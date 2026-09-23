import time
import requests
from django.conf import settings
from django.utils import timezone
from common.contracts import DomainError, iso, request_window, validate_predictions
from weather_archive.services import get_turbine, get_weather
from .models import Forecast


def ml_payload(snapshot):
    keys = ('target_time', 'wind_speed', 'wind_direction', 'temperature', 'pressure', 'weather_model', 'forecast_age_hours')
    return {'turbine_id': snapshot.turbine_id, 'records': [{k: row[k] for k in keys} for row in snapshot.records]}


def predict(snapshot):
    payload = ml_payload(snapshot)
    if settings.ML_BACKEND == 'demo':
        # Contract stub only. Never presented as a trained model.
        return {'model_version': 'demo-power-curve-v1', 'records': [
            {'target_time': row['target_time'], 'predicted_normalized_power': round(
                0 if row['wind_speed'] < 3 or row['wind_speed'] >= 25 else min(1, (row['wind_speed'] ** 3 - 27) / (12 ** 3 - 27)), 5)}
            for row in payload['records']]}
    if settings.ML_BACKEND != 'http':
        raise DomainError('ml_configuration', 'ML_BACKEND должен быть demo или http.', 503)
    headers = {'Accept': 'application/json'}
    if settings.ML_SERVICE_TOKEN:
        headers['Authorization'] = 'Bearer ' + settings.ML_SERVICE_TOKEN
    try:
        response = requests.post(settings.ML_SERVICE_URL, json=payload, headers=headers, timeout=settings.HTTP_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        raise DomainError('ml_timeout', 'ML-сервис не ответил вовремя.', 504)
    except (requests.RequestException, ValueError):
        raise DomainError('ml_unavailable', 'ML-сервис недоступен или вернул некорректный JSON.', 502)


def execute_forecast(data, refresh=False, previous=None):
    turbine_id, as_of, start = request_window(data)
    turbine = get_turbine(turbine_id)
    params = {'turbine_id': turbine_id, 'as_of': iso(as_of), 'start_time': iso(start),
              'provider': data.get('provider', 'open_meteo'), 'weather_model': data.get('weather_model', 'gfs_global')}
    if not isinstance(params['provider'], str) or not isinstance(params['weather_model'], str):
        raise DomainError('invalid_value', 'provider и weather_model должны быть строками.')
    run = Forecast.objects.create(turbine=turbine, as_of=as_of, start_time=start, request_params=params,
                                  is_demo=settings.ML_BACKEND == 'demo' or params['provider'] == 'demo')
    started = time.monotonic()
    def event(stage, message):
        run.trace.append({'stage': stage, 'message': message, 'at': iso(timezone.now()), 'elapsed_ms': round((time.monotonic() - started) * 1000)})
        run.save(update_fields=['trace'])
    try:
        event('request', 'Запрос принят. Горизонт: 48 часов.')
        snapshot = get_weather(turbine_id, start, as_of, params['provider'], params['weather_model'], refresh)
        run.snapshot = snapshot
        run.save(update_fields=['snapshot'])
        event('weather', 'Погода получена. Снимок ' + snapshot.pk[:12])
        event('validation', '48 часов: сетка, единицы и доступность на as_of проверены.')
        if previous and previous.snapshot_id == snapshot.pk and previous.status == 'completed' and previous.is_demo == run.is_demo:
            run.records, run.model_version = previous.records, previous.model_version
            event('model', 'Входные данные не изменились; сохранён предыдущий расчёт. Для новой версии ML создайте новый прогноз.')
        else:
            try:
                result = validate_predictions(predict(snapshot), snapshot.records)
            except DomainError as exc:
                if exc.status == 422:
                    raise DomainError('invalid_ml_response', exc.message, 502)
                raise
            run.records, run.model_version = result['records'], result['model_version']
            if run.model_version.startswith(('demo-', 'mock-')):
                run.is_demo = True
            event('model', 'Получено 48 прогнозов мощности: ' + run.model_version)
        powers = [r['predicted_normalized_power'] for r in run.records]
        warnings = []
        if run.is_demo:
            warnings.append('Демонстрационный результат: не использовать для оценки точности или управления ВЭС.')
        if snapshot.provenance.get('timing') == 'estimated_fixed_lead':
            warnings.append('Время выпуска погоды оценочное. Для строгого backtest импортируйте точные архивные выпуски.')
        if max(powers) == min(powers):
            warnings.append('Постоянный прогноз на всём горизонте: проверьте входные данные и модель.')
        run.analysis = {'mean_normalized_power': sum(powers) / 48, 'peak_normalized_power': max(powers),
                        'normalized_energy_hours': sum(powers), 'max_hourly_ramp': max(abs(b-a) for a,b in zip(powers,powers[1:])),
                        'warnings': warnings}
        run.status = 'completed'
        event('analysis', 'Результат проверен и сохранён; доступны CSV и JSON.')
    except DomainError as exc:
        run.status = 'failed'
        run.error = {'code': exc.code, 'message': exc.message, 'status': exc.status}
        event('failed', exc.message)
    run.save()
    return run


def serialize(run, full=True):
    data = {'id': str(run.pk), 'turbine_id': run.turbine_id, 'as_of': iso(run.as_of), 'start_time': iso(run.start_time),
            'status': run.status, 'is_demo': run.is_demo, 'model_version': run.model_version,
            'created_at': iso(run.created_at), 'snapshot_id': run.snapshot_id, 'error': run.error,
            'analysis': run.analysis, 'request': run.request_params}
    if full:
        data.update(records=run.records, trace=run.trace,
                    weather=run.snapshot.records if run.snapshot_id else [],
                    provenance=run.snapshot.provenance if run.snapshot_id else {})
    return data
