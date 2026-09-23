import time
from django.conf import settings
from django.utils import timezone
from common.contracts import (
    DomainError, iso, request_window, validate_ml_input, validate_predictions,
    INPUT_SCHEMA, OUTPUT_SCHEMA, WEATHER_SCHEMA, WEATHER_MODEL, RECORD_FIELDS,
)
from weather_archive.services import get_turbine, get_weather
from .ml_client import get_metadata, check_capability, call_service
from .models import Forecast


def ml_payload(snapshot):
    if snapshot.provenance.get('schema_version') != WEATHER_SCHEMA or snapshot.provenance.get('wind_height_m') != 10:
        raise DomainError('incompatible_snapshot', 'Старый снимок не содержит подтверждённых признаков ветра на 10 м. Получите новый прогноз погоды.', 409)
    payload = {'schema_version': INPUT_SCHEMA, 'turbine_id': snapshot.turbine_id,
               'weather_model': snapshot.weather_model,
               'records': [{key: row.get(key) for key in RECORD_FIELDS} for row in snapshot.records]}
    return validate_ml_input(payload)


def predict(payload):
    if settings.ML_BACKEND == 'demo':
        return {'schema_version': OUTPUT_SCHEMA, 'turbine_id': payload['turbine_id'],
                'weather_model': payload['weather_model'], 'model_version': 'demo-contract-v2',
                'alignment_confirmed': False, 'records': [
                    {'target_time': row['target_time'], 'predicted_normalized_power': round(
                        0 if row['wind_speed_10m_ms'] < 3 or row['wind_speed_10m_ms'] >= 25 else
                        min(1, (row['wind_speed_10m_ms'] ** 3 - 27) / (12 ** 3 - 27)), 5)}
                    for row in payload['records']]}
    return call_service('POST', settings.ML_SERVICE_URL, payload)


def execute_forecast(data, refresh=False, previous=None):
    turbine_id, as_of, start = request_window(data)
    turbine = get_turbine(turbine_id)
    mode = data.get('mode', 'live')
    params = {'mode': mode, 'turbine_id': turbine_id, 'as_of': iso(as_of), 'start_time': iso(start),
              'provider': data.get('provider', 'open_meteo'), 'weather_model': data.get('weather_model', WEATHER_MODEL)}
    if not isinstance(params['provider'], str) or not isinstance(params['weather_model'], str):
        raise DomainError('invalid_value', 'provider и weather_model должны быть строками.')
    run = Forecast.objects.create(turbine=turbine, as_of=as_of, start_time=start, request_params=params,
                                  is_demo=settings.ML_BACKEND == 'demo' or params['provider'] == 'demo')
    started = time.monotonic()

    def event(stage, message):
        run.trace.append({'stage': stage, 'message': message, 'at': iso(timezone.now()),
                          'elapsed_ms': round((time.monotonic() - started) * 1000)})
        run.save(update_fields=['trace'])

    try:
        event('request', 'Запрос принят. Режим: ' + mode + '. Горизонт: 48 часов.')
        metadata = get_metadata()  # Fresh discovery before every cache lookup.
        check_capability(metadata, turbine_id, params['weather_model'])
        run.is_demo = run.is_demo or metadata['is_demo']
        event('metadata', 'Загруженный артефакт: ' + metadata['model_version'])
        if settings.ML_BACKEND == 'http' and params['provider'] == 'demo':
            raise DomainError('synthetic_weather_rejected', 'Внешнему ML нужны реальные признаки JMA GSM. Выберите Open-Meteo или импортированный архив; демо-погода доступна с ML_BACKEND=demo.')
        snapshot = get_weather(turbine_id, start, as_of, params['provider'], params['weather_model'], refresh, mode=mode)
        run.snapshot = snapshot
        run.save(update_fields=['snapshot'])
        event('weather', 'Погода получена. Снимок ' + snapshot.pk[:12])
        payload = ml_payload(snapshot)
        event('validation', '48 часов JMA GSM: ветер 10 м, температура 2 м; контракт проверен.')
        cached = Forecast.objects.filter(
            snapshot=snapshot, turbine_id=turbine_id, model_version=metadata['model_version'],
            response_schema_version=OUTPUT_SCHEMA, status='completed', is_demo=run.is_demo,
        ).exclude(alignment_confirmed=None).first()
        if cached:
            run.records, run.model_version = cached.records, cached.model_version
            run.alignment_confirmed = cached.alignment_confirmed
            event('model', 'Кэш: совпали снимок погоды, турбина и версия модели.')
        else:
            result = validate_predictions(predict(payload), payload, expected_version=metadata['model_version'])
            run.records, run.model_version = result['records'], result['model_version']
            run.alignment_confirmed = result['alignment_confirmed']
            event('model', 'Получено 48 прогнозов мощности: ' + run.model_version)
        run.response_schema_version = OUTPUT_SCHEMA
        powers = [r['predicted_normalized_power'] for r in run.records]
        warnings = []
        if run.is_demo:
            warnings.append('Демонстрационный результат: не использовать для оценки точности или управления ВЭС.')
        if run.alignment_confirmed is False:
            warnings.append('ML-сервис вернул alignment_confirmed=false: согласование данных моделью не подтверждено.')
        if snapshot.provenance.get('timing') == 'estimated_fixed_lead':
            warnings.append('Время выпуска погоды оценочное. Для строгого backtest импортируйте точные архивные выпуски.')
        if max(powers) == min(powers):
            warnings.append('Постоянный прогноз на всём горизонте: проверьте входные данные и модель.')
        run.analysis = {'mean_normalized_power': sum(powers)/48, 'peak_normalized_power': max(powers),
                        'normalized_energy_hours': sum(powers),
                        'max_hourly_ramp': max(abs(b-a) for a, b in zip(powers, powers[1:])), 'warnings': warnings}
        run.status = 'completed'
        event('analysis', 'Результат проверен и сохранён; доступны CSV и JSON.')
    except DomainError as exc:
        run.status = 'failed'
        run.error = {'code': exc.code, 'message': exc.message, 'status': exc.status}
        event('failed', exc.message)
    run.save()
    return run


def refresh_request(previous):
    params = dict(previous.request_params)
    params.setdefault('mode', 'replay')  # Existing legacy runs were historical.
    if params['mode'] == 'live':
        params.pop('as_of', None)
        params.pop('start_time', None)
    return params


def serialize(run, full=True):
    data = {'id': str(run.pk), 'turbine_id': run.turbine_id, 'as_of': iso(run.as_of), 'start_time': iso(run.start_time),
            'status': run.status, 'is_demo': run.is_demo, 'model_version': run.model_version,
            'schema_version': run.response_schema_version or None,
            'alignment_confirmed': run.alignment_confirmed,
            'weather_model': run.request_params.get('weather_model'),
            'created_at': iso(run.created_at), 'snapshot_id': run.snapshot_id, 'error': run.error,
            'analysis': run.analysis, 'request': run.request_params}
    if full:
        data.update(records=run.records, trace=run.trace,
                    weather=run.snapshot.records if run.snapshot_id else [],
                    provenance=run.snapshot.provenance if run.snapshot_id else {})
    return data
