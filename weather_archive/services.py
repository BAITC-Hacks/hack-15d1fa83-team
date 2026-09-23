import hashlib
import json
import math
from datetime import timedelta
from urllib.parse import urlparse
import requests
from django.conf import settings
from common.contracts import DomainError, iso, timestamp, text, validate_weather
from .models import Turbine, WeatherSnapshot, ArchivedRun

VARIABLES = {'wind_speed': 'wind_speed_100m', 'wind_direction': 'wind_direction_100m',
             'temperature': 'temperature_2m', 'pressure': 'surface_pressure'}
MODELS = ('gfs_global', 'icon_global', 'ecmwf_ifs025')
ENDPOINT = 'https://previous-runs-api.open-meteo.com/v1/forecast'


def digest(value):
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    except (ValueError, TypeError):
        raise DomainError('invalid_value', 'Данные содержат NaN, Infinity или неподдерживаемый тип.')
    return hashlib.sha256(encoded).hexdigest()


def get_turbine(turbine_id):
    try:
        return Turbine.objects.get(pk=turbine_id)
    except Turbine.DoesNotExist:
        raise DomainError('unknown_turbine', 'Турбина не найдена.', 404)


def demo_weather(start, as_of, turbine_id):
    issued = as_of.replace(minute=0, second=0, microsecond=0) - timedelta(hours=12)
    shift = int(hashlib.sha256(turbine_id.encode()).hexdigest()[:4], 16) % 11
    return [{
        'target_time': iso(start + timedelta(hours=i)), 'issued_at': iso(issued), 'available_at': iso(issued + timedelta(hours=8)),
        'wind_speed': round(7.4 + 3 * math.sin((i + shift) / 7) + .6 * math.cos(i / 2), 2),
        'wind_direction': round((225 + 34 * math.sin(i / 11)) % 360, 1),
        'temperature': round(-5 + 4 * math.sin(i / 5), 1), 'pressure': round(950 + 5 * math.cos(i / 13), 1),
        'weather_model': 'synthetic-demo',
    } for i in range(48)]


def previous_runs(turbine, start, as_of, model):
    if model not in MODELS:
        raise DomainError('unsupported_model', 'Поддерживаются: ' + ', '.join(MODELS))
    if turbine.latitude is None or turbine.longitude is None:
        raise DomainError('coordinates_required', 'Укажите подтверждённые координаты турбины в Django Admin.', 409)
    # Conservative fixed-lead selection: retain an additional publication margin.
    lag = settings.WEATHER_PUBLICATION_LAG_HOURS
    targets = [start + timedelta(hours=i) for i in range(48)]
    offsets = [max(1, math.ceil(((t - as_of).total_seconds() / 3600 + lag) / 24)) for t in targets]
    if max(offsets) > 7:
        raise DomainError('unsupported_window', 'Архив поддерживает заблаговременность до 7 суток.')
    variables = [f'{v}_previous_day{d}' for d in sorted(set(offsets)) for v in VARIABLES.values()]
    params = {'latitude': turbine.latitude, 'longitude': turbine.longitude, 'start_date': start.date().isoformat(),
              'end_date': targets[-1].date().isoformat(), 'hourly': ','.join(variables), 'models': model,
              'wind_speed_unit': 'ms', 'temperature_unit': 'celsius', 'timezone': 'UTC'}
    try:
        response = requests.get(ENDPOINT, params=params, timeout=settings.HTTP_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        raise DomainError('weather_unavailable', 'Архив погоды недоступен или не содержит выбранный период. Повторите позже либо импортируйте выпуск.', 502)
    try:
        hourly = payload['hourly']
        units = payload['hourly_units']
        expected_units = {'wind_speed': 'm/s', 'wind_direction': '°', 'temperature': '°C', 'pressure': 'hPa'}
        for day in set(offsets):
            for field, variable in VARIABLES.items():
                if units[f'{variable}_previous_day{day}'] != expected_units[field]:
                    raise ValueError('Unexpected units')
        index = {timestamp(t + 'Z'): i for i, t in enumerate(hourly['time'])}
        records = []
        for target, day in zip(targets, offsets):
            issued = target - timedelta(days=day)
            row = {'target_time': iso(target), 'issued_at': iso(issued), 'available_at': iso(issued + timedelta(hours=lag)), 'weather_model': model}
            row.update({field: hourly[f'{variable}_previous_day{day}'][index[target]] for field, variable in VARIABLES.items()})
            records.append(row)
    except (KeyError, TypeError, IndexError, ValueError):
        raise DomainError('invalid_weather_response', 'Архив вернул неполные данные или неожиданные единицы.', 502)
    provenance = {'kind': 'archived_forecast', 'timing': 'estimated_fixed_lead', 'source_url': ENDPOINT,
                  'publication_lag_hours': lag, 'wind_height_m': 100, 'temperature_height_m': 2,
                  'pressure_type': 'surface', 'request_params': params, 'raw_sha256': digest(payload),
                  'notice': 'Архивные прогнозы с фиксированной заблаговременностью. issued_at и available_at оценочные; точный выпуск и фактическое время публикации API не возвращает. Для строгой проверки используйте импорт выпусков.'}
    return records, provenance


def import_run(data):
    turbine = get_turbine(text(data.get('turbine_id'), 'turbine_id', 64))
    issued = timestamp(data.get('issued_at'), 'issued_at')
    available = timestamp(data.get('available_at'), 'available_at')
    model = text(data.get('weather_model'), 'weather_model')
    source = text(data.get('source_url'), 'source_url', 1000)
    if urlparse(source).scheme not in ('https', 'http') or not urlparse(source).netloc:
        raise DomainError('invalid_source', 'source_url должен указывать на источник архива.')
    if data.get('data_kind') != 'archived_forecast':
        raise DomainError('invalid_source', 'Разрешены только archived_forecast, не наблюдения и не реанализ.')
    records = data.get('records')
    if not isinstance(records, list) or not 48 <= len(records) <= 384:
        raise DomainError('invalid_horizon', 'Импортируйте от 48 до 384 последовательных часов.')
    if any(not isinstance(r, dict) for r in records):
        raise DomainError('invalid_record', 'Записи должны быть JSON-объектами.')
    prepared = [{**r, 'target_time': iso(timestamp(r.get('target_time'))), 'issued_at': iso(issued), 'available_at': iso(available), 'weather_model': model} for r in records]
    start = timestamp(prepared[0].get('target_time'))
    if start.minute or start.second or start.microsecond:
        raise DomainError('invalid_grid', 'Данные должны начинаться с целого часа UTC.')
    # Validate overlapping 48h windows, including the tail, before any write.
    for offset in list(range(0, len(prepared) - 47, 48)) + [len(prepared) - 48]:
        validate_weather(prepared[offset:offset + 48], start + timedelta(hours=offset), available)
    identity = digest({'turbine': turbine.pk, 'model': model, 'issued': iso(issued), 'available': iso(available), 'source': source, 'records': prepared})
    run, created = ArchivedRun.objects.get_or_create(pk=identity, defaults={'turbine': turbine, 'issued_at': issued,
        'available_at': available, 'weather_model': model, 'source_url': source, 'records': prepared})
    return run, created


def archived_weather(turbine, start, as_of, model):
    for run in ArchivedRun.objects.filter(turbine=turbine, weather_model=model, available_at__lte=as_of, issued_at__lte=as_of):
        by_time = {r['target_time']: r for r in run.records}
        times = [iso(start + timedelta(hours=i)) for i in range(48)]
        if all(t in by_time for t in times):
            return [by_time[t] for t in times], {'kind': 'archived_forecast', 'timing': 'imported_exact_run',
                'source_url': run.source_url, 'run_id': run.pk, 'wind_height_m': 100, 'temperature_height_m': 2,
                'pressure_type': 'surface', 'notice': 'Время публикации и происхождение предоставлены владельцем импортированного архива.'}
    raise DomainError('archive_not_found', 'Нет доступного на as_of выпуска, покрывающего все 48 часов.', 404)


def get_weather(turbine_id, start, as_of, provider='open_meteo', model='gfs_global', refresh=False):
    turbine = get_turbine(turbine_id)
    if provider not in ('demo', 'open_meteo', 'archive'):
        raise DomainError('unsupported_provider', 'Источник: open_meteo, archive или demo.')
    if provider == 'demo':
        model = 'synthetic-demo'
    if not refresh and provider != 'archive':
        cached = WeatherSnapshot.objects.filter(turbine=turbine, as_of=as_of, start_time=start, provider=provider, weather_model=model).first()
        if cached:
            return cached
    if provider == 'demo':
        records, provenance = demo_weather(start, as_of, turbine_id), {'kind': 'synthetic', 'timing': 'synthetic', 'notice': 'Синтетические данные для проверки интерфейса. Не использовать для оценки модели.'}
    elif provider == 'open_meteo':
        records, provenance = previous_runs(turbine, start, as_of, model)
    else:
        records, provenance = archived_weather(turbine, start, as_of, model)
    records = validate_weather(records, start, as_of)
    identity = digest({'turbine_id': turbine_id, 'as_of': iso(as_of), 'start_time': iso(start), 'provider': provider, 'records': records, 'provenance': provenance})
    snapshot, _ = WeatherSnapshot.objects.get_or_create(pk=identity, defaults={'turbine': turbine, 'as_of': as_of,
        'start_time': start, 'provider': provider, 'weather_model': model, 'records': records, 'provenance': provenance})
    return snapshot


def serialize(snapshot):
    return {'schema_version': '1.0', 'snapshot_id': snapshot.pk, 'turbine_id': snapshot.turbine_id,
            'as_of': iso(snapshot.as_of), 'start_time': iso(snapshot.start_time), 'provider': snapshot.provider,
            'provenance': snapshot.provenance, 'units': {'wind_speed': 'm/s', 'wind_direction': 'degrees',
            'temperature': 'celsius', 'pressure': 'hPa', 'forecast_age_hours': 'hours'}, 'records': snapshot.records}
