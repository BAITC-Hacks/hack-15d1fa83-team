import hashlib
import json
import math
from datetime import timedelta
from urllib.parse import urlparse
import requests
from django.conf import settings
from django.utils import timezone
from common.contracts import DomainError, WEATHER_SCHEMA, WEATHER_MODEL, iso, timestamp, text, validate_weather
from .models import Turbine, WeatherSnapshot, ArchivedRun

VARIABLES = {'wind_speed_10m_ms': 'wind_speed_10m', 'wind_direction_10m_deg': 'wind_direction_10m',
             'temperature_2m_c': 'temperature_2m'}
EXPECTED_UNITS = {'wind_speed_10m_ms': 'm/s', 'wind_direction_10m_deg': '°', 'temperature_2m_c': '°C'}
ENDPOINT = 'https://previous-runs-api.open-meteo.com/v1/forecast'
LIVE_ENDPOINT = 'https://api.open-meteo.com/v1/forecast'


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
        'target_time': iso(start + timedelta(hours=i)), 'issued_at': iso(issued),
        'available_at': iso(issued + timedelta(hours=8)),
        'wind_speed_10m_ms': round(5.4 + 2 * math.sin((i + shift) / 7) + .6 * math.cos(i / 2), 2),
        'wind_direction_10m_deg': round((225 + 34 * math.sin(i / 11)) % 360, 1),
        'temperature_2m_c': round(-5 + 4 * math.sin(i / 5), 1),
        'pressure': None, 'weather_model': WEATHER_MODEL,
    } for i in range(48)]


def provider_forecast(turbine, start, as_of, model, live=False):
    if model != WEATHER_MODEL:
        raise DomainError('weather_source_mismatch', 'Текущий ML-артефакт требует только jma_gsm.')
    if turbine.latitude is None or turbine.longitude is None:
        raise DomainError('coordinates_required', 'Укажите подтверждённые координаты турбины в Django Admin.', 409)
    lag = settings.WEATHER_PUBLICATION_LAG_HOURS
    targets = [start + timedelta(hours=i) for i in range(48)]
    offsets = [0] * 48 if live else [
        max(1, math.ceil(((t - as_of).total_seconds() / 3600 + lag) / 24)) for t in targets]
    if max(offsets) > 7:
        raise DomainError('unsupported_window', 'Архив поддерживает заблаговременность до 7 суток.')
    def key(variable, day):
        return variable if live else f'{variable}_previous_day{day}'
    variables = [key(v, day) for day in sorted(set(offsets)) for v in VARIABLES.values()]
    params = {'latitude': turbine.latitude, 'longitude': turbine.longitude, 'start_date': start.date().isoformat(),
              'end_date': targets[-1].date().isoformat(), 'hourly': ','.join(variables), 'models': model,
              'wind_speed_unit': 'ms', 'temperature_unit': 'celsius', 'timezone': 'UTC'}
    endpoint = LIVE_ENDPOINT if live else ENDPOINT
    try:
        response = requests.get(endpoint, params=params, timeout=settings.HTTP_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        raise DomainError('weather_unavailable', 'Прогноз JMA GSM недоступен для выбранного периода. Другие модели и демо не подставляются.', 502)
    fetched_at = timezone.now()
    try:
        hourly, units = payload['hourly'], payload['hourly_units']
        for day in set(offsets):
            for field, variable in VARIABLES.items():
                if units[key(variable, day)] != EXPECTED_UNITS[field]:
                    raise ValueError('Unexpected units')
        times = [timestamp(t + 'Z') for t in hourly['time']]
        if len(set(times)) != len(times):
            raise ValueError('Duplicate upstream hours')
        index = {t: i for i, t in enumerate(times)}
        records = []
        for target, day in zip(targets, offsets):
            issued = None if live else target - timedelta(days=day)
            row = {'target_time': iso(target), 'issued_at': iso(issued) if issued else None,
                   'available_at': iso(issued + timedelta(hours=lag)) if issued else None,
                   'weather_model': model, 'pressure': None}
            row.update({field: hourly[key(variable, day)][index[target]] for field, variable in VARIABLES.items()})
            records.append(row)
    except (KeyError, TypeError, IndexError, ValueError):
        raise DomainError('invalid_weather_response', 'JMA GSM вернул неполные данные или неожиданные единицы.', 502)
    provenance = {
        'schema_version': WEATHER_SCHEMA, 'wind_height_m': 10, 'temperature_height_m': 2,
        'kind': 'live_forecast' if live else 'archived_forecast',
        'timing': 'live_retrieval' if live else 'estimated_fixed_lead',
        'source_url': endpoint, 'fetched_at': iso(fetched_at), 'request_params': params,
        'raw_sha256': digest(payload), 'publication_lag_hours': None if live else lag,
        'notice': ('Текущий прогноз JMA GSM, полученный по запросу. Время выпуска API не возвращает; issued_at и возраст неизвестны.'
                   if live else 'Архивные прогнозы JMA GSM с фиксированной заблаговременностью. Время выпуска и доступности оценочное; для строгого replay используйте точные выпуски.'),
    }
    return records, provenance


def previous_runs(turbine, start, as_of, model=WEATHER_MODEL):
    return provider_forecast(turbine, start, as_of, model, live=False)


def import_run(data):
    turbine = get_turbine(text(data.get('turbine_id'), 'turbine_id', 64))
    issued, available = timestamp(data.get('issued_at'), 'issued_at'), timestamp(data.get('available_at'), 'available_at')
    model = text(data.get('weather_model'), 'weather_model')
    if model != WEATHER_MODEL:
        raise DomainError('weather_source_mismatch', 'Для текущей модели импортируйте jma_gsm.')
    if data.get('wind_height_m') != 10 or data.get('temperature_height_m') != 2:
        raise DomainError('incompatible_features', 'Укажите wind_height_m=10 и temperature_height_m=2; ветер 100 м не заменяет ветер 10 м.')
    source = text(data.get('source_url'), 'source_url', 1000)
    if urlparse(source).scheme not in ('https', 'http') or not urlparse(source).netloc:
        raise DomainError('invalid_source', 'source_url должен указывать на источник архива.')
    if data.get('data_kind') != 'archived_forecast':
        raise DomainError('invalid_source', 'Разрешены только archived_forecast, не наблюдения и не реанализ.')
    records = data.get('records')
    if not isinstance(records, list) or not 48 <= len(records) <= 384 or any(not isinstance(r, dict) for r in records):
        raise DomainError('invalid_horizon', 'Импортируйте от 48 до 384 последовательных почасовых объектов.')
    prepared = [{**r, 'target_time': iso(timestamp(r.get('target_time'))),
                 'issued_at': iso(issued), 'available_at': iso(available), 'weather_model': model} for r in records]
    start = timestamp(prepared[0]['target_time'])
    for offset in list(range(0, len(prepared) - 47, 48)) + [len(prepared) - 48]:
        validate_weather(prepared[offset:offset + 48], start + timedelta(hours=offset), available)
    identity = digest({'schema_version': WEATHER_SCHEMA, 'turbine': turbine.pk, 'model': model,
                       'issued': iso(issued), 'available': iso(available), 'source': source, 'records': prepared})
    run, created = ArchivedRun.objects.get_or_create(pk=identity, defaults={
        'schema_version': WEATHER_SCHEMA, 'turbine': turbine, 'issued_at': issued,
        'available_at': available, 'weather_model': model, 'source_url': source, 'records': prepared})
    return run, created


def archived_weather(turbine, start, as_of, model):
    for run in ArchivedRun.objects.filter(turbine=turbine, weather_model=model, schema_version=WEATHER_SCHEMA,
                                           available_at__lte=as_of, issued_at__lte=as_of):
        by_time = {r['target_time']: r for r in run.records}
        times = [iso(start + timedelta(hours=i)) for i in range(48)]
        if all(t in by_time for t in times):
            return [by_time[t] for t in times], {
                'schema_version': WEATHER_SCHEMA, 'kind': 'archived_forecast', 'timing': 'imported_exact_run',
                'source_url': run.source_url, 'run_id': run.pk, 'wind_height_m': 10, 'temperature_height_m': 2,
                'notice': 'Время публикации и происхождение предоставлены владельцем импортированного архива JMA GSM.'}
    raise DomainError('archive_not_found', 'Нет доступного на as_of выпуска JMA GSM с ветром 10 м, покрывающего все 48 часов.', 404)


def get_weather(turbine_id, start, as_of, provider='open_meteo', model=WEATHER_MODEL, refresh=False, mode='replay'):
    turbine = get_turbine(turbine_id)
    if model != WEATHER_MODEL:
        raise DomainError('weather_source_mismatch', 'Поддерживается только jma_gsm. Необходимо переобучение ML для другого источника.')
    if provider not in ('demo', 'open_meteo', 'archive'):
        raise DomainError('unsupported_provider', 'Источник: open_meteo, archive или demo.')
    if mode not in ('live', 'replay'):
        raise DomainError('invalid_mode', 'Режим должен быть live или replay.')
    if not refresh and provider != 'archive':
        cached = WeatherSnapshot.objects.filter(
            turbine=turbine, as_of=as_of, start_time=start, provider=provider, weather_model=model,
            provenance__schema_version=WEATHER_SCHEMA, provenance__mode=mode,
        ).first()
        if cached:
            return cached
    if provider == 'demo':
        records, provenance = demo_weather(start, as_of, turbine_id), {
            'schema_version': WEATHER_SCHEMA, 'kind': 'synthetic', 'timing': 'synthetic',
            'wind_height_m': 10, 'temperature_height_m': 2,
            'notice': 'Синтетические признаки формата JMA GSM для UI. Это не реальные прогнозы JMA; внешнему ML не отправляются.'}
    elif provider == 'open_meteo':
        records, provenance = provider_forecast(turbine, start, as_of, model, live=mode == 'live')
    else:
        records, provenance = archived_weather(turbine, start, as_of, model)
    records = validate_weather(records, start, as_of, historical=mode == 'replay')
    provenance['mode'] = mode
    # Retrieval time and generationtime_ms in the raw response must not invalidate identical weather.
    stable_provenance = {k: v for k, v in provenance.items() if k not in ('fetched_at', 'raw_sha256')}
    identity = digest({'turbine_id': turbine_id, 'as_of': iso(as_of), 'start_time': iso(start),
                       'provider': provider, 'records': records, 'provenance': stable_provenance})
    snapshot, _ = WeatherSnapshot.objects.get_or_create(pk=identity, defaults={
        'turbine': turbine, 'as_of': as_of, 'start_time': start, 'provider': provider,
        'weather_model': model, 'records': records, 'provenance': provenance})
    return snapshot


def serialize(snapshot):
    return {'schema_version': snapshot.provenance.get('schema_version', 'legacy'), 'snapshot_id': snapshot.pk,
            'turbine_id': snapshot.turbine_id, 'weather_model': snapshot.weather_model,
            'as_of': iso(snapshot.as_of), 'start_time': iso(snapshot.start_time), 'provider': snapshot.provider,
            'provenance': snapshot.provenance, 'units': {'wind_speed_10m_ms': 'm/s', 'wind_direction_10m_deg': 'degrees',
            'temperature_2m_c': 'celsius', 'pressure': 'hPa', 'forecast_age_hours': 'hours'}, 'records': snapshot.records}
