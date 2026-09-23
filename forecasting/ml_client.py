"""Transport and capability discovery for the external, already-trained ML service."""
from urllib.parse import urlsplit, urlunsplit
import requests
from django.conf import settings
from common.contracts import DomainError, INPUT_SCHEMA, OUTPUT_SCHEMA, WEATHER_MODEL, RECORD_FIELDS, text

FIELD_DEFINITIONS = {
    'target_time': 'Timezone-aware ISO 8601; consecutive whole UTC hours.',
    'wind_speed_10m_ms': 'Wind speed 10 m above ground, m/s.',
    'wind_direction_10m_deg': 'Direction FROM, clockwise from north, 0–360 degrees.',
    'temperature_2m_c': 'Air temperature at 2 m, Celsius.',
}


def metadata_url():
    if settings.ML_METADATA_URL:
        return settings.ML_METADATA_URL
    parts = urlsplit(settings.ML_SERVICE_URL)
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rsplit('/', 1)[0] + '/metadata', '', ''))


def call_service(method, url, payload=None):
    parts = urlsplit(url)
    if parts.scheme not in ('https', 'http') or not parts.netloc or parts.hostname in ('github.com', 'raw.githubusercontent.com'):
        raise DomainError('ml_configuration', 'Укажите HTTP(S) URL развёрнутого ML API, не ссылку на GitHub.', 503)
    headers = {'Accept': 'application/json'}
    if settings.ML_SERVICE_TOKEN:
        headers['Authorization'] = 'Bearer ' + settings.ML_SERVICE_TOKEN
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers, timeout=settings.HTTP_TIMEOUT)
        else:
            response = requests.post(url, json=payload, headers=headers, timeout=settings.HTTP_TIMEOUT)
        if response.status_code == 401:
            raise DomainError('ml_unauthorized', 'ML-сервис отклонил сервисный токен. ML_SERVICE_TOKEN должен совпадать на обоих сервисах; это не NVIDIA API key.', 401)
        if response.status_code == 422:
            raise DomainError('ml_request_rejected', 'ML-сервис отклонил запрос: проверьте контракт, поддержку турбины и источник jma_gsm.', 422)
        if response.status_code != 200:
            raise DomainError('ml_unavailable', f'ML-сервис вернул HTTP {response.status_code}.', 502)
        return response.json()
    except requests.Timeout:
        raise DomainError('ml_timeout', 'ML-сервис не ответил вовремя.', 504)
    except (requests.RequestException, ValueError):
        raise DomainError('ml_unavailable', 'ML-сервис недоступен или вернул некорректный JSON.', 502)


def get_metadata():
    if settings.ML_BACKEND == 'demo':
        return {'model_version': 'demo-contract-v2', 'supported_turbines': ['turbine_1', 'turbine_2'],
                'weather_model': WEATHER_MODEL, 'field_definitions': FIELD_DEFINITIONS,
                'is_demo': True, 'metadata_source': 'local_demo'}
    if settings.ML_BACKEND != 'http':
        raise DomainError('ml_configuration', 'ML_BACKEND должен быть demo или http.', 503)
    data = call_service('GET', metadata_url())
    # Metadata JSON field names were not fully specified by the teammate.
    # Keep this mapping in one place; never infer turbine support from a model name.
    try:
        if not isinstance(data, dict):
            raise DomainError('invalid_metadata', 'metadata должен быть объектом.')
        version = text(data.get('model_version'), 'model_version')
        turbines = data.get('supported_turbines', data.get('turbine_ids'))
        if not isinstance(turbines, list) or not turbines:
            raise DomainError('invalid_metadata', 'metadata должен содержать supported_turbines (или turbine_ids).')
        for turbine in turbines:
            text(turbine, 'supported_turbines item', 64)
        source = data.get('weather_model', data.get('weather_source'))
        if isinstance(source, dict):
            source = source.get('weather_model', source.get('model'))
        source = text(source, 'weather_model')
        definitions = data.get('field_definitions', data.get('input_fields', data.get('feature_names')))
        if not isinstance(definitions, (dict, list)) or not definitions:
            raise DomainError('invalid_metadata', 'metadata должен содержать field_definitions, input_fields или feature_names.')
        return {'model_version': version, 'supported_turbines': turbines, 'weather_model': source,
                'field_definitions': definitions, 'is_demo': version.startswith(('mock-', 'demo-')),
                'metadata_source': 'loaded_artifact', 'raw': data}
    except DomainError as exc:
        raise DomainError('invalid_ml_metadata', exc.message, 502)


def check_capability(metadata, turbine_id, weather_model):
    if turbine_id not in metadata['supported_turbines']:
        raise DomainError('unsupported_turbine', 'Загруженный ML-артефакт не поддерживает ' + turbine_id + '.', 422)
    if weather_model != WEATHER_MODEL or metadata['weather_model'] != weather_model:
        raise DomainError('weather_source_mismatch', 'Загруженный артефакт и погода должны использовать jma_gsm. Автоматическая замена источника запрещена.', 422)
