import math
from datetime import datetime, timezone, timedelta
from django.utils import timezone as django_timezone

INPUT_SCHEMA = 'windpower.input.v1'
OUTPUT_SCHEMA = 'windpower.output.v1'
WEATHER_SCHEMA = 'windpower.weather.v2'
WEATHER_MODEL = 'jma_gsm'
FEATURE_FIELDS = ('wind_speed_10m_ms', 'wind_direction_10m_deg', 'temperature_2m_c')
RECORD_FIELDS = ('target_time', *FEATURE_FIELDS)


class DomainError(Exception):
    def __init__(self, code, message, status=422):
        self.code, self.message, self.status = code, message, status
        super().__init__(message)


def timestamp(value, field='time'):
    if not isinstance(value, str):
        raise DomainError('invalid_time', f'{field}: требуется ISO 8601 со смещением часового пояса.')
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError()
        return dt.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        raise DomainError('invalid_time', f'{field}: требуется ISO 8601 со смещением часового пояса.')


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def number(value, field, low, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainError('invalid_value', f'{field}: требуется конечное JSON-число.')
    try:
        valid = math.isfinite(value) and value >= low and (high is None or value <= high)
    except OverflowError:
        valid = False
    if not valid:
        raise DomainError('invalid_value', f'{field}: число вне допустимого диапазона.')
    return float(value)


def text(value, field, max_length=160):
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise DomainError('invalid_value', f'{field}: требуется непустая строка (до {max_length} символов).')
    return value


def exact_fields(value, required, optional=()):
    if not isinstance(value, dict):
        raise DomainError('invalid_body', 'Ожидается JSON-объект.')
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        raise DomainError('invalid_fields', f'Недостающие поля: {sorted(missing)}. Неизвестные поля: {sorted(extra)}.')


def request_window(data):
    exact_fields(data, ('turbine_id',), ('mode', 'as_of', 'start_time', 'provider', 'weather_model'))
    turbine = text(data['turbine_id'], 'turbine_id', 64)
    mode = data.get('mode', 'live')
    if mode == 'live':
        if 'as_of' in data or 'start_time' in data:
            raise DomainError('server_time_required', 'В live время выбирает Django; не передавайте as_of или start_time.')
        as_of = django_timezone.now()
        start = as_of.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    elif mode == 'replay':
        as_of = timestamp(data.get('as_of'), 'as_of')
        start = timestamp(data.get('start_time'), 'start_time')
        if as_of > django_timezone.now():
            raise DomainError('invalid_window', 'Момент исторического решения не может быть в будущем.')
        if start.minute or start.second or start.microsecond:
            raise DomainError('invalid_window', 'start_time должен совпадать с началом часа UTC.')
        if start <= as_of or start > as_of + timedelta(hours=24):
            raise DomainError('invalid_window', 'Начало горизонта должно быть после as_of, не более чем через 24 часа.')
    else:
        raise DomainError('invalid_mode', 'Режим должен быть live или replay.')
    return turbine, as_of, start


def feature_record(row):
    return {
        'target_time': iso(timestamp(row.get('target_time'), 'target_time')),
        'wind_speed_10m_ms': number(row.get('wind_speed_10m_ms'), 'wind_speed_10m_ms', 0, 100),
        'wind_direction_10m_deg': number(row.get('wind_direction_10m_deg'), 'wind_direction_10m_deg', 0, 360),
        'temperature_2m_c': number(row.get('temperature_2m_c'), 'temperature_2m_c', -100, 70),
    }


def validate_ml_input(data):
    exact_fields(data, ('schema_version', 'turbine_id', 'weather_model', 'records'))
    if data['schema_version'] != INPUT_SCHEMA:
        raise DomainError('invalid_schema', 'Ожидается ' + INPUT_SCHEMA)
    text(data['turbine_id'], 'turbine_id', 64)
    if data['weather_model'] != WEATHER_MODEL:
        raise DomainError('weather_source_mismatch', 'Текущий контракт модели требует jma_gsm с ветром на 10 м.')
    records = data['records']
    if not isinstance(records, list) or len(records) != 48:
        raise DomainError('invalid_horizon', 'Нужны ровно 48 последовательных почасовых записей.')
    start = None
    for i, row in enumerate(records):
        exact_fields(row, RECORD_FIELDS)
        clean = feature_record(row)
        target = timestamp(clean['target_time'])
        start = target if start is None else start
        if target.minute or target.second or target.microsecond or target != start + timedelta(hours=i):
            raise DomainError('invalid_grid', 'Нужны последовательные целые часы без пропусков и повторов.')
    return data


def validate_weather(records, start, as_of, historical=True):
    if not isinstance(records, list) or len(records) != 48:
        raise DomainError('invalid_horizon', 'Нужны ровно 48 последовательных почасовых записей.')
    clean = []
    for i, row in enumerate(records):
        if not isinstance(row, dict):
            raise DomainError('invalid_record', f'Запись {i} должна быть объектом.')
        prepared = feature_record(row)
        target = timestamp(prepared['target_time'])
        if target != start + timedelta(hours=i) or target.minute or target.second or target.microsecond:
            raise DomainError('invalid_grid', 'Часы должны идти по порядку, без пропусков и повторов.')
        model = text(row.get('weather_model'), 'weather_model')
        if model != WEATHER_MODEL:
            raise DomainError('weather_source_mismatch', 'Требуется jma_gsm; подмена источника не допускается.')
        issued = timestamp(row['issued_at'], 'issued_at') if row.get('issued_at') is not None else None
        available = timestamp(row['available_at'], 'available_at') if row.get('available_at') is not None else None
        if historical and (issued is None or available is None):
            raise DomainError('missing_availability', 'Для replay нужны сведения о времени выпуска и доступности.')
        if (issued and issued > target) or (available and available > as_of) or (issued and available and issued > available):
            raise DomainError('future_leakage', 'Прогноз погоды не был доступен на момент as_of.')
        prepared.update(
            issued_at=iso(issued) if issued else None, available_at=iso(available) if available else None,
            weather_model=model, pressure=number(row['pressure'], 'surface pressure (hPa)', 300, 1100) if row.get('pressure') is not None else None,
            forecast_age_hours=(as_of-issued).total_seconds()/3600 if issued else None,
            lead_time_hours=(target-issued).total_seconds()/3600 if issued else None,
        )
        clean.append(prepared)
    return clean


def validate_predictions(data, payload, expected_version=None):
    try:
        exact_fields(data, ('schema_version', 'turbine_id', 'weather_model', 'model_version', 'alignment_confirmed', 'records'))
        if data['schema_version'] != OUTPUT_SCHEMA:
            raise DomainError('invalid_schema', 'Ожидается ' + OUTPUT_SCHEMA)
        if data['turbine_id'] != payload['turbine_id'] or data['weather_model'] != payload['weather_model']:
            raise DomainError('identity_mismatch', 'ML-ответ относится к другой турбине или погодной модели.')
        version = text(data['model_version'], 'model_version')
        if type(data['alignment_confirmed']) is not bool:
            raise DomainError('invalid_alignment', 'alignment_confirmed должен быть JSON boolean.')
        if expected_version is not None and version != expected_version:
            raise DomainError('ml_model_changed', 'Версия ML изменилась между metadata и predict. Создайте новый запрос.', 409)
        values = data['records']
        if not isinstance(values, list) or len(values) != 48:
            raise DomainError('invalid_horizon', 'ML-сервис должен вернуть ровно 48 записей.')
        clean = []
        for row, weather in zip(values, payload['records']):
            exact_fields(row, ('target_time', 'predicted_normalized_power'))
            if timestamp(row['target_time']) != timestamp(weather['target_time']):
                raise DomainError('invalid_grid', 'Временная сетка ML-ответа не совпадает с запросом.')
            clean.append({'target_time': weather['target_time'],
                          'predicted_normalized_power': number(row['predicted_normalized_power'], 'predicted_normalized_power', 0, 1)})
        return {**data, 'records': clean}
    except DomainError as exc:
        if exc.status == 409:
            raise
        raise DomainError('invalid_ml_response', exc.message, 502)
