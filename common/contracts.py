import math
from datetime import datetime, timezone, timedelta


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
    except ValueError:
        raise DomainError('invalid_time', f'{field}: требуется ISO 8601 со смещением часового пояса.')


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def number(value, field, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise DomainError('invalid_value', f'{field}: требуется конечное число от {low} до {high}.')
    return float(value)


def text(value, field, max_length=160):
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise DomainError('invalid_value', f'{field}: требуется непустая строка (до {max_length} символов).')
    return value


def request_window(data):
    if not isinstance(data, dict):
        raise DomainError('invalid_body', 'Ожидается JSON-объект.')
    turbine = text(data.get('turbine_id'), 'turbine_id', 64)
    as_of = timestamp(data.get('as_of'), 'as_of')
    start = timestamp(data.get('start_time'), 'start_time')
    if start.minute or start.second or start.microsecond:
        raise DomainError('invalid_window', 'start_time должен совпадать с началом часа UTC.')
    if start <= as_of or start > as_of + timedelta(hours=24):
        raise DomainError('invalid_window', 'Начало горизонта должно быть после as_of, не более чем через 24 часа.')
    return turbine, as_of, start


def validate_weather(records, start, as_of):
    if not isinstance(records, list) or len(records) != 48:
        raise DomainError('invalid_horizon', 'Нужны ровно 48 последовательных почасовых записей.')
    clean = []
    for i, row in enumerate(records):
        if not isinstance(row, dict):
            raise DomainError('invalid_record', f'Запись {i} должна быть объектом.')
        target = timestamp(row.get('target_time'), 'target_time')
        issued = timestamp(row.get('issued_at'), 'issued_at')
        available = timestamp(row.get('available_at'), 'available_at')
        if target != start + timedelta(hours=i):
            raise DomainError('invalid_grid', 'Часы должны идти по порядку, без пропусков и повторов.')
        if issued > available or available > as_of or issued > target:
            raise DomainError('future_leakage', 'Прогноз погоды не был доступен на момент as_of.')
        clean.append({
            'target_time': iso(target), 'issued_at': iso(issued), 'available_at': iso(available),
            'wind_speed': number(row.get('wind_speed'), 'wind_speed (m/s)', 0, 150),
            'wind_direction': number(row.get('wind_direction'), 'wind_direction (degrees)', 0, 360),
            'temperature': number(row.get('temperature'), 'temperature (C)', -100, 70),
            'pressure': number(row.get('pressure'), 'surface pressure (hPa)', 300, 1100),
            'weather_model': text(row.get('weather_model'), 'weather_model'),
            'forecast_age_hours': (as_of - issued).total_seconds() / 3600,
            'lead_time_hours': (target - issued).total_seconds() / 3600,
        })
    return clean


def validate_predictions(data, records):
    if not isinstance(data, dict):
        raise DomainError('invalid_ml_response', 'ML-сервис вернул не JSON-объект.', 502)
    version = text(data.get('model_version'), 'model_version')
    values = data.get('records')
    if not isinstance(values, list) or len(values) != 48:
        raise DomainError('invalid_ml_response', 'ML-сервис должен вернуть 48 записей.', 502)
    clean = []
    for row, weather in zip(values, records):
        if not isinstance(row, dict) or timestamp(row.get('target_time')) != timestamp(weather['target_time']):
            raise DomainError('invalid_ml_response', 'Временная сетка ML-ответа не совпадает с запросом.', 502)
        clean.append({'target_time': weather['target_time'],
                      'predicted_normalized_power': number(row.get('predicted_normalized_power'), 'predicted_normalized_power', 0, 1)})
    return {'records': clean, 'model_version': version}
