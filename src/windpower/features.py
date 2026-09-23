"""One feature implementation shared by training and inference."""
import numpy as np
import pandas as pd

WEATHER_COLUMNS = ["wind_speed_10m_ms", "wind_direction_10m_deg", "temperature_2m_c"]
BASE_FEATURES = ["wind_speed_10m_ms", "wind_speed_cubed_scaled", "wind_direction_sin", "wind_direction_cos", "temperature_2m_c", "hour_sin", "hour_cos", "year_sin", "year_cos"]

def make_features(frame: pd.DataFrame, turbines: list[str]) -> np.ndarray:
    missing = set(WEATHER_COLUMNS + ["valid_time_utc", "turbine_id"]) - set(frame)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not len(frame):
        raise ValueError("No forecast rows")
    unknown = set(frame.turbine_id) - set(turbines)
    if unknown:
        raise ValueError(f"No trained model for turbines: {sorted(unknown)}")
    weather = frame[WEATHER_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(weather).all():
        raise ValueError("Weather values must be finite; missing values are not zero")
    speed, direction, temp = weather.T
    if (speed < 0).any() or ((direction < 0) | (direction > 360)).any():
        raise ValueError("Wind speed must be nonnegative and direction must be 0..360 degrees")
    t = pd.to_datetime(frame.valid_time_utc, utc=True, errors="raise")
    angle = np.deg2rad(direction)
    hour = 2 * np.pi * (t.dt.hour + t.dt.minute / 60) / 24
    year = 2 * np.pi * (t.dt.dayofyear - 1) / np.where(t.dt.is_leap_year, 366, 365)
    x = np.column_stack([speed, (speed / 20) ** 3, np.sin(angle), np.cos(angle), temp, np.sin(hour), np.cos(hour), np.sin(year), np.cos(year)] + [(frame.turbine_id == turbine).to_numpy(dtype=float) for turbine in turbines])
    if not np.isfinite(x).all():
        raise ValueError("Derived features must be finite")
    return x.astype(np.float32)

