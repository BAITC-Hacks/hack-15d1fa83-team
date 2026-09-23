import numpy as np
import pandas as pd
import pytest
from windpower.dataset import assemble, hourly_targets
from windpower.features import make_features
from windpower.train import split_masks

def measurements(path, dates, powers=None):
    pd.DataFrame({"Статистическое время": dates, "Нормализованная активная мощность": powers if powers is not None else [0.4]*len(dates)}).to_csv(path, index=False)

def test_local_timezone_changes_in_2024(tmp_path):
    path = tmp_path / "t.csv"
    dates = list(pd.date_range("2023-03-11 06:00", periods=6, freq="10min")) + list(pd.date_range("2025-03-11 05:00", periods=6, freq="10min"))
    measurements(path, dates)
    frame, audit = hourly_targets(path, "turbine_2", "Asia/Almaty", "start")
    assert frame.valid_time_utc.dt.hour.tolist() == [0, 0]
    assert audit["complete_hours"] == 2

def test_incomplete_and_invalid_hours_not_filled(tmp_path):
    path = tmp_path / "t.csv"
    dates = pd.date_range("2025-01-01", periods=17, freq="10min")
    measurements(path, dates, [0.4]*6 + [0.3]*5 + [float("nan")] + [0.7]*5)
    frame, audit = hourly_targets(path, "turbine_2", "UTC", "start")
    assert len(frame) == 1
    assert audit["incomplete_hours"] == 2
    assert audit["invalid_or_ambiguous_rows"] == 1

def test_interval_end_changes_bucket(tmp_path):
    path = tmp_path / "t.csv"
    measurements(path, pd.date_range("2025-01-01 00:10", periods=6, freq="10min"))
    frame, _ = hourly_targets(path, "turbine_2", "UTC", "end")
    assert len(frame) == 1 and frame.valid_time_utc.iloc[0].hour == 0
    start, _ = hourly_targets(path, "turbine_2", "UTC", "start")
    assert start.empty

def test_duplicate_measurements_rejected(tmp_path):
    path = tmp_path / "t.csv"
    measurements(path, ["2025-01-01"]*2)
    with pytest.raises(ValueError, match="Duplicate"):
        hourly_targets(path, "turbine_2", "UTC", "start")

def test_assembly_excludes_february_and_missing_weather(tmp_path):
    path = tmp_path / "t.csv"
    dates = list(pd.date_range("2026-01-31 23:00", periods=12, freq="10min"))
    measurements(path, dates)
    rows = []
    for timestamp in ["2026-01-31T23:00Z", "2026-02-01T00:00Z"]:
        for offset in [1, 2]:
            rows.append(dict(turbine_id="turbine_2", model="jma_gsm", valid_time_utc=timestamp, provider_offset_days=offset, wind_speed_10m_ms=4, wind_direction_10m_deg=90, temperature_2m_c=2))
    rows[1]["wind_speed_10m_ms"] = np.nan
    weather = tmp_path / "w.csv"; pd.DataFrame(rows).to_csv(weather, index=False)
    joined, audit = assemble(weather, {"turbine_2": path}, "UTC", "start")
    assert len(joined) == 1 and joined.target_power.iloc[0] == pytest.approx(0.4)
    assert audit["missing_weather_rows_dropped"] == 1
    assert not audit["alignment_confirmed"]

def test_direction_wrap_and_unknown_turbine():
    rows = pd.DataFrame(dict(turbine_id=["turbine_2"]*2, valid_time_utc=["2025-01-01T00:00Z"]*2, wind_speed_10m_ms=[4]*2, wind_direction_10m_deg=[0, 360], temperature_2m_c=[2]*2))
    x = make_features(rows, ["turbine_2"])
    np.testing.assert_allclose(x[0], x[1], atol=1e-6)
    with pytest.raises(ValueError, match="No trained model"):
        make_features(rows, ["turbine_1"])

def test_all_offsets_for_a_target_stay_in_same_split():
    frame = pd.DataFrame(dict(valid_time_utc=["2025-09-30T23:00Z"]*2+["2025-10-01T00:00Z"]*2+["2026-01-01T00:00Z"]*2))
    masks = split_masks(frame, "2025-10-01", "2026-01-01", "2026-02-01")
    assert [int(mask.sum()) for mask in masks.values()] == [2, 2, 2]
    assert not (masks["train"] & masks["validation"]).any()
