import json
import numpy as np
import pandas as pd
import pytest
from windpower.dataset import assemble, sha256
from windpower.train import train
from windpower.model import Predictor


def test_weather_and_measurements_join_by_turbine_not_just_time(tmp_path):
    files = {}
    weather = []
    for turbine, speed, power in [("turbine_1", 2, .1), ("turbine_2", 8, .8)]:
        files[turbine] = tmp_path / (turbine + ".csv")
        pd.DataFrame({"Статистическое время": pd.date_range("2025-01-01", periods=6, freq="10min"), "Нормализованная активная мощность": power}).to_csv(files[turbine], index=False)
        for offset in (1, 2):
            weather.append(dict(turbine_id=turbine, model="jma_gsm", valid_time_utc="2025-01-01T00:00Z", provider_offset_days=offset, wind_speed_10m_ms=speed, wind_direction_10m_deg=90, temperature_2m_c=2))
    path = tmp_path / "weather.csv"
    pd.DataFrame(weather).to_csv(path, index=False)
    frame, _ = assemble(path, files, "UTC", "start")
    assert len(frame) == 4
    assert frame.loc[frame.turbine_id.eq("turbine_1"), "wind_speed_10m_ms"].eq(2).all()
    assert frame.loc[frame.turbine_id.eq("turbine_2"), "target_power"].tolist() == pytest.approx([.8, .8])


def test_joint_export_supports_both_ids_and_reports_separate_baselines(trained_fixture, tmp_path):
    root, _ = trained_fixture
    second = pd.read_csv(root / "training.csv")
    first = second.copy()
    first["turbine_id"] = "turbine_1"
    first["target_power"] = np.clip(first.target_power * .4, 0, 1)
    frame = pd.concat([first, second], ignore_index=True)
    path = tmp_path / "training.csv"
    frame.to_csv(path, index=False)
    metadata = json.loads((root / "training.metadata.json").read_text())
    metadata["dataset_sha256"] = sha256(path)
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata))
    report = train(path, tmp_path / "model", device="cpu", epochs=2, batch_size=64)
    predictor = Predictor(tmp_path / "model/model.json")
    assert predictor.turbines == ["turbine_1", "turbine_2"]
    for turbine in predictor.turbines:
        selected = frame.loc[frame.turbine_id.eq(turbine)]
        assert np.isfinite(predictor.predict(selected, "jma_gsm")).all()
        result = report["metrics_by_turbine"]["test"][turbine]
        assert result["models"]["neural"]["rows"] == 48
        assert result["unique_target_hours"] == 24
        assert set(result["neural_by_offset"]) == {"1", "2"}
    assert report["metrics_by_turbine"]["test"]["turbine_1"]["models"]["constant"]["mae"] != report["metrics_by_turbine"]["test"]["turbine_2"]["models"]["constant"]["mae"]
