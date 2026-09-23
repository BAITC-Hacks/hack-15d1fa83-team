import datetime as dt
import json
import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from windpower.api import create_app
from windpower.features import make_features
from windpower.model import Predictor

def weather(start=None):
    start = start or dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    return dict(turbine_id="turbine_2", weather_model="jma_gsm", hourly=[dict(time=(start+dt.timedelta(hours=i)).isoformat(), wind_speed_10m_ms=4+i/10, wind_direction_10m_deg=90, temperature_2m_c=2) for i in range(48)])

def client_for(trained_fixture, payload=None, historical=True, transport=None):
    root, _ = trained_fixture
    transport = transport or httpx.MockTransport(lambda request: httpx.Response(200, json=payload or weather()))
    return TestClient(create_app(root/"model"/"model.json", "https://weather.test/weather/forecast", allow_historical=historical, transport=transport))

def test_training_exports_matching_feature_scaling_and_real_weights(trained_fixture):
    root, report = trained_fixture
    artifact = json.loads((root/"model"/"model.json").read_text())
    frame = pd.read_csv(root/"training.csv")
    train_rows = frame.loc[frame.valid_time_utc < "2025-10-01"]
    np.testing.assert_allclose(artifact["mean"], make_features(train_rows, ["turbine_2"]).mean(axis=0))
    assert report["parameter_count"] > 1000
    assert report["best_epoch"] >= 1
    assert set(report["metrics"]["test"]) == {"neural", "constant", "wind_curve"}
    pred = Predictor(root/"model"/"model.json").predict(frame, "jma_gsm")
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    assert np.ptp(pred) > 0

def test_on_demand_service_fetches_and_predicts(trained_fixture):
    calls = []
    def provider(request):
        calls.append(request)
        return httpx.Response(200, json=weather())
    with client_for(trained_fixture, transport=httpx.MockTransport(provider)) as client:
        result = client.post("/power/forecast", json={"turbine_id": "turbine_2", "hours": 48})
        assert result.status_code == 200
        body = result.json()
        assert len(body["hourly"]) == 48
        assert calls[0].url.params["turbine_id"] == "turbine_2"
        assert all(0 <= h["normalized_power"] <= 1 for h in body["hourly"])
        assert client.get("/health").json()["status"] == "ready"

@pytest.mark.parametrize("problem", ["gap", "duplicate", "missing", "wrong_model", "wrong_turbine", "nan", "short", "naive_time"])
def test_invalid_provider_data_never_becomes_a_prediction(trained_fixture, problem):
    payload = weather()
    if problem == "gap": payload["hourly"].pop(1)
    if problem == "duplicate": payload["hourly"][1] = payload["hourly"][0]
    if problem == "missing": del payload["hourly"][0]["temperature_2m_c"]
    if problem == "wrong_model": payload["weather_model"] = "gfs_global"
    if problem == "wrong_turbine": payload["turbine_id"] = "turbine_1"
    if problem == "nan": payload["hourly"][0]["wind_speed_10m_ms"] = "NaN"
    if problem == "short": payload["hourly"] = payload["hourly"][:5]
    if problem == "naive_time": payload["hourly"][0]["time"] = "2026-01-01T00:00:00"
    with client_for(trained_fixture, payload) as client:
        assert client.post("/power/forecast", json={"turbine_id": "turbine_2"}).status_code == 502

def test_unknown_turbine_rejected_without_provider_call(trained_fixture):
    def provider(request): raise AssertionError("Provider should not be called")
    with client_for(trained_fixture, transport=httpx.MockTransport(provider)) as client:
        assert client.post("/power/forecast", json={"turbine_id": "turbine_1"}).status_code == 422

def test_stale_provider_rejected_in_live_mode(trained_fixture):
    with client_for(trained_fixture, historical=False) as client:
        assert client.post("/power/forecast", json={"turbine_id": "turbine_2"}).status_code == 502

def test_live_next_hour_is_accepted(trained_fixture):
    start = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)+dt.timedelta(hours=1)
    with client_for(trained_fixture, weather(start), historical=False) as client:
        assert client.post("/power/forecast", json={"turbine_id": "turbine_2", "hours": 24}).status_code == 200

def test_transient_provider_error_retried(trained_fixture):
    calls = []
    def provider(request):
        calls.append(request)
        return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, json=weather())
    with client_for(trained_fixture, transport=httpx.MockTransport(provider)) as client:
        assert client.post("/power/forecast", json={"turbine_id": "turbine_2"}).status_code == 200
    assert len(calls) == 2

def test_provisional_artifact_requires_explicit_development_setting(trained_fixture, tmp_path):
    root, _ = trained_fixture
    artifact = json.loads((root/"model"/"model.json").read_text())
    artifact["dataset_metadata"]["alignment_confirmed"] = False
    path = tmp_path/"model.json"; path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="provisional"): Predictor(path)
    assert Predictor(path, allow_provisional=True)

def test_missing_artifact_prevents_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path/"missing.json", "https://weather.test")): pass
