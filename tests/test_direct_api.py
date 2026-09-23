import copy
import datetime as dt
import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from windpower.api import create_app
from windpower.model import Predictor


def payload():
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    return {"schema_version": "windpower.input.v1", "turbine_id": "turbine_2", "weather_model": "jma_gsm", "records": [
        {"target_time": (start + dt.timedelta(hours=i)).isoformat(), "wind_speed_10m_ms": 4 + i / 20, "wind_direction_10m_deg": 270, "temperature_2m_c": -5}
        for i in range(48)]}


def direct_client(trained_fixture, **kwargs):
    root, _ = trained_fixture
    def no_network(request):
        raise AssertionError("Direct inference must not fetch weather")
    return TestClient(create_app(root / "model/model.json", provider_url="", transport=httpx.MockTransport(no_network), **kwargs))


def test_direct_inference_without_weather_service_is_deterministic(trained_fixture):
    body = payload()
    with direct_client(trained_fixture) as client:
        first = client.post("/v1/predict", json=body)
        second = client.post("/v1/predict", json=body)
        assert first.status_code == 200, first.text
        assert first.json() == second.json()
        response = first.json()
        frame = pd.DataFrame(body["records"]).rename(columns={"target_time": "valid_time_utc"})
        frame["turbine_id"] = body["turbine_id"]
        model = Predictor(trained_fixture[0] / "model/model.json")
        np.testing.assert_allclose([r["predicted_normalized_power"] for r in response["records"]], model.predict(frame, "jma_gsm"), atol=1e-6)
        assert response["schema_version"] == "windpower.output.v1"
        assert response["records"][0]["target_time"] == "2026-01-01T00:00:00Z"
        assert client.post("/power/forecast", json={"turbine_id": "turbine_2"}).status_code == 503
        meta = client.get("/v1/metadata").json()
        assert meta["supported_turbines"] == ["turbine_2"]
        assert meta["supported_weather_models"] == ["jma_gsm"]
        assert meta["model_version"] == response["model_version"]
        assert meta["turbine_locations"]["turbine_2"]["latitude"] == 43.643198


@pytest.mark.parametrize("problem", ["short", "long", "duplicate", "gap", "unordered", "naive", "partial_hour", "numeric_time", "numeric_string", "boolean", "nan", "negative_wind", "extra_field", "wrong_version", "missing_version", "old_names"])
def test_direct_rejects_invalid_contract(trained_fixture, problem):
    body = payload()
    row = body["records"][0]
    if problem == "short": body["records"].pop()
    if problem == "long": body["records"].append(copy.deepcopy(body["records"][-1]))
    if problem == "duplicate": body["records"][1] = copy.deepcopy(row)
    if problem == "gap": body["records"][1]["target_time"] = "2026-01-01T05:00Z"
    if problem == "unordered": body["records"].reverse()
    if problem == "naive": row["target_time"] = "2026-01-01T00:00:00"
    if problem == "partial_hour": row["target_time"] = "2026-01-01T00:10:00Z"
    if problem == "numeric_time": row["target_time"] = 1767225600
    if problem == "numeric_string": row["wind_speed_10m_ms"] = "4"
    if problem == "boolean": row["wind_speed_10m_ms"] = True
    if problem == "nan": row["wind_speed_10m_ms"] = "NaN"
    if problem == "negative_wind": row["wind_speed_10m_ms"] = -1
    if problem == "extra_field": row["pressure"] = 950
    if problem == "wrong_version": body["schema_version"] = "unknown"
    if problem == "missing_version": del body["schema_version"]
    if problem == "old_names": row["wind_speed"] = row.pop("wind_speed_10m_ms")
    with direct_client(trained_fixture) as client:
        response = client.post("/v1/predict", json=body)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"


def test_wrong_source_and_untrained_turbine_are_explicit_errors(trained_fixture):
    with direct_client(trained_fixture) as client:
        body = payload(); body["weather_model"] = "gfs_global"
        assert client.post("/v1/predict", json=body).json()["error"]["code"] == "weather_model_mismatch"
        body = payload(); body["turbine_id"] = "turbine_1"
        assert client.post("/v1/predict", json=body).json()["error"]["code"] == "unsupported_turbine"


def test_authentication_applies_to_inference_and_metadata(trained_fixture):
    with direct_client(trained_fixture, service_token="test-only-token") as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/metadata").status_code == 401
        response = client.post("/v1/predict", json=payload(), headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"
        assert client.post("/v1/predict", json=payload(), headers={"Authorization": "Bearer test-only-token"}).status_code == 200


def test_timezone_offsets_are_normalized_and_do_not_change_predictions(trained_fixture):
    body = payload()
    equivalent = copy.deepcopy(body)
    for row in equivalent["records"]:
        row["target_time"] = dt.datetime.fromisoformat(row["target_time"]).astimezone(dt.timezone(dt.timedelta(hours=5))).isoformat()
    with direct_client(trained_fixture) as client:
        assert client.post("/v1/predict", json=body).json() == client.post("/v1/predict", json=equivalent).json()
