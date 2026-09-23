"""Generate the service schemas and a complete synthetic weather request example."""
import datetime as dt
import json
from pathlib import Path
from windpower.api import create_app
from windpower.contracts import PredictRequest, PredictResponse


def main():
    docs = Path(__file__).resolve().parents[1] / "docs"
    (docs / "examples").mkdir(exist_ok=True)
    for name, model in [("ml-input.schema.json", PredictRequest), ("ml-output.schema.json", PredictResponse)]:
        schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **model.model_json_schema()}
        (docs / name).write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    (docs / "openapi.json").write_text(json.dumps(create_app().openapi(), indent=2) + "\n", encoding="utf-8")
    start = dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc)
    body = dict(schema_version="windpower.input.v1", turbine_id="turbine_2", weather_model="jma_gsm", records=[
        dict(target_time=(start + dt.timedelta(hours=i)).isoformat().replace("+00:00", "Z"), wind_speed_10m_ms=round(4.5 + (i % 12) * .2, 2), wind_direction_10m_deg=220.0, temperature_2m_c=-4.0)
        for i in range(48)])
    PredictRequest.model_validate(body)
    (docs / "examples/direct-input.json").write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print("Exported schemas, OpenAPI and the complete 48-hour synthetic example.")


if __name__ == "__main__":
    main()
