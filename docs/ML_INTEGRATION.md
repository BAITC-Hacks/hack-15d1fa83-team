# ARYS — design and ML integration contract v1.0

Audience: the developer/agent implementing the trained wind-power model. Python 3.12 / Django 5.2 backend; vanilla frontend. This branch does not train a model or consume measured power. No other developer's branch is modified.

## Responsibilities and boundaries

The Django **forecasting module** owns request parameters, turbine metadata, weather acquisition, strict validation of 48 consecutive hours, calls to your service, validation of returned power, persistence, UI, CSV, run trace and change-triggered recalculation. The **weather module** owns historical forecast selection and provenance. Your **ML service** owns preprocessing learned during training, model loading, trained inference, reproducible model versioning and its own runtime/dependencies.

Workflow: request a power forecast -> retrieve a weather snapshot -> validate/prepare 48 hourly rows -> POST your model -> validate 48 predictions -> persist/analyse -> show/export. `forecasting/services.py:ml_payload` is the only mapping into the ML contract. Replace the adapter if a different transport is agreed; do not put model training into Django views.

## HTTP transport

- Set `ML_BACKEND=http`. Django sends a synchronous POST to the complete `ML_SERVICE_URL` (default `http://127.0.0.1:8001/v1/predict`). That endpoint is implemented **by your service**, not by Django.
- UTF-8 JSON; `Content-Type: application/json`, `Accept: application/json`. Optional `Authorization: Bearer <ML_SERVICE_TOKEN>`.
- 200 with a JSON object on success. Error/non-JSON responses become a persisted upstream error; timeout is 504, other failures 502. Default connect/read timeout is 30 seconds via `HTTP_TIMEOUT` (requests timeout, not a total wall-clock deadline).
- No automatic fallback to demo, no automatic repeated inference on a failed POST. Inference should be deterministic and free of side effects.
- Request and response schemas: `ml-input.schema.json`, `ml-output.schema.json`; complete executable examples: `examples/ml-input.json`, `examples/ml-output.json` (synthetic contract fixtures, not training/evaluation data).

## Request (exact names)

Top-level `turbine_id` string (`turbine_1` / `turbine_2`) and `records` array of exactly 48 objects:

| Field | Meaning | Unit / constraints |
| --- | --- | --- |
| target_time | Time this prediction is for | ISO 8601 UTC with Z, exact hour; strictly ascending, interval 1 hour |
| wind_speed | Predicted wind speed at **100 m** | m/s, finite, [0,150] |
| wind_direction | Meteorological wind direction at **100 m**, direction from which wind blows | degrees clockwise from north, [0,360] |
| temperature | Forecast air temperature at **2 m** | Celsius, [-100,70] |
| pressure | Forecast **surface** pressure, not sea-level pressure | hPa, [300,1100] |
| weather_model | Actual selected model identifier | string: gfs_global, icon_global, ecmwf_ifs025, or imported identifier |
| forecast_age_hours | Age of forecast at the decision time | `(as_of - issued_at).total_seconds()/3600`, nonnegative hours |

**Age is not lead time.** Lead time is `target_time - issued_at`; the weather API also returns `lead_time_hours` but this is deliberately not passed in the minimal ML contract. With one exact imported run, all 48 records have the same age. With fixed-lead archive rows, nominal initialization times differ and ages can differ; do not assume one issuance per request. `issued_at` and `available_at` are normalized archive metadata and remain in weather snapshots, not in the ML body.

Do not silently use the existing SCADA wind-height convention, Kelvin, Pa, km/h or mean sea-level pressure. The model developer must explicitly agree this weather-height and feature definition matches training. Convert direction to sine/cosine inside your pipeline if needed. Turbine ID mapping is fixed here; map it to your own internal IDs. If rated power is needed for normalization, own and version that mapping in the ML project.

No nulls, NaN, Infinity, missing hours, duplicates, out-of-order rows or automatic interpolation. Invalid weather is rejected before inference. The documented bounds are sanity checks, not measured turbine limits. A model receiving weather outside its training distribution should raise a meaningful error or implement its own explicitly versioned strategy.

## Response

Top-level `model_version`: nonempty stable identifier, maximum 160 characters, ideally training version + artifact hash. Top-level `records`: exactly 48 objects with:

- `target_time`: matches the corresponding input time. Offsets are normalized to UTC during validation; array order must remain the same.
- `predicted_normalized_power`: finite JSON number **0 ≤ p ≤ 1**, not a percentage, kW or MW. Overshoots are rejected, never silently clipped by Django.

No per-record model_version is required. Additional fields are ignored; do not rely on them reaching the UI. Uncertainty intervals are out of the current contract. A response `model_version=mock-contract-v1` or `demo-*` must be marked as demonstration by the integration layer.

## Public Django API

`POST /api/v1/forecasts/` accepts turbine_id, as_of, start_time, provider, weather_model. Start is a full UTC hour strictly after as_of and at most 24h later; horizon is always 48 records `[start_time, start_time + 48h)`. The UI and API expose **only 48h** because this is the agreed model contract. Case allows 24–48h; a separate shorter ML contract is unnecessary.

`POST /api/v1/weather/forecast/` accepts the same parameters and returns the weather records independently of ML. `GET /api/v1/forecasts/{uuid}/ml-input/` gives the exact body for replaying inference locally. See `openapi.json` and README for other paths and authentication. External clients should use an environment-configured API_TOKEN, never browser CSRF exemptions.

Forecast success: HTTP 201, status completed, records, model_version, weather, snapshot_id, trace, analysis, is_demo. Upstream failure: HTTP 502/504 with a saved forecast object and error `{code,message,status}`; clients can retrieve the failed run by id. Validation errors before run creation use `{error:{code,message}}` and 400/422; unknown IDs use 404. All frontend rendering uses textContent for untrusted values.

## Historical integrity

See `WEATHER_ARCHIVE.md`. Never treat ERA5, observations, a stitched recent-hours forecast series, or a retrospectively generated hindcast as a forecast demonstrably available at the decision time. Imported exact runs enforce `issued_at <= available_at <= as_of`; the supplying developer is responsible for proving source provenance and publication timestamps. Fixed-lead Open-Meteo archive rows carry clearly identified estimated metadata and an 8h publication margin. They are not proof of exact historical public availability.

For final competition evaluation, use source-verified exact archived runs and keep their provenance. Forecast snapshots are content-addressed (SHA256 of inputs and provenance), immutable through public APIs, and retained with every run. An imported source URL is metadata, never fetched by the import endpoint. This avoids user-supplied-URL server requests.

## Agent behaviour and recalculation

The orchestrator is a deterministic state machine: request -> weather -> validation -> model -> analysis -> completed/failed. Trace stores stage timestamps and elapsed milliseconds. Analysis includes mean/peak normalized power, sum of hourly normalized power (equivalent full-load hours), maximum adjacent-hour ramp and explicit warnings. There are no accuracy metrics until actual generation is supplied.

`POST /forecasts/{id}/refresh/` re-fetches the weather with the same historical decision time. A changed snapshot triggers inference, an unchanged one reuses the old prediction. This checks input updates, not model deployment changes: explicitly create a new forecast after deploying a new model. `python manage.py refresh_forecasts` performs one bounded sweep; schedule with the server's task scheduler if needed. No scheduler is installed automatically. Backtest command runs daily decisions, preserving overlapping target times under separate forecast IDs.

## Integration acceptance checklist

1. Start your service and Django in HTTP mode; use `docs/examples/ml-input.json` as a POST body for an isolated smoke check.
2. Return 48 matching timestamps, finite values within [0,1] and a real artifact version. Confirm both turbine IDs and weather models are handled.
3. Fetch real archived weather for February 2026, agree on height/units/age semantics, then run a full Django forecast.
4. Verify a timeout, an invalid model response and unavailable archive fail visibly rather than substituting synthetic output.
5. Replay each daily decision without training on February targets; use exact archive mode for strict timing and report evaluation metrics separately in your project.

## Deliberate scope limits

No training framework dependency, no ML code merged from another branch, no fabricated uncertainty or accuracy, no aggregate MW without turbine ratings. SQLite + synchronous HTTP is a hackathon deployment choice, not a production job queue. A killed request process can leave status running; production needs worker ownership/heartbeat and recovery. Credentials are environment variables and are never placed in saved request bodies.
