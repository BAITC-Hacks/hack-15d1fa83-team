# ML integration — windpower.input.v1 / windpower.output.v1

## Ownership and flow

Frontend → Django → weather module → prepared weather → ML service → Django → frontend.

Django owns retrieval, forecast selection, historical availability checks, caching, immutable weather snapshots and forecast storage. The ML service uses its loaded artifact for inference only. It does not fetch weather or train during a request. One request is made per turbine; callers may run independent requests concurrently.

Live requests use Django's current UTC time and start at the next whole UTC hour. Users do not enter issue times. Replay requests use a simulated decision time `as_of` plus the first target hour. Neither timestamp belongs in the ML body.

## POST /v1/predict

Content-Type: application/json. Optional `Authorization: Bearer <ML_SERVICE_TOKEN>`: configure the same shared token on Django and the ML service only if the ML service requires it. This is not an NVIDIA key. Django's inbound `API_TOKEN` is a separate credential.

Exact top-level fields:

| Field | Value |
| --- | --- |
| schema_version | windpower.input.v1 |
| turbine_id | Supported ID reported by the loaded artifact |
| weather_model | jma_gsm |
| records | Exactly 48 objects |

Each record contains ONLY:

| Field | Meaning |
| --- | --- |
| target_time | Aware ISO timestamp; preferably UTC Z; ordered consecutive whole hours |
| wind_speed_10m_ms | Wind speed 10 m above ground, m/s, nonnegative |
| wind_direction_10m_deg | Direction wind comes FROM, clockwise from north, 0–360° |
| temperature_2m_c | Air temperature 2 m above ground, Celsius |

Values must be finite JSON numbers, never strings, null, booleans, NaN or Infinity. Unknown fields are rejected. Gaps, duplicates, fractional hours and naive timestamps are rejected. Django and ML both enforce wind speed 0–100 m/s and temperature −100–70 °C.

Pressure, forecast age, issue/publication time and provenance stay in the weather snapshot. They are NOT sent to ML. No 100 m substitution or fallback to GFS/ICON is permitted. Changing weather source requires retraining/deployment and a contract update.

See [complete 48-hour input](examples/ml-input.json) and [JSON Schema](ml-input.schema.json). Schemas cover shape/types; Python validation additionally enforces the temporal grid and response alignment.

## HTTP 200 response

Exact fields: `schema_version=windpower.output.v1`, `turbine_id`, `weather_model`, `model_version`, boolean `alignment_confirmed`, and exactly 48 `records`.

Every record contains `target_time` matching the corresponding request hour and `predicted_normalized_power` in [0,1]. This is a fraction, not MW or percent. The UI converts to percentages only for display; JSON/CSV retain fractions. Selecting 24 hours shows the first 24 of a full 48-hour response.

Django preserves `alignment_confirmed` verbatim in the database, JSON, CSV and UI. False is a visible warning, not an inference failure or an implicit true. Older results have null/unknown status; migration does not invent confirmation.

[Complete output example](examples/ml-output.json), [output schema](ml-output.schema.json).

## GET /v1/metadata — verified joint-model structure

The committed artifact is mlp-76540e5972a6 and supports both turbine_1 and turbine_2. The service actually returns:

```json
{
  "input_schema_version": "windpower.input.v1",
  "output_schema_version": "windpower.output.v1",
  "model_version": "mlp-76540e5972a6",
  "supported_turbines": ["turbine_1", "turbine_2"],
  "supported_weather_models": ["jma_gsm"],
  "required_records": 48,
  "time_step_hours": 1,
  "timezone": "UTC",
  "fields": {
    "wind_speed_10m_ms": {"unit": "m/s", "height_m": 10},
    "wind_direction_10m_deg": {"unit": "degrees clockwise from north, direction FROM", "height_m": 10},
    "temperature_2m_c": {"unit": "Celsius", "height_m": 2}
  },
  "alignment_confirmed": false
}
```

Additional metadata includes turbine_locations and archive_semantics. Django validates the schema versions, hourly horizon and feature heights, then exposes normalized weather_model/field_definitions at GET /api/v1/ml/metadata/. It retains the full raw upstream metadata. Legacy aliases remain only for test-adapter compatibility.

Fresh metadata is required before cache lookup. Support comes from the loaded artifact, not configured Django turbine rows. Default upstream metadata URL is the /metadata sibling of ML_SERVICE_URL; ML_METADATA_URL can override it.

The committed model requires explicit --allow-provisional or ALLOW_PROVISIONAL_MODEL=1 because measurement interval alignment remains unconfirmed. Never flip the artifact flag just to make it start.

## Weather and caching

Open-Meteo: `models=jma_gsm`, `hourly=wind_speed_10m,wind_direction_10m,temperature_2m`, `wind_speed_unit=ms`, `temperature_unit=celsius`, `timezone=UTC`. Previous Runs uses the corresponding `_previous_dayN` variables for historical fixed-lead retrieval. See [weather module](WEATHER_ARCHIVE.md) for availability limitations.

Snapshots carry `schema_version=windpower.weather.v2` and explicit feature heights. Existing 100 m snapshots are left intact and excluded from the new inference path. Existing archive rows default to legacy; reimport verified 10 m JMA records to use them.

Prediction cache key: immutable weather snapshot ID + turbine ID + loaded model_version (also response schema and demo mode). Fresh metadata is required even on cache hits. A new artifact version forces a new call. A response version different from metadata fails visibly with 409 so a deployment race cannot poison the cache. Identical refreshes reuse predictions; changed weather produces a new snapshot. Exact live decision timestamps intentionally identify separate snapshots, so separate live clicks need not hit the cache.

## Failures and operational configuration

- 401: shared service token rejected.
- 422: malformed features, unsupported turbine or weather source mismatch.
- 502: malformed output/metadata or upstream service failure.
- 504: timeout.
- 409: incompatible old snapshot or artifact changed during request.

Failures during orchestration are saved in history with an empty prediction list and the error. Invalid public request shape is rejected before a run is created. There is NO automatic dummy fallback. Explicit `ML_BACKEND=demo` is an offline development mode, clearly marked; synthetic weather is rejected when `ML_BACKEND=http`.

PowerShell, from Source:

```powershell
$env:ML_BACKEND = 'http'
$env:ML_SERVICE_URL = 'https://YOUR-DEPLOYMENT/v1/predict'
$env:ML_SERVICE_TOKEN = 'same-shared-token-if-required'
.venv/Scripts/python.exe manage.py migrate
.venv/Scripts/python.exe manage.py runserver
```

Omit/unset the token if not configured upstream. `.env.example` is documentation; .env is not automatically loaded. Restart Django after changes. A GitHub branch is not a hosted API. The unified repository includes the actual CPU service and weights. Local real-artifact integration is covered by tests/test_unified_integration.py; an external hosted URL is only needed if deploying on another machine.

For a local HTTP-only contract test: `python tools/mock_ml_service.py`. It defaults to turbine_2, version mock-contract-v2 and alignment=false. Set `MOCK_SUPPORTED_TURBINES=turbine_1,turbine_2` to simulate joint deployment. `MOCK_PORT` defaults to 8001; `MOCK_MODEL_VERSION` and `ML_SERVICE_TOKEN` are configurable. This is a test server, not trained ML. With Django HTTP mode use real weather/an imported verified archive, not the synthetic UI provider.

## Public Django requests

Live:
```json
{"mode":"live","turbine_id":"turbine_2","provider":"open_meteo","weather_model":"jma_gsm"}
```

Replay:
```json
{"mode":"replay","turbine_id":"turbine_2","provider":"archive","weather_model":"jma_gsm","as_of":"2026-01-31T18:00:00Z","start_time":"2026-02-01T00:00:00Z"}
```

POST these bodies to `/api/v1/forecasts/` or the independent `/api/v1/weather/forecast/`. Live rejects client-supplied as_of/start_time; replay requires a past aware as_of and a whole-hour start strictly after it, at most 24 hours later. No public request accepts issued_at.
