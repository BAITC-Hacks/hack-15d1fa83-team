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

Values must be finite JSON numbers, never strings, null, booleans, NaN or Infinity. Unknown fields are rejected. Gaps, duplicates, fractional hours and naive timestamps are rejected. Django additionally rejects temperatures below absolute zero.

Pressure, forecast age, issue/publication time and provenance stay in the weather snapshot. They are NOT sent to ML. No 100 m substitution or fallback to GFS/ICON is permitted. Changing weather source requires retraining/deployment and a contract update.

See [complete 48-hour input](examples/ml-input.json) and [JSON Schema](ml-input.schema.json). Schemas cover shape/types; Python validation additionally enforces the temporal grid and response alignment.

## HTTP 200 response

Exact fields: `schema_version=windpower.output.v1`, `turbine_id`, `weather_model`, `model_version`, boolean `alignment_confirmed`, and exactly 48 `records`.

Every record contains `target_time` matching the corresponding request hour and `predicted_normalized_power` in [0,1]. This is a fraction, not MW or percent. The UI converts to percentages only for display; JSON/CSV retain fractions. Selecting 24 hours shows the first 24 of a full 48-hour response.

Django preserves `alignment_confirmed` verbatim in the database, JSON, CSV and UI. False is a visible warning, not an inference failure or an implicit true. Older results have null/unknown status; migration does not invent confirmation.

[Complete output example](examples/ml-output.json), [output schema](ml-output.schema.json).

## GET /v1/metadata — adapter boundary to confirm

Django reads fresh metadata before every inference/cache decision. Support comes from the loaded artifact, never from the two turbines registered in Django. The turbine-2 artifact must report only turbine_2; add turbine_1 only when joint weights are deployed.

The teammate contract specifies metadata semantics but not its exact JSON structure. Until an actual response is provided, the adapter in `forecasting/ml_client.py:get_metadata` expects:

```json
{
  "model_version": "mlp-<artifact-hash>",
  "supported_turbines": ["turbine_2"],
  "weather_model": "jma_gsm",
  "field_definitions": {
    "wind_speed_10m_ms": "10 m wind, m/s",
    "wind_direction_10m_deg": "FROM north clockwise, degrees",
    "temperature_2m_c": "2 m air temperature, Celsius"
  }
}
```

Accepted aliases: `turbine_ids`; `weather_source` as string or object containing `weather_model`/ `model`; `input_fields` or `feature_names` for field definitions. Missing required capability information produces a visible 502; there is no invented default turbine support. Raw metadata is retained in the normalized API response for inspection. Confirm this mapping against the real deployed service before integration sign-off.

Django exposes normalized metadata at `GET /api/v1/ml/metadata/`. Default upstream URL is the sibling `/v1/metadata` of `ML_SERVICE_URL`; override with `ML_METADATA_URL` for a gateway.

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

Omit/unset the token if not configured upstream. `.env.example` is documentation; .env is not automatically loaded. Restart Django after changes. A GitHub branch is not a hosted API. No real deployment URL or real metadata response has been supplied, so real-artifact end-to-end compatibility still needs verification.

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
