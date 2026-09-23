# Wind-power ML service: implemented contract v1

This is the implemented HTTP contract in the unified main project, originally developed on feature/wind-power-baseline. It supersedes TEAM_INTEGRATION_PROPOSAL.md. Django integration and the actual metadata adapter are documented in ML_INTEGRATION.md; the root README is the single setup guide. No training job or weather download is triggered by direct inference.

## Ownership and flow

Frontend -> Django orchestrator -> weather module -> Django -> `POST /v1/predict` -> Django -> frontend.

Django chooses the turbine and horizon, obtains weather, validates its availability, persists the weather snapshot, calls this service, checks/saves predictions, and displays them. Our service loads a trained artifact at startup, applies its learned preprocessing and performs CPU inference. It needs no GPU or NVIDIA credential. Do not call `/power/forecast` for this integration: that optional convenience endpoint fetches weather itself.

Live mode: Django uses server time internally, selects the next UTC hour and obtains 48 hours. Replay mode: Django uses a simulated decision time and only weather available at that time. Both modes use the same ML endpoint and body. ML accepts past/future timestamps and does not compare them with its current clock. No issue time or `as_of` is supplied to ML. Provenance and leakage prevention belong to the weather/orchestration layer; unknown archive issue times must not be invented. Current JMA fixed-offset training is not proof of exact issue-time replay compliance.

For two turbines, make one request per turbine, optionally concurrently. Keep each turbine's weather snapshot, output and errors separate. Turbine 1 is at (43.645150, 78.535604), turbine 2 at (43.643198, 78.538828). Use each requested pin even when the provider resolves both to the same coarse weather grid cell. Never reuse one turbine's predicted output as the other's.

## Deployment and authentication

Configure Django:

```dotenv
ML_BACKEND=http
ML_SERVICE_URL=http://MODEL_HOST:8000/v1/predict
ML_SERVICE_TOKEN=YOUR_SHARED_SERVICE_TOKEN
```

`MODEL_HOST` must be reachable from Django. `127.0.0.1` works only when services share that network namespace; containers generally use a service DNS name. Set the same `ML_SERVICE_TOKEN` in the model service. When configured, it requires `Authorization: Bearer <token>` on `/v1/predict`, `/v1/metadata` and the convenience prediction endpoint. The token is a team-chosen service credential, not a Brev/NVIDIA API key. Omit it on both sides only for a local development service. `GET /health` is unauthenticated.

Run the model service with the committed weights at `models/production/model.json`:

```bash
export MODEL_PATH=models/production/model.json
export ALLOW_PROVISIONAL_MODEL=1
# Supply ML_SERVICE_TOKEN through your deployment's environment or secret manager.
uvicorn windpower.api:app --host 0.0.0.0 --port 8000
```

Current artifacts have unconfirmed measurement interval alignment, so the explicit provisional flag is necessary and the response reports `alignment_confirmed: false`. Display that limitation in the UI. Confirm/retrain before removing the flag. `WEATHER_PROVIDER_URL` is not needed for direct inference. A missing, invalid or disallowed model prevents startup. Models load once: restart the service after replacing the artifact. The selected joint weights and their hash manifest are committed in models/production/.

## Capability discovery

Before selecting a source, call authenticated `GET /v1/metadata`. It returns the exact loaded `model_version`, `supported_turbines`, `supported_weather_models`, field definitions, turbine coordinates, `required_records: 48`, schema versions and alignment status. The committed joint model mlp-76540e5972a6 supports both turbines. Metadata reports the artifact actually loaded; there is no fallback.

Current source: **`jma_gsm`**, with 10 m wind and 2 m temperature. Training and inference must use the same provider, height, units and grid/downscaling settings. Request Open-Meteo variables `wind_speed_10m,wind_direction_10m,temperature_2m`, `models=jma_gsm`, `wind_speed_unit=ms`, `temperature_unit=celsius`, `timezone=UTC`. Changing to GFS/ICON/ECMWF requires an appropriately trained/evaluated artifact; do not relabel another source as JMA. The transport schema can remain v1 for a future source using the same physical field definitions.

## Request

`POST /v1/predict`, `Content-Type: application/json`, optional Bearer header as configured. Exactly these top-level fields:

| Field | Type | Required value/meaning |
|---|---|---|
| schema_version | string | `windpower.input.v1` |
| turbine_id | string | `turbine_1` or `turbine_2`, supported by the loaded artifact |
| weather_model | string | Actual source identity; currently `jma_gsm` |
| records | array | Exactly 48 ordered consecutive hourly objects |

Each record contains exactly:

| Field | Type | Meaning |
|---|---|---|
| target_time | ISO 8601 string | Time being predicted. Explicit timezone required; whole UTC hour. Prefer `Z`; offsets are normalized to UTC. |
| wind_speed_10m_ms | JSON number | Forecast wind at 10 m above ground, m/s, 0..100 |
| wind_direction_10m_deg | JSON number | Wind at 10 m, meteorological direction FROM, degrees clockwise from north, 0..360 |
| temperature_2m_c | JSON number | Forecast temperature at 2 m, Celsius, -100..70 |

No missing/duplicate/out-of-order hours, nulls, NaN, infinity, numeric strings or booleans. Unknown fields are rejected. Bounds are sanity limits, not a guarantee of model accuracy at extremes. Do not include `pressure`, `forecast_age_hours`, issue times or per-row `weather_model` in this body; retain them in Django's snapshot if available. Do not substitute observed future wind for forecast wind.

The full executable example is [examples/direct-input.json](examples/direct-input.json). The following excerpt shows one row only; sending it alone returns 422:

```json
{
  "schema_version": "windpower.input.v1",
  "turbine_id": "turbine_2",
  "weather_model": "jma_gsm",
  "records": [
    {
      "target_time": "2026-02-01T00:00:00Z",
      "wind_speed_10m_ms": 5.2,
      "wind_direction_10m_deg": 220.0,
      "temperature_2m_c": -4.0
    }
  ]
}
```

For a 24-hour UI view, request 48 hours and display the first 24. Do not send a 24-row request to this v1 endpoint.

## Response

HTTP 200. Full shape below, shortened to one row. Values illustrate the contract, not a claimed forecast:

```json
{
  "schema_version": "windpower.output.v1",
  "turbine_id": "turbine_2",
  "weather_model": "jma_gsm",
  "model_version": "mlp-<artifact-hash>",
  "alignment_confirmed": false,
  "records": [
    {"target_time": "2026-02-01T00:00:00Z", "predicted_normalized_power": 0.31}
  ]
}
```

Always 48 records with matching timestamp instants and order. Power is a finite fraction in [0,1], not percent, kW, MW or energy. Do not sum normalized turbine values to claim plant MW without rated capacities. No fabricated uncertainty bands or accuracy are returned. The model response is deterministic for a fixed artifact and request, including historical dates; it contains no generated wall-clock timestamp.

The other branch already validates `model_version` and `records[].target_time/predicted_normalized_power`; those names remain compatible. Its input mapper and weather source/height settings must change. Preserve and surface `alignment_confirmed`, which its old validator currently discards.

## Errors, retries and caching

Errors use `{"error":{"code":"...","message":"..."}}`; invalid requests also include a `details` array of field paths and messages, without echoing input values.

| HTTP | Code | Action |
|---|---|---|
| 401 | unauthorized | Correct the shared Bearer token |
| 422 | invalid_request | Correct schema, values or hourly grid |
| 422 | unsupported_turbine | Deploy weights supporting this ID; check metadata |
| 422 | weather_model_mismatch | Select the source the artifact was trained with |
| 500 | invalid_model_output | Treat as model-service failure, never substitute dummy output |

Missing model prevents startup rather than returning fake predictions. Unexpected server/proxy errors can be non-JSON; handle all non-2xx responses as failures. Django should persist the upstream code/message where available. Do not retry 401/422. A bounded retry of network failure, timeout or 5xx is safe because inference has no side effects. Their existing 30-second timeout is adequate as an integration starting point; measure actual latency under load.

Cache by turbine ID + canonical input snapshot hash + actual returned model version. Detect model-version changes through metadata when deploying; do not reuse predictions solely because the weather snapshot is unchanged. A changed input or deployed model triggers a new prediction. The upstream archive retrieval/cache lifecycle belongs to Django, not this service.

## Files and acceptance checks

- [ml-input.schema.json](ml-input.schema.json), [ml-output.schema.json](ml-output.schema.json): generated from the actual Pydantic models.
- [openapi.json](openapi.json): generated API description; running service also exposes `/openapi.json` and `/docs`.
- [examples/direct-input.json](examples/direct-input.json): 48 synthetic weather rows for contract testing only.

```bash
curl http://MODEL_HOST:8000/v1/metadata -H "Authorization: Bearer $ML_SERVICE_TOKEN"
curl http://MODEL_HOST:8000/v1/predict -H 'Content-Type: application/json' -H "Authorization: Bearer $ML_SERVICE_TOKEN" --data-binary @docs/examples/direct-input.json
```

Local tests verify strict validation, no outbound weather request, deterministic predictions, both turbine IDs with an actually trained tiny test artifact, timezone normalization, source mismatch and authentication. Actual Django integration must still be exercised by the other party against a running deployed instance; repository endpoints are not public hosting URLs.
