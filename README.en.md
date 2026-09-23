> The complete integrated application is now available: [setup and verification](docs/PRE_FINAL.en.md). This document covers the model and training details.

# Wind-power forecasting: integration and operations

[Русская версия](README.ru.md) · [Training toolkit](training_toolkit/README.en.md) · [Exact API contract](docs/ML_SERVICE_CONTRACT.md)

## 1. What is delivered

This repository contains the team's weather-to-power component: a neural network trained from random initialization, its committed weights, a CPU HTTP inference service, reference requests for both turbines, tests, and a separate training toolkit. It is ready for the other developer to connect and exercise. It is not a hosted public API URL, and it does not implement the team's complete dashboard or agent.

The selected release is **`mlp-76540e5972a6`**, in [`models/production/model.json`](models/production/model.json). A clone contains everything needed to run predictions; there is no separate weight download. No raw turbine measurement CSVs, NVIDIA keys, service tokens, billing coupons, or local job credentials are committed.

The artifact supports **both turbine IDs** and **JMA GSM** weather. Predictions are normalized power fractions, not energy or MW. Inference needs neither a GPU nor PyTorch, CatBoost, Brev, an LLM, or access to a weather API. Training does not happen during startup or prediction requests.

### Readiness and limitations

- Joint CUDA training completed. CPU inference reproduced all 2,956 January evaluation predictions within approximately 1.82e-7.
- Both turbines have reference cases from real archived weather and an executable HTTP smoke test.
- Measurement timestamps are interpreted in `Asia/Almaty`, with historical UTC-offset changes. Whether each timestamp marks the start or end of its ten-minute interval remains unconfirmed. Current training assumes **start**.
- `alignment_confirmed` is **false**. Startup requires `--allow-provisional` or `ALLOW_PROVISIONAL_MODEL=1`. The response preserves this flag.
- Provider fixed-offset archives do not establish the precise forecast issue/publication time required for a fully auditable competition replay.
- Supplied measurements end January 31, 2026. February labels were not supplied; no February accuracy result is claimed.
- The pre-final integration passed real Django-to-model HTTP tests for both turbine locations; see `reports/pre-final-live.json`. Docker build/run could not be verified because no Docker engine was available. Executed release checks are recorded in [`reports/release_verification.json`](reports/release_verification.json).

## 2. Repository map

```text
README.en.md / README.ru.md         Full integration documentation
models/production/
  model.json                       Selected weights, scaling and provenance
  manifest.json                    Model identity, SHA-256 and selection rationale
  metrics.json                     Training history and evaluation metrics
  independent_verification.json     Independent export/API verification
inference/
  serve.py                         CPU launcher, default port 8001
  smoke_test.py                    HTTP checks against reference cases
  examples/                        48-hour requests/responses for both turbines
training_toolkit/
  run.py                           Separate model-creation command entry point
  README.en.md / README.ru.md       Archive, dataset, training and Brev commands
src/windpower/                     Shared features, model, API and training code
scripts/                          Low-level controllers and verification tools
docs/                             JSON schemas, OpenAPI and contracts
reports/                          Dataset audits and measured experiments
tests/                            Functional and real-artifact checks
requirements-runtime.txt           Pinned inference dependencies
Dockerfile / compose.yaml          Container deployment configuration
```

The toolkit imports the shared feature implementation instead of copying it, preventing train/serve drift. Existing `scripts/` commands remain supported for earlier job launchers and receipts.

## 3. Quick start: Python 3.12, CPU only

Run commands from the repository root. Use **Python 3.12** for the pinned environment verified here. The package declares Python >=3.11, but these release pins were exercised with 3.12.

### Linux, macOS or WSL

```bash
git clone --branch feature/wind-power-baseline https://github.com/BAITC-Hacks/hack-15d1fa83-team.git
cd hack-15d1fa83-team
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-runtime.txt .
python inference/serve.py --allow-provisional
```

The private repository requires your own GitHub access. If already cloned, use the existing checkout of this branch. Do not save a GitHub token in a command committed to the repository.

### Windows PowerShell: WSL is not needed for serving

```powershell
git clone --branch feature/wind-power-baseline https://github.com/BAITC-Hacks/hack-15d1fa83-team.git
cd hack-15d1fa83-team
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt .
.\.venv\Scripts\python.exe inference\serve.py --allow-provisional
```

Calling the venv interpreter directly avoids PowerShell activation-policy issues. WSL is used for the tested Brev CLI workflow, not CPU inference.

The default listener is **http://127.0.0.1:8001**. Keep the terminal open; stop with Ctrl+C. `/docs` provides interactive API documentation and `/health` reports readiness. The launcher finds the bundled artifact relative to the repository; no custom `MODEL_PATH` is needed.

### Verify the running release

In a second terminal, from the same repository/environment:

```bash
python inference/smoke_test.py --url http://127.0.0.1:8001
```

PowerShell:

```powershell
.\.venv\Scripts\python.exe inference\smoke_test.py --url http://127.0.0.1:8001
```

This sends both real archived 48-hour examples, verifies model identity, timestamps and predictions, repeats requests, and checks rejection of incomplete horizons and a wrong weather source. With `ML_SERVICE_TOKEN`, it also checks authentication. Failure produces a nonzero exit code. These are historical integration fixtures, not current forecasts.

## 4. Deployment configuration

| Setting | Meaning |
|---|---|
| `MODEL_PATH` | Optional artifact override. Direct Uvicorn defaults to `models/production/model.json` relative to the working directory; `inference/serve.py` uses the repository path. |
| `ALLOW_PROVISIONAL_MODEL` | `1` accepts unconfirmed alignment; default `0`. |
| `ML_SERVICE_TOKEN` | Shared Bearer token for Django and ML. Empty disables auth for local development. |
| `WEATHER_PROVIDER_URL` | Legacy `/power/forecast` only; leave unset for direct inference. |
| `WEATHER_PROVIDER_TOKEN` | Legacy weather endpoint token; not an NVIDIA credential. |
| `WEATHER_TIMEOUT_SECONDS` | Legacy weather timeout, default 15 seconds. |
| `ALLOW_HISTORICAL_FORECASTS` | Legacy endpoint only; no effect on `/v1/predict`. |

The application does **not** automatically read `.env`. Export variables, configure a process manager, or use Compose. `.env.example` documents names. The shared service token is team-chosen, not a Brev API key.

For authentication, set the same private value in both processes. Linux: `export ML_SERVICE_TOKEN='your-private-token'`. PowerShell: `$env:ML_SERVICE_TOKEN='your-private-token'`. The smoke-test terminal needs the same value. Never commit the real token.

For other machines/container networks, start with `--host 0.0.0.0 --port 8000`, enable authentication, and configure networking/reverse proxy. `0.0.0.0` is a bind address, not Django's destination URL. Use HTTPS at the reverse proxy for public traffic.

### Docker Compose

Compose now starts both Django and ML, with a private model network and web at localhost:18080. Follow the [integrated startup guide](docs/PRE_FINAL.en.md), including configuration generation and account creation. No host ML port is exposed. Compose validation passed; Docker build/run remains unverified because the local engine was unavailable.

## 5. Request flow and ownership

```text
User / scheduler -> Django agent -> weather module
                 -> Django stores weather snapshot
                 -> POST /v1/predict, once per turbine
                 -> Django validates/stores response -> dashboard
```

The weather/orchestration side chooses live or replay time, obtains forecasts for the right coordinates, verifies historical availability and retains provenance. ML accepts prepared weather, validates it and returns power plus model identity. ML does not obtain latest weather, invent issue times, poll for updates, schedule recalculation, or store team forecasts.

A changed weather snapshot or deployed model triggers a new Django request. Cache by turbine ID, canonical weather snapshot hash and actual model version. Identical inputs and artifact produce identical output; inference has no write side effects.

| Turbine ID | Latitude | Longitude |
|---|---:|---:|
| `turbine_1` | 43.645150 | 78.535604 |
| `turbine_2` | 43.643198 | 78.538828 |

Use each turbine's coordinates and ID. Nearby pins can resolve to the same coarse grid, but target histories and turbine features remain distinct. Never copy one turbine's power prediction to the other.

## 6. HTTP interface

| Endpoint | Purpose | Auth if configured |
|---|---|---|
| `GET /health` | Readiness and loaded identity | No |
| `GET /v1/metadata` | Turbines, source, units, coordinates, alignment | Yes |
| `POST /v1/predict` | Prepared weather to power | Yes |
| `GET /docs`, `GET /openapi.json` | API documentation | No |
| `POST /power/forecast` | Optional legacy weather fetching; not the Django integration path | Yes |

The request contains exactly four top-level fields:

| Field | Meaning |
|---|---|
| `schema_version` | Exactly `windpower.input.v1` |
| `turbine_id` | `turbine_1` or `turbine_2` |
| `weather_model` | Exactly `jma_gsm` for these weights |
| `records` | Exactly 48 ordered consecutive hourly objects |

Each record contains exactly:

| Field | Units and validation |
|---|---|
| `target_time` | ISO 8601 string, explicit timezone, whole UTC hour; prefer `Z`. Explicit offsets normalize to UTC. |
| `wind_speed_10m_ms` | JSON number, 0..100, m/s at 10 m above ground |
| `wind_direction_10m_deg` | JSON number, 0..360; meteorological direction FROM, clockwise from north, at 10 m |
| `temperature_2m_c` | JSON number, -100..70, Celsius at 2 m |

Nulls, booleans, numeric strings, non-finite numbers, gaps, duplicates, out-of-order hours and unknown fields are rejected. Bounds are sanity checks, not accuracy guarantees. Pressure, 100 m wind, forecast age and issue time are not ML input fields.

Full executable requests: [`turbine_1-48h-request.json`](inference/examples/turbine_1-48h-request.json), [`turbine_2-48h-request.json`](inference/examples/turbine_2-48h-request.json). For a 24-hour UI, send 48 rows and display the first 24. Sending 24 rows to v1 returns 422.

```bash
curl http://127.0.0.1:8001/v1/predict \
  -H 'Content-Type: application/json' \
  --data-binary @inference/examples/turbine_1-48h-request.json
```

With authentication, add `-H "Authorization: Bearer $ML_SERVICE_TOKEN"`. In PowerShell use `curl.exe`, one line, and `$env:ML_SERVICE_TOKEN`.

Response: `schema_version: windpower.output.v1`, `turbine_id`, `weather_model`, `model_version`, `alignment_confirmed`, and 48 `records`. Each record has `target_time` and `predicted_normalized_power`. Values are finite in [0,1], and timestamps preserve requested instants/order. Exact reference responses are beside the request files.

**Power interpretation:** 0.5 means half of normalized full scale. It is not automatically 0.5 MW or MWh. Conversion requires organizer-confirmed capacity and normalization definition. Summing two normalized fractions does not directly produce plant MW.

### Weather-source consistency

For Open-Meteo, use `models=jma_gsm`, `hourly=wind_speed_10m,wind_direction_10m,temperature_2m`, `wind_speed_unit=ms`, `temperature_unit=celsius`, `timezone=UTC`, and each turbine's coordinates. Keep grid/downscaling settings consistent with training. Do not relabel another source as JMA or 100 m wind as 10 m wind. A provider change requires appropriately trained/evaluated weights; the schema can remain unchanged when physical inputs remain unchanged.

## 7. Django integration

```dotenv
ML_BACKEND=http
ML_SERVICE_URL=http://127.0.0.1:8001/v1/predict
ML_SERVICE_TOKEN=
```

This URL assumes the processes share the host outside separate containers. Minimal Python caller:

```python
import os
import httpx

def predict_power(payload):
    token = os.getenv('ML_SERVICE_TOKEN', '')
    headers = {'Authorization': 'Bearer ' + token} if token else {}
    response = httpx.post(os.environ['ML_SERVICE_URL'],
                          json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    result = response.json()
    if result['turbine_id'] != payload['turbine_id'] or len(result['records']) != 48:
        raise ValueError('Unexpected prediction response')
    return result
```

The caller should also validate schema version, finite output range, timestamps and model identity, and preserve `alignment_confirmed` in storage/UI. Store issue-time/provenance information with the weather snapshot, not as unknown fields in this request. Call once per turbine, optionally concurrently.

### Errors, retries and model updates

| HTTP / code | Meaning | Action |
|---|---|---|
| 401 / `unauthorized` | Missing/wrong shared token | Fix credentials; no unchanged retry |
| 422 / `invalid_request` | Wrong schema or hourly series | Correct request |
| 422 / `unsupported_turbine` | Loaded weights lack this ID | Check metadata/deployment |
| 422 / `weather_model_mismatch` | Wrong forecast source | Correct source/artifact pairing |
| 500 / `invalid_model_output` | Output validation failed | Record failure; no fabricated power fallback |
| Timeout, connection error, other 5xx | Network/process/proxy failure | Bounded retries |

Structured errors contain `error.code` and `error.message`; invalid requests also contain `error.details`. Unexpected server/proxy failures may be non-JSON. Handle status before assuming JSON. Missing or invalid model files prevent startup. Weights load once: restart after replacement and invalidate caches keyed by the previous model version.

## 8. Model, dataset and measured results

The selected model has 2,881 parameters: 64 → 32 → 1 dense layers, ReLU hidden activations, sigmoid output. Features derive from three forecast variables, cyclic UTC hour/day-of-year and turbine identity. Scaling uses training rows only. Training used random initialization, AdamW, MSE loss, and validation-MAE early stopping. It ran 28 GPU epochs; checkpoint 18 was selected.

The joint dataset has 96,900 rows representing 48,450 turbine-hour targets, each with two archive offsets. Hourly targets require six valid ten-minute readings. Missing/ambiguous hours are not fabricated. Turbine 1 contributes 23,666 complete hours; turbine 2 contributes 24,784. See [`reports/dataset_assembly_both.json`](reports/dataset_assembly_both.json) for coverage and source hashes.

| UTC period | Role |
|---|---|
| Before October 2025 | Weight fitting and scaling |
| October–December 2025 | Checkpoint/model selection |
| January 2026 | Evaluation |
| February 2026 | Competition target period; no supplied labels for scoring |

All offset versions of a target hour remain in the same split. January contains 2,956 examples but only 739 distinct target hours per turbine; offset copies and neighboring hours are correlated.

| January metric | Combined | Turbine 1 | Turbine 2 |
|---|---:|---:|---:|
| MAE | 0.178525 | 0.173897 | 0.183153 |
| RMSE | 0.238635 | 0.232772 | 0.244357 |
| Bias, prediction minus target | +0.066447 | +0.059136 | +0.073758 |

Errors are fractions of normalized full scale, not MAPE or an accuracy percentage. Combined MAE is 29.1% below the empirical wind-curve baseline (0.251898). The constant baseline MAE is 0.302035.

### Why retain this model?

The [comparison](docs/MODEL_COMPARISON.md) tested different losses, a larger network, CatBoost, separate turbine models and ensembles. The validation-selected MAE ensemble improved January MAE to 0.166403 but worsened RMSE to 0.245725. The user chose to retain the squared-error model to favor lower large-error sensitivity. RMSE penalizes large errors more strongly; it does not guarantee every extreme prediction is better.

A CPU MSE control had slightly lower January RMSE (0.235943) but worse 2025 validation MAE. The release was not switched based on January alone. January was inspected during development and is not an untouched external benchmark. No February accuracy or guaranteed future improvement is claimed.

## 9. Creating another model

Use the separate [`training_toolkit/`](training_toolkit/README.en.md) entry point and documentation. It covers archives, both CSVs, timezone handling, CPU/GPU training, Brev, comparisons, verification and promotion. Data access is needed for retraining, not inference.

Serving uses `requirements-runtime.txt`. Training uses the `train` extra; comparisons also use `benchmark`. Runtime installation does not install those extras. Put new candidates in new directories; do not silently overwrite the selected release.

## 10. Checks and troubleshooting

Developer regression suite:

```bash
python -m pip install -e '.[train,test]'
python -m pytest -q
```

This installs PyTorch for tiny training fixtures. The HTTP smoke test and inference do not need it. Tests cover input errors, chronology, export parity, both real release cases and controllers. GitHub Actions is configured; local passing checks do not prove the hosted CI runner completed. Inspect the actual GitHub run.

| Symptom | Check |
|---|---|
| `ModuleNotFoundError: windpower` | Install root package using the interpreter that starts the service. |
| Provisional/alignment startup error | Explicitly allow the provisional model; do not edit metadata to fake confirmation. |
| Model missing | Use repository launcher or valid `MODEL_PATH`; confirm weights were cloned. |
| Port occupied | Stop old service or use `--port 8002`; update caller URL. |
| 422 | Exact fields, numeric types, 48 whole UTC hours, source and turbine ID. |
| 401 | Same token on both sides, with `Bearer ` header prefix. |
| Container cannot connect | Use service DNS/network address, not caller-container localhost. |
| Legacy endpoint returns 503 | Use `/v1/predict`; legacy weather fetching is unconfigured. |
| New weights have no effect | Restart the service. |
| Poor forecast skill | Check source, units, coordinates and alignment before retraining. Never substitute observed future weather. |

Acceptance flow: clone, install runtime only, start service, discover both turbines/JMA, pass reference requests, then send one actual snapshot per turbine from Django and verify persistence, display and errors. The teammate's branch is unchanged.
