# Complete application — pre-final

[Русская версия](PRE_FINAL.ru.md) · [Model documentation](../README.en.md) · [Training toolkit](../training_toolkit/README.en.md)

## What is integrated

This branch merges `feature/wind-power-baseline` at `e3455465897a6e9ef84b7963404cf4b5ae0dcba6` and `yevgeniy` at `d34ac600506a974bdf1e71b72e672fe89cacd96a`. It does not require changes to `main` or either source branch. The selected weights are unchanged.

Django owns the dashboard, authentication, weather retrieval, immutable weather snapshots, history, validation, refresh and export. A separate FastAPI process loads the committed NumPy model and converts each 48-hour JMA GSM weather window to normalized power. Neither runtime trains a model or needs GPU/NVIDIA credentials.

```text
Browser / authenticated API client
  -> Django (web:8000, host localhost:18080)
     -> GET windpower-ml:8000/v1/metadata
     -> Open-Meteo JMA GSM at the selected turbine coordinates
     -> validate units/heights/hours/values; save weather snapshot
     -> POST windpower-ml:8000/v1/predict
     -> validate response/version, save history and analysis, render/export
```

Metadata was the blocking incompatibility: the model returns `supported_weather_models` and `fields`; the original web adapter expected `weather_model` and `field_definitions`. The adapter now maps the actual contract and retains explicit older aliases. Web-side feature bounds match the ML service. HTTP inference is the default and the dashboard hides synthetic weather in that mode.

## Docker setup

Prerequisites: Git, Python 3.12+ for generating configuration, Docker Compose v2 with a running Linux-container engine. Use a **separate checkout** so work on main stays independent. Do not switch an actively used main checkout.

```sh
git clone --branch pre-final https://github.com/BAITC-Hacks/hack-15d1fa83-team.git windpower-pre-final
cd windpower-pre-final
python deployment/configure.py
docker compose config --quiet
docker compose up --build -d
docker compose ps
docker compose exec web python manage.py createsuperuser
```

Cloning requires access to the private repository. `createsuperuser` asks you to choose login credentials; there is no default password. Open **http://localhost:18080**, log in, select either turbine, choose Open-Meteo and create a live forecast. The 24-hour UI option changes display only; inference always receives 48 hours.

`configure.py` uses only the standard library. It creates `.env` with random `DJANGO_SECRET_KEY`, `API_TOKEN`, `ML_SERVICE_TOKEN`, project name and port. It never prints secrets and refuses to overwrite an existing file. Keep the generated `.env` across restarts. Compose reads it automatically; ordinary Python/Django commands do not.

The images have separate environments: `Dockerfile` installs the pinned ML runtime; `Dockerfile.web` installs Django, requests, Waitress and WhiteNoise. Both run as non-root users. No PyTorch, CUDA, CSV dataset or training job is included. `.dockerignore` admits only required source and model files.

ML is internal to the Compose network at `http://windpower-ml:8000/v1/predict`; this setup does **not** expose host port 8001. Web binds to loopback **18080**. Project name **windpower-pre-final** scopes its containers, network and database volume. There are no fixed container names. If the port is occupied, change `WEB_PORT` in `.env`.

Web startup runs migrations, `seed_demo`, and `collectstatic`, then starts Waitress. Despite its legacy name, `seed_demo` only creates the two real turbine identities/coordinates; it creates no synthetic forecasts and preserves existing turbine settings. WhiteNoise serves static assets with debug disabled. The SQLite database persists in the project volume.

```sh
docker compose logs --tail=100 web windpower-ml
docker compose stop
docker compose start
# Remove this stack's containers/network, retain its database:
docker compose down
```

`down -v` would delete saved data. For internet-facing use add HTTPS, configure allowed hosts, secure cookies and redirects; the supplied profile is local HTTP. `API_TOKEN` authenticates web API clients; `ML_SERVICE_TOKEN` is the shared web-to-model secret. Neither is an NVIDIA key. Browser sessions use Django login and CSRF protection.

## Native setup and verification

Use Python 3.12 and a fresh environment, separate from any other running checkout:

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux alternative: source .venv/bin/activate
python -m pip install -r requirements-runtime.txt -r requirements.txt -e .
python -m pip check
python deployment/verify_stack.py --live --report reports/local-stack.json
```

The verifier starts two real HTTP services on free loopback ports, uses random credentials and a temporary database, exercises both turbines and stops only its own processes. `--live` retrieves current JMA weather; omit it for startup, metadata, authentication, static assets and real-model fixture checks without weather access. It does not invent issue times. The temporary database is discarded afterwards. Static collection creates `staticfiles/` in this checkout.

For interactive native development, run `python inference/serve.py --allow-provisional --port 18081` in one terminal. In another activated terminal:

```powershell
$env:ML_BACKEND = 'http'
$env:ML_SERVICE_URL = 'http://127.0.0.1:18081/v1/predict'
python manage.py migrate
python manage.py seed_demo
python manage.py runserver 127.0.0.1:18080
```

This native example uses Django's development/debug settings and is for loopback only. If `ML_SERVICE_TOKEN` is set, it must match in both terminals. The Docker profile provides debug-disabled authentication and persistence.

The complete suite needs CPU PyTorch for small training tests:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
python manage.py check
python manage.py makemigrations --check --dry-run
pytest -q
```

## Interfaces and ownership

| Interface | Purpose |
|---|---|
| Web `POST /api/v1/forecasts/` | Body `{"turbine_id":"turbine_1","mode":"live","provider":"open_meteo"}`; substitute turbine_2 for the other location |
| Web `GET /api/v1/ml/metadata/` | Fresh normalized capabilities from the loaded artifact |
| Web `GET /api/v1/forecasts/<id>/` | Saved result, weather, provenance, trace and warnings |
| Web `POST /api/v1/forecasts/<id>/refresh/` | Refetch/recompute with history |
| Web `GET /api/v1/forecasts/<id>/export.csv` | 48 predictions plus header |
| Web `GET /api/v1/forecasts/<id>/ml-input/` | Exact payload sent to the model |
| ML `GET /health` | Public liveness |
| ML `GET /v1/metadata`, `POST /v1/predict` | Token-protected service boundary |

References: [dashboard OpenAPI](dashboard-openapi.json), [ML OpenAPI](openapi.json), [input schema](ml-input.schema.json), [output schema](ml-output.schema.json), [integration contract](ML_INTEGRATION.md), [weather/replay](WEATHER_ARCHIVE.md). The two OpenAPI documents describe different servers.

| Turbine | Latitude | Longitude |
|---|---|---|
| turbine_1 | 43.645150 | 78.535604 |
| turbine_2 | 43.643198 | 78.538828 |

Each turbine gets its own location request and model identity. Nearby points may share similar gridded weather; turbine identities remain separate. Inputs: 10 m wind speed in m/s (0–100), 10 m direction FROM north clockwise in degrees (0–360), 2 m temperature in Celsius (-100–70), 48 consecutive whole UTC hours. No 100 m feature is required. Output is normalized power [0,1], not MW; physical conversion needs the correct rating and normalization definition.

The workflow is deterministic orchestration, without an LLM decision-maker or continuously running refresh scheduler. `python manage.py refresh_forecasts --limit 20` runs one sweep over saved successful windows; scheduling is external. Live refresh uses current server time and weather; replay retains historical decision time. Fresh metadata is required before cache reuse. Failures do not trigger fallback to synthetic data, another weather model or dummy power.

## Evidence and limits

On 2026-09-23: **93 tests and 31 subtests passed**, pip found no dependency conflicts, Django checks passed and no missing migrations were found. A real two-server test with debug disabled and authentication enabled retrieved live JMA weather for both coordinates, produced 48 hours each, persisted/exported the results and exactly matched direct model predictions. See [live report](../reports/pre-final-live.json) and [verification summary](../reports/pre-final-verification.json).

Compose configuration passed validation. Docker Engine was unavailable, so image builds, Linux container execution and volume ownership are **not yet runtime-verified**. The combined suite is configured in GitHub Actions; local success does not establish remote CI success.

`alignment_confirmed=false` remains truthful: measurement interval semantics need confirmation. We retained the selected MSE-trained model; integration tests do not establish new accuracy. See the model README for held-out metrics.

Live weather issue/availability times remain null when unknown. Previous Runs replay uses fixed lead offsets and estimated availability, with visible warnings. Strict competition replay needs verified archived run provenance; February ground-truth evaluation remains separate. Passing integration is not proof of forecast accuracy or exact historical availability.
