# Unified project acceptance — 2026-09-23

## Sources and result

Integrated Django/UI/weather from yevgeniy (d34ac60), the ML release from feature/wind-power-baseline (e345546), and main history through 28cefae. Original feature branches are preserved. README.md is the single setup/operation entry point; detailed training instructions are retained in TRAINING.md.

The original branches were not plug-compatible at metadata discovery: ML returns supported_weather_models and fields, while Django expected weather_model and field_definitions. The adapter now handles the actual service, validates versions/horizon/heights and preserves raw metadata and alignment. Django feature bounds now match the deployed model (wind 0–100 m/s, temperature −100–70 °C).

## Verified locally

Windows, Python 3.12.4, Django 5.2.17, NumPy 2.5.3, CPU inference on the unchanged committed artifact:

- Model version: mlp-76540e5972a6.
- Artifact SHA-256: 997066f7ddaeedbb6db840e2edfbaae947da487146b76949f076d1504b8ea55d — matches committed manifest.
- python -m pytest -q: **95 passed, 28 subtests passed**, 2 dependency deprecation warnings. Includes both source suites plus real loopback HTTP integration between Django and the actual ML model.
- python manage.py check: no issues.
- python manage.py makemigrations --check --dry-run: no pending model changes.
- node --check static/app.js: valid JavaScript.
- pip check: dependency consistency checked.
- inference/smoke_test.py against a separately running CPU service: both turbines reproduce reference predictions within 2e-6, preserve timestamps/schema/alignment, are repeatable, and reject invalid horizons/sources.
- Integration regression also verifies shared-token failure (401), no prediction fallback, metadata compatibility, Django persistence/CSV, version-aware cache reuse and unsupported feature ranges.
- tools/run_project.py starts the two local processes and waits for readiness. Without explicit provisional consent it refuses the committed artifact before starting services.
- Browser: actual model version shown, both turbines available; turbine_2 replay for 2026-02-01 through 2026-02-02 uses a previously retrieved real JMA snapshot and returns 48 real-model predictions. Mean 19.1%, peak 43.9% for that saved run (not a quality metric). The 24-hour toggle changes the display, not API horizon. alignment_confirmed=false and estimated weather issuance are visibly warned. No JavaScript console errors observed.
- Local documentation links checked; duplicate bilingual READMEs consolidated into README.md and docs/TRAINING.md.

A first full run with PyTorch 2.14.0+cpu failed to load c10.dll on this Windows, including outside the restricted execution environment. Replacing only the local test dependency with the supported CPU PyTorch 2.6.0 resolved it, and the full suite then passed. CI/test instructions pin 2.6.0. No production weights were retrained or edited. Production inference uses NumPy and does not need PyTorch. The GPU training environment versions in original reports remain historical facts, not this local test environment.

Two warnings are non-failing upstream deprecations: Starlette's httpx TestClient adapter and pandas/NumPy generic timedelta conversion in a dataset fixture.

## Weather verification inherited from the same workspace

With explicit user permission, real JMA GSM requests for turbine_2 (43.643198, 78.538828) returned 48 validated rows for live 2026-09-23 and replay 2026-02-01 00:00 to 2026-02-02 23:00 UTC with as_of 2026-01-31 18:00 UTC. New integration used the saved replay snapshot; no observations or reanalysis were substituted. Reference-based regression tests for both turbines do not require external weather access.

## Not claimed / still required externally

- Docker Engine is unavailable here. The combined Compose configuration is supplied for local demonstration, but image build/container execution has not been verified.
- Local tests do not establish hosted GitHub Actions status or public production deployment.
- alignment_confirmed remains false. Measurement interval start/end semantics still require organizer confirmation.
- Fixed-offset archive plus an 8-hour publication buffer does not prove exact historical forecast availability. Strict competition replay requires original verified issuance/publication records.
- No February target measurements or confirmed MW conversion are available, so neither February accuracy nor station MW output is claimed.

Old reports under reports/ and models/production/ are preserved as ML experiment/release evidence. Their earlier statements about pending teammate integration describe that prior stage; this document records the unified project's current acceptance.

Local SQLite, venv, generated test XML/temp folders, raw source documents, downloaded weather and credentials remain excluded from Git.
