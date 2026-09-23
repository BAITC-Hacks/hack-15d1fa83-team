# Verification record — contract adaptation

Verified locally on Windows, Python 3.12.4, Django 5.2.17, 2026-09-23.

- Django system check: no issues. New migrations add nullable forecast alignment, response schema, and archive schema (legacy for existing rows). Applied locally without deleting existing data.
- Django test suite: strict ML input/output schemas, aware consecutive 48-hour grid, finite numbers, unknown-field rejection, live server time, replay availability, exact JMA 10 m mapping, legacy exclusion, optional token, metadata turbine support, version-aware caching, rollout race, saved 401/422/timeout failures and no dummy fallback.
- JavaScript syntax checked with Node --check.
- Browser: live request succeeds with no user time inputs; replay exposes decision time and horizon start; alignment_confirmed=false is visible. 24-hour display shows first 24 hours while CSV/JSON remain full 48. Graph, metadata panel and weather use the new fields. Desktop layout inspected.
- With explicit user permission, actual Open-Meteo requests for turbine_2 (43.643198, 78.538828) returned 48 validated JMA GSM records for live 2026-09-23 and replay 2026-02-01 00:00 through 2026-02-02 23:00 UTC, as_of 2026-01-31 18:00 UTC. Wind is 10 m, m/s; temperature 2 m, Celsius. Live issuance metadata remains null. Replay is marked estimated_fixed_lead, NOT proven exact historical availability.
- Actual local HTTP integration with tools/mock_ml_service.py: Django queried metadata, obtained JMA weather and sent the new strict body to POST /v1/predict. Received mock-contract-v2, 48 predictions, alignment=false; repeated identical replay reused the prediction cache.
- Mock output is not trained ML and provides no evidence of forecast accuracy.

Real trained-service deployment URL and concrete GET /v1/metadata response have not been supplied. Exact metadata field mapping must be confirmed against that service before real-artifact integration sign-off. No trained model, training pipeline, accuracy metrics, MW conversion or assumed alignment confirmation is included.

The old verification of GFS/100 m is not evidence for this contract. Old snapshots are retained but excluded from new inference. Local SQLite, test outputs, virtual environment, input PDF/image and secrets are not committed.
