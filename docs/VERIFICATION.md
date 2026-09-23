# Verification record

Verified locally on Windows, Python 3.12, Django 5.2.17.

- `python manage.py check`: no issues.
- `python manage.py makemigrations --check --dry-run`: no changes.
- `python manage.py test`: 15 tests pass, including historical availability checks, malformed weather/model responses, CSRF/Bearer authentication, snapshot reuse and changed-input inference.
- JavaScript syntax checked with Node `--check`; Python modules compiled.
- Browser: demo forecast returns 48 hours, graph and hourly slider render, weather table has 48 rows, history opens stored runs. Desktop 1440px and mobile 390px layouts visually inspected; mobile chart uses fewer readable time labels.
- Actual network smoke test: Open-Meteo Previous Runs `gfs_global`, coordinates 43.645150 / 78.535604, decision 2026-01-31 18:00 UTC, horizon 2026-02-01 00:00 through 2026-02-02 23:00 UTC. Complete normalized 48-row weather snapshot and demo power output saved locally. No observed weather used. This verifies API access and mapping, not the provider's exact historical publication timestamps.
- Actual local HTTP integration: separate `tools/mock_ml_service.py` receives request, returns 48 records and `mock-contract-v1`; Django accepts and marks the result as demo.
- One-day `backtest --provider demo` smoke test produces a 48-row CSV with metadata; this is a workflow test, not a model accuracy test.

Local SQLite, output CSVs, virtual environments and input PDF/image are intentionally not committed. No trained model is included. Strict competition replay still requires source-verified archival issue/availability metadata and the other developer's trained model.
