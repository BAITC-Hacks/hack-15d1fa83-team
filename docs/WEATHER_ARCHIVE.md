# Weather module — JMA GSM, 10 m

Only forecasts, never observations or reanalysis. The trained artifact requires JMA GSM wind at 10 m and temperature at 2 m. No silent replacement by wind at 100 m or another model.

## Sources

- Live: https://api.open-meteo.com/v1/forecast. Django selects now and the next whole-hour start internally. Requests `models=jma_gsm`, hourly wind_speed_10m, wind_direction_10m, temperature_2m, wind_speed_unit=ms, temperature_unit=celsius, timezone=UTC. Map to wind_speed_10m_ms, wind_direction_10m_deg, temperature_2m_c.
- Replay convenience source: https://previous-runs-api.open-meteo.com/v1/forecast. Uses those variables with _previous_dayN suffixes. N = ceil((target_time − as_of + 8 hours)/24 hours), minimum 1, maximum 7. This is a forecast archive, not Historical Weather/ERA5.
- Exact imported archive: a verified JMA forecast run supplied by the data owner with issue/publication metadata.
- Demo: explicit synthetic data for offline UI development only. Never sent to the external ML adapter.

Official provider references: [Forecast](https://open-meteo.com/en/docs), [JMA](https://open-meteo.com/en/docs/jma-api), [Previous Runs](https://open-meteo.com/en/docs/previous-runs-api).

## Availability caveat

Previous Runs represents fixed lead times, not exact named issuance runs. We estimate issued_at as target minus N days and available_at as issued_at + 8 hours. The snapshot is visibly marked timing=estimated_fixed_lead. This buffer does NOT prove historical availability and is not sufficient for strict leakage-free competition evaluation. The required archive/model/period can be unavailable; that becomes a visible error, not substituted weather.

For strict replay, import exact verified runs. The importer requires archived_forecast, weather_model=jma_gsm, wind_height_m=10, temperature_height_m=2, aware issued_at and available_at, source_url, turbine_id and 48–384 ordered hourly records with the three explicit features. Optional pressure (hPa) remains in the snapshot only. The owner is responsible for truthful source timestamps. Reject observations/reanalysis; select newest run whose available_at and issued_at are <= simulated decision time and which covers all 48 hours. Null features, gaps and duplicates are rejected.

The live API does not expose verified issuance timestamps: issued_at, available_at, forecast_age_hours and lead_time_hours are null. Retrieval time is stored in provenance; it is not presented as issue time. Live snapshots are not treated as exact historical archive runs.

## Snapshot schema and storage

New snapshots: windpower.weather.v2. Provenance includes source URL, mode, wind/temperature height, request parameters, timing quality and (for Open-Meteo) retrieval time and raw-response hash. Internal rows retain pressure, weather_model, issued_at, available_at, age and lead time. ML receives only the four fields per record required by windpower.input.v1.

Existing snapshots and forecasts are kept readable without conversion; old 100 m records cannot enter new ML inference. Existing imported runs are marked legacy by migration and excluded from selection. Reimport verified JMA 10 m data, not relabelled 100 m values.

Weather cache distinguishes schema, turbine, decision time, start, provider, weather model and mode. Archive selection is rerun so a newer available run can be chosen. Explicit refresh refetches weather. Volatile retrieval timestamps/raw response generation metadata do not invalidate unchanged contents within the same window. Prediction cache also requires turbine and fresh loaded model version.

`python manage.py import_weather path/to/verified-run.json` imports an exact run. The bundled [synthetic example](examples/archived-run-synthetic.json) deliberately uses data_kind=synthetic and is REJECTED; it demonstrates format, not real archive provenance.
