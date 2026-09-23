# Training data and provenance

Locations resolved from the organizer's map pins:

| Turbine | Latitude | Longitude |
|---|---:|---:|
| turbine_1 | 43.645150 | 78.535604 |
| turbine_2 | 43.643198 | 78.538828 |

The supplied turbine-2 CSV has 149,499 records, mostly ten-minute intervals, spanning March 11, 2023 through January 31, 2026. The filename mentions February but the file has no February records. Only this turbine's targets are currently available.

## Forecast source

[Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api), explicitly `jma_gsm`, provides the three required fields at `previous_day1` and `previous_day2` offsets. The collected archive has no missing requested JMA values across the nominal training range. The reproducible downloader preserves raw response caches, exact request URLs, retrieval times and checksums.

The [JMA API documentation](https://open-meteo.com/en/docs/jma-api) describes this global GSM product as 0.5-degree spatial resolution and six-hourly source values interpolated to hourly output. Do not describe this as native hourly, turbine-resolution weather. The turbines can share a weather grid point. Their rows are not independent weather realizations.

The returned fixed-offset series does not expose the original run/publication time per value. Do not construct an apparent issue time simply by subtracting 24 or 48 hours. The current model is a first weather-to-power baseline, not a completed exact-as-of historical replay. Original NOAA GFS files were separately sampled successfully in March 2023 and January 2026; full original-run collection remains separate work.

This draft excludes pressure and 100 m wind from its model contract so the same feature set is available throughout the JMA training period. GFS point-archive winds had earlier gaps; they are not silently mixed into JMA data.

## Time alignment and targets

The user identifies the timestamps as local time. Use `Asia/Almaty` for the supplied locations, not a constant UTC+5 across all years. Kazakhstan introduced nationwide UTC+5 on March 1, 2024 ([government notice](https://www.gov.kz/memleket/entities/mfa-jakarta/press/news/details/715495?lang=en)). Any source-system historical clock convention still needs confirmation.

The interval-start/end convention has not been supplied. The assembled draft uses explicit `start` semantics and remains marked provisional. Ambiguous transition-hour timestamps are excluded and reported. Six valid ten-minute power readings are required for an hourly mean; partially observed hours are excluded. No interpolation of the target is performed.

Input variables are archived forecast wind speed, direction and temperature. The measured wind and temperature columns are not used as substitutes for forecast features. Target power outside 0..1 or non-finite values is excluded and counted. The 0..1 range is observed in the data; its exact engineering normalization should still be confirmed.

## Splits and interpretation

Target timestamps before 2025-10-01 UTC form training. October–December 2025 is validation; January 2026 is held-out evaluation. February targets are forbidden. Forecast offsets for the same turbine and target hour remain in the same split. Scaling and comparison baselines use training labels only.

A 24-hour and a 48-hour forecast for the same target are two predictor examples with the same label. The row count therefore overstates independent target observations; reports retain unique-hour counts. These are provider offset categories, not verified model-run lead times. Metrics from this split diagnose the model but are not a substitute for the competition's rolling issue-time replay.

Raw turbine data is not committed. Reproduce locally using the organizer-provided file and `windpower-fetch`, or use the already assembled dataset shared outside Git. Credit Open-Meteo and JMA for weather data; see [Open-Meteo licensing](https://open-meteo.com/en/licence).
