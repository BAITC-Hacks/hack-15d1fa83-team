# Proposed shared weather contract and request flow

> Historical proposal. The unified project now implements this integration; use ML_INTEGRATION.md and the root README for current behavior. The statements below about unconfirmed adaptation refer to the original proposal date.

This records the original design proposal. The direct-input endpoint is now implemented on our branch; [ML_SERVICE_CONTRACT.md](ML_SERVICE_CONTRACT.md) is authoritative for the exact current fields and errors. The other party has not yet confirmed adaptation. The `yevgeniy` branch was inspected read-only at commit 97987bbda12f6fd165a7e8d32be1410b024d9e1a. No changes to that branch or messages to its owner were made.

## Weather choice

Keep `jma_gsm` with 10 m wind speed/direction and 2 m temperature as the initial shared weather source. Our downloaded fixed-offset archive has all three variables throughout March 2023-January 2026 at both requested turbine pins. This provides a working baseline, not evidence JMA is the most accurate weather model.

Our existing GFS download has 10 m wind only from 2024-01-19 12:00 UTC at day-1 offset; 100 m wind begins 2024-02-16 06:00 UTC. New small coverage probes at turbine 2 found ICON's three required variables available for both day-1/day-2 offsets on 2024-02-01 and 2026-01-15, but absent on 2023-03-11. ECMWF IFS025 supplied all three on 2026-01-15 but none on the other two sampled dates. These probes do not establish full coverage or relative predictive skill.

JMA GSM is coarse: approximately 0.5 degrees, native six-hour values interpolated hourly by Open-Meteo. Higher-resolution GFS or ICON could improve power forecasts despite less training history; benchmark them on the same covered training and validation timestamps with 10 m features before switching. Preserve January as held-out reporting data and exclude February targets from training. Do not train with one provider and silently substitute another at inference, or use automatic best-match blends.

Sources: [JMA documentation](https://open-meteo.com/en/docs/jma-api), [archive availability](https://open-meteo.com/en/docs/previous-runs-api), [GFS](https://open-meteo.com/en/docs/gfs-api), [ICON](https://open-meteo.com/en/docs/dwd-api).

## Turbine locations and training

| Turbine | Latitude | Longitude | Complete measurement hours | Training examples |
|---|---:|---:|---:|---:|
| turbine_1 | 43.645150 | 78.535604 | 23,666 | 47,332 |
| turbine_2 | 43.643198 | 78.538828 | 24,784 | 49,568 |

The turbines are approximately 340 m apart. Weather requests use their respective coordinates and training joins on turbine ID plus UTC target time. The current JMA inputs are identical at all 46,886 overlapping forecast examples. Actual normalized power differs by an average absolute 0.02605 on those overlaps. Identical coarse-grid inputs must not be presented as resolving local turbine-scale weather.

Train one shared network with turbine-ID features and both turbines' measured targets. Report MAE, RMSE and bias separately for each turbine and archive offset, against turbine-specific constant and wind-curve baselines. Calendar-based splits put simultaneous measurements from both turbines in the same partition. If shared training hurts one turbine, compare separate models on validation data. Coordinates identify weather acquisition; they are not claimed to resolve spatial detail absent from the upstream forecast.

The new turbine-2 file is byte-identical to the original. Both CSVs stop at 2026-01-31 23:50 local time, despite their filenames mentioning February. Measurement interval-start semantics remain provisional; Asia/Almaty handles the historical UTC-offset change. Six ambiguous readings per turbine are excluded rather than guessed. No new production weights exist for both turbines until the cloud retraining finishes.

## Recommended ownership and request flow

1. Frontend requests a forecast from Django. In live mode Django sets the decision time to now and selects the next UTC hour; operators need not enter issue times. Replay mode supplies a simulated clock internally.
2. Django asks its weather module for the requested turbine's next 48 hourly rows, explicitly selecting `jma_gsm` and the same units, interpolation and grid-selection settings used for training.
3. The weather module saves an immutable snapshot with source, model, requested/resolved coordinates, units and available timing provenance. Actual issue time must remain unknown when a product does not expose it; estimated fixed-offset timing must remain labelled as estimated. A strict historical replay needs source-verified archived runs.
4. Django posts the prepared weather to our proposed `POST /v1/predict`. Our service loads trained weights once and performs inference; it does not fetch weather or train during the request.
5. Django checks the returned timestamps and values, saves the result and model version, and updates the UI. For both turbines it makes two independent requests (they can run concurrently), preserving separate predictions and failures.
6. Cache results by input-snapshot hash, turbine ID and deployed model version. New weather or a new model version triggers recalculation. Missing required weather or a mismatched provider fails visibly; no silent source replacement.

The current `/power/forecast` weather-fetching endpoint remains a standalone convenience interface. The implemented direct-input adapter is the intended integration path; Django calling the convenience endpoint would duplicate weather acquisition and weaken snapshot reproducibility. See ML_SERVICE_CONTRACT.md for deployment and complete examples.

## Proposed direct-input body

This example shows one record for readability; a real request must contain exactly 48 consecutive UTC hours.

```json
{
  "schema_version": "windpower.input.v1",
  "turbine_id": "turbine_1",
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

Keep the other branch's response names: `model_version` and `records`, each with `target_time` and `predicted_normalized_power` in [0,1]. The input turbine ID selects learned turbine behavior. Unsupported turbines and mismatched weather models return explicit errors. UTC timestamps must be preserved exactly by instant and order.

Pressure and forecast age may remain in Django's weather snapshots but are not required inputs for this baseline. Forecast age is not forecast lead time, and neither should be fabricated. A future model can add verified lead-time/provenance features under a new contract version. A proposed metadata endpoint can advertise supported turbines, weather model and input schema to prevent configuration drift.

Changes requested from the other developer: use JMA GSM for the current artifact; request 10 m wind and 2 m temperature in m/s and Celsius; update the ML payload mapper/schema to the explicit names above; keep forecast provenance in their weather layer; select HTTP ML mode; invalidate predictions when the deployed model changes. Our direct-input adapter is implemented; deploy verified joint weights when training completes to enable both turbines. No raw-data upload or GPU is required for inference.

Do not sum two normalized outputs to claim plant MW or station capacity factor without knowing each turbine's rated power.
