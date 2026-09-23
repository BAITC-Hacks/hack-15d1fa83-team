# Weather module integration

The power endpoint accepts only a turbine identifier and the number of hours (1–48). No caller-supplied issue time is required. Internally it sends:

```http
GET /weather/forecast?turbine_id=turbine_2&hours=48
```

Configure the full URL in `WEATHER_PROVIDER_URL`. Optional bearer authentication comes from `WEATHER_PROVIDER_TOKEN`. Credentials are not accepted in prediction request bodies.

The response must have this structure, with `hourly` containing the requested number of records. The values below illustrate the schema, not a stored default response:

```json
{
  "turbine_id": "turbine_2",
  "weather_model": "jma_gsm",
  "hourly": [
    {
      "time": "2026-02-01T00:00:00Z",
      "wind_speed_10m_ms": 4.8,
      "wind_direction_10m_deg": 64,
      "temperature_2m_c": 2.0
    }
  ]
}
```

Rules:

- Forecast times must contain timezone offsets, lie on whole UTC hours, and be ordered, unique and hourly-contiguous.
- In live mode the first record is for the next UTC hour at request receipt. The response can contain additional later hours; the service uses the requested count.
- The three weather fields must be finite numbers in the named units. Unknown fields are rejected so contract mismatches are visible.
- Meteorological wind direction is degrees from north. Wind speed is at 10 m above ground.
- The initial model is trained on JMA GSM. A different upstream model must not masquerade as JMA; train/validate a corresponding artifact before changing providers.
- Simulated/historical weather uses the same schema and `ALLOW_HISTORICAL_FORECASTS=1` on the power service. No mocked power values are involved.
- Predictions are normalized power, not MW or MWh. Confirm the normalization and rated capacity before adding physical-unit conversion.

The power response includes turbine/model identity, the server's automatically generated timestamp, an alignment-confirmation flag, and hourly `{time, normalized_power}` records. The generated timestamp is operational metadata, not a manually entered input or model feature.

An artifact trained only on turbine 2 cannot serve turbine 1. Add its real target CSV and retrain before enabling it.
