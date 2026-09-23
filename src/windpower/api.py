"""On-demand power forecast service; the weather provider is supplied by the team."""
import asyncio
from contextlib import asynccontextmanager
import datetime as dt
import os
from pathlib import Path
import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, AwareDatetime, ValidationError, model_validator
from .model import Predictor

class Hour(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    time: AwareDatetime
    wind_speed_10m_ms: float = Field(ge=0, le=100)
    wind_direction_10m_deg: float = Field(ge=0, le=360)
    temperature_2m_c: float = Field(ge=-100, le=70)

class WeatherForecast(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turbine_id: str
    weather_model: str
    hourly: list[Hour] = Field(min_length=1, max_length=168)

    @model_validator(mode="after")
    def ordered_hours(self):
        times = [h.time.astimezone(dt.timezone.utc) for h in self.hourly]
        if any(t.minute or t.second or t.microsecond for t in times):
            raise ValueError("Forecasts must use whole UTC hours")
        if any(b-a != dt.timedelta(hours=1) for a, b in zip(times, times[1:])):
            raise ValueError("Forecast hours must be ordered, unique and contiguous")
        return self

class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turbine_id: str
    hours: int = Field(default=48, ge=1, le=48)

class PredictionHour(BaseModel):
    time: AwareDatetime
    normalized_power: float = Field(ge=0, le=1)

class ForecastResponse(BaseModel):
    turbine_id: str
    model_version: str
    weather_model: str
    generated_at_utc: AwareDatetime
    alignment_confirmed: bool
    hourly: list[PredictionHour]

class ProviderError(Exception):
    pass

async def fetch_weather(client, url, token, request):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for attempt in range(3):
        try:
            response = await client.get(url, params=request.model_dump(), headers=headers)
            if response.status_code in [429, 500, 502, 503, 504] and attempt < 2:
                await asyncio.sleep(0.25 * 2 ** attempt); continue
            response.raise_for_status()
            return WeatherForecast.model_validate(response.json())
        except httpx.TransportError:
            if attempt == 2: raise ProviderError("Weather provider unavailable") from None
            await asyncio.sleep(0.25 * 2 ** attempt)
        except (httpx.HTTPStatusError, ValidationError, ValueError):
            # Do not expose upstream URLs, credentials, bodies or validation inputs.
            raise ProviderError("Weather provider returned an invalid response") from None
    raise ProviderError("Weather provider unavailable")

def create_app(model_path=None, provider_url=None, allow_provisional=None, allow_historical=None, transport=None):
    path = Path(model_path or os.getenv("MODEL_PATH", "artifacts/model.json"))
    url = provider_url or os.getenv("WEATHER_PROVIDER_URL", "")
    provisional = allow_provisional if allow_provisional is not None else os.getenv("ALLOW_PROVISIONAL_MODEL", "0") == "1"
    historical = allow_historical if allow_historical is not None else os.getenv("ALLOW_HISTORICAL_FORECASTS", "0") == "1"

    @asynccontextmanager
    async def lifespan(app):
        # A missing or invalid artifact prevents startup. No fabricated predictions.
        app.state.predictor = Predictor(path, allow_provisional=provisional)
        if not url.startswith(("http://", "https://")):
            raise ValueError("WEATHER_PROVIDER_URL must be the team's HTTP(S) forecast endpoint")
        async with httpx.AsyncClient(timeout=float(os.getenv("WEATHER_TIMEOUT_SECONDS", "15")), transport=transport) as client:
            app.state.client = client
            yield

    app = FastAPI(title="Wind Power Forecast", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        predictor = app.state.predictor
        return {"status": "ready", "model_version": predictor.bundle["model_version"], "supported_turbines": predictor.turbines}

    @app.post("/power/forecast", response_model=ForecastResponse)
    async def forecast(request: ForecastRequest):
        predictor = app.state.predictor
        if request.turbine_id not in predictor.turbines:
            raise HTTPException(422, "No trained model for this turbine")
        requested_at = dt.datetime.now(dt.timezone.utc)
        expected_start = requested_at.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
        try:
            weather = await fetch_weather(app.state.client, url, os.getenv("WEATHER_PROVIDER_TOKEN", ""), request)
        except ProviderError as error:
            raise HTTPException(502, str(error)) from None
        if weather.turbine_id != request.turbine_id or len(weather.hourly) < request.hours:
            raise HTTPException(502, "Weather provider returned the wrong turbine or insufficient hours")
        if not historical and weather.hourly[0].time != expected_start:
            raise HTTPException(502, "Weather forecast must begin at the next UTC hour")
        hours = weather.hourly[:request.hours]
        frame = pd.DataFrame([{**hour.model_dump(), "valid_time_utc": hour.time, "turbine_id": request.turbine_id} for hour in hours])
        try:
            prediction = predictor.predict(frame, weather.weather_model)
        except ValueError:
            raise HTTPException(502, "Weather model or features do not match the trained predictor") from None
        return ForecastResponse(turbine_id=request.turbine_id, model_version=predictor.bundle["model_version"], weather_model=weather.weather_model, generated_at_utc=dt.datetime.now(dt.timezone.utc), alignment_confirmed=predictor.bundle["dataset_metadata"]["alignment_confirmed"], hourly=[PredictionHour(time=hour.time, normalized_power=float(value)) for hour, value in zip(hours, prediction)])

    return app

app = create_app()
