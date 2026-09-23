"""On-demand power forecast service; the weather provider is supplied by the team."""
import asyncio
from contextlib import asynccontextmanager
import datetime as dt
import os
import math
import secrets
from pathlib import Path
import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, AwareDatetime, ValidationError, model_validator
from .model import Predictor
from .archive import SITES
from .contracts import INPUT_SCHEMA, OUTPUT_SCHEMA, PredictRequest, PredictResponse, OutputRecord, InferenceError, ErrorResponse

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

def create_app(model_path=None, provider_url=None, allow_provisional=None, allow_historical=None, transport=None, service_token=None):
    path = Path(model_path or os.getenv("MODEL_PATH", "models/production/model.json"))
    url = provider_url or os.getenv("WEATHER_PROVIDER_URL", "")
    provisional = allow_provisional if allow_provisional is not None else os.getenv("ALLOW_PROVISIONAL_MODEL", "0") == "1"
    historical = allow_historical if allow_historical is not None else os.getenv("ALLOW_HISTORICAL_FORECASTS", "0") == "1"
    token = service_token if service_token is not None else os.getenv("ML_SERVICE_TOKEN", "")

    @asynccontextmanager
    async def lifespan(app):
        # A missing or invalid artifact prevents startup. No fabricated predictions.
        app.state.predictor = Predictor(path, allow_provisional=provisional)
        if url and not url.startswith(("http://", "https://")):
            raise ValueError("WEATHER_PROVIDER_URL must be the team's HTTP(S) forecast endpoint")
        if url:
            async with httpx.AsyncClient(timeout=float(os.getenv("WEATHER_TIMEOUT_SECONDS", "15")), transport=transport) as client:
                app.state.client = client
                yield
        else:
            yield

    app = FastAPI(title="Wind Power Forecast", version="0.2.0", lifespan=lifespan)

    def require_auth(authorization: str | None = Header(default=None)):
        if token and not secrets.compare_digest((authorization or "").encode(), ("Bearer " + token).encode()):
            raise InferenceError(401, "unauthorized", "A valid Bearer token is required")

    @app.exception_handler(InferenceError)
    async def inference_error(request: Request, error: InferenceError):
        return JSONResponse(status_code=error.status, content={"error": {"code": error.code, "message": error.message}}, headers={"WWW-Authenticate": "Bearer"} if error.status == 401 else None)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        if request.url.path != "/v1/predict":
            return await request_validation_exception_handler(request, error)
        return JSONResponse(status_code=422, content={"error": {
            "code": "invalid_request", "message": "Request does not match " + INPUT_SCHEMA,
            "details": [{"path": ".".join(map(str, item["loc"])), "message": item["msg"]} for item in error.errors()],
        }})

    @app.get("/v1/metadata", dependencies=[Depends(require_auth)])
    def metadata():
        predictor = app.state.predictor
        return {
            "input_schema_version": INPUT_SCHEMA, "output_schema_version": OUTPUT_SCHEMA,
            "model_version": predictor.bundle["model_version"],
            "supported_turbines": predictor.turbines,
            "supported_weather_models": [predictor.bundle["weather_model"]],
            "required_records": 48, "time_step_hours": 1, "timezone": "UTC",
            "fields": {"wind_speed_10m_ms": {"unit": "m/s", "height_m": 10}, "wind_direction_10m_deg": {"unit": "degrees clockwise from north, direction FROM", "height_m": 10}, "temperature_2m_c": {"unit": "Celsius", "height_m": 2}},
            "turbine_locations": {name: {"latitude": lat, "longitude": lon} for name, lat, lon in SITES if name in predictor.turbines},
            "alignment_confirmed": predictor.bundle["dataset_metadata"]["alignment_confirmed"],
            "archive_semantics": predictor.bundle["dataset_metadata"].get("archive_semantics", "unknown"),
        }

    @app.post("/v1/predict", response_model=PredictResponse, dependencies=[Depends(require_auth)],
              responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
    def predict(request: PredictRequest):
        predictor = app.state.predictor
        if request.turbine_id not in predictor.turbines:
            raise InferenceError(422, "unsupported_turbine", "Loaded artifact has no trained model for this turbine; check /v1/metadata")
        if request.weather_model != predictor.bundle["weather_model"]:
            raise InferenceError(422, "weather_model_mismatch", "Weather source does not match the loaded artifact; check /v1/metadata")
        frame = pd.DataFrame([{**row.model_dump(), "valid_time_utc": row.target_time, "turbine_id": request.turbine_id} for row in request.records])
        # Pure inference: no weather lookup, wall-clock filtering, training or persistence.
        values = predictor.predict(frame, request.weather_model)
        if len(values) != 48 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise InferenceError(500, "invalid_model_output", "Model produced an invalid prediction")
        return PredictResponse(turbine_id=request.turbine_id, weather_model=request.weather_model,
            model_version=predictor.bundle["model_version"], alignment_confirmed=predictor.bundle["dataset_metadata"]["alignment_confirmed"],
            records=[OutputRecord(target_time=row.target_time, predicted_normalized_power=float(value)) for row, value in zip(request.records, values)])

    @app.get("/health")
    def health():
        predictor = app.state.predictor
        return {"status": "ready", "model_version": predictor.bundle["model_version"], "supported_turbines": predictor.turbines}

    @app.post("/power/forecast", response_model=ForecastResponse, dependencies=[Depends(require_auth)])
    async def forecast(request: ForecastRequest):
        predictor = app.state.predictor
        if request.turbine_id not in predictor.turbines:
            raise HTTPException(422, "No trained model for this turbine")
        if not url:
            raise HTTPException(503, "Weather provider not configured; use /v1/predict with prepared weather")
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
