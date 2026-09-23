"""Versioned direct weather-to-power HTTP contract shared with the orchestrator."""
import datetime as dt
from typing import Literal
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

INPUT_SCHEMA = "windpower.input.v1"
OUTPUT_SCHEMA = "windpower.output.v1"


class InputRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    target_time: AwareDatetime
    wind_speed_10m_ms: float = Field(ge=0, le=100, strict=True)
    wind_direction_10m_deg: float = Field(ge=0, le=360, strict=True)
    temperature_2m_c: float = Field(ge=-100, le=70, strict=True)

    @field_validator("target_time", mode="before")
    @classmethod
    def iso_string(cls, value):
        if not isinstance(value, str):
            raise ValueError("target_time must be an ISO 8601 string with timezone")
        return value

    @field_validator("target_time")
    @classmethod
    def utc_hour(cls, value):
        value = value.astimezone(dt.timezone.utc)
        if value.minute or value.second or value.microsecond:
            raise ValueError("target_time must be a whole UTC hour")
        return value


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["windpower.input.v1"]
    turbine_id: str = Field(min_length=1, max_length=64, strict=True)
    weather_model: str = Field(min_length=1, max_length=160, strict=True)
    records: list[InputRecord] = Field(min_length=48, max_length=48)

    @model_validator(mode="after")
    def contiguous_hours(self):
        if any(b.target_time - a.target_time != dt.timedelta(hours=1) for a, b in zip(self.records, self.records[1:])):
            raise ValueError("records must contain 48 ordered consecutive UTC hours without duplicates")
        return self


class OutputRecord(BaseModel):
    target_time: AwareDatetime
    predicted_normalized_power: float = Field(ge=0, le=1, allow_inf_nan=False)


class PredictResponse(BaseModel):
    schema_version: Literal["windpower.output.v1"] = OUTPUT_SCHEMA
    turbine_id: str
    weather_model: str
    model_version: str
    alignment_confirmed: bool
    records: list[OutputRecord] = Field(min_length=48, max_length=48)


class InferenceError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message
        super().__init__(message)


class ErrorDetail(BaseModel):
    path: str
    message: str


class ErrorInfo(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] | None = None


class ErrorResponse(BaseModel):
    error: ErrorInfo
