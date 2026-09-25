"""Pydantic schemas for POST /predict.

Every feature field is OPTIONAL here: the API contract does not change when
the production model changes (baseline today, possibly a trained candidate
later). What's actually REQUIRED for a given prediction is decided at
request time against ProductionModel.required_features, not hardcoded here.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    timestamp: str = Field(..., description="The target hour being forecast, e.g. '2022-09-15T14:00:00'.")

    @field_validator("timestamp")
    @classmethod
    def _timestamp_must_be_parseable(cls, value: str) -> str:
        try:
            pd.Timestamp(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"'timestamp' is not a valid date/time: {value!r}") from exc
        return value

    # Calendar
    hour: Optional[int] = None
    day_of_week: Optional[int] = None
    is_weekend: Optional[int] = None
    month: Optional[int] = None
    quarter: Optional[int] = None
    week_of_year: Optional[int] = None
    is_holiday: Optional[int] = None

    # Weather (assumed forecast for the target hour; see docs/MODELING_FINDINGS.md)
    irradiance: Optional[float] = None
    temperature: Optional[float] = None
    dewpoint: Optional[float] = None
    specific_humidity: Optional[float] = Field(default=None, alias="specific humidity")
    wind_speed: Optional[float] = Field(default=None, alias="wind speed")

    # Lags and rolling stats
    lag_24: Optional[float] = None
    lag_48: Optional[float] = None
    lag_168: Optional[float] = None
    rolling_mean_24: Optional[float] = None
    rolling_std_24: Optional[float] = None
    rolling_mean_168: Optional[float] = None
    rolling_std_168: Optional[float] = None

    def to_feature_row(self) -> pd.DataFrame:
        """One-row DataFrame indexed by the target timestamp, with real column names
        (including the two with spaces), and only the fields the caller actually set."""
        data = self.model_dump(by_alias=True, exclude={"timestamp"}, exclude_none=True)
        return pd.DataFrame([data], index=pd.DatetimeIndex([pd.Timestamp(self.timestamp)], name="Timestamp"))


class PredictResponse(BaseModel):
    timestamp: str
    predicted_load_mw: float
    model_type: str
    model_name: Optional[str] = None


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest]


class BatchResultItem(BaseModel):
    index: int
    success: bool
    result: Optional[PredictResponse] = None
    error: Optional[str] = None


class BatchPredictResponse(BaseModel):
    results: list[BatchResultItem]
