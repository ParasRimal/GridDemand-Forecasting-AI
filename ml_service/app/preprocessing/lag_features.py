"""Lag features for the day-ahead forecast.

For a row at target hour t, lag_h is the load at exactly t - h hours.
Because every h >= FORECAST_HORIZON_HOURS (24), the value was already known
at forecast time, so the future cannot leak into the features.

Lags are looked up by TIMESTAMP, not by row position, so they stay correct
even if a row is missing. Unknown values stay NaN (no interpolation).
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import pandas as pd

from ml_service.app import config

logger = logging.getLogger(__name__)


def lag_column_name(hours: int) -> str:
    """Name of the lag column, e.g. 24 -> 'lag_24'."""
    return f"lag_{hours}"


def add_lag_features(
    df: pd.DataFrame,
    lags: Optional[Sequence[int]] = None,
    target: str = config.TARGET_COLUMN,
) -> pd.DataFrame:
    """Return a copy of df with one lag column per entry in `lags`."""
    if lags is None:
        lags = config.LAG_HOURS

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")
    if not df.index.is_unique:
        raise ValueError("df index has duplicate timestamps.")
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found.")

    too_short = [lag for lag in lags if lag < config.FORECAST_HORIZON_HOURS]
    if too_short:
        raise ValueError(
            f"Lags {too_short} are shorter than the {config.FORECAST_HORIZON_HOURS}h "
            "forecast horizon and would leak the future."
        )

    out = df.copy()
    series = df[target]
    for lag in lags:
        # shift(freq=...) moves each value forward in TIME by `lag` hours,
        # so the value from t - lag ends up at t. reindex aligns it to our rows.
        shifted = series.shift(freq=pd.Timedelta(hours=lag))
        out[lag_column_name(lag)] = shifted.reindex(out.index)

    logger.info("Added lag features: %s", [lag_column_name(lag) for lag in lags])
    return out
