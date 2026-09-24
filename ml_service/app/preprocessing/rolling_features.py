"""Rolling features for the day-ahead forecast.

For a row at target hour t, the forecast is made at t - 24 h, and the newest
load known at that moment is the load at t - 24. So every rolling window must
END at t - 24, never at t:

    rolling_mean_24 at t  = mean of the load from t-47h to t-24h    (24 hours)
    rolling_mean_168 at t = mean of the load from t-191h to t-24h   (168 hours)

Method: compute an ordinary trailing rolling statistic that ends at each hour s,
then move the result forward in TIME by 24 h so it lands on t = s + 24.
Missing loads are ignored inside a window. A window needs at least
ROLLING_MIN_FRACTION of its hours to have data, otherwise the result is NaN.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Sequence

import pandas as pd

from ml_service.app import config

logger = logging.getLogger(__name__)


def rolling_column_names(window: int) -> tuple[str, str]:
    """Names of the mean and std columns, e.g. 24 -> ('rolling_mean_24', 'rolling_std_24')."""
    return f"rolling_mean_{window}", f"rolling_std_{window}"


def add_rolling_features(
    df: pd.DataFrame,
    windows: Optional[Sequence[int]] = None,
    target: str = config.TARGET_COLUMN,
) -> pd.DataFrame:
    """Return a copy of df with rolling mean/std columns that end at t - horizon."""
    if windows is None:
        windows = config.ROLLING_WINDOWS_HOURS

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")
    if not df.index.is_unique:
        raise ValueError("df index has duplicate timestamps.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be sorted oldest to newest.")
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found.")
    too_small = [w for w in windows if w < 2]
    if too_small:
        raise ValueError(f"Windows {too_small} are too small; a std needs at least 2 hours.")

    out = df.copy()
    series = df[target]
    horizon = pd.Timedelta(hours=config.FORECAST_HORIZON_HOURS)

    for w in windows:
        min_periods = max(2, math.ceil(w * config.ROLLING_MIN_FRACTION))
        # Trailing window over the last w hours, including hour s itself.
        roll = series.rolling(f"{w}h", min_periods=min_periods, closed="right")
        mean_col, std_col = rolling_column_names(w)
        # shift(freq=...) moves each result forward in TIME by the horizon,
        # so the window that ended at s = t - 24 lands on row t.
        out[mean_col] = roll.mean().shift(freq=horizon).reindex(out.index)
        out[std_col] = roll.std().shift(freq=horizon).reindex(out.index)

    logger.info("Added rolling features for windows %s", list(windows))
    return out
