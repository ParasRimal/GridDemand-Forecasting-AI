"""Calendar features for the target hour.

All of these are known in advance, so they cannot leak future information.
"""
from __future__ import annotations

import holidays
import pandas as pd

from ml_service.app import config

CALENDAR_FEATURES = [
    "hour", "day_of_week", "is_weekend", "month",
    "quarter", "week_of_year", "is_holiday",
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with calendar feature columns added."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")

    out = df.copy()
    idx = out.index

    out["hour"] = idx.hour
    out["day_of_week"] = idx.dayofweek          # 0 = Monday
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    out["week_of_year"] = idx.isocalendar().week.astype(int).to_numpy()

    years = range(idx.min().year, idx.max().year + 1)
    holiday_dates = holidays.country_holidays(
        config.HOLIDAY_COUNTRY, subdiv=config.HOLIDAY_SUBDIVISION, years=years
    )
    holiday_set = set(holiday_dates.keys())
    out["is_holiday"] = [int(d in holiday_set) for d in idx.date]

    return out
