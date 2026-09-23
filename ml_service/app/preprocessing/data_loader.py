"""Load, validate and clean the Ahmedabad weather + load dataset.

Cleaning rules (all defined in config.py):
  * irradiance == -999  -> NaN  (sensor "no measurement" code)
  * load <= 0           -> NaN  (meter dropout)
Bad values become NaN. We never interpolate the target.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ml_service.app import config

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = config.DATE_PARTS + config.WEATHER_COLUMNS + [config.TARGET_COLUMN]


def load_raw(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and check that every required column is present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info("Loaded %d rows from %s", len(df), path.name)
    return df


def build_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    """Replace YEAR/Month/Day/Hour with a single DatetimeIndex called 'Timestamp'."""
    timestamps = pd.to_datetime(
        dict(year=df["YEAR"], month=df["Month"], day=df["Day"], hour=df["Hour"])
    )
    out = df[config.WEATHER_COLUMNS + [config.TARGET_COLUMN]].copy()
    out.index = pd.DatetimeIndex(timestamps, name="Timestamp")
    return out


def check_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Make the index unique, sorted and strictly hourly."""
    dupes = int(df.index.duplicated().sum())
    if dupes > 0:
        raise ValueError(f"Found {dupes} duplicate timestamps; refusing to continue.")

    if not df.index.is_monotonic_increasing:
        logger.warning("Timestamps were not sorted; sorting chronologically.")
        df = df.sort_index()

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h", name="Timestamp")
    n_missing = len(full_index) - len(df)
    if n_missing > 0:
        logger.warning("%d hourly timestamps were missing; adding them as NaN rows.", n_missing)
        df = df.reindex(full_index)
    else:
        logger.info("Timeline check passed: %d rows, every hour present.", len(df))
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Turn known-bad values into NaN (no filling, no deleting rows)."""
    df = df.copy()

    bad_irradiance = df["irradiance"] == config.IRRADIANCE_MISSING_CODE
    df.loc[bad_irradiance, "irradiance"] = float("nan")

    bad_load = df[config.TARGET_COLUMN] <= config.MIN_VALID_LOAD_MW
    df.loc[bad_load, config.TARGET_COLUMN] = float("nan")

    logger.info(
        "Cleaning: %d irradiance values and %d load values set to NaN.",
        int(bad_irradiance.sum()),
        int(bad_load.sum()),
    )
    return df


def load_clean_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Full loading pipeline: read -> timestamp -> timeline check -> clean."""
    df = load_raw(path)
    df = build_timestamp_index(df)
    df = check_timeline(df)
    return clean(df)
