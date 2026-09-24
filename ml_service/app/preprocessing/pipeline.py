"""Full feature pipeline: raw CSV -> model-ready table.

Steps: load + clean -> calendar -> lags -> rolling -> keep usable rows.
Row filtering happens LAST, so lags and rolling windows are computed on the
complete hourly timeline and stay exact.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ml_service.app import config
from ml_service.app.preprocessing.calendar_features import CALENDAR_FEATURES, add_calendar_features
from ml_service.app.preprocessing.data_loader import load_clean_data
from ml_service.app.preprocessing.lag_features import add_lag_features, lag_column_name
from ml_service.app.preprocessing.rolling_features import add_rolling_features, rolling_column_names

logger = logging.getLogger(__name__)


def feature_columns() -> list[str]:
    """The model input columns, in a fixed order (the target is NOT included)."""
    cols = list(CALENDAR_FEATURES) + list(config.WEATHER_COLUMNS)
    cols += [lag_column_name(lag) for lag in config.LAG_HOURS]
    for window in config.ROLLING_WINDOWS_HOURS:
        cols += list(rolling_column_names(window))
    return cols


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Turn the cleaned hourly table into the model-ready feature table."""
    df = add_calendar_features(clean)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = df[feature_columns() + [config.TARGET_COLUMN]]

    n_start = len(df)
    warmup_end = df.index.min() + pd.Timedelta(hours=max(config.LAG_HOURS))
    df = df[df.index >= warmup_end]
    n_after_warmup = len(df)
    df = df[df[config.TARGET_COLUMN].notna()]

    logger.info(
        "Rows: %d -> %d after dropping the %dh warm-up -> %d after dropping missing targets.",
        n_start, n_after_warmup, max(config.LAG_HOURS), len(df),
    )
    return df


def build_and_save(output_path: Optional[Path] = None) -> pd.DataFrame:
    """Run the whole pipeline from the raw CSV and save the result as CSV."""
    output_path = Path(output_path or config.PROCESSED_DATA_PATH)
    df = build_features(load_clean_data())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    logger.info("Saved %d rows x %d columns to %s", len(df), df.shape[1], output_path)
    return df


def load_processed(path: Optional[Path] = None) -> pd.DataFrame:
    """Read the saved processed dataset back, with its Timestamp index."""
    path = Path(path or config.PROCESSED_DATA_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}. Run build_and_save() first.")
    return pd.read_csv(path, index_col="Timestamp", parse_dates=["Timestamp"])
