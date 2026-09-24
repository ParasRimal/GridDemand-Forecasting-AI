"""Rolling-origin (expanding window) backtest, using train + validation data ONLY.

For each calendar month M from `first_fold` on:
    train on all rows before (start of M - gap), then test on month M.
The gap is the forecast horizon (24 h): labels from the last 24 hours before M
would not be known yet when the forecast for M's first hour is made.

The test period (after config.VALIDATION_END) is never used.
Each fold scores the model and the lag baseline on the SAME rows.

target_mode:
    "level"           the model predicts the load directly.
    "delta_baseline"  the model predicts (load - baseline lag) and the lag is
                      added back. Training rows with an unknown lag are skipped.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import pandas as pd

from ml_service.app import config
from ml_service.app.evaluation.metrics import mae
from ml_service.app.preprocessing.pipeline import feature_columns
from ml_service.app.training.model_factory import build_model

logger = logging.getLogger(__name__)

TARGET_MODES = ("level", "delta_baseline")


def backtest(
    df: pd.DataFrame,
    model_name: str,
    first_fold: str = "2022-02",
    features: Optional[Sequence[str]] = None,
    gap_hours: Optional[int] = None,
    target_mode: str = "level",
) -> pd.DataFrame:
    """Return one row per monthly fold with model and baseline scores."""
    if target_mode not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {TARGET_MODES}, got '{target_mode}'.")
    features = list(features) if features is not None else feature_columns()
    gap = pd.Timedelta(hours=config.FORECAST_HORIZON_HOURS if gap_hours is None else gap_hours)
    baseline_col = f"lag_{config.BASELINE_LAG_HOURS}"
    target = config.TARGET_COLUMN
    delta = target_mode == "delta_baseline"

    data = df[df.index <= pd.Timestamp(config.VALIDATION_END)]  # never touch the test period
    if data.empty:
        raise ValueError("No data available before the end of validation.")

    rows = []
    for month in pd.period_range(first_fold, data.index.max().to_period("M"), freq="M"):
        start, end = month.start_time, (month + 1).start_time
        train_part = data[data.index < start - gap]
        if delta:
            train_part = train_part[train_part[baseline_col].notna()]
        test_part = data[(data.index >= start) & (data.index < end)]
        test_part = test_part[test_part[baseline_col].notna()]  # rows the baseline can predict
        if train_part.empty or test_part.empty:
            continue

        y_train = train_part[target] - train_part[baseline_col] if delta else train_part[target]
        model = build_model(model_name)
        model.fit(train_part[features], y_train)
        raw = model.predict(test_part[features])
        if delta:
            raw = raw + test_part[baseline_col].to_numpy()
        pred = pd.Series(raw, index=test_part.index)
        y = test_part[target]

        model_mae = mae(y, pred)
        baseline_mae = mae(y, test_part[baseline_col])
        rows.append({
            "month": str(month),
            "train_rows": len(train_part),
            "test_rows": len(test_part),
            "actual_mean": float(y.mean()),
            "model_bias": float((pred - y).mean()),
            "model_mae": model_mae,
            "baseline_mae": baseline_mae,
            "beats_baseline": model_mae < baseline_mae,
        })

    if not rows:
        raise ValueError("No usable folds; check first_fold against the data range.")
    return pd.DataFrame(rows).set_index("month")
