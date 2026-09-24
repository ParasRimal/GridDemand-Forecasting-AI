"""Naive baselines that every real model must beat.

    lag_24 baseline:   predict tomorrow's load = the load 24 hours earlier
    lag_168 baseline:  predict the load = the load exactly one week earlier

No learning is involved. Rows where the lag value is unknown (NaN) cannot be
predicted, so they are skipped and counted in the result.
"""
from __future__ import annotations

import pandas as pd

from ml_service.app.evaluation.metrics import evaluate


def naive_lag_prediction(X: pd.DataFrame, lag_hours: int = 24) -> pd.Series:
    """Return the lag column as the prediction (NaN where the lag is unknown)."""
    column = f"lag_{lag_hours}"
    if column not in X.columns:
        raise KeyError(f"Column '{column}' not found in the features.")
    return X[column]


def evaluate_naive_baseline(X: pd.DataFrame, y: pd.Series, lag_hours: int = 24) -> dict:
    """Score the lag baseline on the rows where it can make a prediction."""
    prediction = naive_lag_prediction(X, lag_hours)
    usable = prediction.notna()
    if not usable.any():
        raise ValueError("The baseline has no usable rows (every lag value is NaN).")

    result = evaluate(y[usable], prediction[usable])
    result["rows_scored"] = int(usable.sum())
    result["rows_skipped"] = int((~usable).sum())
    return result
