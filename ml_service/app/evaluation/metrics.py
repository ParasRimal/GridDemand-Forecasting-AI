"""Forecast accuracy metrics: MAE, RMSE, MAPE and R^2.

MAE   average absolute error, in MW.                      lower is better
RMSE  square root of the average squared error, in MW.    lower is better
      Punishes large misses more than MAE does.
MAPE  average absolute error as a percentage of the actual load.  lower is better
R2    1 - (model error / error of always predicting the mean).    higher is better
      1.0 = perfect, 0.0 = no better than the mean.

Bad input (different lengths, empty, NaN/inf) raises ValueError instead of
silently producing a misleading score.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import pandas as pd

ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]

METRIC_NAMES = ["mae", "rmse", "mape", "r2"]


def _prepare(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to float arrays and validate them."""
    true = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    if true.shape != pred.shape:
        raise ValueError(f"Length mismatch: {true.shape[0]} actual vs {pred.shape[0]} predicted.")
    if true.size == 0:
        raise ValueError("Cannot compute metrics on empty input.")
    if not (np.isfinite(true).all() and np.isfinite(pred).all()):
        raise ValueError("Inputs contain NaN or infinite values.")
    return true, pred


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error (same unit as the load: MW)."""
    true, pred = _prepare(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error (MW)."""
    true, pred = _prepare(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Percentage Error, in percent (10.0 means 10%)."""
    true, pred = _prepare(y_true, y_pred)
    if np.any(true == 0):
        raise ValueError("MAPE is undefined when an actual value is exactly 0.")
    return float(np.mean(np.abs((true - pred) / true)) * 100)


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination R^2."""
    true, pred = _prepare(y_true, y_pred)
    ss_total = float(np.sum((true - true.mean()) ** 2))
    if ss_total == 0:
        raise ValueError("R2 is undefined when all actual values are identical.")
    ss_residual = float(np.sum((true - pred) ** 2))
    return 1.0 - ss_residual / ss_total


def evaluate(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Return all four metrics as {'mae', 'rmse', 'mape', 'r2'}."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
