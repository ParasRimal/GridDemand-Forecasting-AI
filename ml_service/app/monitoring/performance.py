"""Performance-degradation check: is recent forecast error meaningfully worse
than a documented reference error?

This deliberately does NOT compare against "however well the model did
during training" (which would always look worse over time due to natural
noise) but against an explicit, documented reference MAE the caller
supplies -- e.g. the production model's known test-set MAE.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ml_service.app import config
from ml_service.app.evaluation.metrics import evaluate


def check_performance(
    y_true: pd.Series,
    y_pred: pd.Series,
    reference_mae: float,
    degradation_threshold: float = config.PERFORMANCE_DEGRADATION_THRESHOLD,
) -> dict:
    """Compare recent (y_true, y_pred) error against a documented reference MAE.

    Returns:
        {
          "degraded": bool,
          "recent_mae": float,
          "reference_mae": float,
          "relative_increase": float,   # (recent - reference) / reference
          "threshold_used": float,
          "metrics": {mae, rmse, mape, r2},
        }
    """
    if reference_mae <= 0:
        raise ValueError("reference_mae must be positive.")
    if len(y_true) == 0:
        raise ValueError("y_true/y_pred must be non-empty.")

    metrics = evaluate(y_true, y_pred)
    recent_mae = metrics["mae"]
    relative_increase = (recent_mae - reference_mae) / reference_mae

    return {
        "degraded": bool(relative_increase > degradation_threshold),
        "recent_mae": recent_mae,
        "reference_mae": reference_mae,
        "relative_increase": relative_increase,
        "threshold_used": degradation_threshold,
        "metrics": metrics,
    }
