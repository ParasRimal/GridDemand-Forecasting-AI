"""Combined retraining decision: drift and/or performance degradation.

Per project requirement: retraining must be based on STATISTICALLY MEANINGFUL
evidence (drift and/or degradation), never "retrain whenever data changes."
Both checks below already have their own documented thresholds
(config.DRIFT_SHARE_THRESHOLD, config.PERFORMANCE_DEGRADATION_THRESHOLD);
this module just combines their verdicts and logs a clear reason, mirroring
the promotion gate's style (evaluation.selection / training.registry).
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from ml_service.app.monitoring.drift import check_drift
from ml_service.app.monitoring.performance import check_performance


def decide_retraining(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    reference_mae: float,
    feature_columns: Optional[Sequence[str]] = None,
) -> dict:
    """Run both checks and return a single retraining decision.

    Returns:
        {
          "retrain": bool,               # drift OR degradation
          "drift": {...},                # full check_drift() result
          "performance": {...},          # full check_performance() result
          "reason": str,
        }
    """
    drift_result = check_drift(reference_features, current_features, columns=feature_columns)
    perf_result = check_performance(y_true, y_pred, reference_mae)

    drifted = drift_result["dataset_drifted"]
    degraded = perf_result["degraded"]
    retrain = drifted or degraded

    parts = []
    if drifted:
        parts.append(
            f"data drift detected ({drift_result['drifted_column_share']:.0%} of "
            f"{drift_result['n_columns']} features drifted, threshold "
            f"{drift_result['threshold_used']:.0%})"
        )
    if degraded:
        parts.append(
            f"performance degraded (MAE {perf_result['recent_mae']:.2f} vs reference "
            f"{perf_result['reference_mae']:.2f}, +{perf_result['relative_increase']:.0%}, "
            f"threshold +{perf_result['threshold_used']:.0%})"
        )
    if retrain:
        reason = "Retraining recommended: " + "; ".join(parts) + "."
    else:
        reason = (
            f"No retraining needed: drift share {drift_result['drifted_column_share']:.0%} "
            f"(threshold {drift_result['threshold_used']:.0%}) and performance within "
            f"{perf_result['relative_increase']:.0%} of reference "
            f"(threshold +{perf_result['threshold_used']:.0%})."
        )

    return {"retrain": retrain, "drift": drift_result, "performance": perf_result, "reason": reason}
