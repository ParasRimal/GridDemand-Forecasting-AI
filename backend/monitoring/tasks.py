"""Celery task wrapping the Phase 4 retraining decision (drift + performance).

Currently runs against the project's static train/test split (the same
comparison validated in Task 61), since there is not yet a live stream of
new production data distinct from that dataset. Once real forecasts
accumulate actual_load_mw values (see forecasts.models.Forecast), this
should be updated to compare against that instead -- documented here so
the limitation isn't silently forgotten.
"""
import logging
import sys
from pathlib import Path

from celery import shared_task
from django.utils import timezone

logger = logging.getLogger(__name__)

# The ml_service package lives outside backend/, at the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@shared_task
def run_drift_and_performance_check() -> dict:
    """Run decide_retraining() and persist the result as a DataMonitoring row."""
    from ml_service.app import config as ml_config
    from ml_service.app.monitoring.retraining import decide_retraining
    from ml_service.app.preprocessing.data_loader import load_clean_data
    from ml_service.app.preprocessing.pipeline import build_features, feature_columns
    from ml_service.app.training.split import split_chronological, split_xy

    from .models import DataMonitoring

    features = build_features(load_clean_data())
    train, val, test = split_chronological(features)
    X_test, y_test = split_xy(test)

    lag_col = f"lag_{ml_config.BASELINE_LAG_HOURS}"
    same_rows = X_test[lag_col].notna()
    y_true = y_test[same_rows]
    y_pred = X_test.loc[same_rows, lag_col]

    decision = decide_retraining(
        reference_features=train,
        current_features=test,
        y_true=y_true,
        y_pred=y_pred,
        reference_mae=9.01,
        feature_columns=feature_columns(),
    )

    record = DataMonitoring.objects.create(
        checked_at=timezone.now(),
        drift_detected=decision["drift"]["dataset_drifted"],
        drifted_column_share=decision["drift"]["drifted_column_share"],
        drifted_column_count=decision["drift"]["drifted_column_count"],
        n_columns=decision["drift"]["n_columns"],
        performance_degraded=decision["performance"]["degraded"],
        recent_mae=decision["performance"]["recent_mae"],
        reference_mae=decision["performance"]["reference_mae"],
        relative_increase=decision["performance"]["relative_increase"],
        retrain_recommended=decision["retrain"],
        reason=decision["reason"],
    )
    logger.info("DataMonitoring record #%s created: %s", record.id, decision["reason"])
    return {"record_id": record.id, "retrain": decision["retrain"], "reason": decision["reason"]}
