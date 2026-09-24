"""Train every model, score it on validation, and select the best one.

All models and the baseline are scored on the SAME rows: the rows where the
baseline lag is known. The test set is not touched here.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import pandas as pd

from ml_service.app import config
from ml_service.app.evaluation.selection import select_best_model
from ml_service.app.training.baseline import evaluate_naive_baseline
from ml_service.app.training.model_factory import available_models
from ml_service.app.training.trainer import score_model, train_model

logger = logging.getLogger(__name__)


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_names: Optional[Sequence[str]] = None,
) -> dict:
    """Return {'baseline', 'scores', 'models', 'selection'}."""
    names = list(model_names) if model_names is not None else available_models()

    baseline = evaluate_naive_baseline(X_val, y_val, config.BASELINE_LAG_HOURS)
    same_rows = X_val[f"lag_{config.BASELINE_LAG_HOURS}"].notna()

    models, scores = {}, {}
    for name in names:
        models[name] = train_model(name, X_train, y_train)
        scores[name] = score_model(models[name], X_val, y_val, same_rows)
        logger.info("%s validation MAE=%.2f RMSE=%.2f", name, scores[name]["mae"], scores[name]["rmse"])

    selection = select_best_model(scores, baseline["mae"])
    logger.info("Selection: %s", selection.reason)
    return {"baseline": baseline, "scores": scores, "models": models, "selection": selection}
