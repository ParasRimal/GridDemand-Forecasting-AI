"""Full training run: features -> split -> compare -> select -> test score -> save.

Selection uses VALIDATION scores only. The test set is scored once, for the
winner, purely to report an honest final number. The saved model is trained on
the training set only and is marked as a 'candidate' (promotion comes later).
If no model beats the baseline, nothing is saved.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ml_service.app import config
from ml_service.app.preprocessing.data_loader import load_clean_data
from ml_service.app.preprocessing.pipeline import build_features
from ml_service.app.training.artifacts import save_artifact
from ml_service.app.training.baseline import evaluate_naive_baseline
from ml_service.app.training.compare import compare_models
from ml_service.app.training.split import split_chronological, split_xy
from ml_service.app.training.trainer import score_model

logger = logging.getLogger(__name__)


def run_training(
    features: Optional[pd.DataFrame] = None,
    models_dir: Optional[Path] = None,
) -> dict:
    """Run the full training pipeline and return a summary dict."""
    if features is None:
        features = build_features(load_clean_data())

    train, val, test = split_chronological(features)
    X_tr, y_tr = split_xy(train)
    X_val, y_val = split_xy(val)
    X_te, y_te = split_xy(test)

    result = compare_models(X_tr, y_tr, X_val, y_val)
    selection = result["selection"]
    if selection.winner is None:
        logger.warning("No model saved. %s", selection.reason)
        return {"winner": None, "artifact_folder": None, "reason": selection.reason}

    winner = selection.winner
    model = result["models"][winner]

    # The one and only look at the test set: report only, never used to choose.
    lag_col = f"lag_{config.BASELINE_LAG_HOURS}"
    same_rows = X_te[lag_col].notna()
    test_scores = {
        "baseline": evaluate_naive_baseline(X_te, y_te, config.BASELINE_LAG_HOURS),
        "model_same_rows": score_model(model, X_te, y_te, same_rows),
        "model_all_rows": score_model(model, X_te, y_te),
    }

    metadata = {
        "status": "candidate",
        "target": config.TARGET_COLUMN,
        "feature_columns": list(X_tr.columns),
        "forecast_horizon_hours": config.FORECAST_HORIZON_HOURS,
        "lag_hours": config.LAG_HOURS,
        "rolling_windows_hours": config.ROLLING_WINDOWS_HOURS,
        "weather_assumption": config.WEATHER_ASSUMPTION,
        "random_seed": config.RANDOM_SEED,
        "model_params": model.get_params(),
        "trained_on": "train split only",
        "split": {
            "train_start": train.index.min(),
            "train_end": train.index.max(),
            "validation_start": val.index.min(),
            "validation_end": val.index.max(),
            "test_start": test.index.min(),
            "test_end": test.index.max(),
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
        },
        "selection_reason": selection.reason,
        "validation_scores": result["scores"][winner],
        "validation_all_models": result["scores"],
        "validation_baseline": result["baseline"],
        "test_scores": test_scores,
    }
    folder = save_artifact(model, winner, metadata, base_dir=models_dir)
    return {
        "winner": winner,
        "artifact_folder": folder,
        "reason": selection.reason,
        "validation_scores": result["scores"][winner],
        "test_scores": test_scores,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out = run_training()
    print()
    if out["winner"] is None:
        print("NO MODEL SAVED:", out["reason"])
        return

    def line(label, r):
        return f"{label:<34} MAE={r['mae']:6.2f}  RMSE={r['rmse']:6.2f}  MAPE={r['mape']:5.1f}%  R2={r['r2']:6.3f}"

    t = out["test_scores"]
    print("WINNER:", out["winner"])
    print("SAVED TO:", out["artifact_folder"])
    print()
    print(line("validation (winner)", out["validation_scores"]))
    print(line("TEST baseline lag_24", t["baseline"]) + f"   rows={t['baseline']['rows_scored']}")
    print(line("TEST winner (same rows)", t["model_same_rows"]))
    print(line("TEST winner (all rows)", t["model_all_rows"]))


if __name__ == "__main__":
    main()
