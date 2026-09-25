"""Train the backtest-selected candidate (Random Forest, no season columns)
on train + validation data, save it, and report its test score ONCE.

The test score is a labeled report only. It must never be used to pick or
tune anything; the promotion gate (a later task) decides whether this
candidate replaces the production model.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ml_service.app import config
from ml_service.app.preprocessing.data_loader import load_clean_data
from ml_service.app.preprocessing.pipeline import build_features, feature_columns
from ml_service.app.training.artifacts import save_artifact
from ml_service.app.training.baseline import evaluate_naive_baseline
from ml_service.app.training.split import split_chronological, split_xy
from ml_service.app.training.trainer import score_model, train_model

logger = logging.getLogger(__name__)

MODEL_NAME = "random_forest"
SEASON_FEATURES = config.SEASON_FEATURES


def run(features: Optional[pd.DataFrame] = None, models_dir: Optional[Path] = None) -> dict:
    if features is None:
        features = build_features(load_clean_data())

    train, val, test = split_chronological(features)
    train_val = pd.concat([train, val]).sort_index()  # everything up to 31 Aug 2022

    candidate_features = [c for c in feature_columns() if c not in SEASON_FEATURES]
    X_tv, y_tv = split_xy(train_val)
    X_tv = X_tv[candidate_features]
    X_te, y_te = split_xy(test)
    X_te = X_te[candidate_features]

    model = train_model(MODEL_NAME, X_tv, y_tv)

    lag_col = f"lag_{config.BASELINE_LAG_HOURS}"
    same_rows = X_te[lag_col].notna()
    test_scores = {
        "baseline": evaluate_naive_baseline(X_te, y_te, config.BASELINE_LAG_HOURS),
        "model_same_rows": score_model(model, X_te, y_te, same_rows),
        "model_all_rows": score_model(model, X_te, y_te),
    }
    beats_baseline_on_test = test_scores["model_same_rows"]["mae"] < test_scores["baseline"]["mae"]

    metadata = {
        "status": "candidate",
        "selected_by": "rolling_backtest",
        "target": config.TARGET_COLUMN,
        "feature_columns": candidate_features,
        "season_features_excluded": SEASON_FEATURES,
        "forecast_horizon_hours": config.FORECAST_HORIZON_HOURS,
        "random_seed": config.RANDOM_SEED,
        "model_params": model.get_params(),
        "trained_on": "train + validation (everything up to end of validation)",
        "train_val_rows": len(train_val),
        "train_val_start": train_val.index.min(),
        "train_val_end": train_val.index.max(),
        "test_start": test.index.min(),
        "test_end": test.index.max(),
        "test_scores": test_scores,
        "test_score_note": (
            "This test score is a REPORT ONLY. The test period was already viewed once "
            "for an earlier candidate, so this is not a clean unseen evaluation. It was "
            "not used to pick this candidate or tune any setting."
        ),
        "beats_baseline_on_test": bool(beats_baseline_on_test),
    }
    folder = save_artifact(model, MODEL_NAME, metadata, base_dir=models_dir)
    return {"artifact_folder": folder, "test_scores": test_scores,
            "beats_baseline_on_test": beats_baseline_on_test}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out = run()

    def line(label, r):
        return f"{label:<28} MAE={r['mae']:6.2f}  RMSE={r['rmse']:6.2f}  MAPE={r['mape']:5.1f}%  R2={r['r2']:6.3f}"

    t = out["test_scores"]
    print("\nSAVED TO:", out["artifact_folder"])
    print(line("TEST baseline lag_24", t["baseline"]) + f"   rows={t['baseline']['rows_scored']}")
    print(line("TEST candidate (same rows)", t["model_same_rows"]))
    print(line("TEST candidate (all rows)", t["model_all_rows"]))
    print("\nBeats baseline on test:", out["beats_baseline_on_test"])


if __name__ == "__main__":
    main()
