"""Backtest feature/target variants. Uses train + validation data only.

Decision rule (fixed BEFORE running):
  eligible = beats the baseline in >= 5 of 7 folds AND mean MAE < baseline mean MAE
  winner   = lowest mean MAE; combinations within 0.1 MW of the best are tied,
             and the one that beats the baseline in the most folds wins.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from ml_service.app.preprocessing.data_loader import load_clean_data
from ml_service.app.preprocessing.pipeline import build_features, feature_columns
from ml_service.app.training.backtest import backtest
from ml_service.app.training.model_factory import available_models

pd.set_option("display.width", 200)

SEASON = ["month", "quarter", "week_of_year"]
ALL = feature_columns()
NO_SEASON = [c for c in ALL if c not in SEASON]
VARIANTS = {
    "full": (ALL, "level"),
    "no_season": (NO_SEASON, "level"),
    "full_delta": (ALL, "delta_baseline"),
    "no_season_delta": (NO_SEASON, "delta_baseline"),
}
MIN_FOLDS, TIE_MW = 5, 0.1

df = build_features(load_clean_data())
rows = []
for variant, (features, mode) in VARIANTS.items():
    for model in available_models():
        r = backtest(df, model, features=features, target_mode=mode)
        rows.append({
            "variant": variant,
            "model": model,
            "folds_beaten": int(r["beats_baseline"].sum()),
            "n_folds": len(r),
            "mean_mae": r["model_mae"].mean(),
            "mean_abs_bias": r["model_bias"].abs().mean(),
            "worst_fold_mae": r["model_mae"].max(),
            "baseline_mean_mae": r["baseline_mae"].mean(),
        })

res = pd.DataFrame(rows).sort_values("mean_mae").reset_index(drop=True)
base = res["baseline_mean_mae"].iloc[0]
res["eligible"] = (res["folds_beaten"] >= MIN_FOLDS) & (res["mean_mae"] < base)
print(f"Baseline (same hour yesterday): mean MAE = {base:.2f}\n")
print(res.drop(columns="baseline_mean_mae").round(2).to_string())

ok = res[res["eligible"]]
print()
if ok.empty:
    print("NO ELIGIBLE COMBINATION under the rule.")
else:
    best = ok["mean_mae"].min()
    tied = ok[ok["mean_mae"] <= best + TIE_MW + 1e-12]
    win = tied.sort_values(["folds_beaten", "mean_mae"], ascending=[False, True]).iloc[0]
    print(f"WINNER: {win['variant']} + {win['model']}  "
          f"(mean MAE {win['mean_mae']:.2f}, beats baseline in {win['folds_beaten']}/{win['n_folds']} folds)")
