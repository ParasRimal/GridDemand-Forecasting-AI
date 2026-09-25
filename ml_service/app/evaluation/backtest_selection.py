"""Model selection from rolling-backtest results (many time windows, not one).

Input: {candidate_name: DataFrame returned by training.backtest.backtest()}.

Rule (settings in config.py):
  1. Eligible = beats the baseline in >= BACKTEST_MIN_FOLD_FRACTION of the folds
     AND mean model MAE < mean baseline MAE.
  2. The lowest mean MAE among eligible candidates wins.
  3. Eligible candidates within the tolerance of the best mean MAE are tied;
     the one that beats the baseline in the most folds wins.
  4. If nothing is eligible, winner is None.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional

import pandas as pd

from ml_service.app import config
from ml_service.app.evaluation.selection import SelectionResult

REQUIRED_COLUMNS = ["model_mae", "baseline_mae"]
_EPSILON = 1e-12  # guards the tolerance comparison against floating-point noise


def select_from_backtests(
    results: Mapping[str, pd.DataFrame],
    min_fold_fraction: float = config.BACKTEST_MIN_FOLD_FRACTION,
    tie_tolerance_mw: float = config.BACKTEST_TIE_TOLERANCE_MW,
) -> SelectionResult:
    """Apply the selection rule to a dict of backtest result tables."""
    if not results:
        raise ValueError("No backtest results were given.")

    rows = []
    for name, bt in results.items():
        missing = [c for c in REQUIRED_COLUMNS if c not in bt.columns]
        if missing:
            raise ValueError(f"Backtest for '{name}' is missing columns: {missing}")
        if bt.empty:
            raise ValueError(f"Backtest for '{name}' has no folds.")

        n_folds = len(bt)
        # round() first: 10 * 0.7 is 7.000000000000001 in floating point, and ceil would give 8.
        folds_needed = math.ceil(round(n_folds * min_fold_fraction, 9))
        folds_beaten = int((bt["model_mae"] < bt["baseline_mae"]).sum())
        mean_mae = float(bt["model_mae"].mean())
        baseline_mean_mae = float(bt["baseline_mae"].mean())
        rows.append({
            "name": name,
            "folds": n_folds,
            "folds_beaten": folds_beaten,
            "folds_needed": folds_needed,
            "mean_mae": mean_mae,
            "baseline_mean_mae": baseline_mean_mae,
            "eligible": folds_beaten >= folds_needed and mean_mae < baseline_mean_mae,
        })

    table = pd.DataFrame(rows).set_index("name").sort_values("mean_mae")
    eligible = table[table["eligible"]]
    if eligible.empty:
        return SelectionResult(
            winner=None,
            eligible=[],
            reason="No candidate beat the baseline in enough backtest folds with a lower mean MAE.",
            table=table,
        )

    best = float(eligible["mean_mae"].min())
    contenders = eligible[eligible["mean_mae"] <= best + tie_tolerance_mw + _EPSILON]
    ranked = contenders.reset_index().sort_values(
        ["folds_beaten", "mean_mae", "name"], ascending=[False, True, True]
    )
    winner = str(ranked.iloc[0]["name"])
    w = table.loc[winner]

    if len(contenders) == 1:
        reason = (
            f"{winner} has the lowest mean backtest MAE ({w['mean_mae']:.2f} MW vs baseline "
            f"{w['baseline_mean_mae']:.2f}) and beats the baseline in {int(w['folds_beaten'])}/{int(w['folds'])} folds."
        )
    else:
        reason = (
            f"{', '.join(sorted(contenders.index))} are within {tie_tolerance_mw} MW of the best mean MAE "
            f"({best:.2f} MW); {winner} beats the baseline in the most folds "
            f"({int(w['folds_beaten'])}/{int(w['folds'])})."
        )
    return SelectionResult(winner=winner, eligible=list(eligible.index), reason=reason, table=table)
