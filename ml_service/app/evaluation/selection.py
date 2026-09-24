"""Automatic model selection on validation scores.

Rule (documented in config.py):
  1. A model is eligible only if its MAE is strictly below the baseline MAE.
  2. Among eligible models, the lowest MAE wins.
  3. Eligible models within `tie_tolerance_mw` of the best MAE are tied,
     and the lowest RMSE among them wins.
  4. If no model is eligible, there is no winner (winner is None).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import pandas as pd

from ml_service.app import config

_EPSILON = 1e-12  # guards the tolerance comparison against floating-point noise


@dataclass(frozen=True)
class SelectionResult:
    winner: Optional[str]
    eligible: list[str]
    reason: str
    table: pd.DataFrame  # one row per model, sorted by MAE


def select_best_model(
    scores: Mapping[str, Mapping[str, float]],
    baseline_mae: float,
    tie_tolerance_mw: float = config.SELECTION_TIE_TOLERANCE_MW,
) -> SelectionResult:
    """Apply the selection rule to {model_name: {'mae': ..., 'rmse': ..., ...}}."""
    if not scores:
        raise ValueError("No model scores were given.")
    for name, s in scores.items():
        if "mae" not in s or "rmse" not in s:
            raise ValueError(f"Scores for '{name}' must include 'mae' and 'rmse'.")

    table = pd.DataFrame({name: dict(s) for name, s in scores.items()}).T
    table["eligible"] = table["mae"] < baseline_mae
    table = table.sort_values("mae")

    eligible = [n for n in table.index if table.loc[n, "eligible"]]
    if not eligible:
        return SelectionResult(
            winner=None,
            eligible=[],
            reason=f"No model beat the baseline MAE of {baseline_mae:.2f} MW.",
            table=table,
        )

    best_mae = min(scores[n]["mae"] for n in eligible)
    contenders = [n for n in eligible if scores[n]["mae"] <= best_mae + tie_tolerance_mw + _EPSILON]
    winner = min(contenders, key=lambda n: (scores[n]["rmse"], scores[n]["mae"], n))

    if len(contenders) == 1:
        reason = f"{winner} has the lowest validation MAE ({scores[winner]['mae']:.2f} MW)."
    else:
        reason = (
            f"{', '.join(sorted(contenders))} are within {tie_tolerance_mw} MW of the best MAE "
            f"({best_mae:.2f} MW); {winner} has the lowest RMSE ({scores[winner]['rmse']:.2f} MW)."
        )
    return SelectionResult(winner=winner, eligible=eligible, reason=reason, table=table)
