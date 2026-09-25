"""Production model registry: what is currently in production, and why.

registry.json (tracked in Git, unlike model files) records:
  - production: the model/rule currently serving predictions
  - history: every promotion decision ever made, promoted or not

Default production, before anything is ever promoted, is the naive
"same hour yesterday" baseline. A challenger only replaces production
if it beats it by more than PROMOTION_MIN_MAE_IMPROVEMENT_MW on a fair,
shared evaluation.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ml_service.app import config

logger = logging.getLogger(__name__)

BASELINE_ENTRY = {
    "type": "baseline",
    "name": "naive_lag_24",
    "description": "Same hour yesterday (lag_24). Default production model until a challenger proves better.",
}


def _default_registry() -> dict:
    return {"production": dict(BASELINE_ENTRY), "history": []}


def load_registry(path: Optional[Path] = None) -> dict:
    """Load registry.json, or return the baseline-default registry if it doesn't exist yet."""
    path = Path(path or config.REGISTRY_PATH)
    if not path.exists():
        return _default_registry()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: dict, path: Optional[Path] = None) -> Path:
    path = Path(path or config.REGISTRY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, default=str)
    return path


def evaluate_promotion(
    challenger_name: str,
    challenger_mae: float,
    challenger_metrics: dict,
    production_mae: float,
    min_improvement_mw: float = config.PROMOTION_MIN_MAE_IMPROVEMENT_MW,
) -> dict:
    """Decide whether a challenger should replace the current production model.

    Returns a decision record; does NOT mutate the registry (see promote()).
    """
    improvement = production_mae - challenger_mae
    promoted = improvement > min_improvement_mw
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "challenger": challenger_name,
        "challenger_mae": challenger_mae,
        "challenger_metrics": dict(challenger_metrics),
        "production_mae_before": production_mae,
        "improvement_mw": improvement,
        "min_improvement_required_mw": min_improvement_mw,
        "promoted": promoted,
        "reason": (
            f"{challenger_name} beat production by {improvement:.2f} MW "
            f"(> required {min_improvement_mw} MW)."
            if promoted else
            f"{challenger_name} did not beat production by more than {min_improvement_mw} MW "
            f"(improvement was {improvement:.2f} MW)."
        ),
    }


def promote(
    challenger_entry: dict[str, Any],
    challenger_mae: float,
    challenger_metrics: dict,
    path: Optional[Path] = None,
    min_improvement_mw: float = config.PROMOTION_MIN_MAE_IMPROVEMENT_MW,
) -> dict:
    """Run the promotion decision against the CURRENT registry, apply it if it wins,
    record it in history either way, and save. Returns the decision record.
    """
    registry = load_registry(path)
    production_mae = registry["production"].get("mae")
    if production_mae is None:
        raise ValueError(
            "Current production entry has no recorded 'mae'; cannot compare fairly. "
            "Re-score production on the same evaluation rows as the challenger first."
        )

    decision = evaluate_promotion(
        challenger_entry.get("name", "candidate"),
        challenger_mae,
        challenger_metrics,
        production_mae,
        min_improvement_mw,
    )
    registry["history"].append(decision)
    if decision["promoted"]:
        new_production = dict(challenger_entry)
        new_production["mae"] = challenger_mae
        new_production["metrics"] = dict(challenger_metrics)
        new_production["promoted_at"] = decision["timestamp"]
        registry["production"] = new_production
        logger.info("PROMOTED: %s", decision["reason"])
    else:
        logger.info("NOT promoted: %s", decision["reason"])

    save_registry(registry, path)
    return decision
