"""Feature drift detection using Evidently (0.7.x API).

Wraps evidently.Report + DataDriftPreset so the rest of the codebase works
with a plain dict, not Evidently's internal Snapshot/JSON structure.

Evidently auto-selects a drift-detection method per column depending on its
type, and different method families point in OPPOSITE directions:
  - distance methods (Wasserstein, Jensen-Shannon, PSI, ...): drifted when
    value > threshold (a bigger distance means more different).
  - statistical-test p-value methods (K-S, Z-test, chi-square, ...): drifted
    when value < threshold (a small p-value rejects "same distribution").
Evidently's own Snapshot.dict()["tests"] list is empty for this preset in
this version, so we cannot rely on it for a ready-made pass/fail; we
determine "drifted" ourselves per method family below.

DRIFT_SHARE_THRESHOLD is OUR threshold on top of the per-column flags: the
fraction of columns that must be individually flagged before the dataset as
a whole is considered meaningfully drifted. This deliberately avoids
"retrain whenever data changes" -- a couple of flaky columns should not
trigger anything.
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

from ml_service.app import config

# Method name (as reported in config["method"]) -> comparison direction.
# "above": drifted when value > threshold (distance-style methods).
# "below": drifted when value < threshold (p-value-style methods).
_METHOD_DIRECTION = {
    "Wasserstein distance (normed)": "above",
    "Jensen-Shannon distance": "above",
    "Population Stability Index": "above",
    "K-S p_value": "below",
    "Z-test p_value": "below",
    "Chi-square p_value": "below",
}


def _is_drifted(method: str, value: float, threshold: float) -> bool:
    direction = _METHOD_DIRECTION.get(method)
    if direction is None:
        raise ValueError(
            f"Unknown drift method '{method}'; add its comparison direction to "
            "_METHOD_DIRECTION in drift.py before trusting this result."
        )
    return bool(value > threshold) if direction == "above" else bool(value < threshold)


def _extract(snapshot_dict: dict) -> dict:
    """Turn Evidently's raw metrics list into {'summary': ..., 'columns': {...}}.

    The summary's own drifted-count/share (from DriftedColumnsCount) is NOT
    used here, because we recompute "drifted" per column ourselves (see
    module docstring on p-value direction), so our count may legitimately
    differ from Evidently's if its internal test wiring disagrees.
    """
    columns = {}
    for entry in snapshot_dict["metrics"]:
        cfg = entry.get("config", {})
        if not cfg.get("type", "").endswith("ValueDrift"):
            continue
        column, method, threshold, value = cfg["column"], cfg.get("method"), cfg.get("threshold"), entry["value"]
        columns[column] = {
            "method": method,
            "threshold": threshold,
            "drift_score": value,
            "drifted": _is_drifted(method, value, threshold),
        }
    if not columns:
        raise ValueError("No ValueDrift entries found in the Evidently report.")
    return {"columns": columns}


def check_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    drift_share_threshold: float = config.DRIFT_SHARE_THRESHOLD,
) -> dict:
    """Compare `current` against `reference` and return a drift summary.

    Returns:
        {
          "dataset_drifted": bool,           # share of drifted columns > threshold
          "drifted_column_share": float,
          "drifted_column_count": int,
          "n_columns": int,
          "threshold_used": float,
          "columns": {col: {"method", "threshold", "drift_score", "drifted"}, ...},
        }
    """
    cols = list(columns) if columns is not None else list(reference.columns)
    missing_ref = [c for c in cols if c not in reference.columns]
    missing_cur = [c for c in cols if c not in current.columns]
    if missing_ref or missing_cur:
        raise KeyError(f"Columns missing from reference={missing_ref}, current={missing_cur}")
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("reference and current must both be non-empty.")

    ref_ds = Dataset.from_pandas(reference[cols], data_definition=DataDefinition())
    cur_ds = Dataset.from_pandas(current[cols], data_definition=DataDefinition())

    report = Report([DataDriftPreset()])
    snapshot = report.run(cur_ds, ref_ds)
    extracted = _extract(snapshot.dict())

    n = len(extracted["columns"])
    n_drifted = sum(1 for c in extracted["columns"].values() if c["drifted"])
    share = n_drifted / n

    return {
        "dataset_drifted": bool(share > drift_share_threshold),
        "drifted_column_share": share,
        "drifted_column_count": n_drifted,
        "n_columns": n,
        "threshold_used": drift_share_threshold,
        "columns": extracted["columns"],
    }
