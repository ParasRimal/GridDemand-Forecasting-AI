"""Tests for performance degradation checking and the combined retraining decision."""
import numpy as np
import pandas as pd
import pytest

from ml_service.app.monitoring.performance import check_performance
from ml_service.app.monitoring.retraining import decide_retraining


def make_reference(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.uniform(0, 1, n)})


def offset_series(offset: float, n: int = 20, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """y_true varies (so R2 is defined); y_pred is a constant offset from it,
    so MAE == abs(offset) exactly."""
    rng = np.random.default_rng(seed)
    y_true = pd.Series(rng.uniform(80.0, 120.0, n))
    y_pred = y_true + offset
    return y_true, y_pred


# --- performance ---

def test_error_within_threshold_is_not_degraded():
    y_true, y_pred = offset_series(5.0)  # MAE = 5, reference = 5 -> 0% increase
    r = check_performance(y_true, y_pred, reference_mae=5.0)
    assert r["degraded"] is False
    assert r["relative_increase"] == pytest.approx(0.0)


def test_error_just_above_threshold_is_degraded():
    y_true, y_pred = offset_series(6.0)  # MAE = 6, reference = 5 -> +20% > 15% threshold
    r = check_performance(y_true, y_pred, reference_mae=5.0)
    assert r["degraded"] is True
    assert r["relative_increase"] == pytest.approx(0.2)


def test_error_below_reference_is_not_degraded():
    y_true, y_pred = offset_series(2.0)  # better than reference
    r = check_performance(y_true, y_pred, reference_mae=5.0)
    assert r["degraded"] is False
    assert r["relative_increase"] < 0


def test_non_positive_reference_mae_is_rejected():
    with pytest.raises(ValueError):
        check_performance(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), reference_mae=0.0)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        check_performance(pd.Series([], dtype=float), pd.Series([], dtype=float), reference_mae=5.0)


# --- combined retraining decision ---

def test_no_drift_no_degradation_means_no_retrain():
    ref = make_reference()
    cur = make_reference(seed=1)
    y_true, y_pred = offset_series(2.0)  # MAE 2, ref 5 -> fine
    d = decide_retraining(ref, cur, y_true, y_pred, reference_mae=5.0)
    assert d["retrain"] is False
    assert d["drift"]["dataset_drifted"] is False
    assert d["performance"]["degraded"] is False


def test_drift_alone_triggers_retrain():
    ref = make_reference()
    cur = ref.copy()
    cur["a"] += 10.0
    cur["b"] += 10.0  # both columns shifted -> drift
    y_true, y_pred = offset_series(2.0)  # performance fine
    d = decide_retraining(ref, cur, y_true, y_pred, reference_mae=5.0)
    assert d["retrain"] is True
    assert d["drift"]["dataset_drifted"] is True
    assert d["performance"]["degraded"] is False
    assert "drift detected" in d["reason"]
    assert "degraded" not in d["reason"]


def test_degradation_alone_triggers_retrain():
    ref = make_reference()
    cur = make_reference(seed=1)  # no drift
    y_true, y_pred = offset_series(8.0)  # MAE 8 vs reference 5 -> +60%, degraded
    d = decide_retraining(ref, cur, y_true, y_pred, reference_mae=5.0)
    assert d["retrain"] is True
    assert d["drift"]["dataset_drifted"] is False
    assert d["performance"]["degraded"] is True
    assert "performance degraded" in d["reason"]


def test_both_drift_and_degradation_are_both_named_in_the_reason():
    ref = make_reference()
    cur = ref.copy()
    cur["a"] += 10.0
    cur["b"] += 10.0
    y_true, y_pred = offset_series(8.0)
    d = decide_retraining(ref, cur, y_true, y_pred, reference_mae=5.0)
    assert d["retrain"] is True
    assert "drift detected" in d["reason"]
    assert "performance degraded" in d["reason"]
