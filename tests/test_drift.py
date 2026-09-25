"""Tests for feature drift detection."""
import numpy as np
import pandas as pd
import pytest

from ml_service.app.monitoring.drift import check_drift
from ml_service.app.preprocessing.data_loader import load_clean_data
from ml_service.app.preprocessing.pipeline import build_features, feature_columns
from ml_service.app.training.split import split_chronological


def make_reference(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.uniform(0, 1, n),
        "c": rng.integers(0, 2, n),
    })


def test_identical_distributions_show_no_drift():
    ref = make_reference()
    cur = make_reference(seed=1)  # same distribution, different random draw
    result = check_drift(ref, cur)
    assert result["dataset_drifted"] is False
    assert result["drifted_column_share"] < 0.5


def test_all_columns_shifted_shows_drift():
    ref = make_reference()
    cur = ref.copy()
    cur["a"] += 10.0
    cur["b"] += 10.0
    cur["c"] = 1 - cur["c"]
    result = check_drift(ref, cur)
    assert result["dataset_drifted"] is True
    assert result["drifted_column_share"] > 0.5
    assert result["columns"]["a"]["drifted"] is True


def test_only_one_of_three_columns_shifted_does_not_trigger_dataset_drift():
    ref = make_reference()
    cur = ref.copy()
    cur["a"] += 10.0  # only 1 of 3 columns shifted -> share = 0.33, below default 0.5
    result = check_drift(ref, cur)
    assert result["columns"]["a"]["drifted"] is True
    assert result["dataset_drifted"] is False


def test_threshold_is_configurable():
    ref = make_reference()
    cur = ref.copy()
    cur["a"] += 10.0
    lenient = check_drift(ref, cur, drift_share_threshold=0.9)
    strict = check_drift(ref, cur, drift_share_threshold=0.1)
    assert lenient["dataset_drifted"] is False
    assert strict["dataset_drifted"] is True


def test_column_subset_is_respected():
    ref = make_reference()
    cur = ref.copy()
    cur["a"] += 10.0
    result = check_drift(ref, cur, columns=["b", "c"])
    assert set(result["columns"]) == {"b", "c"}
    assert result["n_columns"] == 2


def test_missing_column_is_rejected():
    ref, cur = make_reference(), make_reference()
    with pytest.raises(KeyError):
        check_drift(ref, cur, columns=["not_a_column"])


def test_empty_input_is_rejected():
    ref = make_reference()
    with pytest.raises(ValueError):
        check_drift(ref, ref.iloc[0:0])


def test_real_train_vs_test_split_shows_major_drift():
    """Locks in today's finding: the test period looks very different from training."""
    features = build_features(load_clean_data())
    train, val, test = split_chronological(features)
    result = check_drift(train, test, columns=feature_columns())
    assert result["dataset_drifted"] is True
    assert result["drifted_column_share"] > 0.5
