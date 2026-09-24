"""Tests for the naive lag baselines."""
import numpy as np
import pandas as pd
import pytest

from ml_service.app.preprocessing.data_loader import load_clean_data
from ml_service.app.preprocessing.pipeline import build_features
from ml_service.app.training.baseline import evaluate_naive_baseline, naive_lag_prediction
from ml_service.app.training.split import split_chronological, split_xy


def tiny_data():
    X = pd.DataFrame({"lag_24": [100.0, 200.0, np.nan], "lag_168": [90.0, 210.0, 40.0]})
    y = pd.Series([110.0, 180.0, 50.0])
    return X, y


def test_prediction_is_the_lag_column():
    X, _ = tiny_data()
    pd.testing.assert_series_equal(naive_lag_prediction(X, 24), X["lag_24"])
    pd.testing.assert_series_equal(naive_lag_prediction(X, 168), X["lag_168"])


def test_rows_with_unknown_lag_are_skipped_and_counted():
    X, y = tiny_data()
    result = evaluate_naive_baseline(X, y, 24)
    assert result["rows_scored"] == 2
    assert result["rows_skipped"] == 1
    # Actual = [110, 180], prediction = lag = [100, 200] (roles swapped vs the metrics example).
    assert result["mae"] == pytest.approx(15.0)
    assert result["mape"] == pytest.approx(100 * (10 / 110 + 20 / 180) / 2)
    assert result["r2"] == pytest.approx(1 - 500 / 2450)


def test_missing_lag_column_is_rejected():
    X, y = tiny_data()
    with pytest.raises(KeyError):
        evaluate_naive_baseline(X, y, 48)


def test_all_nan_lag_is_rejected():
    X = pd.DataFrame({"lag_24": [np.nan, np.nan]})
    y = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        evaluate_naive_baseline(X, y, 24)


def test_inputs_are_not_modified():
    X, y = tiny_data()
    X_before, y_before = X.copy(), y.copy()
    evaluate_naive_baseline(X, y, 24)
    pd.testing.assert_frame_equal(X, X_before)
    pd.testing.assert_series_equal(y, y_before)


def test_real_validation_rows_are_all_accounted_for():
    _, val, _ = split_chronological(build_features(load_clean_data()))
    X, y = split_xy(val)
    result = evaluate_naive_baseline(X, y, 24)
    assert result["rows_scored"] + result["rows_skipped"] == len(val)
