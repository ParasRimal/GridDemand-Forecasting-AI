"""Tests for the forecast metrics."""
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from ml_service.app.evaluation.metrics import METRIC_NAMES, evaluate, mae, mape, r2, rmse


def test_hand_calculated_example():
    actual = [100.0, 200.0]
    predicted = [110.0, 180.0]
    assert mae(actual, predicted) == pytest.approx(15.0)
    assert rmse(actual, predicted) == pytest.approx(math.sqrt(250))
    assert mape(actual, predicted) == pytest.approx(10.0)
    assert r2(actual, predicted) == pytest.approx(0.9)


def test_perfect_prediction():
    values = [50.0, 60.0, 70.0]
    assert evaluate(values, values) == pytest.approx(
        {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 1.0}
    )


def test_predicting_the_mean_gives_r2_zero():
    actual = [10.0, 20.0, 30.0, 40.0]
    assert r2(actual, [25.0] * 4) == pytest.approx(0.0)


def test_matches_scikit_learn_on_random_data():
    rng = np.random.default_rng(0)
    actual = rng.uniform(20, 150, size=500)
    predicted = actual + rng.normal(0, 8, size=500)
    assert mae(actual, predicted) == pytest.approx(mean_absolute_error(actual, predicted))
    assert rmse(actual, predicted) == pytest.approx(math.sqrt(mean_squared_error(actual, predicted)))
    assert mape(actual, predicted) == pytest.approx(
        mean_absolute_percentage_error(actual, predicted) * 100
    )
    assert r2(actual, predicted) == pytest.approx(r2_score(actual, predicted))


def test_accepts_pandas_series_with_different_indexes():
    actual = pd.Series([100.0, 200.0], index=[5, 6])
    predicted = pd.Series([110.0, 180.0], index=["a", "b"])
    assert mae(actual, predicted) == pytest.approx(15.0)


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        mae([1.0, 2.0, 3.0], [1.0, 2.0])


@pytest.mark.parametrize("which", ["actual", "predicted"])
def test_nan_is_rejected(which):
    good = [1.0, 2.0, 3.0]
    bad = [1.0, float("nan"), 3.0]
    args = (bad, good) if which == "actual" else (good, bad)
    with pytest.raises(ValueError):
        rmse(*args)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        mae([], [])


def test_mape_rejects_a_zero_actual():
    with pytest.raises(ValueError):
        mape([0.0, 10.0], [1.0, 9.0])


def test_r2_rejects_constant_actual_values():
    with pytest.raises(ValueError):
        r2([5.0, 5.0, 5.0], [4.0, 5.0, 6.0])


def test_evaluate_returns_the_four_metrics_in_order():
    result = evaluate([100.0, 200.0], [110.0, 180.0])
    assert list(result) == METRIC_NAMES
