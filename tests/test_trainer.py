"""Tests for the model factory and trainer, using small synthetic data."""
import numpy as np
import pandas as pd
import pytest

from ml_service.app.evaluation.metrics import evaluate, r2
from ml_service.app.training.model_factory import available_models, build_model
from ml_service.app.training.trainer import predict, score_model, train_model


def make_data(n: int = 300, seed: int = 0):
    """y = 3a + b + small noise, with some missing values in b."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="h", name="Timestamp")
    a = rng.uniform(0, 10, n)
    b = rng.uniform(0, 10, n)
    y = pd.Series(3 * a + b + rng.normal(0, 0.1, n), index=idx, name="y")
    X = pd.DataFrame({"a": a, "b": b}, index=idx)
    X.iloc[::17, 1] = np.nan
    return X, y


def test_random_forest_is_available():
    assert "random_forest" in available_models()


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        build_model("not_a_model")


def test_prediction_keeps_the_timestamp_index():
    X, y = make_data()
    model = train_model("random_forest", X, y)
    pred = predict(model, X)
    assert len(pred) == len(X)
    assert pred.index.equals(X.index)
    assert np.isfinite(pred).all()


def test_same_seed_gives_identical_predictions():
    X, y = make_data()
    p1 = predict(train_model("random_forest", X, y), X)
    p2 = predict(train_model("random_forest", X, y), X)
    pd.testing.assert_series_equal(p1, p2)


def test_model_learns_a_pattern_on_unseen_rows_despite_nan():
    X, y = make_data()
    model = train_model("random_forest", X.iloc[:200], y.iloc[:200])
    assert r2(y.iloc[200:], predict(model, X.iloc[200:])) > 0.8


def test_missing_target_is_rejected():
    X, y = make_data()
    y.iloc[5] = np.nan
    with pytest.raises(ValueError):
        train_model("random_forest", X, y)


def test_empty_training_data_is_rejected():
    X, y = make_data()
    with pytest.raises(ValueError):
        train_model("random_forest", X.iloc[:0], y.iloc[:0])


def test_score_model_only_uses_the_selected_rows():
    X, y = make_data()
    model = train_model("random_forest", X.iloc[:200], y.iloc[:200])
    X_new, y_new = X.iloc[200:], y.iloc[200:]
    rows = pd.Series(np.arange(len(X_new)) % 2 == 0, index=X_new.index)
    expected = evaluate(y_new[rows], predict(model, X_new)[rows])
    assert score_model(model, X_new, y_new, rows) == pytest.approx(expected)
