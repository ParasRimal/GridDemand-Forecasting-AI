"""The same checks for every model in the factory, on small synthetic data."""
import numpy as np
import pandas as pd
import pytest

from ml_service.app.evaluation.metrics import r2
from ml_service.app.training.model_factory import available_models
from ml_service.app.training.trainer import predict, train_model

MODELS = ["random_forest", "xgboost", "lightgbm"]


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


def test_all_three_models_are_available():
    assert set(MODELS) <= set(available_models())


@pytest.mark.parametrize("name", MODELS)
def test_trains_and_predicts_with_nan_features(name):
    X, y = make_data()
    pred = predict(train_model(name, X, y), X)
    assert pred.index.equals(X.index)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("name", MODELS)
def test_same_seed_gives_identical_predictions(name):
    X, y = make_data()
    p1 = predict(train_model(name, X, y), X)
    p2 = predict(train_model(name, X, y), X)
    pd.testing.assert_series_equal(p1, p2)


@pytest.mark.parametrize("name", MODELS)
def test_learns_a_pattern_on_unseen_rows(name):
    X, y = make_data()
    model = train_model(name, X.iloc[:200], y.iloc[:200])
    assert r2(y.iloc[200:], predict(model, X.iloc[200:])) > 0.8
