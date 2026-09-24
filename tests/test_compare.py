"""Tests for the model comparison runner, using small synthetic data."""
import numpy as np
import pandas as pd
import pytest

from ml_service.app.training.compare import compare_models

MODELS = ["random_forest", "xgboost", "lightgbm"]


def make_split(n: int = 400, seed: int = 0):
    """y = 3a + b + tiny noise. lag_24 is a noisy copy of y, with some NaN."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="h", name="Timestamp")
    a = rng.uniform(0, 10, n)
    b = rng.uniform(0, 10, n)
    y = pd.Series(3 * a + b + rng.normal(0, 0.1, n), index=idx, name="y")
    lag = y + rng.normal(0, 5, n)
    lag.iloc[::10] = np.nan
    X = pd.DataFrame({"a": a, "b": b, "lag_24": lag.to_numpy()}, index=idx)
    return X.iloc[:300], y.iloc[:300], X.iloc[300:], y.iloc[300:]


@pytest.fixture(scope="module")
def result():
    return compare_models(*make_split())


def test_every_model_gets_scores_and_is_returned(result):
    assert set(result["scores"]) == set(MODELS)
    assert set(result["models"]) == set(MODELS)


def test_baseline_and_models_are_scored_on_the_same_rows(result):
    _, _, X_val, _ = make_split()
    assert result["baseline"]["rows_scored"] == int(X_val["lag_24"].notna().sum())


def test_winner_beats_the_baseline(result):
    winner = result["selection"].winner
    assert winner in MODELS
    assert result["scores"][winner]["mae"] < result["baseline"]["mae"]


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        compare_models(*make_split(), model_names=["not_a_model"])


def test_missing_baseline_column_is_rejected():
    X_tr, y_tr, X_val, y_val = make_split()
    with pytest.raises(KeyError):
        compare_models(X_tr, y_tr, X_val.drop(columns="lag_24"), y_val)
