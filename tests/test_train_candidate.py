"""Tests for training and saving the backtest-selected candidate."""
import pandas as pd
import pytest

from ml_service.app.preprocessing.data_loader import load_clean_data
from ml_service.app.preprocessing.pipeline import build_features
from ml_service.app.training import train_candidate
from ml_service.app.training.artifacts import load_artifact


@pytest.fixture(scope="module")
def features():
    return build_features(load_clean_data())


@pytest.fixture(scope="module")
def run(features, tmp_path_factory):
    models_dir = tmp_path_factory.mktemp("models")
    return models_dir, train_candidate.run(features=features, models_dir=models_dir)


def test_saved_in_exactly_one_folder(run):
    models_dir, out = run
    assert out["artifact_folder"].parent == models_dir
    assert len(list(models_dir.iterdir())) == 1


def test_season_features_are_excluded(run):
    _, out = run
    _, meta = load_artifact(out["artifact_folder"])
    for col in train_candidate.SEASON_FEATURES:
        assert col not in meta["feature_columns"]
    assert "lag_24" in meta["feature_columns"]  # a normal feature must still be there


def test_trained_on_train_plus_validation(run):
    _, out = run
    _, meta = load_artifact(out["artifact_folder"])
    assert meta["trained_on"].startswith("train + validation")
    assert pd.Timestamp(meta["train_val_end"]) < pd.Timestamp(meta["test_start"])


def test_metadata_is_explicit_about_the_test_score_caveat(run):
    _, out = run
    _, meta = load_artifact(out["artifact_folder"])
    assert "REPORT ONLY" in meta["test_score_note"]
    assert meta["status"] == "candidate"


def test_test_scores_are_internally_consistent(run):
    _, out = run
    t = out["test_scores"]
    assert out["beats_baseline_on_test"] == (t["model_same_rows"]["mae"] < t["baseline"]["mae"])


def test_saved_model_predicts_on_the_test_period(run, features):
    _, out = run
    model, meta = load_artifact(out["artifact_folder"])
    X_test = features.loc[features.index >= pd.Timestamp(meta["test_start"]), meta["feature_columns"]]
    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert pd.Series(pred).notna().all()
