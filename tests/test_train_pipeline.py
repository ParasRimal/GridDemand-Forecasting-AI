"""Tests for the full training run (uses the real processed data, saves to a temp folder)."""
import pandas as pd
import pytest

from ml_service.app.evaluation.selection import SelectionResult
from ml_service.app.preprocessing.data_loader import load_clean_data
from ml_service.app.preprocessing.pipeline import build_features, feature_columns
from ml_service.app.training import train_pipeline
from ml_service.app.training.artifacts import load_artifact
from ml_service.app.training.model_factory import available_models


@pytest.fixture(scope="module")
def features():
    return build_features(load_clean_data())


@pytest.fixture(scope="module")
def run(features, tmp_path_factory):
    models_dir = tmp_path_factory.mktemp("models")
    return models_dir, train_pipeline.run_training(features=features, models_dir=models_dir)


def test_a_winner_is_chosen_and_saved_in_exactly_one_folder(run):
    models_dir, out = run
    assert out["winner"] in available_models()
    assert out["artifact_folder"].parent == models_dir
    assert len(list(models_dir.iterdir())) == 1


def test_metadata_describes_the_model_honestly(run):
    _, out = run
    _, meta = load_artifact(out["artifact_folder"])
    assert meta["status"] == "candidate"
    assert meta["trained_on"] == "train split only"
    assert meta["feature_columns"] == feature_columns()
    assert meta["forecast_horizon_hours"] == 24
    assert meta["split"]["train_end"] < meta["split"]["validation_start"]
    assert meta["split"]["validation_end"] < meta["split"]["test_start"]


def test_metadata_has_validation_and_test_scores(run):
    _, out = run
    _, meta = load_artifact(out["artifact_folder"])
    for key in ["mae", "rmse", "mape", "r2"]:
        assert key in meta["validation_scores"]
        assert key in meta["test_scores"]["model_same_rows"]
        assert key in meta["test_scores"]["baseline"]


def test_saved_model_predicts_on_the_test_period(run, features):
    _, out = run
    model, meta = load_artifact(out["artifact_folder"])
    test_start = pd.Timestamp(meta["split"]["test_start"])
    X_test = features.loc[features.index >= test_start, feature_columns()]
    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert pd.Series(pred).notna().all()


def test_nothing_is_saved_when_no_model_beats_the_baseline(features, tmp_path, monkeypatch):
    fake = {
        "baseline": {"mae": 1.0},
        "scores": {},
        "models": {},
        "selection": SelectionResult(
            winner=None, eligible=[], reason="No model beat the baseline.", table=pd.DataFrame()
        ),
    }
    monkeypatch.setattr(train_pipeline, "compare_models", lambda *a, **k: fake)
    out = train_pipeline.run_training(features=features, models_dir=tmp_path)
    assert out["winner"] is None
    assert out["artifact_folder"] is None
    assert list(tmp_path.iterdir()) == []
