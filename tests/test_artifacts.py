"""Tests for saving and loading model artifacts."""
import json

import numpy as np
import pandas as pd
import pytest

from ml_service.app.training.artifacts import (
    METADATA_FILE,
    MODEL_FILE,
    load_artifact,
    save_artifact,
)
from ml_service.app.training.trainer import predict, train_model


def make_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="h", name="Timestamp")
    X = pd.DataFrame({"a": rng.uniform(0, 10, n), "b": rng.uniform(0, 10, n)}, index=idx)
    y = pd.Series(3 * X["a"] + X["b"], index=idx, name="y")
    return X, y


@pytest.fixture(scope="module")
def trained():
    X, y = make_data()
    return X, train_model("xgboost", X, y)


def test_save_creates_both_files_in_a_new_folder(tmp_path, trained):
    _, model = trained
    folder = save_artifact(model, "xgboost", {"note": "x"}, base_dir=tmp_path)
    assert folder.parent == tmp_path
    assert folder.name.startswith("xgboost_")
    assert (folder / MODEL_FILE).exists()
    assert (folder / METADATA_FILE).exists()


def test_loaded_model_predicts_identically(tmp_path, trained):
    X, model = trained
    folder = save_artifact(model, "xgboost", {}, base_dir=tmp_path)
    loaded, _ = load_artifact(folder)
    pd.testing.assert_series_equal(predict(model, X), predict(loaded, X))


def test_metadata_round_trip_and_added_fields(tmp_path, trained):
    _, model = trained
    meta = {"feature_columns": ["a", "b"], "validation": {"mae": 8.47}}
    folder = save_artifact(model, "xgboost", meta, base_dir=tmp_path)
    _, loaded = load_artifact(folder)
    assert loaded["feature_columns"] == ["a", "b"]
    assert loaded["validation"]["mae"] == 8.47
    assert loaded["model_name"] == "xgboost"
    assert "saved_at" in loaded


def test_numpy_numbers_in_metadata_are_saved_as_plain_numbers(tmp_path, trained):
    _, model = trained
    meta = {"mae": np.float64(8.47), "rows": np.int64(1448)}
    folder = save_artifact(model, "xgboost", meta, base_dir=tmp_path)
    raw = json.loads((folder / METADATA_FILE).read_text())
    assert raw["mae"] == 8.47
    assert raw["rows"] == 1448


def test_input_metadata_is_not_modified(tmp_path, trained):
    _, model = trained
    meta = {"a": 1}
    save_artifact(model, "xgboost", meta, base_dir=tmp_path)
    assert meta == {"a": 1}


def test_loading_a_missing_folder_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "does_not_exist")
