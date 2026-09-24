"""Tests for the chronological split."""
import numpy as np
import pandas as pd
import pytest

from ml_service.app import config
from ml_service.app.preprocessing.data_loader import load_clean_data
from ml_service.app.preprocessing.pipeline import build_features, feature_columns
from ml_service.app.training.split import split_chronological, split_xy

TARGET = config.TARGET_COLUMN


@pytest.fixture(scope="module")
def features():
    return build_features(load_clean_data())


@pytest.fixture(scope="module")
def parts(features):
    return split_chronological(features)


def small_df() -> pd.DataFrame:
    idx = pd.date_range("2022-06-29", "2022-09-03", freq="h", name="Timestamp")
    return pd.DataFrame({TARGET: np.arange(len(idx), dtype=float)}, index=idx)


def test_parts_add_up_to_the_whole_table(features, parts):
    assert sum(len(p) for p in parts) == len(features)


def test_parts_are_in_time_order_without_overlap(parts):
    train, val, test = parts
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_boundaries_follow_the_config(parts):
    train, val, test = parts
    assert train.index.max() <= pd.Timestamp(config.TRAIN_END)
    assert val.index.min() > pd.Timestamp(config.TRAIN_END)
    assert val.index.max() <= pd.Timestamp(config.VALIDATION_END)
    assert test.index.min() > pd.Timestamp(config.VALIDATION_END)


def test_expected_sizes_on_real_data(parts):
    assert [len(p) for p in parts] == [5583, 1468, 1481]


def test_unsorted_input_is_rejected():
    with pytest.raises(ValueError):
        split_chronological(small_df().iloc[::-1])


def test_empty_split_is_rejected():
    with pytest.raises(ValueError):
        split_chronological(small_df(), train_end="2020-01-01", validation_end="2020-02-01")


def test_split_dates_in_wrong_order_are_rejected():
    with pytest.raises(ValueError):
        split_chronological(small_df(), train_end="2022-08-31", validation_end="2022-07-01")


def test_split_xy_keeps_the_target_out_of_the_features(parts):
    X, y = split_xy(parts[0])
    assert list(X.columns) == feature_columns()
    assert TARGET not in X.columns
    assert y.name == TARGET
    assert len(X) == len(y)
