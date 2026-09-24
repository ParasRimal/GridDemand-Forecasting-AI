"""End-to-end tests for the feature pipeline."""
import pandas as pd
import pytest

from ml_service.app import config
from ml_service.app.preprocessing.data_loader import load_clean_data
from ml_service.app.preprocessing.pipeline import (
    build_and_save,
    build_features,
    feature_columns,
    load_processed,
)

TARGET = config.TARGET_COLUMN


@pytest.fixture(scope="module")
def clean():
    return load_clean_data()


@pytest.fixture(scope="module")
def features(clean):
    return build_features(clean)


def test_output_columns_are_features_plus_target(features):
    assert list(features.columns) == feature_columns() + [TARGET]


def test_target_is_never_a_feature():
    assert TARGET not in feature_columns()


def test_no_missing_target(features):
    assert features[TARGET].notna().all()


def test_warmup_rows_are_removed(clean, features):
    first_allowed = clean.index.min() + pd.Timedelta(hours=max(config.LAG_HOURS))
    assert features.index.min() >= first_allowed


def test_index_is_sorted_and_unique(features):
    assert features.index.is_monotonic_increasing
    assert features.index.is_unique


def test_every_lag_column_respects_the_horizon(features):
    lag_cols = [c for c in features.columns if c.startswith("lag_")]
    assert lag_cols
    assert all(int(c.split("_")[1]) >= config.FORECAST_HORIZON_HOURS for c in lag_cols)


def test_features_ignore_load_from_t_minus_23h_to_t(clean, features):
    """End-to-end leak check: wreck the target from t-23h to t, features at t must not change."""
    t = pd.Timestamp("2022-03-10 10:00:00")
    wrecked = clean.copy()
    wrecked.loc[t - pd.Timedelta(hours=23) : t, TARGET] = 9999.0
    features_wrecked = build_features(wrecked)
    pd.testing.assert_series_equal(
        features.loc[t, feature_columns()],
        features_wrecked.loc[t, feature_columns()],
        check_names=False,
    )


def test_save_and_reload_round_trip(tmp_path):
    path = tmp_path / "processed.csv"
    saved = build_and_save(path)
    loaded = load_processed(path)
    pd.testing.assert_frame_equal(
        saved, loaded, check_dtype=False, check_freq=False, check_index_type=False
    )
