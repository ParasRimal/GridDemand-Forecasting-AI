"""Tests proving the lag features point at the right hours and cannot leak."""
import numpy as np
import pandas as pd
import pytest

from ml_service.app import config
from ml_service.app.preprocessing.data_loader import load_clean_data
from ml_service.app.preprocessing.lag_features import add_lag_features

TARGET = config.TARGET_COLUMN


def make_df(n: int = 300) -> pd.DataFrame:
    """Hourly data where the load at row i equals i, so lags are easy to check."""
    idx = pd.date_range("2022-01-01", periods=n, freq="h", name="Timestamp")
    return pd.DataFrame({TARGET: np.arange(n, dtype=float)}, index=idx)


def test_lag_points_to_exact_hours_earlier():
    out = add_lag_features(make_df())
    for lag in config.LAG_HOURS:
        assert out[f"lag_{lag}"].iloc[250] == 250 - lag


def test_first_rows_are_nan_and_the_rest_are_filled():
    out = add_lag_features(make_df())
    for lag in config.LAG_HOURS:
        col = out[f"lag_{lag}"]
        assert col.iloc[:lag].isna().all()
        assert col.iloc[lag:].notna().all()


def test_lag_uses_timestamps_not_row_positions():
    base = make_df()
    df = base.drop(index=base.index[30])  # remove one hour from the timeline
    out = add_lag_features(df, lags=[24])
    start = pd.Timestamp("2022-01-01")
    # Hour 54 looks back to hour 30, which was removed -> must be NaN.
    assert np.isnan(out.loc[start + pd.Timedelta(hours=54), "lag_24"])
    # Hour 55 looks back to hour 31, which exists -> must be 31.
    assert out.loc[start + pd.Timedelta(hours=55), "lag_24"] == 31.0


@pytest.mark.parametrize("bad_lag", [1, 3, 23])
def test_lag_shorter_than_horizon_is_rejected(bad_lag):
    with pytest.raises(ValueError):
        add_lag_features(make_df(), lags=[bad_lag])


def test_input_dataframe_is_not_modified():
    df = make_df()
    before = df.copy()
    add_lag_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_missing_load_stays_missing_in_the_lag():
    df = make_df()
    df.iloc[100, 0] = np.nan  # a meter dropout at hour 100
    out = add_lag_features(df, lags=[24])
    assert np.isnan(out["lag_24"].iloc[124])  # 124 looks back to 100
    assert out["lag_24"].iloc[125] == 101.0


def test_real_data_lags_match_the_earlier_hours():
    df = load_clean_data()
    out = add_lag_features(df)
    t = pd.Timestamp("2021-11-08 10:00:00")
    for lag in config.LAG_HOURS:
        expected = df.loc[t - pd.Timedelta(hours=lag), TARGET]
        assert out.loc[t, f"lag_{lag}"] == expected
