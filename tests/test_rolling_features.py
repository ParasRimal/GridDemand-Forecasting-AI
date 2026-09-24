"""Tests proving the rolling features end at t-24 and cannot leak the future."""
import math

import numpy as np
import pandas as pd
import pytest

from ml_service.app import config
from ml_service.app.preprocessing.data_loader import load_clean_data
from ml_service.app.preprocessing.rolling_features import add_rolling_features

TARGET = config.TARGET_COLUMN


def make_df(n: int = 400) -> pd.DataFrame:
    """Hourly data where the load at row i equals i, so results are easy to predict."""
    idx = pd.date_range("2022-01-01", periods=n, freq="h", name="Timestamp")
    return pd.DataFrame({TARGET: np.arange(n, dtype=float)}, index=idx)


@pytest.mark.parametrize("w", [24, 168])
def test_rolling_mean_and_std_match_the_formula(w):
    i = 300
    out = add_rolling_features(make_df(), windows=[w])
    # Window covers rows (i-24-w+1) .. (i-24). For load = row number:
    # mean = (i-24) - (w-1)/2 ; sample std of w consecutive integers = sqrt(w(w+1)/12)
    assert out[f"rolling_mean_{w}"].iloc[i] == pytest.approx((i - 24) - (w - 1) / 2)
    assert out[f"rolling_std_{w}"].iloc[i] == pytest.approx(math.sqrt(w * (w + 1) / 12))


def test_hours_after_forecast_time_cannot_change_the_feature():
    i = 300
    df = make_df()
    base = add_rolling_features(df, windows=[24])

    # Hours t-23 ... t are NOT known at forecast time. Wreck them.
    future = df.copy()
    future.iloc[i - 23 : i + 1, 0] = 9999.0  # the only column is the target
    after_future = add_rolling_features(future, windows=[24])
    for col in ["rolling_mean_24", "rolling_std_24"]:
        assert after_future[col].iloc[i] == pytest.approx(base[col].iloc[i])

    # Sanity check: hour t-24 IS known, so changing it must change the feature.
    known = df.copy()
    known.iloc[i - 24, 0] = 9999.0
    after_known = add_rolling_features(known, windows=[24])
    assert after_known["rolling_mean_24"].iloc[i] != pytest.approx(base["rolling_mean_24"].iloc[i])


def test_first_rows_are_nan_until_enough_history_exists():
    out = add_rolling_features(make_df(), windows=[24])
    # min_periods = ceil(0.5 * 24) = 12 known hours; the first result exists at row 35.
    assert out[["rolling_mean_24", "rolling_std_24"]].iloc[:35].isna().all().all()
    assert out[["rolling_mean_24", "rolling_std_24"]].iloc[35].notna().all()


def test_missing_loads_are_ignored_inside_the_window():
    i = 300
    df = make_df()
    df.iloc[271, 0] = np.nan  # inside the window rows 253..276
    out = add_rolling_features(df, windows=[24])
    expected = np.mean([v for v in range(253, 277) if v != 271])
    assert out["rolling_mean_24"].iloc[i] == pytest.approx(expected)


def test_input_dataframe_is_not_modified():
    df = make_df()
    before = df.copy()
    add_rolling_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_real_data_matches_a_manual_calculation():
    df = load_clean_data()
    out = add_rolling_features(df, windows=[24])
    t = pd.Timestamp("2021-11-20 10:00:00")
    known = df.loc[t - pd.Timedelta(hours=47) : t - pd.Timedelta(hours=24), TARGET]
    assert len(known) == 24
    assert out.loc[t, "rolling_mean_24"] == pytest.approx(known.mean())
    assert out.loc[t, "rolling_std_24"] == pytest.approx(known.std())
