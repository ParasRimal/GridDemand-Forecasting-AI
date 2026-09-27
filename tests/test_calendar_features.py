"""Tests for calendar_features.py -- verified manually in Task 19 but never
given a dedicated automated test file, unlike lag/rolling features."""
import pandas as pd
import pytest

from ml_service.app.preprocessing.calendar_features import (
    CALENDAR_FEATURES,
    add_calendar_features,
)


def make_df():
    idx = pd.date_range("2021-11-01", "2022-11-01 23:00", freq="h", name="Timestamp")
    return pd.DataFrame({"Electric Load (MW)": range(len(idx))}, index=idx)


def test_all_seven_calendar_columns_are_added():
    out = add_calendar_features(make_df())
    for col in CALENDAR_FEATURES:
        assert col in out.columns
    assert len(CALENDAR_FEATURES) == 7


def test_no_missing_values_in_any_calendar_column():
    out = add_calendar_features(make_df())
    assert out[CALENDAR_FEATURES].isna().sum().sum() == 0


def test_diwali_2021_is_correctly_flagged_as_a_holiday():
    """Diwali fell on 2021-11-04, and Gujarat holidays (config.HOLIDAY_SUBDIVISION)
    should mark it -- this was manually verified in Task 19."""
    out = add_calendar_features(make_df())
    row = out.loc["2021-11-04 10:00:00"]
    assert row["is_holiday"] == 1
    assert row["day_of_week"] == 3  # Thursday


def test_a_known_sunday_is_flagged_as_weekend_not_holiday():
    """2022-01-02 was a Sunday, manually verified in Task 19."""
    out = add_calendar_features(make_df())
    row = out.loc["2022-01-02 10:00:00"]
    assert row["day_of_week"] == 6
    assert row["is_weekend"] == 1
    assert row["is_holiday"] == 0


def test_weekend_flag_is_true_only_for_saturday_and_sunday():
    out = add_calendar_features(make_df())
    weekend_rows = out[out["is_weekend"] == 1]
    assert set(weekend_rows["day_of_week"].unique()) == {5, 6}


def test_holiday_count_matches_the_known_27_gujarat_holidays():
    """27 Gujarat holiday days x 24 hours = 648 holiday-flagged rows, per Task 19."""
    out = add_calendar_features(make_df())
    assert out["is_holiday"].sum() == 648


def test_requires_a_datetime_index():
    df = pd.DataFrame({"Electric Load (MW)": [1, 2, 3]})  # plain integer index
    with pytest.raises(TypeError):
        add_calendar_features(df)
