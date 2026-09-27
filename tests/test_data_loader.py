"""Tests for data_loader.py's validation: duplicate timestamps, unsorted
data, and missing required columns -- none of which had a dedicated test
before, despite the loader code explicitly guarding against them."""
import pandas as pd
import pytest

from ml_service.app.preprocessing.data_loader import (
    check_timeline,
    load_raw,
)


def make_hourly_df(n=10):
    idx = pd.date_range("2022-01-01", periods=n, freq="h", name="Timestamp")
    return pd.DataFrame({"Electric Load (MW)": range(n)}, index=idx)


def test_duplicate_timestamps_are_rejected():
    df = make_hourly_df()
    dupe = pd.concat([df, df.iloc[[0]]])  # duplicate the first row's timestamp
    with pytest.raises(ValueError, match="duplicate"):
        check_timeline(dupe)


def test_unsorted_timestamps_are_sorted_not_rejected():
    """check_timeline should fix unsorted order rather than error -- confirms
    the documented 'sort, don't crash' behavior from data_loader.py."""
    df = make_hourly_df()
    shuffled = df.sample(frac=1, random_state=0)
    result = check_timeline(shuffled)
    assert result.index.is_monotonic_increasing


def test_missing_hours_are_filled_as_nan_rows():
    df = make_hourly_df(10)
    with_gap = df.drop(df.index[5])  # remove one hour from the middle
    result = check_timeline(with_gap)
    assert len(result) == 10  # the gap hour is reintroduced
    assert result["Electric Load (MW)"].isna().sum() == 1


def test_load_raw_rejects_missing_required_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"YEAR": [2022], "Month": [1]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_raw(bad_csv)


def test_load_raw_rejects_a_nonexistent_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "does_not_exist.csv")
