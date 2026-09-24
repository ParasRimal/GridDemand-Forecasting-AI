"""Tests for the rolling backtest, using synthetic hourly data."""
import numpy as np
import pandas as pd
import pytest

from ml_service.app import config
from ml_service.app.training import backtest as bt

TARGET = config.TARGET_COLUMN
FEATURES = ["a", "b"]


def make_df(seed: int = 0) -> pd.DataFrame:
    """Jan-Sep 2022, so the data extends past the end of validation (31 Aug)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", "2022-09-30 23:00", freq="h", name="Timestamp")
    a = rng.uniform(0, 10, len(idx))
    b = rng.uniform(0, 10, len(idx))
    y = pd.Series(3 * a + b + rng.normal(0, 0.5, len(idx)), index=idx)
    return pd.DataFrame({"a": a, "b": b, TARGET: y, "lag_24": y.shift(24)}, index=idx)


def test_the_test_period_is_never_used():
    out = bt.backtest(make_df(), "xgboost", first_fold="2022-03", features=FEATURES)
    assert out.index.max() == "2022-08"
    assert "2022-09" not in out.index


def test_training_stops_at_least_24h_before_each_fold(monkeypatch):
    seen = []

    class Spy:
        def fit(self, X, y):
            seen.append(X.index.max())

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(bt, "build_model", lambda name: Spy())
    out = bt.backtest(make_df(), "xgboost", first_fold="2022-03", features=FEATURES)
    assert len(seen) == len(out)
    for month, last_train in zip(out.index, seen):
        assert last_train < pd.Period(month).start_time - pd.Timedelta(hours=24)


def test_no_usable_folds_raises():
    with pytest.raises(ValueError):
        bt.backtest(make_df(), "xgboost", first_fold="2023-01", features=FEATURES)


def test_output_has_one_row_per_month_and_expected_columns():
    out = bt.backtest(make_df(), "xgboost", first_fold="2022-03", features=FEATURES)
    assert list(out.index) == [f"2022-0{m}" for m in range(3, 9)]
    for col in ["train_rows", "test_rows", "actual_mean", "model_bias",
                "model_mae", "baseline_mae", "beats_baseline"]:
        assert col in out.columns


def test_beats_flag_matches_the_scores():
    out = bt.backtest(make_df(), "xgboost", first_fold="2022-03", features=FEATURES)
    assert (out["beats_baseline"] == (out["model_mae"] < out["baseline_mae"])).all()
    # y = 3a + b is learnable from a and b, while lag_24 is just noise, so the model should win.
    assert out["beats_baseline"].all()


def test_unknown_target_mode_is_rejected():
    with pytest.raises(ValueError):
        bt.backtest(make_df(), "xgboost", first_fold="2022-03", features=FEATURES, target_mode="nope")


def test_delta_mode_trains_on_the_change_from_the_lag(monkeypatch):
    seen = {}

    class Spy:
        def fit(self, X, y):
            seen["X"], seen["y"] = X, y

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(bt, "build_model", lambda name: Spy())
    df = make_df()
    bt.backtest(df, "xgboost", first_fold="2022-08", features=FEATURES, target_mode="delta_baseline")
    idx = seen["X"].index
    expected = df.loc[idx, TARGET] - df.loc[idx, "lag_24"]
    pd.testing.assert_series_equal(seen["y"], expected, check_names=False, check_freq=False)
    assert seen["y"].notna().all()


def test_delta_mode_adds_the_lag_back_to_the_predictions(monkeypatch):
    class Zero:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(bt, "build_model", lambda name: Zero())
    out = bt.backtest(make_df(), "xgboost", first_fold="2022-03", features=FEATURES,
                      target_mode="delta_baseline")
    # A predicted change of 0 means "same as yesterday", so the model must equal the baseline.
    assert out["model_mae"].to_numpy() == pytest.approx(out["baseline_mae"].to_numpy())
