"""Tests for backtest-based model selection, using hand-made fold tables."""
import pandas as pd
import pytest

from ml_service.app.evaluation.backtest_selection import select_from_backtests


def bt(model_maes, baseline_maes=None):
    """A fake backtest table. The baseline MAE is 9.0 in every fold unless given."""
    n = len(model_maes)
    baseline_maes = baseline_maes if baseline_maes is not None else [9.0] * n
    idx = pd.Index([f"2022-{m:02d}" for m in range(1, n + 1)], name="month")
    return pd.DataFrame({"model_mae": model_maes, "baseline_mae": baseline_maes}, index=idx)


def test_lowest_mean_mae_wins_when_not_a_tie():
    r = select_from_backtests({"a": bt([8.0] * 7), "b": bt([8.5] * 7)})
    assert r.winner == "a"


def test_tie_within_tolerance_the_candidate_beating_more_folds_wins():
    a = bt([7.4] * 5 + [9.5] * 2)   # mean 8.00, beats the baseline in 5 of 7 folds
    b = bt([8.05] * 7)              # mean 8.05, beats the baseline in 7 of 7 folds
    assert select_from_backtests({"a": a, "b": b}).winner == "b"


def test_outside_tolerance_the_lower_mean_still_wins():
    a = bt([7.4] * 5 + [9.5] * 2)   # mean 8.00, 5 of 7 folds
    b = bt([8.30] * 7)              # mean 8.30, 7 of 7 folds, but 0.3 MW worse
    assert select_from_backtests({"a": a, "b": b}).winner == "a"


def test_beating_the_baseline_in_too_few_folds_is_not_eligible():
    # Great mean (8.0), but it beats the baseline in only 4 of 7 folds.
    r = select_from_backtests({"a": bt([5, 5, 5, 5, 12, 12, 12])})
    assert r.winner is None
    assert r.eligible == []


def test_enough_folds_but_a_worse_mean_is_not_eligible():
    # Beats the baseline in 5 of 7 folds, but two huge misses make the mean worse than 9.0.
    r = select_from_backtests({"a": bt([8, 8, 8, 8, 8, 30, 30])})
    assert r.winner is None


def test_no_eligible_candidate_returns_no_winner():
    r = select_from_backtests({"a": bt([10.0] * 7), "b": bt([11.0] * 7)})
    assert r.winner is None
    assert "No candidate" in r.reason


def test_fold_requirement_scales_with_the_number_of_folds():
    # 10 folds x 0.7 = 7 folds needed (the floating-point trap would demand 8).
    seven_of_ten = bt([8.0] * 7 + [10.0] * 3)   # mean 8.6
    six_of_ten = bt([8.0] * 6 + [10.0] * 4)     # mean 8.8
    assert select_from_backtests({"a": seven_of_ten}).winner == "a"
    assert select_from_backtests({"a": six_of_ten}).winner is None


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        select_from_backtests({})
    with pytest.raises(ValueError):
        select_from_backtests({"a": bt([8.0]).iloc[0:0]})


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError):
        select_from_backtests({"a": bt([8.0] * 7).drop(columns="baseline_mae")})


def test_table_lists_every_candidate_sorted_by_mean_mae():
    r = select_from_backtests({"a": bt([8.5] * 7), "b": bt([8.0] * 7), "c": bt([9.5] * 7)})
    assert list(r.table.index) == ["b", "a", "c"]
    assert list(r.table["eligible"]) == [True, True, False]
