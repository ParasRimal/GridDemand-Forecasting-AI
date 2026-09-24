"""Tests for the automatic model selection rule."""
import pytest

from ml_service.app.evaluation.selection import select_best_model


def s(mae, rmse):
    return {"mae": mae, "rmse": rmse}


def test_lowest_mae_wins_when_not_a_tie():
    r = select_best_model({"a": s(8.0, 12.0), "b": s(8.5, 9.0)}, baseline_mae=9.0)
    assert r.winner == "a"


def test_tie_within_tolerance_lower_rmse_wins():
    r = select_best_model({"a": s(8.50, 12.0), "b": s(8.55, 11.0)}, baseline_mae=9.25)
    assert r.winner == "b"


def test_just_outside_tolerance_does_not_use_rmse():
    r = select_best_model({"a": s(8.50, 12.0), "b": s(8.61, 11.0)}, baseline_mae=9.25)
    assert r.winner == "a"


def test_model_that_loses_to_baseline_is_never_selected():
    # "a" has a superb RMSE but is worse than the baseline on MAE.
    r = select_best_model({"a": s(9.5, 1.0), "b": s(8.9, 20.0)}, baseline_mae=9.0)
    assert r.winner == "b"
    assert "a" not in r.eligible


def test_equal_to_baseline_is_not_eligible():
    r = select_best_model({"a": s(9.0, 5.0)}, baseline_mae=9.0)
    assert r.winner is None


def test_no_eligible_model_returns_no_winner():
    r = select_best_model({"a": s(10.0, 5.0), "b": s(11.0, 6.0)}, baseline_mae=9.0)
    assert r.winner is None
    assert r.eligible == []


def test_our_real_validation_scores_select_xgboost():
    scores = {
        "random_forest": s(8.52, 11.58),
        "xgboost": s(8.47, 11.46),
        "lightgbm": s(8.61, 11.67),
    }
    r = select_best_model(scores, baseline_mae=9.25)
    assert r.winner == "xgboost"
    assert set(r.eligible) == set(scores)


def test_empty_scores_are_rejected():
    with pytest.raises(ValueError):
        select_best_model({}, baseline_mae=9.0)


def test_scores_without_rmse_are_rejected():
    with pytest.raises(ValueError):
        select_best_model({"a": {"mae": 8.0}}, baseline_mae=9.0)


def test_table_lists_every_model_sorted_by_mae():
    r = select_best_model({"a": s(8.5, 1.0), "b": s(8.0, 2.0), "c": s(9.5, 3.0)}, baseline_mae=9.0)
    assert list(r.table.index) == ["b", "a", "c"]
    assert list(r.table["eligible"]) == [True, True, False]
