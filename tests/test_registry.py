"""Tests for the production model registry and promotion gate."""
import json

import pytest

from ml_service.app.training.registry import (
    BASELINE_ENTRY,
    evaluate_promotion,
    load_registry,
    promote,
    save_registry,
)


def test_missing_registry_file_defaults_to_baseline_production(tmp_path):
    reg = load_registry(tmp_path / "does_not_exist.json")
    assert reg["production"]["type"] == "baseline"
    assert reg["production"]["name"] == BASELINE_ENTRY["name"]
    assert reg["history"] == []


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "registry.json"
    reg = {"production": {"type": "candidate", "name": "x", "mae": 8.0}, "history": []}
    save_registry(reg, path)
    assert load_registry(path) == reg


def test_clear_improvement_is_promoted():
    d = evaluate_promotion("rf", challenger_mae=8.0, challenger_metrics={}, production_mae=9.01,
                           min_improvement_mw=0.3)
    assert d["promoted"] is True
    assert d["improvement_mw"] == pytest.approx(1.01)


def test_improvement_below_the_margin_is_not_promoted():
    d = evaluate_promotion("rf", challenger_mae=8.9, challenger_metrics={}, production_mae=9.01,
                           min_improvement_mw=0.3)
    assert d["promoted"] is False


def test_a_worse_challenger_is_not_promoted():
    d = evaluate_promotion("rf", challenger_mae=10.4, challenger_metrics={}, production_mae=9.01,
                           min_improvement_mw=0.3)
    assert d["promoted"] is False
    assert d["improvement_mw"] < 0


def test_promote_updates_registry_when_challenger_wins(tmp_path):
    path = tmp_path / "registry.json"
    save_registry({"production": {"type": "baseline", "name": "naive_lag_24", "mae": 9.01}, "history": []}, path)
    decision = promote(
        {"type": "candidate", "name": "random_forest_no_season"},
        challenger_mae=8.0, challenger_metrics={"mae": 8.0, "rmse": 11.0},
        path=path, min_improvement_mw=0.3,
    )
    assert decision["promoted"] is True
    reg = load_registry(path)
    assert reg["production"]["name"] == "random_forest_no_season"
    assert reg["production"]["mae"] == 8.0
    assert len(reg["history"]) == 1


def test_promote_keeps_production_when_challenger_loses_but_logs_the_attempt(tmp_path):
    path = tmp_path / "registry.json"
    save_registry({"production": {"type": "baseline", "name": "naive_lag_24", "mae": 9.01}, "history": []}, path)
    decision = promote(
        {"type": "candidate", "name": "random_forest_no_season"},
        challenger_mae=10.4, challenger_metrics={"mae": 10.4}, path=path,
    )
    assert decision["promoted"] is False
    reg = load_registry(path)
    assert reg["production"]["name"] == "naive_lag_24"  # unchanged
    assert len(reg["history"]) == 1
    assert reg["history"][0]["promoted"] is False


def test_promote_refuses_when_production_has_no_recorded_mae(tmp_path):
    path = tmp_path / "registry.json"
    save_registry({"production": {"type": "baseline", "name": "naive_lag_24"}, "history": []}, path)
    with pytest.raises(ValueError):
        promote({"type": "candidate", "name": "x"}, challenger_mae=8.0, challenger_metrics={}, path=path)


def test_our_real_result_would_not_promote_either_candidate(tmp_path):
    """Sanity check against this project's actual test-set numbers."""
    path = tmp_path / "registry.json"
    save_registry({"production": {"type": "baseline", "name": "naive_lag_24", "mae": 9.01}, "history": []}, path)
    for name, mae in [("xgboost_full", 10.67), ("random_forest_no_season", 10.40)]:
        decision = promote({"type": "candidate", "name": name}, challenger_mae=mae,
                            challenger_metrics={"mae": mae}, path=path)
        assert decision["promoted"] is False
    reg = load_registry(path)
    assert reg["production"]["name"] == "naive_lag_24"
    assert len(reg["history"]) == 2
