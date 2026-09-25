"""Tests for POST /predict/batch."""
from fastapi.testclient import TestClient

from ml_service.app.main import app


def test_batch_with_all_valid_items_succeeds():
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"items": [
            {"timestamp": "2022-09-15T14:00:00", "lag_24": 85.3},
            {"timestamp": "2022-09-16T14:00:00", "lag_24": 90.0},
        ]})
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 2
        assert all(r["success"] for r in results)
        assert results[0]["result"]["predicted_load_mw"] == 85.3
        assert results[1]["result"]["predicted_load_mw"] == 90.0


def test_batch_partial_failure_does_not_fail_the_whole_batch():
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"items": [
            {"timestamp": "2022-09-15T14:00:00", "lag_24": 85.3},
            {"timestamp": "2022-09-16T14:00:00"},  # missing lag_24
        ]})
        assert response.status_code == 200
        results = response.json()["results"]
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert "lag_24" in results[1]["error"]


def test_batch_results_preserve_input_order_via_index():
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"items": [
            {"timestamp": "2022-09-15T14:00:00", "lag_24": 1.0},
            {"timestamp": "2022-09-16T14:00:00", "lag_24": 2.0},
            {"timestamp": "2022-09-17T14:00:00", "lag_24": 3.0},
        ]})
        results = response.json()["results"]
        assert [r["index"] for r in results] == [0, 1, 2]
        assert [r["result"]["predicted_load_mw"] for r in results] == [1.0, 2.0, 3.0]


def test_empty_batch_returns_empty_results():
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"items": []})
        assert response.status_code == 200
        assert response.json()["results"] == []
