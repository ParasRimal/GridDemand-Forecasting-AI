"""Tests for GET /model and GET /metrics."""
from fastapi.testclient import TestClient

from ml_service.app.main import app


def test_model_endpoint_describes_the_baseline_and_its_required_features():
    with TestClient(app) as client:
        response = client.get("/model")
        assert response.status_code == 200
        body = response.json()
        assert body["type"] == "baseline"
        assert body["required_features"] == ["lag_24"]


def test_metrics_endpoint_returns_the_baseline_mae():
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        body = response.json()
        assert body["model_type"] == "baseline"
        assert body["mae"] == 9.01
