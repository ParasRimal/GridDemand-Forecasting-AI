"""Tests for the FastAPI service startup and /health endpoint."""
from fastapi.testclient import TestClient

from ml_service.app.main import app


def test_health_returns_ok_and_current_production_info():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["production_model"]["type"] in ("baseline", "candidate")


def test_health_reflects_the_real_registry_state():
    """As of this project's current state, production should be the baseline."""
    with TestClient(app) as client:
        response = client.get("/health")
        body = response.json()
        assert body["production_model"]["type"] == "baseline"
        assert body["production_model"]["name"] == "naive_lag_24"
