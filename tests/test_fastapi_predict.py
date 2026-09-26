"""Tests for POST /predict, against whatever is actually in production."""
from fastapi.testclient import TestClient

from ml_service.app.main import app


def test_predict_with_only_lag_24_succeeds_against_the_baseline():
    """As of this project's current state, production is the baseline, which
    only needs lag_24."""
    with TestClient(app) as client:
        response = client.post("/predict", json={
            "timestamp": "2022-09-15T14:00:00",
            "lag_24": 85.3,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_load_mw"] == 85.3  # baseline just copies lag_24
        assert body["model_type"] == "baseline"


def test_predict_missing_the_required_feature_returns_422():
    with TestClient(app) as client:
        response = client.post("/predict", json={"timestamp": "2022-09-15T14:00:00"})
        assert response.status_code == 422
        assert "lag_24" in response.json()["detail"]


def test_predict_ignores_irrelevant_extra_fields_gracefully():
    """Even if the caller sends weather data the baseline doesn't need, it should
    still work -- extra optional fields are harmless."""
    with TestClient(app) as client:
        response = client.post("/predict", json={
            "timestamp": "2022-09-15T14:00:00",
            "lag_24": 90.0,
            "temperature": 33.5,
            "wind speed": 3.2,
        })
        assert response.status_code == 200
        assert response.json()["predicted_load_mw"] == 90.0


def test_invalid_timestamp_format_is_rejected():
    with TestClient(app) as client:
        response = client.post("/predict", json={
            "timestamp": "not-a-date",
            "lag_24": 90.0,
        })
        assert response.status_code in (400, 422)


def test_repeated_identical_request_is_served_from_cache():
    """The second identical call should return the same result; this also
    exercises the real Redis instance since prediction_cache isn't mocked here."""
    with TestClient(app) as client:
        payload = {"timestamp": "2022-09-20T09:00:00", "lag_24": 77.7}
        first = client.post("/predict", json=payload)
        second = client.post("/predict", json=payload)
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json() == second.json()
