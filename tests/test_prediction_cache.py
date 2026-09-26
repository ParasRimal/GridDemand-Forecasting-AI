"""Tests for the Redis prediction cache: real-Redis behavior plus fail-open guarantees."""
from unittest.mock import MagicMock, patch

import redis

from ml_service.app.caching import prediction_cache as pc


def test_cache_key_is_stable_regardless_of_dict_order():
    a = {"timestamp": "2022-09-15T14:00:00", "lag_24": 85.3}
    b = {"lag_24": 85.3, "timestamp": "2022-09-15T14:00:00"}
    assert pc.make_cache_key(a) == pc.make_cache_key(b)


def test_cache_key_differs_for_different_payloads():
    a = {"timestamp": "2022-09-15T14:00:00", "lag_24": 85.3}
    b = {"timestamp": "2022-09-15T14:00:00", "lag_24": 90.0}
    assert pc.make_cache_key(a) != pc.make_cache_key(b)


# --- fail-open behavior (mocked, no real Redis needed) ---

def test_get_returns_none_when_redis_raises():
    pc._client = MagicMock()
    pc._client.get.side_effect = redis.RedisError("connection refused")
    result = pc.get_cached_prediction({"timestamp": "x", "lag_24": 1.0})
    assert result is None
    pc._client = None  # reset shared client for other tests


def test_set_does_not_raise_when_redis_raises():
    pc._client = MagicMock()
    pc._client.set.side_effect = redis.RedisError("connection refused")
    pc.set_cached_prediction({"timestamp": "x", "lag_24": 1.0}, {"predicted_load_mw": 1.0})
    pc._client = None  # reset shared client for other tests


# --- real Redis integration (this environment has a live Redis on 127.0.0.1:6379) ---

def test_set_then_get_round_trips_through_real_redis():
    pc._client = None  # force a fresh real connection
    payload = {"timestamp": "2022-09-15T14:00:00", "lag_24": 85.3}
    response = {"predicted_load_mw": 85.3, "model_type": "baseline"}

    pc.set_cached_prediction(payload, response)
    result = pc.get_cached_prediction(payload)

    assert result == response


def test_get_returns_none_for_a_key_never_set():
    pc._client = None
    payload = {"timestamp": "2099-01-01T00:00:00", "lag_24": 999.99}
    assert pc.get_cached_prediction(payload) is None
