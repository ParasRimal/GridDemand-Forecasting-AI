"""Redis-backed cache for /predict responses.

Fail-open design: if Redis is unreachable, get()/set() calls are swallowed
and logged as a warning rather than raised. Caching is an optimization, not
a correctness requirement, so its failure must never break prediction
serving -- this was a real lesson from this project's own environment,
where 'localhost' silently failed to resolve while '127.0.0.1' worked.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

import redis

from ml_service.app import config

logger = logging.getLogger(__name__)

_client: Optional[redis.Redis] = None


def _get_client() -> redis.Redis:
    """Lazily create a single shared Redis client (real connection attempts
    only happen on first actual use, not at import time)."""
    global _client
    if _client is None:
        _client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=2,
        )
    return _client


def make_cache_key(payload: dict) -> str:
    """A stable key from a request's field values, independent of key order."""
    canonical = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{config.PREDICTION_CACHE_KEY_PREFIX}{digest}"


def get_cached_prediction(payload: dict) -> Optional[dict]:
    """Return the cached response dict, or None on a miss OR any Redis failure."""
    try:
        client = _get_client()
        raw = client.get(make_cache_key(payload))
        if raw is None:
            return None
        return json.loads(raw)
    except redis.RedisError as exc:
        logger.warning("Redis unavailable on GET, skipping cache: %s", exc)
        return None


def set_cached_prediction(payload: dict, response: dict) -> None:
    """Store a response dict; silently no-ops on any Redis failure."""
    try:
        client = _get_client()
        client.set(
            make_cache_key(payload),
            json.dumps(response, default=str),
            ex=config.PREDICTION_CACHE_TTL_SECONDS,
        )
    except redis.RedisError as exc:
        logger.warning("Redis unavailable on SET, skipping cache: %s", exc)
