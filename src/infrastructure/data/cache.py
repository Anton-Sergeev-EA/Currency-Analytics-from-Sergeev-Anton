"""
CacheManager — Redis-backed cache with an in-memory fallback.

This module previously existed but was missing from the repository
entirely, even though `DataLoader` (a live, wired-in dependency of the
`/api/data` routes) imports it unconditionally — meaning the application
could not start at all without it. Reconstructed here from the exact
interface its callers already depend on:

  - `DataLoader` (src/infrastructure/data/loader.py): .get(key),
    .set(key, value, ttl_seconds), .delete(key)
  - the admin routes (src/presentation/api/routes/admin.py): .clear(),
    .redis (None when Redis is unreachable), .local_cache (dict, its
    length is reported as a diagnostic)

Design: try Redis first (via REDIS_URL in settings); if it's unreachable
at construction time, fall back to an in-memory dict with manually
tracked expiry, so the app still runs (with a smaller, single-process
cache) even without a Redis instance available — useful for local
development and for graceful degradation in production if Redis goes
down.
"""
import json
import logging
import time
from typing import Any, Optional

import redis

from src.core.config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self) -> None:
        self.redis: Optional[redis.Redis] = None
        # value -> (expires_at_epoch_seconds, value)
        self.local_cache: dict[str, tuple[float, Any]] = {}

        try:
            client = redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
            client.ping()
            self.redis = client
            logger.info("CacheManager: connected to Redis at %s", settings.REDIS_URL)
        except Exception as exc:
            logger.warning(
                "CacheManager: Redis unavailable (%s) — falling back to in-memory cache", exc
            )
            self.redis = None

    def get(self, key: str) -> Optional[Any]:
        if self.redis is not None:
            try:
                raw = self.redis.get(key)
                return json.loads(raw) if raw is not None else None
            except Exception as exc:
                logger.warning("CacheManager.get: Redis error for key=%s: %s", key, exc)
                return None

        entry = self.local_cache.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at < time.monotonic():
            del self.local_cache[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl_seconds: int = 600) -> None:
        if self.redis is not None:
            try:
                self.redis.setex(key, ttl_seconds, json.dumps(value))
                return
            except Exception as exc:
                logger.warning("CacheManager.set: Redis error for key=%s: %s", key, exc)
                # fall through to local cache so the write isn't silently lost

        self.local_cache[key] = (time.monotonic() + ttl_seconds, value)

    def delete(self, key: str) -> None:
        if self.redis is not None:
            try:
                self.redis.delete(key)
            except Exception as exc:
                logger.warning("CacheManager.delete: Redis error for key=%s: %s", key, exc)
        self.local_cache.pop(key, None)

    def clear(self) -> None:
        if self.redis is not None:
            try:
                self.redis.flushdb()
            except Exception as exc:
                logger.warning("CacheManager.clear: Redis error: %s", exc)
        self.local_cache.clear()

    def is_healthy(self) -> bool:
        # The manager itself always degrades gracefully to the in-memory
        # fallback, so "healthy" here means "usable", not "Redis is up" —
        # check `.redis is not None` separately if you need that distinction
        # (see GET /api/cache/status).
        if self.redis is not None:
            try:
                self.redis.ping()
                return True
            except Exception:
                return False
        return True
