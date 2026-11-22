"""A minimal in-memory key-value store with TTL and transaction support."""

from __future__ import annotations

import time
from typing import Any, Callable


class MiniKVStore:
    """In-memory key-value store supporting TTL expiration and transactions."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock: Callable[[], float] = clock or time.monotonic
        self._data: dict[str, Any] = {}
        self._expires: dict[str, float] = {}
        self._in_tx: bool = False
        self._backup: dict[str, Any] = {}

    def _is_expired(self, key: str) -> bool:
        if key not in self._expires:
            return False
        # BUG: uses strict `>` instead of `>=`
        return self._clock() > self._expires[key]

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        self._data[key] = value
        if ttl is not None:
            # BUG: does not handle ttl <= 0, and else branch forgets to clear old TTL
            self._expires[key] = self._clock() + ttl

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._data:
            return default
        if self._is_expired(key):
            self._data.pop(key, None)
            self._expires.pop(key, None)
            return default
        return self._data[key]

    def delete(self, key: str) -> bool:
        # BUG: does not check expiration before reporting deletion
        if key in self._data:
            del self._data[key]
            self._expires.pop(key, None)
            return True
        return False

    def exists(self, key: str) -> bool:
        # BUG: ignores expiration
        return key in self._data

    def keys(self) -> list[str]:
        # BUG: includes expired keys and is not guaranteed sorted
        return list(self._data.keys())

    def begin(self) -> None:
        # BUG: overwrites single backup instead of maintaining a stack for nested transactions
        self._in_tx = True
        self._backup = dict(self._data)

    def commit(self) -> None:
        # BUG: silently no-ops if no active transaction, and does not support nesting
        if not self._in_tx:
            return
        self._in_tx = False
        self._backup = {}

    def rollback(self) -> None:
        # BUG: silently no-ops if no active transaction, forgets _expires, and breaks nesting
        if not self._in_tx:
            return
        self._data = dict(self._backup)
        self._in_tx = False
        self._backup = {}
