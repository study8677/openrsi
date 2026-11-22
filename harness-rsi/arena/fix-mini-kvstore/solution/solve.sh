#!/bin/bash
set -euo pipefail

cat <<'EOF' > /app/kvstore.py
"""A minimal in-memory key-value store with TTL and transaction support."""

from __future__ import annotations

import time
from typing import Any, Callable


class MiniKVStore:
    """In-memory key-value store supporting TTL expiration and nested transactions."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock: Callable[[], float] = clock or time.monotonic
        self._data: dict[str, Any] = {}
        self._expires: dict[str, float] = {}
        self._tx_stack: list[tuple[dict[str, Any], dict[str, float]]] = []

    def _purge_if_expired(self, key: str) -> bool:
        expires_at = self._expires.get(key)
        if expires_at is not None and self._clock() >= expires_at:
            self._data.pop(key, None)
            self._expires.pop(key, None)
            return True
        return False

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        if ttl is not None and ttl <= 0:
            self._data.pop(key, None)
            self._expires.pop(key, None)
            return
        self._data[key] = value
        if ttl is None:
            self._expires.pop(key, None)
        else:
            self._expires[key] = self._clock() + ttl

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._data or self._purge_if_expired(key):
            return default
        return self._data[key]

    def delete(self, key: str) -> bool:
        if key not in self._data or self._purge_if_expired(key):
            return False
        del self._data[key]
        self._expires.pop(key, None)
        return True

    def exists(self, key: str) -> bool:
        if key not in self._data or self._purge_if_expired(key):
            return False
        return True

    def keys(self) -> list[str]:
        live_keys = []
        for key in list(self._data.keys()):
            if not self._purge_if_expired(key):
                live_keys.append(key)
        return sorted(live_keys)

    def begin(self) -> None:
        self._tx_stack.append((dict(self._data), dict(self._expires)))

    def commit(self) -> None:
        if not self._tx_stack:
            raise RuntimeError("No active transaction to commit")
        self._tx_stack.pop()

    def rollback(self) -> None:
        if not self._tx_stack:
            raise RuntimeError("No active transaction to rollback")
        prev_data, prev_expires = self._tx_stack.pop()
        self._data = prev_data
        self._expires = prev_expires
EOF
