# Fix `MiniKVStore` in `/app/kvstore.py`

You are given a minimal in-memory key-value store implementation in `/app/kvstore.py` (with a basic smoke test in `/app/test_smoke.py`).

Fix all bugs in `/app/kvstore.py` so that `MiniKVStore` satisfies the full specification below:

## Specification

1. **Clock & Expiration (TTL)**
   - `MiniKVStore(clock=None)` accepts an optional zero-argument callable returning the current timestamp in seconds (defaults to `time.monotonic`).
   - `set(key, value, ttl=None)`:
     - Stores `value` under `key`.
     - If `ttl is not None` and `ttl > 0`, the key expires at `clock() + ttl`.
     - A key is considered expired whenever `clock() >= expires_at` (inclusive boundary).
     - If `ttl is not None` and `ttl <= 0`, the key is immediately expired (must not be present, and any previous value for `key` must be removed).
     - If `ttl is None`, any previously set TTL on `key` must be cleared so the key does not expire.
   - `get(key, default=None)`:
     - Returns `default` if `key` does not exist or has expired (and cleans up expired entries).
   - `delete(key) -> bool`:
     - Removes `key` and returns `True` only if `key` currently exists and is **not** expired; returns `False` if missing or already expired.
   - `exists(key) -> bool`:
     - Returns `True` only if `key` exists and is not expired.
   - `keys() -> list[str]`:
     - Returns a **sorted** list of all currently non-expired keys.

2. **Nested Transactions (`begin`, `commit`, `rollback`)**
   - `begin()` starts a new transaction. Transactions may be arbitrarily nested (`begin()` inside `begin()`).
   - `commit()` commits the innermost active transaction:
     - If nested, changes become part of the enclosing outer transaction (so a subsequent outer `rollback()` still reverts them).
     - If outermost, changes are permanently applied.
     - Raises `RuntimeError` if there is no active transaction.
   - `rollback()` discards all changes (including both values and TTL metadata) made since the innermost `begin()`:
     - Restores deleted keys and their original TTLs, removes newly added keys, and restores overwritten values/TTLs.
     - Raises `RuntimeError` if there is no active transaction.

## Requirements

- Edit `/app/kvstore.py` in place.
- Keep the public class name `MiniKVStore` and all method signatures backward-compatible.
- Do not add external third-party runtime dependencies.
