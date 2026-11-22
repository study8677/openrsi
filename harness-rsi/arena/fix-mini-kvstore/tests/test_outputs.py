"""Hidden verification suite for fix-mini-kvstore (10 deterministic tests)."""

import sys
import pytest

sys.path.insert(0, "/app")
from kvstore import MiniKVStore


class FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


def test_01_ttl_exact_expiration_boundary() -> None:
    clock = FakeClock(100.0)
    store = MiniKVStore(clock=clock)
    store.set("token", "abc", ttl=5.0)
    clock.advance(4.999)
    assert store.get("token") == "abc"
    clock.advance(0.001)  # now == 105.0 == expires_at
    assert store.get("token") is None


def test_02_ttl_non_positive_immediately_expired() -> None:
    clock = FakeClock(100.0)
    store = MiniKVStore(clock=clock)
    store.set("a", 1)
    store.set("a", 2, ttl=0)
    assert store.get("a") is None
    assert store.exists("a") is False

    store.set("b", 3, ttl=-10.0)
    assert store.get("b") is None
    assert store.exists("b") is False


def test_03_overwrite_clears_previous_ttl() -> None:
    clock = FakeClock(100.0)
    store = MiniKVStore(clock=clock)
    store.set("k", "v1", ttl=5.0)
    clock.advance(2.0)
    store.set("k", "v2", ttl=None)
    clock.advance(100.0)
    assert store.get("k") == "v2"
    assert store.exists("k") is True


def test_04_exists_and_keys_exclude_expired_and_sorted() -> None:
    clock = FakeClock(100.0)
    store = MiniKVStore(clock=clock)
    store.set("zebra", 1)
    store.set("alpha", 2, ttl=5.0)
    store.set("middle", 3)
    clock.advance(5.0)
    assert store.exists("alpha") is False
    assert store.exists("zebra") is True
    assert store.keys() == ["middle", "zebra"]


def test_05_delete_expired_key_returns_false() -> None:
    clock = FakeClock(100.0)
    store = MiniKVStore(clock=clock)
    store.set("temp", "val", ttl=3.0)
    clock.advance(3.0)
    assert store.delete("temp") is False
    assert store.get("temp") is None


def test_06_commit_and_rollback_without_tx_raise() -> None:
    store = MiniKVStore()
    with pytest.raises(RuntimeError):
        store.commit()
    with pytest.raises(RuntimeError):
        store.rollback()


def test_07_rollback_restores_deleted_key_and_ttl() -> None:
    clock = FakeClock(100.0)
    store = MiniKVStore(clock=clock)
    store.set("session", "s1", ttl=10.0)

    store.begin()
    assert store.delete("session") is True
    assert store.get("session") is None
    store.rollback()

    clock.advance(5.0)
    assert store.get("session") == "s1"
    clock.advance(5.0)  # original TTL=10.0 should now expire
    assert store.get("session") is None


def test_08_nested_tx_inner_rollback_preserves_outer() -> None:
    store = MiniKVStore()
    store.set("base", 0)

    store.begin()
    store.set("a", 1)
    store.begin()
    store.set("a", 2)
    store.set("b", 3)
    store.rollback()  # revert inner only

    assert store.get("a") == 1
    assert store.get("b") is None

    store.commit()  # commit outer
    assert store.get("base") == 0
    assert store.get("a") == 1
    assert store.get("b") is None
    with pytest.raises(RuntimeError):
        store.commit()  # stack should now be empty


def test_09_nested_tx_inner_commit_then_outer_rollback() -> None:
    store = MiniKVStore()
    store.set("k", "orig")

    store.begin()
    store.set("k", "outer")
    store.begin()
    store.set("k", "inner")
    store.set("new_key", 99)
    store.commit()  # merge into outer

    assert store.get("k") == "inner"
    assert store.get("new_key") == 99

    store.rollback()  # revert outer (including merged inner changes)
    assert store.get("k") == "orig"
    assert store.get("new_key") is None


def test_10_transaction_ttl_extension_rollback() -> None:
    clock = FakeClock(100.0)
    store = MiniKVStore(clock=clock)
    store.set("lease", "v1", ttl=5.0)

    store.begin()
    store.set("lease", "v2", ttl=50.0)
    assert store.get("lease") == "v2"
    store.rollback()

    assert store.get("lease") == "v1"
    clock.advance(5.0)
    assert store.get("lease") is None
