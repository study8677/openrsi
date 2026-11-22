"""Basic public smoke tests for MiniKVStore."""

from kvstore import MiniKVStore


def test_basic_set_get_delete() -> None:
    store = MiniKVStore()
    store.set("name", "openrsi")
    assert store.get("name") == "openrsi"
    assert store.delete("name") is True
    assert store.get("name") is None


def test_basic_transaction_commit() -> None:
    store = MiniKVStore()
    store.begin()
    store.set("x", 42)
    store.commit()
    assert store.get("x") == 42
