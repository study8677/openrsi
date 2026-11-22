"""Hidden verification suite for t2_config_migrator (8 constraints)."""

import copy
import pytest
from migrator import migrate_config


def test_01_schema_v3_and_nested_server() -> None:
    out = migrate_config({"host": "0.0.0.0", "port": 9090})
    assert out["schema_version"] == 3
    assert out["server"] == {"host": "0.0.0.0", "port": 9090}


def test_02_defaults_applied() -> None:
    out = migrate_config({})
    assert out["schema_version"] == 3
    assert out["server"] == {"host": "127.0.0.1", "port": 8000}
    assert out["timeout_sec"] == 5.0
    assert out["debug"] is False
    assert out["tags"] == []
    assert out["extra"] == {}


def test_03_timeout_ms_converted_to_float_seconds() -> None:
    out = migrate_config({"timeout_ms": 2500})
    assert isinstance(out["timeout_sec"], float)
    assert out["timeout_sec"] == 2.5
    assert "timeout_ms" not in out


def test_04_port_coercion_and_range_validation() -> None:
    out = migrate_config({"port": "8080"})
    assert out["server"]["port"] == 8080
    with pytest.raises(ValueError):
        migrate_config({"port": 0})
    with pytest.raises(ValueError):
        migrate_config({"port": 70000})


def test_05_tags_normalization() -> None:
    out1 = migrate_config({"tags": " prod, api , , prod, beta "})
    assert out1["tags"] == ["api", "beta", "prod"]
    out2 = migrate_config({"tags": [" z ", "a", "a", ""]})
    assert out2["tags"] == ["a", "z"]


def test_06_unknown_keys_preserved_in_extra() -> None:
    out = migrate_config({"host": "localhost", "region": "us-east", "retries": 3})
    assert out["extra"] == {"region": "us-east", "retries": 3}
    assert "region" not in out


def test_07_debug_string_coercion_and_no_input_mutation() -> None:
    raw = {"debug": "false", "tags": ["b", "a"], "custom": {"nested": 1}}
    snap = copy.deepcopy(raw)
    out = migrate_config(raw)
    assert raw == snap
    assert out["debug"] is False
    assert migrate_config({"debug": "0"})["debug"] is False
    assert migrate_config({"debug": "true"})["debug"] is True


def test_08_idempotent_on_v3_and_deep_copy() -> None:
    v3 = {
        "schema_version": 3,
        "server": {"host": "10.0.0.1", "port": 443},
        "timeout_sec": 1.5,
        "debug": False,
        "tags": ["edge"],
        "extra": {"nested": [1, 2]},
    }
    out = migrate_config(v3)
    assert out == v3
    out["extra"]["nested"].append(3)
    assert v3["extra"]["nested"] == [1, 2]
    bad_v3 = copy.deepcopy(v3)
    bad_v3["server"]["port"] = 99999
    with pytest.raises(ValueError):
        migrate_config(bad_v3)
