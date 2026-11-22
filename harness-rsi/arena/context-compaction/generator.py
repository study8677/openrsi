"""Deterministic benchmark generator for `arena/context-compaction`.

Creates 2 Train cases (`t1_rate_limiter`, `t2_config_migrator`) and
1 Held-out case (`h1_billing_ledger`). Each case contains:
  - A long (~10k-14k token) multi-turn trajectory with:
      * Early global invariants (turns 1-3)
      * Requirement superseding (mid-turn override of earlier spec)
      * Negative memory (failed approach / pitfall to avoid)
      * Buried critical specs inside noisy tool logs
      * Heavy trailing noise (so naive sliding window loses early constraints)
  - A minimal starter file in `starter_files/`
  - A `final_task.md` prompt
  - A deterministic 8-test hidden `test_hidden.py` suite
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")
CASES_DIR = Path(__file__).parent / "cases"


def _tok(text: str) -> int:
    return len(_ENC.encode(text))


def _noisy_pytest_log(module_name: str, count: int = 55, buried_note: str = "") -> str:
    lines = [
        "============================= test session starts ==============================",
        "platform linux -- Python 3.12.11, pytest-8.4.1, pluggy-1.6.0",
        "rootdir: /workspace",
        f"collected {count} items",
        "",
    ]
    for i in range(1, count + 1):
        lines.append(
            f"tests/legacy/test_{module_name}_compat_{i:03d}.py::test_case_{i:03d} PASSED "
            f"[{(i * 100) // count:3d}%] (duration=0.01{(i % 9)}s, worker=gw{i % 4})"
        )
        if i == count // 2 and buried_note:
            lines.append(f"WARNING [spec-check]: {buried_note}")
    lines.append(f"============================== {count} passed in 1.42s ==============================")
    return "\n".join(lines)


def _noisy_file_dump(filename: str, lines_count: int = 90, buried_line: str = "") -> str:
    out = [f"$ cat {filename}"]
    for i in range(1, lines_count + 1):
        if i == lines_count // 2 and buried_line:
            out.append(f"{i:04d} | {buried_line}")
        else:
            out.append(
                f"{i:04d} | # auto-generated boilerplate diagnostic record id=rec_{i:04d} "
                f"checksum=9f8e7d6c5b4a{i:04d} status=OK latency_ms={12 + (i % 17)}"
            )
    return "\n".join(out)


def _noisy_trailing_turns(start_turn: int, project_name: str) -> list[dict[str, Any]]:
    """Trailing noisy turns (~2800-3500 tokens) that flush out naive sliding windows."""
    return [
        {
            "turn": start_turn,
            "role": "user",
            "content": f"Before writing the final implementation for `{project_name}`, can you check the repo linter logs and docs?",
        },
        {
            "turn": start_turn + 1,
            "role": "tool",
            "content": _noisy_file_dump(f"logs/{project_name}_lint_audit.log", lines_count=85),
        },
        {
            "turn": start_turn + 2,
            "role": "assistant",
            "content": "Lint audit log looks clean — all legacy modules pass formatting and type checks with zero warnings.",
        },
        {
            "turn": start_turn + 3,
            "role": "user",
            "content": "Also dump the CI artifact manifest so we know the runner environment is healthy.",
        },
        {
            "turn": start_turn + 4,
            "role": "tool",
            "content": _noisy_file_dump(f"ci/{project_name}_manifest.txt", lines_count=85),
        },
        {
            "turn": start_turn + 5,
            "role": "assistant",
            "content": "CI runner manifest verified: Python 3.12, pytest 8.4.1, clean environment. Ready whenever you want me to write the final module.",
        },
    ]


# ---------------------------------------------------------------------------
# Case 1 (Train): t1_rate_limiter
# ---------------------------------------------------------------------------

def _build_t1_rate_limiter() -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {
            "turn": 1,
            "role": "user",
            "content": (
                "We are building a pure-Python rate limiter module at `rate_limiter.py`.\n"
                "Hard global constraints from day 1:\n"
                "1. Pure Python standard library only (no third-party imports).\n"
                "2. Expose class `RateLimiter(limit: int, window_sec: float, clock=None)` where `clock` defaults to `time.monotonic`.\n"
                "3. Method `allow(client_id: str) -> dict` must return a dict with exact keys: "
                "`allowed` (bool), `remaining` (int), `retry_after` (float), `error_code` (str | None)."
            ),
        },
        {
            "turn": 2,
            "role": "assistant",
            "content": "Understood. I will keep it stdlib-only and expose `RateLimiter` with `allow(client_id)` returning those 4 keys.",
        },
        {
            "turn": 3,
            "role": "user",
            "content": (
                "Initial algorithm idea: let's use a simple fixed-window counter per client_id. "
                "When rejected, set `error_code` to `'RATE_LIMITED'`."
            ),
        },
        {
            "turn": 4,
            "role": "tool",
            "content": _noisy_pytest_log(
                "gateway",
                count=60,
                buried_note=(
                    "Client SDK v3 requires `retry_after` to be 0.0 when allowed=True, "
                    "and rounded to 3 decimal places when allowed=False."
                ),
            ),
        },
        {
            "turn": 5,
            "role": "assistant",
            "content": "Noted the gateway log and SDK v3 requirement on `retry_after` formatting.",
        },
        {
            "turn": 6,
            "role": "user",
            "content": (
                "Wait—we tried a fixed-window prototype last sprint and it failed load testing: "
                "users burst 2x the limit across a window boundary (e.g. at t=59.9s and t=60.1s). "
                "Do NOT use fixed-window buckets! Also never mutate or sleep on the system clock."
            ),
        },
        {
            "turn": 7,
            "role": "assistant",
            "content": "Good catch. We will avoid fixed-window buckets and never call `time.sleep`.",
        },
        {
            "turn": 8,
            "role": "tool",
            "content": _noisy_file_dump(
                "docs/api_errors.txt",
                lines_count=75,
                buried_line=(
                    "OFFICIAL_ERROR_CODE_OVERRIDE: Rejections MUST use error_code='ERR_QUOTA_EXCEEDED' "
                    "(the old 'RATE_LIMITED' string is deprecated and breaks billing alerts)."
                ),
            ),
        },
        {
            "turn": 9,
            "role": "user",
            "content": (
                "Confirmed! Official requirement update (supersedes Turn 3):\n"
                "- Switch algorithm to **sliding window log** (exact timestamp window `[now - window_sec, now)` — "
                "an event at `t` expires once `now - t >= window_sec`, inclusive boundary!).\n"
                "- Rejected calls MUST return `error_code='ERR_QUOTA_EXCEEDED'` (NOT `'RATE_LIMITED'`), "
                "and `error_code=None` when allowed.\n"
                "- Also add `reset(client_id: str | None = None) -> None`: if `client_id` is given, clear only that client; "
                "if `None`, clear all clients."
            ),
        },
        {
            "turn": 10,
            "role": "assistant",
            "content": (
                "Got it: sliding window log with inclusive expiration `now - t >= window_sec`, "
                "`error_code='ERR_QUOTA_EXCEEDED'` on rejection, and `reset(client_id=None)`."
            ),
        },
        {
            "turn": 11,
            "role": "user",
            "content": (
                "One more edge-case contract: if `limit <= 0` or `window_sec <= 0`, `__init__` must raise `ValueError`. "
                "And when a request is rejected, do NOT record its timestamp in the window log (otherwise retries extend the lockout!)."
            ),
        },
        {
            "turn": 12,
            "role": "tool",
            "content": _noisy_pytest_log("metrics", count=55),
        },
        {
            "turn": 13,
            "role": "assistant",
            "content": "Recorded both edge-case rules: validate `limit > 0` and `window_sec > 0`, and never log rejected timestamps.",
        },
    ]
    messages.extend(_noisy_trailing_turns(14, "rate_limiter"))

    starter = '''"""Rate limiter module — implement according to the agreed conversation specs."""

from __future__ import annotations


class RateLimiter:
    pass
'''

    final_task = (
        "Using ONLY the compacted conversation context provided above, implement the complete "
        "`rate_limiter.py` in the current workspace directory so it satisfies every active constraint, "
        "override, and edge case discussed in the conversation."
    )

    hidden_tests = '''"""Hidden verification suite for t1_rate_limiter (8 constraints)."""

import pytest
from rate_limiter import RateLimiter


class FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def test_01_init_validation() -> None:
    with pytest.raises(ValueError):
        RateLimiter(limit=0, window_sec=10.0)
    with pytest.raises(ValueError):
        RateLimiter(limit=5, window_sec=0.0)
    with pytest.raises(ValueError):
        RateLimiter(limit=-1, window_sec=5.0)


def test_02_return_schema_when_allowed() -> None:
    clock = FakeClock(10.0)
    rl = RateLimiter(limit=2, window_sec=10.0, clock=clock)
    res = rl.allow("alice")
    assert set(res.keys()) == {"allowed", "remaining", "retry_after", "error_code"}
    assert res["allowed"] is True
    assert res["remaining"] == 1
    assert res["retry_after"] == 0.0
    assert res["error_code"] is None


def test_03_superseded_error_code_on_rejection() -> None:
    clock = FakeClock(10.0)
    rl = RateLimiter(limit=1, window_sec=10.0, clock=clock)
    assert rl.allow("alice")["allowed"] is True
    rej = rl.allow("alice")
    assert rej["allowed"] is False
    assert rej["remaining"] == 0
    assert rej["error_code"] == "ERR_QUOTA_EXCEEDED"


def test_04_sliding_window_prevents_boundary_burst() -> None:
    clock = FakeClock(0.0)
    rl = RateLimiter(limit=2, window_sec=10.0, clock=clock)
    clock.advance(9.0)  # t=9.0
    assert rl.allow("u1")["allowed"] is True
    clock.advance(0.5)  # t=9.5
    assert rl.allow("u1")["allowed"] is True
    clock.advance(1.0)  # t=10.5 (fixed window would wrongly reset at 10.0!)
    assert rl.allow("u1")["allowed"] is False


def test_05_inclusive_expiration_boundary() -> None:
    clock = FakeClock(50.0)
    rl = RateLimiter(limit=1, window_sec=5.0, clock=clock)
    assert rl.allow("u1")["allowed"] is True
    clock.advance(4.999)
    assert rl.allow("u1")["allowed"] is False
    clock.advance(0.001)  # now - t == 5.0 == window_sec -> expired!
    assert rl.allow("u1")["allowed"] is True


def test_06_retry_after_rounding_and_no_rejected_recording() -> None:
    clock = FakeClock(100.0)
    rl = RateLimiter(limit=1, window_sec=10.0, clock=clock)
    assert rl.allow("u1")["allowed"] is True
    clock.advance(3.33333)
    rej = rl.allow("u1")
    assert rej["allowed"] is False
    assert rej["retry_after"] == round(10.0 - 3.33333, 3)
    # Rejected call must NOT be recorded; at t=110.0 client must be allowed again
    clock.now = 110.0
    assert rl.allow("u1")["allowed"] is True


def test_07_per_client_isolation() -> None:
    clock = FakeClock(0.0)
    rl = RateLimiter(limit=1, window_sec=10.0, clock=clock)
    assert rl.allow("a")["allowed"] is True
    assert rl.allow("a")["allowed"] is False
    assert rl.allow("b")["allowed"] is True


def test_08_reset_single_and_all() -> None:
    clock = FakeClock(0.0)
    rl = RateLimiter(limit=1, window_sec=10.0, clock=clock)
    rl.allow("a")
    rl.allow("b")
    rl.reset("a")
    assert rl.allow("a")["allowed"] is True
    assert rl.allow("b")["allowed"] is False
    rl.reset()
    assert rl.allow("b")["allowed"] is True
'''

    return {
        "id": "t1_rate_limiter",
        "split": "train",
        "target_file": "rate_limiter.py",
        "budget_ratio": 0.18,
        "messages": messages,
        "starter_files": {"rate_limiter.py": starter},
        "final_task": final_task,
        "hidden_tests": hidden_tests,
    }


# ---------------------------------------------------------------------------
# Case 2 (Train): t2_config_migrator
# ---------------------------------------------------------------------------

def _build_t2_config_migrator() -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {
            "turn": 1,
            "role": "user",
            "content": (
                "We need a pure-Python config migration module at `migrator.py`.\n"
                "Hard constraints:\n"
                "1. Pure stdlib only.\n"
                "2. Expose function `migrate_config(raw: dict) -> dict`.\n"
                "3. Never mutate the input dict `raw` in place (must return a deep-copied new dict)."
            ),
        },
        {
            "turn": 2,
            "role": "assistant",
            "content": "Understood: stdlib-only `migrate_config(raw: dict) -> dict` with zero in-place mutation.",
        },
        {
            "turn": 3,
            "role": "user",
            "content": (
                "In v1 configs, fields are flat: `host`, `port`, `timeout_ms`, `debug`, `tags`.\n"
                "Initially we planned to output `schema_version: 2` and rename `timeout_ms` to `timeout` (keeping milliseconds)."
            ),
        },
        {
            "turn": 4,
            "role": "tool",
            "content": _noisy_file_dump(
                "logs/prod_schema_errors.log",
                lines_count=80,
                buried_line=(
                    "INCIDENT-402: Downstream parser crashes if `server.port` is a string; "
                    "migrator MUST coerce numeric string ports like '8080' to int 8080, "
                    "and raise ValueError if port is out of range 1..65535."
                ),
            ),
        },
        {
            "turn": 5,
            "role": "assistant",
            "content": "Captured INCIDENT-402: coerce numeric string ports to int and validate 1..65535.",
        },
        {
            "turn": 6,
            "role": "user",
            "content": (
                "Update (supersedes Turn 3 on timeout and schema_version):\n"
                "- Output `schema_version` MUST be `3` (NOT `2`).\n"
                "- Move `host` and `port` under nested dict `server: {'host': ..., 'port': ...}` (defaults: host='127.0.0.1', port=8000).\n"
                "- Convert `timeout_ms` (default 5000) into `timeout_sec` as a `float` in SECONDS (e.g., 2500 -> 2.5), and drop `timeout_ms`."
            ),
        },
        {
            "turn": 7,
            "role": "assistant",
            "content": "Updated: `schema_version=3`, nested `server`, and `timeout_sec` in float seconds.",
        },
        {
            "turn": 8,
            "role": "tool",
            "content": _noisy_pytest_log(
                "migrator",
                count=55,
                buried_note=(
                    "PITFALL: Do NOT drop unknown extra top-level keys! Preserve any unknown keys "
                    "under a nested dict `extra: {...}` (empty dict `{}` if none)."
                ),
            ),
        },
        {
            "turn": 9,
            "role": "user",
            "content": (
                "Yes, good catch from that warning! Also normalize `tags`: "
                "if `tags` is a comma-separated string (e.g. `' api, prod ,api '`), split by comma, strip whitespace, "
                "drop empty strings, deduplicate, and sort alphabetically. If `tags` is a list, strip/dedup/sort it the same way. "
                "If missing, default to `[]`."
            ),
        },
        {
            "turn": 10,
            "role": "assistant",
            "content": "Got it: normalize `tags` from str or list into sorted unique stripped list, and preserve unknown keys under `extra`.",
        },
        {
            "turn": 11,
            "role": "user",
            "content": (
                "Final idempotency rule: if `raw` ALREADY has `schema_version == 3`, return a deep copy of `raw` unchanged "
                "(still validating `server.port` is in 1..65535). "
                "And `debug` should always be coerced to bool (default False; string `'false'` / `'0'` must become `False`, NOT `True`!)."
            ),
        },
        {
            "turn": 12,
            "role": "tool",
            "content": _noisy_file_dump("docs/changelog_old.md", lines_count=70),
        },
    ]
    messages.extend(_noisy_trailing_turns(13, "config_migrator"))

    starter = '''"""Config migrator module — implement according to the agreed conversation specs."""

from __future__ import annotations
from typing import Any


def migrate_config(raw: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError
'''

    final_task = (
        "Using ONLY the compacted conversation context provided above, implement the complete "
        "`migrator.py` in the current workspace directory so it satisfies every active constraint, "
        "override, and edge case discussed in the conversation."
    )

    hidden_tests = '''"""Hidden verification suite for t2_config_migrator (8 constraints)."""

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
'''

    return {
        "id": "t2_config_migrator",
        "split": "train",
        "target_file": "migrator.py",
        "budget_ratio": 0.18,
        "messages": messages,
        "starter_files": {"migrator.py": starter},
        "final_task": final_task,
        "hidden_tests": hidden_tests,
    }


# ---------------------------------------------------------------------------
# Case 3 (Held-out): h1_billing_ledger
# ---------------------------------------------------------------------------

def _build_h1_billing_ledger() -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {
            "turn": 1,
            "role": "user",
            "content": (
                "We are implementing a financial billing ledger at `ledger.py`.\n"
                "Hard global constraints:\n"
                "1. Pure stdlib only; MUST use `decimal.Decimal` internally (never binary floats for money math!).\n"
                "2. Expose class `BillingLedger(tax_rate: str = '0.10')`.\n"
                "3. Methods: `add_item(sku: str, unit_price: str, qty: int) -> None`, "
                "`apply_discount(percent: str) -> None`, `summary() -> dict`."
            ),
        },
        {
            "turn": 2,
            "role": "assistant",
            "content": "Understood: `Decimal`-based `BillingLedger` with `add_item`, `apply_discount`, and `summary`.",
        },
        {
            "turn": 3,
            "role": "user",
            "content": (
                "Initial thought: apply tax first, then apply discount on the taxed total. "
                "Default tax_rate is 10% (`'0.10'`)."
            ),
        },
        {
            "turn": 4,
            "role": "tool",
            "content": _noisy_file_dump(
                "audit/finance_compliance.txt",
                lines_count=80,
                buried_line=(
                    "AUDIT-RULE-19: All monetary values in `summary()` MUST be quantized to 2 decimal places "
                    "using `ROUND_HALF_EVEN` (banker's rounding) and returned as strings (e.g. `'12.34'`), "
                    "with exact keys: `subtotal`, `discount`, `tax`, `total`."
                ),
            ),
        },
        {
            "turn": 5,
            "role": "assistant",
            "content": "Noted AUDIT-RULE-19: banker's rounding (`ROUND_HALF_EVEN`) to 2 decimal strings with those 4 exact keys.",
        },
        {
            "turn": 6,
            "role": "user",
            "content": (
                "CRITICAL finance override (supersedes Turn 3!):\n"
                "- Tax MUST be calculated AFTER discount (i.e. `taxable = subtotal - discount`, `tax = taxable * tax_rate`, "
                "`total = taxable + tax`). Computing tax before discount caused an audit failure last quarter!\n"
                "- Default `tax_rate` is changed to `'0.085'` (8.5%, NOT 10%)."
            ),
        },
        {
            "turn": 7,
            "role": "assistant",
            "content": "Updated: discount applies before tax (`taxable = subtotal - discount`), and default `tax_rate='0.085'`.",
        },
        {
            "turn": 8,
            "role": "tool",
            "content": _noisy_pytest_log(
                "billing",
                count=55,
                buried_note=(
                    "BUG-HISTORY: Calling `apply_discount` multiple times must REPLACE the active discount percent, "
                    "NOT stack/add them. Percent must be in [0, 100]; otherwise raise ValueError."
                ),
            ),
        },
        {
            "turn": 9,
            "role": "user",
            "content": (
                "Also add validation & void support:\n"
                "- `add_item`: `qty` must be `int > 0` and `Decimal(unit_price) >= 0`, else raise `ValueError`.\n"
                "- Add `void_sku(sku: str) -> int`: removes all line items matching `sku` and returns the count of removed items."
            ),
        },
        {
            "turn": 10,
            "role": "assistant",
            "content": "Got it: input validation on `add_item`/`apply_discount`, replace-not-stack discount, and `void_sku(sku) -> int`.",
        },
    ]
    messages.extend(_noisy_trailing_turns(11, "billing_ledger"))

    starter = '''"""Billing ledger module — implement according to the agreed conversation specs."""

from __future__ import annotations


class BillingLedger:
    pass
'''

    final_task = (
        "Using ONLY the compacted conversation context provided above, implement the complete "
        "`ledger.py` in the current workspace directory so it satisfies every active constraint, "
        "override, and edge case discussed in the conversation."
    )

    hidden_tests = '''"""Hidden verification suite for h1_billing_ledger (8 constraints)."""

import pytest
from ledger import BillingLedger


def test_01_default_tax_rate_override_and_schema() -> None:
    led = BillingLedger()
    led.add_item("A1", "100.00", 1)
    s = led.summary()
    assert set(s.keys()) == {"subtotal", "discount", "tax", "total"}
    assert s["subtotal"] == "100.00"
    assert s["discount"] == "0.00"
    assert s["tax"] == "8.50"
    assert s["total"] == "108.50"


def test_02_discount_applied_before_tax() -> None:
    led = BillingLedger(tax_rate="0.10")
    led.add_item("PRO", "100.00", 2)  # subtotal 200.00
    led.apply_discount("25")          # discount 50.00 -> taxable 150.00 -> tax 15.00 -> total 165.00
    s = led.summary()
    assert s["subtotal"] == "200.00"
    assert s["discount"] == "50.00"
    assert s["tax"] == "15.00"
    assert s["total"] == "165.00"


def test_03_bankers_rounding_half_even() -> None:
    led = BillingLedger(tax_rate="0.05")
    # 0.10 * 0.05 = 0.005 -> ROUND_HALF_EVEN -> 0.00 (whereas ROUND_HALF_UP -> 0.01)
    led.add_item("PENNY", "0.10", 1)
    assert led.summary()["tax"] == "0.00"
    # 0.30 * 0.05 = 0.015 -> ROUND_HALF_EVEN -> 0.02
    led2 = BillingLedger(tax_rate="0.05")
    led2.add_item("PENNY", "0.30", 1)
    assert led2.summary()["tax"] == "0.02"


def test_04_decimal_precision_no_float_drift() -> None:
    led = BillingLedger(tax_rate="0.00")
    led.add_item("X", "0.10", 1)
    led.add_item("Y", "0.20", 1)
    assert led.summary()["subtotal"] == "0.30"
    assert led.summary()["total"] == "0.30"


def test_05_apply_discount_replaces_not_stacks() -> None:
    led = BillingLedger(tax_rate="0.00")
    led.add_item("A", "100.00", 1)
    led.apply_discount("10")
    led.apply_discount("20")
    s = led.summary()
    assert s["discount"] == "20.00"
    assert s["total"] == "80.00"


def test_06_validation_errors() -> None:
    led = BillingLedger()
    with pytest.raises(ValueError):
        led.add_item("A", "10.00", 0)
    with pytest.raises(ValueError):
        led.add_item("A", "-1.00", 1)
    with pytest.raises(ValueError):
        led.apply_discount("-5")
    with pytest.raises(ValueError):
        led.apply_discount("101")


def test_07_void_sku_removes_matching_items() -> None:
    led = BillingLedger(tax_rate="0.00")
    led.add_item("A", "10.00", 1)
    led.add_item("B", "25.00", 2)
    led.add_item("A", "5.00", 3)
    removed = led.void_sku("A")
    assert removed == 2
    assert led.void_sku("NOPE") == 0
    assert led.summary()["subtotal"] == "50.00"


def test_08_empty_ledger_summary() -> None:
    led = BillingLedger()
    assert led.summary() == {
        "subtotal": "0.00",
        "discount": "0.00",
        "tax": "0.00",
        "total": "0.00",
    }
'''

    return {
        "id": "h1_billing_ledger",
        "split": "held_out",
        "target_file": "ledger.py",
        "budget_ratio": 0.18,
        "messages": messages,
        "starter_files": {"ledger.py": starter},
        "final_task": final_task,
        "hidden_tests": hidden_tests,
    }


def generate_all(output_dir: Path = CASES_DIR) -> list[dict[str, Any]]:
    """Materialize all benchmark cases onto disk and return metadata summary."""
    builders = [
        _build_t1_rate_limiter,
        _build_t2_config_migrator,
        _build_h1_billing_ledger,
    ]
    summaries: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for build_fn in builders:
        case = build_fn()
        case_dir = output_dir / case["id"]
        starter_dir = case_dir / "starter_files"
        starter_dir.mkdir(parents=True, exist_ok=True)

        raw_text = "\n".join(
            f"[Turn {m['turn']} | {m['role'].upper()}]\n{m['content']}\n"
            for m in case["messages"]
        )
        raw_tokens = _tok(raw_text)
        max_tokens = max(800, int(raw_tokens * case["budget_ratio"]))

        meta = {
            "id": case["id"],
            "split": case["split"],
            "target_file": case["target_file"],
            "raw_tokens": raw_tokens,
            "max_tokens": max_tokens,
            "budget_ratio": case["budget_ratio"],
            "num_turns": len(case["messages"]),
        }
        (case_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (case_dir / "trajectory.json").write_text(
            json.dumps(case["messages"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (case_dir / "final_task.md").write_text(case["final_task"], encoding="utf-8")
        (case_dir / "test_hidden.py").write_text(case["hidden_tests"], encoding="utf-8")

        for fname, fcontent in case["starter_files"].items():
            (starter_dir / fname).write_text(fcontent, encoding="utf-8")

        summaries.append(meta)

    return summaries


if __name__ == "__main__":
    for s in generate_all():
        print(
            f"{s['id']:22s} split={s['split']:8s} turns={s['num_turns']:2d} "
            f"raw_tokens={s['raw_tokens']:5d} max_tokens={s['max_tokens']:4d}"
        )
