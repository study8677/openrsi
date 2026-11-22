"""Hidden verification suite for t1_rate_limiter (8 constraints)."""

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
