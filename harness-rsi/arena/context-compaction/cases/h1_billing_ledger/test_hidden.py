"""Hidden verification suite for h1_billing_ledger (8 constraints)."""

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
