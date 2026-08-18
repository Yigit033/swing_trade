"""UTC ISO coercion for last-price-update timestamps."""
from swing_trader.paper_trading.storage import _as_utc_iso, _utc_now_iso


def test_naive_iso_is_treated_as_utc():
    assert _as_utc_iso("2026-08-18T12:02:00.123456") == "2026-08-18T12:02:00Z"


def test_zulu_passthrough():
    assert _as_utc_iso("2026-08-18T12:02:00Z") == "2026-08-18T12:02:00Z"


def test_offset_passthrough():
    assert _as_utc_iso("2026-08-18T15:02:00+03:00") == "2026-08-18T15:02:00+03:00"


def test_none_and_empty():
    assert _as_utc_iso(None) is None
    assert _as_utc_iso("") is None


def test_utc_now_has_z():
    s = _utc_now_iso()
    assert s.endswith("Z")
    assert "T" in s
