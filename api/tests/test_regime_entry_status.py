"""Ana sayfadaki "alım açık mı" satırı — canlı auto-scan frenleriyle aynı karar."""
from types import SimpleNamespace

from api.routers.regime import entry_status

POLICY = SimpleNamespace(bear_block_new_entries=True, caution_max_concurrent=2)


def test_h6s_follows_small_cap_regime_not_spy():
    assert entry_status("H6S", "BEAR", "BULL", POLICY)["status"] == "BLOCKED"
    assert entry_status("H6S", "BULL", "BEAR", POLICY)["status"] == "OPEN"
    out = entry_status("H6S", "CAUTION", "BULL", POLICY)
    assert (out["status"], out["max_open"], out["index"]) == ("LIMITED", 1, "IWM")


def test_h6s_unknown_when_small_cap_regime_missing():
    assert entry_status("H6S", None, "BEAR", POLICY)["status"] == "UNKNOWN"
    assert entry_status("H6S", "UNKNOWN", "BEAR", POLICY)["status"] == "OPEN"   # auto-scan yalnız BEAR'de keser


def test_legacy_follows_market_regime_and_policy():
    assert entry_status("LEGACY", "BULL", "BEAR", POLICY)["status"] == "BLOCKED"
    out = entry_status("LEGACY", None, "CAUTION", POLICY)
    assert (out["status"], out["max_open"], out["index"]) == ("LIMITED", 2, "SPY")
    assert entry_status("LEGACY", None, "BULL", POLICY)["status"] == "OPEN"
    off = SimpleNamespace(bear_block_new_entries=False, caution_max_concurrent=2)
    assert entry_status("LEGACY", None, "BEAR", off)["status"] == "OPEN"
