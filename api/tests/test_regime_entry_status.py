"""Ana sayfadaki "alım açık mı" satırı — canlı auto-scan frenleriyle aynı karar."""
from types import SimpleNamespace

from api.routers.regime import entry_status

POLICY = SimpleNamespace(bear_block_new_entries=True, caution_max_concurrent=2)


def test_grw_buys_only_when_spy_regime_is_bull():
    assert entry_status("GRW", "BULL", POLICY) == dict(strategy="GRW", index="SPY", regime="BULL", status="OPEN")
    for regime in ("BEAR", "CAUTION", "UNKNOWN", None):
        assert entry_status("GRW", regime, POLICY)["status"] == "BLOCKED"


def test_legacy_follows_market_regime_and_policy():
    assert entry_status("LEGACY", "BEAR", POLICY)["status"] == "BLOCKED"
    out = entry_status("LEGACY", "CAUTION", POLICY)
    assert (out["status"], out["max_open"], out["index"]) == ("LIMITED", 2, "SPY")
    assert entry_status("LEGACY", "BULL", POLICY)["status"] == "OPEN"
    off = SimpleNamespace(bear_block_new_entries=False, caution_max_concurrent=2)
    assert entry_status("LEGACY", "BEAR", off)["status"] == "OPEN"


def test_retired_h6s_setting_is_read_as_grw():
    from swing_trader.small_cap.settings_models_extra import AutoScanSettings
    assert AutoScanSettings(strategy="H6S").strategy == "GRW"
    assert AutoScanSettings().strategy == "GRW" and AutoScanSettings(strategy="LEGACY").strategy == "LEGACY"
