"""Daily-run skip is per completed NYSE session, not calendar weekday."""

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from api import auto_scan
from api.auto_scan import auto_entry_capacity, run_covers_auto_session, session_already_auto_scanned


FRIDAY = date(2026, 8, 28)


def test_live_auto_entry_capacity_matches_backtest_regime_policy():
    policy = SimpleNamespace(bear_block_new_entries=True, caution_max_concurrent=1)
    assert auto_entry_capacity("BEAR", policy, active_count=0) == 0
    assert auto_entry_capacity("CAUTION", policy, active_count=0) == 1
    assert auto_entry_capacity("CAUTION", policy, active_count=1) == 0
    assert auto_entry_capacity("BULL", policy, active_count=9) is None


def test_auto_run_with_bar_session_covers():
    stats = {"scan_source": "auto", "bar_session": "2026-08-28", "reason": "no_qualifying"}
    assert run_covers_auto_session(stats, "2026-08-29T01:00:00Z", FRIDAY)


def test_manual_run_does_not_cover_even_same_session():
    stats = {"bar_session": "2026-08-28", "reason": "no_qualifying"}
    assert not run_covers_auto_session(stats, "2026-08-28T21:00:00Z", FRIDAY)


def test_aborted_auto_run_does_not_cover():
    stats = {"scan_source": "auto", "bar_session": "2026-08-28", "reason": "stale_data"}
    assert not run_covers_auto_session(stats, "2026-08-28T21:00:00Z", FRIDAY)


def test_auto_run_created_at_fallback_after_close():
    stats = {"scan_source": "auto", "reason": "success"}
    # Friday 20:50 ET = Saturday 00:50 UTC (EDT)
    assert run_covers_auto_session(stats, "2026-08-29T00:50:00Z", FRIDAY)


def test_auto_run_created_at_before_close_is_previous_session():
    stats = {"scan_source": "auto", "reason": "success"}
    # Friday 01:29 ET = Friday 05:29 UTC — still Thursday's window
    assert not run_covers_auto_session(stats, "2026-08-28T05:29:00Z", FRIDAY)
    assert run_covers_auto_session(stats, "2026-08-28T05:29:00Z", date(2026, 8, 27))


def test_session_already_auto_scanned_reads_storage(monkeypatch):
    rows = [
        {
            "id": 108,
            "created_at": "2026-08-28T21:00:00Z",
            "stats": {"bar_session": "2026-08-28", "reason": "no_qualifying"},
        },
        {
            "id": 109,
            "created_at": "2026-08-29T01:10:00Z",
            "stats": {
                "scan_source": "auto",
                "bar_session": "2026-08-28",
                "reason": "no_qualifying",
            },
        },
    ]

    class _Fake:
        def list_recent_stats(self, limit=24):
            assert limit >= 2
            return rows

    import api.deps as deps

    monkeypatch.setattr(deps, "get_signal_history_storage", lambda: _Fake())
    assert session_already_auto_scanned(FRIDAY) is True


def test_session_already_auto_scanned_ignores_manual_only(monkeypatch):
    rows = [
        {
            "id": 108,
            "created_at": datetime(2026, 8, 28, 21, 0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
            "stats": {"bar_session": "2026-08-28", "reason": "no_qualifying"},
        },
    ]

    class _Fake:
        def list_recent_stats(self, limit=24):
            return rows

    import api.deps as deps

    monkeypatch.setattr(deps, "get_signal_history_storage", lambda: _Fake())
    assert session_already_auto_scanned(FRIDAY) is False


def test_saturday_maintenance_still_scans_friday(monkeypatch):
    """GitHub Cuma cron'u Cumartesi'ye kayınca taramayı atlama."""
    import api.auto_scan as auto_scan

    auto_scan._last_run_session = None
    monkeypatch.setattr(auto_scan, "last_completed_session", lambda now=None: FRIDAY)
    monkeypatch.setattr(auto_scan, "session_already_auto_scanned", lambda s, lookback=24: False)
    monkeypatch.setattr(auto_scan, "_run_auto_scan_once", lambda: "ran")

    class _Tracker:
        def confirm_pending_trades(self, _data):
            return []

        def update_all_open_trades(self, _data):
            return []

    class _Auto:
        enabled = True

    class _Settings:
        auto_scan = _Auto()

    monkeypatch.setattr("api.deps.get_paper_tracker", lambda: _Tracker())
    monkeypatch.setattr(
        "swing_trader.small_cap.settings_config.load_settings",
        lambda: _Settings(),
    )

    out = auto_scan.run_daily_maintenance()
    assert out["scan"] == "ran"
    assert out["bar_session"] == "2026-08-28"
    assert auto_scan._last_run_session == FRIDAY


def test_active_notional_counts_open_pending_and_partial(tmp_path):
    from swing_trader.paper_trading.storage import PaperTradeStorage
    st = PaperTradeStorage(str(tmp_path / "t.db"))
    base = dict(entry_date="2026-10-01", stop_loss=9.0, target=12.0, swing_type="A")
    st.add_trade(dict(base, ticker="OPN", status="OPEN", entry_price=10.0, position_size=100))
    st.add_trade(dict(base, ticker="PND", status="PENDING", entry_price=20.0, position_size=50))
    prt = st.add_trade(dict(base, ticker="PRT", status="OPEN", entry_price=10.0, position_size=300))
    st.update_trade(prt, {"partial_exit_price": 11.0, "partial_exit_pct": 33.0})   # T1 kısmi satış
    st.add_trade(dict(base, ticker="CLS", status="STOPPED", entry_price=10.0, position_size=999))
    st.add_trade(dict(base, ticker="USR", status="OPEN", entry_price=10.0, position_size=999), user_id="u1")
    got = st.active_notional_for_owner(None)
    assert got == pytest.approx(1000 + 1000 + 3000 * 0.67)     # kapanan ve kullanıcınınki sayılmaz


def test_auto_scan_never_exceeds_cash_account(monkeypatch):
    """Midas gibi nakit hesap: açık+bekleyen+yeni ≤ portfolio_value; sıradaki küçük sinyal sığarsa girer."""
    from unittest.mock import Mock
    from swing_trader.small_cap.settings_config import SmallCapSettings
    import api.routers.scanner as scanner
    import api.scanner_jobs as jobs
    import api.deps as deps
    s = SmallCapSettings()
    s.auto_scan.strategy = "LEGACY"                 # bu test eski kuralların nakit sınırını ölçer
    monkeypatch.setattr("swing_trader.small_cap.settings_config.load_settings", lambda *a, **k: s)
    monkeypatch.setattr(jobs, "create_exclusive_scan_job", lambda: "job")
    monkeypatch.setattr(jobs, "release_scan_slot", lambda j: None)
    sigs = [dict(ticker="BIG", entry_price=50.0, position_size=100),     # 5.000$
            dict(ticker="TOO", entry_price=40.0, position_size=100),     # 4.000$ → sığmaz
            dict(ticker="FIT", entry_price=10.0, position_size=100)]     # 1.000$ → sığar
    monkeypatch.setattr(scanner, "_execute_smallcap_scan",
                        lambda *a, **k: dict(signals=sigs, market_regime="BULL", stats={}))
    tracker = Mock()
    tracker.storage.active_notional_for_owner.return_value = 3500.0      # zaten bağlı
    tracker.add_trade_from_signal.side_effect = lambda sig, owner: 1
    monkeypatch.setattr(deps, "get_paper_tracker", lambda: tracker)
    assert auto_scan._run_auto_scan_once() == "ran"
    opened = [c.args[0]["ticker"] for c in tracker.add_trade_from_signal.call_args_list]
    assert opened == ["BIG", "FIT"]                                      # 3.5K + 5K + 1K ≤ 10K



@pytest.mark.parametrize("regime,expected", [("BULL", ["BIG", "MID"]), ("CAUTION", ["BIG"]), ("BEAR", [])])
def test_auto_scan_h6s_entries(monkeypatch, regime, expected):
    """H6S ana strateji: VCE isabetleri → IWM rejimi → %10 pozisyon, nakit sınırı; eski kurallar ayrıca açar."""
    from unittest.mock import Mock
    from swing_trader.small_cap.settings_config import SmallCapSettings
    from swing_trader.shadow import runner as shadow_runner, sources as shadow_src
    import api.routers.scanner as scanner
    import api.scanner_jobs as jobs
    import api.deps as deps
    s = SmallCapSettings()
    monkeypatch.setattr("swing_trader.small_cap.settings_config.load_settings", lambda *a, **k: s)
    monkeypatch.setattr(jobs, "create_exclusive_scan_job", lambda: "job")
    monkeypatch.setattr(jobs, "release_scan_slot", lambda j: None)
    legacy = [dict(ticker="OLD", entry_price=10.0, position_size=10)]
    monkeypatch.setattr(scanner, "_execute_smallcap_scan",
                        lambda *a, **k: dict(signals=legacy, market_regime="BULL", stats={}))
    hits = [dict(ticker="MID", dv20=5e7, close=20.0, atr14=1.0),
            dict(ticker="BIG", dv20=9e7, close=50.0, atr14=2.0),
            dict(ticker="TOO", dv20=1e7, close=40.0, atr14=1.0)]
    monkeypatch.setattr(shadow_src, "live_vce_hits", lambda asof: dict(hits=hits, stats={}))
    monkeypatch.setattr(shadow_runner, "_small_cap_regime", lambda src, asof: {"regime": regime})
    tracker = Mock()
    tracker.storage.count_active_trades_for_owner.return_value = 0
    held = {"H6S": 8000.0, "LEGACY": 0.0}                    # H6S hesabında 8.000$ zaten bağlı
    tracker.storage.active_notional_for_owner.side_effect = lambda owner, strategy=None: held[strategy]
    tracker.storage.get_closed_trades.return_value = []                    # kayıp serisi yok → tam boyut
    tracker.add_trade_from_signal.side_effect = lambda sig, owner: 1
    monkeypatch.setattr(deps, "get_paper_tracker", lambda: tracker)
    assert auto_scan._run_auto_scan_once() == "ran"
    calls = [c.args[0] for c in tracker.add_trade_from_signal.call_args_list]
    h6s = [c for c in calls if c.get("strategy") == "H6S"]
    assert [c["ticker"] for c in h6s] == expected             # dolar hacim sırası; TOO nakde sığmaz
    for c in h6s:
        assert c["position_size"] == int(1000 / c["entry_price"]) and c["atr14"] > 0
    assert [c["ticker"] for c in calls if c.get("strategy") != "H6S"] == ["OLD"]   # kontrol grubu


def test_loss_streak_brake_rule():
    from swing_trader.paper_trading.risk_rules import loss_streak_active
    L = lambda d, p, st="TIMEOUT": dict(status=st, exit_date=d, realized_pnl_pct=p)
    assert loss_streak_active([L("2026-11-03", -2.0), L("2026-11-05", 0.3), L("2026-11-04", -1.0)])  # +0.3 brüt = zarar
    assert not loss_streak_active([L("2026-11-03", -2.0), L("2026-11-04", -1.0), L("2026-11-05", 4.0)])
    assert not loss_streak_active([L("2026-11-03", -2.0), L("2026-11-04", -1.0), L("2026-11-05", 9.0, "REJECTED")])
    assert not loss_streak_active([L("2026-11-03", -2.0), L("2026-11-04", -1.0)])           # 3 kapanış yok


def test_auto_scan_h6s_halves_size_after_loss_streak(monkeypatch):
    from unittest.mock import Mock
    from swing_trader.small_cap.settings_config import SmallCapSettings
    from swing_trader.shadow import runner as shadow_runner, sources as shadow_src
    import api.routers.scanner as scanner
    import api.scanner_jobs as jobs
    import api.deps as deps
    s = SmallCapSettings()
    s.auto_scan.legacy_control = False
    monkeypatch.setattr("swing_trader.small_cap.settings_config.load_settings", lambda *a, **k: s)
    monkeypatch.setattr(jobs, "create_exclusive_scan_job", lambda: "job")
    monkeypatch.setattr(jobs, "release_scan_slot", lambda j: None)
    monkeypatch.setattr(scanner, "_execute_smallcap_scan", lambda *a, **k: dict(signals=[], market_regime="BULL", stats={}))
    monkeypatch.setattr(shadow_src, "live_vce_hits",
                        lambda asof: dict(hits=[dict(ticker="BIG", dv20=9e7, close=50.0, atr14=2.0)], stats={}))
    monkeypatch.setattr(shadow_runner, "_small_cap_regime", lambda src, asof: {"regime": "BULL"})
    tracker = Mock()
    tracker.storage.count_active_trades_for_owner.return_value = 0
    tracker.storage.active_notional_for_owner.return_value = 0.0
    tracker.storage.get_closed_trades.return_value = [
        dict(strategy="H6S", user_id=None, status="STOPPED", exit_date=f"2026-11-1{i}", realized_pnl_pct=-3.0)
        for i in range(3)]
    tracker.add_trade_from_signal.side_effect = lambda sig, owner: 1
    monkeypatch.setattr(deps, "get_paper_tracker", lambda: tracker)
    auto_scan._run_auto_scan_once()
    sig = tracker.add_trade_from_signal.call_args.args[0]
    assert sig["position_size"] == int(0.05 * 10000 / 50.0) == 10          # yarım boyut



def test_auto_scan_h6s_brake_fails_safe_to_half_size(monkeypatch):
    """Kapanan işlemler okunamazsa girişler atlanmaz; temkinli yarım boyut."""
    from unittest.mock import Mock
    from swing_trader.small_cap.settings_config import SmallCapSettings
    from swing_trader.shadow import runner as shadow_runner, sources as shadow_src
    import api.routers.scanner as scanner
    import api.scanner_jobs as jobs
    import api.deps as deps
    s = SmallCapSettings()
    s.auto_scan.legacy_control = False
    monkeypatch.setattr("swing_trader.small_cap.settings_config.load_settings", lambda *a, **k: s)
    monkeypatch.setattr(jobs, "create_exclusive_scan_job", lambda: "job")
    monkeypatch.setattr(jobs, "release_scan_slot", lambda j: None)
    monkeypatch.setattr(scanner, "_execute_smallcap_scan", lambda *a, **k: dict(signals=[], market_regime="BULL", stats={}))
    monkeypatch.setattr(shadow_src, "live_vce_hits",
                        lambda asof: dict(hits=[dict(ticker="BIG", dv20=9e7, close=50.0, atr14=2.0)], stats={}))
    monkeypatch.setattr(shadow_runner, "_small_cap_regime", lambda src, asof: {"regime": "BULL"})
    tracker = Mock()
    tracker.storage.count_active_trades_for_owner.return_value = 0
    tracker.storage.active_notional_for_owner.return_value = 0.0
    tracker.storage.get_closed_trades.side_effect = RuntimeError("db down")
    tracker.add_trade_from_signal.side_effect = lambda sig, owner: 1
    monkeypatch.setattr(deps, "get_paper_tracker", lambda: tracker)
    auto_scan._run_auto_scan_once()
    assert tracker.add_trade_from_signal.call_args.args[0]["position_size"] == 10
