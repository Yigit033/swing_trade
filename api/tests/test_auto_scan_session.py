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
    s.auto_scan.h6s_position_pct = 0.10   # sabit: bu testler kuralı ölçer, varsayılanı değil
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


def test_loss_streak_brake_rule():
    from swing_trader.paper_trading.risk_rules import loss_streak_active
    L = lambda d, p, st="TIMEOUT": dict(status=st, exit_date=d, realized_pnl_pct=p)
    assert loss_streak_active([L("2026-11-03", -2.0), L("2026-11-05", 0.3), L("2026-11-04", -1.0)])  # +0.3 brüt = zarar
    assert not loss_streak_active([L("2026-11-03", -2.0), L("2026-11-04", -1.0), L("2026-11-05", 4.0)])
    assert not loss_streak_active([L("2026-11-03", -2.0), L("2026-11-04", -1.0), L("2026-11-05", 9.0, "REJECTED")])
    assert not loss_streak_active([L("2026-11-03", -2.0), L("2026-11-04", -1.0)])           # 3 kapanış yok


def _grw_env(monkeypatch, hits, regime="BULL", growth=None, closed=None, held=0.0, legacy=None, legacy_control=False):
    """GRW otomatik girişinin sahte ortamı: kırılım isabetleri, SPY rejimi, SEC büyüme cevabı, sanal hesap."""
    from unittest.mock import Mock
    from swing_trader.small_cap.settings_config import SmallCapSettings
    from swing_trader.shadow import runner as shadow_runner, sources as shadow_src
    import api.routers.scanner as scanner
    import api.scanner_jobs as jobs
    import api.deps as deps
    s = SmallCapSettings()
    s.auto_scan.legacy_control = legacy_control
    assert s.auto_scan.strategy == "GRW"                       # ana kural (2026-11-28)
    monkeypatch.setattr("swing_trader.small_cap.settings_config.load_settings", lambda *a, **k: s)
    monkeypatch.setattr(jobs, "create_exclusive_scan_job", lambda: "job")
    monkeypatch.setattr(jobs, "release_scan_slot", lambda j: None)
    monkeypatch.setattr(scanner, "_execute_smallcap_scan",
                        lambda *a, **k: dict(signals=legacy or [], market_regime="BULL", stats={}))
    monkeypatch.setattr(shadow_src, "grw_hits", lambda asof: dict(hits=hits, stats=dict(candidates=len(hits))))
    monkeypatch.setattr(shadow_src, "fetch_bars", lambda *a, **k: {})
    monkeypatch.setattr(shadow_runner, "_regime_on", lambda bars, index, day: {"regime": regime})
    asked = []

    def growth_check(ticker, day):
        asked.append(ticker)
        g = (growth or {}).get(ticker, dict(known=True, ok=True))
        if isinstance(g, Exception):
            raise g
        return g
    monkeypatch.setattr(shadow_src, "growth_check", growth_check)
    tracker = Mock()
    tracker.storage.check_duplicate.return_value = False
    tracker.storage.active_notional_for_owner.side_effect = (
        lambda owner, strategy=None: held if strategy == "GRW" else 0.0)
    if isinstance(closed, Exception):
        tracker.storage.get_closed_trades.side_effect = closed
    else:
        tracker.storage.get_closed_trades.return_value = closed or []
    tracker.add_trade_from_signal.side_effect = lambda sig, owner: 1
    monkeypatch.setattr(deps, "get_paper_tracker", lambda: tracker)
    return tracker, asked


def _grw_calls(tracker):
    return [c.args[0] for c in tracker.add_trade_from_signal.call_args_list if c.args[0].get("strategy") == "GRW"]


GRW_HITS = [dict(ticker="LOW", cap=2e9, close=20.0, low=19.5, dv20=5e7, key=0.2),      # dip %2.5 aşağıda
            dict(ticker="TOP", cap=9e9, close=50.0, low=44.0, dv20=9e7, key=1.5),      # dip %12 aşağıda → stop %8
            dict(ticker="BAD", cap=3e9, close=40.0, low=39.0, dv20=1e7, key=0.9)]


def test_auto_scan_grw_entries_ranked_sized_by_risk_and_growth_filtered(monkeypatch):
    """GRW ana kural: 12 aylık getiri sırası, büyüme şartı, boyut = min(%25, %1 risk ÷ stop uzaklığı)."""
    legacy = [dict(ticker="OLD", entry_price=10.0, position_size=10)]
    tracker, asked = _grw_env(monkeypatch, GRW_HITS, growth={"BAD": dict(known=True, ok=False)},
                              legacy=legacy, legacy_control=True)
    assert auto_scan._run_auto_scan_once() == "ran"
    calls = _grw_calls(tracker)
    assert [c["ticker"] for c in calls] == ["TOP", "LOW"] and asked == ["TOP", "BAD", "LOW"]
    top, low = calls
    assert top["position_size"] == int(0.01 * 10000 / 0.08 / 50.0) == 25      # stop %8 → pozisyon %12.5
    assert low["position_size"] == int(0.25 * 10000 / 20.0) == 125            # stop %2.5 → %40 olurdu, tavan %25
    assert (top["signal_low"], top["entry_price"], top["regime"]) == (44.0, 50.0, "BULL")
    others = [c.args[0]["ticker"] for c in tracker.add_trade_from_signal.call_args_list
              if c.args[0].get("strategy") != "GRW"]
    assert others == ["OLD"]                                                   # kontrol grubu ayrıca açar


@pytest.mark.parametrize("regime", ["BEAR", "CAUTION", "UNKNOWN"])
def test_auto_scan_grw_buys_only_in_bull_and_skips_sec(monkeypatch, regime):
    tracker, asked = _grw_env(monkeypatch, GRW_HITS, regime=regime)
    auto_scan._run_auto_scan_once()
    assert _grw_calls(tracker) == [] and asked == []                           # alım yok, SEC'e gidilmez


def test_auto_scan_grw_cash_limit_and_busy_ticker(monkeypatch):
    tracker, asked = _grw_env(monkeypatch, GRW_HITS, held=8000.0)              # hesapta 8.000$ zaten bağlı
    tracker.storage.check_duplicate.side_effect = lambda t, user_id=None, strategy=None: t == "BAD"
    auto_scan._run_auto_scan_once()
    assert [c["ticker"] for c in _grw_calls(tracker)] == ["TOP"]               # 8.000 + 1.250 sığar; LOW (2.500) sığmaz
    assert asked == ["TOP", "LOW"]                                             # elde olan hisse için SEC'e gidilmez


def test_auto_scan_grw_halves_risk_after_two_losses(monkeypatch):
    losers = [dict(strategy="GRW", user_id=None, status="STOPPED", exit_date=f"2026-11-1{i}", realized_pnl_pct=-3.0,
                   entry_price=10.0, position_size=10, realized_pnl=-3.0) for i in range(2)]
    tracker, _ = _grw_env(monkeypatch, GRW_HITS[1:2], closed=losers)
    auto_scan._run_auto_scan_once()
    equity = 10_000 + 2 * (-3.0 - 100 * 0.20 / 100 - 2 * 1.5)                  # iki zarar + spread + emir ücretleri
    assert _grw_calls(tracker)[0]["position_size"] == int(0.005 * equity / 0.08 / 50.0) == 12


def test_auto_scan_grw_fails_safe_to_half_risk_when_history_unreadable(monkeypatch):
    tracker, _ = _grw_env(monkeypatch, GRW_HITS[1:2], closed=RuntimeError("db down"))
    auto_scan._run_auto_scan_once()
    assert _grw_calls(tracker)[0]["position_size"] == int(0.005 * 10000 / 0.08 / 50.0) == 12


def test_auto_scan_grw_defers_when_sec_is_unreachable(monkeypatch):
    """SEC'e ulaşılamazsa "büyüme verisi yok" denmez; kalan adaylar sonraki koşuya bırakılır."""
    tracker, asked = _grw_env(monkeypatch, GRW_HITS, growth={"BAD": RuntimeError("sec 503")})
    auto_scan._run_auto_scan_once()
    assert [c["ticker"] for c in _grw_calls(tracker)] == ["TOP"] and asked == ["TOP", "BAD"]
