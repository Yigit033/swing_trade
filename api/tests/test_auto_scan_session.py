"""Daily-run skip is per completed NYSE session, not calendar weekday."""

from datetime import date, datetime, timezone

from api.auto_scan import run_covers_auto_session, session_already_auto_scanned


FRIDAY = date(2026, 8, 28)


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
