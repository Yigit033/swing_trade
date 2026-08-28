"""Universe vs signal forward-return cohort — membership snapshot + tracker isolation."""

from __future__ import annotations

from pathlib import Path

import pytest

from swing_trader.data.forward_returns import (
    KIND_SIGNAL,
    KIND_UNIVERSE,
    assemble_scan_membership,
)


def test_assemble_scan_membership_keeps_reject_reason_and_no_fake_q():
    members = assemble_scan_membership(
        universe_tickers=["AAA", "BBB", "CCC", "DDD"],
        signals=[{"ticker": "AAA", "date": "2026-08-28", "quality_score": 82, "trigger_pathway": "vce_breakout"}],
        outcomes=[
            {"ticker": "BBB", "reject_reason": "no_trigger", "date": "2026-08-28"},
            {"ticker": "CCC", "reject_reason": "quality_type_a", "date": "2026-08-28", "quality": 61.0},
        ],
        fallback_date="2026-08-28",
    )
    by = {m["ticker"]: m for m in members}
    assert by["AAA"]["kind"] == KIND_SIGNAL
    assert by["AAA"]["quality"] == 82
    assert by["AAA"]["reject_reason"] is None
    assert by["BBB"]["kind"] == KIND_UNIVERSE
    assert by["BBB"]["reject_reason"] == "no_trigger"
    assert by["BBB"]["quality"] is None
    assert by["CCC"]["quality"] == 61.0
    assert by["DDD"]["reject_reason"] == "unknown"


@pytest.fixture()
def isolated_tracker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import swing_trader.data.forward_returns as fr

    monkeypatch.setattr(fr, "DATABASE_URL", None)
    monkeypatch.setattr(fr, "_MODE", "sqlite")
    monkeypatch.setattr(fr, "DB_PATH", tmp_path / "fwd.db")
    monkeypatch.setattr(fr, "_tracker", None)
    return fr.get_forward_tracker()


def test_universe_rows_do_not_enter_quality_buckets(isolated_tracker):
    isolated_tracker.record_signals(
        "job-1",
        [{"ticker": "SIG", "date": "2026-08-01", "quality_score": 85, "trigger_pathway": "vce_breakout"}],
    )
    isolated_tracker.record_universe(
        "job-1",
        [{"ticker": "UNI", "date": "2026-08-01", "kind": "universe", "reject_reason": "no_trigger"}],
        regime="BULL",
    )
    # Fill r10 without yfinance — write directly
    import sqlite3
    import swing_trader.data.forward_returns as fr

    conn = sqlite3.connect(str(fr.DB_PATH))
    conn.execute(
        "UPDATE signal_forward_returns SET r10 = 12.0, status = 'complete' WHERE ticker = 'SIG'"
    )
    conn.execute(
        "UPDATE signal_forward_returns SET r10 = 40.0, status = 'complete' WHERE ticker = 'UNI'"
    )
    conn.commit()
    conn.close()

    stats = isolated_tracker.get_stats()
    assert stats["n_tracked"] == 1
    assert stats["universe"]["n_tracked"] == 1
    q80 = next(b for b in stats["quality_buckets"] if b["label"] == "Q80-100")
    assert q80["n"] == 1
    assert q80["mean"] == 12.0
    assert stats["cohort_split"]["signal"]["mean"] == 12.0
    assert stats["cohort_split"]["universe"]["mean"] == 40.0
    assert stats["signals"][0]["ticker"] == "SIG"
    assert stats["universe_rows"][0]["ticker"] == "UNI"
    assert stats["universe_rows"][0]["reject_reason"] == "no_trigger"


def test_signal_wins_unique_ticker_date(isolated_tracker):
    isolated_tracker.record_signals(
        "j",
        [{"ticker": "FOO", "date": "2026-08-10", "quality_score": 90}],
    )
    n = isolated_tracker.record_universe(
        "j",
        [{"ticker": "FOO", "date": "2026-08-10", "kind": KIND_UNIVERSE, "reject_reason": "no_trigger"}],
    )
    assert n == 0
    stats = isolated_tracker.get_stats()
    assert stats["n_tracked"] == 1
    assert stats["universe"]["n_tracked"] == 0
