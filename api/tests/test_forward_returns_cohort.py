"""Universe vs signal forward-return cohort — membership snapshot + tracker isolation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from swing_trader.data.forward_returns import (
    KIND_SIGNAL,
    KIND_UNIVERSE,
    _compute_forward_values,
    assemble_scan_membership,
    build_edge_report,
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


def _edge_row(ticker, date, quality, r10, *, regime="BULL", kind="signal"):
    return {
        "ticker": ticker,
        "signal_date": date,
        "quality": quality,
        "r10": r10,
        "regime": regime,
        "kind": kind,
        "effective_min_quality": None,
        "is_actionable": None,
    }


def test_policy_split_uses_regime_threshold_not_fixed_q80():
    rows = [
        _edge_row("BULL78", "2026-01-02", 78, 5.0, regime="BULL"),
        _edge_row("CAUTION78", "2026-01-03", 78, -2.0, regime="CAUTION"),
        _edge_row("CAUTION80", "2026-01-04", 80, 3.0, regime="CAUTION"),
    ]
    report = build_edge_report(rows)
    assert report["policy_split"]["eligible"]["n"] == 2
    assert report["policy_split"]["eligible"]["mean"] == 4.0
    assert report["policy_split"]["excluded"]["n"] == 1


def test_persisted_scan_threshold_wins_over_current_fallback():
    row = _edge_row("STRICT", "2026-01-02", 80, 4.0)
    row["effective_min_quality"] = 82
    row["is_actionable"] = 0
    report = build_edge_report([row])
    assert report["policy_split"]["eligible"]["n"] == 0
    assert report["policy_split"]["excluded"]["n"] == 1
    assert report["policy_split"]["legacy_inferred_rows"] == 0


def test_matched_edge_uses_same_date_controls_and_equal_weights_dates():
    rows = [
        _edge_row("S1", "2026-01-02", 80, 10.0),
        _edge_row("S2", "2026-01-02", 81, 6.0),
        _edge_row("S3", "2026-01-03", 82, 2.0),
        _edge_row("U1", "2026-01-02", None, 2.0, kind="universe"),
        _edge_row("U2", "2026-01-02", None, 4.0, kind="universe"),
        _edge_row("U3", "2026-01-03", None, -2.0, kind="universe"),
        _edge_row("U4", "2026-02-01", None, 100.0, kind="universe"),
    ]
    matched = build_edge_report(rows)["matched_edge"]
    assert matched["matched_dates"] == 2
    assert matched["matched_signals"] == 3
    assert matched["mean"] == 4.5


def test_forward_values_reject_truncated_history_instead_of_shifting_entry():
    df = pd.DataFrame({
        "Date": pd.date_range("2026-02-01", periods=12, freq="B"),
        "Open": [100.0] * 12,
        "High": [101.0] * 12,
        "Low": [99.0] * 12,
        "Close": [100.0] * 12,
    })
    assert _compute_forward_values(df, "2025-12-01") is None


def test_forward_values_enter_next_session_open():
    df = pd.DataFrame({
        "Date": pd.date_range("2026-01-05", periods=12, freq="B"),
        "Open": [10.0, 20.0] + [20.0] * 10,
        "High": [10.0, 22.0] + [22.0] * 10,
        "Low": [10.0, 18.0] + [18.0] * 10,
        "Close": [10.0, 21.0] + [22.0] * 10,
    })
    values = _compute_forward_values(df, "2026-01-05")
    assert values is not None
    assert values["entry_open"] == 20.0
    assert values["r3"] == 10.0
    assert values["r10"] == 10.0
