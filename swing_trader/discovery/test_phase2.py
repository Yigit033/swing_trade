"""Deterministic, network-free ingestion, parity, timing, replay and outcomes."""
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from .calendar import is_session, last_completed, session_close, sessions, next_session
from .bars import (normalize, rows_frame, prepare_one, prepare, store_object,
                   read_object, pilot_records, object_id, MODE)
from .shadow import evaluate, scan, combine
from .opportunity import measure, register
from .master import build_snapshot
from .test_phase1 import payloads
from swing_trader.small_cap.settings_config import SmallCapSettings
from swing_trader.small_cap.signals import SmallCapSignals

SESSION = date(2026, 10, 9)
ASOF = "2026-10-10T20:00:00+00:00"


def frame(end=SESSION, n=130, seed=1):
    idx = pd.to_datetime(sessions(date(2026, 1, 2), end)[-n:])
    rng = np.random.default_rng(seed)
    close = 30 + np.arange(len(idx))*.025 + rng.normal(0, .35, len(idx))
    return pd.DataFrame({"Open": close-.1, "High": close+.5, "Low": close-.5,
                         "Close": close, "Volume": 1e6, "Adj Close": close,
                         "Dividends": 0., "Stock Splits": 0.}, index=idx)


def record(symbol="AAA"):
    return {"symbol": symbol, "exchange": "NASDAQ", "security_type": "COMMON_STOCK",
            "discovery_status": "CANDIDATE", "discovery_reason": "equity", "yfinance_symbol_candidate": symbol}


def obj(df=None, symbol="AAA"):
    rows, validation = normalize(frame() if df is None else df, SESSION)
    return {"symbol": symbol, "vendor_symbol": symbol, "mode": MODE, "rows": rows,
            "validation": validation, "session_requested": SESSION.isoformat(),
            "identity": {"symbol": symbol, "exchangeName": "NMS", "instrumentType": "EQUITY", "currency": "USD"},
            "fetched_at": "2026-10-10T19:00:00+00:00"}


@pytest.mark.parametrize("stamp,expected", [
    ("2026-10-09T19:59:00+00:00", "2026-10-08"),
    ("2026-10-09T20:29:00+00:00", "2026-10-08"),
    ("2026-10-09T20:30:00+00:00", "2026-10-09"),
    ("2026-11-27T18:29:00+00:00", "2026-11-25"),
    ("2026-11-27T18:30:00+00:00", "2026-11-27"),
    ("2026-12-24T18:30:00+00:00", "2026-12-24"),
    ("2026-03-09T20:30:00+00:00", "2026-03-09"),
])
def test_completed_and_early_close(stamp, expected):
    assert last_completed(datetime.fromisoformat(stamp)).isoformat() == expected


def test_calendar_special_closures_and_range():
    assert not is_session(date(2025, 1, 9))
    assert is_session(date(2027, 12, 31))
    assert next_session(date(2026, 10, 9)) == date(2026, 10, 12)
    with pytest.raises(ValueError, match="RANGE"):
        is_session(date(2029, 1, 2))
    with pytest.raises(ValueError, match="AWARE"):
        last_completed(datetime(2026, 10, 9))


def test_adjustment_leaves_volume_unchanged():
    df = frame()
    df["Adj Close"] *= .9
    rows, _ = normalize(df, SESSION)
    adjusted = rows_frame(rows)
    np.testing.assert_allclose(adjusted.Close, df.Close*.9)
    np.testing.assert_allclose(adjusted.Volume, df.Volume)
    from yfinance.utils import auto_adjust
    np.testing.assert_allclose(adjusted[["Open", "High", "Low", "Close", "Volume"]],
                               auto_adjust(df)[["Open", "High", "Low", "Close", "Volume"]])


@pytest.mark.parametrize("mutation,reason", [
    (lambda f: pd.concat([f, f.tail(1)]), "DUPLICATE"),
    (lambda f: f.iloc[::-1], "UNSORTED"),
    (lambda f: f.assign(High=0), "NONPOSITIVE"),
    (lambda f: f.assign(Volume=-1), "NEGATIVE"),
    (lambda f: f.assign(Close=np.nan), "NONFINITE"),
    (lambda f: f.assign(Low=f.High+1), "ENVELOPE"),
    (lambda f: f.drop(columns="Adj Close"), "MISSING"),
])
def test_reject_corrupt_data(mutation, reason):
    with pytest.raises(ValueError, match=reason):
        normalize(mutation(frame()), SESSION)


def test_future_rows_removed_and_missing_reported():
    df = frame(end=date(2026, 10, 12))
    rows, validation = normalize(df, SESSION)
    assert rows[-1]["Date"] == "2026-10-09"
    assert validation["future_or_partial_rows_removed"] == 1
    _, validation = normalize(frame().drop(frame().index[-3]), SESSION)
    assert validation["missing_sessions"] == ["2026-10-07"]


class Provider:
    def __init__(self, df=None, error=None):
        self.df, self.error, self.calls = df, error, []

    def history(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        if self.error:
            raise self.error
        return self.df.loc[pd.Timestamp(start):pd.Timestamp(end)], {"symbol": symbol}


def test_initial_cache_hit_incremental_and_rebase(tmp_path):
    df = frame()
    provider = Provider(df)
    first = prepare_one(record(), SESSION, tmp_path, provider)
    assert first["status"] == "OK"
    assert read_object(tmp_path, first["object_id"])["rows"][-1]["Date"] == str(SESSION)
    hit = prepare_one(record(), SESSION, tmp_path, provider, first)
    assert hit["status"] == "CACHE_HIT" and len(provider.calls) == 1
    later = date(2026, 10, 12)
    more = df.copy()
    more.loc[pd.Timestamp(later)] = df.iloc[-1]
    provider.df = more
    second = prepare_one(record(), later, tmp_path, provider, first)
    assert second["status"] == "OK"
    assert provider.calls[-1][1] == date(2026, 10, 8)
    revised = more.copy()
    revised["Adj Close"] *= .99
    revised.loc[pd.Timestamp(later), "Dividends"] = .1
    provider.df = revised
    rebuilt = prepare_one(record(), later, tmp_path, provider, first)
    assert rebuilt["revision_rebuild"]
    assert rebuilt["attempts"] == 2
    assert read_object(tmp_path, first["object_id"])["rows"] == normalize(df, SESSION)[0]


def test_rate_limit_stops_without_retry_and_sanitizes(tmp_path):
    provider = Provider(error=RuntimeError("429 token=SECRET https://private"))
    result = prepare_one(record(), SESSION, tmp_path, provider, sleeper=lambda _: None)
    assert result["status"] == "RATE_LIMIT_STOP" and result["attempts"] == 1
    assert "SECRET" not in json.dumps(result)


def test_retry_and_circuit_checkpoint_resume(tmp_path):
    provider = Provider(error=RuntimeError("403 denied"))
    records = [record("AAA"), record("BBB")]
    first = prepare(records, SESSION, tmp_path, tmp_path / "run", provider)
    assert first["circuit"] == "ACCESS_DENIED_STOP"
    assert first["outcomes"][1]["status"] == "NOT_ATTEMPTED_CIRCUIT_OPEN"
    assert prepare(records, SESSION, tmp_path, tmp_path / "run", provider) == first
    assert len(provider.calls) == 1
    with pytest.raises(ValueError, match="PLAN_MISMATCH"):
        prepare(records[::-1], SESSION, tmp_path, tmp_path / "run", provider)


def test_object_integrity(tmp_path):
    key = store_object(tmp_path, {"x": 1})
    (tmp_path / "objects" / f"{key}.json").write_text('{"x":2}')
    with pytest.raises(ValueError, match="CHECKSUM"):
        read_object(tmp_path, key)
    with pytest.raises(ValueError, match="OBJECT_ID"):
        read_object(tmp_path, "../../secrets")


def test_pilot_diverse_reproducible_and_not_capped():
    snapshot = {"records": [{**record(f"AAA{i}"), "exchange": "NYSE" if i % 2 else "NASDAQ"} for i in range(510)]}
    a, meta = pilot_records(snapshot, 240)
    assert a == pilot_records(snapshot, 240)[0]
    assert len(a) == 240 and len({r["exchange"] for r in a}) == 2
    assert len(pilot_records(snapshot, 510)[0]) == 510


@pytest.mark.parametrize("seed", range(8))
def test_trigger_parity_and_unknown_not_pass(seed):
    df = frame(seed=seed)
    df.loc[df.index[-1], "Volume"] = (1.5+seed*.5)*1e6
    settings = SmallCapSettings()
    out = evaluate(record(), obj(df), SESSION, settings, {"regime": "BULL", "confidence": "CONFIRMED"}, decision_at=ASOF)
    from dateutil.relativedelta import relativedelta
    cut = df.loc[df.index >= pd.Timestamp(SESSION+timedelta(days=1)-relativedelta(months=6))]
    sig = SmallCapSignals(settings=settings)
    expected, details = sig.check_all_triggers(cut)
    assert out["trigger_positive"] == expected
    assert (out["stages"]["vce"]["status"] == "PASS") == sig.check_vce_breakout(cut)[0]
    assert (out["stages"]["rvol"]["status"] == "PASS") == sig.check_rvol_thrust(cut)[0]
    assert out["stages"]["market_cap"]["status"] == "UNKNOWN"
    assert out["stages"]["earnings"]["status"] == "UNKNOWN"
    assert out["final_status"] != "PASS"


def test_unknown_regime_does_not_relax_quality_and_future_facts_excluded():
    facts = {"observed_at": "2026-10-11T20:00:00+00:00", "session": str(SESSION), "market_cap": 1e9,
             "earnings_coverage": "VERIFIED", "earnings_dates": []}
    result = evaluate(record(), obj(), SESSION, SmallCapSettings(), {"regime": "UNKNOWN"}, facts, ASOF)
    assert result["stages"]["market_cap"]["status"] == "UNKNOWN"
    assert result["stages"]["quality"]["status"] == "UNKNOWN"
    assert combine([{"status": "PASS"}, {"status": "UNKNOWN"}]) == "UNKNOWN"


def test_deterministic_scan_cutoff_and_legacy_incomplete(tmp_path):
    snapshot = build_snapshot(payloads(), "2026-10-10T18:00:00+00:00")
    key = store_object(tmp_path, obj())
    prep = {"session": str(SESSION), "outcomes": [{"symbol": "AAA", "object_id": key, "status": "OK"}]}
    settings = SmallCapSettings()
    first = scan(snapshot, prep, tmp_path, settings, ASOF)
    assert first == scan(snapshot, prep, tmp_path, settings, ASOF)
    assert first["comparison"]["broad_trigger_only"] is None
    assert first["metrics"]["master_candidates"] == 3
    with pytest.raises(ValueError, match="BEFORE_EOD"):
        scan(snapshot, prep, tmp_path, settings, "2026-10-09T19:00:00+00:00")


def observation():
    return {"observation_id": "x", "signal_date": "2026-10-02", "ticker": "AAA", "independent_observation": True,
            "cohort": "HISTORICAL_RESEARCH"}


def test_forward_pending_next_open_costs_and_no_gap_filling():
    df = frame()
    r = measure(observation(), obj(df), SESSION)["horizons"]
    assert r["5"]["status"] == "MATURE" and r["10"]["status"] == "PENDING"
    assert r["5"]["entry_session"] == "2026-10-05"
    expected = 100*(df.loc["2026-10-09", "Close"]/df.loc["2026-10-05", "Open"]-1)
    assert r["5"]["forward_return_pct"] == pytest.approx(expected)
    assert r["5"]["net_return_pct_by_roundtrip_bps"]["100"] < r["5"]["net_return_pct_by_roundtrip_bps"]["20"] < expected
    broken = obj(df.drop(pd.Timestamp("2026-10-05")))
    assert measure(observation(), broken, SESSION)["horizons"]["5"]["status"] == "MISSING_DATA"


def test_overlap_not_counted_independent():
    row = {"trigger_positive": True, "ticker": "AAA", "setup": "rvol_thrust", "legacy_membership": "UNKNOWN",
           "reference_close": 20., "atr_percent": .03, "volume": 1e6, "avg_dollar_volume_20": 2e7,
           "quality_score": 80., "rank_score": 80., "stages": {}, "final_status": "UNKNOWN"}
    scan_obj = {"scan_id": "1", "inputs": {"session": "2026-10-02", "decision_at": "2026-10-02T22:00:00+00:00"}, "rows": [row]}
    first = register(scan_obj, "HISTORICAL_RESEARCH")
    assert register(scan_obj, "HISTORICAL_RESEARCH", first)["observations"] == first["observations"]
    scan_obj["inputs"]["session"] = "2026-10-05"
    second = register(scan_obj, "HISTORICAL_RESEARCH", first)
    assert len(second["observations"]) == 2
    assert not second["observations"][1]["independent_observation"]
    assert not second["observations"][0]["broad_only_confirmed"]
    with pytest.raises(ValueError, match="AFTER_REFERENCE_ENTRY"):
        register(scan_obj, "PROSPECTIVE_SHADOW")


@pytest.mark.parametrize("seed", [5, 7, 11, 16, 18, "vce"])
def test_engine_score_type_rank_and_eligibility_parity(seed):
    from swing_trader.small_cap.engine import SmallCapEngine
    from swing_trader.small_cap.filters import SmallCapFilters
    from swing_trader.small_cap.scoring import SmallCapScoring
    from dateutil.relativedelta import relativedelta
    settings = SmallCapSettings()
    df = frame(seed=5 if seed == "vce" else seed)
    df.loc[df.index[-1], "Volume"] = 3e6
    if seed == "vce":
        ix = df.index[-15:-1]
        df.loc[ix, "Open"] = df.loc[ix, "Close"]-.01
        df.loc[ix, "High"] = df.loc[ix, "Close"]+.03
        df.loc[ix, "Low"] = df.loc[ix, "Close"]-.03
        df.loc[df.index[-1], ["Open", "High", "Low", "Close", "Adj Close"]] = [35, 35.7, 34.9, 35.6, 35.6]
    facts = {"observed_at": ASOF, "session": str(SESSION), "float_shares": 1e6, "market_cap": 1e9,
             "earnings_coverage": "VERIFIED", "earnings_dates": []}
    shadow = evaluate(record(), obj(df), SESSION, settings, {"regime": "BULL", "confidence": "CONFIRMED"}, facts, ASOF)
    engine = SmallCapEngine.__new__(SmallCapEngine)  # avoid DB/settings loader
    engine.settings, engine.config = settings, {}
    engine.signals = SmallCapSignals(settings=settings)
    engine.filters = SmallCapFilters(settings=settings)
    engine.scoring = SmallCapScoring(settings=settings)
    # No execution/risk work: return the signal unchanged at the test boundary.
    engine.risk = SimpleNamespace(add_risk_management=lambda signal, *_a, **_k: signal)
    cut = df.loc[pd.Timestamp(SESSION+timedelta(days=1)-relativedelta(months=6)):]
    champion = engine.scan_stock("AAA", cut, {"marketCap": 1e9, "floatShares": 1e6},
                                 backtest_mode=True, earnings_dates=[], regime="BULL")
    assert champion is not None
    assert shadow["final_status"] == "PASS"
    assert champion["quality_score"] == round(shadow["quality_score"], 1)
    assert champion["rank_score"] == round(shadow["rank_score"], 1)
    assert champion["swing_type"] == shadow["swing_type"]
    assert champion["trigger_pathway"] == shadow["setup"]
    if seed == "vce":
        assert shadow["stages"]["vce"]["status"] == shadow["stages"]["rvol"]["status"] == "PASS"
        assert shadow["setup"] == "vce_breakout"
        assert shadow["rank_score"] - shadow["quality_score"] == 13
        assert shadow["stages"]["rsi"]["value"] > 70 and shadow["stages"]["rsi"]["status"] == "PASS"


def test_complete_legacy_comparison_distinguishes_unchecked(tmp_path):
    snapshot = build_snapshot(payloads(), "2026-10-10T18:00:00+00:00")
    df = frame(seed=5)
    df.loc[df.index[-1], "Volume"] = 3e6
    key = store_object(tmp_path, obj(df))
    prep = {"session": str(SESSION), "outcomes": [{"symbol": "AAA", "object_id": key, "status": "OK"}]}
    legacy = {"bar_session": str(SESSION), "complete": True,
              "finviz_membership_inferred_from_rank_telemetry": True, "finviz_selected": ["ADR"]}
    result = scan(snapshot, prep, tmp_path, SmallCapSettings(), ASOF, legacy)
    assert result["comparison"]["broad_trigger_only"] == ["AAA"]
    assert result["comparison"]["legacy_without_broad_trigger"] == []
    assert result["comparison"]["legacy_data_unknown"] == ["ADR"]


def test_provider_identity_mismatch_is_not_eligible():
    cached = obj()
    cached["identity"]["instrumentType"] = "ETF"
    result = evaluate(record(), cached, SESSION, SmallCapSettings(), {"regime": "UNKNOWN"}, decision_at=ASOF)
    assert result["stages"]["provider_identity"]["status"] == "FAIL"


def test_reference_index_calendar_alignment_is_explicit():
    df = frame()
    df.loc[pd.Timestamp("2026-07-03")] = df.iloc[0]  # cash-equity holiday
    df = df.sort_index()
    with pytest.raises(ValueError, match="NON_SESSION"):
        normalize(df, SESSION)
    rows, report = normalize(df, SESSION, reference_calendar=True)
    assert report["reference_non_cash_sessions_excluded"] == ["2026-07-03"]
    assert "2026-07-03" not in [r["Date"] for r in rows]


def test_future_bar_does_not_change_signal_features():
    cached = obj()
    original = evaluate(record(), cached, SESSION, SmallCapSettings(), {"regime": "UNKNOWN"}, decision_at=ASOF)
    cached["rows"].append({**cached["rows"][-1], "Date": "2026-10-12", "Close": 1000., "High": 1000., "Volume": 1e12})
    altered = evaluate(record(), cached, SESSION, SmallCapSettings(), {"regime": "UNKNOWN"}, decision_at=ASOF)
    for k in ("stages", "quality_score", "rank_score", "trigger_details", "reference_close", "final_status"):
        assert original[k] == altered[k]


def test_calendar_estimate_range_not_false_event_free_gap():
    from .facts import interpret_calendar
    coverage, dates = interpret_calendar({"Earnings Date": [date(2026, 10, 10), date(2026, 10, 13)]}, SESSION)
    assert coverage == "VERIFIED" and dates == ["2026-10-10", "2026-10-11", "2026-10-12", "2026-10-13"]
    assert interpret_calendar({}, SESSION)[0] == "UNKNOWN"
    assert interpret_calendar({"Earnings Date": [date(2026, 9, 1)]}, SESSION)[0] == "UNKNOWN"


def test_unknown_uppercase_exception_does_not_leak_secrets(tmp_path):
    provider = Provider(error=ValueError("SECRET_API_TOKEN=ABC123"))
    result = prepare_one(record(), SESSION, tmp_path, provider, sleeper=lambda _: None)
    assert result["status"] == "ValueError" and result["retries"] == 1
    assert "ABC123" not in json.dumps(result)


def test_empty_partial_retry_and_gap_repair(tmp_path):
    class Flaky(Provider):
        def history(self, symbol, start, end):
            if not self.calls:
                self.calls.append((symbol, start, end))
                return pd.DataFrame(), {}
            return super().history(symbol, start, end)
    p = Flaky(frame())
    result = prepare_one(record(), SESSION, tmp_path, p, sleeper=lambda _: None)
    assert result["status"] == "OK" and result["retries"] == 1 and result["attempts"] == 2
    broken = obj(frame().drop(pd.Timestamp("2026-10-06")))
    prior = {"object_id": store_object(tmp_path, broken)}
    p = Provider(frame())
    repaired = prepare_one(record(), SESSION, tmp_path, p, prior)
    assert repaired["status"] == "OK" and p.calls[0][1] == date(2026, 10, 6)
    assert not read_object(tmp_path, repaired["object_id"])["validation"]["missing_sessions"]


@pytest.mark.parametrize("ratio", [2., .1])
def test_split_reverse_split_rebuild_preserves_old_revision(tmp_path, ratio):
    df = frame()
    p = Provider(df)
    old = prepare_one(record(), SESSION, tmp_path, p)
    later = date(2026, 10, 12)
    revised = df.copy()
    revised.loc[pd.Timestamp(later)] = df.iloc[-1]
    revised[["Open", "High", "Low", "Close", "Adj Close"]] /= ratio
    revised["Volume"] *= ratio
    revised.loc[pd.Timestamp(later), "Stock Splits"] = ratio
    p.df = revised
    new = prepare_one(record(), later, tmp_path, p, old)
    assert new["revision_rebuild"] and new["attempts"] == 2
    before, after = read_object(tmp_path, old["object_id"]), read_object(tmp_path, new["object_id"])
    assert after["rows"][0]["Close"] == pytest.approx(before["rows"][0]["Close"]/ratio)
    assert after["rows"][0]["Volume"] == pytest.approx(before["rows"][0]["Volume"]*ratio)


def test_interrupted_run_resumes_only_uncheckpointed_symbol(tmp_path):
    class Interrupted(Provider):
        def history(self, symbol, start, end):
            if symbol == "BBB":
                raise KeyboardInterrupt()
            return super().history(symbol, start, end)
    records = [record(), record("BBB")]
    p = Interrupted(frame())
    with pytest.raises(KeyboardInterrupt):
        prepare(records, SESSION, tmp_path, tmp_path / "run", p)
    assert len(list((tmp_path / "run" / "checkpoints").glob("*.json"))) == 1
    resumed = Provider(frame())
    result = prepare(records, SESSION, tmp_path, tmp_path / "run", resumed)
    assert result["counts"] == {"OK": 2}
    assert [c[0] for c in resumed.calls] == ["BBB"]


def test_mapping_failures_do_not_trip_provider_circuit(tmp_path):
    records = [{**record(f"AAA{i}"), "yfinance_symbol_candidate": None} for i in range(6)] + [record("BBB")]
    result = prepare(records, SESSION, tmp_path, tmp_path / "run", Provider(frame()))
    assert result["circuit"] is None and result["outcomes"][-1]["status"] == "OK"


def test_zero_volume_warning_does_not_change_trigger_algorithm():
    df = frame(seed=5)
    df.loc[df.index[-1], "Volume"] = 3e6
    df.loc[df.index[-30], "Volume"] = 0
    result = evaluate(record(), obj(df), SESSION, SmallCapSettings(), {"regime": "BULL", "confidence": "CONFIRMED"}, decision_at=ASOF)
    assert result["trigger_positive"]
    assert result["stages"]["data_quality"]["status"] == "UNKNOWN"
    assert result["stages"]["quality"]["status"] == "UNKNOWN"  # missing float must not become a rejection
    assert result["quality_range_given_unknown_float"][0] <= result["quality_range_given_unknown_float"][1]


def test_bad_cache_reported_per_symbol_without_network(tmp_path):
    p = Provider(frame())
    result = prepare_one(record(), SESSION, tmp_path, p, {"object_id": "f"*64})
    assert result["status"] == "CACHE_INTEGRITY_REVIEW_REQUIRED" and p.calls == []


def test_scan_source_bundle_and_exclusive_manifest(tmp_path):
    from .shadow import save_scan
    from .master import digest
    snapshot = build_snapshot(payloads(), "2026-10-10T18:00:00+00:00")
    prep = {"session": str(SESSION), "outcomes": []}
    result = scan(snapshot, prep, tmp_path, SmallCapSettings(), ASOF)
    path = tmp_path / "scan"
    save_scan(path, result, {"example": "frozen"}, {"elapsed_seconds": 0})
    manifest = json.loads((path / "manifest.json").read_text())
    for name, checksum in manifest["sha256"].items():
        assert digest((path / name).read_bytes()) == checksum
    assert "source/swing_trader/small_cap/signals.py" in manifest["sha256"]
    with pytest.raises(FileExistsError):
        save_scan(path, result)


def test_52_bar_ipo_rvol_is_not_blocked_by_55_bar_vce_requirement():
    df = frame(seed=5).tail(52).copy()
    df.loc[df.index[-1], "Volume"] = 3e6
    result = evaluate(record(), obj(df), SESSION, SmallCapSettings(), {"regime": "UNKNOWN"}, decision_at=ASOF)
    assert result["stages"]["ohlcv"]["status"] == "PASS"
    assert result["stages"]["vce"]["status"] == "UNKNOWN"
    assert result["stages"]["rvol"]["status"] == "PASS"
    assert result["trigger_positive"] and result["setup"] == "rvol_thrust"
