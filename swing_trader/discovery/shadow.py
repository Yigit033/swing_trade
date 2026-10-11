"""Pure offline broad-market adapter to champion signal/score functions.

No risk planner, execution, scheduler, DB, settings loader or provider calls.
Diagnostic scores are computed even for rejected triggers; eligibility remains
tri-state. Unknown float uses the EXACT production unknown-float score, labeled.
"""
from __future__ import annotations

import json
import platform
import math
from importlib.metadata import version
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from dateutil.relativedelta import relativedelta

import pandas as pd

from swing_trader.small_cap.engine import SmallCapEngine
from swing_trader.small_cap.filters import SmallCapFilters
from swing_trader.small_cap.signals import SmallCapSignals
from swing_trader.small_cap.scoring import SmallCapScoring
from swing_trader.small_cap.patterns import detect_weinstein_stage
from swing_trader.small_cap.regime_logic import regime_from_spy_close, regime_unknown
from swing_trader.small_cap.thresholds import effective_scan_thresholds

from .bars import rows_frame, object_id, read_object, MIN_EQUITY_BARS
from .calendar import sessions, last_completed
from .master import digest, write_json


def gate(passed: bool | None, reason: str, value=None) -> dict:
    return {"status": "UNKNOWN" if passed is None else "PASS" if passed else "FAIL",
            "reason": reason, "value": value}


def combine(stages: list[dict]) -> str:
    statuses = {s["status"] for s in stages}
    return "FAIL" if "FAIL" in statuses else "UNKNOWN" if "UNKNOWN" in statuses else "PASS"


def positive_number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0 else None


def code_manifest() -> dict:
    root = Path(__file__).resolve().parents[2]
    files = list((root / "swing_trader" / "discovery").glob("*.py"))
    files += [root / "swing_trader" / "small_cap" / x for x in (
        "signals.py", "scoring.py", "engine.py", "filters.py", "patterns.py", "trend_quality.py",
        "settings_config.py", "settings_models_extra.py", "regime_logic.py", "thresholds.py")]
    files += [root / "swing_trader" / "utils" / "market_calendar.py"]
    return {str(p.relative_to(root)).replace("\\", "/"): digest(p.read_bytes()) for p in sorted(files)}


def frame_status(obj: dict | None, session: date, minimum=MIN_EQUITY_BARS) -> tuple[pd.DataFrame | None, dict]:
    if not obj:
        return None, gate(None, "NO_CACHE_OBJECT")
    if obj.get("mode") != "YAHOO_SPLIT_OHLC_VOLUME__DIVIDEND_RATIO_OHLC_ONLY":
        return None, gate(None, "UNVERIFIED_ADJUSTMENT_MODE")
    df = rows_frame(obj["rows"])
    df = df.loc[df.index.date <= session]
    if df.empty:
        return None, gate(None, "NO_BARS_AT_SESSION")
    if df.index[-1].date() != session:
        return None, gate(None, "STALE", df.index[-1].date().isoformat())
    if len(df) < minimum:
        return None, gate(None, "INSUFFICIENT_BARS", len(df))
    expected = sessions(df.index[-minimum].date(), session)
    if list(df.index[-minimum:].date) != expected:
        return None, gate(None, "GAPS_IN_REQUIRED_LOOKBACK")
    return df, gate(True, "COMPLETED_CONTIGUOUS_BARS", len(df))


def evaluate(record: dict, obj: dict | None, session: date, settings, regime: dict,
             facts: dict | None = None, decision_at: str | None = None) -> dict:
    facts = facts or {}
    sig, filt, score = SmallCapSignals(settings=settings), SmallCapFilters(settings=settings), SmallCapScoring(settings=settings)
    stages = {"universe": gate(record["discovery_status"] == "CANDIDATE", record["discovery_reason"], record["security_type"])}
    minimum_bars = max(MIN_EQUITY_BARS, settings.volume_surge_baseline_days + 1,
                       settings.universe_filters.atr_period + 1)
    df, stages["ohlcv"] = frame_status(obj, session, minimum_bars)
    result = {"ticker": record["symbol"], "exchange": record["exchange"], "security_type": record["security_type"],
              "security_flags": {k: record.get(k, "UNKNOWN") for k in
                  ("classification_confidence", "adr_status", "spac_status", "share_class", "ipo_status", "symbol_mapping_status")},
              "session": session.isoformat(), "stages": stages, "trigger_positive": False,
              "quality_score": None, "rank_score": None, "hard_gate_status": "UNKNOWN",
              "known_hard_gates_pass": False, "final_status": "UNKNOWN", "cache_object_id": object_id(obj) if obj else None}
    for name in ("data_quality", "provider_identity", "price", "dollar_volume", "market_cap", "earnings", "vce", "rvol", "swing", "rsi", "weinstein", "quality"):
        stages[name] = gate(None, "NOT_EVALUATED")
    if df is None or stages["universe"]["status"] != "PASS":
        result["rejection_reasons"] = [stages["ohlcv"]["reason"]]
        return result
    if obj["symbol"] != record["symbol"]:
        raise ValueError("CACHE_SYMBOL_MISMATCH")
    zero_volume = int((df.Volume.tail(minimum_bars) == 0).sum())
    stages["data_quality"] = gate(None if zero_volume else True,
        "ZERO_VOLUME_CAUTION_TRIGGER_STILL_EVALUATED" if zero_volume else "REQUIRED_BAR_VALIDATION_PASSED", zero_volume)
    identity = obj.get("identity", {})
    exchanges = {"NASDAQ": {"NMS", "NGM", "NCM"}, "NYSE": {"NYQ"}}
    evidence = [identity.get(k) for k in ("symbol", "exchangeName", "instrumentType", "currency")]
    identity_ok = (identity.get("symbol") == obj.get("vendor_symbol")
                   and identity.get("exchangeName") in exchanges.get(record["exchange"], set())
                   and identity.get("instrumentType") == "EQUITY" and identity.get("currency") == "USD")
    stages["provider_identity"] = gate(identity_ok if all(evidence) else None,
                                       "SYMBOL_EXCHANGE_EQUITY_USD_MATCH_ONLY_NOT_PERMANENT_ID")
    # Production fetcher requests 6mo; preserve its EMA initialization window.
    # Freeze the window relative to EOD session instead of wall-clock replay day.
    from datetime import timedelta
    window_start = session + timedelta(days=1) - relativedelta(months=6)
    df = df.loc[df.index >= pd.Timestamp(window_start)]
    if len(df) < minimum_bars:
        stages["ohlcv"] = gate(None, "INSUFFICIENT_SIX_MONTH_WINDOW")
        result["rejection_reasons"] = ["INSUFFICIENT_SIX_MONTH_WINDOW"]
        return result
    history_gaps = sorted(set(sessions(df.index[0].date(), session)) - set(df.index.date))
    if history_gaps:
        stages["data_quality"] = gate(None, "SIX_MONTH_SCORE_WINDOW_GAPS_TRIGGER_LOOKBACK_COMPLETE",
                                       [d.isoformat() for d in history_gaps])
    price, dv = float(df.Close.iloc[-1]), filt.calculate_avg_dollar_volume(df)
    passed, reason = filt.check_price(price)
    stages["price"] = gate(passed, reason, price)
    passed, reason = filt.check_dollar_volume(dv)
    stages["dollar_volume"] = gate(passed, reason, dv)
    # Facts must be observed no later than decision and explicitly applicable
    # to this session; current fundamentals are never backdated in replay.
    try:
        observed = datetime.fromisoformat(facts["observed_at"])
        decision = datetime.fromisoformat(decision_at)
        facts_valid = (observed.utcoffset() is not None and decision.utcoffset() is not None
                       and facts.get("session") == session.isoformat() and observed <= decision)
    except (KeyError, ValueError, TypeError):
        facts_valid = False
    result["metadata_observation"] = {"valid_for_decision": facts_valid, "observed_at": facts.get("observed_at"),
        "source": facts.get("source"), "earnings_evidence": facts.get("earnings_evidence"),
        "errors": facts.get("errors", [])}
    cap = positive_number(facts.get("market_cap")) if facts_valid else None
    if cap is not None and cap > 0:
        passed, reason = filt.check_market_cap(cap)
        stages["market_cap"] = gate(passed, reason, cap)
    else:
        stages["market_cap"] = gate(None, "NO_POINT_IN_TIME_MARKET_CAP")
    if facts_valid and facts.get("earnings_coverage") == "VERIFIED":
        passed, reason = filt.check_earnings(record["symbol"], session.isoformat(), facts.get("earnings_dates", []))
        stages["earnings"] = gate(passed, reason)
    else:
        stages["earnings"] = gate(None, "NO_VERIFIED_EVENT_CALENDAR")
    triggered, td = sig.check_all_triggers(df)  # EXACT champion pathway logic
    for name, key in (("vce", "vce_breakout"), ("rvol", "rvol_thrust")):
        t = td["triggers"][key]
        ready = len(df) >= (55 if name == "vce" else 52)
        stages[name] = gate(bool(t["passed"]) if ready else None, t["reason"], t["metrics"])
    result.update(trigger_positive=bool(triggered), setup=td.get("trigger_pathway"),
                  trigger_details=td, reference_close=price, avg_dollar_volume_20=dv,
                  volume=float(df.Volume.iloc[-1]), atr_percent=float(td["atr_percent"]),
                  input_bars=len(df), input_first_session=str(df.index[0].date()))
    boosters = sig.check_boosters(df)
    details = boosters["swing_details"]
    boosters["rsi_divergence"] = sig.detect_rsi_divergence(df, lookback=14)["divergence_found"]
    macd = sig.calculate_macd(df)
    boosters["macd_bullish"] = macd["bullish_cross"] or (macd["above_zero"] and macd["expanding"])
    spread = float(df.High.iloc[-1] - df.Low.iloc[-1])
    swing_type, _, _ = SmallCapEngine._classify_swing_type(SimpleNamespace(settings=settings),
        details.get("five_day_momentum", {}).get("return", 0), boosters["rsi"], td["volume_surge"], boosters["higher_lows"],
        close_position=(price - float(df.Low.iloc[-1])) / spread if spread > 0 else .5,
        ma20_distance=details.get("above_ma20", {}).get("distance", 0),
        rsi_divergence=boosters["rsi_divergence"], macd_bullish=boosters["macd_bullish"])
    boosters["swing_type"] = swing_type
    stages["swing"] = gate(bool(boosters["swing_ready"]), "CHAMPION_SWING_CONFIRMATION")
    stages["rsi"] = gate(boosters["rsi"] <= settings.max_entry_rsi or td.get("trigger_pathway") == "vce_breakout",
                         "CHAMPION_RSI_GATE_VCE_EXEMPT", float(boosters["rsi"]))
    stage = detect_weinstein_stage(df)["stage"]
    sg = settings.scan_gates
    stages["weinstein"] = gate(not ((stage == 3 and sg.reject_stage3) or (stage == 4 and sg.reject_stage4)),
                               "CHAMPION_WEINSTEIN_GATE", stage)
    float_shares = positive_number(facts.get("float_shares")) if facts_valid else None
    q = score.calculate_quality_score(df, td["volume_surge"], td["atr_percent"], float_shares, boosters)
    metrics = td.get("vce_metrics", {})
    rank = q + (8 if metrics.get("is_premium") else 0) + (5 if 0 < metrics.get("squeeze_ratio", 1) < .65 else 0)
    result.update(quality_score=float(q), rank_score=float(rank), swing_type=swing_type,
                  float_shares=float_shares,
                  float_status="KNOWN" if float_shares else "UNKNOWN_PRODUCTION_DEFAULT_SCORE",
                  score_semantics="DIAGNOSTIC_CHAMPION_FORMULA_NOT_TRADE_ELIGIBILITY")
    if regime.get("regime") in {"BULL", "CAUTION", "BEAR"}:
        minimum, top = effective_scan_thresholds(regime["regime"], regime["confidence"],
            settings.auto_scan.min_quality, settings.auto_scan.top_n, settings.regime_thresholds)
        result["quality_gate_at_production_unknown_float_default"] = "PASS" if q >= minimum else "FAIL"
        stages["quality"] = gate(q >= minimum if float_shares and float_shares > 0 else None,
            "CHAMPION_EFFECTIVE_QUALITY_FLOOR" if float_shares and float_shares > 0 else "FLOAT_UNKNOWN_SCORE_IS_DIAGNOSTIC",
            {"minimum": minimum, "top_n": top, "request_policy": "EXPLICIT_AUTO_SCAN_SETTINGS_NO_JOB_EXECUTION"})
        if triggered and not (float_shares and float_shares > 0):
            # These are data-uncertainty bounds, NOT tuned parameters/strategies.
            bands = settings.scoring_tuning.float_millions_bands
            alternatives = [b.max_millions_le * 1e6 for b in bands]
            alternatives += [max(alternatives) * 2]
            scores = [score.calculate_quality_score(df, td["volume_surge"], td["atr_percent"], f, boosters)
                      for f in alternatives]
            result["quality_range_given_unknown_float"] = [float(min(scores)), float(max(scores))]
    else:
        stages["quality"] = gate(None, "REGIME_UNAVAILABLE_NO_RELAXED_FLOOR")
    hard = [stages[x] for x in ("price", "dollar_volume", "market_cap", "earnings")]
    result["hard_gate_status"] = combine(hard)
    result["known_hard_gates_pass"] = all(s["status"] != "FAIL" for s in hard)
    result["final_status"] = combine(hard + [stages[x] for x in ("data_quality", "provider_identity", "swing", "rsi", "weinstein", "quality")]
                                     + [gate(bool(triggered), "ANY_CHAMPION_TRIGGER")])
    result["rejection_reasons"] = [k for k, v in stages.items() if v["status"] == "FAIL" and k not in {"vce", "rvol"}]
    if not triggered:
        result["rejection_reasons"].append("NO_TRIGGER")
    result["unknown_reasons"] = [k for k, v in stages.items() if v["status"] == "UNKNOWN"]
    # JSON-primitive conversion of numpy scalar values from production helpers.
    return json.loads(json.dumps(result, default=lambda x: x.item() if hasattr(x, "item") else str(x), allow_nan=False))


def scan(snapshot: dict, preparation: dict, cache: Path, settings, decision_at: str,
         legacy: dict | None = None, facts: dict | None = None) -> dict:
    session = date.fromisoformat(preparation["session"])
    asof = datetime.fromisoformat(decision_at.replace("Z", "+00:00"))
    if session > last_completed(asof):
        raise ValueError("SCAN_BEFORE_EOD_FINALIZATION")
    if datetime.fromisoformat(snapshot["observed_at"]) > asof:
        raise ValueError("UNIVERSE_OBSERVED_AFTER_DECISION")
    outcomes = {x["symbol"]: x for x in preparation["outcomes"]}
    objects = {s: read_object(cache, o["object_id"]) for s, o in outcomes.items() if o.get("object_id")}
    if any(datetime.fromisoformat(o["fetched_at"]) > asof for o in objects.values()):
        raise ValueError("CACHE_OBSERVED_AFTER_DECISION")
    spy, _ = frame_status(objects.get("SPY"), session, 204)
    vix, _ = frame_status(objects.get("^VIX"), session, 5)
    regime = regime_from_spy_close(spy.Close, float(vix.Close.iloc[-1])) if spy is not None and vix is not None else regime_unknown("SPY_OR_VIX_INCOMPLETE")
    legacy = legacy or {}
    same = (legacy.get("bar_session") == session.isoformat() and legacy.get("complete") is True
            and legacy.get("finviz_membership_inferred_from_rank_telemetry") is True)
    members = set(legacy.get("finviz_selected", [])) if same else set()
    records = sorted((r for r in snapshot["records"] if r["discovery_status"] == "CANDIDATE"), key=lambda r: r["symbol"])
    if len({r["symbol"] for r in records}) != len(records):
        raise ValueError("DUPLICATE_UNIVERSE_SYMBOL")
    rows = []
    for r in records:
        row = evaluate(r, objects.get(r["symbol"]), session, settings, regime, (facts or {}).get(r["symbol"]), decision_at)
        row["legacy_membership"] = ("YES" if r["symbol"] in members else "NO") if same else "UNKNOWN"
        row["preparation_status"] = outcomes.get(r["symbol"], {}).get("status", "NOT_REQUESTED")
        rows.append(row)
    triggers = {r["ticker"] for r in rows if r["trigger_positive"]}
    evaluated = {r["ticker"] for r in rows if r["stages"]["ohlcv"]["status"] == "PASS"}
    fresh = {s for s, o in objects.items() if any(bar["Date"] == session.isoformat() for bar in o["rows"])}
    fresh &= {r["ticker"] for r in rows}
    full = sum(all(r["stages"][k]["status"] != "UNKNOWN" for k in
                   ("ohlcv", "data_quality", "provider_identity", "price", "dollar_volume", "market_cap", "earnings", "vce", "rvol", "swing", "rsi", "weinstein", "quality")) for r in rows)
    metrics = {"master_candidates": len(rows), "data_available": sum(r["ticker"] in objects for r in rows),
        "last_session_fresh": len(fresh), "fresh_coverage_pct": 100*len(fresh)/len(rows) if rows else 0,
        "fresh_sufficient_ohlcv": len(evaluated), "usable_coverage_pct": 100*len(evaluated)/len(rows) if rows else 0,
        "full_evaluation": full, "full_evaluation_coverage_pct": 100*full/len(rows) if rows else 0,
        "trigger_positive": len(triggers), "vce_positive": sum(r["stages"]["vce"]["status"] == "PASS" for r in rows),
        "rvol_positive": sum(r["stages"]["rvol"]["status"] == "PASS" for r in rows),
        "hard_gate_eligible": sum(r["hard_gate_status"] == "PASS" for r in rows),
        "trigger_and_hard_gate_eligible": sum(r["trigger_positive"] and r["hard_gate_status"] == "PASS" for r in rows),
        "trigger_known_hard_gates_pass": sum(r["trigger_positive"] and r["known_hard_gates_pass"] for r in rows),
        "final_pass": sum(r["final_status"] == "PASS" for r in rows),
        "final_conclusive_fail": sum(r["final_status"] == "FAIL" for r in rows),
        "final_unknown": sum(r["final_status"] == "UNKNOWN" for r in rows),
        "trigger_metadata_unknown": sum(r["trigger_positive"] and any(r["stages"][k]["status"] == "UNKNOWN" for k in ("market_cap", "earnings")) for r in rows),
        "stage_counts": {k: dict(Counter(r["stages"].get(k, {"status": "UNKNOWN"})["status"] for r in rows))
                         for k in sorted({k for r in rows for k in r["stages"]})},
        "ohlcv_reason_counts": dict(Counter(r["stages"]["ohlcv"]["reason"] for r in rows)),
        "rejection_reason_counts_nonexclusive": dict(Counter(k for r in rows for k in r["rejection_reasons"])),
        "trigger_rejected_downstream": sum(r["trigger_positive"] and any(r["stages"][k]["status"] == "FAIL"
                  for k in ("swing", "rsi", "weinstein", "quality")) for r in rows),
        "fresh_by_exchange": dict(Counter(r["exchange"] for r in rows if r["ticker"] in evaluated)),
    }
    comparison = {"status": ("COMPLETE" if len(evaluated) == len(rows) and members <= evaluated else "INCOMPLETE_DATA_COVERAGE") if same else "UNKNOWN_INCOMPLETE",
        "legacy_membership_status": "SAME_SESSION_VERIFIED" if same else "UNKNOWN",
        "universe_point_in_time": False,
        "not_executable_recall": True,
        "semantics": "LEGACY_DISCOVERY_MEMBERSHIP_VS_BROAD_TRIGGER_NOT_LEGACY_SIGNALS",
        "legacy_session": legacy.get("bar_session"), "legacy_count": len(members) if same else None,
        "broad_trigger_only": sorted(triggers - members) if same else None,
        "overlap": sorted(triggers & members) if same else None,
        "legacy_without_broad_trigger": sorted((members & evaluated) - triggers) if same else None,
        "legacy_data_unknown": sorted(members - evaluated) if same else None}
    inputs = {"universe_id": object_id(snapshot), "preparation_id": object_id(preparation),
              "config": settings.model_dump(mode="json"), "code": code_manifest(), "facts_id": object_id(facts or {}),
              "runtime": {"python": platform.python_version(), **{k: version(k) for k in ("pandas", "numpy", "yfinance", "pydantic")}},
              "legacy_id": object_id(legacy), "decision_at": decision_at, "session": session.isoformat()}
    return {"schema": 1, "inputs": inputs, "scan_id": object_id(inputs), "regime": regime,
            "input_window": "EQUITY_SIX_CALENDAR_MONTHS_END_SESSION_PLUS_ONE_DAY;SPY_FULL_CACHE_MIN204",
            "universe_is_historical_point_in_time": False, "metrics": metrics, "comparison": comparison, "rows": rows}


def save_scan(path: Path, result: dict, inputs: dict | None = None, execution: dict | None = None):
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "scan.json", result)
    files = ["scan.json"]
    if inputs is not None:
        write_json(path / "inputs.json", inputs)
        files.append("inputs.json")
    if execution is not None:
        write_json(path / "execution.json", execution)
        files.append("execution.json")
    source_root = Path(__file__).resolve().parents[2]
    for name, expected in result["inputs"]["code"].items():
        raw = (source_root / name).read_bytes()
        if digest(raw) != expected:
            raise ValueError("SOURCE_CHANGED_DURING_SCAN")
        relative = "source/" + name
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(raw)
        files.append(relative)
    write_json(path / "manifest.json", {"complete": True, "sha256": {f: digest((path / f).read_bytes()) for f in files},
                                       "scan_id": result["scan_id"]})
