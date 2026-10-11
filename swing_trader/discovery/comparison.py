"""Snapshot set comparison and bounded OFFLINE historical-cache diagnostics.

No historical discovery performance claim: today's membership and today's
Finviz result cannot serve as a historical control cohort.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd


def resolve_legacy_symbols(records: list[dict], symbols: set[str]) -> tuple[set[str], dict]:
    """Use unique official aliases or explicit class-share spelling only.

    Ambiguous aliases stay unresolved; never merge different share classes.
    """
    index = {}
    for row in records:
        aliases = {row["symbol"], row["raw"].get("CQS Symbol"), row["raw"].get("NASDAQ Symbol")}
        if row["symbol_mapping_status"] == "CLASS_ALIAS_UNVERIFIED":
            aliases.add(row["yfinance_symbol_candidate"])
        for alias in aliases - {None, ""}:
            index.setdefault(alias, set()).add(row["symbol"])
    matched, details = set(), {}
    for symbol in sorted(symbols):
        targets = index.get(symbol, set())
        if len(targets) == 1:
            target = next(iter(targets))
            matched.add(target)
            if target != symbol:
                details[symbol] = {"status": "ALIAS_MATCH", "official_symbol": target}
        else:
            details[symbol] = {"status": "AMBIGUOUS" if targets else "UNRESOLVED", "targets": sorted(targets)}
    return matched, details


def compare(snapshot: dict, legacy: dict) -> dict:
    scope = [r for r in snapshot["records"] if r["exchange"] in {"NASDAQ", "NYSE"} and not r["test_issue"]]
    candidates = [r for r in scope if r["discovery_status"] == "CANDIDATE"]
    raw_legacy_set = set(legacy["selected"])
    legacy_set, mapping = resolve_legacy_symbols(scope, raw_legacy_set)
    finviz_set = set(legacy["finviz_selected"])
    scope_symbols = {r["symbol"] for r in scope}
    missing = [r for r in scope if r["symbol"] not in legacy_set]
    new_candidates = [r for r in candidates if r["symbol"] not in legacy_set]
    exclusions = [r for r in scope if r["symbol"] in legacy["exclusions"]]
    observed_rows = {r["Ticker"]: r for r in legacy["rows"]}
    def numeric_known(symbol, key):
        value = pd.to_numeric(observed_rows.get(symbol, {}).get(key), errors="coerce")
        return pd.notna(value) and np.isfinite(value) and value > 0
    dates = [m["file_creation_time_raw"] for m in snapshot["sources"]]
    return {"snapshot_observed_at": snapshot["observed_at"], "source_creation_times": dates,
            "legacy_observed_at": legacy["observed_at"], "legacy_complete": legacy["complete"],
            "legacy_kind": legacy["kind"], "legacy_bar_session": legacy.get("bar_session"),
            "comparison_semantics": "OBSERVED_SET_DIFFERENCE_NOT_MISSED_PROFIT_OR_HISTORICAL_RECALL",
            "same_market_session_verified": False,
            "summary": snapshot["summary"],
            "legacy_finviz_count": len(finviz_set) if legacy["complete"] and
                (legacy["kind"] != "RECORDED_LEGACY_RUN" or legacy.get("finviz_membership_inferred_from_rank_telemetry")) else None,
            "legacy_observed_or_inferred_finviz_selected_count": len(finviz_set),
            "legacy_final_count": len(raw_legacy_set), "symbol_resolution": mapping,
            "legacy_in_scope_count": len(legacy_set & scope_symbols),
            "legacy_not_in_scope": sorted(t for t, m in mapping.items() if m["status"] != "ALIAS_MATCH"),
            "master_not_in_legacy_count": len(missing),
            "candidate_not_in_legacy_count": len(new_candidates),
            "absent_by_exchange": dict(Counter(r["exchange"] for r in missing)),
            "candidate_absent_by_exchange": dict(Counter(r["exchange"] for r in new_candidates)),
            "candidate_not_in_legacy": [r["symbol"] for r in new_candidates],
            "listed_hardcoded_exclusions": [{"symbol": r["symbol"], "exchange": r["exchange"],
                "type": r["security_type"], "observed_finviz_rejection":
                (r["symbol"] in legacy["funnel"]["excluded"] if legacy["funnel"]["excluded"] is not None else None),
                "per_ticker_rationale": "NOT_PRESENT_IN_CODE"} for r in exclusions],
            "metadata_coverage_from_observed_legacy_rows": {
                "denominator_candidate_count": len(candidates),
                "missing_price": sum(not numeric_known(r["symbol"], "Price") for r in candidates),
                "missing_current_volume": sum(not numeric_known(r["symbol"], "Volume") for r in candidates),
                "missing_market_cap": sum(not numeric_known(r["symbol"], "Market Cap") for r in candidates),
                "note": "Not enrichment; only current legacy rows. Current volume is not average dollar volume."},
            "legacy_funnel": legacy["funnel"]}


def inspect_cache(snapshot: dict, legacy: dict, bars: dict, settings, as_of: str,
                  cache_metadata: dict) -> dict:
    """Trigger mechanics only on existing cache, sliced to explicit past date.

    Known non-equities/test issues are retained in master but not scanned here.
    No engine.scan_stock (metadata/API/earnings/risk side effects). No network.
    Unknown market cap/earnings mean these are NOT qualified trading signals.
    """
    from swing_trader.small_cap.signals import SmallCapSignals
    detector = SmallCapSignals(settings=settings)
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is not None:
        raise ValueError("as_of must be a date without timezone")
    candidates = [r for r in snapshot["records"] if r["discovery_status"] == "CANDIDATE"]
    legacy_set, _ = resolve_legacy_symbols(snapshot["records"], set(legacy["selected"]))
    coverage = Counter()
    observed_dates, trigger_rows = [], []
    groups = {"all_candidates": Counter(), "not_in_current_legacy": Counter()}
    for record in candidates:
        ticker = record["symbol"]
        group_keys = ["all_candidates"] + (["not_in_current_legacy"] if ticker not in legacy_set else [])
        for key in group_keys:
            groups[key]["denominator"] += 1
        # Prefer official spelling. Aliases only used when exact cache key absent.
        lookup = ticker if ticker in bars else record["yfinance_symbol_candidate"]
        if lookup is None or lookup not in bars:
            coverage["no_cached_symbol"] += 1
            continue
        for key in group_keys:
            groups[key]["cached_symbol"] += 1
        df = bars[lookup].copy()
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(df.columns):
            coverage["missing_columns"] += 1
            continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
        df = df[df["Date"] <= cutoff].sort_values("Date")
        if df.empty:
            coverage["no_bars_on_or_before_as_of"] += 1
            continue
        if df["Date"].duplicated().any():
            coverage["duplicate_dates"] += 1
            continue
        observed_dates.append(df["Date"].iloc[-1].strftime("%Y-%m-%d"))
        if df["Date"].iloc[-1].normalize() != cutoff:
            coverage["stale_for_as_of"] += 1
            continue
        values = df[list(required - {"Date"})].apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(values).all(axis=1) & (values[["Open", "High", "Low", "Close"]] > 0).all(axis=1) & (values["Volume"] >= 0)
        valid &= (values["High"] >= values[["Open", "Close", "Low"]].max(axis=1)) & (values["Low"] <= values[["Open", "Close", "High"]].min(axis=1))
        if not valid.tail(55).all():
            coverage["invalid_recent_ohlcv"] += 1
            continue
        if len(df) < 55:
            coverage["insufficient_55_bars"] += 1
            continue
        for col in values:
            df[col] = values[col]
        for key in group_keys:
            groups[key]["trigger_evaluable"] += 1
        coverage["trigger_evaluable"] += 1
        vce, _, _ = detector.check_vce_breakout(df)
        rvol, _, _ = detector.check_rvol_thrust(df)
        if vce or rvol:
            price = float(df["Close"].iloc[-1])
            dv = float((df["Close"] * df["Volume"]).tail(20).mean())
            trigger_rows.append({"symbol": ticker, "cache_key": lookup, "exchange": record["exchange"],
                                 "bar_date": as_of, "vce": bool(vce), "rvol": bool(rvol),
                                 "close": price, "avg_dollar_volume20": dv,
                                 "legacy_price_dvol_pass": bool(7 <= price <= 200 and dv >= 5_000_000),
                                 "absent_from_current_legacy": ticker not in legacy_set,
                                 "historical_legacy_membership": "UNKNOWN",
                                 "market_cap_at_signal": "UNKNOWN", "earnings_at_signal": "UNKNOWN"})
    for value in groups.values():
        value["trigger_evaluable_fraction"] = value["trigger_evaluable"] / value["denominator"] if value["denominator"] else None
    snapshot_date = datetime.fromisoformat(snapshot["observed_at"].replace("Z", "+00:00")).date().isoformat()
    return {"kind": "RETROSPECTIVE_CACHE_MECHANICS_ONLY", "as_of": as_of,
            "snapshot_date": snapshot_date, "cache": cache_metadata,
            "historical_point_in_time": False, "same_date_legacy_comparator": False,
            "coverage": dict(coverage), "groups": {k: dict(v) for k, v in groups.items()},
            "latest_cached_date_distribution": dict(Counter(observed_dates)),
            "trigger_rows": trigger_rows,
            "warning": "Current directory survivors + old research cache. Not current signals, missed historical opportunities, or profitability evidence."}
