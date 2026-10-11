"""Observe the unchanged Finviz champion in a standalone research process.

Explicit settings only: never load .env, production DB, API dependencies or jobs.
The four queries and post-filter behavior execute from the existing class.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd

from .master import utc_now


def collect_legacy(settings_patch: dict | None = None) -> dict:
    from swing_trader.small_cap.settings_config import SmallCapSettings
    from swing_trader.small_cap.universe import SmallCapUniverse, _ticker_safe_overview_cls

    settings = SmallCapSettings.model_validate(settings_patch or {})

    class ObservedUniverse(SmallCapUniverse):
        def __init__(self):
            super().__init__(settings=settings)
            self.observations = []

        def _run_finviz_query(self, filters_dict, label):
            observed = {"label": label, "filters": filters_dict, "observed_at": utc_now()}
            try:
                overview = _ticker_safe_overview_cls()()
                overview.set_filter(filters_dict=filters_dict)
                frame = overview.screener_view()
                frame = frame if frame is not None else pd.DataFrame()
                observed.update(status="OK", rows=json.loads(frame.to_json(orient="records")))
            except Exception as exc:
                # Exception text/URLs can contain credentials; persist type only.
                observed.update(status="FAILED", error_type=type(exc).__name__, rows=[])
                response = getattr(exc, "response", None)
                observed["http_status"] = getattr(response, "status_code", None)
                frame = pd.DataFrame()
            self.observations.append(observed)
            return frame

    universe = ObservedUniverse()
    selected = universe.get_universe(force_refresh=True)
    result = summarize_legacy(universe.observations, settings.universe_scan.model_dump(),
                              sorted(universe.EXCLUDED_TICKERS), selected)
    code_file = Path(inspect.getfile(SmallCapUniverse))
    result["legacy_code_sha256"] = hashlib.sha256(code_file.read_bytes()).hexdigest()
    result["settings_provenance"] = "EXPLICIT_RESEARCH_INPUT_NOT_IMPLICIT_PRODUCTION_DB"
    return result


def summarize_legacy(queries: list[dict], settings: dict, exclusions: list[str], selected: list[str]) -> dict:
    """Sequential funnel counts; upstream Finviz predicate attrition is unknown."""
    from swing_trader.small_cap.universe import SmallCapUniverse

    rows = [row for q in queries for row in q["rows"]]
    unique = {}
    for row in rows:
        ticker = str(row["Ticker"]).strip().upper()
        unique.setdefault(ticker, {**row, "Ticker": ticker})
    excluded = sorted(set(unique) & set(exclusions))
    post_exclusion = [r for t, r in unique.items() if t not in exclusions]
    pmin, pmax = settings["post_filter_price_min"], settings["post_filter_price_max"]
    # Match pandas concat semantics: if any query supplied Price, missing Price
    # cells in other queries become NaN and fail the numeric post-filter.
    has_price = any("Price" in row for row in rows)
    priced, missing_price, outside_price = [], [], []
    for row in post_exclusion:
        price = pd.to_numeric(row.get("Price"), errors="coerce")
        if has_price and pd.isna(price):
            missing_price.append(row["Ticker"])
        elif has_price and not pmin <= price <= pmax:
            outside_price.append(row["Ticker"])
        else:
            priced.append(row)
    ranked = pd.DataFrame(priced)
    ranked_tickers = []
    if not ranked.empty:
        parser = SmallCapUniverse._parse_volume
        ranked["dollar_volume"] = ranked["Volume"].apply(lambda v: parser(None, v)) * pd.to_numeric(ranked["Price"], errors="coerce").fillna(15.0)
        ranked_tickers = ranked.sort_values("dollar_volume", ascending=False)["Ticker"].tolist()
    cap = settings["max_scan_tickers"]
    finviz_selected = ranked_tickers[:cap]
    fallback = len(finviz_selected) < settings["min_finviz_tickers_skip_static_merge"] or not settings["use_finviz"]
    return {"observed_at": utc_now(), "kind": "LEGACY_CURRENT_OBSERVATION",
            "complete": len(queries) == 4 and all(q["status"] == "OK" for q in queries),
            "queries": queries, "settings": settings, "exclusions": exclusions,
            "finviz_selected": finviz_selected, "selected": selected,
            "static_merge_used": fallback, "static_added": sorted(set(selected) - set(finviz_selected)),
            "funnel": {"query_rows_with_overlap": len(rows), "unique_query_members": len(unique),
                       "duplicate_query_rows": len(rows) - len(unique),
                       "excluded": excluded, "missing_price": missing_price,
                       "outside_price": outside_price, "after_price": len(priced),
                       "cap_cut": ranked_tickers[cap:], "finviz_selected_count": len(finviz_selected),
                       "final_selected_count": len(selected)},
            "upstream_per_predicate_rejection_counts": None,
            "upstream_count_reason": "Finviz returns conjunction results, not the pre-query population",
            "rows": list(unique.values())}


def recorded_run_legacy(run: dict) -> dict:
    """Adapt a sanitized read-only scan-history export; no historical query invention.

    Membership is only complete when every selected ticker has a recorded row.
    A run missing tickers must not turn failed fetches into discovery exclusions.
    """
    from swing_trader.small_cap.universe import SmallCapUniverse
    stats = run["stats"]
    members = sorted({r["ticker"] for r in stats.get("scanned_members", [])})
    complete = len(members) == stats.get("stocks_scanned") and bool(members)
    finviz_confirmed = complete and stats.get("universe_ranked_total") is not None and not stats.get("universe_fallback")
    return {"kind": "RECORDED_LEGACY_RUN", "observed_at": run["created_at"],
            "bar_session": stats.get("bar_session"), "run_id": run["id"],
            "complete": complete, "finviz_membership_inferred_from_rank_telemetry": finviz_confirmed,
            "selected": members, "finviz_selected": members if finviz_confirmed else [],
            "exclusions": sorted(SmallCapUniverse.EXCLUDED_TICKERS), "rows": [],
            "funnel": {"excluded": None, "missing_price": None, "outside_price": None,
                       "cap_cut": stats.get("universe_cut_tickers"),
                       "final_selected_count": stats.get("stocks_scanned"),
                       "stocks_with_data": stats.get("stocks_with_data"),
                       "reject_counts": stats.get("reject_counts"),
                       "upstream_query_attrition": "NOT_RECORDED"}}
