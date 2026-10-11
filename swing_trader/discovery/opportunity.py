"""Append-only observations and unlevered next-open horizon diagnostics.

NOT a trading simulator: no T1/T2/stop modifications, no portfolio PnL claims.
All triggers (including rejects/UNKNOWN) are recorded to avoid winner selection.
"""
from datetime import date, datetime, time, timezone
from collections import Counter
from pathlib import Path

import pandas as pd

from .calendar import NY, next_session, sessions, last_completed
from .bars import object_id, read_object, rows_frame
from .master import utc_now


def horizon_sessions(signal: date, count=30):
    days, day = [], signal
    for _ in range(count):
        day = next_session(day)
        days.append(day)
    return days


def register(scan: dict, cohort: str, previous: dict | None = None) -> dict:
    if cohort not in {"PROSPECTIVE_SHADOW", "HISTORICAL_RESEARCH"}:
        raise ValueError("COHORT_REQUIRED")
    session = date.fromisoformat(scan["inputs"]["session"])
    decision = datetime.fromisoformat(scan["inputs"]["decision_at"])
    entry_day = next_session(session)
    now = datetime.now(timezone.utc)
    if cohort == "PROSPECTIVE_SHADOW" and max(now, decision) >= datetime.combine(entry_day, time(9, 30), NY):
        raise ValueError("PROSPECTIVE_REGISTRATION_AFTER_REFERENCE_ENTRY")
    observations = list((previous or {}).get("observations", []))
    cohort_dates = [date.fromisoformat(r["signal_date"]) for r in observations if r["cohort"] == cohort]
    if cohort_dates and session < max(cohort_dates):
        raise ValueError("OUT_OF_ORDER_COHORT_REGISTRATION")
    existing = {r["observation_id"] for r in observations}
    for row in scan["rows"]:
        if not row["trigger_positive"]:
            continue
        identity = object_id({"ticker": row["ticker"], "session": session.isoformat(), "cohort": cohort})
        if identity in existing:
            continue
        recent = [r for r in observations if r["ticker"] == row["ticker"] and r["cohort"] == cohort
                  and date.fromisoformat(r["signal_date"]) <= session <= date.fromisoformat(r["independent_until"])]
        parent = recent[0]["independence_cluster"] if recent else identity
        observations.append({"observation_id": identity, "scan_id": scan["scan_id"],
            "cohort": cohort, "registered_at": now.isoformat(), "signal_date": session.isoformat(),
            "decision_at": decision.isoformat(), "ticker": row["ticker"], "setup": row["setup"],
            "legacy_membership": row["legacy_membership"],
            "security_flags": row.get("security_flags", {}),
            "broad_only_confirmed": row["legacy_membership"] == "NO",
            "reference_close": row["reference_close"], "atr_percent": row["atr_percent"],
            "volume": row["volume"], "avg_dollar_volume_20": row["avg_dollar_volume_20"],
            "quality_score": row["quality_score"], "rank_score": row["rank_score"],
            "stages": row["stages"], "final_status": row["final_status"],
            "entry_method": "NEXT_SCHEDULED_SESSION_OPEN_NOT_NEXT_AVAILABLE_BAR",
            "entry_session": entry_day.isoformat(), "entry_price": None,
            "independence_cluster": parent, "independent_observation": not recent,
            "independent_until": recent[0]["independent_until"] if recent else horizon_sessions(session)[-1].isoformat(),
            "future_status": "PENDING", "not_trade_advice": True})
    return {"schema": 1, "observations": observations,
            "independence_policy": "ONE_PRIMARY_OBSERVATION_PER_TICKER_PER_30_SESSION_WINDOW_WITHIN_COHORT",
            "scope": "ALL_BROAD_TRIGGERS_INCLUDING_FAIL_AND_UNKNOWN;BROAD_ONLY_REQUIRES_VERIFIED_LEGACY"}


def measure(observation: dict, obj: dict | None, completed: date) -> dict:
    days = horizon_sessions(date.fromisoformat(observation["signal_date"]))
    result = {"observation_id": observation["observation_id"], "ticker": observation["ticker"],
              "independent_observation": observation["independent_observation"],
              "cohort": observation["cohort"], "horizons": {}}
    df = rows_frame(obj["rows"]) if obj else None
    for n in (5, 10, 20, 30):
        period = days[:n]
        end = period[-1]
        if end > completed:
            result["horizons"][str(n)] = {"status": "PENDING", "maturity_session": end.isoformat()}
            continue
        if df is None or any(pd.Timestamp(d) not in df.index for d in period):
            result["horizons"][str(n)] = {"status": "MISSING_DATA", "maturity_session": end.isoformat()}
            continue
        sample = df.loc[[pd.Timestamp(d) for d in period]]
        entry, close = float(sample.Open.iloc[0]), float(sample.Close.iloc[-1])
        if entry <= 0 or (sample.Volume <= 0).any():
            result["horizons"][str(n)] = {"status": "UNRELIABLE_FILL_PROXY"}
            continue
        result["horizons"][str(n)] = {"status": "MATURE", "maturity_session": end.isoformat(),
            "entry_adjusted_open": entry, "entry_session": period[0].isoformat(),
            "exit_adjusted_close": close, "forward_return_pct": 100*(close/entry-1),
            "mfe_pct": 100*max(0, float(sample.High.max())/entry-1),
            "mae_pct": 100*min(0, float(sample.Low.min())/entry-1),
            "net_return_pct_by_roundtrip_bps": {str(bps): 100*(close*(1-bps/20000)/(entry*(1+bps/20000))-1)
                                               for bps in (20, 50, 100)},
            "bar_object_id": object_id(obj), "return_basis": "YAHOO_SPLIT_DIVIDEND_ADJUSTED_PROXY",
            "execution_warning": "OPEN_IS_REFERENCE_NOT_GUARANTEED_MANUAL_FILL;NO_IMPACT_OR_SPREAD_MEASUREMENT"}
    return result


def forward(book: dict, preparation: dict, cache: Path) -> dict:
    completed = min(date.fromisoformat(preparation["session"]), last_completed(datetime.now(timezone.utc)))
    objects = {r["symbol"]: read_object(cache, r["object_id"]) for r in preparation["outcomes"] if r.get("object_id")}
    results = [measure(r, objects.get(r["ticker"]), completed) for r in book["observations"]]
    return {"schema": 1, "book_id": object_id(book), "preparation_id": object_id(preparation),
            "evaluated_at": utc_now(), "completed_session": completed.isoformat(), "results": results,
            "status_counts": dict(Counter(h["status"] for r in results for h in r["horizons"].values()))}
