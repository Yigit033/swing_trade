"""Opt-in, serial/rate-bounded Yahoo research ingestion and immutable JSON cache.

No credentials, fallback providers, production imports with side effects, or
pickle loading. Existing DataFrame OHLCV column conventions are reused. Raw
Yahoo OHLC is already split-adjusted; Adj Close/Close reproduces yf auto_adjust.
Volume stays in Yahoo's split-adjusted share units, NEVER dividend-adjusted.
This is a consistency check, not independent corporate-action certification.
"""
from __future__ import annotations

import json
import logging
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .calendar import sessions, last_completed, VERSION as CALENDAR_VERSION
from .master import digest, write_json, utc_now

MODE = "YAHOO_SPLIT_OHLC_VOLUME__DIVIDEND_RATIO_OHLC_ONLY"
COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Adj Close", "Dividends", "Stock Splits"]
# Largest finite window: regime MA200 plus its 5-session confirmation = 204.
# Equity VCE minimum55 / RVOL52. EMA has no finite exact convergence: scanner
# explicitly uses the production six-calendar-month input window for equities.
INITIAL_DAYS = 400
MIN_EQUITY_BARS = 52  # RVOL can fire at52; VCE individually requires55
SAFE_VALIDATION_ERRORS = {
    "EMPTY_RESPONSE", "MISSING_OHLCV_OR_ADJUSTED_CLOSE", "DUPLICATE_SESSION", "UNSORTED_SESSIONS",
    "NO_COMPLETED_BARS", "NON_SESSION_BAR", "NO_COMPLETED_CASH_SESSION_REFERENCE", "NONFINITE_OHLCV",
    "NONPOSITIVE_PRICE_OR_NEGATIVE_VOLUME", "OHLC_ENVELOPE_INVALID", "INVALID_CORPORATE_ACTION",
    "SUSPICIOUS_ADJUSTMENT_FACTOR", "PROVIDER_SYMBOL_MISMATCH", "PROVIDER_CIRCUIT_STOP",
    "CALENDAR_OUTSIDE_VERIFIED_RANGE",
}


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def object_id(value) -> str:
    return digest(canonical(value))


def store_object(root: Path, value: dict) -> str:
    key = object_id(value)
    target = root / "objects" / f"{key}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != value:
            raise ValueError("OBJECT_COLLISION")
    else:
        write_json(target, value)
    return key


def read_object(root: Path, key: str) -> dict:
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("INVALID_OBJECT_ID")
    obj = json.loads((root / "objects" / f"{key}.json").read_text(encoding="utf-8"))
    if object_id(obj) != key:
        raise ValueError("CACHE_CHECKSUM_MISMATCH")
    return obj


def rows_frame(rows: list[dict], adjusted: bool = True) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows).set_index("Date")
    df.index = pd.to_datetime(df.index)
    if adjusted:
        ratio = df["Adj Close"] / df["Close"]
        df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].mul(ratio, axis=0)
    return df


def normalize(df: pd.DataFrame, end: date, reference_calendar: bool = False) -> tuple[list[dict], dict]:
    if df is None or df.empty:
        raise ValueError("EMPTY_RESPONSE")
    if not set(COLUMNS[:6]).issubset(df.columns):
        raise ValueError("MISSING_OHLCV_OR_ADJUSTED_CLOSE")
    frame = df.copy()
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_convert("America/New_York").tz_localize(None)
    frame.index = index.normalize()
    if frame.index.duplicated().any():
        raise ValueError("DUPLICATE_SESSION")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("UNSORTED_SESSIONS")
    removed = int(sum(frame.index.date > end))
    frame = frame.loc[frame.index.date <= end].copy()
    if frame.empty:
        raise ValueError("NO_COMPLETED_BARS")
    allowed = set(sessions(frame.index[0].date(), end))
    extra = [x.date().isoformat() for x in frame.index if x.date() not in allowed]
    if extra and not reference_calendar:
        raise ValueError("NON_SESSION_BAR")
    if extra:
        # VIX is a reference INDEX, not a NASDAQ/NYSE equity. Retain the list
        # of excluded non-cash sessions and align only its cash-session close.
        frame = frame.loc[[x.date() in allowed for x in frame.index]].copy()
    if frame.empty:
        raise ValueError("NO_COMPLETED_CASH_SESSION_REFERENCE")
    for c in COLUMNS:
        if c not in frame:
            frame[c] = 0.0
    vals = frame[COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("NONFINITE_OHLCV")
    if (frame[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any().any() or (frame.Volume < 0).any():
        raise ValueError("NONPOSITIVE_PRICE_OR_NEGATIVE_VOLUME")
    if ((frame.High < frame[["Open", "Close", "Low"]].max(axis=1)) |
            (frame.Low > frame[["Open", "Close", "High"]].min(axis=1))).any():
        raise ValueError("OHLC_ENVELOPE_INVALID")
    if (frame["Stock Splits"] < 0).any() or (frame["Dividends"] < 0).any():
        raise ValueError("INVALID_CORPORATE_ACTION")
    ratio = frame["Adj Close"] / frame.Close
    # This is a quarantine rule on provider consistency, NOT a trading filter.
    if ((ratio <= 0) | (ratio > 1.05)).any():
        raise ValueError("SUSPICIOUS_ADJUSTMENT_FACTOR")
    rows = [{"Date": day.date().isoformat(), **{c: float(row[c]) for c in COLUMNS}}
            for day, row in frame.iterrows()]
    missing = sorted(d.isoformat() for d in allowed if d >= frame.index[0].date() and d not in set(frame.index.date))
    return rows, {"future_or_partial_rows_removed": removed, "missing_sessions": missing,
                  "reference_non_cash_sessions_excluded": extra,
                  "split_events": [{"date": r["Date"], "ratio": r["Stock Splits"]}
                                   for r in rows if r["Stock Splits"]],
                  "zero_volume_sessions": int((frame.Volume == 0).sum()),
                  "adjustment_consistency": "RATIO_AND_ENVELOPE_CHECKED_NOT_INDEPENDENTLY_VERIFIED"}


class YahooProvider:
    """Shared pacing for at most two workers; no restriction-bypass mechanism.

    min_interval is conservative pacing, NOT a claimed Yahoo licensed quota.
    Library may make internal auth/timezone requests; report logical calls only.
    """
    def __init__(self, min_interval: float = 1.0):
        if min_interval < .5:
            raise ValueError("MAXIMUM_TWO_LOGICAL_CALLS_PER_SECOND")
        self.interval, self.last_call, self.calls = min_interval, 0.0, 0
        self.lock = threading.Lock()
        self.blocked = threading.Event()

    def history(self, symbol: str, start: date, end: date):
        import yfinance as yf
        with self.lock:
            if self.blocked.is_set():
                raise ValueError("PROVIDER_CIRCUIT_STOP")
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_call)))
            self.last_call = time.monotonic()
            self.calls += 1
        ticker = yf.Ticker(symbol)
        try:
            frame = ticker.history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                                   interval="1d", auto_adjust=False, actions=True, repair=False,
                                   raise_errors=True, timeout=20)
        except Exception as exc:
            if safe_error(exc).endswith("_STOP"):
                self.blocked.set()
            raise
        # Populated by the same history call; never fetch .info here.
        meta = getattr(getattr(ticker, "_price_history", None), "_history_metadata", {}) or {}
        safe = {k: meta.get(k) for k in ("symbol", "exchangeName", "instrumentType", "currency", "firstTradeDate", "exchangeTimezoneName")}
        if safe.get("symbol") and safe["symbol"].upper() != symbol.upper():
            raise ValueError("PROVIDER_SYMBOL_MISMATCH")
        return frame, safe


def safe_error(exc: Exception) -> str:
    # Never persist raw provider messages, cookies, URLs, headers or keys.
    msg = str(exc).lower()
    if "429" in msg or "ratelimit" in type(exc).__name__.lower() or "too many" in msg:
        return "RATE_LIMIT_STOP"
    if "403" in msg or "401" in msg or "forbidden" in msg or "unauthorized" in msg:
        return "ACCESS_DENIED_STOP"
    if isinstance(exc, ValueError) and str(exc) in SAFE_VALIDATION_ERRORS:
        return str(exc)
    return type(exc).__name__


def prepare_one(record: dict, session: date, root: Path, provider, previous: dict | None = None,
                retries: int = 1, sleeper=time.sleep) -> dict:
    symbol = record["symbol"]
    vendor = record.get("yfinance_symbol_candidate")
    base = {"symbol": symbol, "exchange": record["exchange"], "attempts": 0, "retries": 0,
            "errors": [], "object_id": None}
    if not vendor:
        return {**base, "status": "MAPPING_REVIEW_REQUIRED"}
    try:
        old = read_object(root, previous["object_id"]) if previous and previous.get("object_id") else None
    except (OSError, ValueError):
        return {**base, "status": "CACHE_INTEGRITY_REVIEW_REQUIRED", "errors": ["CACHE_INTEGRITY_REVIEW_REQUIRED"]}
    if old and (old["vendor_symbol"] != vendor or old["mode"] != MODE):
        return {**base, "status": "CACHE_IDENTITY_OR_MODE_MISMATCH"}
    if old and old["session_requested"] == session.isoformat() and old["rows"][-1]["Date"] == session.isoformat() and not old["validation"]["missing_sessions"]:
        return {**base, "object_id": previous["object_id"], "status": "CACHE_HIT"}
    initial = session - timedelta(days=30 if symbol == "^VIX" else INITIAL_DAYS)
    start = initial
    if old:
        # Missing tail/internal sessions plus TWO overlap bars detect revised
        # adjustment factors. IPO prefix isn't repeatedly requested as a hole.
        dates = [date.fromisoformat(r["Date"]) for r in old["rows"]]
        holes = [date.fromisoformat(x) for x in old["validation"]["missing_sessions"]]
        start = max(initial, min([dates[max(0, len(dates)-2)], *holes]))
    last_error = "UNKNOWN_ERROR"
    for attempt in range(retries + 1):
        base["attempts"] += 1
        base["retries"] = attempt
        try:
            raw, identity = provider.history(vendor, start, session)
            rows, validation = normalize(raw, session, symbol == "^VIX")
            rebuild = False
            if old:
                old_by_date = {r["Date"]: r for r in old["rows"]}
                for row in rows:
                    prior = old_by_date.get(row["Date"])
                    if prior and any(not math.isclose(row[c], prior[c], rel_tol=1e-7, abs_tol=1e-8) for c in COLUMNS):
                        rebuild = True
                    if not prior and (row["Stock Splits"] or row["Dividends"]):
                        rebuild = True
                if rebuild:
                    base["attempts"] += 1
                    raw, identity = provider.history(vendor, initial, session)
                    rows, validation = normalize(raw, session, symbol == "^VIX")
                else:
                    old_by_date.update({r["Date"]: r for r in rows})
                    removed_partial = validation["future_or_partial_rows_removed"]
                    excluded_reference = validation["reference_non_cash_sessions_excluded"]
                    rows, validation = normalize(rows_frame(sorted(old_by_date.values(), key=lambda x: x["Date"]), False), session)
                    validation["future_or_partial_rows_removed"] = removed_partial
                    validation["reference_non_cash_sessions_excluded"] = excluded_reference
            obj = {"schema": 1, "symbol": symbol, "vendor_symbol": vendor, "source": "Yahoo/yfinance",
                   "mode": MODE, "fetched_at": utc_now(), "session_requested": session.isoformat(),
                   "last_valid_session": rows[-1]["Date"], "requested_start": start.isoformat(),
                   "revision_rebuild": rebuild, "identity": identity,
                   "identity_status": "SYMBOL_MATCH_ONLY" if identity.get("symbol") else "UNKNOWN",
                   "rename_history": "UNKNOWN_NO_PERMANENT_SECURITY_ID",
                   "validation": validation, "rows": rows}
            key = store_object(root, obj)
            status = "OK" if rows[-1]["Date"] == session.isoformat() else "STALE"
            if validation["missing_sessions"]:
                status = "PARTIAL"
            return {**base, "object_id": key, "status": status, "bars": len(rows),
                    "last_valid_session": rows[-1]["Date"], "revision_rebuild": rebuild}
        except Exception as exc:
            last_error = safe_error(exc)
            base["errors"].append(last_error)
            if last_error.endswith("_STOP"):
                break
            if attempt < retries:
                sleeper(5 * (2 ** attempt))
    # Preserve previous immutable revision for diagnostics, never call it fresh.
    return {**base, "object_id": previous.get("object_id") if previous else None, "status": last_error}


def pilot_records(snapshot: dict, count: int = 240, liquidity: dict | None = None) -> tuple[list[dict], dict]:
    """Deterministic exchange x old-dollar-volume strata, includes UNKNOWN.

    Cached May liquidity is for operational sampling ONLY, not a current gate.
    Round-robin hash ordering prevents alphabetically selecting the first300.
    """
    groups = defaultdict(list)
    candidates = [r for r in snapshot["records"] if r["discovery_status"] == "CANDIDATE"]
    for r in candidates:
        dv = (liquidity or {}).get(r["symbol"])
        band = "UNKNOWN" if dv is None else "LT5M" if dv < 5e6 else "5M_25M" if dv < 25e6 else "GE25M"
        groups[(r["exchange"], band)].append(r)
    for key in groups:
        groups[key].sort(key=lambda r: digest(("phase2-pilot-v1:" + r["symbol"]).encode()))
    selected, strata = [], Counter()
    while len(selected) < min(count, len(candidates)):
        for key in sorted(groups):
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
                strata[":".join(key)] += 1
    return selected, {"method": "EXCHANGE_X_STALE_LIQUIDITY_ROUND_ROBIN_HASH_V1",
                      "strata": dict(strata), "liquidity_is_current": False,
                      "selection_not_performance_representative": True}


def prepare(records: list[dict], session: date, root: Path, run_dir: Path, provider=None,
            previous: dict | None = None, metadata: dict | None = None, workers: int = 1) -> dict:
    if workers not in (1, 2):
        raise ValueError("WORKERS_MUST_BE_ONE_OR_TWO")
    if len({r["symbol"] for r in records}) != len(records):
        raise ValueError("DUPLICATE_PLAN_SYMBOL")
    if session > last_completed(datetime.now(timezone.utc)):
        raise ValueError("UNCOMPLETED_SESSION")
    run_dir.mkdir(parents=True, exist_ok=True)
    plan = {"session": session.isoformat(), "symbols": [r["symbol"] for r in records],
            "metadata": metadata or {}, "calendar": CALENDAR_VERSION, "workers": workers}
    plan_path = run_dir / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("RESUME_PLAN_MISMATCH")
    else:
        write_json(plan_path, plan)
    result_path = run_dir / "preparation.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    checkpoint = run_dir / "checkpoints"
    checkpoint.mkdir(exist_ok=True)
    provider = provider or YahooProvider()
    started = time.monotonic()
    outcomes, consecutive_errors, circuit = [], 0, None
    # At most two outstanding tasks: a circuit stop never leaves a 5000-job
    # queue running. Checkpoints are committed in plan order by the main thread.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(records), workers):
            pending = {}
            for i in range(offset, min(offset + workers, len(records))):
                record = records[i]
                path = checkpoint / f"{i:05d}.json"
                if not path.exists() and not circuit:
                    pending[i] = pool.submit(prepare_one, record, session, root, provider,
                                             (previous or {}).get(record["symbol"]))
            for i in range(offset, min(offset + workers, len(records))):
                record = records[i]
                path = checkpoint / f"{i:05d}.json"
                if path.exists():
                    outcome = json.loads(path.read_text())
                    if outcome["symbol"] != record["symbol"]:
                        raise ValueError("CHECKPOINT_IDENTITY_MISMATCH")
                elif i in pending:
                    outcome = pending[i].result()
                    write_json(path, outcome)
                else:
                    outcome = {"symbol": record["symbol"], "exchange": record["exchange"],
                               "status": "NOT_ATTEMPTED_CIRCUIT_OPEN", "circuit": circuit,
                               "attempts": 0, "retries": 0, "errors": [], "object_id": None}
                    write_json(path, outcome)
                success = outcome["status"] in {"OK", "CACHE_HIT", "STALE", "PARTIAL"}
                if success:
                    consecutive_errors = 0
                elif outcome["attempts"]:
                    consecutive_errors += 1
                if not circuit and (outcome["status"].endswith("_STOP") or consecutive_errors >= 5):
                    circuit = outcome["status"]
                outcomes.append(outcome)
                if (i + 1) % 20 == 0 or (circuit and outcome["status"] != "NOT_ATTEMPTED_CIRCUIT_OPEN"):
                    print(json.dumps({"completed": i+1, "total": len(records), "last_status": outcome["status"], "circuit": circuit}), flush=True)
    result = {**plan, "created_at": utc_now(), "outcomes": outcomes, "elapsed_seconds": round(time.monotonic()-started, 3),
              "counts": dict(Counter(o["status"] for o in outcomes)),
              "logical_history_calls": sum(o["attempts"] for o in outcomes),
              "retries": sum(o["retries"] for o in outcomes), "circuit": circuit,
              "cash_data_cost_usd": 0, "internal_http_request_count": "NOT_INSTRUMENTED"}
    write_json(result_path, result)
    return result
