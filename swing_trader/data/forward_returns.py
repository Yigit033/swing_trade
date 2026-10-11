"""
Live Signal Forward-Return Tracking — the system's continuous feedback loop.

Every live scan signal is recorded here; after 3/5/10 trading days the actual
forward returns (R3/R5/R10, MFE/MAE) are filled in automatically. This lets us
compare LIVE performance against the measured harness edge (VCE: R10 ≈ +5-8%
over benchmark) every week — so we never fly blind again.

Entry convention matches the harness AND the live PENDING mechanic exactly:
    entry = next trading day's OPEN after the signal date
    R_N   = close of the N-th bar after entry (entry day = bar 1) / entry - 1
    MFE/MAE = max high / min low over the first 10 bars from entry

Dual-mode storage (SQLite local / PostgreSQL Supabase), same pattern as
signal_history_storage.py.
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL")
DB_PATH = Path(__file__).parent.parent.parent / "data" / "signal_history.db"

HORIZONS = (3, 5, 10)
MFE_WINDOW = 10
EDGE_POLICY_VERSION = "regime-quality-v1"

# Legacy rows pre-date persisted effective thresholds.  These fallbacks mirror
# the current conservative policy; new rows always store the exact threshold
# used by the scan, so future settings changes do not rewrite history.
REGIME_QUALITY_FLOORS = {
    "BULL": 78.0,
    "CAUTION": 80.0,
    "BEAR": 80.0,
    "UNKNOWN": 80.0,
}


def _url_mode() -> str:
    if DATABASE_URL and DATABASE_URL.startswith(("postgresql", "postgres")):
        return "pg"
    return "sqlite"


_MODE = _url_mode()


def _connect():
    if _MODE == "pg":
        from swing_trader.utils.pg_connect import connect as pg_connect

        return pg_connect(DATABASE_URL)
    import sqlite3

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _ph() -> str:
    return "%s" if _MODE == "pg" else "?"


_CREATE_PG = """
CREATE TABLE IF NOT EXISTS signal_forward_returns (
    id            SERIAL PRIMARY KEY,
    created_at    TEXT NOT NULL,
    updated_at    TEXT,
    run_id        TEXT,
    ticker        TEXT NOT NULL,
    signal_date   TEXT NOT NULL,
    quality       REAL,
    swing_type    TEXT,
    pathway       TEXT,
    regime        TEXT,
    entry_open    REAL,
    r3            REAL,
    r5            REAL,
    r10           REAL,
    mfe10         REAL,
    mae10         REAL,
    status        TEXT DEFAULT 'pending',
    kind          TEXT NOT NULL DEFAULT 'signal',
    reject_reason TEXT,
    effective_min_quality REAL,
    is_actionable INTEGER,
    policy_version TEXT,
    UNIQUE (ticker, signal_date)
)
"""

_CREATE_SQLITE = _CREATE_PG.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")

KIND_SIGNAL = "signal"
KIND_UNIVERSE = "universe"


def _migrate_forward_returns_schema(cur) -> None:
    """Add telemetry columns on existing DBs. Unique stays ticker + session."""
    if _MODE == "pg":
        for name, sql_type in (
            ("kind", "TEXT"),
            ("reject_reason", "TEXT"),
            ("effective_min_quality", "REAL"),
            ("is_actionable", "INTEGER"),
            ("policy_version", "TEXT"),
        ):
            cur.execute(
                f"ALTER TABLE signal_forward_returns ADD COLUMN IF NOT EXISTS {name} {sql_type}"
            )
        cur.execute(
            "UPDATE signal_forward_returns SET kind = %s WHERE kind IS NULL OR kind = ''",
            (KIND_SIGNAL,),
        )
        return
    cur.execute("PRAGMA table_info(signal_forward_returns)")
    cols = {row[1] for row in cur.fetchall()}
    if "kind" not in cols:
        cur.execute(
            "ALTER TABLE signal_forward_returns ADD COLUMN kind TEXT NOT NULL DEFAULT 'signal'"
        )
    if "reject_reason" not in cols:
        cur.execute("ALTER TABLE signal_forward_returns ADD COLUMN reject_reason TEXT")
    if "effective_min_quality" not in cols:
        cur.execute("ALTER TABLE signal_forward_returns ADD COLUMN effective_min_quality REAL")
    if "is_actionable" not in cols:
        cur.execute("ALTER TABLE signal_forward_returns ADD COLUMN is_actionable INTEGER")
    if "policy_version" not in cols:
        cur.execute("ALTER TABLE signal_forward_returns ADD COLUMN policy_version TEXT")


def _policy_floor(row: Dict[str, Any]) -> float:
    persisted = row.get("effective_min_quality")
    if persisted is not None:
        return float(persisted)
    return REGIME_QUALITY_FLOORS.get(
        str(row.get("regime") or "UNKNOWN").upper(),
        REGIME_QUALITY_FLOORS["UNKNOWN"],
    )


def _is_actionable(row: Dict[str, Any]) -> bool:
    persisted = row.get("is_actionable")
    if persisted is not None:
        return bool(persisted)
    quality = row.get("quality")
    return quality is not None and float(quality) >= _policy_floor(row)


def _summary(values: Sequence[float]) -> Optional[Dict[str, Any]]:
    """Distribution summary with a small-sample Student-t confidence interval."""
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    result: Dict[str, Any] = {
        "n": n,
        "mean": round(mean, 2),
        "median": round(median(vals), 2),
        "win_rate": round(sum(v > 0 for v in vals) / n * 100, 1),
        "best": round(max(vals), 2),
        "worst": round(min(vals), 2),
        "ci95_low": None,
        "ci95_high": None,
        "standard_error": None,
        "p_value_positive": None,
    }
    if n < 2:
        return result
    variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
    se = math.sqrt(variance / n)
    if se == 0:
        result.update({
            "ci95_low": round(mean, 2),
            "ci95_high": round(mean, 2),
            "standard_error": 0.0,
            "p_value_positive": 0.0 if mean > 0 else 1.0,
        })
        return result
    try:
        from scipy.stats import t as student_t

        critical = float(student_t.ppf(0.975, n - 1))
        p_positive = float(student_t.sf(mean / se, n - 1))
    except Exception:
        critical = 1.96
        p_positive = None
    result.update({
        "ci95_low": round(mean - critical * se, 2),
        "ci95_high": round(mean + critical * se, 2),
        "standard_error": round(se, 2),
        "p_value_positive": round(p_positive, 4) if p_positive is not None else None,
    })
    return result


def _side(pool: Sequence[Dict[str, Any]], key: str = "r10") -> Dict[str, Any]:
    return _summary([r[key] for r in pool if r.get(key) is not None]) or {
        "n": 0,
        "mean": None,
        "median": None,
        "win_rate": None,
        "best": None,
        "worst": None,
        "ci95_low": None,
        "ci95_high": None,
        "standard_error": None,
        "p_value_positive": None,
    }


def _clustered_by(pool: Sequence[Dict[str, Any]], field: str) -> Dict[str, Any]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in pool:
        value = row.get("r10")
        group = row.get(field)
        if value is not None and group:
            grouped[str(group)].append(float(value))
    values = [sum(vs) / len(vs) for vs in grouped.values()]
    result = _summary(values) or _side([])
    result["groups"] = len(grouped)
    return result


def build_edge_report(recs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build statistically honest live-edge telemetry from database records.

    The policy split uses the threshold that was active when available. Legacy
    rows are evaluated against today's regime floor and explicitly counted.
    Same-date excess return avoids comparing signals from one market regime to
    universe rows from a different date.
    """
    signal_recs = [r for r in recs if (r.get("kind") or KIND_SIGNAL) != KIND_UNIVERSE]
    universe_recs = [r for r in recs if (r.get("kind") or KIND_SIGNAL) == KIND_UNIVERSE]
    eligible = [r for r in signal_recs if _is_actionable(r)]
    excluded = [r for r in signal_recs if not _is_actionable(r)]

    universe_by_date: Dict[str, List[float]] = defaultdict(list)
    for row in universe_recs:
        if row.get("signal_date") and row.get("r10") is not None:
            universe_by_date[str(row["signal_date"])].append(float(row["r10"]))

    matched_rows: List[Dict[str, Any]] = []
    for row in eligible:
        controls = universe_by_date.get(str(row.get("signal_date")), [])
        if row.get("r10") is None or not controls:
            continue
        control_mean = sum(controls) / len(controls)
        matched_rows.append({
            "signal_date": row.get("signal_date"),
            "excess": float(row["r10"]) - control_mean,
            "control_n": len(controls),
        })
    matched_by_date: Dict[str, List[float]] = defaultdict(list)
    for row in matched_rows:
        matched_by_date[str(row["signal_date"])].append(float(row["excess"]))
    matched_daily = [sum(vs) / len(vs) for vs in matched_by_date.values()]
    matched_summary = _summary(matched_daily) or _side([])
    matched_summary.update({
        "matched_signals": len(matched_rows),
        "matched_dates": len(matched_daily),
        "method": "Her sinyal R10 eksi aynı tarama tarihindeki evren ortalama R10; tarihler eşit ağırlıklı.",
    })

    eligible_mature = [r for r in eligible if r.get("r10") is not None]
    by_ticker = _clustered_by(eligible_mature, "ticker")
    by_date = _clustered_by(eligible_mature, "signal_date")
    without_best = list(eligible_mature)
    if without_best:
        best_idx = max(range(len(without_best)), key=lambda i: float(without_best[i]["r10"]))
        without_best.pop(best_idx)
    eligible_sum = sum(float(r["r10"]) for r in eligible_mature)
    top_trade = max((float(r["r10"]) for r in eligible_mature), default=0.0)
    concentration = {
        "unique_tickers": len({r.get("ticker") for r in eligible_mature if r.get("ticker")}),
        "unique_dates": len({r.get("signal_date") for r in eligible_mature if r.get("signal_date")}),
        "equal_ticker_weighted": by_ticker,
        "date_clustered": by_date,
        "without_best_trade": _side(without_best),
        "top_trade_contribution_pct": (
            round(top_trade / eligible_sum * 100, 1) if eligible_sum > 0 else None
        ),
    }

    eligible_summary = _side(eligible)
    ci_low = eligible_summary.get("ci95_low")
    if eligible_summary["n"] < 30 or by_date.get("groups", 0) < 10:
        status = "early"
        label = "Umut verici, örneklem küçük"
    elif ci_low is not None and ci_low > 0:
        status = "confirmed"
        label = "Canlı edge istatistiksel olarak pozitif"
    elif eligible_summary.get("mean") is not None and eligible_summary["mean"] > 0:
        status = "inconclusive"
        label = "Pozitif ortalama, güven aralığı belirsiz"
    else:
        status = "negative"
        label = "Canlı edge doğrulanmıyor"

    legacy_policy_rows = sum(
        1 for r in signal_recs if r.get("effective_min_quality") is None
    )
    return {
        "signal_recs": signal_recs,
        "universe_recs": universe_recs,
        "eligible": eligible,
        "excluded": excluded,
        "policy_split": {
            "version": EDGE_POLICY_VERSION,
            "floors": REGIME_QUALITY_FLOORS,
            "eligible": eligible_summary,
            "excluded": _side(excluded),
            "legacy_inferred_rows": legacy_policy_rows,
        },
        "matched_edge": matched_summary,
        "robustness": concentration,
        "verdict": {
            "status": status,
            "label": label,
            "minimum_signals": 30,
            "minimum_dates": 10,
        },
    }


def _compute_forward_values(df, signal_date: str) -> Optional[Dict[str, Optional[float]]]:
    """Calculate t+1-open forward returns without silently shifting old rows.

    A missing signal session is allowed only for a nearby non-trading label.
    Previously, a truncated 3-month download could turn its first available bar
    into the entry for a much older signal and corrupt the scorecard.
    """
    import pandas as pd

    dates = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d").tolist()
    if not dates:
        return None
    if signal_date in dates:
        entry_idx = dates.index(signal_date) + 1
    else:
        after = [i for i, date in enumerate(dates) if date > signal_date]
        if not after:
            return None
        entry_idx = after[0]
        gap_days = (
            datetime.strptime(dates[entry_idx], "%Y-%m-%d")
            - datetime.strptime(signal_date[:10], "%Y-%m-%d")
        ).days
        if gap_days > 7:
            return None
    if entry_idx >= len(df):
        return None

    o = df["Open"].astype(float).values
    c = df["Close"].astype(float).values
    h = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    entry = float(o[entry_idx])
    if not math.isfinite(entry) or entry <= 0:
        return None

    vals: Dict[str, Optional[float]] = {"entry_open": round(entry, 4)}
    for n in HORIZONS:
        j = entry_idx + n - 1
        value = float(c[j]) if j < len(df) else None
        vals[f"r{n}"] = (
            round((value / entry - 1) * 100, 2)
            if value is not None and math.isfinite(value)
            else None
        )
    end = min(entry_idx + MFE_WINDOW, len(df))
    high = float(h[entry_idx:end].max())
    low_value = float(low[entry_idx:end].min())
    vals["mfe10"] = round((high / entry - 1) * 100, 2) if math.isfinite(high) else None
    vals["mae10"] = round((low_value / entry - 1) * 100, 2) if math.isfinite(low_value) else None
    return vals


def assemble_scan_membership(
    *,
    universe_tickers: List[str],
    signals: List[Dict[str, Any]],
    outcomes: List[Dict[str, Any]],
    fallback_date: str,
) -> List[Dict[str, Any]]:
    """One snapshot row per scanned ticker for history JSON (compact, no OHLCV)."""
    by_ticker = {o.get("ticker"): o for o in outcomes if o.get("ticker")}
    sig_by_ticker = {s.get("ticker"): s for s in signals if s.get("ticker")}
    members: List[Dict[str, Any]] = []
    seen = set()
    for ticker in universe_tickers:
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        sig = sig_by_ticker.get(ticker)
        if sig:
            date = (sig.get("date") or sig.get("signal_date") or fallback_date)[:10]
            members.append({
                "ticker": ticker,
                "kind": KIND_SIGNAL,
                "date": date,
                "quality": sig.get("quality_score"),
                "reject_reason": None,
                "pathway": sig.get("trigger_pathway")
                or (sig.get("trigger_details") or {}).get("trigger_pathway"),
            })
            continue
        out = by_ticker.get(ticker) or {}
        date = (out.get("date") or fallback_date or "")[:10]
        members.append({
            "ticker": ticker,
            "kind": KIND_UNIVERSE,
            "date": date,
            "quality": out.get("quality"),
            "reject_reason": out.get("reject_reason") or "unknown",
            "pathway": None,
        })
    return members


class ForwardReturnTracker:
    def __init__(self):
        self._init_db()
        logger.info("ForwardReturnTracker initialized (mode=%s)", _MODE)

    def _init_db(self) -> None:
        try:
            conn = _connect()
            cur = conn.cursor()
            cur.execute(_CREATE_PG if _MODE == "pg" else _CREATE_SQLITE)
            _migrate_forward_returns_schema(cur)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_fwd_status ON signal_forward_returns(status)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_fwd_kind_status ON signal_forward_returns(kind, status)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_fwd_signal_date ON signal_forward_returns(signal_date)"
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("forward_returns table init failed: %s", e)

    # ------------------------------------------------------------------
    # RECORD — called right after each live scan persists its signals
    # ------------------------------------------------------------------
    def record_signals(
        self,
        run_id: Optional[str],
        signals: List[Dict[str, Any]],
        *,
        effective_min_quality: Optional[float] = None,
        policy_version: str = EDGE_POLICY_VERSION,
    ) -> int:
        """Insert one tracking row per signal. Idempotent on (ticker, signal_date)."""
        if not signals:
            return 0
        ph = _ph()
        inserted = 0
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        conn = None
        try:
            conn = _connect()
            cur = conn.cursor()
            for s in signals:
                ticker = s.get("ticker")
                sig_date = (s.get("signal_date") or s.get("date") or now)[:10]
                if not ticker:
                    continue
                quality = float(s.get("quality_score") or 0)
                threshold = (
                    float(effective_min_quality)
                    if effective_min_quality is not None
                    else REGIME_QUALITY_FLOORS.get(
                        str(s.get("market_regime") or "UNKNOWN").upper(),
                        REGIME_QUALITY_FLOORS["UNKNOWN"],
                    )
                )
                actionable = 1 if quality >= threshold else 0
                try:
                    if _MODE == "pg":
                        cur.execute(
                            f"""INSERT INTO signal_forward_returns
                                (created_at, run_id, ticker, signal_date, quality,
                                 swing_type, pathway, regime, status, kind,
                                 effective_min_quality, is_actionable, policy_version)
                                VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},'pending',{ph},{ph},{ph},{ph})
                                ON CONFLICT (ticker, signal_date) DO NOTHING""",
                            (
                                now, run_id, ticker, sig_date,
                                quality,
                                s.get("swing_type"),
                                s.get("trigger_pathway")
                                or (s.get("trigger_details") or {}).get("trigger_pathway"),
                                s.get("market_regime"),
                                KIND_SIGNAL,
                                threshold,
                                actionable,
                                policy_version,
                            ),
                        )
                    else:
                        cur.execute(
                            f"""INSERT OR IGNORE INTO signal_forward_returns
                                (created_at, run_id, ticker, signal_date, quality,
                                 swing_type, pathway, regime, status, kind,
                                 effective_min_quality, is_actionable, policy_version)
                                VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},'pending',{ph},{ph},{ph},{ph})""",
                            (
                                now, run_id, ticker, sig_date,
                                quality,
                                s.get("swing_type"),
                                s.get("trigger_pathway")
                                or (s.get("trigger_details") or {}).get("trigger_pathway"),
                                s.get("market_regime"),
                                KIND_SIGNAL,
                                threshold,
                                actionable,
                                policy_version,
                            ),
                        )
                    inserted += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
                except Exception as e:
                    logger.debug("record_signals row failed (%s): %s", ticker, e)
            conn.commit()
            if inserted:
                logger.info("ForwardReturnTracker: %d new signal(s) queued for tracking", inserted)
            return inserted
        except Exception as e:
            logger.error("record_signals failed: %s", e)
            return 0
        finally:
            if conn:
                conn.close()

    def record_universe(
        self,
        run_id: Optional[str],
        members: List[Dict[str, Any]],
        *,
        regime: Optional[str] = None,
    ) -> int:
        """Queue scanned-but-no-signal names. Same t+1 / R3-R10 fill as signals.

        Skips tickers already stored that day (unique ticker+date) — call AFTER
        record_signals so a real signal is never overwritten by a universe row.
        Quality is stored only when the engine actually scored (e.g. type floor).
        """
        if not members:
            return 0
        ph = _ph()
        inserted = 0
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        conn = None
        try:
            conn = _connect()
            cur = conn.cursor()
            for m in members:
                if m.get("kind") == KIND_SIGNAL:
                    continue
                ticker = m.get("ticker")
                sig_date = (m.get("date") or now)[:10]
                if not ticker or not sig_date:
                    continue
                q = m.get("quality")
                quality = float(q) if q is not None else None
                try:
                    if _MODE == "pg":
                        cur.execute(
                            f"""INSERT INTO signal_forward_returns
                                (created_at, run_id, ticker, signal_date, quality,
                                 swing_type, pathway, regime, status, kind, reject_reason)
                                VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},'pending',{ph},{ph})
                                ON CONFLICT (ticker, signal_date) DO NOTHING""",
                            (
                                now, run_id, ticker, sig_date, quality,
                                None, None, regime, KIND_UNIVERSE,
                                m.get("reject_reason"),
                            ),
                        )
                    else:
                        cur.execute(
                            f"""INSERT OR IGNORE INTO signal_forward_returns
                                (created_at, run_id, ticker, signal_date, quality,
                                 swing_type, pathway, regime, status, kind, reject_reason)
                                VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},'pending',{ph},{ph})""",
                            (
                                now, run_id, ticker, sig_date, quality,
                                None, None, regime, KIND_UNIVERSE,
                                m.get("reject_reason"),
                            ),
                        )
                    inserted += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
                except Exception as e:
                    logger.debug("record_universe row failed (%s): %s", ticker, e)
            conn.commit()
            if inserted:
                logger.info(
                    "ForwardReturnTracker: %d universe name(s) queued for tracking",
                    inserted,
                )
            return inserted
        except Exception as e:
            logger.error("record_universe failed: %s", e)
            return 0
        finally:
            if conn:
                conn.close()

    # ------------------------------------------------------------------
    # UPDATE — fill matured forward returns (called after each scan / daily)
    # ------------------------------------------------------------------
    def update_pending(self, fetcher=None, max_tickers: int = 500) -> int:
        """
        For every non-complete row, fetch price history and fill R3/R5/R10 +
        MFE/MAE as bars become available. Marks 'complete' once R10 is known.
        Returns number of rows updated.
        """
        ph = _ph()
        conn = None
        try:
            conn = _connect()
            cur = conn.cursor()
            cur.execute(
                "SELECT id, ticker, signal_date FROM signal_forward_returns "
                "WHERE status != 'complete' "
                "ORDER BY CASE WHEN COALESCE(kind, 'signal') = 'signal' THEN 0 ELSE 1 END, "
                "signal_date ASC"
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.error("update_pending select failed: %s", e)
            if conn:
                conn.close()
            return 0

        if not rows:
            if conn:
                conn.close()
            return 0

        if fetcher is None:
            from swing_trader.data.fetcher import DataFetcher

            fetcher = DataFetcher()

        selected = rows[:max_tickers]
        tickers = list(dict.fromkeys(
            (row[1] if not hasattr(row, "keys") else row["ticker"])
            for row in selected
        ))
        signal_dates = [
            str(row[2] if not hasattr(row, "keys") else row["signal_date"])
            for row in selected
        ]
        earliest = min(signal_dates)
        start_date = (
            datetime.strptime(earliest[:10], "%Y-%m-%d") - timedelta(days=7)
        ).strftime("%Y-%m-%d")
        end_date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            data_by_ticker = fetcher.fetch_multiple_stocks_batch(
                tickers,
                start_date=start_date,
                end_date=end_date,
                chunk_size=40,
                min_rows=5,
            )
        except Exception as e:
            logger.warning("ForwardReturnTracker batch fetch failed: %s", e)
            data_by_ticker = {}

        updated = 0
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        for row in selected:
            rid = row[0] if not hasattr(row, "keys") else row["id"]
            ticker = row[1] if not hasattr(row, "keys") else row["ticker"]
            sig_date = row[2] if not hasattr(row, "keys") else row["signal_date"]
            try:
                df = data_by_ticker.get(ticker)
                if df is None or len(df) < 5 or "Date" not in df.columns:
                    continue
                vals = _compute_forward_values(df, str(sig_date))
                if vals is None:
                    continue
                status = "complete" if vals.get("r10") is not None else "partial"

                cur = conn.cursor()
                cur.execute(
                    f"""UPDATE signal_forward_returns
                        SET entry_open={ph}, r3={ph}, r5={ph}, r10={ph},
                            mfe10={ph}, mae10={ph}, status={ph}, updated_at={ph}
                        WHERE id={ph}""",
                    (
                        vals["entry_open"], vals.get("r3"), vals.get("r5"), vals.get("r10"),
                        vals.get("mfe10"), vals.get("mae10"), status, now, rid,
                    ),
                )
                updated += 1
            except Exception as e:
                logger.debug("update_pending %s failed: %s", ticker, e)

        conn.commit()
        conn.close()
        if updated:
            logger.info("ForwardReturnTracker: %d signal(s) updated", updated)
        return updated

    # ------------------------------------------------------------------
    # STATS — live edge vs harness expectation
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        conn = None
        try:
            conn = _connect()
            cur = conn.cursor()
            cur.execute(
                "SELECT ticker, signal_date, quality, swing_type, pathway, regime, "
                "entry_open, r3, r5, r10, mfe10, mae10, status, "
                "COALESCE(kind, 'signal') AS kind, reject_reason, "
                "effective_min_quality, is_actionable, policy_version "
                "FROM signal_forward_returns ORDER BY signal_date DESC"
            )
            rows = cur.fetchall()
            cols = ["ticker", "signal_date", "quality", "swing_type", "pathway", "regime",
                    "entry_open", "r3", "r5", "r10", "mfe10", "mae10", "status",
                    "kind", "reject_reason", "effective_min_quality", "is_actionable",
                    "policy_version"]
            recs = [dict(zip(cols, r)) for r in rows]
        except Exception as e:
            logger.error("get_stats failed: %s", e)
            return {"n_tracked": 0, "signals": [], "aggregates": {}}
        finally:
            if conn:
                conn.close()

        report = build_edge_report(recs)
        signal_recs = report["signal_recs"]
        universe_recs = report["universe_recs"]

        def _agg(pool: List[Dict[str, Any]], key: str) -> Optional[Dict[str, float]]:
            vals = [row[key] for row in pool if row.get(key) is not None]
            if not vals:
                return None
            vals_sorted = sorted(vals)
            mid = len(vals) // 2
            median = vals_sorted[mid] if len(vals) % 2 else (vals_sorted[mid - 1] + vals_sorted[mid]) / 2
            return {
                "n": len(vals),
                "mean": round(sum(vals) / len(vals), 2),
                "median": round(median, 2),
                "win_rate": round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1),
            }

        # ── Kalite kovaları ──────────────────────────────────────────────
        # "Skor gerçekten ayırt ediyor mu?" sorusunun CANLI cevabı. Eşik-altı
        # sinyaller bilerek kaydediliyor (scanner.py raw signals) — onlar
        # olmadan eşiğin doğru yerde olduğunu ölçmek imkânsız olurdu.
        def _bucket(lo: float, hi: float, horizon: str = "r10") -> Dict[str, Any]:
            vals = [
                r[horizon] for r in signal_recs
                if r.get("quality") is not None
                and lo <= r["quality"] < hi
                and r.get(horizon) is not None
            ]
            pending = sum(
                1 for r in signal_recs
                if r.get("quality") is not None
                and lo <= r["quality"] < hi
                and r.get(horizon) is None
            )
            if not vals:
                return {"label": f"Q{int(lo)}-{int(hi) - 1}", "n": 0, "pending": pending,
                        "mean": None, "win_rate": None}
            return {
                "label": f"Q{int(lo)}-{int(hi) - 1}",
                "n": len(vals),
                "pending": pending,
                "mean": round(sum(vals) / len(vals), 2),
                "win_rate": round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1),
            }

        # Eşik referansı: rejime göre 78-82 arasında oynuyor; 80 temsili kesim.
        REF_THRESHOLD = 80.0

        def _side(above: bool, horizon: str = "r10") -> Dict[str, Any]:
            vals = [
                r[horizon] for r in signal_recs
                if r.get("quality") is not None
                and ((r["quality"] >= REF_THRESHOLD) if above else (r["quality"] < REF_THRESHOLD))
                and r.get(horizon) is not None
            ]
            if not vals:
                return {"n": 0, "mean": None, "win_rate": None, "best": None, "worst": None}
            return {
                "n": len(vals),
                "mean": round(sum(vals) / len(vals), 2),
                "win_rate": round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1),
                "best": round(max(vals), 2),
                "worst": round(min(vals), 2),
            }

        def _r10_side(pool: List[Dict[str, Any]]) -> Dict[str, Any]:
            vals = [r["r10"] for r in pool if r.get("r10") is not None]
            if not vals:
                return {"n": 0, "mean": None, "win_rate": None, "best": None, "worst": None}
            return {
                "n": len(vals),
                "mean": round(sum(vals) / len(vals), 2),
                "win_rate": round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1),
                "best": round(max(vals), 2),
                "worst": round(min(vals), 2),
            }

        reason_stats: List[Dict[str, Any]] = []
        for reason in sorted({(r.get("reject_reason") or "unknown") for r in universe_recs}):
            pool = [r for r in universe_recs if (r.get("reject_reason") or "unknown") == reason]
            r10 = _r10_side(pool)
            reason_stats.append({
                "reason": reason,
                "n": len(pool),
                "pending": sum(1 for r in pool if r.get("r10") is None),
                "n_mature": r10["n"],
                "mean": r10["mean"],
                "win_rate": r10["win_rate"],
            })
        reason_stats.sort(key=lambda x: -x["n"])

        return {
            "n_tracked": len(signal_recs),
            "n_complete": sum(1 for r in signal_recs if r.get("status") == "complete"),
            "aggregates": {f"r{n}": _agg(signal_recs, f"r{n}") for n in HORIZONS},
            "mfe10": _agg(signal_recs, "mfe10"),
            "mae10": _agg(signal_recs, "mae10"),
            "quality_buckets": [
                _bucket(0, 60), _bucket(60, 70), _bucket(70, 78),
                _bucket(78, 80), _bucket(80, 101),
            ],
            "threshold_split": {
                "reference": REF_THRESHOLD,
                "above": _side(True),
                "below": _side(False),
            },
            "policy_split": report["policy_split"],
            "matched_edge": report["matched_edge"],
            "robustness": report["robustness"],
            "verdict": report["verdict"],
            "harness_expectation": {
                "r10_mean": "Pathway ve sürüme göre değişir; canlı politika kohortu esas alınır.",
                "kaynak": "PIPELINE_MAP.md + GATE_AUDIT.md; canlı sonuçlar bu ekranda ayrıca ölçülür.",
            },
            "signals": signal_recs[:150],
            "universe": {
                "n_tracked": len(universe_recs),
                "n_complete": sum(1 for r in universe_recs if r.get("status") == "complete"),
                "aggregates": {f"r{n}": _agg(universe_recs, f"r{n}") for n in HORIZONS},
                "r10": _r10_side(universe_recs),
                "reject_reasons": reason_stats,
            },
            "cohort_split": {
                "signal": _r10_side(signal_recs),
                "universe": _r10_side(universe_recs),
            },
            "universe_rows": universe_recs[:250],
        }


_tracker: Optional[ForwardReturnTracker] = None


def get_forward_tracker() -> ForwardReturnTracker:
    global _tracker
    if _tracker is None:
        _tracker = ForwardReturnTracker()
    return _tracker
