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
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _url_mode() -> str:
    if DATABASE_URL and DATABASE_URL.startswith(("postgresql", "postgres")):
        return "pg"
    return "sqlite"


_MODE = _url_mode()


def _connect():
    if _MODE == "pg":
        import psycopg2

        return psycopg2.connect(DATABASE_URL, connect_timeout=5)
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
    UNIQUE (ticker, signal_date)
)
"""

_CREATE_SQLITE = _CREATE_PG.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")


class ForwardReturnTracker:
    def __init__(self):
        self._init_db()
        logger.info("ForwardReturnTracker initialized (mode=%s)", _MODE)

    def _init_db(self) -> None:
        try:
            conn = _connect()
            cur = conn.cursor()
            cur.execute(_CREATE_PG if _MODE == "pg" else _CREATE_SQLITE)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_fwd_status ON signal_forward_returns(status)"
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("forward_returns table init failed: %s", e)

    # ------------------------------------------------------------------
    # RECORD — called right after each live scan persists its signals
    # ------------------------------------------------------------------
    def record_signals(self, run_id: Optional[str], signals: List[Dict[str, Any]]) -> int:
        """Insert one tracking row per signal. Idempotent on (ticker, signal_date)."""
        if not signals:
            return 0
        ph = _ph()
        inserted = 0
        now = datetime.utcnow().isoformat() + "Z"
        conn = None
        try:
            conn = _connect()
            cur = conn.cursor()
            for s in signals:
                ticker = s.get("ticker")
                sig_date = (s.get("signal_date") or s.get("date") or now)[:10]
                if not ticker:
                    continue
                try:
                    if _MODE == "pg":
                        cur.execute(
                            f"""INSERT INTO signal_forward_returns
                                (created_at, run_id, ticker, signal_date, quality,
                                 swing_type, pathway, regime, status)
                                VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},'pending')
                                ON CONFLICT (ticker, signal_date) DO NOTHING""",
                            (
                                now, run_id, ticker, sig_date,
                                float(s.get("quality_score") or 0),
                                s.get("swing_type"),
                                s.get("trigger_pathway")
                                or (s.get("trigger_details") or {}).get("trigger_pathway"),
                                s.get("market_regime"),
                            ),
                        )
                    else:
                        cur.execute(
                            f"""INSERT OR IGNORE INTO signal_forward_returns
                                (created_at, run_id, ticker, signal_date, quality,
                                 swing_type, pathway, regime, status)
                                VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},'pending')""",
                            (
                                now, run_id, ticker, sig_date,
                                float(s.get("quality_score") or 0),
                                s.get("swing_type"),
                                s.get("trigger_pathway")
                                or (s.get("trigger_details") or {}).get("trigger_pathway"),
                                s.get("market_regime"),
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

    # ------------------------------------------------------------------
    # UPDATE — fill matured forward returns (called after each scan / daily)
    # ------------------------------------------------------------------
    def update_pending(self, fetcher=None, max_tickers: int = 25) -> int:
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
                "WHERE status != 'complete' ORDER BY signal_date ASC"
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

        import pandas as pd

        updated = 0
        now = datetime.utcnow().isoformat() + "Z"
        for row in rows[:max_tickers]:
            rid = row[0] if not hasattr(row, "keys") else row["id"]
            ticker = row[1] if not hasattr(row, "keys") else row["ticker"]
            sig_date = row[2] if not hasattr(row, "keys") else row["signal_date"]
            try:
                df = fetcher.fetch_stock_data(ticker, period="3mo")
                if df is None or len(df) < 5 or "Date" not in df.columns:
                    continue
                dates = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d").tolist()
                if sig_date not in dates:
                    # signal day older than window or non-trading label — try first bar AFTER it
                    after = [i for i, d in enumerate(dates) if d > sig_date]
                    if not after:
                        continue
                    entry_idx = after[0]
                else:
                    entry_idx = dates.index(sig_date) + 1  # next-day-open entry
                if entry_idx >= len(df):
                    continue  # entry day not traded yet

                o = df["Open"].astype(float).values
                c = df["Close"].astype(float).values
                h = df["High"].astype(float).values
                low = df["Low"].astype(float).values
                entry = float(o[entry_idx])
                if entry <= 0:
                    continue

                vals: Dict[str, Optional[float]] = {"entry_open": round(entry, 4)}
                for n in HORIZONS:
                    j = entry_idx + n - 1
                    vals[f"r{n}"] = round((float(c[j]) / entry - 1) * 100, 2) if j < len(df) else None
                end = min(entry_idx + MFE_WINDOW, len(df))
                vals["mfe10"] = round((float(h[entry_idx:end].max()) / entry - 1) * 100, 2)
                vals["mae10"] = round((float(low[entry_idx:end].min()) / entry - 1) * 100, 2)
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
                conn.commit()
                updated += 1
            except Exception as e:
                logger.debug("update_pending %s failed: %s", ticker, e)

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
                "entry_open, r3, r5, r10, mfe10, mae10, status "
                "FROM signal_forward_returns ORDER BY signal_date DESC"
            )
            rows = cur.fetchall()
            cols = ["ticker", "signal_date", "quality", "swing_type", "pathway", "regime",
                    "entry_open", "r3", "r5", "r10", "mfe10", "mae10", "status"]
            recs = [dict(zip(cols, r)) for r in rows]
        except Exception as e:
            logger.error("get_stats failed: %s", e)
            return {"n_tracked": 0, "signals": [], "aggregates": {}}
        finally:
            if conn:
                conn.close()

        def _agg(key: str) -> Optional[Dict[str, float]]:
            vals = [r[key] for r in recs if r.get(key) is not None]
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

        return {
            "n_tracked": len(recs),
            "n_complete": sum(1 for r in recs if r.get("status") == "complete"),
            "aggregates": {f"r{n}": _agg(f"r{n}") for n in HORIZONS},
            "mfe10": _agg("mfe10"),
            "mae10": _agg("mae10"),
            # Harness reference (v13 measurement) — live numbers should converge here
            "harness_expectation": {
                "r10_mean": "+7 ile +10% arası (benchmark +2.2%)",
                "r5_win_rate": "55-60%",
                "kaynak": "scripts/measure_signal_edge.py 2026-06-10 (n=123 VCE, OOS t=2.29)",
            },
            "signals": recs[:50],
        }


_tracker: Optional[ForwardReturnTracker] = None


def get_forward_tracker() -> ForwardReturnTracker:
    global _tracker
    if _tracker is None:
        _tracker = ForwardReturnTracker()
    return _tracker
