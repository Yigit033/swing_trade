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
    UNIQUE (ticker, signal_date)
)
"""

_CREATE_SQLITE = _CREATE_PG.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")

KIND_SIGNAL = "signal"
KIND_UNIVERSE = "universe"


def _migrate_forward_returns_schema(cur) -> None:
    """Add kind/reject_reason on existing DBs. Unique stays (ticker, signal_date)."""
    if _MODE == "pg":
        cur.execute(
            "ALTER TABLE signal_forward_returns ADD COLUMN IF NOT EXISTS kind TEXT"
        )
        cur.execute(
            "ALTER TABLE signal_forward_returns ADD COLUMN IF NOT EXISTS reject_reason TEXT"
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
                                 swing_type, pathway, regime, status, kind)
                                VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},'pending',{ph})
                                ON CONFLICT (ticker, signal_date) DO NOTHING""",
                            (
                                now, run_id, ticker, sig_date,
                                float(s.get("quality_score") or 0),
                                s.get("swing_type"),
                                s.get("trigger_pathway")
                                or (s.get("trigger_details") or {}).get("trigger_pathway"),
                                s.get("market_regime"),
                                KIND_SIGNAL,
                            ),
                        )
                    else:
                        cur.execute(
                            f"""INSERT OR IGNORE INTO signal_forward_returns
                                (created_at, run_id, ticker, signal_date, quality,
                                 swing_type, pathway, regime, status, kind)
                                VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},'pending',{ph})""",
                            (
                                now, run_id, ticker, sig_date,
                                float(s.get("quality_score") or 0),
                                s.get("swing_type"),
                                s.get("trigger_pathway")
                                or (s.get("trigger_details") or {}).get("trigger_pathway"),
                                s.get("market_regime"),
                                KIND_SIGNAL,
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
        now = datetime.utcnow().isoformat() + "Z"
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
    def update_pending(self, fetcher=None, max_tickers: int = 40) -> int:
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
                "entry_open, r3, r5, r10, mfe10, mae10, status, "
                "COALESCE(kind, 'signal') AS kind, reject_reason "
                "FROM signal_forward_returns ORDER BY signal_date DESC"
            )
            rows = cur.fetchall()
            cols = ["ticker", "signal_date", "quality", "swing_type", "pathway", "regime",
                    "entry_open", "r3", "r5", "r10", "mfe10", "mae10", "status",
                    "kind", "reject_reason"]
            recs = [dict(zip(cols, r)) for r in rows]
        except Exception as e:
            logger.error("get_stats failed: %s", e)
            return {"n_tracked": 0, "signals": [], "aggregates": {}}
        finally:
            if conn:
                conn.close()

        def _kind_of(r: Dict[str, Any]) -> str:
            k = r.get("kind") or KIND_SIGNAL
            return KIND_UNIVERSE if k == KIND_UNIVERSE else KIND_SIGNAL

        signal_recs = [r for r in recs if _kind_of(r) == KIND_SIGNAL]
        universe_recs = [r for r in recs if _kind_of(r) == KIND_UNIVERSE]

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
            "harness_expectation": {
                "r10_mean": "+7 ile +10% arası (benchmark +2.2%)",
                "r5_win_rate": "55-60%",
                "kaynak": "scripts/measure_signal_edge.py 2026-06-10 (n=123 VCE, OOS t=2.29)",
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
