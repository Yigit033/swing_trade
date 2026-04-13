"""
SmallCap Signal History Storage - Dual-mode: SQLite (local dev) + PostgreSQL (production).

Persists scanner outputs (signals + stats + thresholds/regime context) so we can:
- audit what the scanner produced at a given time
- segment/analyze historical signal quality vs outcomes
- debug changes to settings / thresholds over time
"""

from __future__ import annotations

import json
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


def _url_mode() -> str:
    if DATABASE_URL and (DATABASE_URL.startswith("postgresql") or DATABASE_URL.startswith("postgres")):
        return "pg"
    return "sqlite"


_MODE = _url_mode()


def _connect():
    if _MODE == "pg":
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        return conn
    import sqlite3
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _ph() -> str:
    return "%s" if _MODE == "pg" else "?"


def _rows_to_dicts(cursor, rows):
    if _MODE == "pg":
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in rows]
    return [dict(r) for r in rows]


_CREATE_TABLE_PG = """
CREATE TABLE IF NOT EXISTS smallcap_signal_runs (
    id                  SERIAL PRIMARY KEY,
    created_at          TEXT NOT NULL,
    job_id              TEXT,
    user_id             UUID,
    portfolio_value     REAL,
    request_min_quality INTEGER,
    request_top_n       INTEGER,
    effective_min_quality INTEGER,
    effective_top_n     INTEGER,
    market_regime       TEXT,
    regime_confidence   TEXT,
    stats_json          TEXT,
    signals_json        TEXT
)
"""

_CREATE_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS smallcap_signal_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    job_id              TEXT,
    user_id             TEXT,
    portfolio_value     REAL,
    request_min_quality INTEGER,
    request_top_n       INTEGER,
    effective_min_quality INTEGER,
    effective_top_n     INTEGER,
    market_regime       TEXT,
    regime_confidence   TEXT,
    stats_json          TEXT,
    signals_json        TEXT
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_smallcap_runs_created_at ON smallcap_signal_runs(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_smallcap_runs_user_id ON smallcap_signal_runs(user_id)",
]


class SmallCapSignalHistoryStorage:
    def __init__(self):
        self._init_db()
        mode_label = "PostgreSQL" if _MODE == "pg" else f"SQLite @ {DB_PATH}"
        logger.info("SmallCapSignalHistoryStorage initialized (mode=%s)", mode_label)

    def _init_db(self) -> None:
        try:
            conn = _connect()
            cur = conn.cursor()
            cur.execute(_CREATE_TABLE_PG if _MODE == "pg" else _CREATE_TABLE_SQLITE)
            for q in _CREATE_INDEXES:
                cur.execute(q)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to init smallcap_signal_runs table: %s", e)

    def save_run(
        self,
        *,
        job_id: Optional[str],
        user_id: Optional[str],
        portfolio_value: float,
        request_min_quality: int,
        request_top_n: int,
        effective_min_quality: int,
        effective_top_n: int,
        market_regime: str,
        regime_confidence: str,
        stats: Dict[str, Any],
        signals: List[Dict[str, Any]],
    ) -> Optional[int]:
        """
        Persist a scanner run. Returns id if inserted.

        Notes:
        - Stores full signals payload as JSON to enable future offline segmentation.
        - `user_id` may be None for anonymous/local usage.
        """
        created_at = datetime.utcnow().isoformat() + "Z"
        try:
            stats_json = json.dumps(stats, ensure_ascii=False, default=str)
            signals_json = json.dumps(signals, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning("Signal run JSON serialization failed: %s", e)
            return None

        ph = _ph()
        conn = None
        try:
            conn = _connect()
            cur = conn.cursor()
            cols = (
                "created_at, job_id, user_id, portfolio_value, "
                "request_min_quality, request_top_n, effective_min_quality, effective_top_n, "
                "market_regime, regime_confidence, stats_json, signals_json"
            )
            vals = ", ".join([ph] * 12)
            q = f"INSERT INTO smallcap_signal_runs ({cols}) VALUES ({vals})"
            params = (
                created_at,
                job_id,
                user_id,
                float(portfolio_value or 0),
                int(request_min_quality or 0),
                int(request_top_n or 0),
                int(effective_min_quality or 0),
                int(effective_top_n or 0),
                str(market_regime or "UNKNOWN"),
                str(regime_confidence or ""),
                stats_json,
                signals_json,
            )
            if _MODE == "pg":
                cur.execute(q + " RETURNING id", params)
                row_id = cur.fetchone()[0]
            else:
                cur.execute(q, params)
                row_id = cur.lastrowid
            conn.commit()
            return int(row_id) if row_id is not None else None
        except Exception as e:
            logger.error("Failed to save signal run: %s", e)
            return None
        finally:
            if conn:
                conn.close()

    def list_runs(self, limit: int = 20, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        ph = _ph()
        conn = None
        try:
            conn = _connect()
            cur = conn.cursor()
            if user_id:
                cur.execute(
                    f"""
                    SELECT id, created_at, job_id, user_id, portfolio_value,
                           request_min_quality, request_top_n, effective_min_quality, effective_top_n,
                           market_regime, regime_confidence
                    FROM smallcap_signal_runs
                    WHERE user_id IS NULL OR user_id = {ph}
                    ORDER BY id DESC
                    LIMIT {ph}
                    """,
                    (user_id, limit),
                )
            else:
                cur.execute(
                    f"""
                    SELECT id, created_at, job_id, user_id, portfolio_value,
                           request_min_quality, request_top_n, effective_min_quality, effective_top_n,
                           market_regime, regime_confidence
                    FROM smallcap_signal_runs
                    ORDER BY id DESC
                    LIMIT {ph}
                    """,
                    (limit,),
                )
            rows = cur.fetchall()
            return _rows_to_dicts(cur, rows)
        except Exception as e:
            logger.error("Failed to list signal runs: %s", e)
            return []
        finally:
            if conn:
                conn.close()

    def get_run(self, run_id: int, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        ph = _ph()
        conn = None
        try:
            conn = _connect()
            cur = conn.cursor()
            if user_id:
                cur.execute(
                    f"SELECT * FROM smallcap_signal_runs WHERE id = {ph} AND (user_id IS NULL OR user_id = {ph})",
                    (run_id, user_id),
                )
            else:
                cur.execute(f"SELECT * FROM smallcap_signal_runs WHERE id = {ph}", (run_id,))
            row = cur.fetchone()
            if not row:
                return None
            out = _rows_to_dicts(cur, [row])[0]
            # Expand JSON fields for API convenience
            try:
                out["stats"] = json.loads(out.get("stats_json") or "{}")
            except Exception:
                out["stats"] = {}
            try:
                out["signals"] = json.loads(out.get("signals_json") or "[]")
            except Exception:
                out["signals"] = []
            return out
        except Exception as e:
            logger.error("Failed to get signal run %s: %s", run_id, e)
            return None
        finally:
            if conn:
                conn.close()

