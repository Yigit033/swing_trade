"""
Market Regime History Storage - Dual-mode: SQLite (local dev) + PostgreSQL (production).

Tracks regime changes over time for analysis and dashboard display.
Follows the same pattern as paper_trading/storage.py.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Connection mode ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL")
DB_PATH = Path(__file__).parent.parent.parent / "data" / "regime_history.db"


def _url_mode() -> str:
    if DATABASE_URL and (DATABASE_URL.startswith("postgresql") or DATABASE_URL.startswith("postgres")):
        return "pg"
    return "sqlite"


_MODE = _url_mode()


def _connect():
    global _MODE
    if _MODE == "pg":
        try:
            import psycopg2
            conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
            return conn
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed ({e.__class__.__name__}): {e}")
            raise
    import sqlite3
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _ph():
    return "%s" if _MODE == "pg" else "?"


def _rows_to_dicts(cursor, rows):
    if _MODE == "pg":
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in rows]
    return [dict(row) for row in rows]


# ── CREATE TABLE SQL ─────────────────────────────────────────────────────────
_CREATE_TABLE_PG = """
CREATE TABLE IF NOT EXISTS regime_history (
    id              SERIAL PRIMARY KEY,
    detected_at     TEXT NOT NULL,
    regime          TEXT NOT NULL,
    confidence      TEXT NOT NULL,
    score_multiplier REAL NOT NULL,
    spy_price       REAL,
    ma50            REAL,
    ma200           REAL,
    vix             REAL,
    spy_5d_return   REAL,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS regime_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at     TEXT NOT NULL,
    regime          TEXT NOT NULL,
    confidence      TEXT NOT NULL,
    score_multiplier REAL NOT NULL,
    spy_price       REAL,
    ma50            REAL,
    ma200           REAL,
    vix             REAL,
    spy_5d_return   REAL,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_regime_history_date ON regime_history(detected_at)
"""


class RegimeHistoryStorage:
    """Persist and query market regime history."""

    def __init__(self):
        self._init_db()
        logger.info(f"RegimeHistoryStorage initialized (mode={_MODE})")

    def _init_db(self):
        try:
            conn = _connect()
            cursor = conn.cursor()
            sql = _CREATE_TABLE_PG if _MODE == "pg" else _CREATE_TABLE_SQLITE
            cursor.execute(sql)
            cursor.execute(_CREATE_INDEX)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to init regime_history table: {e}")

    def save_regime(self, regime_data: Dict) -> Optional[int]:
        """
        Save a regime detection result.
        Skips insert if the latest record has the same regime+confidence (no change).
        Returns the row id if inserted, None if skipped.
        """
        regime = regime_data.get('regime', 'UNKNOWN')
        confidence = regime_data.get('confidence', 'TENTATIVE')
        multiplier = regime_data.get('score_multiplier', 1.0)
        spy_price = regime_data.get('spy_price', 0)
        ma50 = regime_data.get('ma50', 0)
        ma200 = regime_data.get('ma200', 0)
        vix = regime_data.get('vix', 0)
        spy_5d_return = regime_data.get('spy_5d_return', 0)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        try:
            conn = _connect()
            cursor = conn.cursor()

            # Check if regime changed from last record
            cursor.execute("SELECT regime, confidence FROM regime_history ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                last = dict(row) if _MODE == "sqlite" else dict(zip(['regime', 'confidence'], row))
                if last.get('regime') == regime and last.get('confidence') == confidence:
                    conn.close()
                    return None  # No change, skip

            p = _ph()
            cursor.execute(f"""
                INSERT INTO regime_history
                (detected_at, regime, confidence, score_multiplier, spy_price, ma50, ma200, vix, spy_5d_return)
                VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})
            """, (now, regime, confidence, multiplier, spy_price, ma50, ma200, vix, spy_5d_return))

            conn.commit()
            row_id = cursor.lastrowid
            conn.close()
            logger.info(f"Regime saved: {regime} ({confidence}) multiplier={multiplier}")
            return row_id

        except Exception as e:
            logger.error(f"Failed to save regime: {e}")
            return None

    def get_latest(self) -> Optional[Dict]:
        """Get the most recent regime record."""
        try:
            conn = _connect()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM regime_history ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            if row:
                return _rows_to_dicts(cursor, [row])[0] if _MODE == "pg" else dict(row)
            return None
        except Exception as e:
            logger.error(f"Failed to get latest regime: {e}")
            return None

    def get_history(self, limit: int = 30) -> List[Dict]:
        """Get regime history (most recent first)."""
        try:
            conn = _connect()
            cursor = conn.cursor()
            p = _ph()
            cursor.execute(f"SELECT * FROM regime_history ORDER BY id DESC LIMIT {p}", (limit,))
            rows = cursor.fetchall()
            result = _rows_to_dicts(cursor, rows) if _MODE == "pg" else [dict(r) for r in rows]
            conn.close()
            return result
        except Exception as e:
            logger.error(f"Failed to get regime history: {e}")
            return []
