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

# Yeni satır eklemeden önce aynı rejimde en az bu kadar saniye geçmeli (SPY/VIX güncel kalsın)
REGIME_SNAPSHOT_INTERVAL_SEC = int(os.environ.get("REGIME_SNAPSHOT_INTERVAL_SEC", "300"))


def _snapshot_age_seconds(detected_at: Optional[str]) -> float:
    if not detected_at or not str(detected_at).strip():
        return float("inf")
    try:
        s = str(detected_at).strip()[:19]
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return (datetime.now() - dt).total_seconds()
    except ValueError:
        return float("inf")


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
    spy_price       REAL,
    ma50            REAL,
    ma200           REAL,
    vix             REAL,
    spy_5d_return   REAL,
    detect_error    TEXT,
    score_multiplier REAL NOT NULL DEFAULT 1.0,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS regime_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at     TEXT NOT NULL,
    regime          TEXT NOT NULL,
    confidence      TEXT NOT NULL,
    spy_price       REAL,
    ma50            REAL,
    ma200           REAL,
    vix             REAL,
    spy_5d_return   REAL,
    detect_error    TEXT,
    score_multiplier REAL NOT NULL DEFAULT 1.0,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_regime_history_date ON regime_history(detected_at)
"""


def _migrate_add_detect_error(cursor) -> None:
    """Add detect_error for existing deployments (SQLite + PostgreSQL)."""
    if _MODE == "pg":
        cursor.execute(
            "ALTER TABLE regime_history ADD COLUMN IF NOT EXISTS detect_error TEXT"
        )
        return
    cursor.execute("PRAGMA table_info(regime_history)")
    cols = {r[1] for r in cursor.fetchall()}
    if "detect_error" not in cols:
        cursor.execute("ALTER TABLE regime_history ADD COLUMN detect_error TEXT")


def _migrate_ensure_score_multiplier(cursor) -> None:
    """
    Legacy Supabase / older DBs may have score_multiplier NOT NULL; app always
    persists 1.0 (regime no longer scales scores).
    """
    if _MODE == "pg":
        cursor.execute(
            "ALTER TABLE regime_history ADD COLUMN IF NOT EXISTS score_multiplier REAL DEFAULT 1.0"
        )
        return
    cursor.execute("PRAGMA table_info(regime_history)")
    cols = {r[1] for r in cursor.fetchall()}
    if "score_multiplier" not in cols:
        cursor.execute(
            "ALTER TABLE regime_history ADD COLUMN score_multiplier REAL NOT NULL DEFAULT 1.0"
        )


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
            _migrate_add_detect_error(cursor)
            _migrate_ensure_score_multiplier(cursor)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to init regime_history table: {e}")

    def save_regime(self, regime_data: Dict) -> Optional[int]:
        """
        Save a regime detection result.
        Skips insert if the latest row matches regime, confidence, and detect_error
        (so UNKNOWN + new error reason still creates a row; identical spam is avoided).
        Returns the row id if inserted, None if skipped.
        """
        regime = regime_data.get('regime', 'UNKNOWN')
        confidence = regime_data.get('confidence', 'TENTATIVE')
        spy_price = regime_data.get('spy_price', 0)
        ma50 = regime_data.get('ma50', 0)
        ma200 = regime_data.get('ma200', 0)
        vix = regime_data.get('vix', 0)
        spy_5d_return = regime_data.get('spy_5d_return', 0)
        try:
            score_multiplier = float(regime_data.get("score_multiplier", 1.0) or 1.0)
        except (TypeError, ValueError):
            score_multiplier = 1.0
        raw_err = regime_data.get("detect_error")
        detect_error: Optional[str]
        if raw_err is None or (isinstance(raw_err, str) and not raw_err.strip()):
            detect_error = None
        else:
            detect_error = str(raw_err).strip()[:2000]
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        def _norm_err(e: Optional[str]) -> str:
            return (e or "").strip()

        try:
            conn = _connect()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT regime, confidence, detect_error, detected_at FROM regime_history ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                if _MODE == "sqlite":
                    last = dict(row)
                else:
                    last = dict(
                        zip(["regime", "confidence", "detect_error", "detected_at"], row)
                    )
                same_state = (
                    last.get("regime") == regime
                    and last.get("confidence") == confidence
                    and _norm_err(last.get("detect_error")) == _norm_err(detect_error)
                )
                # Önceden: aynı rejimde hiç insert yoktu → dashboard SPY/VIX haftalarca eski kalıyordu.
                if same_state and _snapshot_age_seconds(last.get("detected_at")) < REGIME_SNAPSHOT_INTERVAL_SEC:
                    conn.close()
                    return None

            p = _ph()
            placeholders = ", ".join([p] * 10)
            insert_sql = f"""
                INSERT INTO regime_history
                (detected_at, regime, confidence, spy_price, ma50, ma200, vix, spy_5d_return, detect_error, score_multiplier)
                VALUES ({placeholders})
            """
            params = (
                now,
                regime,
                confidence,
                spy_price,
                ma50,
                ma200,
                vix,
                spy_5d_return,
                detect_error,
                score_multiplier,
            )
            if _MODE == "pg":
                cursor.execute(insert_sql + " RETURNING id", params)
                row_id = cursor.fetchone()[0]
            else:
                cursor.execute(insert_sql, params)
                row_id = cursor.lastrowid

            conn.commit()
            conn.close()
            err_note = f" err={detect_error[:80]}…" if detect_error and len(detect_error) > 80 else (f" err={detect_error}" if detect_error else "")
            logger.info(f"Regime saved: {regime} ({confidence}){err_note}")
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
