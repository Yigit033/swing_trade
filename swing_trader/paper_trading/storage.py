"""
Paper Trading Storage - Dual-mode: SQLite (local dev) + PostgreSQL (production).

DATABASE_URL env var yoksa SQLite kullanır (local geliştirme).
DATABASE_URL set edilmişse PostgreSQL kullanır (Supabase / Fly.io).
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Bağlantı modu ─────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL")
DB_PATH = Path(__file__).parent.parent.parent / "data" / "paper_trades.db"
_MODE = "pg" if DATABASE_URL else "sqlite"


def _connect():
    """Aktif moda göre DB bağlantısı döndürür."""
    if _MODE == "pg":
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    else:
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        return conn


def _ph():
    """SQL placeholder: PostgreSQL → %s, SQLite → ?"""
    return "%s" if _MODE == "pg" else "?"


def _rows_to_dicts(cursor, rows):
    """Cursor sonuçlarını Sözlük listesine çevirir."""
    if _MODE == "pg":
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in rows]
    else:
        return [dict(row) for row in rows]


# ── CREATE TABLE SQL ───────────────────────────────────────────────────────────
_CREATE_TABLE_PG = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id              SERIAL PRIMARY KEY,
    ticker          TEXT NOT NULL,
    entry_date      TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    stop_loss       REAL NOT NULL,
    target          REAL NOT NULL,
    swing_type      TEXT,
    quality_score   REAL,
    position_size   INTEGER DEFAULT 100,
    max_hold_days   INTEGER DEFAULT 7,
    status          TEXT DEFAULT 'OPEN',
    exit_date       TEXT,
    exit_price      REAL,
    realized_pnl    REAL,
    realized_pnl_pct REAL,
    notes           TEXT,
    trailing_stop   REAL,
    initial_stop    REAL,
    atr             REAL,
    signal_price    REAL,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    entry_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    target REAL NOT NULL,
    swing_type TEXT,
    quality_score REAL,
    position_size INTEGER DEFAULT 100,
    max_hold_days INTEGER DEFAULT 7,
    status TEXT DEFAULT 'OPEN',
    exit_date TEXT,
    exit_price REAL,
    realized_pnl REAL,
    realized_pnl_pct REAL,
    notes TEXT,
    trailing_stop REAL,
    initial_stop REAL,
    atr REAL,
    signal_price REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


class PaperTradeStorage:
    """
    Dual-mode paper trade storage.
    - DATABASE_URL yok → SQLite (local)
    - DATABASE_URL var → PostgreSQL (Supabase / Fly.io)
    """

    def __init__(self, db_path: str = None):
        if db_path:
            # Explicit path override (legacy / test)
            import sqlite3
            self._override_path = db_path
        else:
            self._override_path = None
        self._init_db()
        mode_label = f"PostgreSQL @ Supabase" if _MODE == "pg" else f"SQLite @ {DB_PATH}"
        logger.info(f"PaperTradeStorage initialized: {mode_label}")

    def _get_conn(self):
        if self._override_path:
            import sqlite3
            conn = sqlite3.connect(self._override_path)
            conn.row_factory = sqlite3.Row
            return conn
        return _connect()

    def _init_db(self):
        """Create table if not exists. Also run SQLite migrations for existing DBs."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            if _MODE == "pg" and not self._override_path:
                cursor.execute(_CREATE_TABLE_PG)
            else:
                cursor.execute(_CREATE_TABLE_SQLITE)
                # SQLite migrations for existing DBs
                v3_columns = [
                    ('trailing_stop', 'REAL'),
                    ('initial_stop', 'REAL'),
                    ('atr', 'REAL'),
                    ('signal_price', 'REAL'),
                ]
                import sqlite3 as _sq
                for col_name, col_type in v3_columns:
                    try:
                        cursor.execute(f"ALTER TABLE paper_trades ADD COLUMN {col_name} {col_type}")
                    except _sq.OperationalError:
                        pass

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add_trade(self, trade: Dict) -> int:
        """Add a new paper trade. Returns trade ID."""
        ph = _ph()
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            if _MODE == "pg" and not self._override_path:
                cursor.execute(f"""
                    INSERT INTO paper_trades
                    (ticker, entry_date, entry_price, stop_loss, target,
                     swing_type, quality_score, position_size, max_hold_days, notes,
                     trailing_stop, initial_stop, atr, signal_price, status)
                    VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})
                    RETURNING id
                """, (
                    trade['ticker'], trade['entry_date'], trade['entry_price'],
                    trade['stop_loss'], trade['target'],
                    trade.get('swing_type', 'A'), trade.get('quality_score', 0),
                    trade.get('position_size', 100), trade.get('max_hold_days', 7),
                    trade.get('notes', ''),
                    trade.get('trailing_stop', trade['stop_loss']),
                    trade.get('initial_stop', trade['stop_loss']),
                    trade.get('atr', 0),
                    trade.get('signal_price', trade['entry_price']),
                    trade.get('status', 'OPEN'),
                ))
                trade_id = cursor.fetchone()[0]
            else:
                cursor.execute(f"""
                    INSERT INTO paper_trades
                    (ticker, entry_date, entry_price, stop_loss, target,
                     swing_type, quality_score, position_size, max_hold_days, notes,
                     trailing_stop, initial_stop, atr, signal_price, status)
                    VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})
                """, (
                    trade['ticker'], trade['entry_date'], trade['entry_price'],
                    trade['stop_loss'], trade['target'],
                    trade.get('swing_type', 'A'), trade.get('quality_score', 0),
                    trade.get('position_size', 100), trade.get('max_hold_days', 7),
                    trade.get('notes', ''),
                    trade.get('trailing_stop', trade['stop_loss']),
                    trade.get('initial_stop', trade['stop_loss']),
                    trade.get('atr', 0),
                    trade.get('signal_price', trade['entry_price']),
                    trade.get('status', 'OPEN'),
                ))
                trade_id = cursor.lastrowid

            conn.commit()
            conn.close()
            logger.info(f"Added paper trade: {trade['ticker']} (ID: {trade_id})")
            return trade_id

        except Exception as e:
            logger.error(f"Error adding trade: {e}")
            return -1

    def get_open_trades(self) -> List[Dict]:
        """Get all OPEN and PENDING trades."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM paper_trades
                WHERE status IN ('OPEN', 'PENDING')
                ORDER BY entry_date DESC
            """)
            rows = cursor.fetchall()
            result = _rows_to_dicts(cursor, rows) if _MODE == "pg" and not self._override_path else [dict(r) for r in rows]
            conn.close()
            return result
        except Exception as e:
            logger.error(f"Error getting open trades: {e}")
            return []

    def get_closed_trades(self, limit: int = 50) -> List[Dict]:
        """Get closed trades."""
        ph = _ph()
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM paper_trades
                WHERE status NOT IN ('OPEN', 'PENDING')
                ORDER BY exit_date DESC
                LIMIT {ph}
            """, (limit,))
            rows = cursor.fetchall()
            result = _rows_to_dicts(cursor, rows) if _MODE == "pg" and not self._override_path else [dict(r) for r in rows]
            conn.close()
            return result
        except Exception as e:
            logger.error(f"Error getting closed trades: {e}")
            return []

    def get_all_trades(self) -> List[Dict]:
        """Get all trades."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM paper_trades ORDER BY entry_date DESC")
            rows = cursor.fetchall()
            result = _rows_to_dicts(cursor, rows) if _MODE == "pg" and not self._override_path else [dict(r) for r in rows]
            conn.close()
            return result
        except Exception as e:
            logger.error(f"Error getting all trades: {e}")
            return []

    def get_trade_by_id(self, trade_id: int) -> Optional[Dict]:
        """Get a specific trade by ID."""
        ph = _ph()
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM paper_trades WHERE id = {ph}", (trade_id,))
            row = cursor.fetchone()
            conn.close()
            if row is None:
                return None
            if _MODE == "pg" and not self._override_path:
                cols = [desc[0] for desc in cursor.description]
                return dict(zip(cols, row))
            return dict(row)
        except Exception as e:
            logger.error(f"Error getting trade {trade_id}: {e}")
            return None

    def update_trade(self, trade_id: int, updates: Dict) -> bool:
        """Update fields on a trade."""
        ph = _ph()
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            set_clauses = [f"{key} = {ph}" for key in updates]
            values = list(updates.values())
            set_clauses.append(f"updated_at = {ph}")
            values.append(datetime.now().isoformat())
            values.append(trade_id)

            query = f"UPDATE paper_trades SET {', '.join(set_clauses)} WHERE id = {ph}"
            cursor.execute(query, values)
            conn.commit()
            conn.close()
            logger.info(f"Updated paper trade ID: {trade_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating trade {trade_id}: {e}")
            return False

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_date: str,
        status: str,
        notes: str = ""
    ) -> bool:
        """Close a paper trade and calculate P/L."""
        try:
            trade = self.get_trade_by_id(trade_id)
            if not trade:
                return False
            entry_price = trade['entry_price']
            position_size = trade['position_size']
            pnl = (exit_price - entry_price) * position_size
            pnl_pct = ((exit_price / entry_price) - 1) * 100
            return self.update_trade(trade_id, {
                'status': status,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'realized_pnl': round(pnl, 2),
                'realized_pnl_pct': round(pnl_pct, 2),
                'notes': notes,
            })
        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
            return False

    def delete_trade(self, trade_id: int) -> bool:
        """Delete a trade permanently."""
        ph = _ph()
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM paper_trades WHERE id = {ph}", (trade_id,))
            conn.commit()
            conn.close()
            logger.info(f"Deleted paper trade ID: {trade_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting trade {trade_id}: {e}")
            return False

    def check_duplicate(self, ticker: str, entry_date: str) -> bool:
        """Return True if a PENDING/OPEN trade for this ticker+date already exists."""
        ph = _ph()
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT COUNT(*) FROM paper_trades
                WHERE ticker = {ph} AND entry_date = {ph}
                AND status IN ('OPEN', 'PENDING')
            """, (ticker, entry_date))
            count = cursor.fetchone()[0]
            conn.close()
            return count > 0
        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return False
