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

# ── Bağlantı modu ─────────────────────────────────────────────────────────────────────────────────
# .env dosyasını yükle — Streamlit ve API'nin AYNI DB'ye bağlanması için kritik
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL")
DB_PATH = Path(__file__).parent.parent.parent / "data" / "paper_trades.db"


# Mode detection based on URL scheme only (non-blocking, no DNS/TCP at import time).
# Actual connection errors are handled gracefully in _connect().
def _url_mode() -> str:
    if DATABASE_URL and (DATABASE_URL.startswith("postgresql") or DATABASE_URL.startswith("postgres")):
        logger.info("PostgreSQL modu secildi (baglanti ilk istekte denenir)")
        return "pg"
    return "sqlite"


_MODE = _url_mode()


def _connect():
    """Aktif moda gore DB baglantisi dondurur. PG basarisiz olursa SQLite'a gec."""
    global _MODE
    if _MODE == "pg":
        try:
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
            return conn
        except Exception as e:
            logger.warning(f"PostgreSQL baglantisi basarisiz ({e.__class__.__name__}): {e}")
            raise Exception(f"Database connection failed: {e}")
    import sqlite3
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
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
    user_id         UUID,
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
    current_price   REAL,
    unrealized_pnl  REAL,
    unrealized_pnl_pct REAL,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
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
    current_price REAL,
    unrealized_pnl REAL,
    unrealized_pnl_pct REAL,
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
        """Create table if not exists. Run migrations for both PG and SQLite."""
        # Auth: user_id for multi-tenant (Supabase Auth)
        auth_columns = [
            ('user_id', 'UUID' if _MODE == "pg" and not self._override_path else 'TEXT'),
        ]
        # v3 columns that may be missing from older schema
        v3_columns = [
            ('trailing_stop', 'REAL'),
            ('initial_stop', 'REAL'),
            ('atr', 'REAL'),
            ('signal_price', 'REAL'),
            ('current_price', 'REAL'),
            ('unrealized_pnl', 'REAL'),
            ('unrealized_pnl_pct', 'REAL'),
            # v3.1: Partial exit (T1/T2 dual target)
            ('target_2', 'REAL'),
            ('partial_exit_price', 'REAL'),
            ('partial_exit_pct', 'REAL'),
        ]
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            if _MODE == "pg" and not self._override_path:
                cursor.execute(_CREATE_TABLE_PG)
                # Auth: user_id column
                for col_name, col_type in auth_columns:
                    try:
                        cursor.execute(
                            f"ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
                        )
                    except Exception:
                        pass
                # PostgreSQL migrations for existing tables
                for col_name, col_type in v3_columns:
                    try:
                        cursor.execute(
                            f"ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
                        )
                    except Exception:
                        pass
                # Meta table for lightweight key/value data (e.g. last price update)
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS paper_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                # Indexes for common queries
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status)",
                    "CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker ON paper_trades(ticker)",
                    "CREATE INDEX IF NOT EXISTS idx_paper_trades_entry_date ON paper_trades(entry_date)",
                    "CREATE INDEX IF NOT EXISTS idx_paper_trades_exit_date ON paper_trades(exit_date)",
                ]:
                    try:
                        cursor.execute(idx_sql)
                    except Exception:
                        pass
            else:
                cursor.execute(_CREATE_TABLE_SQLITE)
                # SQLite migrations for existing DBs
                import sqlite3 as _sq
                for col_name, col_type in auth_columns:
                    try:
                        cursor.execute(f"ALTER TABLE paper_trades ADD COLUMN {col_name} {col_type}")
                    except _sq.OperationalError:
                        pass
                for col_name, col_type in v3_columns:
                    try:
                        cursor.execute(f"ALTER TABLE paper_trades ADD COLUMN {col_name} {col_type}")
                    except _sq.OperationalError:
                        pass
                # Meta table for lightweight key/value data (e.g. last price update)
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS paper_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                # Indexes for common queries
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status)",
                    "CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker ON paper_trades(ticker)",
                    "CREATE INDEX IF NOT EXISTS idx_paper_trades_entry_date ON paper_trades(entry_date)",
                    "CREATE INDEX IF NOT EXISTS idx_paper_trades_exit_date ON paper_trades(exit_date)",
                ]:
                    try:
                        cursor.execute(idx_sql)
                    except _sq.OperationalError:
                        pass

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add_trade(self, trade: Dict, user_id: Optional[str] = None) -> int:
        """Add a new paper trade. Returns trade ID."""
        ph = _ph()

        # ── numpy → native Python dönüşümü (psycopg2 np.float64 tanımıyor) ──
        def _native(v):
            """Convert numpy types to Python native types."""
            if v is None:
                return v
            try:
                import numpy as _np
                if isinstance(v, (_np.integer,)):
                    return int(v)
                if isinstance(v, (_np.floating,)):
                    return float(v)
                if isinstance(v, _np.ndarray):
                    return v.tolist()
            except ImportError:
                pass
            return v

        trade = {k: _native(v) for k, v in trade.items()}

        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            if user_id:
                _cols = """(user_id, ticker, entry_date, entry_price, stop_loss, target,
                         swing_type, quality_score, position_size, max_hold_days, notes,
                         trailing_stop, initial_stop, atr, signal_price, status, target_2)"""
                _vals = ",".join([ph] * 17)
                _params = (
                    user_id,
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
                    trade.get('target_2', trade['target']),
                )
            else:
                _cols = """(ticker, entry_date, entry_price, stop_loss, target,
                         swing_type, quality_score, position_size, max_hold_days, notes,
                         trailing_stop, initial_stop, atr, signal_price, status, target_2)"""
                _vals = ",".join([ph] * 16)
                _params = (
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
                    trade.get('target_2', trade['target']),
                )

            if _MODE == "pg" and not self._override_path:
                cursor.execute(f"INSERT INTO paper_trades {_cols} VALUES ({_vals}) RETURNING id", _params)
                trade_id = cursor.fetchone()[0]
            else:
                cursor.execute(f"INSERT INTO paper_trades {_cols} VALUES ({_vals})", _params)
                trade_id = cursor.lastrowid

            conn.commit()
            logger.info(f"Added paper trade: {trade['ticker']} (ID: {trade_id})")
            return trade_id

        except Exception as e:
            logger.error(f"Error adding trade: {e}")
            return -1
        finally:
            if conn:
                conn.close()

    def get_open_trades(self, user_id: Optional[str] = None) -> List[Dict]:
        """Get all OPEN and PENDING trades."""
        ph = _ph()
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            if user_id:
                cursor.execute(f"""
                    SELECT * FROM paper_trades
                    WHERE status IN ('OPEN', 'PENDING') AND (user_id IS NULL OR user_id = {ph})
                    ORDER BY entry_date DESC
                """, (user_id,))
            else:
                cursor.execute("""
                    SELECT * FROM paper_trades
                    WHERE status IN ('OPEN', 'PENDING')
                    ORDER BY entry_date DESC
                """)
            rows = cursor.fetchall()
            return _rows_to_dicts(cursor, rows) if _MODE == "pg" and not self._override_path else [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error getting open trades: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_closed_trades(self, limit: int = 50, user_id: Optional[str] = None) -> List[Dict]:
        """Get closed trades."""
        ph = _ph()
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            if user_id:
                cursor.execute(f"""
                    SELECT * FROM paper_trades
                    WHERE status NOT IN ('OPEN', 'PENDING') AND (user_id IS NULL OR user_id = {ph})
                    ORDER BY exit_date DESC
                    LIMIT {ph}
                """, (user_id, limit))
            else:
                cursor.execute(f"""
                    SELECT * FROM paper_trades
                    WHERE status NOT IN ('OPEN', 'PENDING')
                    ORDER BY exit_date DESC
                    LIMIT {ph}
                """, (limit,))
            rows = cursor.fetchall()
            return _rows_to_dicts(cursor, rows) if _MODE == "pg" and not self._override_path else [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error getting closed trades: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_all_trades(self, user_id: Optional[str] = None) -> List[Dict]:
        """Get all trades."""
        ph = _ph()
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            if user_id:
                cursor.execute(f"""
                    SELECT * FROM paper_trades
                    WHERE user_id IS NULL OR user_id = {ph}
                    ORDER BY entry_date DESC
                """, (user_id,))
            else:
                cursor.execute("SELECT * FROM paper_trades ORDER BY entry_date DESC")
            rows = cursor.fetchall()
            return _rows_to_dicts(cursor, rows) if _MODE == "pg" and not self._override_path else [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error getting all trades: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def _set_meta(self, key: str, value: str) -> None:
        """Upsert a key/value pair into paper_meta."""
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            if _MODE == "pg" and not self._override_path:
                cursor.execute(
                    """
                    INSERT INTO paper_meta (key, value, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = NOW()
                    """,
                    (key, value),
                )
            else:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO paper_meta (key, value, updated_at)
                    VALUES (?, ?, datetime('now'))
                    """,
                    (key, value),
                )
            conn.commit()
        except Exception as e:
            logger.error(f"Error setting meta '{key}': {e}")
        finally:
            if conn:
                conn.close()

    def _get_meta(self, key: str) -> Optional[str]:
        """Fetch a value from paper_meta by key."""
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            placeholder = "%s" if _MODE == "pg" and not self._override_path else "?"
            cursor.execute(f"SELECT value FROM paper_meta WHERE key = {placeholder}", (key,))
            row = cursor.fetchone()
            if not row:
                return None
            return row[0] if isinstance(row, (tuple, list)) else row
        except Exception as e:
            logger.error(f"Error getting meta '{key}': {e}")
            return None
        finally:
            if conn:
                conn.close()

    def touch_last_price_update(self) -> None:
        """Record the moment when prices were last refreshed."""
        self._set_meta("last_price_update", datetime.now().isoformat())

    def get_last_update_timestamp(self) -> Optional[str]:
        """
        Return the last time prices were refreshed via the API.
        
        This is stored in the paper_meta table and updated whenever
        /api/trades/update-prices is called.
        """
        return self._get_meta("last_price_update")

    def get_trade_by_id(self, trade_id: int, user_id: Optional[str] = None) -> Optional[Dict]:
        """Get a specific trade by ID."""
        ph = _ph()
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            if user_id:
                cursor.execute(f"""
                    SELECT * FROM paper_trades
                    WHERE id = {ph} AND (user_id IS NULL OR user_id = {ph})
                """, (trade_id, user_id))
            else:
                cursor.execute(f"SELECT * FROM paper_trades WHERE id = {ph}", (trade_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            if _MODE == "pg" and not self._override_path:
                cols = [desc[0] for desc in cursor.description]
                return dict(zip(cols, row))
            return dict(row)
        except Exception as e:
            logger.error(f"Error getting trade {trade_id}: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def update_trade(self, trade_id: int, updates: Dict, user_id: Optional[str] = None) -> bool:
        """Update fields on a trade."""
        if not updates:
            return True
        ph = _ph()
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            set_clauses = [f"{key} = {ph}" for key in updates]
            values = list(updates.values())
            set_clauses.append(f"updated_at = {ph}")
            values.append(datetime.now().isoformat())
            values.append(trade_id)
            if user_id:
                values.append(user_id)
                query = f"UPDATE paper_trades SET {', '.join(set_clauses)} WHERE id = {ph} AND (user_id IS NULL OR user_id = {ph})"
            else:
                query = f"UPDATE paper_trades SET {', '.join(set_clauses)} WHERE id = {ph}"
            cursor.execute(query, values)
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating trade {trade_id}: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_date: str,
        status: str,
        notes: str = "",
        user_id: Optional[str] = None,
    ) -> bool:
        """Close a paper trade and calculate P/L.

        v3.1: Blended PnL when partial exit exists.
        If 50% was already sold at T1 (partial_exit_price), the final PnL
        combines partial + remaining position performance.
        """
        try:
            trade = self.get_trade_by_id(trade_id, user_id)
            if not trade:
                return False
            entry_price = trade['entry_price']
            position_size = trade['position_size']
            partial_exit_price = trade.get('partial_exit_price')
            partial_pct = trade.get('partial_exit_pct') or 0

            if partial_exit_price and partial_pct > 0:
                # Blended PnL: partial sold at T1, remainder at final exit
                partial_ratio = partial_pct / 100  # e.g. 0.50
                remaining_ratio = 1 - partial_ratio
                partial_shares = int(position_size * partial_ratio)
                remaining_shares = position_size - partial_shares
                partial_pnl = partial_shares * (partial_exit_price - entry_price)
                remaining_pnl = remaining_shares * (exit_price - entry_price)
                pnl = partial_pnl + remaining_pnl
                # Blended % = total_pnl / total_cost
                total_cost = position_size * entry_price
                pnl_pct = (pnl / total_cost) * 100 if total_cost > 0 else 0
            else:
                pnl = (exit_price - entry_price) * position_size
                pnl_pct = ((exit_price / entry_price) - 1) * 100

            return self.update_trade(trade_id, {
                'status': status,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'realized_pnl': round(pnl, 2),
                'realized_pnl_pct': round(pnl_pct, 2),
                'notes': notes,
            }, user_id)
        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
            return False

    def delete_trade(self, trade_id: int, user_id: Optional[str] = None) -> bool:
        """Delete a trade permanently."""
        ph = _ph()
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            if user_id:
                cursor.execute(f"""
                    DELETE FROM paper_trades
                    WHERE id = {ph} AND (user_id IS NULL OR user_id = {ph})
                """, (trade_id, user_id))
            else:
                cursor.execute(f"DELETE FROM paper_trades WHERE id = {ph}", (trade_id,))
            conn.commit()
            logger.info(f"Deleted paper trade ID: {trade_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting trade {trade_id}: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def check_duplicate(self, ticker: str, entry_date: str, user_id: Optional[str] = None) -> bool:
        """Return True if a PENDING/OPEN trade for this ticker+date already exists."""
        ph = _ph()
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            if user_id:
                cursor.execute(f"""
                    SELECT COUNT(*) FROM paper_trades
                    WHERE ticker = {ph} AND entry_date = {ph}
                    AND status IN ('OPEN', 'PENDING')
                    AND (user_id IS NULL OR user_id = {ph})
                """, (ticker, entry_date, user_id))
            else:
                cursor.execute(f"""
                    SELECT COUNT(*) FROM paper_trades
                    WHERE ticker = {ph} AND entry_date = {ph}
                    AND status IN ('OPEN', 'PENDING')
                """, (ticker, entry_date))
            count = cursor.fetchone()[0]
            return count > 0
        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return False
        finally:
            if conn:
                conn.close()
