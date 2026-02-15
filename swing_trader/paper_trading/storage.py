"""
Paper Trading Storage - SQLite database operations for paper trades.
"""

import sqlite3
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent.parent / "data" / "paper_trades.db"


class PaperTradeStorage:
    """
    SQLite storage for paper trades.
    
    Table Schema:
    - id: INTEGER PRIMARY KEY
    - ticker: TEXT
    - entry_date: TEXT (YYYY-MM-DD)
    - entry_price: REAL
    - stop_loss: REAL
    - target: REAL
    - swing_type: TEXT (A, B, C, S)
    - quality_score: REAL
    - position_size: INTEGER (shares)
    - max_hold_days: INTEGER
    - status: TEXT (OPEN, STOPPED, TARGET, TIMEOUT, MANUAL)
    - exit_date: TEXT
    - exit_price: REAL
    - realized_pnl: REAL
    - realized_pnl_pct: REAL
    - notes: TEXT
    - created_at: TEXT
    - updated_at: TEXT
    """
    
    def __init__(self, db_path: str = None):
        """Initialize storage with database path."""
        self.db_path = db_path or str(DB_PATH)
        self._init_db()
        logger.info(f"PaperTradeStorage initialized: {self.db_path}")
    
    def _init_db(self):
        """Create database and tables if not exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
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
            """)
            
            # Migration: add V3 columns to existing databases
            v3_columns = [
                ('trailing_stop', 'REAL'),
                ('initial_stop', 'REAL'),
                ('atr', 'REAL'),
                ('signal_price', 'REAL'),
            ]
            for col_name, col_type in v3_columns:
                try:
                    cursor.execute(f"ALTER TABLE paper_trades ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass  # Column already exists
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def add_trade(self, trade: Dict) -> int:
        """
        Add a new paper trade.
        
        Args:
            trade: Dict with keys: ticker, entry_date, entry_price, stop_loss,
                   target, swing_type, quality_score, position_size, max_hold_days
        
        Returns:
            Trade ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO paper_trades 
                (ticker, entry_date, entry_price, stop_loss, target, 
                 swing_type, quality_score, position_size, max_hold_days, notes,
                 trailing_stop, initial_stop, atr, signal_price, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade['ticker'],
                trade['entry_date'],
                trade['entry_price'],
                trade['stop_loss'],
                trade['target'],
                trade.get('swing_type', 'A'),
                trade.get('quality_score', 0),
                trade.get('position_size', 100),
                trade.get('max_hold_days', 7),
                trade.get('notes', ''),
                trade.get('trailing_stop', trade['stop_loss']),
                trade.get('initial_stop', trade['stop_loss']),
                trade.get('atr', 0),
                trade.get('signal_price', trade['entry_price']),
                trade.get('status', 'OPEN')
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
        """Get all open paper trades."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM paper_trades 
                WHERE status IN ('OPEN', 'PENDING')
                ORDER BY entry_date DESC
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting open trades: {e}")
            return []
    
    def get_closed_trades(self, limit: int = 50) -> List[Dict]:
        """Get closed paper trades."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM paper_trades 
                WHERE status NOT IN ('OPEN', 'PENDING')
                ORDER BY exit_date DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting closed trades: {e}")
            return []
    
    def get_all_trades(self) -> List[Dict]:
        """Get all paper trades."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM paper_trades 
                ORDER BY entry_date DESC
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting all trades: {e}")
            return []
    
    def get_trade_by_id(self, trade_id: int) -> Optional[Dict]:
        """Get a specific trade by ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM paper_trades WHERE id = ?", (trade_id,))
            row = cursor.fetchone()
            conn.close()
            
            return dict(row) if row else None
            
        except Exception as e:
            logger.error(f"Error getting trade {trade_id}: {e}")
            return None
    
    def update_trade(self, trade_id: int, updates: Dict) -> bool:
        """
        Update a paper trade.
        
        Args:
            trade_id: Trade ID to update
            updates: Dict with fields to update
        
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build update query
            set_clauses = []
            values = []
            for key, value in updates.items():
                set_clauses.append(f"{key} = ?")
                values.append(value)
            
            set_clauses.append("updated_at = ?")
            values.append(datetime.now().isoformat())
            values.append(trade_id)
            
            query = f"UPDATE paper_trades SET {', '.join(set_clauses)} WHERE id = ?"
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
        """
        Close a paper trade.
        
        Args:
            trade_id: Trade ID
            exit_price: Exit price
            exit_date: Exit date (YYYY-MM-DD)
            status: STOPPED, TARGET, TIMEOUT, MANUAL
            notes: Optional notes
        """
        try:
            # Get trade to calculate P/L
            trade = self.get_trade_by_id(trade_id)
            if not trade:
                return False
            
            entry_price = trade['entry_price']
            position_size = trade['position_size']
            
            # Calculate P/L
            pnl = (exit_price - entry_price) * position_size
            pnl_pct = ((exit_price / entry_price) - 1) * 100
            
            return self.update_trade(trade_id, {
                'status': status,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'realized_pnl': round(pnl, 2),
                'realized_pnl_pct': round(pnl_pct, 2),
                'notes': notes
            })
            
        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
            return False
    
    def delete_trade(self, trade_id: int) -> bool:
        """Delete a paper trade."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM paper_trades WHERE id = ?", (trade_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted paper trade ID: {trade_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting trade {trade_id}: {e}")
            return False
    
    def check_duplicate(self, ticker: str, entry_date: str) -> bool:
        """Check if a trade already exists for this ticker and date."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM paper_trades 
                WHERE ticker = ? AND entry_date = ? AND status = 'OPEN'
            """, (ticker, entry_date))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count > 0
            
        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return False
