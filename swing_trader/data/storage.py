"""
Database storage module for managing stock data in SQLite.
"""

import logging
import sqlite3
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database operations for stock data.
    
    Attributes:
        db_path (str): Path to SQLite database file
    """
    
    def __init__(self, db_path: str = "data/stocks.db"):
        """
        Initialize DatabaseManager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure directory exists
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DatabaseManager initialized with path: {db_path}")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Yields:
            sqlite3.Connection: Database connection
        
        Example:
            >>> db = DatabaseManager()
            >>> with db.get_connection() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT * FROM stock_data LIMIT 1")
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}", exc_info=True)
            raise
        finally:
            if conn:
                conn.close()
    
    def initialize_database(self) -> bool:
        """
        Create database tables if they don't exist.
        
        Returns:
            True if successful, False otherwise
        
        Example:
            >>> db = DatabaseManager()
            >>> success = db.initialize_database()
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Stock data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stock_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker TEXT NOT NULL,
                        date DATE NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume INTEGER NOT NULL,
                        adj_close REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(ticker, date)
                    )
                """)
                
                # Stock info table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stock_info (
                        ticker TEXT PRIMARY KEY,
                        name TEXT,
                        sector TEXT,
                        industry TEXT,
                        market_cap REAL,
                        currency TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Trading signals table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker TEXT NOT NULL,
                        date DATE NOT NULL,
                        signal_type TEXT NOT NULL,
                        score INTEGER,
                        entry_price REAL,
                        stop_loss REAL,
                        target_1 REAL,
                        target_2 REAL,
                        position_size INTEGER,
                        risk_amount REAL,
                        conditions TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(ticker, date, signal_type)
                    )
                """)
                
                # Create indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_stock_data_ticker_date 
                    ON stock_data(ticker, date)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_signals_ticker_date 
                    ON trading_signals(ticker, date)
                """)
                
                logger.info("Database initialized successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)
            return False
    
    def insert_stock_data(self, ticker: str, df: pd.DataFrame) -> bool:
        """
        Insert or update stock OHLCV data.
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with OHLCV data (columns: Date, Open, High, Low, Close, Volume)
        
        Returns:
            True if successful, False otherwise
        
        Example:
            >>> db = DatabaseManager()
            >>> data = fetcher.fetch_stock_data('AAPL')
            >>> success = db.insert_stock_data('AAPL', data)
        """
        if df is None or df.empty:
            logger.warning(f"No data to insert for {ticker}")
            return False
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    INSERT OR REPLACE INTO stock_data 
                    (ticker, date, open, high, low, close, volume, adj_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                records = []
                for _, row in df.iterrows():
                    # Handle date conversion
                    date = row['Date']
                    if isinstance(date, pd.Timestamp):
                        date = date.strftime('%Y-%m-%d')
                    
                    records.append((
                        ticker,
                        date,
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        int(row['Volume']),
                        float(row.get('Adj Close', row['Close']))
                    ))
                
                cursor.executemany(query, records)
                logger.info(f"Inserted {len(records)} records for {ticker}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to insert data for {ticker}: {e}", exc_info=True)
            return False
    
    def get_stock_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve stock data from database.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            limit: Maximum number of rows to return
        
        Returns:
            DataFrame with stock data or None if not found
        
        Example:
            >>> db = DatabaseManager()
            >>> data = db.get_stock_data('AAPL', start_date='2024-01-01')
        """
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT date, open, high, low, close, volume, adj_close
                    FROM stock_data
                    WHERE ticker = ?
                """
                params = [ticker]
                
                if start_date:
                    query += " AND date >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND date <= ?"
                    params.append(end_date)
                
                query += " ORDER BY date DESC"
                
                if limit:
                    query += f" LIMIT {limit}"
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    logger.warning(f"No data found for {ticker}")
                    return None
                
                # Convert date column
                df['date'] = pd.to_datetime(df['date'])
                
                # Rename columns to match standard format
                df.rename(columns={
                    'date': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'adj_close': 'Adj Close'
                }, inplace=True)
                
                # Sort by date ascending
                df = df.sort_values('Date').reset_index(drop=True)
                
                logger.debug(f"Retrieved {len(df)} records for {ticker}")
                return df
                
        except Exception as e:
            logger.error(f"Failed to retrieve data for {ticker}: {e}", exc_info=True)
            return None
    
    def insert_stock_info(self, info: Dict[str, Any]) -> bool:
        """
        Insert or update stock information.
        
        Args:
            info: Dictionary with stock info (ticker, name, sector, industry, marketCap, currency)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    INSERT OR REPLACE INTO stock_info 
                    (ticker, name, sector, industry, market_cap, currency, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """
                
                cursor.execute(query, (
                    info['ticker'],
                    info.get('name', ''),
                    info.get('sector', 'Unknown'),
                    info.get('industry', 'Unknown'),
                    info.get('marketCap', 0),
                    info.get('currency', 'USD')
                ))
                
                logger.debug(f"Updated info for {info['ticker']}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to insert stock info: {e}", exc_info=True)
            return False
    
    def get_stock_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve stock information.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary with stock info or None if not found
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT ticker, name, sector, industry, market_cap, currency
                    FROM stock_info
                    WHERE ticker = ?
                """, (ticker,))
                
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve info for {ticker}: {e}")
            return None
    
    def insert_trading_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Insert a trading signal.
        
        Args:
            signal: Dictionary with signal data
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    INSERT OR REPLACE INTO trading_signals 
                    (ticker, date, signal_type, score, entry_price, stop_loss, 
                     target_1, target_2, position_size, risk_amount, conditions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                cursor.execute(query, (
                    signal['ticker'],
                    signal['date'],
                    signal.get('signal_type', 'BUY'),
                    signal.get('score', 0),
                    signal.get('entry_price', 0),
                    signal.get('stop_loss', 0),
                    signal.get('target_1', 0),
                    signal.get('target_2', 0),
                    signal.get('position_size', 0),
                    signal.get('risk_amount', 0),
                    str(signal.get('conditions', {}))
                ))
                
                logger.debug(f"Inserted signal for {signal['ticker']}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to insert signal: {e}", exc_info=True)
            return False
    
    def get_latest_date(self, ticker: str) -> Optional[str]:
        """
        Get the latest date for which we have data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Latest date as string (YYYY-MM-DD) or None if no data
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT MAX(date) as latest_date
                    FROM stock_data
                    WHERE ticker = ?
                """, (ticker,))
                
                result = cursor.fetchone()
                if result and result['latest_date']:
                    return result['latest_date']
                return None
                
        except Exception as e:
            logger.error(f"Failed to get latest date for {ticker}: {e}")
            return None
    
    def get_all_tickers(self) -> List[str]:
        """
        Get list of all tickers in database.
        
        Returns:
            List of ticker symbols
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT ticker FROM stock_data ORDER BY ticker")
                return [row['ticker'] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get ticker list: {e}")
            return []

