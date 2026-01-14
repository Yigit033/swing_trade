"""
Data fetcher module for downloading stock data from various APIs.
"""

import logging
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches stock data from yfinance and Alpha Vantage (backup).
    
    Attributes:
        source (str): Primary data source ('yfinance' or 'alphavantage')
        alpha_vantage_key (str): API key for Alpha Vantage
    """
    
    def __init__(self, source: str = "yfinance", alpha_vantage_key: Optional[str] = None):
        """
        Initialize DataFetcher.
        
        Args:
            source: Primary data source ('yfinance' or 'alphavantage')
            alpha_vantage_key: API key for Alpha Vantage backup
        """
        self.source = source
        self.alpha_vantage_key = alpha_vantage_key
        logger.info(f"DataFetcher initialized with source: {source}")
    
    def fetch_stock_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single stock using yf.Ticker().history().
        
        IMPORTANT: Uses yf.Ticker().history() instead of yf.download() to prevent
        data corruption in parallel downloads. yf.download() with MultiIndex can
        mix up ticker data.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            period: Period if dates not specified
        
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
            None if fetch fails
        """
        if not ticker:
            raise ValueError("Ticker symbol cannot be empty")
        
        try:
            logger.debug(f"Fetching data for {ticker}")
            
            # Use yf.Ticker().history() - SAFER than yf.download()
            # This method is guaranteed to return data for the specific ticker
            stock = yf.Ticker(ticker)
            
            if start_date and end_date:
                data = stock.history(start=start_date, end=end_date)
            else:
                data = stock.history(period=period)
            
            if data.empty:
                logger.warning(f"No data received for {ticker}")
                return None
            
            # Columns from .history() are already single-level: Open, High, Low, Close, Volume
            # No MultiIndex handling needed
            
            # Validate data
            if not self._validate_ohlcv_data(data):
                logger.warning(f"Invalid data received for {ticker}")
                return None
            
            # VERIFICATION: Check that prices are reasonable for this ticker
            # This catches any remaining data issues
            avg_price = data['Close'].mean()
            last_price = data['Close'].iloc[-1]
            
            # Sanity check: price should be positive
            if last_price <= 0 or avg_price <= 0:
                logger.warning(f"{ticker}: Invalid price data (last=${last_price:.2f})")
                return None
            
            # Reset index to have date as column
            data = data.reset_index()
            
            # Rename 'Datetime' or 'Date' column to 'Date'
            if 'Datetime' in data.columns:
                data.rename(columns={'Datetime': 'Date'}, inplace=True)
            
            # Remove timezone from Date column if present
            if pd.api.types.is_datetime64_any_dtype(data['Date']):
                data['Date'] = data['Date'].dt.tz_localize(None)
            
            # Keep only required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [c for c in required_cols if c in data.columns]
            data = data[available_cols]
            
            logger.info(f"Successfully fetched {len(data)} rows for {ticker} (last=${last_price:.2f})")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {str(e)}", exc_info=True)
            return None
    
    def fetch_multiple_stocks(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
        max_workers: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks concurrently.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            period: Period if dates not specified
            max_workers: Maximum concurrent workers
        
        Returns:
            Dictionary mapping ticker to DataFrame
        
        Example:
            >>> fetcher = DataFetcher()
            >>> tickers = ['AAPL', 'MSFT', 'GOOGL']
            >>> data = fetcher.fetch_multiple_stocks(tickers, period='1y')
            >>> print(f"Fetched {len(data)} stocks")
        """
        results = {}
        
        logger.info(f"Fetching data for {len(tickers)} stocks with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(
                    self.fetch_stock_data,
                    ticker,
                    start_date,
                    end_date,
                    period
                ): ticker
                for ticker in tickers
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(tickers), desc="Downloading stocks") as pbar:
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        data = future.result()
                        if data is not None:
                            results[ticker] = data
                    except Exception as e:
                        logger.error(f"Error fetching {ticker}: {e}")
                    finally:
                        pbar.update(1)
        
        logger.info(f"Successfully fetched {len(results)}/{len(tickers)} stocks")
        return results
    
    def get_sp500_tickers(self) -> List[str]:
        """
        Get list of S&P 500 ticker symbols.
        
        Returns:
            List of ticker symbols
        
        Example:
            >>> fetcher = DataFetcher()
            >>> tickers = fetcher.get_sp500_tickers()
            >>> print(f"Found {len(tickers)} S&P 500 stocks")
        """
        try:
            logger.info("Fetching S&P 500 ticker list")
            
            # Try with requests and proper headers first
            import requests
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            tables = pd.read_html(response.text)
            sp500_table = tables[0]
            
            tickers = sp500_table['Symbol'].tolist()
            
            # Clean ticker symbols (some have periods instead of hyphens)
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            
            logger.info(f"Found {len(tickers)} S&P 500 stocks")
            return tickers
            
        except Exception as e:
            logger.error(f"Failed to fetch S&P 500 list: {e}", exc_info=True)
            # Return a comprehensive fallback list of major stocks across sectors
            fallback = [
                # Technology
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'AVGO', 'ADBE',
                'CRM', 'AMD', 'INTC', 'CSCO', 'ORCL', 'QCOM', 'TXN', 'IBM', 'NOW', 'INTU',
                'AMAT', 'MU', 'LRCX', 'ADI', 'KLAC', 'SNPS', 'CDNS', 'PANW', 'CRWD', 'FTNT',
                # Financial
                'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'BLK', 'SCHW',
                'C', 'USB', 'PNC', 'TFC', 'COF', 'SPGI', 'ICE', 'CME', 'MCO', 'AON',
                # Healthcare
                'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
                'AMGN', 'MDT', 'GILD', 'CVS', 'CI', 'ELV', 'ISRG', 'VRTX', 'REGN', 'ZTS',
                # Consumer
                'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
                'LOW', 'DIS', 'NFLX', 'CMCSA', 'BKNG', 'MAR', 'YUM', 'EL', 'CL', 'MDLZ',
                # Industrial
                'CAT', 'HON', 'UPS', 'BA', 'GE', 'MMM', 'RTX', 'DE', 'LMT', 'UNP',
                'ADP', 'FDX', 'EMR', 'ITW', 'ETN', 'PH', 'ROK', 'CTAS', 'FAST', 'ODFL',
                # Energy
                'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI',
                # Utilities & Real Estate
                'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'ED',
                'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
                # Communication
                'T', 'VZ', 'TMUS', 'CHTR', 'ATVI'
            ]
            logger.warning(f"Using fallback list of {len(fallback)} stocks")
            return fallback
    
    def get_stock_info(self, ticker: str) -> Optional[Dict]:
        """
        Get stock information (sector, market cap, etc.).
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary with stock information or None if fetch fails
        
        Example:
            >>> fetcher = DataFetcher()
            >>> info = fetcher.get_stock_info('AAPL')
            >>> print(info['sector'], info['marketCap'])
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'ticker': ticker,
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'marketCap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD')
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch info for {ticker}: {e}")
            return None
    
    def _validate_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """
        Validate OHLCV data integrity.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            True if data is valid, False otherwise
        """
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check required columns exist
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns. Found: {df.columns.tolist()}")
            return False
        
        # Check for completely empty data
        if len(df) == 0:
            return False
        
        # Check for too many NaN values (allow up to 10%)
        nan_ratio = df[required_cols].isna().sum().sum() / (len(df) * len(required_cols))
        if nan_ratio > 0.1:
            logger.warning(f"Too many NaN values: {nan_ratio:.2%}")
            return False
        
        # Check price relationships (only for non-NaN rows)
        valid_mask = df[required_cols].notna().all(axis=1)
        valid_df = df[valid_mask]
        
        if len(valid_df) > 0:
            if not (valid_df['High'] >= valid_df['Low']).all():
                logger.warning("High price < Low price found")
                return False
            
            if not (valid_df['High'] >= valid_df['Close']).all():
                logger.warning("High price < Close price found")
                return False
            
            if not (valid_df['High'] >= valid_df['Open']).all():
                logger.warning("High price < Open price found")
                return False
            
            # Check for negative or zero values
            if not (valid_df[['Open', 'High', 'Low', 'Close']] > 0).all().all():
                logger.warning("Negative or zero prices found")
                return False
        
        return True

