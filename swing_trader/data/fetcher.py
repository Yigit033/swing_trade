"""
Data fetcher module for downloading stock data from various APIs.
"""

import logging
import time
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Suppress urllib3 connection pool full warnings from yf.download parallel threads
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


class DataFetcher:
    """
    Fetches stock data from yfinance and Alpha Vantage (backup).
    
    Attributes:
        source (str): Primary data source ('yfinance' or 'alphavantage')
        alpha_vantage_key (str): API key for Alpha Vantage
    """
    
    def __init__(
        self,
        source: str = "yfinance",
        alpha_vantage_key: Optional[str] = None,
        tiingo_key: Optional[str] = None,
        finnhub_key: Optional[str] = None,
    ):
        self.source = source
        self.alpha_vantage_key = alpha_vantage_key
        self.tiingo_key = tiingo_key or ""
        self.finnhub_key = finnhub_key or ""
        fallbacks = [s for s, k in [("tiingo", self.tiingo_key), ("finnhub", self.finnhub_key)] if k]
        logger.info(f"DataFetcher initialized: primary=yfinance fallbacks={fallbacks or ['none']}")
    
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
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(f"Fetching data for {ticker}")
                stock = yf.Ticker(ticker)

                if start_date and end_date:
                    data = stock.history(start=start_date, end=end_date)
                else:
                    data = stock.history(period=period)

                if data.empty:
                    logger.warning(f"No data received for {ticker}")
                    return None

                if not self._validate_ohlcv_data(data):
                    logger.warning(f"Invalid data received for {ticker}")
                    return None

                avg_price = data['Close'].mean()
                last_price = data['Close'].iloc[-1]
                if last_price <= 0 or avg_price <= 0:
                    logger.warning(f"{ticker}: Invalid price data (last=${last_price:.2f})")
                    return None

                data = data.reset_index()
                if 'Datetime' in data.columns:
                    data.rename(columns={'Datetime': 'Date'}, inplace=True)
                if pd.api.types.is_datetime64_any_dtype(data['Date']):
                    data['Date'] = data['Date'].dt.tz_localize(None)

                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                available_cols = [c for c in required_cols if c in data.columns]
                data = data[available_cols]

                logger.info(f"Successfully fetched {len(data)} rows for {ticker} (last=${last_price:.2f})")
                return data

            except Exception as e:
                err = str(e).lower()
                if ('rate' in err or 'too many' in err or '429' in err) and attempt < max_retries - 1:
                    wait = 5 * (2 ** attempt)
                    logger.warning(f"{ticker}: Rate limited, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                logger.warning(f"yfinance failed for {ticker}: {str(e)}")
                break  # fall through to fallback sources

        # yfinance exhausted — try Tiingo then Finnhub
        s_date, e_date = (start_date, end_date) if (start_date and end_date) else self._period_to_dates(period)

        if self.tiingo_key:
            data = self._fetch_tiingo_single(ticker, s_date, e_date)
            if data is not None:
                logger.info(f"[Tiingo] {ticker}: {len(data)} rows")
                return data

        if self.finnhub_key:
            data = self._fetch_finnhub_single(ticker, s_date, e_date)
            if data is not None:
                logger.info(f"[Finnhub] {ticker}: {len(data)} rows")
                return data

        return None
    
    def fetch_multiple_stocks(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
        max_workers: int = 3
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

    def fetch_multiple_stocks_batch(
        self,
        tickers: List[str],
        period: str = "3mo",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        chunk_size: int = 25,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for many tickers using yf.download() in chunks.
        One HTTP request per chunk vs N individual requests — much less likely to hit rate limits.
        Falls back to empty dict on failure so caller can use individual fetches.
        """
        if not tickers:
            return {}

        results: Dict[str, pd.DataFrame] = {}
        chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
        logger.info(f"Batch fetching {len(tickers)} tickers in {len(chunks)} chunk(s) of {chunk_size}")

        for chunk_idx, chunk in enumerate(chunks):
            for attempt in range(3):
                try:
                    if start_date and end_date:
                        raw = yf.download(chunk, start=start_date, end=end_date,
                                          group_by='ticker', auto_adjust=True, progress=False)
                    else:
                        raw = yf.download(chunk, period=period,
                                          group_by='ticker', auto_adjust=True, progress=False)

                    if raw is None or raw.empty:
                        break

                    for ticker in chunk:
                        try:
                            df = raw[ticker] if len(chunk) > 1 else raw
                            df = df.dropna(how='all').reset_index()

                            if 'Datetime' in df.columns:
                                df.rename(columns={'Datetime': 'Date'}, inplace=True)
                            if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
                                df['Date'] = df['Date'].dt.tz_localize(None)

                            required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                            df = df[[c for c in required if c in df.columns]]

                            if len(df) >= 20:
                                results[ticker] = df
                        except Exception:
                            pass

                    break  # chunk succeeded

                except Exception as e:
                    err = str(e).lower()
                    if ('rate' in err or 'too many' in err or '429' in err) and attempt < 2:
                        wait = 10 * (2 ** attempt)
                        logger.warning(f"Batch chunk {chunk_idx+1}: rate limited, waiting {wait}s")
                        time.sleep(wait)
                    else:
                        logger.warning(f"Batch chunk {chunk_idx+1} failed: {e}")
                        break

            if chunk_idx < len(chunks) - 1:
                time.sleep(1)  # 1s between chunks to be polite

        # yfinance returned nothing — try alternative sources before giving up
        if not results:
            s_date, e_date = (start_date, end_date) if (start_date and end_date) else self._period_to_dates(period)

            if self.tiingo_key:
                logger.warning("yfinance batch returned 0 tickers — trying Tiingo")
                results = self._batch_tiingo(tickers, s_date, e_date)

            if not results and self.finnhub_key:
                logger.warning("Tiingo batch empty — trying Finnhub")
                results = self._batch_finnhub(tickers, s_date, e_date)

        logger.info(f"Batch fetch complete: {len(results)}/{len(tickers)} tickers")
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
    
    # ------------------------------------------------------------------
    # Fallback data source helpers
    # ------------------------------------------------------------------

    def _period_to_dates(self, period: str) -> tuple:
        """Convert yfinance period string to (start_date, end_date) YYYY-MM-DD strings."""
        days = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
        end = datetime.now()
        start = end - timedelta(days=days.get(period, 90))
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    def _fetch_tiingo_single(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV from Tiingo REST API. Free tier: 1000 req/day, 50 req/min, 500 symbols."""
        if not self.tiingo_key:
            return None
        import requests
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        params = {"startDate": start_date, "endDate": end_date, "token": self.tiingo_key,
                  "columns": "date,open,high,low,close,volume"}
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            raw = resp.json()
            if not raw:
                return None
            df = pd.DataFrame(raw)
            df.rename(columns={"date": "Date", "open": "Open", "high": "High",
                                "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            required = ["Date", "Open", "High", "Low", "Close", "Volume"]
            df = df[[c for c in required if c in df.columns]]
            return df if len(df) >= 20 else None
        except Exception as e:
            logger.warning(f"Tiingo failed for {ticker}: {e}")
            return None

    def _fetch_finnhub_single(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV from Finnhub REST API. Free tier: 60 req/min."""
        if not self.finnhub_key:
            return None
        import requests
        from_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        to_ts   = int(datetime.strptime(end_date,   "%Y-%m-%d").timestamp())
        url = "https://finnhub.io/api/v1/stock/candle"
        params = {"symbol": ticker, "resolution": "D", "from": from_ts, "to": to_ts,
                  "token": self.finnhub_key}
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
            if raw.get("s") != "ok" or not raw.get("t"):
                return None
            df = pd.DataFrame({
                "Date":   pd.to_datetime(raw["t"], unit="s"),
                "Open":   raw["o"], "High": raw["h"],
                "Low":    raw["l"], "Close": raw["c"], "Volume": raw["v"],
            })
            return df if len(df) >= 20 else None
        except Exception as e:
            logger.warning(f"Finnhub failed for {ticker}: {e}")
            return None

    def _batch_tiingo(self, tickers: List[str], start_date: str, end_date: str,
                      max_workers: int = 5) -> Dict[str, pd.DataFrame]:
        """Concurrent Tiingo fetch for multiple tickers."""
        if not self.tiingo_key:
            return {}
        results: Dict[str, pd.DataFrame] = {}
        logger.info(f"Tiingo: fetching {len(tickers)} tickers ({start_date} → {end_date})")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._fetch_tiingo_single, t, start_date, end_date): t
                       for t in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    df = future.result()
                    if df is not None:
                        results[ticker] = df
                except Exception:
                    pass
        logger.info(f"Tiingo: {len(results)}/{len(tickers)} tickers fetched")
        return results

    def _batch_finnhub(self, tickers: List[str], start_date: str, end_date: str,
                       max_workers: int = 3) -> Dict[str, pd.DataFrame]:
        """Concurrent Finnhub fetch for multiple tickers (max 60 req/min — keep workers low)."""
        if not self.finnhub_key:
            return {}
        results: Dict[str, pd.DataFrame] = {}
        logger.info(f"Finnhub: fetching {len(tickers)} tickers ({start_date} → {end_date})")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._fetch_finnhub_single, t, start_date, end_date): t
                       for t in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    df = future.result()
                    if df is not None:
                        results[ticker] = df
                except Exception:
                    pass
        logger.info(f"Finnhub: {len(results)}/{len(tickers)} tickers fetched")
        return results

    # ------------------------------------------------------------------

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

