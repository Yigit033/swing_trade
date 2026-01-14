"""
Data updater module for daily stock data updates.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
from .fetcher import DataFetcher
from .storage import DatabaseManager

logger = logging.getLogger(__name__)


class DataUpdater:
    """
    Manages daily updates of stock data.
    
    Attributes:
        fetcher (DataFetcher): Data fetcher instance
        db (DatabaseManager): Database manager instance
    """
    
    def __init__(self, config: Dict):
        """
        Initialize DataUpdater.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.fetcher = DataFetcher(
            source=config['data']['source']
        )
        self.db = DatabaseManager(
            db_path=config['data']['database_path']
        )
        logger.info("DataUpdater initialized")
    
    def initial_download(self, tickers: List[str], days: int = 250) -> Dict[str, bool]:
        """
        Download initial historical data for multiple stocks.
        
        Args:
            tickers: List of ticker symbols
            days: Number of days of historical data
        
        Returns:
            Dictionary mapping ticker to success status
        
        Example:
            >>> updater = DataUpdater(config)
            >>> tickers = ['AAPL', 'MSFT', 'GOOGL']
            >>> results = updater.initial_download(tickers, days=250)
        """
        logger.info(f"Starting initial download for {len(tickers)} stocks ({days} days)")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch data for all stocks
        data_dict = self.fetcher.fetch_multiple_stocks(
            tickers=tickers,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            max_workers=10
        )
        
        # Save to database
        results = {}
        for ticker, df in data_dict.items():
            success = self.db.insert_stock_data(ticker, df)
            results[ticker] = success
            
            # Also fetch and save stock info
            if success:
                info = self.fetcher.get_stock_info(ticker)
                if info:
                    self.db.insert_stock_info(info)
        
        successful = sum(1 for v in results.values() if v)
        logger.info(f"Initial download complete: {successful}/{len(tickers)} successful")
        
        return results
    
    def update_stock_data(self, ticker: str) -> bool:
        """
        Update data for a single stock (fetch only missing dates).
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            True if successful, False otherwise
        
        Example:
            >>> updater = DataUpdater(config)
            >>> success = updater.update_stock_data('AAPL')
        """
        try:
            # Get latest date in database
            latest_date = self.db.get_latest_date(ticker)
            
            if latest_date:
                # Convert to datetime
                latest_dt = datetime.strptime(latest_date, '%Y-%m-%d')
                
                # Check if we need to update
                today = datetime.now().date()
                if latest_dt.date() >= today:
                    logger.debug(f"{ticker} is already up to date")
                    return True
                
                # Fetch data from day after latest date
                start_date = (latest_dt + timedelta(days=1)).strftime('%Y-%m-%d')
                end_date = today.strftime('%Y-%m-%d')
                
                logger.debug(f"Updating {ticker} from {start_date} to {end_date}")
            else:
                # No data exists, fetch last 250 days
                logger.debug(f"No existing data for {ticker}, fetching last 250 days")
                end_date = datetime.now()
                start_date = (end_date - timedelta(days=250)).strftime('%Y-%m-%d')
                end_date = end_date.strftime('%Y-%m-%d')
            
            # Fetch new data
            df = self.fetcher.fetch_stock_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or df.empty:
                logger.warning(f"No new data for {ticker}")
                return False
            
            # Save to database
            success = self.db.insert_stock_data(ticker, df)
            
            if success:
                logger.info(f"Updated {ticker}: {len(df)} new records")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update {ticker}: {e}", exc_info=True)
            return False
    
    def daily_update(self, tickers: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Perform daily update for all stocks or specified list.
        
        Args:
            tickers: Optional list of tickers to update (if None, updates all in database)
        
        Returns:
            Dictionary mapping ticker to success status
        
        Example:
            >>> updater = DataUpdater(config)
            >>> results = updater.daily_update()
            >>> print(f"Updated {sum(results.values())}/{len(results)} stocks")
        """
        if tickers is None:
            tickers = self.db.get_all_tickers()
        
        if not tickers:
            logger.warning("No tickers to update")
            return {}
        
        logger.info(f"Starting daily update for {len(tickers)} stocks")
        
        results = {}
        for ticker in tickers:
            results[ticker] = self.update_stock_data(ticker)
        
        successful = sum(1 for v in results.values() if v)
        logger.info(f"Daily update complete: {successful}/{len(tickers)} successful")
        
        return results
    
    def apply_filters(self, ticker: str) -> bool:
        """
        Check if a stock meets filter criteria.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            True if stock passes filters, False otherwise
        
        Example:
            >>> updater = DataUpdater(config)
            >>> if updater.apply_filters('AAPL'):
            ...     print("AAPL passes all filters")
        """
        try:
            # Get latest data
            df = self.db.get_stock_data(ticker, limit=1)
            if df is None or df.empty:
                return False
            
            latest = df.iloc[-1]
            
            # Get stock info
            info = self.db.get_stock_info(ticker)
            
            # Apply filters from config
            filters = self.config['filters']
            
            # Volume filter
            if latest['Volume'] < filters['min_volume']:
                logger.debug(f"{ticker} failed volume filter: {latest['Volume']}")
                return False
            
            # Price filter
            if not (filters['min_price'] <= latest['Close'] <= filters['max_price']):
                logger.debug(f"{ticker} failed price filter: {latest['Close']}")
                return False
            
            # Market cap filter
            if info and info.get('market_cap', 0) < filters['min_market_cap']:
                logger.debug(f"{ticker} failed market cap filter")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying filters for {ticker}: {e}")
            return False
    
    def get_filtered_tickers(self) -> List[str]:
        """
        Get list of tickers that pass all filters.
        
        Returns:
            List of ticker symbols that meet filter criteria
        
        Example:
            >>> updater = DataUpdater(config)
            >>> valid_tickers = updater.get_filtered_tickers()
            >>> print(f"Found {len(valid_tickers)} valid stocks")
        """
        all_tickers = self.db.get_all_tickers()
        filtered = [ticker for ticker in all_tickers if self.apply_filters(ticker)]
        
        logger.info(f"Filtered {len(all_tickers)} stocks -> {len(filtered)} valid stocks")
        return filtered

