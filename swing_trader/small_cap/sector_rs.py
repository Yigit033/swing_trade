"""
Sector Relative Strength Module - Compare stock performance to sector ETF.
Senior Trader v2.1 Feature
"""

import logging
from typing import Dict, Optional, Tuple
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

# Sector to ETF mapping
SECTOR_ETF_MAP = {
    # SPDR Sector ETFs
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financial Services': 'XLF',
    'Financials': 'XLF',
    'Consumer Cyclical': 'XLY',
    'Consumer Defensive': 'XLP',
    'Energy': 'XLE',
    'Industrials': 'XLI',
    'Basic Materials': 'XLB',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Utilities': 'XLU',
    'Communication Services': 'XLC',
    'Communication': 'XLC',
    
    # Default fallback
    'Unknown': 'SPY',
}


class SectorRS:
    """
    Calculate Sector Relative Strength for a stock.
    
    Logic:
    1. Get stock's 5-day return
    2. Get sector ETF's 5-day return
    3. RS = (stock_return - sector_return)
    4. If RS > +15%, award +12 bonus points
    
    Why: Sector rotation provides +25% extra momentum in small caps.
    """
    
    # Cache for sector ETF data (avoid repeated API calls)
    _sector_cache: Dict[str, Dict] = {}
    
    @classmethod
    def get_sector_etf(cls, sector: str) -> str:
        """Map sector name to ETF symbol."""
        return SECTOR_ETF_MAP.get(sector, 'SPY')
    
    @classmethod
    def get_5day_return(cls, ticker: str) -> Optional[float]:
        """Get 5-day return for a ticker."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='10d')
            
            if hist is None or len(hist) < 5:
                return None
            
            close_now = hist['Close'].iloc[-1]
            close_5d_ago = hist['Close'].iloc[-5] if len(hist) >= 5 else hist['Close'].iloc[0]
            
            return ((close_now - close_5d_ago) / close_5d_ago) * 100
            
        except Exception as e:
            logger.debug(f"Error getting 5d return for {ticker}: {e}")
            return None
    
    @classmethod
    def calculate_sector_rs(cls, ticker: str, sector: str, ticker_5d_return: float = None) -> Dict:
        """
        Calculate Sector Relative Strength.
        
        Args:
            ticker: Stock ticker
            sector: Stock's sector name
            ticker_5d_return: Pre-calculated 5d return (optional, saves API call)
        
        Returns:
            {
                'sector_etf': str,
                'ticker_5d': float,
                'sector_5d': float,
                'rs_score': float,
                'is_leader': bool,
                'bonus': int
            }
        """
        result = {
            'sector_etf': None,
            'ticker_5d': 0.0,
            'sector_5d': 0.0,
            'rs_score': 0.0,
            'is_leader': False,
            'bonus': 0
        }
        
        try:
            # Get sector ETF
            sector_etf = cls.get_sector_etf(sector)
            result['sector_etf'] = sector_etf
            
            # Get ticker's 5d return
            if ticker_5d_return is not None:
                result['ticker_5d'] = ticker_5d_return
            else:
                result['ticker_5d'] = cls.get_5day_return(ticker) or 0.0
            
            # Get sector ETF's 5d return (use cache if available)
            if sector_etf in cls._sector_cache:
                result['sector_5d'] = cls._sector_cache[sector_etf].get('return', 0.0)
            else:
                sector_return = cls.get_5day_return(sector_etf)
                if sector_return is not None:
                    cls._sector_cache[sector_etf] = {'return': sector_return}
                    result['sector_5d'] = sector_return
            
            # Calculate RS
            result['rs_score'] = result['ticker_5d'] - result['sector_5d']
            
            # Check if sector leader (+15% outperformance)
            if result['rs_score'] > 15:
                result['is_leader'] = True
                result['bonus'] = 12  # Senior Trader: +12 bonus
            elif result['rs_score'] > 10:
                result['bonus'] = 8   # Strong outperformance
            elif result['rs_score'] > 5:
                result['bonus'] = 4   # Moderate outperformance
            elif result['rs_score'] < -10:
                result['bonus'] = -5  # Underperforming sector (penalty)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Sector RS for {ticker}: {e}")
            return result
    
    @classmethod
    def clear_cache(cls):
        """Clear the sector ETF cache."""
        cls._sector_cache.clear()


def get_sector_rs_bonus(ticker: str, sector: str, ticker_5d_return: float = None) -> int:
    """
    Convenience function to get just the bonus points.
    
    Usage in scoring.py:
        from swing_trader.small_cap.sector_rs import get_sector_rs_bonus
        bonus += get_sector_rs_bonus(ticker, sector, five_day_return)
    """
    result = SectorRS.calculate_sector_rs(ticker, sector, ticker_5d_return)
    return result['bonus']
