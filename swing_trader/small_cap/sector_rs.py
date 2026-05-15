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
    def _get_multi_period_return(cls, ticker: str) -> Dict[str, Optional[float]]:
        """Get 5d, 10d, 20d returns in a single API call (efficient)."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='25d')
            if hist is None or len(hist) < 5:
                return {'5d': None, '10d': None, '20d': None}
            closes = hist['Close']
            def ret(n):
                return ((float(closes.iloc[-1]) / float(closes.iloc[-n]) - 1) * 100
                        if len(closes) >= n else None)
            return {'5d': ret(5), '10d': ret(10), '20d': ret(20)}
        except Exception:
            return {'5d': None, '10d': None, '20d': None}

    @classmethod
    def calculate_sector_rs(cls, ticker: str, sector: str, ticker_5d_return: float = None) -> Dict:
        """
        Calculate Sector Relative Strength using a weighted multi-period RS score.

        v2.2: Uses 5d (50%) + 10d (30%) + 20d (20%) weighted composite instead of
        5d alone. This prevents a stock that underperformed for 3 weeks but had a
        good 2-day bounce from looking like a "sector leader."

        Returns:
            {
                'sector_etf': str,
                'ticker_5d': float,
                'sector_5d': float,
                'rs_score': float,      # weighted composite RS
                'rs_5d': float,
                'rs_10d': float,
                'rs_20d': float,
                'is_leader': bool,
                'bonus': int
            }
        """
        result = {
            'sector_etf': None,
            'ticker_5d': 0.0,
            'sector_5d': 0.0,
            'rs_score': 0.0,
            'rs_5d': 0.0,
            'rs_10d': 0.0,
            'rs_20d': 0.0,
            'is_leader': False,
            'bonus': 0
        }

        try:
            sector_etf = cls.get_sector_etf(sector)
            result['sector_etf'] = sector_etf

            # Get ticker multi-period returns
            ticker_returns = cls._get_multi_period_return(ticker)
            t5 = ticker_returns['5d'] if ticker_returns['5d'] is not None else (ticker_5d_return or 0.0)
            t10 = ticker_returns['10d'] or 0.0
            t20 = ticker_returns['20d'] or 0.0
            result['ticker_5d'] = t5

            # Get sector ETF multi-period returns (cached per session)
            if sector_etf in cls._sector_cache:
                s_returns = cls._sector_cache[sector_etf]
            else:
                s_returns = cls._get_multi_period_return(sector_etf)
                cls._sector_cache[sector_etf] = s_returns

            s5 = s_returns.get('5d') or 0.0
            s10 = s_returns.get('10d') or 0.0
            s20 = s_returns.get('20d') or 0.0
            result['sector_5d'] = s5

            # Per-period RS
            rs5 = t5 - s5
            rs10 = t10 - s10
            rs20 = t20 - s20
            result['rs_5d'] = rs5
            result['rs_10d'] = rs10
            result['rs_20d'] = rs20

            # Weighted composite: recent periods carry more weight,
            # but all three must agree for "leader" status
            composite_rs = rs5 * 0.50 + rs10 * 0.30 + rs20 * 0.20
            result['rs_score'] = round(composite_rs, 2)

            # Only award leader bonus when ALL periods positive (genuine sustained leader)
            all_positive = rs5 > 0 and rs10 > 0 and rs20 > 0
            if composite_rs > 15 and all_positive:
                result['is_leader'] = True
                result['bonus'] = 12
            elif composite_rs > 10 and all_positive:
                result['bonus'] = 8
            elif composite_rs > 5:
                result['bonus'] = 4
            elif composite_rs < -10:
                result['bonus'] = -5

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
