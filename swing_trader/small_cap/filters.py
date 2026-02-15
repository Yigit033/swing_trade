"""
Small Cap Universe Filters - Hard filters for stock selection.
Completely independent from LargeCap filters.

SENIOR TRADER OPTIMIZED v2.0
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SmallCapFilters:
    """
    Universe selection filters for Small Cap Momentum Engine.
    
    SENIOR TRADER SPECS (Agresif):
    - Market Cap: $250M - $2.5B (true small-cap)
    - Float: ≤ 60M shares (EXPLOSION POTENTIAL!)
    - Avg Volume (20d): >= 750K shares
    - ATR%: >= 3.5% (type-specific)
    - Price: $3 - $200
    - Earnings ±7 days: REJECT
    
    Float Tiering:
    - ≤15M:  ATOMIC (+20 pts)
    - 15-30M: MICRO (+15 pts)
    - 30-45M: SMALL (+10 pts)
    - 45-60M: TIGHT (+5 pts)
    - >150M: REJECT
    """
    
    # SENIOR TRADER FILTER CONSTANTS
    MIN_MARKET_CAP = 250_000_000      # $250M (was $300M)
    MAX_MARKET_CAP = 2_500_000_000    # $2.5B (was $3B)
    MIN_AVG_VOLUME = 750_000          # 750K shares (was 1M)
    MIN_ATR_PERCENT = 0.035           # 3.5% minimum
    MAX_FLOAT = 150_000_000            # 150M shares (was 80M - raised based on data analysis)
    IDEAL_FLOAT = 60_000_000          # 60M - main filter for explosion potential
    MIN_PRICE = 3.00                  # $3 (avoid penny stocks)
    MAX_PRICE = 200.00                # $200 (realistic small-cap)
    EARNINGS_EXCLUSION_DAYS = 3       # ±3 days (was 7 - tightened to only block pre-event risk)
    ATR_PERIOD = 10                   # 10-period ATR (was 14, faster reaction)
    
    # FLOAT TIERING THRESHOLDS (for scoring bonus)
    FLOAT_TIER_ATOMIC = 15_000_000    # ≤15M: +20 pts
    FLOAT_TIER_MICRO = 30_000_000     # 15-30M: +15 pts
    FLOAT_TIER_SMALL = 45_000_000     # 30-45M: +10 pts
    FLOAT_TIER_TIGHT = 60_000_000     # 45-60M: +5 pts
    
    def __init__(self, config: Dict = None):
        """Initialize SmallCapFilters."""
        self.config = config or {}
        logger.info("SmallCapFilters initialized (Senior Trader v2.0)")
    
    def calculate_atr_percent(self, df: pd.DataFrame, period: int = None) -> float:
        """Calculate ATR as percentage of close price. Uses 10-period for faster reaction."""
        if period is None:
            period = self.ATR_PERIOD  # Default 10
            
        if df is None or len(df) < period + 1:
            return 0.0
        
        try:
            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values
            
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1])
                )
            )
            
            atr = np.mean(tr[-period:])
            current_close = close[-1]
            
            return atr / current_close if current_close > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating ATR%: {e}")
            return 0.0
    
    def calculate_avg_volume(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate average volume over period."""
        if df is None or len(df) < period:
            return 0.0
        
        try:
            return float(df['Volume'].tail(period).mean())
        except Exception:
            return 0.0
    
    def check_market_cap(self, market_cap: float) -> Tuple[bool, str]:
        """Check if market cap is within small-cap range."""
        if market_cap is None or market_cap <= 0:
            return False, "Unknown market cap"
        
        if market_cap < self.MIN_MARKET_CAP:
            return False, f"Market cap too small (${market_cap/1e6:.0f}M < $250M)"
        
        if market_cap > self.MAX_MARKET_CAP:
            return False, f"Market cap too large (${market_cap/1e9:.1f}B > $2.5B)"
        
        return True, f"Market cap OK (${market_cap/1e6:.0f}M)"
    
    def check_volume(self, avg_volume: float) -> Tuple[bool, str]:
        """Check if average volume meets minimum threshold."""
        if avg_volume < self.MIN_AVG_VOLUME:
            return False, f"Volume too low ({avg_volume/1e6:.2f}M < 750K)"
        return True, f"Volume OK ({avg_volume/1e6:.2f}M)"
    
    def check_atr_percent(self, atr_pct: float) -> Tuple[bool, str]:
        """Check if ATR% meets volatility threshold."""
        if atr_pct < self.MIN_ATR_PERCENT:
            return False, f"ATR% too low ({atr_pct*100:.1f}% < 3.5%)"
        return True, f"ATR% OK ({atr_pct*100:.1f}%)"
    
    def check_float(self, float_shares: float) -> Tuple[bool, str]:
        """Check if float is within small-cap range (≤150M, ideal ≤60M)."""
        if float_shares is None or float_shares <= 0:
            return True, "Float unknown (allowing)"  # Allow if unknown
        
        if float_shares > self.MAX_FLOAT:
            return False, f"Float too large ({float_shares/1e6:.0f}M > 150M)"
        
        # Show tier info
        if float_shares <= self.FLOAT_TIER_ATOMIC:
            return True, f"Float ATOMIC ({float_shares/1e6:.0f}M) +20pts"
        elif float_shares <= self.FLOAT_TIER_MICRO:
            return True, f"Float MICRO ({float_shares/1e6:.0f}M) +15pts"
        elif float_shares <= self.FLOAT_TIER_SMALL:
            return True, f"Float SMALL ({float_shares/1e6:.0f}M) +10pts"
        elif float_shares <= self.FLOAT_TIER_TIGHT:
            return True, f"Float TIGHT ({float_shares/1e6:.0f}M) +5pts"
        elif float_shares <= 80_000_000:
            return True, f"Float OK ({float_shares/1e6:.0f}M) +0pts"
        else:
            return True, f"Float WIDE ({float_shares/1e6:.0f}M) -5pts penalty"
    
    def get_float_tier_bonus(self, float_shares: float) -> int:
        """Get bonus points for float tier. Negative bonus for wide floats."""
        if float_shares is None or float_shares <= 0:
            return 0
        
        if float_shares <= self.FLOAT_TIER_ATOMIC:
            return 20  # ATOMIC
        elif float_shares <= self.FLOAT_TIER_MICRO:
            return 15  # MICRO
        elif float_shares <= self.FLOAT_TIER_SMALL:
            return 10  # SMALL
        elif float_shares <= self.FLOAT_TIER_TIGHT:
            return 5   # TIGHT
        elif float_shares <= 80_000_000:
            return 0   # OK - no bonus no penalty
        elif float_shares <= self.MAX_FLOAT: # 80M < float_shares <= 150M
            return -5  # WIDE - penalty but not rejected
        else:
            return -100 # Rejected by hard filter, but return a large penalty if somehow reached
    
    def check_price(self, price: float) -> Tuple[bool, str]:
        """Check if price is within acceptable range ($3-$200)."""
        if price is None or price <= 0:
            return False, "Unknown price"
        
        if price < self.MIN_PRICE:
            return False, f"Price too low (${price:.2f} < $3 - penny stock risk)"
        
        if price > self.MAX_PRICE:
            return False, f"Price too high (${price:.2f} > $200)"
        
        return True, f"Price OK (${price:.2f})"
    
    def check_earnings(self, ticker: str, signal_date) -> Tuple[bool, str]:
        """Check if stock has earnings within ±3 days (tightened from ±7)."""
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            
            stock = yf.Ticker(ticker)
            
            # Try to get earnings dates
            try:
                earnings_dates = stock.earnings_dates
                if earnings_dates is not None and len(earnings_dates) > 0:
                    if isinstance(signal_date, str):
                        signal_date = datetime.strptime(signal_date, '%Y-%m-%d')
                    
                    for earn_date in earnings_dates.index:
                        earn_dt = earn_date.to_pydatetime().replace(tzinfo=None)
                        days_diff = (earn_dt.date() - signal_date.date()).days
                        
                        # Only block PRE-earnings (upcoming 3 days)
                        if 0 <= days_diff <= self.EARNINGS_EXCLUSION_DAYS:
                            return False, f"Earnings in {days_diff} days (pre-event risk)"
                        
                        # Also block day-of (just happened today)
                        if days_diff == -1:
                            return False, f"Earnings yesterday (wait for dust to settle)"
            except:
                pass
            
            return True, "No nearby earnings"
            
        except Exception as e:
            logger.debug(f"Earnings check error for {ticker}: {e}")
            return True, "Earnings check skipped"
    
    def apply_all_filters(
        self, 
        ticker: str, 
        df: pd.DataFrame, 
        stock_info: Dict,
        signal_date = None
    ) -> Tuple[bool, Dict]:
        """
        Apply all hard filters to a stock.
        
        Returns:
            Tuple of (passed: bool, results: dict with filter details)
        """
        from datetime import datetime
        
        results = {
            'ticker': ticker,
            'passed': False,
            'filters': {}
        }
        
        if signal_date is None:
            signal_date = datetime.now()
        
        # 1. Market Cap
        market_cap = stock_info.get('marketCap', 0) or stock_info.get('market_cap', 0)
        passed, reason = self.check_market_cap(market_cap)
        results['filters']['market_cap'] = {'passed': passed, 'reason': reason, 'value': market_cap}
        if not passed:
            return False, results
        
        # 2. Average Volume
        avg_vol = self.calculate_avg_volume(df)
        passed, reason = self.check_volume(avg_vol)
        results['filters']['avg_volume'] = {'passed': passed, 'reason': reason, 'value': avg_vol}
        if not passed:
            return False, results
        
        # 3. ATR%
        atr_pct = self.calculate_atr_percent(df)
        passed, reason = self.check_atr_percent(atr_pct)
        results['filters']['atr_percent'] = {'passed': passed, 'reason': reason, 'value': atr_pct}
        if not passed:
            return False, results
        
        # 4. Float
        float_shares = stock_info.get('floatShares', 0) or stock_info.get('float_shares', 0)
        passed, reason = self.check_float(float_shares)
        results['filters']['float'] = {'passed': passed, 'reason': reason, 'value': float_shares}
        if not passed:
            return False, results
        
        # 5. Earnings
        passed, reason = self.check_earnings(ticker, signal_date)
        results['filters']['earnings'] = {'passed': passed, 'reason': reason}
        if not passed:
            return False, results
        
        results['passed'] = True
        results['atr_percent'] = atr_pct
        results['avg_volume'] = avg_vol
        results['market_cap'] = market_cap
        results['float_shares'] = float_shares
        
        return True, results
