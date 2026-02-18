"""
Small Cap Signal Triggers - Momentum breakout detection.
Completely independent from LargeCap signals.

SENIOR TRADER OPTIMIZED v2.0
"""

import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SmallCapSignals:
    """
    Signal triggers for Small Cap Momentum Engine.
    
    SENIOR TRADER TRIGGERS (3-Tier):
    
    TIER 1 - MANDATORY:
    - Volume surge >= 1.8x (was 1.5x)
    - ATR% >= 3.5%
    
    TIER 2 - CATALYST BOOSTERS:
    - Gap >= 3% + Volume 2x
    - MACD bullish cross
    - RSI bullish divergence (Game Changer!)
    - Close > VWAP
    
    TIER 3 - TECHNICAL CONFLUENCE:
    - MACD > Signal Line
    - Close > VWAP
    - Higher Highs (3 days)
    """
    
    # SENIOR TRADER SIGNAL CONSTANTS
    MIN_VOLUME_SURGE = 1.2            # 1.2x (was 1.8x — audit showed most stocks rejected at 1.3-1.5x)
    MIN_ATR_PERCENT_TRIGGER = 0.035   # 3.5%
    ATR_PERIOD = 10                   # 10-period ATR (faster)
    GAP_THRESHOLD = 0.03              # 3% gap for catalyst boost
    
    def __init__(self, config: Dict = None):
        """Initialize SmallCapSignals."""
        self.config = config or {}
        logger.info("SmallCapSignals initialized (Senior Trader v2.0)")
    
    def calculate_volume_surge(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate current volume relative to 20-day average."""
        if df is None or len(df) < period + 1:
            return 0.0
        
        try:
            current_vol = df['Volume'].iloc[-1]
            avg_vol = df['Volume'].tail(period + 1).head(period).mean()
            
            return current_vol / avg_vol if avg_vol > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volume surge: {e}")
            return 0.0
    
    def calculate_relative_volume(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate RVOL (same as volume surge but for clarity)."""
        return self.calculate_volume_surge(df, period)
    
    def calculate_atr_percent(self, df: pd.DataFrame, period: int = None) -> float:
        """Calculate ATR as percentage of close price. Uses 10-period."""
        if period is None:
            period = self.ATR_PERIOD
            
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
    
    # ============================================================
    # MACD ANALYSIS (Senior Trader)
    # ============================================================
    def calculate_macd(self, df: pd.DataFrame) -> Dict:
        """
        Calculate MACD indicators.
        Returns: {macd_line, signal_line, histogram, bullish_cross, above_zero}
        """
        result = {
            'macd_line': 0.0,
            'signal_line': 0.0,
            'histogram': 0.0,
            'bullish_cross': False,
            'above_zero': False,
            'expanding': False
        }
        
        if df is None or len(df) < 26:
            return result
        
        try:
            close = df['Close']
            
            # Calculate EMAs
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            
            # MACD line
            macd_line = ema12 - ema26
            
            # Signal line (9-period EMA of MACD)
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            
            # Histogram
            histogram = macd_line - signal_line
            
            # Current values
            result['macd_line'] = float(macd_line.iloc[-1])
            result['signal_line'] = float(signal_line.iloc[-1])
            result['histogram'] = float(histogram.iloc[-1])
            
            # Bullish cross (MACD crosses above signal)
            if len(macd_line) >= 2:
                prev_macd = macd_line.iloc[-2]
                prev_signal = signal_line.iloc[-2]
                curr_macd = macd_line.iloc[-1]
                curr_signal = signal_line.iloc[-1]
                
                result['bullish_cross'] = (prev_macd <= prev_signal) and (curr_macd > curr_signal)
            
            # Above zero line
            result['above_zero'] = result['macd_line'] > 0
            
            # Histogram expanding (bullish)
            if len(histogram) >= 2:
                result['expanding'] = histogram.iloc[-1] > histogram.iloc[-2] and histogram.iloc[-1] > 0
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return result
    
    # ============================================================
    # RSI BULLISH DIVERGENCE (Game Changer!)
    # ============================================================
    def detect_rsi_divergence(self, df: pd.DataFrame, lookback: int = 10) -> Dict:
        """
        Detect RSI Bullish Divergence.
        
        Logic:
        1. Find 2 local price lows in last 10 days
        2. Second low <= First low (price)
        3. Second RSI > First RSI (RSI)
        4. RSI diff >= 5 points
        
        Returns: {divergence_found, rsi_diff, price_diff, confidence}
        """
        result = {
            'divergence_found': False,
            'rsi_diff': 0.0,
            'price_diff': 0.0,
            'confidence': 0,
            'type': None
        }
        
        if df is None or len(df) < lookback + 14:
            return result
        
        try:
            # Calculate RSI for the full period
            rsi_values = self._calculate_rsi_series(df, 14)
            if rsi_values is None or len(rsi_values) < lookback:
                return result
            
            close = df['Close'].values
            lows = df['Low'].values
            
            # Find local lows in price (last 10 days)
            local_lows = []
            for i in range(-lookback, -1):
                if i == -lookback:
                    if lows[i] < lows[i + 1]:
                        local_lows.append(i)
                elif i == -2:
                    if lows[i] < lows[i - 1]:
                        local_lows.append(i)
                else:
                    if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                        local_lows.append(i)
            
            if len(local_lows) < 2:
                return result
            
            # Get first and second low
            first_low_idx = local_lows[-2]
            second_low_idx = local_lows[-1]
            
            first_low_price = lows[first_low_idx]
            second_low_price = lows[second_low_idx]
            first_low_rsi = rsi_values.iloc[first_low_idx]
            second_low_rsi = rsi_values.iloc[second_low_idx]
            
            # Check for bullish divergence
            price_lower = second_low_price <= first_low_price
            rsi_higher = second_low_rsi > first_low_rsi
            rsi_diff = second_low_rsi - first_low_rsi
            
            if price_lower and rsi_higher and rsi_diff >= 5:
                result['divergence_found'] = True
                result['rsi_diff'] = float(rsi_diff)
                result['price_diff'] = float((second_low_price - first_low_price) / first_low_price * 100)
                result['type'] = 'BULLISH'
                
                # Confidence scoring
                if rsi_diff >= 15:
                    result['confidence'] = 3  # Strong
                elif rsi_diff >= 10:
                    result['confidence'] = 2  # Medium
                else:
                    result['confidence'] = 1  # Weak
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting RSI divergence: {e}")
            return result
    
    def _calculate_rsi_series(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI as a series for divergence analysis."""
        try:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return None
    
    # ============================================================
    # VWAP ANALYSIS
    # ============================================================
    def calculate_vwap_position(self, df: pd.DataFrame) -> Dict:
        """
        Calculate VWAP and position relative to it.
        Note: True VWAP is intraday, this is an approximation using daily HLC.
        
        Returns: {vwap, close_vs_vwap, above_vwap, days_above_vwap}
        """
        result = {
            'vwap': 0.0,
            'close_vs_vwap': 0.0,
            'above_vwap': False,
            'days_above_vwap': 0
        }
        
        if df is None or len(df) < 5:
            return result
        
        try:
            # Typical Price * Volume for VWAP approximation
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            cumulative_tp_vol = (typical_price * df['Volume']).rolling(5).sum()
            cumulative_vol = df['Volume'].rolling(5).sum()
            
            vwap = cumulative_tp_vol / cumulative_vol
            
            current_close = df['Close'].iloc[-1]
            current_vwap = vwap.iloc[-1]
            
            result['vwap'] = float(current_vwap)
            result['close_vs_vwap'] = float((current_close - current_vwap) / current_vwap * 100)
            result['above_vwap'] = current_close > current_vwap
            
            # Count consecutive days above VWAP
            days_above = 0
            for i in range(-1, -min(6, len(df)), -1):
                if df['Close'].iloc[i] > vwap.iloc[i]:
                    days_above += 1
                else:
                    break
            result['days_above_vwap'] = days_above
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return result
    
    # ============================================================
    # GAP ANALYSIS
    # ============================================================
    def calculate_gap(self, df: pd.DataFrame) -> Dict:
        """Calculate gap up/down percentage."""
        result = {
            'gap_percent': 0.0,
            'gap_up': False,
            'gap_held': False
        }
        
        if df is None or len(df) < 2:
            return result
        
        try:
            prev_close = df['Close'].iloc[-2]
            today_open = df['Open'].iloc[-1]
            today_close = df['Close'].iloc[-1]
            today_low = df['Low'].iloc[-1]
            
            gap_pct = (today_open - prev_close) / prev_close * 100
            result['gap_percent'] = float(gap_pct)
            result['gap_up'] = gap_pct >= self.GAP_THRESHOLD * 100  # 3%+
            
            # Gap held if close > open and low didn't fill gap
            if result['gap_up']:
                gap_fill_level = prev_close * 1.01  # 1% buffer
                result['gap_held'] = today_low > gap_fill_level and today_close > today_open
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating gap: {e}")
            return result
    
    def check_breakout(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check if Close > Previous day High (momentum breakout)."""
        if df is None or len(df) < 2:
            return False, "Insufficient data"
        
        try:
            current_close = df['Close'].iloc[-1]
            prev_high = df['High'].iloc[-2]
            
            if current_close > prev_high:
                pct_above = (current_close - prev_high) / prev_high * 100
                return True, f"Breakout +{pct_above:.1f}% above prev high"
            else:
                return False, f"No breakout (Close {current_close:.2f} <= Prev High {prev_high:.2f})"
                
        except Exception as e:
            logger.error(f"Error checking breakout: {e}")
            return False, str(e)
    
    def check_volume_surge(self, volume_surge: float) -> Tuple[bool, str]:
        """Check if volume surge meets threshold."""
        if volume_surge >= self.MIN_VOLUME_SURGE:
            return True, f"Volume surge {volume_surge:.1f}x >= 1.2x"
        return False, f"Volume surge {volume_surge:.1f}x < 1.2x"
    
    def check_atr_percent(self, atr_pct: float) -> Tuple[bool, str]:
        """Check if ATR% meets signal trigger threshold."""
        if atr_pct >= self.MIN_ATR_PERCENT_TRIGGER:
            return True, f"ATR% {atr_pct*100:.1f}% >= 4%"
        return False, f"ATR% {atr_pct*100:.1f}% < 4%"
    
    def check_all_triggers(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Check signal conditions for scoring.
        
        NEW APPROACH: No hard triggers - all conditions contribute to quality score.
        This allows signals even in low volatility markets.
        
        Returns:
            Tuple of (triggered: bool, details: dict)
        """
        details = {
            'triggered': False,
            'triggers': {}
        }
        
        # Calculate metrics (no hard failures)
        volume_surge = self.calculate_volume_surge(df)
        atr_pct = self.calculate_atr_percent(df)
        breakout_passed, breakout_reason = self.check_breakout(df)
        
        # Store all metrics
        details['triggers']['volume_surge'] = {
            'passed': volume_surge >= 1.0,  # At least average volume
            'reason': f"Volume surge {volume_surge:.1f}x",
            'value': volume_surge
        }
        
        details['triggers']['atr_percent'] = {
            'passed': atr_pct >= 0.02,  # At least 2% ATR
            'reason': f"ATR% {atr_pct*100:.1f}%",
            'value': atr_pct
        }
        
        details['triggers']['breakout'] = {
            'passed': breakout_passed, 
            'reason': breakout_reason,
            'optional': True
        }
        
        # ALWAYS trigger if we have minimum thresholds
        # Let the quality score determine ranking
        min_vol_ok = volume_surge >= 1.3    # v2.3: At least 30% above average (was 1.0x)
        min_atr_ok = atr_pct >= 0.02        # At least 2% volatility
        
        # ALWAYS store values for display (even if not triggered)
        details['volume_surge'] = volume_surge
        details['atr_percent'] = atr_pct
        details['has_breakout'] = breakout_passed
        
        if min_vol_ok and min_atr_ok:
            details['triggered'] = True
        
        return details['triggered'], details
    
    # ============================================================
    # SWING TRADE CONFIRMATION CHECKS (NEW)
    # These checks ensure we're finding SWING candidates, not spikes
    # ============================================================
    
    def check_five_day_momentum(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Check if 5-day momentum is positive.
        Ensures we're in an uptrend, not catching a falling knife.
        """
        if df is None or len(df) < 6:
            return False, 0.0
        
        try:
            close_today = df['Close'].iloc[-1]
            close_5_days_ago = df['Close'].iloc[-6]
            
            five_day_return = (close_today / close_5_days_ago - 1) * 100
            
            return five_day_return > 0, five_day_return
            
        except Exception as e:
            logger.error(f"Error checking 5-day momentum: {e}")
            return False, 0.0
    
    def check_above_ma20(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Check if Close > 20-day Moving Average.
        Eliminates dead cat bounces and downtrend rallies.
        """
        if df is None or len(df) < 21:
            return False, 0.0
        
        try:
            close_today = df['Close'].iloc[-1]
            ma_20 = df['Close'].rolling(20).mean().iloc[-1]
            
            above_ma = close_today > ma_20
            distance_pct = (close_today / ma_20 - 1) * 100
            
            return above_ma, distance_pct
            
        except Exception as e:
            logger.error(f"Error checking MA20: {e}")
            return False, 0.0
    
    def check_higher_lows(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Check for higher lows pattern in last 3-5 days.
        Strong sign of accumulation and continuation.
        """
        if df is None or len(df) < 4:
            return False, "Insufficient data"
        
        try:
            low_today = df['Low'].iloc[-1]
            low_2_days_ago = df['Low'].iloc[-3]
            low_3_days_ago = df['Low'].iloc[-4]
            
            # Pattern: Recent low > Earlier low
            higher_low = low_today > low_3_days_ago or low_today > low_2_days_ago
            
            if higher_low:
                return True, "Higher lows pattern detected"
            else:
                return False, "No higher lows"
                
        except Exception as e:
            logger.error(f"Error checking higher lows: {e}")
            return False, str(e)
    
    def check_multi_day_volume_surge(self, df: pd.DataFrame) -> Tuple[bool, int]:
        """
        Check if at least 1 day in last 3 days had Volume >= 2x avg.
        Better than single-day check for swing setups.
        """
        if df is None or len(df) < 24:  # Need 20 days for avg + 3 days to check
            return False, 0
        
        try:
            avg_vol = df['Volume'].iloc[-24:-4].mean()  # 20-day avg before last 3 days
            
            surge_days = 0
            for i in range(-3, 0):  # Check last 3 days
                day_vol = df['Volume'].iloc[i]
                if day_vol >= 2 * avg_vol:
                    surge_days += 1
            
            return surge_days >= 1, surge_days
            
        except Exception as e:
            logger.error(f"Error checking multi-day volume: {e}")
            return False, 0
    
    def check_not_overextended(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Check if stock is NOT overextended (chasing protection).
        Returns True if stock is SAFE to enter.
        """
        if df is None or len(df) < 6:
            return False, {}
        
        try:
            result = {'today_change': 0, 'max_single_day': 0, 'five_day_total': 0}
            
            # Today's change
            today_change = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
            result['today_change'] = today_change
            
            # Max single day change in last 3 days
            max_single_day = 0
            for i in range(-3, 0):
                if i-1 >= -len(df):
                    day_change = abs((df['Close'].iloc[i] / df['Close'].iloc[i-1] - 1) * 100)
                    max_single_day = max(max_single_day, day_change)
            result['max_single_day'] = max_single_day
            
            # 5-day total
            five_day_total = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100
            result['five_day_total'] = five_day_total
            
            # SAFE if:
            # - Today < +15%
            # - No single day > +25%
            # - 5-day total between +10% and +40%
            is_safe = (
                today_change <= 15 and
                max_single_day <= 25 and
                10 <= five_day_total <= 40
            )
            
            return is_safe, result
            
        except Exception as e:
            logger.error(f"Error checking overextension: {e}")
            return False, {}
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI for penalty scoring."""
        if df is None or len(df) < period + 2:
            return 50.0  # Neutral
        
        try:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def check_swing_confirmation(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Master swing confirmation check.
        Combines all swing-specific criteria.
        
        Required for swing trade:
        - 5-day momentum > 0
        - Close > 20-day MA
        
        Returns (passed, details)
        """
        details = {
            'five_day_momentum': {},
            'above_ma20': {},
            'higher_lows': {},
            'multi_day_volume': {},
            'overextension': {},
            'rsi': 0,
            'swing_ready': False
        }
        
        # 1. 5-Day Momentum (REQUIRED)
        passed_5d, return_5d = self.check_five_day_momentum(df)
        details['five_day_momentum'] = {'passed': passed_5d, 'return': return_5d}
        
        # 2. Above MA20 (REQUIRED)
        passed_ma, distance = self.check_above_ma20(df)
        details['above_ma20'] = {'passed': passed_ma, 'distance': distance}
        
        # 3. Higher Lows (BOOSTER)
        passed_hl, reason_hl = self.check_higher_lows(df)
        details['higher_lows'] = {'passed': passed_hl, 'reason': reason_hl}
        
        # 4. Multi-Day Volume (BOOSTER)
        passed_vol, surge_days = self.check_multi_day_volume_surge(df)
        details['multi_day_volume'] = {'passed': passed_vol, 'surge_days': surge_days}
        
        # 5. Not Overextended (WARNING)
        is_safe, ext_details = self.check_not_overextended(df)
        details['overextension'] = {'safe': is_safe, 'details': ext_details}
        
        # 6. RSI (for penalty scoring)
        rsi = self.calculate_rsi(df)
        details['rsi'] = rsi
        
        # SWING READY if required conditions met
        details['swing_ready'] = passed_5d and passed_ma
        
        return details['swing_ready'], details
    
    # Optional Boosters
    def check_boosters(self, df: pd.DataFrame) -> Dict:
        """
        Check optional boosters that increase quality score.
        NOW INCLUDES SWING CONFIRMATION CHECKS.
        """
        boosters = {}
        
        # 1. RVOL >= 3
        rvol = self.calculate_relative_volume(df)
        boosters['high_rvol'] = rvol >= 3.0
        boosters['rvol_value'] = rvol
        
        # 2. Gap Up with Continuation
        if len(df) >= 2:
            prev_close = df['Close'].iloc[-2]
            today_open = df['Open'].iloc[-1]
            today_close = df['Close'].iloc[-1]
            
            gap_pct = (today_open - prev_close) / prev_close
            continuation = today_close > today_open
            
            boosters['gap_continuation'] = gap_pct > 0.02 and continuation
            boosters['gap_percent'] = gap_pct
        else:
            boosters['gap_continuation'] = False
            boosters['gap_percent'] = 0
        
        # 3. Higher High
        if len(df) >= 3:
            today_high = df['High'].iloc[-1]
            prev_high = df['High'].iloc[-2]
            prev2_high = df['High'].iloc[-3]
            
            boosters['higher_highs'] = today_high > prev_high > prev2_high
        else:
            boosters['higher_highs'] = False
        
        # 4. SWING CONFIRMATION (NEW)
        swing_ready, swing_details = self.check_swing_confirmation(df)
        boosters['swing_ready'] = swing_ready
        boosters['swing_details'] = swing_details
        boosters['higher_lows'] = swing_details.get('higher_lows', {}).get('passed', False)
        boosters['multi_day_volume'] = swing_details.get('multi_day_volume', {}).get('passed', False)
        boosters['rsi'] = swing_details.get('rsi', 50)
        
        return boosters

