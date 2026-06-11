"""
Small Cap Signal Triggers - Momentum breakout detection.
Completely independent from LargeCap signals.

SENIOR TRADER OPTIMIZED v2.0
"""

import logging
from typing import TYPE_CHECKING, Dict, Optional, Tuple
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from .settings_config import SmallCapSettings

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
    MIN_VOLUME_SURGE = 1.2            # Default soft check; live from settings.min_volume_surge_soft
    MIN_ATR_PERCENT_TRIGGER = 0.03    # Default; live from settings.min_atr_percent
    ATR_PERIOD = 10                   # 10-period ATR (faster)
    GAP_THRESHOLD = 0.03              # 3% gap for catalyst boost
    
    def __init__(self, config: Dict = None, settings: Optional["SmallCapSettings"] = None):
        """Initialize SmallCapSignals."""
        from .settings_config import load_settings

        self.config = config or {}
        self._settings = settings if settings is not None else load_settings()
        scfg = self._settings.signal_confirmation
        self._overext_today_max = scfg.overext_today_change_max
        self._overext_single_day_max = scfg.overext_single_day_max
        self._overext_five_day_total_max = scfg.overext_five_day_total_max
        self._ma20_max_below_pct = scfg.ma20_max_distance_below_pct
        self.ATR_PERIOD = self._settings.universe_filters.atr_period
        logger.info("SmallCapSignals initialized (Senior Trader v2.0)")
    
    def calculate_volume_surge(self, df: pd.DataFrame, period: int = 50) -> float:
        """
        Calculate current volume relative to the N-day MEDIAN baseline.

        Default period = 50 days (not 20) so that Finviz momentum stocks —
        which already have an elevated 20-day baseline from a recent rally —
        are compared against their longer-term "normal" activity level.

        Example: BKKT was trading 1.5M/day before a rally, then 5M/day for 3 weeks.
        - 20-day median baseline → ~4.5M  →  today's 7M = 1.56x  (no trigger at 2.0x)
        - 50-day median baseline → ~2.5M  →  today's 7M = 2.80x  (triggers cleanly)

        Median (not mean) prevents single spike days from inflating the baseline.
        """
        if df is None or len(df) < period + 1:
            # Graceful fallback: use whatever bars we have (min 20+1)
            if df is None or len(df) < 22:
                return 0.0
            period = len(df) - 1

        try:
            current_vol = df['Volume'].iloc[-1]
            baseline = df['Volume'].tail(period + 1).head(period).median()

            return current_vol / baseline if baseline > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating volume surge: {e}")
            return 0.0

    def calculate_relative_volume(self, df: pd.DataFrame, period: int = 50) -> float:
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
        """
        Professional breakout detection — multi-criteria check.
        
        Criteria:
        1. Close > 5-bar rolling high (not just 1 prev bar)
        2. Close in upper 40% of today's range (close strength)
        3. Volume on breakout bar >= 1.2x average (volume confirmation)
        4. Minimum 0.3% above breakout level (noise filter)
        """
        if df is None or len(df) < 7:
            return False, "Insufficient data"
        
        try:
            current_close = float(df['Close'].iloc[-1])
            current_high = float(df['High'].iloc[-1])
            current_low = float(df['Low'].iloc[-1])
            current_vol = float(df['Volume'].iloc[-1])
            
            # 1. BREAKOUT LEVEL: 5-bar rolling high (excluding today)
            lookback_highs = df['High'].iloc[-6:-1]  # 5 bars before today
            breakout_level = float(lookback_highs.max())
            
            # Check basic price breakout
            if current_close <= breakout_level:
                return False, (
                    f"No breakout (Close {current_close:.2f} <= "
                    f"5-Bar High {breakout_level:.2f})"
                )
            
            # 2. MINIMUM % ABOVE BREAKOUT LEVEL (noise filter)
            pct_above = (current_close - breakout_level) / breakout_level * 100
            if pct_above < 0.3:
                return False, (
                    f"Breakout too small (+{pct_above:.2f}% < 0.3% min, "
                    f"Close {current_close:.2f} vs Level {breakout_level:.2f})"
                )
            
            # 3. CLOSE STRENGTH (upper 40% of range)
            day_range = current_high - current_low
            close_position = 0.5
            if day_range > 0:
                close_position = (current_close - current_low) / day_range
                if close_position < 0.40:
                    return False, (
                        f"Weak close ({close_position:.0%} of range, need 40%+). "
                        f"Close {current_close:.2f}, Range {current_low:.2f}-{current_high:.2f}"
                    )
            
            # 4. VOLUME CONFIRMATION
            vol_window = df['Volume'].iloc[-21:-1] if len(df) >= 21 else df['Volume'].iloc[:-1]
            avg_vol_20 = float(vol_window.mean())
            vol_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 0
            
            if vol_ratio < 1.2:
                return False, (
                    f"Low volume breakout ({vol_ratio:.1f}x avg, need 1.2x+). "
                    f"Close {current_close:.2f} > Level {breakout_level:.2f} (+{pct_above:.1f}%)"
                )
            
            # ALL PASSED - VALID BREAKOUT
            return True, (
                f"Breakout +{pct_above:.1f}% above 5-bar high ${breakout_level:.2f} | "
                f"Close strength {close_position:.0%} | Vol {vol_ratio:.1f}x"
            )
            
        except Exception as e:
            logger.error(f"Error checking breakout: {e}")
            return False, str(e)
    
    
    def check_volume_surge(self, volume_surge: float) -> Tuple[bool, str]:
        """Check if volume surge meets threshold."""
        soft = self._settings.min_volume_surge_soft
        if volume_surge >= soft:
            return True, f"Volume surge {volume_surge:.1f}x >= {soft}x"
        return False, f"Volume surge {volume_surge:.1f}x < {soft}x"
    
    def check_atr_percent(self, atr_pct: float) -> Tuple[bool, str]:
        """Check if ATR% meets signal trigger threshold."""
        thr = self._settings.min_atr_percent
        threshold_pct = thr * 100
        if atr_pct >= thr:
            return True, f"ATR% {atr_pct*100:.1f}% >= {threshold_pct:.1f}%"
        return False, f"ATR% {atr_pct*100:.1f}% < {threshold_pct:.1f}%"
    
    def check_continuation_setup(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Trend continuation entry — stock already in established uptrend,
        today extends the trend with strength (no fresh breakout needed).

        Senior trader logic: in bull regimes most setups are NOT fresh breakouts
        but continuation moves on established trends (Minervini Stage 2, O'Neil
        leadership). Volume/breakout pathways alone systematically miss these
        because Finviz pre-screens for stocks already in motion — they made
        their 5-bar high days ago and are now consolidating up.

        Requires (all must pass):
        - Close > MA20  (in uptrend, not just a dead-cat bounce)
        - Close > prev_close  (green day, momentum holding)
        - Close in upper 50% of today's range  (close strength, not weak fade)
        - MA20 distance <= 12%  (not parabolic / overextended above support)

        Downstream gates (swing_confirmation, RSI, distribution, quality bar)
        filter out the weak continuations.
        """
        if df is None or len(df) < 21:
            return False, "Insufficient data"
        try:
            close_today = float(df['Close'].iloc[-1])
            close_prev = float(df['Close'].iloc[-2])
            high_today = float(df['High'].iloc[-1])
            low_today = float(df['Low'].iloc[-1])
            ma20 = float(df['Close'].rolling(20).mean().iloc[-1])

            if close_today <= ma20:
                return False, f"Below MA20 ({close_today:.2f} <= {ma20:.2f})"

            if close_today <= close_prev:
                return False, f"Red/flat day (close {close_today:.2f} <= prev {close_prev:.2f})"

            day_range = high_today - low_today
            if day_range > 0:
                close_pos = (close_today - low_today) / day_range
                if close_pos < 0.5:
                    return False, f"Weak close ({close_pos:.0%} of range, need 50%+)"

            ma20_dist_pct = (close_today / ma20 - 1) * 100
            if ma20_dist_pct > 12.0:
                return False, f"Overextended above MA20 (+{ma20_dist_pct:.1f}% > 12%)"

            return True, (
                f"Continuation: +{ma20_dist_pct:.1f}% above MA20, "
                f"green +{(close_today/close_prev-1)*100:.1f}%, close strong"
            )
        except Exception as e:
            logger.error(f"Error checking continuation: {e}")
            return False, str(e)

    # ============================================================
    # VOLATILITY CONTRACTION → EXPANSION BREAKOUT (VCE) — v13 PRIMARY
    # ============================================================
    # Empirically validated edge (scripts/measure_signal_edge.py +
    # scripts/test_edge_hypotheses.py + scripts/validate_volsqueeze.py,
    # 57 small/mid-caps, 2024-06 → 2026-05, 24k bar benchmark):
    #   R10 edge +5.16% vs random entry, Welch t=2.6
    #   Out-of-sample (2025-06+): R10 edge +6.17%, t=2.29 — edge holds on unseen data
    # All other tested pathways (early accumulation, fresh 5-bar breakout,
    # plain continuation, VCP, oversold bounce, new-high momentum) measured
    # at zero or NEGATIVE edge on the same benchmark.
    #
    # The constants below are the exact validated rule — do not tune them
    # without re-running the measurement harness.
    VCE_ATR_PERIOD = 14            # ATR% via TR.rolling(14).mean() / Close
    VCE_SQUEEZE_RATIO = 0.8        # yesterday ATR% < 80% of baseline = squeeze
    VCE_BASELINE_OFFSET = (20, 5)  # baseline = mean ATR% of bars [-20, -5) before today
    VCE_BREAKOUT_LOOKBACK = 20     # close must exceed prior 20-day high
    VCE_VOLUME_MULT = 1.5          # breakout volume >= 1.5x 20-day average
    VCE_CLOSE_POS_MIN = 0.6        # close in upper 40% of day range

    def check_vce_breakout(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """
        Volatility Contraction→Expansion breakout — the system's PRIMARY entry.

        The spring: volatility (ATR%) dries up to <80% of its recent baseline,
        then price breaks the prior 20-day high in an uptrend (>MA50) on
        1.5x volume with a strong close. We catch the expansion as it starts,
        not after the move has run.

        Returns (passed, reason, metrics).
        """
        metrics = {
            'atr_now_pct': 0.0,
            'atr_baseline_pct': 0.0,
            'squeeze_ratio': 0.0,
            'breakout_level': 0.0,
            'vol_ratio': 0.0,
            'close_position': 0.0,
        }
        if df is None or len(df) < 55:
            return False, "Insufficient data (<55 bars)", metrics

        try:
            close = df['Close'].astype(float)
            high = df['High'].astype(float)
            low = df['Low'].astype(float)
            volume = df['Volume'].astype(float)

            # ATR% series (14-period SMA of true range, as % of close)
            tr = pd.concat(
                [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
                axis=1,
            ).max(axis=1)
            atr_pct = tr.rolling(self.VCE_ATR_PERIOD).mean() / close * 100

            # 1. SQUEEZE: yesterday's ATR% vs baseline (bars -20..-6, before the move)
            atr_now = float(atr_pct.iloc[-2])
            off_far, off_near = self.VCE_BASELINE_OFFSET
            atr_base = float(atr_pct.iloc[-1 - off_far:-1 - off_near].mean())
            metrics['atr_now_pct'] = round(atr_now, 2)
            metrics['atr_baseline_pct'] = round(atr_base, 2)
            if np.isnan(atr_now) or np.isnan(atr_base) or atr_base <= 0:
                return False, "ATR baseline unavailable", metrics
            squeeze_ratio = atr_now / atr_base
            metrics['squeeze_ratio'] = round(squeeze_ratio, 2)
            if squeeze_ratio >= self.VCE_SQUEEZE_RATIO:
                return False, (
                    f"No squeeze (ATR {atr_now:.1f}% = {squeeze_ratio:.0%} of baseline "
                    f"{atr_base:.1f}%, need <{self.VCE_SQUEEZE_RATIO:.0%})"
                ), metrics

            # 2. BREAKOUT: close above prior 20-day high (excluding today)
            c = float(close.iloc[-1])
            breakout_level = float(high.iloc[-1 - self.VCE_BREAKOUT_LOOKBACK:-1].max())
            metrics['breakout_level'] = round(breakout_level, 2)
            if c <= breakout_level:
                return False, (
                    f"No breakout (close {c:.2f} <= 20d high {breakout_level:.2f})"
                ), metrics

            # 3. GREEN DAY
            if c <= float(close.iloc[-2]):
                return False, "Red/flat day on breakout bar", metrics

            # 4. TREND: above MA50
            ma50 = float(close.rolling(50).mean().iloc[-1])
            if np.isnan(ma50) or c <= ma50:
                return False, f"Below MA50 ({c:.2f} <= {ma50:.2f})", metrics

            # 5. VOLUME: 1.5x 20-day average
            vol20 = float(volume.rolling(20).mean().iloc[-1])
            vol_ratio = float(volume.iloc[-1]) / vol20 if vol20 > 0 else 0.0
            metrics['vol_ratio'] = round(vol_ratio, 2)
            if vol_ratio < self.VCE_VOLUME_MULT:
                return False, (
                    f"Volume too low ({vol_ratio:.1f}x < {self.VCE_VOLUME_MULT}x)"
                ), metrics

            # 6. STRONG CLOSE: upper 40% of day's range
            h, l = float(high.iloc[-1]), float(low.iloc[-1])
            day_range = h - l
            close_pos = (c - l) / day_range if day_range > 0 else 0.5
            metrics['close_position'] = round(close_pos, 2)
            if close_pos < self.VCE_CLOSE_POS_MIN:
                return False, (
                    f"Weak close ({close_pos:.0%} of range, need {self.VCE_CLOSE_POS_MIN:.0%}+)"
                ), metrics

            return True, (
                f"VCE: squeeze {squeeze_ratio:.0%} of baseline → breakout above "
                f"${breakout_level:.2f} | vol {vol_ratio:.1f}x | close {close_pos:.0%}"
            ), metrics

        except Exception as e:
            logger.error(f"Error checking VCE breakout: {e}")
            return False, str(e), metrics

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

        # ============================================================
        # v13 — VCE IS THE PRIMARY (AND ONLY) SIGNAL-GENERATING TRIGGER
        # ============================================================
        # Edge measurement (2024-06→2026-05, 24k bar benchmark) showed the
        # previous pathways produce ZERO or NEGATIVE edge vs random entry:
        #   volume_ignition / early accumulation: R5 edge -1.17% (t=-1.83)
        #   technical_breakout (5-bar high):      no measurable edge
        #   trend_continuation:                   R5 edge +0.29% (t=0.65, noise)
        # Only VCE (volatility squeeze → expansion breakout) survived
        # out-of-sample testing: R10 edge +5.2%, t=2.6. So VCE decides;
        # legacy metrics are still computed for display/scoring context.
        #
        # NOTE: the old `min_atr_ok` hard gate is intentionally NOT applied
        # to the VCE pathway — a squeezed stock has LOW recent ATR by
        # definition; requiring ATR>=3% would contradict the validated rule.
        vce_passed, vce_reason, vce_metrics = self.check_vce_breakout(df)

        # Context metrics (display + downstream scoring; NOT trigger decisions)
        volume_surge = self.calculate_volume_surge(df, period=self._settings.volume_surge_baseline_days)
        atr_pct = self.calculate_atr_percent(df)
        breakout_passed, breakout_reason = self.check_breakout(df)
        continuation_passed, continuation_reason = self.check_continuation_setup(df)

        vol_need = self._settings.volume_surge_trigger
        atr_need = self._settings.min_atr_percent

        details['triggers']['vce_breakout'] = {
            'passed': vce_passed,
            'reason': vce_reason,
            'metrics': vce_metrics,
            'primary': True,
        }
        details['triggers']['volume_surge'] = {
            'passed': volume_surge >= vol_need,
            'reason': f"Volume surge {volume_surge:.1f}x (need {vol_need}x)",
            'value': volume_surge,
            'optional': True
        }
        details['triggers']['atr_percent'] = {
            'passed': atr_pct >= atr_need,
            'reason': f"ATR% {atr_pct*100:.1f}% (need {atr_need*100:.1f}%)",
            'value': atr_pct,
            'optional': True
        }
        details['triggers']['breakout'] = {
            'passed': breakout_passed,
            'reason': breakout_reason,
            'optional': True
        }
        details['triggers']['continuation'] = {
            'passed': continuation_passed,
            'reason': continuation_reason,
            'optional': True
        }

        # ALWAYS store values for display (even if not triggered)
        details['volume_surge'] = volume_surge
        details['atr_percent'] = atr_pct
        details['has_breakout'] = breakout_passed
        details['has_continuation'] = continuation_passed
        details['vce_metrics'] = vce_metrics

        if vce_passed:
            details['triggered'] = True
            details['trigger_pathway'] = 'vce_breakout'
            details['trigger_reason'] = vce_reason

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
        V4: Proper 3-bar ascending lows staircase over the last 5 days.
        Each of the last 3 lows must be strictly higher than the previous.
        """
        if df is None or len(df) < 5:
            return False, "Insufficient data"

        try:
            lows = df['Low'].iloc[-5:].values
            # Need at least 3 consecutively higher lows in the 5-bar window
            ascending_count = 0
            for i in range(1, len(lows)):
                if lows[i] > lows[i - 1]:
                    ascending_count += 1
                else:
                    ascending_count = 0
                if ascending_count >= 3:
                    return True, "3+ consecutive higher lows (strong accumulation)"

            return False, "No consistent higher lows"

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
            
            is_safe = (
                today_change <= self._overext_today_max
                and max_single_day <= self._overext_single_day_max
                and five_day_total <= self._overext_five_day_total_max
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
        
        Required for swing trade (v5.0 — Trend Quality Enhanced):
        - 5-day momentum > 0
        - Close > 20-day MA (or within tolerance)
        - MA20 slope must be rising (NEW — prevents entering fading trends)
        - Close must not be too far below MA50 (NEW — long-term downtrend rejection)
        - No rejection candle / bull trap (NEW — distribution candle detection)
        
        Returns (passed, details)
        """
        from .trend_quality import calculate_trend_quality

        details = {
            'five_day_momentum': {},
            'above_ma20': {},
            'higher_lows': {},
            'multi_day_volume': {},
            'overextension': {},
            'trend_quality': {},
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

        # ================================================================
        # 7. TREND QUALITY ANALYSIS (v5.0 — Directional Gates)
        # ================================================================
        trend = calculate_trend_quality(
            df,
            ma20_slope_lookback=5,
            ma50_max_below_pct=self._settings.signal_confirmation.ma50_max_below_pct,
        )
        details['trend_quality'] = trend
        
        # SWING READY: composite check
        ma20_distance = details["above_ma20"].get("distance", 0)
        ma20_ok = passed_ma or ma20_distance >= -self._ma20_max_below_pct

        # MA50 gate: Must not be too far below MA50 (rejects long-term downtrend bounces)
        ma50_ok = trend.get("ma50_ok", True)

        # Soft signals (penalized in scoring but NOT hard gates):
        # MA20 slope and rejection candle are scoring factors only.
        # Removing them from hard gate prevents over-filtering after market corrections —
        # post-selloff many stocks have flat/declining MA20 and recent red candles even
        # while starting to turn bullish (exactly the Type C early entry setup).
        ma20_slope_ok = trend.get("ma20_slope_ok", True)
        no_rejection = not trend.get("rejection_candle", False)

        details['swing_ready'] = (
            passed_5d
            and ma20_ok
            and ma50_ok
        )

        # Log all gate results for debugging (soft gates shown as advisory only)
        if not details['swing_ready']:
            fail_reasons = []
            if not passed_5d:
                fail_reasons.append(f"5d_mom={return_5d:+.1f}%")
            if not ma20_ok:
                fail_reasons.append(f"MA20_dist={ma20_distance:+.1f}%")
            if not ma50_ok:
                fail_reasons.append(f"MA50_dist={trend.get('ma50_distance_pct', 0):+.1f}%")
            logger.debug("Swing confirmation failed: %s", " | ".join(fail_reasons))
        else:
            # Log soft advisory warnings (don't block, but will penalize score)
            soft_warnings = []
            if not ma20_slope_ok:
                soft_warnings.append(f"MA20_slope={trend.get('ma20_slope_value', 0):+.3f}%")
            if not no_rejection:
                rej = trend.get("rejection_details", {})
                soft_warnings.append(
                    f"rejection_candle(close_pos={rej.get('close_position', 0):.2f})"
                )
            if soft_warnings:
                logger.debug("Swing soft warnings (scoring penalty applied): %s", " | ".join(soft_warnings))
        
        return details['swing_ready'], details
    
    # ============================================================
    # PULLBACK-TO-MA20 ENTRY DETECTION (v6.0 — High Win-Rate)
    # ============================================================
    def detect_pullback_setup(self, df: pd.DataFrame) -> Dict:
        """
        Detect pullback-to-MA20 entry setup — higher win-rate alternative to breakout.

        A pullback entry is the opposite of a breakout entry:
        - Breakout: price explodes above resistance on HIGH volume → 20-35% win rate
        - Pullback: price dips to MA20 support on LOW volume, then bounces → 40-55% win rate

        Required criteria (all must pass):
        1. Above MA50: prior uptrend context (no catching falling knives)
        2. MA20 touch: price Low was within 5% below MA20 in last 5 bars
        3. Bounce today: green candle (Close > Open)
        4. Close above MA20: bounce is confirmed (not a dead-cat)
        5. RSI: 33-65 (oversold bounce OK, but not overbought)

        Bonus criteria (improve quality tier):
        - Low pullback volume (5d avg < 90% of 20d baseline = healthy dry-up)
        - Strong close position (upper 60%+ of today's range)
        - Volume expansion on bounce (today >= 1.2x recent pullback average)
        - RSI < 52 (more room to run before hitting overbought)
        """
        result = {
            'detected': False,
            'quality': '',
            'bonus': 0,
            'above_ma50': False,
            'ma20_touched': False,
            'pullback_volume_ok': False,
            'bounce_today': False,
            'close_above_ma20': False,
            'rsi': 50.0,
            'reason': ''
        }

        if df is None or len(df) < 52:
            result['reason'] = 'Insufficient data'
            return result

        try:
            close = df['Close']
            low = df['Low']
            high = df['High']
            volume = df['Volume']

            ma20 = close.rolling(20).mean()
            ma50 = close.rolling(50).mean()

            current_close = float(close.iloc[-1])
            current_open = float(df['Open'].iloc[-1])
            current_low = float(low.iloc[-1])
            current_high = float(high.iloc[-1])
            current_ma20 = float(ma20.iloc[-1])
            current_ma50 = float(ma50.iloc[-1])

            # 1. Above MA50 (uptrend context required)
            above_ma50 = current_close > current_ma50
            result['above_ma50'] = above_ma50
            if not above_ma50:
                result['reason'] = f'Below MA50 ({current_close:.2f} < {current_ma50:.2f})'
                return result

            # 2. MA20 touch: any of last 5 bars had Low within 5% below MA20
            ma20_touched = False
            for i in range(-5, 0):
                bar_low = float(low.iloc[i])
                bar_ma20 = float(ma20.iloc[i])
                if bar_ma20 > 0:
                    distance_pct = (bar_low - bar_ma20) / bar_ma20 * 100
                    if -5.0 <= distance_pct <= 1.5:  # within 5% below or slightly above
                        ma20_touched = True
                        break
            result['ma20_touched'] = ma20_touched
            if not ma20_touched:
                result['reason'] = 'No MA20 touch in last 5 bars'
                return result

            # 3. Bounce today: green candle
            bounce_today = current_close > current_open
            result['bounce_today'] = bounce_today
            if not bounce_today:
                result['reason'] = 'No bounce today (red candle)'
                return result

            # 4. Close above MA20 (confirmed bounce off the MA)
            close_above_ma20 = current_close >= current_ma20 * 0.995  # 0.5% tolerance
            result['close_above_ma20'] = close_above_ma20
            if not close_above_ma20:
                result['reason'] = f'Close below MA20 ({current_close:.2f} < {current_ma20:.2f})'
                return result

            # 5. RSI check
            rsi = self.calculate_rsi(df)
            result['rsi'] = rsi
            if rsi > 65 or rsi < 33:
                result['reason'] = f'RSI out of range ({rsi:.1f})'
                return result

            # All required criteria met
            result['detected'] = True

            # — Quality scoring —
            ma20_distance = (current_close - current_ma20) / current_ma20 * 100

            # Pullback volume dry-up (last 5d vs 20d baseline)
            pullback_volume_ok = False
            pullback_vol_ratio = 1.0
            if len(volume) >= 25:
                recent_5d_vol = float(volume.iloc[-5:].mean())
                baseline_20d = float(volume.iloc[-25:-5].mean())
                if baseline_20d > 0:
                    pullback_vol_ratio = recent_5d_vol / baseline_20d
                    pullback_volume_ok = pullback_vol_ratio < 0.90
            result['pullback_volume_ok'] = pullback_volume_ok

            # Volume expansion on bounce day
            today_vol_ratio = 1.0
            if len(volume) >= 6:
                recent_avg = float(volume.iloc[-6:-1].mean())
                today_vol = float(volume.iloc[-1])
                if recent_avg > 0:
                    today_vol_ratio = today_vol / recent_avg

            # Close strength (upper portion of range)
            day_range = current_high - current_low
            close_pos = (current_close - current_low) / day_range if day_range > 0 else 0.5

            if (pullback_volume_ok and rsi < 52 and ma20_distance < 2.5
                    and today_vol_ratio >= 1.3 and close_pos >= 0.65):
                result['quality'] = 'perfect'
                result['bonus'] = 20
                result['reason'] = (
                    f'Perfect pullback: RSI={rsi:.0f}, '
                    f'MA20_dist={ma20_distance:+.1f}%, vol_bounce={today_vol_ratio:.1f}x'
                )
            elif rsi < 58 and close_pos >= 0.55 and (pullback_volume_ok or today_vol_ratio >= 1.2):
                result['quality'] = 'clean'
                result['bonus'] = 15
                result['reason'] = (
                    f'Clean pullback: RSI={rsi:.0f}, close_pos={close_pos:.0%}'
                )
            else:
                result['quality'] = 'basic'
                result['bonus'] = 8
                result['reason'] = f'Basic pullback: RSI={rsi:.0f}, MA20 touched'

            return result

        except Exception as e:
            logger.error(f'Error detecting pullback setup: {e}')
            result['reason'] = str(e)
            return result

    # ============================================================
    # OBV TREND ANALYSIS (v3.0 — Smart Money Detection)
    # ============================================================
    def calculate_obv_trend(self, df: pd.DataFrame, period: int = 20) -> Dict:
        """
        Calculate On-Balance Volume trend slope.

        Positive OBV slope while price consolidates = smart money accumulation.
        Negative OBV slope while price rises = distribution (warning!).

        Returns:
            {
                'obv_slope': float,       # Normalized slope (-1 to +1)
                'obv_rising': bool,       # OBV trending up
                'accumulation': bool,     # OBV up + price flat/down = smart money
                'distribution': bool,     # OBV down + price up = warning
                'bonus': int              # Scoring bonus (-5 to +8)
            }
        """
        result = {
            'obv_slope': 0.0,
            'obv_rising': False,
            'accumulation': False,
            'distribution': False,
            'bonus': 0
        }

        if df is None or len(df) < period + 2:
            return result

        try:
            close = df['Close'].values
            volume = df['Volume'].values

            # Calculate OBV
            obv = np.zeros(len(close))
            for i in range(1, len(close)):
                if close[i] > close[i - 1]:
                    obv[i] = obv[i - 1] + volume[i]
                elif close[i] < close[i - 1]:
                    obv[i] = obv[i - 1] - volume[i]
                else:
                    obv[i] = obv[i - 1]

            # Calculate OBV slope over last `period` bars (linear regression)
            obv_recent = obv[-period:]
            x = np.arange(period)
            slope = np.polyfit(x, obv_recent, 1)[0]

            # Normalize slope by average volume (makes it comparable)
            avg_vol = np.mean(volume[-period:])
            normalized_slope = slope / avg_vol if avg_vol > 0 else 0

            result['obv_slope'] = round(float(normalized_slope), 4)
            result['obv_rising'] = normalized_slope > 0.05

            # Price trend over same period
            price_change = (close[-1] / close[-period] - 1) * 100

            # Detect accumulation: OBV rising, price flat or down
            if normalized_slope > 0.1 and price_change < 5:
                result['accumulation'] = True
                result['bonus'] = 8  # Strong signal

            # Detect distribution: OBV falling, price rising
            elif normalized_slope < -0.1 and price_change > 5:
                result['distribution'] = True
                result['bonus'] = -8  # Distribution warning — smart money exiting

            # Simple OBV confirmation
            elif normalized_slope > 0.1:
                result['bonus'] = 4  # OBV confirms uptrend

            return result

        except Exception as e:
            logger.error(f"Error calculating OBV trend: {e}")
            return result

    # ============================================================
    # MARKET REGIME DETECTION (v4.0 — Anti-Whipsaw)
    # ============================================================
    def detect_market_regime(self) -> Dict:
        """
        Detect broad market regime using SPY with 5-day confirmation.

        v4.0 improvements over v3.0:
        - 5-day confirmation window prevents whipsaw around MA lines
        - 1y data for real MA200 calculation (was 6mo → MA200 was fake)
        - VIX-based fear adjustment (>30 = forced BEAR)
        - Confidence level (CONFIRMED vs TENTATIVE) — drives API top_n caps
        - BEAR TENTATIVE: 3/5 days below MA200 (between CAUTION and BEAR)
        - CAUTION CONFIRMED: above MA200 but 2+ of last 5 below MA200
        - No score multiplier: regime is informational; top_n caps are applied elsewhere

        Returns:
            {
                'regime': str,           # 'BULL', 'CAUTION', 'BEAR', 'UNKNOWN'
                'confidence': str,       # 'CONFIRMED', 'TENTATIVE'
                'spy_above_ma50': bool,
                'spy_above_ma200': bool,
                'spy_5d_return': float,
                'spy_price': float,
                'ma50': float,
                'ma200': float,
                'vix': float,
            }
            When detection fails, regime is UNKNOWN and detect_error explains why.
        """
        from .regime_logic import regime_from_spy_close, regime_unknown

        import time as _time

        _yf_error: Optional[str] = None
        for _attempt in range(3):
            try:
                import yfinance as yf

                spy = yf.Ticker("SPY")
                hist = spy.history(period="1y")
                if hist is None or len(hist) < 50:
                    r = regime_unknown("insufficient_spy_history")
                    logger.warning("Market regime unavailable: %s", r.get("detect_error"))
                    return r

                close = hist["Close"]
                # Incomplete-bar guard: during US pre/intraday hours yfinance
                # appends today's row with a LIVE partial price (premarket SPY
                # quote), skewing regime values vs the completed-session view.
                # Drop it before 16:00 ET — same rule as DataFetcher.
                try:
                    from zoneinfo import ZoneInfo
                    from datetime import datetime as _dt

                    _now_et = _dt.now(ZoneInfo("America/New_York"))
                    if len(close) > 0:
                        _last_date = pd.Timestamp(close.index[-1]).date()
                        if _last_date == _now_et.date() and _now_et.hour < 16:
                            close = close.iloc[:-1]
                except Exception:
                    pass
                vix_val: float = 0.0
                try:
                    vix = yf.Ticker("^VIX")
                    vix_hist = vix.history(period="5d")
                    if vix_hist is not None and len(vix_hist) > 0:
                        vix_val = float(vix_hist["Close"].iloc[-1])
                except Exception:
                    pass

                result = regime_from_spy_close(close, vix_val)
                if result.get("detect_error"):
                    logger.warning("Market regime unavailable: %s", result.get("detect_error"))
                else:
                    logger.info(
                        f"Market Regime: {result['regime']} ({result['confidence']}) | "
                        f"SPY ${result['spy_price']:.2f} vs MA50 ${result['ma50']:.2f} / MA200 ${result['ma200']:.2f} | "
                        f"VIX: {result['vix']:.1f}"
                    )
                return result

            except Exception as e:
                err = str(e).lower()
                if ('rate' in err or 'too many' in err or '429' in err) and _attempt < 2:
                    wait = 10 * (2 ** _attempt)
                    logger.warning("Market regime rate limited, retrying in %ds (attempt %d/3)", wait, _attempt + 1)
                    _time.sleep(wait)
                    continue
                _yf_error = str(e)
                break

        # yfinance failed — try Tiingo for SPY history
        tiingo_key = (self.config.get('api_keys') or {}).get('tiingo', '')
        if tiingo_key:
            try:
                import requests as _req
                from datetime import date as _date, timedelta as _td
                end = _date.today().isoformat()
                start = (_date.today() - _td(days=400)).isoformat()
                resp = _req.get(
                    f'https://api.tiingo.com/tiingo/daily/SPY/prices',
                    params={'startDate': start, 'endDate': end, 'token': tiingo_key},
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data and len(data) >= 50:
                        closes = pd.Series(
                            [d.get('adjClose') or d.get('close') for d in data],
                            index=pd.to_datetime([d['date'] for d in data]),
                        )
                        result = regime_from_spy_close(closes, 0.0)
                        if not result.get('detect_error'):
                            logger.info(
                                f"Market Regime (Tiingo): {result['regime']} ({result['confidence']}) | "
                                f"SPY ${result['spy_price']:.2f} vs MA50 ${result['ma50']:.2f} / MA200 ${result['ma200']:.2f}"
                            )
                            return result
            except Exception as te:
                logger.warning(f"Tiingo SPY fallback failed: {te}")

        r = regime_unknown(_yf_error or "rate_limit_exhausted")
        logger.warning("Market regime unavailable: %s", r.get("detect_error"))
        return r

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

        # 5. OBV TREND (v3.0 — Smart Money)
        obv_data = self.calculate_obv_trend(df)
        boosters['obv_trend'] = obv_data
        boosters['obv_accumulation'] = obv_data.get('accumulation', False)
        boosters['obv_distribution'] = obv_data.get('distribution', False)
        boosters['obv_bonus'] = obv_data.get('bonus', 0)

        return boosters

