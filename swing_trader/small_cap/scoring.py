"""
Small Cap Quality Scoring - Momentum-focused scoring system.
Completely independent from LargeCap scoring.

SENIOR TRADER OPTIMIZED v2.0
"""

import logging
from typing import Dict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SmallCapScoring:
    """
    Quality scoring for Small Cap Momentum Engine.
    
    SENIOR TRADER SCORING SYSTEM (Max 150 points):
    
    BASE SCORE (max 100):
    - Volume Explosion: 30 pts
    - Volatility Expansion (ATR%): 20 pts
    - Float Tightness: 20 pts (increased!)
    - Momentum Continuity: 15 pts
    - Risk Control: 15 pts
    
    CATALYST BONUS (max 35 pts):
    - Sector Leadership: +10
    - RSI Divergence: +8
    - Short Squeeze Setup: +10
    - News Catalyst: +8
    - Gap Continuation: +6
    - VWAP Above Close: +4
    - MACD Bullish: +4
    - Higher Lows: +5
    
    PENALTY SYSTEM (max -50):
    - Parabolic Move: -25
    - Earnings Week ±7: -20
    - Chasing (today +20%): -15
    - RSI > 90: -20
    - RSI > 85: -15
    - Single Day +30%: -20
    - 5-day > 60%: -15
    - Sector Underperform: -10
    - No Catalyst: -5
    
    FINAL RANGE: -50 to 150
    """
    
    # SENIOR TRADER SCORING WEIGHTS
    WEIGHT_VOLUME = 0.30          # 30 pts max
    WEIGHT_VOLATILITY = 0.20      # 20 pts max
    WEIGHT_FLOAT = 0.20           # 20 pts max (increased from 15%)
    WEIGHT_MOMENTUM = 0.15        # 15 pts max
    WEIGHT_RISK = 0.15            # 15 pts max
    
    def __init__(self, config: Dict = None):
        """Initialize SmallCapScoring."""
        self.config = config or {}
        logger.info("SmallCapScoring initialized (Senior Trader v2.0)")
    
    def score_volume_explosion(self, volume_surge: float, rvol: float = None) -> float:
        """
        Score volume explosion (0-30 points).
        OPTIMIZED tiered scoring based on market microstructure.
        """
        if rvol is None:
            rvol = volume_surge
        
        # Volume surge scoring - OPTIMIZED thresholds
        if volume_surge >= 5.0:
            surge_score = 20  # Parabolic, institutional
        elif volume_surge >= 4.0:
            surge_score = 17  # Very strong, viral
        elif volume_surge >= 3.0:
            surge_score = 14  # Strong, significant
        elif volume_surge >= 2.5:
            surge_score = 10  # Moderate, building
        elif volume_surge >= 2.0:
            surge_score = 7   # Early surge, watchlist
        elif volume_surge >= 1.5:
            surge_score = 5   # Minimum qualification
        else:
            surge_score = 0   # Below threshold
        
        # RVOL bonus (intraday relative volume)
        if rvol >= 5.0:
            rvol_bonus = 10   # Extreme intraday
        elif rvol >= 4.0:
            rvol_bonus = 8    # Very high
        elif rvol >= 3.0:
            rvol_bonus = 6    # High
        elif rvol >= 2.0:
            rvol_bonus = 3    # Moderate
        else:
            rvol_bonus = 0
        
        return min(surge_score + rvol_bonus, 30)
    
    def score_volatility_expansion(self, atr_percent: float) -> float:
        """
        Score volatility expansion (0-25 points).
        Higher ATR% = higher score for momentum plays.
        """
        # ATR% scoring - OPTIMIZED thresholds
        if atr_percent >= 0.15:    # 15%+ - extreme volatility
            return 25
        elif atr_percent >= 0.12:  # 12-15% - very high
            return 22
        elif atr_percent >= 0.10:  # 10-12% - high
            return 18
        elif atr_percent >= 0.08:  # 8-10% - above average
            return 14
        elif atr_percent >= 0.06:  # 6-8% - moderate
            return 10
        elif atr_percent >= 0.04:  # 4-6% - minimum for Type B
            return 7
        elif atr_percent >= 0.035: # 3.5-4% - minimum for Type A/C
            return 5
        else:                       # Below threshold
            return 0
    
    def score_float_tightness(self, float_shares: float) -> float:
        """
        Score float tightness (0-20 points) - SENIOR TRADER TIERING.
        
        Float Tiering (SENIOR TRADER):
        - ≤15M:  ATOMIC (+20 pts) - Parabolic potential
        - 15-30M: MICRO (+15 pts) - Explosive potential
        - 30-45M: SMALL (+10 pts) - Strong potential
        - 45-60M: TIGHT (+5 pts) - Good potential
        - 60-80M: Accept (+0 pts) - No bonus
        - >80M:  REJECT (filtered out)
        """
        if float_shares is None or float_shares <= 0:
            return 5  # Unknown - reduced score
        
        float_millions = float_shares / 1_000_000
        
        # SENIOR TRADER TIERING
        if float_millions <= 15:
            return 20  # ATOMIC - parabolic potential
        elif float_millions <= 30:
            return 15  # MICRO - explosive potential
        elif float_millions <= 45:
            return 10  # SMALL - strong potential
        elif float_millions <= 60:
            return 5   # TIGHT - good potential
        elif float_millions <= 80:
            return 0   # Accept but no bonus
        else:
            return -5  # Too large (should be filtered)
    
    def score_momentum_continuity(self, df: pd.DataFrame) -> float:
        """
        Score momentum continuity (0-15 points).
        Higher highs / higher closes = momentum persisting.
        """
        if df is None or len(df) < 3:
            return 5
        
        score = 0
        
        try:
            # Check last 3 days for higher highs
            highs = df['High'].tail(3).values
            if highs[2] > highs[1] > highs[0]:
                score += 6
            elif highs[2] > highs[1]:
                score += 3
            
            # Check last 3 days for higher closes
            closes = df['Close'].tail(3).values
            if closes[2] > closes[1] > closes[0]:
                score += 6
            elif closes[2] > closes[1]:
                score += 3
            
            # Check if close near high of day (bullish)
            today_close = df['Close'].iloc[-1]
            today_high = df['High'].iloc[-1]
            today_low = df['Low'].iloc[-1]
            
            day_range = today_high - today_low
            if day_range > 0:
                close_position = (today_close - today_low) / day_range
                if close_position >= 0.8:  # Close in top 20% of range
                    score += 3
            
            return min(score, 15)
            
        except Exception as e:
            logger.error(f"Error scoring momentum: {e}")
            return 5
    
    def score_risk_control(self, df: pd.DataFrame, atr_percent: float) -> float:
        """
        Score risk control efficiency (0-15 points).
        Better stop placement = higher score.
        """
        if df is None or len(df) < 1:
            return 5
        
        score = 0
        
        try:
            current_close = df['Close'].iloc[-1]
            
            # ATR-based stop distance
            atr_value = atr_percent * current_close
            stop_distance = 1.5 * atr_value  # 1.5 ATR stop
            stop_pct = stop_distance / current_close
            
            # Tighter stop = better R:R potential
            if stop_pct <= 0.05:  # <= 5% stop
                score += 10
            elif stop_pct <= 0.08:  # 5-8% stop
                score += 7
            elif stop_pct <= 0.10:  # 8-10% stop
                score += 5
            else:
                score += 3
            
            # Reward tight intraday range (easier to define stop)
            today_range = (df['High'].iloc[-1] - df['Low'].iloc[-1]) / current_close
            if today_range <= 0.05:
                score += 5
            elif today_range <= 0.08:
                score += 3
            
            return min(score, 15)
            
        except Exception as e:
            logger.error(f"Error scoring risk: {e}")
            return 5
    
    def calculate_quality_score(
        self, 
        df: pd.DataFrame,
        volume_surge: float,
        atr_percent: float,
        float_shares: float,
        boosters: Dict = None
    ) -> float:
        """
        Calculate composite quality score (0-100).
        
        Components:
        - Volume Explosion: 30%
        - Volatility Expansion: 25%
        - Float Tightness: 15%
        - Momentum Continuity: 15%
        - Risk Control: 15%
        
        + Swing Bonuses (NEW)
        - Penalties for overextension (NEW)
        """
        # Calculate component scores
        volume_score = self.score_volume_explosion(
            volume_surge, 
            boosters.get('rvol_value', volume_surge) if boosters else volume_surge
        )
        volatility_score = self.score_volatility_expansion(atr_percent)
        float_score = self.score_float_tightness(float_shares)
        momentum_score = self.score_momentum_continuity(df)
        risk_score = self.score_risk_control(df, atr_percent)
        
        # Calculate total
        total = volume_score + volatility_score + float_score + momentum_score + risk_score
        
        # ============================================================
        # BOOSTER BONUSES (max +35, expanded from +25)
        # ============================================================
        bonus = 0
        if boosters:
            if boosters.get('high_rvol'):
                bonus += 3
            if boosters.get('gap_continuation'):
                bonus += 4
            if boosters.get('higher_highs'):
                bonus += 3
            
            # SWING TRADE BONUSES
            if boosters.get('swing_ready'):
                bonus += 10  # Major bonus for swing-ready stocks
            if boosters.get('higher_lows'):
                bonus += 5   # Accumulation pattern
            if boosters.get('multi_day_volume'):
                bonus += 3   # Multi-day interest
            
            # SUSTAINED VOLUME PATTERN
            # Count days with 1.5x+ volume in last 5 days
            swing_details = boosters.get('swing_details', {})
            multi_day_vol = swing_details.get('multi_day_volume', {})
            surge_days = multi_day_vol.get('surge_days', 0)
            if surge_days >= 3:
                bonus += 5   # Sustained campaign (3+ days)
            elif surge_days >= 2:
                bonus += 3   # Building interest (2 days)
            
            # EARLY ENTRY BONUS (NEW) - Reward catching moves early!
            five_day_mom = swing_details.get('five_day_momentum', {})
            five_day_return = five_day_mom.get('return', 0)
            if 5 <= five_day_return <= 15:
                bonus += 8   # PERFECT early entry zone!
            elif 0 < five_day_return < 5:
                bonus += 5   # Very early (building phase)
            
            # ============================================================
            # SECTOR RS BONUS (NEW - Senior Trader v2.1)
            # ============================================================
            sector_rs_bonus = boosters.get('sector_rs_bonus', 0)
            bonus += sector_rs_bonus  # Max +12 for sector leader
            
            # ============================================================
            # CATALYST BONUSES (NEW - Senior Trader v2.1)
            # ============================================================
            # Short Interest
            short_interest_bonus = boosters.get('short_interest_bonus', 0)
            bonus += short_interest_bonus  # Max +10 for squeeze candidate
            
            # Insider Buying
            insider_bonus = boosters.get('insider_bonus', 0)
            bonus += insider_bonus  # Max +8 for >$1M insider buying
            
            # News Activity
            news_bonus = boosters.get('news_bonus', 0)
            bonus += news_bonus  # Max +5 for high news activity
            
            # RSI Divergence (already exists but emphasize)
            if boosters.get('rsi_divergence'):
                bonus += 8  # Game changer for early reversal
        
        # ============================================================
        # PENALTY SYSTEM (max -40, expanded from -35)
        # ============================================================
        penalty = 0
        if boosters:
            swing_details = boosters.get('swing_details', {})
            
            # RSI Penalties (Type-aware)
            rsi = boosters.get('rsi', 50)
            swing_type = boosters.get('swing_type', 'A')
            
            if swing_type == 'A':  # Continuation - stricter RSI
                if rsi > 70:
                    penalty += 10
                elif rsi > 65:
                    penalty += 5
            elif swing_type == 'B':  # Momentum - allows higher RSI
                if rsi > 85:
                    penalty += 15
                elif rsi > 80:
                    penalty += 10
                elif rsi > 75:
                    penalty += 5
            else:  # Type C - Early stage
                if rsi > 65:
                    penalty += 10
                elif rsi > 60:
                    penalty += 5
            
            # Overextension Penalties
            ext_info = swing_details.get('overextension', {})
            ext_details = ext_info.get('details', {})
            
            # Single day spike penalty (gap risk)
            max_single_day = ext_details.get('max_single_day', 0)
            if max_single_day > 25:
                penalty += 15  # Major gap risk
            elif max_single_day > 20:
                penalty += 8
            
            # Today chasing penalty
            today_change = ext_details.get('today_change', 0)
            if today_change > 15:
                penalty += 10  # Chasing a spike
            elif today_change > 10:
                penalty += 5
            
            # 5-day total overextension - TIGHTENED FOR EARLY ENTRY FOCUS
            five_day_total = ext_details.get('five_day_total', 0)
            if five_day_total > 40:
                penalty += 15  # Way too extended - likely chasing
            elif five_day_total > 30:
                penalty += 10  # Extended - late entry (TIGHTENED)
            elif five_day_total > 25:
                penalty += 5   # Getting extended (NEW)
            
            # PARABOLIC MOVE DETECTION (NEW)
            # Check if last 3 days show accelerating gains
            if df is not None and len(df) >= 4:
                try:
                    day1_ret = (df['Close'].iloc[-3] / df['Close'].iloc[-4] - 1) * 100
                    day2_ret = (df['Close'].iloc[-2] / df['Close'].iloc[-3] - 1) * 100
                    day3_ret = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
                    
                    # Parabolic = each day bigger than previous AND day3 >= 10%
                    if day1_ret < day2_ret < day3_ret and day3_ret >= 10:
                        penalty += 15  # Parabolic - reversal imminent
                except:
                    pass
            
            # NOT swing ready penalty
            if not boosters.get('swing_ready'):
                penalty += 5  # Missing key swing criteria
        
        # ============================================================
        # FINAL SCORE (Range: -40 to 135)
        # ============================================================
        final_score = total + bonus - penalty
        
        # Clamp between 0 and 135 (expanded range)
        final_score = max(0, min(final_score, 135))
        
        logger.debug(
            f"SmallCap Score: Vol={volume_score}, Volatility={volatility_score}, "
            f"Float={float_score}, Momentum={momentum_score}, Risk={risk_score}, "
            f"Bonus={bonus}, Penalty={penalty} -> Total={final_score}"
        )
        
        return final_score
    
    def is_swing_ready(self, boosters: Dict) -> bool:
        """Check if stock passes swing trade criteria."""
        if boosters is None:
            return False
        return boosters.get('swing_ready', False)

