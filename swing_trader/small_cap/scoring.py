"""
Small Cap Quality Scoring - Momentum-focused scoring system.
Completely independent from LargeCap scoring.

SENIOR TRADER OPTIMIZED v2.0
"""

import logging
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .settings_config import SmallCapSettings

logger = logging.getLogger(__name__)


class SmallCapScoring:
    """
    Quality scoring for Small Cap Momentum Engine.

    SENIOR TRADER SCORING SYSTEM v3.0 (Weighted):

    BASE SCORE (max 100 — weighted):
    - Volume Explosion:    30% weight (raw 0-30, normalized)
    - Volatility (ATR%):   20% weight (raw 0-25, normalized)
    - Float Tightness:     20% weight (raw 0-20, normalized)
    - Momentum Continuity: 15% weight (raw 0-15, normalized)
    - Risk Control:        15% weight (raw 0-15, normalized)

    CATALYST BONUS (max +40 pts, capped):
    PENALTY SYSTEM (max -40 pts)

    FINAL RANGE: 0 to 140
    """
    
    # SENIOR TRADER SCORING WEIGHTS (v3.0 — actually applied!)
    # Each component is normalized to 0-100, then multiplied by weight.
    # Weighted total = 0-100 base score.
    WEIGHT_VOLUME = 0.30          # 30% importance
    WEIGHT_VOLATILITY = 0.20      # 20% importance
    WEIGHT_FLOAT = 0.20           # 20% importance
    WEIGHT_MOMENTUM = 0.15        # 15% importance
    WEIGHT_RISK = 0.15            # 15% importance

    # Raw score maximums (for normalization to 0-100)
    MAX_VOLUME_SCORE = 30
    MAX_VOLATILITY_SCORE = 25
    MAX_FLOAT_SCORE = 20
    MAX_MOMENTUM_SCORE = 15
    MAX_RISK_SCORE = 15
    
    def __init__(self, config: Dict = None, settings: Optional["SmallCapSettings"] = None):
        """Initialize SmallCapScoring from SmallCapSettings.scoring_tuning."""
        from .settings_config import load_settings

        self.config = config or {}
        s = settings if settings is not None else load_settings()
        self._st = s.scoring_tuning
        # Mirror tuning onto class-style names used below
        st = self._st
        self.WEIGHT_VOLUME = st.weight_volume
        self.WEIGHT_VOLATILITY = st.weight_volatility
        self.WEIGHT_FLOAT = st.weight_float
        self.WEIGHT_MOMENTUM = st.weight_momentum
        self.WEIGHT_RISK = st.weight_risk
        self.MAX_VOLUME_SCORE = st.max_volume_score
        self.MAX_VOLATILITY_SCORE = st.max_volatility_score
        self.MAX_FLOAT_SCORE = st.max_float_score
        self.MAX_MOMENTUM_SCORE = st.max_momentum_score
        self.MAX_RISK_SCORE = st.max_risk_score
        self.BONUS_CAP = st.bonus_cap
        self.FINAL_SCORE_MAX = st.final_score_max
        self.RISK_SCORE_ATR_MULT = st.risk_score_atr_mult
        logger.info("SmallCapScoring initialized (Senior Trader v2.0)")
    
    def score_volume_explosion(self, volume_surge: float, rvol: float = None) -> float:
        """
        Score volume explosion (0-30 points).
        
        FIX v2.3: Single-metric scoring only.
        Previous bug: RVOL = volume_surge (same function), causing double-count.
        Now uses unified tiered scoring based on volume_surge alone.
        """
        st = self._st
        for t in sorted(st.volume_surge_tiers, key=lambda x: -x.min_surge):
            if volume_surge >= t.min_surge:
                return t.score
        return 0
    
    def score_volatility_expansion(self, atr_percent: float) -> float:
        """
        Score volatility expansion (0-25 points).
        Higher ATR% = higher score for momentum plays.
        """
        st = self._st
        for t in sorted(st.atr_percent_tiers, key=lambda x: -x.min_atr_frac):
            if atr_percent >= t.min_atr_frac:
                return t.score
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
        st = self._st
        if float_shares is None or float_shares <= 0:
            return st.float_score_unknown

        float_millions = float_shares / 1_000_000
        for b in sorted(st.float_millions_bands, key=lambda x: x.max_millions_le):
            if float_millions <= b.max_millions_le:
                return b.score
        return st.float_score_above_max_band
    
    def score_momentum_continuity(self, df: pd.DataFrame) -> float:
        """
        Score momentum continuity (0-15 points).
        Higher highs / higher closes = momentum persisting.
        """
        mp = self._st.momentum_points
        if df is None or len(df) < 3:
            return float(mp.insufficient_bars_score)

        score = 0

        try:
            highs = df["High"].tail(3).values
            if highs[2] > highs[1] > highs[0]:
                score += mp.higher_highs_full
            elif highs[2] > highs[1]:
                score += mp.higher_highs_partial

            closes = df["Close"].tail(3).values
            if closes[2] > closes[1] > closes[0]:
                score += mp.higher_closes_full
            elif closes[2] > closes[1]:
                score += mp.higher_closes_partial

            today_close = df["Close"].iloc[-1]
            today_high = df["High"].iloc[-1]
            today_low = df["Low"].iloc[-1]

            day_range = today_high - today_low
            if day_range > 0:
                close_position = (today_close - today_low) / day_range
                if close_position >= mp.close_in_top_of_range_min:
                    score += mp.close_near_high_pts

            return min(score, mp.raw_cap)

        except Exception as e:
            logger.error(f"Error scoring momentum: {e}")
            return float(mp.insufficient_bars_score)
    
    def score_risk_control(self, df: pd.DataFrame, atr_percent: float) -> float:
        """
        Score risk control efficiency (0-15 points).
        Better stop placement = higher score.
        """
        rb = self._st.risk_bands
        if df is None or len(df) < 1:
            return float(rb.insufficient_bars_score)

        score = 0

        try:
            current_close = df["Close"].iloc[-1]

            atr_value = atr_percent * current_close
            stop_distance = self.RISK_SCORE_ATR_MULT * atr_value
            stop_pct = stop_distance / current_close

            if stop_pct <= 0.05:
                score += rb.stop_le_05_pts
            elif stop_pct <= 0.08:
                score += rb.stop_le_08_pts
            elif stop_pct <= 0.10:
                score += rb.stop_le_10_pts
            else:
                score += rb.stop_else_pts

            today_range = (df["High"].iloc[-1] - df["Low"].iloc[-1]) / current_close
            if today_range <= 0.05:
                score += rb.range_le_05_pts
            elif today_range <= 0.08:
                score += rb.range_le_08_pts

            return min(score, rb.raw_cap)

        except Exception as e:
            logger.error(f"Error scoring risk: {e}")
            return float(rb.insufficient_bars_score)
    
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
        # Calculate raw component scores
        volume_score_raw = self.score_volume_explosion(
            volume_surge,
            boosters.get('rvol_value', volume_surge) if boosters else volume_surge
        )
        volatility_score_raw = self.score_volatility_expansion(atr_percent)
        float_score_raw = self.score_float_tightness(float_shares)
        momentum_score_raw = self.score_momentum_continuity(df)
        risk_score_raw = self.score_risk_control(df, atr_percent)

        # Normalize each to 0-100, then apply weights (v3.0)
        # V4: float_score allows negative values so large-float penalty actually bites
        volume_score = (max(volume_score_raw, 0) / self.MAX_VOLUME_SCORE) * 100 * self.WEIGHT_VOLUME
        volatility_score = (max(volatility_score_raw, 0) / self.MAX_VOLATILITY_SCORE) * 100 * self.WEIGHT_VOLATILITY
        float_score = (float_score_raw / self.MAX_FLOAT_SCORE) * 100 * self.WEIGHT_FLOAT
        momentum_score = (max(momentum_score_raw, 0) / self.MAX_MOMENTUM_SCORE) * 100 * self.WEIGHT_MOMENTUM
        risk_score = (max(risk_score_raw, 0) / self.MAX_RISK_SCORE) * 100 * self.WEIGHT_RISK

        # Weighted total (0-100 range)
        total = volume_score + volatility_score + float_score + momentum_score + risk_score
        
        st = self._st
        # ============================================================
        # BOOSTER BONUSES (max +35, expanded from +25)
        # ============================================================
        bonus = 0
        if boosters:
            if boosters.get('high_rvol'):
                bonus += st.bonus_high_rvol
            if boosters.get('gap_continuation'):
                bonus += st.bonus_gap_continuation
            if boosters.get('higher_highs'):
                bonus += st.bonus_higher_highs
            
            # SWING TRADE BONUSES
            if boosters.get('swing_ready'):
                bonus += st.bonus_swing_ready
            if boosters.get('higher_lows'):
                bonus += st.bonus_higher_lows
            if boosters.get('multi_day_volume'):
                bonus += st.bonus_multi_day_volume
            
            # SUSTAINED VOLUME PATTERN
            swing_details = boosters.get('swing_details', {})
            multi_day_vol = swing_details.get('multi_day_volume', {})
            surge_days = multi_day_vol.get('surge_days', 0)
            if surge_days >= 3:
                bonus += st.bonus_surge_days_3
            elif surge_days >= 2:
                bonus += st.bonus_surge_days_2
            
            five_day_mom = swing_details.get('five_day_momentum', {})
            five_day_return = five_day_mom.get('return', 0)
            if st.bonus_early_entry_lo <= five_day_return <= st.bonus_early_entry_hi:
                bonus += st.bonus_early_entry_pts
            elif 0 < five_day_return < st.bonus_very_early_hi:
                bonus += st.bonus_very_early_pts
            
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
            
            if boosters.get('rsi_divergence'):
                bonus += st.bonus_rsi_divergence

            # OBV Trend (v3.0 — Smart Money detection)
            obv_bonus = boosters.get('obv_bonus', 0)
            bonus += obv_bonus  # +8 accumulation, +4 confirm, -5 distribution
        
        # ============================================================
        # PENALTY SYSTEM (max -40, expanded from -35)
        # ============================================================
        penalty = 0
        if boosters:
            swing_details = boosters.get('swing_details', {})
            
            rsi = boosters.get('rsi', 50)
            swing_type = boosters.get('swing_type', 'A')
            
            if swing_type == 'A':
                if rsi > 70:
                    penalty += st.pen_a_rsi_gt_70
                elif rsi > 65:
                    penalty += st.pen_a_rsi_gt_65
            elif swing_type == 'B':
                if rsi > 85:
                    penalty += st.pen_b_rsi_gt_85
                elif rsi > 80:
                    penalty += st.pen_b_rsi_gt_80
                elif rsi > 75:
                    penalty += st.pen_b_rsi_gt_75
            else:
                if rsi > 65:
                    penalty += st.pen_c_rsi_gt_65
                elif rsi > 60:
                    penalty += st.pen_c_rsi_gt_60
            
            ext_info = swing_details.get('overextension', {})
            ext_details = ext_info.get('details', {})
            
            max_single_day = ext_details.get('max_single_day', 0)
            if max_single_day > 25:
                penalty += st.pen_ext_day_gt_25
            elif max_single_day > 20:
                penalty += st.pen_ext_day_gt_20
            
            today_change = ext_details.get('today_change', 0)
            if today_change > 15:
                penalty += st.pen_today_gt_15
            elif today_change > 10:
                penalty += st.pen_today_gt_10
            
            five_day_total = ext_details.get('five_day_total', 0)
            if five_day_total > 40:
                penalty += st.pen_5d_gt_40
            elif five_day_total > 30:
                penalty += st.pen_5d_gt_30
            elif five_day_total > 25:
                penalty += st.pen_5d_gt_25
            
            if df is not None and len(df) >= 4:
                try:
                    day1_ret = (df['Close'].iloc[-3] / df['Close'].iloc[-4] - 1) * 100
                    day2_ret = (df['Close'].iloc[-2] / df['Close'].iloc[-3] - 1) * 100
                    day3_ret = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
                    
                    if day1_ret < day2_ret < day3_ret and day3_ret >= st.parabolic_day3_min_pct:
                        penalty += st.pen_parabolic
                except Exception:
                    pass
            
            if not boosters.get('swing_ready'):
                penalty += st.pen_not_swing_ready
        
        bonus = min(bonus, st.bonus_cap)

        final_score = total + bonus - penalty

        final_score = max(0, min(final_score, st.final_score_max))
        
        logger.debug(
            f"SmallCap Score: Vol={volume_score:.1f}(raw {volume_score_raw}), "
            f"Volatility={volatility_score:.1f}(raw {volatility_score_raw}), "
            f"Float={float_score:.1f}(raw {float_score_raw}), "
            f"Momentum={momentum_score:.1f}(raw {momentum_score_raw}), "
            f"Risk={risk_score:.1f}(raw {risk_score_raw}), "
            f"Bonus={bonus}, Penalty={penalty} -> Total={final_score}"
        )
        
        return final_score
    
    def is_swing_ready(self, boosters: Dict) -> bool:
        """Check if stock passes swing trade criteria."""
        if boosters is None:
            return False
        return boosters.get('swing_ready', False)

