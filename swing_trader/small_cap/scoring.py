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

    SENIOR TRADER SCORING SYSTEM v5.0 (Directional Rebalance):

    BASE SCORE (max 100 — weighted):
    - Volume Explosion:    20% weight (raw 0-30, normalized) [was 30%]
    - Volatility (ATR%):   15% weight (raw 0-25, normalized) [was 20%]
    - Float Tightness:     15% weight (raw 0-20, normalized) [was 20%]
    - Momentum Continuity: 20% weight (raw 0-15, normalized) [was 15%]
    - Risk Control:        10% weight (raw 0-15, normalized) [was 15%]
    - Trend Quality:       20% weight (raw 0-25, normalized) [NEW]

    KEY CHANGE: Directional information now = 40% (Momentum + Trend)
    vs old 15%. Non-directional (Volume + ATR + Float) = 50% → 50%.

    CATALYST BONUS (max +40 pts, capped):
    PENALTY SYSTEM (max -60 pts, expanded)

    FINAL RANGE: 0 to 140
    """
    
    # SCORING WEIGHTS v5.0 — Directional Rebalance
    # Directional info (Momentum + Trend) = 40% of base score.
    # This ensures "which direction" matters more than "how much it moved."
    WEIGHT_VOLUME = 0.20          # 20% importance [was 30%]
    WEIGHT_VOLATILITY = 0.15      # 15% importance [was 20%]
    WEIGHT_FLOAT = 0.15           # 15% importance [was 20%]
    WEIGHT_MOMENTUM = 0.20        # 20% importance [was 15%]
    WEIGHT_RISK = 0.10            # 10% importance [was 15%]
    WEIGHT_TREND = 0.20           # 20% importance [NEW]

    # Raw score maximums (for normalization to 0-100)
    MAX_VOLUME_SCORE = 30
    MAX_VOLATILITY_SCORE = 25
    MAX_FLOAT_SCORE = 20
    MAX_MOMENTUM_SCORE = 15
    MAX_RISK_SCORE = 15
    MAX_TREND_SCORE = 25          # NEW
    
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
        self.WEIGHT_TREND = st.weight_trend
        self.MAX_VOLUME_SCORE = st.max_volume_score
        self.MAX_VOLATILITY_SCORE = st.max_volatility_score
        self.MAX_FLOAT_SCORE = st.max_float_score
        self.MAX_MOMENTUM_SCORE = st.max_momentum_score
        self.MAX_RISK_SCORE = st.max_risk_score
        self.MAX_TREND_SCORE = st.max_trend_score
        self.BONUS_CAP = st.bonus_cap
        self.FINAL_SCORE_MAX = st.final_score_max
        self.RISK_SCORE_ATR_MULT = st.risk_score_atr_mult
        logger.info("SmallCapScoring initialized (v5.0 — Directional Rebalance)")
    
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

        Uses a 5-bar window (not 3) to reduce noise and capture the actual
        price structure trend rather than reacting to a single outlier candle.
        Scores on majority-vote higher highs/closes across the window.
        """
        mp = self._st.momentum_points
        if df is None or len(df) < 5:
            return float(mp.insufficient_bars_score)

        score = 0

        try:
            # 5-bar window: majority-vote higher highs (3 of 4 transitions)
            highs = df["High"].tail(5).values
            hh_transitions = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])
            if hh_transitions >= 4:          # 4/4 — strong trend
                score += mp.higher_highs_full
            elif hh_transitions >= 3:        # 3/4 — majority
                score += mp.higher_highs_partial

            closes = df["Close"].tail(5).values
            hc_transitions = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
            if hc_transitions >= 4:
                score += mp.higher_closes_full
            elif hc_transitions >= 3:
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
    
    def score_trend_quality(self, df: pd.DataFrame, boosters: Dict = None) -> float:
        """
        Score trend quality — directional health (0-25 points). NEW in v5.0.

        This is the most important new component: it measures whether the
        stock is in a healthy, constructive uptrend vs a distribution or
        markdown phase.

        Scoring:
        - MA20 rising slope: +6
        - Close above MA50: +5
        - Golden cross (MA20 > MA50): +4
        - Trend phase = markup: +5
        - Higher lows pattern (>60% of lookback): +3
        - No rejection candle: +2
        """
        score = 0
        trend_data = {}

        if boosters:
            swing_details = boosters.get('swing_details', {})
            trend_data = swing_details.get('trend_quality', {})

        if not trend_data:
            # Fallback: calculate inline if not available from boosters
            from .trend_quality import calculate_trend_quality
            if df is not None and len(df) >= 21:
                trend_data = calculate_trend_quality(df)

        if not trend_data:
            return 5  # Neutral fallback

        # MA20 slope rising
        if trend_data.get('ma20_slope_ok', False):
            slope_val = abs(trend_data.get('ma20_slope_value', 0))
            if slope_val > 1.0:
                score += 6  # Strong upslope
            elif slope_val > 0.3:
                score += 4  # Moderate upslope
            else:
                score += 2  # Weak but positive

        # Close above MA50
        ma50_dist = trend_data.get('ma50_distance_pct', 0)
        if ma50_dist > 5:
            score += 5  # Comfortably above
        elif ma50_dist > 0:
            score += 3  # Just above
        elif ma50_dist > -5:
            score += 1  # Slightly below but within range

        # Golden cross
        if trend_data.get('golden_cross', False):
            score += 4

        # Trend phase
        phase = trend_data.get('trend_phase', 'unknown')
        if phase == 'markup':
            score += 5
        elif phase == 'late_markup':
            score += 2
        # distribution/markdown get 0

        # Higher lows pattern
        hl_count = trend_data.get('higher_lows_count', 0)
        if hl_count >= 6:
            score += 3
        elif hl_count >= 4:
            score += 2
        elif hl_count >= 2:
            score += 1

        # No rejection candle
        if not trend_data.get('rejection_candle', False):
            score += 2

        return min(score, 25)

    def calculate_quality_score(
        self, 
        df: pd.DataFrame,
        volume_surge: float,
        atr_percent: float,
        float_shares: float,
        boosters: Dict = None
    ) -> float:
        """
        Calculate composite quality score (0-140).
        
        v5.0 Components (Directional Rebalance):
        - Volume Explosion:    20% [was 30%]
        - Volatility (ATR%):   15% [was 20%]
        - Float Tightness:     15% [was 20%]
        - Momentum Continuity: 20% [was 15%]
        - Risk Control:        10% [was 15%]
        - Trend Quality:       20% [NEW — directional health]
        
        + Bonuses (catalyst, sector RS, OBV, etc.)
        - Penalties (RSI overbought, overextension, trend weakness)
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
        trend_score_raw = self.score_trend_quality(df, boosters)

        # Normalize each to 0-100, then apply weights (v5.0)
        volume_score = (max(volume_score_raw, 0) / self.MAX_VOLUME_SCORE) * 100 * self.WEIGHT_VOLUME
        volatility_score = (max(volatility_score_raw, 0) / self.MAX_VOLATILITY_SCORE) * 100 * self.WEIGHT_VOLATILITY
        float_score = (float_score_raw / self.MAX_FLOAT_SCORE) * 100 * self.WEIGHT_FLOAT
        momentum_score = (max(momentum_score_raw, 0) / self.MAX_MOMENTUM_SCORE) * 100 * self.WEIGHT_MOMENTUM
        risk_score = (max(risk_score_raw, 0) / self.MAX_RISK_SCORE) * 100 * self.WEIGHT_RISK
        trend_score = (max(trend_score_raw, 0) / self.MAX_TREND_SCORE) * 100 * self.WEIGHT_TREND

        # Weighted total (0-100 range)
        total = volume_score + volatility_score + float_score + momentum_score + risk_score + trend_score
        
        st = self._st
        # ============================================================
        # BONUS = SABİT +30 (ölçüldü — 2026-08-04)
        # ============================================================
        # Burada 14 koşullu bonus (high_rvol, gap_continuation, higher_highs,
        # swing_ready, higher_lows, multi_day_volume, surge_days, early_entry,
        # rsi_divergence, golden_cross, confirmed_breakout, volume_on_up_day)
        # + sector_rs_bonus + obv_bonus toplanıp `min(bonus, bonus_cap)` ile
        # 30'a kırpılıyordu.
        #
        # ÖLÇÜM (scripts/measure_score_modifiers.py, 95 sinyal / 21 ay):
        #   • Bonus tavanına dayanan sinyal oranı: %100
        #   • Tavan üstü aşılan miktar: ortalama +29.7 (maks +48)
        #     → ham toplam ~60, tavan 30
        #   • Bırak-birini-çıkar: 14 bonusun HİÇBİRİNİN tek başına etkisi YOK
        #     (hepsinde ΔEV tam 0.00) — çünkü biri gitse toplam yine ≥30.
        #
        # Yani 14 koşulun net çıktısı HER SİNYAL İÇİN AYNI SAYI: +30.
        # Ayırt etme gücü sıfır; sadece skoru sabit bir miktar kaydırıyordu.
        # 40 satır kod, 14 ayarlanabilir parametre ve 14 UI alanı, tek bir
        # sabitin işini yapıyordu. Sabitle değiştirildi: davranış BİREBİR aynı
        # (dolayısıyla Q78/80/82 eşikleri ve tüm ölçümler geçerli kalır).
        #
        # NOT: Alternatif, tavanı kaldırıp bonusların gerçekten ayırt etmesini
        # sağlamaktı — ama bu ÖLÇÜLMEMİŞ bir davranış değişikliği olurdu (ham
        # toplam 40-78 bandına çıkıyor, tüm eşiklerin yeniden kalibrasyonu
        # gerekir). Ölçmeden yapılmaz; sonraki denetim turunun konusu.
        bonus = st.bonus_cap

        # ============================================================
        # CEZALAR — yalnız ölçülmüş olanlar (2026-08-05)
        # ============================================================
        # 21 cezanın hepsi bırak-birini-çıkar ile ölçüldü. Sadece 5'ini
        # çıkarmak EV'yi DÜŞÜRÜYOR (yani işe yarıyorlar): aşağıdaki RSI
        # merdivenleri + today_gt_10. Diğer 16'sı silindi — gerekçeler
        # GATE_AUDIT.md'de tek tek yazılı. Kısaca iki grup:
        #   (a) Hiç ateşlemeyenler (10): girdileri VCE/RVOL tetiğinin zaten
        #       garantilediği şeyleri tekrar kontrol ediyordu (MA50 üstü,
        #       swing_ready, trend fazı, OBV, parabolik, today>15 ...).
        #   (b) Ateşleyip ΔEV 0.00 verenler (6): cezaladıkları sinyaller
        #       ORTALAMANIN ÜSTÜNDE kazandı (5d>25 → +13.24%, 5d>40 → +20.36%,
        #       tek-gün>25 → +21.12%, MA20 düşüyor → +11.24%) — yani ceza
        #       yönü TERSTİ; sadece Q80 seçimini değiştirecek kadar büyük
        #       olmadığı için zarar görünmüyordu.
        penalty = 0
        if boosters:
            rsi = boosters.get('rsi', 50)
            swing_type = boosters.get('swing_type', 'A')

            if swing_type == 'A':
                if rsi > 70:
                    penalty += st.pen_a_rsi_gt_70      # 28 ateşleme, ΔEV -0.07
                elif rsi > 65:
                    penalty += st.pen_a_rsi_gt_65      # 4 ateşleme, ΔEV -0.09
            elif swing_type != 'B':
                # Type C (ve bilinmeyen tip) — en muhafazakâr eşik
                if rsi > 65:
                    penalty += st.pen_c_rsi_gt_65      # 17 ateşleme, ΔEV -0.16
                elif rsi > 60:
                    penalty += st.pen_c_rsi_gt_60      # 12 ateşleme, ΔEV -0.20
            # Type B (momentum/parabolik): RSI cezası YOK. 75/80/85 merdiveni
            # ölçüldü — 75 ve 80 hiç ateşlemedi, 85 bir kez ateşledi ve o sinyal
            # +8.75% kazandı. Type B'nin tanımı zaten "yüksek RSI ile koşuyor".

            # Tek günde >%10 sıçrayıp giriş: 3 ateşleme, ΔEV -0.10 (işe yarıyor)
            today_change = (boosters.get('swing_details', {})
                            .get('overextension', {})
                            .get('details', {})
                            .get('today_change', 0))
            if today_change > 10:
                penalty += st.pen_today_gt_10

        final_score = total + bonus - penalty

        final_score = max(0, min(final_score, st.final_score_max))
        
        logger.debug(
            f"SmallCap Score v5.0: Vol={volume_score:.1f}(raw {volume_score_raw}), "
            f"ATR={volatility_score:.1f}(raw {volatility_score_raw}), "
            f"Float={float_score:.1f}(raw {float_score_raw}), "
            f"Mom={momentum_score:.1f}(raw {momentum_score_raw}), "
            f"Risk={risk_score:.1f}(raw {risk_score_raw}), "
            f"Trend={trend_score:.1f}(raw {trend_score_raw}), "
            f"Bonus={bonus}, Penalty={penalty} -> Total={final_score}"
        )
        
        return final_score
    
    def is_swing_ready(self, boosters: Dict) -> bool:
        """Check if stock passes swing trade criteria."""
        if boosters is None:
            return False
        return boosters.get('swing_ready', False)

