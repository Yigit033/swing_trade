"""
Pattern Detection - VCP (Minervini) + Weinstein Stage Analysis

VCP: Identifies tight consolidation before a breakout (Minervini method).
Stage: Maps stocks to Weinstein's 4-stage cycle (only buy Stage 2).
"""

import logging
from typing import Dict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def detect_vcp(df: pd.DataFrame, min_contractions: int = 2) -> Dict:
    """
    Detect Volatility Contraction Pattern (Minervini).

    Logic:
    1. Prior uptrend: stock must be above where it was 40 bars ago (not a falling knife)
    2. Successive contractions: each 5-day swing range progressively tighter
    3. Volume declining into the base (smart money not distributing)
    4. Final contraction: price range < 12% of price (tight pivot)

    Works with standard 3-month data (≈63 trading days) — no long history needed.

    Returns:
        {
            'detected': bool,
            'contractions': int,          # number of confirmed contractions (2+ = VCP)
            'final_range_pct': float,     # last contraction width as % of price
            'volume_declining': bool,     # volume trend falling into base
            'prior_uptrend': bool,        # above 40-bar-ago price
            'bonus': int                  # +15 perfect VCP, +10 strong, +6 valid, +3 forming
        }
    """
    result = {
        'detected': False,
        'contractions': 0,
        'final_range_pct': 0.0,
        'volume_declining': False,
        'prior_uptrend': False,
        'bonus': 0,
    }

    if df is None or len(df) < 30:
        return result

    try:
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        n = len(df)

        # 1. Prior uptrend: price must be near or above where it was 40 bars ago
        look_back = min(40, n - 1)
        result['prior_uptrend'] = close[-1] > close[-look_back - 1] * 0.92  # 8% tolerance

        # 2. Build 5-day (weekly) swing ranges and volume averages
        week_ranges: list = []
        week_vols: list = []
        for start in range(max(0, n - 55), n, 5):
            end = min(start + 5, n)
            if end - start < 3:
                continue
            w_high = high[start:end].max()
            w_low = low[start:end].min()
            w_range_pct = (w_high - w_low) / w_low * 100 if w_low > 0 else 0
            week_ranges.append(w_range_pct)
            week_vols.append(float(np.mean(volume[start:end])))

        if len(week_ranges) < 4:
            return result

        # 3. Count contractions: each week's range must be ≥10% tighter than prior
        contractions = 0
        for i in range(1, len(week_ranges)):
            if week_ranges[i] < week_ranges[i - 1] * 0.90:
                contractions += 1
            elif week_ranges[i] > week_ranges[i - 1] * 1.25:
                contractions = max(0, contractions - 1)  # expansion resets momentum

        result['contractions'] = contractions

        # 4. Final contraction: average of last 2 weeks' ranges
        final_range_pct = sum(week_ranges[-2:]) / min(2, len(week_ranges))
        result['final_range_pct'] = round(final_range_pct, 1)

        # 5. Volume declining: last 3 weeks vs first 3 weeks (15% lower = meaningful)
        if len(week_vols) >= 6:
            early_vol = sum(week_vols[:3]) / 3
            late_vol = sum(week_vols[-3:]) / 3
            result['volume_declining'] = late_vol < early_vol * 0.85

        # 6. Pattern requires: uptrend + enough contractions + tight base
        tight_base = final_range_pct < 12.0
        result['detected'] = (
            result['prior_uptrend']
            and contractions >= min_contractions
            and tight_base
        )

        # 7. Bonus scoring
        if result['detected'] and result['volume_declining']:
            if final_range_pct < 5.0:
                result['bonus'] = 15   # Perfect VCP: very tight + volume dry-up
            elif final_range_pct < 8.0:
                result['bonus'] = 10   # Strong VCP
            else:
                result['bonus'] = 6    # Valid VCP
        elif contractions >= min_contractions and result['prior_uptrend']:
            result['bonus'] = 3        # Setup forming — volume not confirmed yet

        return result

    except Exception as e:
        logger.error(f"VCP detection error: {e}")
        return result


def detect_weinstein_stage(df: pd.DataFrame) -> Dict:
    """
    Weinstein Stage Analysis — approximated from 3-month price data.

    True Weinstein uses the 30-week (150 trading day) MA. With standard 3-month
    data (≈63 days) we use the 30-day SMA as the trend MA and 50-day SMA as
    the secondary anchor. This captures the same concept at a shorter horizon.

    Stage mapping:
        Stage 1 — Basing:      above MA30 but MA30 flat or just turning up
        Stage 2 — Markup:      above RISING MA30 in upper half of range  ← BUY ZONE
        Stage 3 — Distribution: above MA30 but MA30 starting to roll over
        Stage 4 — Decline:     below FALLING MA30                         ← AVOID

    Returns:
        {
            'stage': int,              # 1-4 (0 = not enough data)
            'stage_label': str,
            'ma30': float,
            'ma30_slope_pct': float,   # % change over last 5 bars (positive = rising)
            'above_ma30': bool,
            'ma30_rising': bool,
            'bonus': int               # +10 Stage 2, +3 Stage 1, -3 Stage 3, -10 Stage 4
        }
    """
    result = {
        'stage': 0,
        'stage_label': 'Unknown',
        'ma30': 0.0,
        'ma30_slope_pct': 0.0,
        'above_ma30': False,
        'ma30_rising': False,
        'bonus': 0,
    }

    if df is None or len(df) < 32:
        return result

    try:
        close = df['Close']
        current_price = float(close.iloc[-1])

        ma30_series = close.rolling(30).mean()
        ma30_now = float(ma30_series.iloc[-1])
        ma30_anchor = float(ma30_series.iloc[-6]) if len(ma30_series) >= 6 else float(ma30_series.iloc[0])

        result['ma30'] = round(ma30_now, 2)
        result['above_ma30'] = current_price > ma30_now

        slope_pct = (ma30_now / ma30_anchor - 1) * 100 if ma30_anchor > 0 else 0
        result['ma30_slope_pct'] = round(slope_pct, 3)
        result['ma30_rising'] = slope_pct > 0.05   # rising ≥ 0.05% per 5 bars

        # Price position in recent range (separates Stage 2 from Stage 1 basing)
        range_30d_high = float(close.tail(30).max())
        range_30d_low = float(close.tail(30).min())
        price_position = (
            (current_price - range_30d_low) / (range_30d_high - range_30d_low)
            if range_30d_high > range_30d_low else 0.5
        )

        above = result['above_ma30']
        rising = result['ma30_rising']

        if above and rising and price_position >= 0.50:
            result['stage'] = 2
            result['stage_label'] = 'Stage 2 — Markup (Buy Zone)'
            result['bonus'] = 10
        elif above and rising:
            result['stage'] = 1
            result['stage_label'] = 'Stage 1 — Basing (Turning Up)'
            result['bonus'] = 3
        elif above and not rising:
            result['stage'] = 3
            result['stage_label'] = 'Stage 3 — Distribution (Caution)'
            result['bonus'] = -3
        elif not above and not rising:
            result['stage'] = 4
            result['stage_label'] = 'Stage 4 — Decline (Avoid)'
            result['bonus'] = -10
        else:
            # Below MA30 but MA30 is turning up — late Stage 4 / early Stage 1
            result['stage'] = 1
            result['stage_label'] = 'Stage 1 — Recovery (Watch)'
            result['bonus'] = 0

        return result

    except Exception as e:
        logger.error(f"Weinstein stage detection error: {e}")
        return result
