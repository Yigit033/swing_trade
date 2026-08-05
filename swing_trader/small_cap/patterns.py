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
        }

    Yalnız `stage` tüketiliyor: engine.py'deki Stage 3/4 hard gate. Ölçüm
    (measure_gate_value.py) bunu EN GÜÇLÜ kapı buldu — kaldırılınca EV
    +3.11% → +2.13%. Eskiden bir de `bonus` (+10/+3/−3/−10) dönüyordu; skorun
    bonus bloğu sabite indirilince (GATE_AUDIT.md "3. tur") okuyanı kalmadı.
    """
    result = {
        'stage': 0,
        'stage_label': 'Unknown',
        'ma30': 0.0,
        'ma30_slope_pct': 0.0,
        'above_ma30': False,
        'ma30_rising': False,
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
        elif above and rising:
            result['stage'] = 1
            result['stage_label'] = 'Stage 1 — Basing (Turning Up)'
        elif above and not rising:
            result['stage'] = 3
            result['stage_label'] = 'Stage 3 — Distribution (Caution)'
        elif not above and not rising:
            result['stage'] = 4
            result['stage_label'] = 'Stage 4 — Decline (Avoid)'
        else:
            # Below MA30 but MA30 is turning up — late Stage 4 / early Stage 1
            result['stage'] = 1
            result['stage_label'] = 'Stage 1 — Recovery (Watch)'

        return result

    except Exception as e:
        logger.error(f"Weinstein stage detection error: {e}")
        return result
