"""
Trend Quality Analysis Module for Small-Cap Swing System.

Provides directional trend health scoring to distinguish 
"stocks that are moving" from "stocks that will continue moving UP."

Components:
1. MA20 Slope — Is the short-term trend accelerating?
2. MA50 Relationship — Is the stock in a long-term uptrend?
3. Rejection Candle (Bull Trap) — Is today's candle a distribution signal?
4. Higher Highs / Higher Lows — Is the price structure constructive?
5. Golden/Death Cross — MA20 vs MA50 relationship

Returns a TrendQuality dict with pass/fail gates + trend_strength score (0-100).
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_trend_quality(
    df: pd.DataFrame,
    *,
    ma20_slope_lookback: int = 5,
    ma50_max_below_pct: float = 8.0,
    rejection_close_position_max: float = 0.30,
    rejection_gap_min_pct: float = 2.0,
    higher_lows_lookback: int = 10,
) -> Dict:
    """
    Master trend quality analysis.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data, expects columns: Open, High, Low, Close, Volume. ≥50 bars ideal.
    ma20_slope_lookback : int
        Number of bars to measure MA20 slope direction.
    ma50_max_below_pct : float
        Max % below MA50 before rejecting (e.g. 8.0 means reject if close < MA50 * 0.92).
    rejection_close_position_max : float
        Close position threshold for rejection candle (0.30 = bottom 30% of daily range).
    rejection_gap_min_pct : float
        Minimum gap-up % for rejection candle detection.
    higher_lows_lookback : int
        Window for higher lows / higher highs pattern.

    Returns
    -------
    dict with keys:
        - trend_strength (int 0-100)
        - ma20_slope_ok (bool) : True if MA20 rising
        - ma50_ok (bool) : True if not too far below MA50
        - rejection_candle (bool) : True if a bull trap candle detected
        - golden_cross (bool) : True if MA20 > MA50
        - higher_lows_count (int) : count of higher lows in lookback
        - higher_highs_count (int)
        - details (dict) : raw values for debugging
    """
    result = {
        "trend_strength": 0,
        "ma20_slope_ok": False,
        "ma20_slope_value": 0.0,
        "ma50_ok": True,       # Default True — benefit of doubt if not enough data
        "ma50_distance_pct": 0.0,
        "rejection_candle": False,
        "rejection_details": {},
        "golden_cross": False,
        "higher_lows_count": 0,
        "higher_highs_count": 0,
        "trend_phase": "unknown",   # accumulation, markup, distribution, markdown
        "details": {},
    }

    if df is None or len(df) < 21:
        return result

    try:
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        open_ = df["Open"].astype(float)
        volume = df["Volume"].astype(float)
        n = len(close)

        # ================================================================
        # 1. MA20 SLOPE — Is the short-term moving average rising?
        # ================================================================
        ma20 = close.rolling(20).mean()
        if len(ma20.dropna()) >= ma20_slope_lookback:
            ma20_now = float(ma20.iloc[-1])
            ma20_prev = float(ma20.iloc[-ma20_slope_lookback])
            if ma20_prev > 0:
                slope_pct = (ma20_now / ma20_prev - 1) * 100
                result["ma20_slope_ok"] = slope_pct > 0
                result["ma20_slope_value"] = round(slope_pct, 3)
                result["details"]["ma20_now"] = round(ma20_now, 2)
                result["details"]["ma20_prev"] = round(ma20_prev, 2)

        # ================================================================
        # 2. MA50 RELATIONSHIP — Are we in a long-term uptrend?
        # ================================================================
        if n >= 50:
            ma50 = close.rolling(50).mean()
            ma50_val = float(ma50.iloc[-1])
            close_val = float(close.iloc[-1])

            if ma50_val > 0:
                dist_pct = (close_val / ma50_val - 1) * 100
                result["ma50_distance_pct"] = round(dist_pct, 2)
                # Reject if too far below MA50
                result["ma50_ok"] = dist_pct >= -ma50_max_below_pct
                result["details"]["ma50_val"] = round(ma50_val, 2)

                # Golden Cross check
                ma20_val = float(ma20.iloc[-1]) if len(ma20.dropna()) > 0 else 0
                if ma20_val > 0 and ma50_val > 0:
                    result["golden_cross"] = ma20_val > ma50_val

        # ================================================================
        # 3. REJECTION CANDLE (Bull Trap Detection)
        # ================================================================
        if n >= 2:
            today_open = float(open_.iloc[-1])
            today_close = float(close.iloc[-1])
            today_high = float(high.iloc[-1])
            today_low = float(low.iloc[-1])
            prev_close = float(close.iloc[-2])

            day_range = today_high - today_low
            if day_range > 0:
                close_position = (today_close - today_low) / day_range
            else:
                close_position = 0.5

            # Gap-up percentage
            gap_pct = ((today_open / prev_close) - 1) * 100 if prev_close > 0 else 0

            # Upper wick ratio (how much selling pressure)
            upper_wick = today_high - max(today_open, today_close)
            body = abs(today_close - today_open)
            upper_wick_ratio = upper_wick / day_range if day_range > 0 else 0

            # Bull Trap = gap up + close in bottom of range + large upper wick
            is_rejection = (
                gap_pct >= rejection_gap_min_pct
                and close_position < rejection_close_position_max
                and today_close < today_open  # red candle
            )

            # Also catch non-gap rejection: huge upper wick + red close in bottom
            is_distribution_candle = (
                upper_wick_ratio >= 0.60
                and close_position < 0.25
                and today_close < today_open
            )

            result["rejection_candle"] = is_rejection or is_distribution_candle
            result["rejection_details"] = {
                "gap_pct": round(gap_pct, 2),
                "close_position": round(close_position, 2),
                "upper_wick_ratio": round(upper_wick_ratio, 2),
                "is_gap_rejection": is_rejection,
                "is_distribution_candle": is_distribution_candle,
                "red_candle": today_close < today_open,
            }

        # ================================================================
        # 4. HIGHER HIGHS / HIGHER LOWS (Price Structure)
        # ================================================================
        lookback = min(higher_lows_lookback, n)
        if lookback >= 4:
            recent_lows = low.iloc[-lookback:].values
            recent_highs = high.iloc[-lookback:].values

            hl_count = sum(
                1 for i in range(1, len(recent_lows))
                if recent_lows[i] > recent_lows[i - 1]
            )
            hh_count = sum(
                1 for i in range(1, len(recent_highs))
                if recent_highs[i] > recent_highs[i - 1]
            )

            result["higher_lows_count"] = hl_count
            result["higher_highs_count"] = hh_count

        # ================================================================
        # 5. TREND PHASE DETECTION (Wyckoff-inspired)
        # ================================================================
        if n >= 20:
            # Volume trend (last 10 days vs prior 10 days)
            vol_recent = float(volume.iloc[-10:].mean()) if n >= 20 else 0
            vol_prior = float(volume.iloc[-20:-10].mean()) if n >= 20 else 0
            vol_expanding = vol_recent > vol_prior * 1.1 if vol_prior > 0 else False

            price_rising = float(close.iloc[-1]) > float(close.iloc[-10]) if n >= 10 else False

            if price_rising and vol_expanding:
                result["trend_phase"] = "markup"       # Best phase to enter
            elif price_rising and not vol_expanding:
                result["trend_phase"] = "late_markup"   # Caution
            elif not price_rising and vol_expanding:
                result["trend_phase"] = "distribution"  # Avoid
            else:
                result["trend_phase"] = "markdown"      # Avoid

            result["details"]["vol_expanding"] = vol_expanding
            result["details"]["price_rising"] = price_rising

        # ================================================================
        # 6. COMPOSITE TREND STRENGTH SCORE (0-100)
        # ================================================================
        score = 0

        # MA20 slope: +20 if rising
        if result["ma20_slope_ok"]:
            score += 20
            # Extra points for strong slope
            if result["ma20_slope_value"] > 1.0:
                score += 5
            elif result["ma20_slope_value"] > 0.5:
                score += 3

        # MA50 relationship: +20 if above
        if result.get("ma50_distance_pct", 0) > 0:
            score += 20
        elif result["ma50_ok"]:
            score += 10  # Below but within tolerance

        # Golden Cross: +10
        if result["golden_cross"]:
            score += 10

        # No rejection candle: +15
        if not result["rejection_candle"]:
            score += 15

        # Higher lows pattern: up to +15
        hl_pct = result["higher_lows_count"] / max(higher_lows_lookback - 1, 1)
        score += min(15, int(hl_pct * 15))

        # Higher highs: up to +10
        hh_pct = result["higher_highs_count"] / max(higher_lows_lookback - 1, 1)
        score += min(10, int(hh_pct * 10))

        # Trend phase bonus
        phase = result["trend_phase"]
        if phase == "markup":
            score += 10
        elif phase == "late_markup":
            score += 5
        elif phase == "distribution":
            score -= 10
        elif phase == "markdown":
            score -= 15

        result["trend_strength"] = max(0, min(100, score))

        return result

    except Exception as e:
        logger.warning("calculate_trend_quality failed: %s", e)
        return result
