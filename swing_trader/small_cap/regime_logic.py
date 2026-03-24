"""
Point-in-time market regime from SPY (and optional VIX) close series.
Shared by live detect_market_regime (yfinance) and walk-forward backtest.
"""

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def regime_unknown(reason: str) -> Dict:
    msg = (reason or "unknown")[:500]
    return {
        "regime": "UNKNOWN",
        "confidence": "TENTATIVE",
        "spy_above_ma50": False,
        "spy_above_ma200": False,
        "spy_5d_return": 0.0,
        "score_multiplier": 1.0,
        "spy_price": 0.0,
        "ma50": 0.0,
        "ma200": 0.0,
        "vix": 0.0,
        "detect_error": msg,
    }


def regime_from_spy_close(close: pd.Series, vix_last: Optional[float] = None) -> Dict:
    """
    Same rules as SmallCapSignals.detect_market_regime v4.0, using pre-aligned SPY closes.
    `close` must be chronological, index or RangeIndex; last row = as-of bar.
    """
    if close is None or len(close) < 50:
        return regime_unknown("insufficient_spy_history")

    try:
        close = close.astype(float)
        current = float(close.iloc[-1])
        ma50_val = float(close.rolling(50).mean().iloc[-1])
        has_ma200 = len(close) >= 200
        ma200_val = float(close.rolling(200).mean().iloc[-1]) if has_ma200 else ma50_val

        spy_5d = 0.0
        if len(close) >= 6:
            spy_5d = round(((current / float(close.iloc[-6])) - 1) * 100, 2)

        vix_val = float(vix_last) if vix_last is not None and vix_last == vix_last else 0.0

        result: Dict = {
            "spy_price": round(current, 2),
            "ma50": round(ma50_val, 2),
            "ma200": round(ma200_val, 2),
            "spy_above_ma50": current > ma50_val,
            "spy_above_ma200": current > ma200_val,
            "spy_5d_return": spy_5d,
            "vix": round(vix_val, 2),
        }

        if vix_val > 30:
            result["regime"] = "BEAR"
            result["confidence"] = "CONFIRMED"
            result["score_multiplier"] = 0.70
            return result

        ma50_series = close.rolling(50).mean()
        ma200_series = close.rolling(200).mean() if has_ma200 else ma50_series

        last_5_close = close.tail(5)
        last_5_ma50 = ma50_series.tail(5)
        last_5_ma200 = ma200_series.tail(5)

        bull_days = int(((last_5_close > last_5_ma50) & (last_5_close > last_5_ma200)).sum())
        bear_days = int((last_5_close < last_5_ma200).sum())

        if bull_days >= 4:
            result["regime"] = "BULL"
            result["confidence"] = "CONFIRMED"
            result["score_multiplier"] = 1.0
        elif bear_days >= 4:
            result["regime"] = "BEAR"
            result["confidence"] = "CONFIRMED"
            result["score_multiplier"] = 0.75
        elif bear_days == 3:
            result["regime"] = "BEAR"
            result["confidence"] = "TENTATIVE"
            result["score_multiplier"] = 0.72
        elif current > ma200_val:
            if bear_days >= 2:
                result["regime"] = "CAUTION"
                result["confidence"] = "CONFIRMED"
                result["score_multiplier"] = 0.80
            else:
                result["regime"] = "CAUTION"
                result["confidence"] = "TENTATIVE"
                result["score_multiplier"] = 0.85
        else:
            result["regime"] = "CAUTION"
            result["confidence"] = "TENTATIVE"
            result["score_multiplier"] = 0.80

        if vix_val > 25 and result["regime"] != "BEAR":
            result["score_multiplier"] = round(result["score_multiplier"] - 0.05, 2)

        return result

    except Exception as e:
        logger.warning("regime_from_spy_close failed: %s", e)
        return regime_unknown(str(e))


def rs_bonus_vs_spy(stock_close: pd.Series, spy_close: pd.Series) -> Dict:
    """
    Sector-style RS bonus using only aligned historical closes (backtest / point-in-time).
    Tier thresholds match SectorRS.calculate_sector_rs.
    """
    out: Dict = {
        "sector_etf": "SPY",
        "ticker_5d": 0.0,
        "sector_5d": 0.0,
        "rs_score": 0.0,
        "is_leader": False,
        "bonus": 0,
    }
    if stock_close is None or spy_close is None:
        return out
    try:
        sc = stock_close.astype(float)
        sp = spy_close.astype(float)
        if len(sc) < 6 or len(sp) < 6:
            return out
        s5 = (float(sc.iloc[-1]) / float(sc.iloc[-6]) - 1.0) * 100.0
        sp5 = (float(sp.iloc[-1]) / float(sp.iloc[-6]) - 1.0) * 100.0
        rs = s5 - sp5
        out["ticker_5d"] = round(s5, 2)
        out["sector_5d"] = round(sp5, 2)
        out["rs_score"] = round(rs, 2)
        if rs > 15:
            out["is_leader"] = True
            out["bonus"] = 12
        elif rs > 10:
            out["bonus"] = 8
        elif rs > 5:
            out["bonus"] = 4
        elif rs < -10:
            out["bonus"] = -5
        return out
    except Exception as e:
        logger.debug("rs_bonus_vs_spy: %s", e)
        return out
