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
    if close is None:
        return regime_unknown("insufficient_spy_history")

    try:
        # NaN guard: during US premarket, data providers can append today's
        # row with NaN close. Without dropna, every derived value (price,
        # MA50, MA200) becomes NaN — the regime label may still come out
        # (comparisons on the remaining valid days) but the payload breaks
        # JSON serialization (/api/regime/current 500) and sector RS math.
        close = pd.to_numeric(close, errors="coerce").dropna()
        if len(close) < 50:
            return regime_unknown("insufficient_spy_history")
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
        elif bear_days >= 4:
            result["regime"] = "BEAR"
            result["confidence"] = "CONFIRMED"
        elif bear_days == 3:
            result["regime"] = "BEAR"
            result["confidence"] = "TENTATIVE"
        elif current > ma200_val:
            if bear_days >= 2:
                result["regime"] = "CAUTION"
                result["confidence"] = "CONFIRMED"
            else:
                result["regime"] = "CAUTION"
                result["confidence"] = "TENTATIVE"
        else:
            result["regime"] = "CAUTION"
            result["confidence"] = "TENTATIVE"

        return result

    except Exception as e:
        logger.warning("regime_from_spy_close failed: %s", e)
        return regime_unknown(str(e))


def relative_strength_vs_spy(stock_close: pd.Series, spy_close: pd.Series) -> Dict:
    """
    5 günlük göreli güç: hissenin getirisi eksi SPY'ın getirisi.

    2026-08-05: eskiden bir de kademeli `bonus` (+12/+8/+4/−5) üretiyordu ve
    adı `rs_bonus_vs_spy`'dı. O bonus skorun bonus bloğuna gidiyordu; blok
    sabite indirilince (tavan %100 bağlıyordu, GATE_AUDIT.md "3. tur") bonusun
    tek tüketicisi kalmadı. `sector_etf` / `ticker_5d` / `sector_5d` alanlarını
    da hiçbir kod okumuyordu. Kalan iki alan gerçekten kullanılıyor:
    `rs_score` (narrative + tarayıcı UI) ve `is_leader` (UI'da "Lider!" etiketi).
    """
    out: Dict = {"rs_score": 0.0, "is_leader": False}
    if stock_close is None or spy_close is None:
        return out
    try:
        sc = pd.to_numeric(stock_close, errors="coerce").dropna()
        sp = pd.to_numeric(spy_close, errors="coerce").dropna()
        if len(sc) < 6 or len(sp) < 6:
            return out
        s5 = (float(sc.iloc[-1]) / float(sc.iloc[-6]) - 1.0) * 100.0
        sp5 = (float(sp.iloc[-1]) / float(sp.iloc[-6]) - 1.0) * 100.0
        rs = s5 - sp5
        out["rs_score"] = round(rs, 2)
        out["is_leader"] = rs > 15
        return out
    except Exception as e:
        logger.debug("relative_strength_vs_spy: %s", e)
        return out
