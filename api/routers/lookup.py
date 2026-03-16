"""
Manual ticker lookup router.
POST /api/lookup  — analyze tickers with full stage-by-stage diagnostic.

Logic mirrors Streamlit's analyze_smallcap_ticker() from dashboard/app.py
so the output is identical to the Streamlit version.
"""

import logging
from datetime import datetime
import yfinance as yf
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from api.deps import get_smallcap_engine, get_fetcher
from api.utils import flatten_yf_df, sanitize_for_json

router = APIRouter()
logger = logging.getLogger(__name__)


class LookupRequest(BaseModel):
    tickers: List[str]
    portfolio_value: float = 10000


def _analyze_smallcap_ticker(ticker: str, df, info: dict, engine, portfolio_value: float) -> dict:
    """
    Exact port of Streamlit's analyze_smallcap_ticker() from dashboard/app.py.
    Returns a result dict with the same keys so the frontend can display
    the same stage-by-stage rejection detail.
    """
    result: dict = {
        "ticker":       ticker,
        "strategy":     "SmallCap",
        "status":       "analyzed",
        "company_name": info.get("longName") or info.get("shortName", ticker),
        "sector":       info.get("sector", "Unknown"),
        "market_cap":   info.get("marketCap", 0) or 0,
        "float_shares": info.get("floatShares", 0) or 0,
    }

    stock_info = {
        "ticker":      ticker,
        "marketCap":   result["market_cap"],
        "floatShares": result["float_shares"],
        "shortName":   result["company_name"],
        "sector":      result["sector"],
    }

    # ── STEP 1: Hard Filters ─────────────────────────────────────────────────
    filter_passed, filter_results = engine.filters.apply_all_filters(
        ticker, df, stock_info, datetime.now()
    )
    result["filter_passed"]  = filter_passed
    result["filter_details"] = filter_results   # sanitized below

    if not filter_passed:
        result["swing_ready"]      = False
        result["rejection_reason"] = "Failed universe filters"
        result["rsi"]              = _safe_rsi(engine, df)
        result["five_day_return"]  = _safe_5d(df)
        return result

    # ── STEP 2: Signal Triggers ───────────────────────────────────────────────
    triggered, trigger_details = engine.signals.check_all_triggers(df)
    result["trigger_passed"]   = triggered
    result["trigger_details"]  = trigger_details   # sanitized below

    if not triggered:
        vol_surge = trigger_details.get("volume_surge", 0)
        atr_pct   = trigger_details.get("atr_percent", 0) * 100
        result["swing_ready"]      = False
        result["rejection_reason"] = (
            f"No signal trigger | VolSurge: {vol_surge:.1f}x | ATR: {atr_pct:.1f}%"
        )
        result["rsi"]             = _safe_rsi(engine, df)
        result["five_day_return"] = _safe_5d(df)
        return result

    # ── STEP 3: Swing Confirmation ────────────────────────────────────────────
    boosters     = engine.signals.check_boosters(df)
    swing_ready  = boosters.get("swing_ready", False)
    swing_details = boosters.get("swing_details", {})

    result["swing_ready"]   = swing_ready
    result["boosters"]      = boosters
    result["swing_details"] = swing_details

    if not swing_ready:
        result["rejection_reason"] = "Failed swing confirmation"
        result["rsi"]             = _safe_rsi(engine, df)
        result["five_day_return"] = _safe_5d(df)
        return result

    # ── STEP 4: Quality Score ─────────────────────────────────────────────────
    volume_surge = trigger_details.get("volume_surge", 2.0)
    atr_percent  = trigger_details.get("atr_percent", 0.06)
    float_sh     = filter_results.get("float_shares", 0)

    quality_score = engine.scoring.calculate_quality_score(
        df, volume_surge, atr_percent, float_sh, boosters
    )
    result["quality_score"] = quality_score

    # ── STEP 5: Swing Type ────────────────────────────────────────────────────
    five_day_return = swing_details.get("five_day_momentum", {}).get("return", 0)
    ma20_distance   = swing_details.get("above_ma20", {}).get("distance", 0)
    rsi             = boosters.get("rsi", 50)
    higher_lows     = boosters.get("higher_lows", False)

    today_high   = float(df["High"].iloc[-1])
    today_low    = float(df["Low"].iloc[-1])
    today_close  = float(df["Close"].iloc[-1])
    day_range    = today_high - today_low
    close_position = (today_close - today_low) / day_range if day_range > 0 else 0.5

    swing_type, hold_days, type_reason = engine._classify_swing_type(
        five_day_return, rsi, volume_surge, higher_lows,
        close_position=close_position, ma20_distance=ma20_distance,
    )

    result.update({
        "swing_type":      swing_type,
        "hold_days":       hold_days,
        "type_reason":     type_reason,
        "five_day_return": five_day_return,
        "rsi":             rsi,
        "volume_surge":    volume_surge,
        "atr_percent":     atr_percent,
        "entry_price":     today_close,
    })

    # ── STEP 6: Risk Management ───────────────────────────────────────────────
    signal = {
        "entry_price": today_close,
        "atr_percent": atr_percent / 100 if atr_percent > 1 else atr_percent,
        "date": datetime.now().strftime("%Y-%m-%d"),
    }
    signal = engine.risk.add_risk_management(signal, df, portfolio_value)
    result["stop_loss"]     = signal.get("stop_loss")
    result["target_1"]      = signal.get("target_1")
    result["target_2"]      = signal.get("target_2")
    result["position_size"] = signal.get("position_size")

    # ── v3.0: OBV Trend ─────────────────────────────────────────────────────
    obv_data = boosters.get("obv_trend", {})
    result["obv_accumulation"] = obv_data.get("accumulation", False)
    result["obv_distribution"] = obv_data.get("distribution", False)
    result["obv_bonus"]        = obv_data.get("bonus", 0)

    return result


def _safe_rsi(engine, df) -> float:
    try:
        return float(engine.signals.calculate_rsi(df))
    except Exception:
        return 50.0


def _safe_5d(df) -> float:
    try:
        if len(df) >= 6:
            return float((df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) * 100)
    except Exception:
        pass
    return 0.0


@router.post("")
def lookup_tickers(body: LookupRequest):
    """Analyze tickers with full stage-by-stage diagnostic breakdown.
    
    Uses the SAME DataFetcher.fetch_stock_data() as Streamlit to ensure
    identical data fetching behavior (session management, validation, etc.).
    """
    engine = get_smallcap_engine()
    fetcher = get_fetcher()
    results = []

    for ticker in body.tickers:
        ticker = ticker.upper().strip()
        try:
            # Use DataFetcher — SAME as Streamlit's fetcher.fetch_stock_data()
            df = fetcher.fetch_stock_data(ticker, period='3mo')

            if df is None or len(df) < 20:
                results.append({
                    "ticker": ticker, "status": "error",
                    "message": "Yetersiz veri (20+ gün gerekli)",
                })
                continue

            info = {}
            try:
                info = yf.Ticker(ticker).info or {}
            except Exception:
                pass

            result = _analyze_smallcap_ticker(ticker, df, info, engine, body.portfolio_value)
            # CRITICAL: convert all numpy types before JSON serialization
            results.append(sanitize_for_json(result))

        except Exception as e:
            logger.warning(f"Lookup failed for {ticker}: {e}", exc_info=True)
            results.append({"ticker": ticker, "status": "error", "message": str(e)})

    signals = [r for r in results if r.get("swing_ready")]
    return {"results": results, "count": len(signals)}
