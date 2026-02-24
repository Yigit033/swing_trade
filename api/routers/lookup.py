"""
Manual ticker lookup router — step-by-step diagnostic analysis.
POST /api/lookup  — analyze one or more tickers with FULL rejection reasons.
"""

import logging
from datetime import datetime
import yfinance as yf
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from api.deps import get_smallcap_engine
from api.utils import flatten_yf_df

router = APIRouter()
logger = logging.getLogger(__name__)


class LookupRequest(BaseModel):
    tickers: List[str]
    portfolio_value: float = 10000


def _analyze_ticker(engine, ticker: str, df, info: dict) -> dict:
    """
    Run all 3 analysis stages manually so we can return diagnostic info at every step.
    Mirrors exactly what scan_stock() does internally.
    """
    from datetime import datetime

    # ── Date ──────────────────────────────────────────────────────────────────
    try:
        if 'Date' in df.columns:
            signal_date_raw = df['Date'].iloc[-1]
        else:
            signal_date_raw = df.index[-1]
        if hasattr(signal_date_raw, 'to_pydatetime'):
            signal_date_dt = signal_date_raw.to_pydatetime().replace(tzinfo=None)
        else:
            signal_date_dt = datetime.strptime(str(signal_date_raw)[:10], '%Y-%m-%d')
    except Exception:
        signal_date_dt = datetime.now()

    # ── Stock meta ─────────────────────────────────────────────────────────────
    market_cap = (info.get('marketCap') or 0) / 1e9
    sector = info.get('sector', 'Unknown')
    company = info.get('longName') or info.get('shortName', ticker)
    float_shares = (info.get('floatShares') or 0) / 1e6
    current_price = float(df['Close'].iloc[-1]) if 'Close' in df.columns else 0
    try:
        rsi5d = ((float(df['Close'].iloc[-1]) / float(df['Close'].iloc[-6])) - 1) * 100
    except Exception:
        rsi5d = 0

    # ── Calculate RSI ─────────────────────────────────────────────────────────
    try:
        import numpy as np
        close = df['Close'].astype(float)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_val = float((100 - (100 / (1 + rs))).iloc[-1])
    except Exception:
        rsi_val = 0

    meta = {
        "ticker":       ticker,
        "company":      company,
        "sector":       sector,
        "market_cap_b": round(market_cap, 2),
        "float_m":      round(float_shares, 1),
        "current_price": round(current_price, 2),
        "rsi":          round(rsi_val, 1),
        "five_day_pct": round(rsi5d, 1),
    }

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1: Hard Filters
    # ─────────────────────────────────────────────────────────────────────────
    filter_passed, filter_results = engine.filters.apply_all_filters(
        ticker, df, info, signal_date_dt
    )
    filters_detail = filter_results.get('filters', {})

    if not filter_passed:
        # Find which filter failed and why
        failed_filter = None
        failed_reason = "Unknown filter failure"
        for fname, fdata in filters_detail.items():
            if not fdata.get('passed', True):
                failed_filter = fname
                failed_reason = fdata.get('reason', 'Failed')
                break
        return {
            **meta,
            "result": "REJECTED",
            "stage_failed": 1,
            "stage_name": "Filters (Market Cap, Volume, ATR%, Float, Earnings)",
            "failed_filter": failed_filter,
            "failed_reason": failed_reason,
            "filters": filters_detail,
            "triggers": None,
            "swing": None,
            "signal": None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2: Signal Triggers
    # ─────────────────────────────────────────────────────────────────────────
    triggered, trigger_details = engine.signals.check_all_triggers(df)
    triggers = trigger_details.get('triggers', trigger_details)

    if not triggered:
        # Find failed trigger
        failed_trigger = None
        failed_reason = "No momentum triggers activated"
        for tname, tdata in (triggers.items() if isinstance(triggers, dict) else {}.items()):
            if isinstance(tdata, dict) and not tdata.get('passed', True):
                failed_trigger = tname
                failed_reason = tdata.get('reason', 'Not triggered')
                break
        return {
            **meta,
            "result": "REJECTED",
            "stage_failed": 2,
            "stage_name": "Signal Triggers (Volume Surge, Volatility, Breakout)",
            "failed_filter": failed_trigger,
            "failed_reason": failed_reason,
            "filters": filters_detail,
            "triggers": triggers,
            "swing": None,
            "signal": None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3: Swing Confirmation
    # ─────────────────────────────────────────────────────────────────────────
    boosters = engine.signals.check_boosters(df)
    swing_ready = boosters.get('swing_ready', False)
    swing_details = boosters.get('swing_details', {})

    if not swing_ready:
        five_d = swing_details.get('five_day_momentum', {})
        ma20   = swing_details.get('above_ma20', {})
        reasons = []
        if not five_d.get('passed', True):
            reasons.append(f"5-day momentum {five_d.get('return', 0)*100:.1f}% (needs > 0%)")
        if not ma20.get('passed', True):
            reasons.append(f"Price below MA20 (distance: {ma20.get('distance', 0)*100:.1f}%)")
        return {
            **meta,
            "result": "REJECTED",
            "stage_failed": 3,
            "stage_name": "Swing Confirmation (5-Day Momentum, MA20, Higher Lows)",
            "failed_filter": "swing_confirmation",
            "failed_reason": "; ".join(reasons) or "Swing confirmation failed",
            "filters": filters_detail,
            "triggers": triggers,
            "swing": {
                "five_day": five_d,
                "above_ma20": ma20,
            },
            "signal": None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # All stages passed — build full signal
    # ─────────────────────────────────────────────────────────────────────────
    try:
        signal = engine.scan_stock(ticker, df, stock_info=info)
    except Exception as e:
        signal = None

    return {
        **meta,
        "result": "SIGNAL",
        "stage_failed": None,
        "stage_name": None,
        "failed_filter": None,
        "failed_reason": None,
        "filters": filters_detail,
        "triggers": triggers,
        "swing": {
            "five_day": swing_details.get('five_day_momentum', {}),
            "above_ma20": swing_details.get('above_ma20', {}),
        },
        "signal": signal,
    }


@router.post("")
def lookup_tickers(body: LookupRequest):
    """Analyze specific tickers with full diagnostic breakdown."""
    engine = get_smallcap_engine()
    results = []

    for ticker in body.tickers:
        ticker = ticker.upper().strip()
        try:
            raw = yf.download(ticker, period="60d", interval="1d",
                              auto_adjust=True, progress=False)
            if raw is None or raw.empty or len(raw) < 20:
                results.append({
                    "ticker": ticker, "result": "ERROR",
                    "failed_reason": "Yetersiz veri (20+ gün gerekli)",
                })
                continue

            df = flatten_yf_df(raw)

            info = {}
            try:
                info = yf.Ticker(ticker).info or {}
            except Exception:
                pass

            result = _analyze_ticker(engine, ticker, df, info)
            results.append(result)

        except Exception as e:
            logger.warning(f"Lookup failed for {ticker}: {e}")
            results.append({"ticker": ticker, "result": "ERROR", "failed_reason": str(e)})

    signals = [r for r in results if r.get("result") == "SIGNAL"]
    return {"results": results, "count": len(signals)}
