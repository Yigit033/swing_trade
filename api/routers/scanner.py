"""
SmallCap Scanner router.
GET  /api/scanner/chart     - OHLCV + indicators for any ticker
POST /api/scanner/smallcap  - run SmallCap momentum scan
POST /api/scanner/track     - add a signal to paper trades via tracker (duplicate-safe)
"""

import logging
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional
from api.deps import get_smallcap_engine, get_paper_tracker, get_fetcher, get_regime_storage
from api.utils import flatten_yf_df, sanitize_for_json, fetch_ticker_history

router = APIRouter()
logger = logging.getLogger(__name__)


class TrackSignalRequest(BaseModel):
    ticker: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: Optional[float] = None
    swing_type: str = "A"
    quality_score: float = 0
    position_size: int = 100
    hold_days_max: int = 7
    atr: Optional[float] = 0
    date: Optional[str] = None


@router.post("/track")
def track_signal(body: TrackSignalRequest):
    """
    Add a signal to paper trades using the tracker (duplicate-safe).
    Returns: {"status": "added", "trade_id": N} or {"status": "duplicate"}
    """
    tracker = get_paper_tracker()
    signal = {
        "ticker": body.ticker,
        "entry_price": body.entry_price,
        "stop_loss": body.stop_loss,
        "target_1": body.target_1,
        "target_2": body.target_2 or body.target_1,
        "swing_type": body.swing_type,
        "quality_score": body.quality_score,
        "position_size": body.position_size,
        "hold_days_max": body.hold_days_max,
        "atr": body.atr or 0,
        "date": body.date or datetime.date.today().isoformat(),
    }
    trade_id = tracker.add_trade_from_signal(signal)
    if trade_id > 0:
        return sanitize_for_json({"status": "added", "trade_id": trade_id})
    else:
        return sanitize_for_json({"status": "duplicate", "trade_id": -1})


@router.get("/chart")
def get_chart_data(ticker: str = Query(...), period: str = Query("3mo")):
    """Return OHLCV + RSI + MACD + EMAs for charting."""
    try:
        fetcher = get_fetcher()
        df = fetcher.fetch_stock_data(ticker.upper(), period=period)
        if df is None or len(df) < 5:
            return {"error": f"Veri alınamadı ({ticker}). Yahoo Finance geçici olarak rate limit uygulamış olabilir — birkaç dakika sonra tekrar deneyin.", "ticker": ticker}

        close = df["Close"].astype(float)
        vol   = df["Volume"].astype(float)

        # RSI (14)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["RSI"] = (100 - (100 / (1 + rs))).round(4)

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD"]        = (ema12 - ema26).round(4)
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean().round(4)
        df["MACD_hist"]   = (df["MACD"] - df["MACD_signal"]).round(4)

        # EMA 20 / 50
        df["EMA20"] = close.ewm(span=20, adjust=False).mean().round(4)
        df["EMA50"] = close.ewm(span=50, adjust=False).mean().round(4)

        # Volume MA 20
        df["Volume_MA"] = vol.rolling(20).mean().round(0)

        def safe(v):
            try:
                f = float(v)
                return None if f != f else round(f, 4)
            except Exception:
                return None

        rows = []
        for _, row in df.iterrows():
            date = str(row.get("Date", ""))[:10]
            rows.append({
                "date":        date,
                "open":        safe(row.get("Open")),
                "high":        safe(row.get("High")),
                "low":         safe(row.get("Low")),
                "close":       safe(row.get("Close")),
                "volume":      safe(row.get("Volume")),
                "rsi":         safe(row.get("RSI")),
                "macd":        safe(row.get("MACD")),
                "macd_signal": safe(row.get("MACD_signal")),
                "macd_hist":   safe(row.get("MACD_hist")),
                "ema20":       safe(row.get("EMA20")),
                "ema50":       safe(row.get("EMA50")),
                "volume_ma":   safe(row.get("Volume_MA")),
            })

        return sanitize_for_json({"ticker": ticker.upper(), "period": period, "data": rows})

    except Exception as e:
        logger.exception(f"Chart data error for {ticker}")
        return sanitize_for_json({"error": str(e), "ticker": ticker})


class ScanRequest(BaseModel):
    portfolio_value: float = 10000
    min_quality: int = 65
    top_n: int = 10


@router.post("/smallcap")
def run_smallcap_scan(body: ScanRequest):
    """Run the SmallCap Momentum Scanner."""
    engine = get_smallcap_engine()
    fetcher = get_fetcher()
    try:
        tickers = engine.get_small_cap_universe(use_finviz=True, max_tickers=200)
        logger.info(f"SmallCap universe: {len(tickers)} tickers")

        # Use fetcher.fetch_stock_data (same as Streamlit) instead of yf.download
        # This ensures identical data pipeline: yf.Ticker().history(), same validation,
        # same period (3mo), same column format → identical technical indicators → identical scores
        data_dict: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                df = fetcher.fetch_stock_data(ticker, period='3mo')
                if df is not None and len(df) >= 20:
                    data_dict[ticker] = df
            except Exception:
                continue

        if not data_dict:
            return {"signals": [], "stats": {"reason": "no_data"}, "market_regime": "UNKNOWN"}

        signals = engine.scan_universe(
            tickers=list(data_dict.keys()),
            data_dict=data_dict,
            portfolio_value=body.portfolio_value,
        )

        # Filter on ORIGINAL quality score (before regime penalty) so users see
        # fundamentally good signals even in BEAR/CAUTION markets.
        # The regime-adjusted score is still displayed for risk awareness.
        filtered = [s for s in signals if s.get("original_quality_score", s.get("quality_score", 0)) >= body.min_quality]
        filtered = filtered[:body.top_n]

        # v4.0: Get actual market regime from signals (engine sets it per-signal)
        actual_regime = "BULL"
        regime_multiplier = 1.0
        regime_confidence = "CONFIRMED"
        source = filtered if filtered else signals
        if source:
            actual_regime = source[0].get("market_regime", "BULL")
            regime_multiplier = source[0].get("regime_multiplier", 1.0)
            regime_confidence = source[0].get("regime_confidence", "CONFIRMED")

        # v4.0: Persist regime to DB (auto-log on every scan)
        try:
            regime_data = getattr(engine, '_last_regime', None)
            if regime_data:
                get_regime_storage().save_regime(regime_data)
        except Exception:
            logger.debug("Regime save skipped (non-critical)")

        stats = {
            "stocks_scanned":  len(tickers),
            "stocks_with_data": len(data_dict),
            "raw_signals":     len(signals),
            "filtered_signals": len(filtered),
            "reason": "success" if filtered else "no_qualifying",
            "regime_multiplier": regime_multiplier,
            "regime_confidence": regime_confidence,
        }
        return sanitize_for_json({"signals": filtered, "stats": stats, "market_regime": actual_regime})

    except Exception as e:
        logger.exception("SmallCap scan failed")
        return sanitize_for_json({"signals": [], "stats": {"reason": "error", "error": str(e)},
                "market_regime": "UNKNOWN", "error": str(e)})
