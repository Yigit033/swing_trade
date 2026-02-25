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
from api.deps import get_smallcap_engine, get_paper_tracker
from api.utils import flatten_yf_df

router = APIRouter()
logger = logging.getLogger(__name__)


class TrackSignalRequest(BaseModel):
    ticker: str
    entry_price: float
    stop_loss: float
    target_1: float
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
        "swing_type": body.swing_type,
        "quality_score": body.quality_score,
        "position_size": body.position_size,
        "hold_days_max": body.hold_days_max,
        "atr": body.atr or 0,
        "date": body.date or datetime.date.today().isoformat(),
    }
    trade_id = tracker.add_trade_from_signal(signal)
    if trade_id > 0:
        return {"status": "added", "trade_id": trade_id}
    else:
        return {"status": "duplicate", "trade_id": -1}


@router.get("/chart")
def get_chart_data(ticker: str = Query(...), period: str = Query("3mo")):
    """Return OHLCV + RSI + MACD + EMAs for charting."""
    try:
        raw = yf.download(ticker.upper(), period=period, interval="1d",
                          auto_adjust=True, progress=False)
        if raw is None or raw.empty or len(raw) < 5:
            return {"error": "No data available for this ticker", "ticker": ticker}

        df = flatten_yf_df(raw)

        close = df["Close"].squeeze().astype(float)
        vol   = df["Volume"].squeeze().astype(float)

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

        return {"ticker": ticker.upper(), "period": period, "data": rows}

    except Exception as e:
        logger.exception(f"Chart data error for {ticker}")
        return {"error": str(e), "ticker": ticker}


class ScanRequest(BaseModel):
    portfolio_value: float = 10000
    min_quality: int = 65
    top_n: int = 10


@router.post("/smallcap")
def run_smallcap_scan(body: ScanRequest):
    """Run the SmallCap Momentum Scanner."""
    engine = get_smallcap_engine()
    try:
        tickers = engine.get_small_cap_universe(use_finviz=True, max_tickers=200)
        logger.info(f"SmallCap universe: {len(tickers)} tickers")

        data_dict: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                raw = yf.download(ticker, period="60d", interval="1d",
                                  auto_adjust=True, progress=False)
                if raw is not None and not raw.empty and len(raw) >= 20:
                    data_dict[ticker] = flatten_yf_df(raw)
            except Exception:
                pass

        if not data_dict:
            return {"signals": [], "stats": {"reason": "no_data"}, "market_regime": "UNKNOWN"}

        signals = engine.scan_universe(
            tickers=list(data_dict.keys()),
            data_dict=data_dict,
            portfolio_value=body.portfolio_value,
        )

        filtered = [s for s in signals if s.get("quality_score", 0) >= body.min_quality]
        filtered = filtered[:body.top_n]

        stats = {
            "stocks_scanned":  len(tickers),
            "stocks_with_data": len(data_dict),
            "raw_signals":     len(signals),
            "filtered_signals": len(filtered),
            "reason": "success" if filtered else "no_qualifying",
        }
        return {"signals": filtered, "stats": stats, "market_regime": "RISK_ON"}

    except Exception as e:
        logger.exception("SmallCap scan failed")
        return {"signals": [], "stats": {"reason": "error", "error": str(e)},
                "market_regime": "UNKNOWN", "error": str(e)}
