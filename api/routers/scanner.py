"""
SmallCap Scanner router.
POST /api/scanner/smallcap  - run SmallCap momentum scan
"""

import logging
import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import APIRouter, Query
from pydantic import BaseModel
from api.deps import get_smallcap_engine

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/chart")
def get_chart_data(ticker: str = Query(...), period: str = Query("3mo")):
    """Return OHLCV + RSI + MACD + Volume data for charting."""
    try:
        df = yf.download(ticker.upper(), period=period, interval="1d",
                         auto_adjust=True, progress=False, timeout=10)
        if df is None or len(df) < 5:
            return {"error": "No data available", "ticker": ticker}

        df = df.reset_index()
        df.columns = [str(c) for c in df.columns]

        # RSI (14)
        close = df["Close"].squeeze()
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

        # EMA 20 / 50
        df["EMA20"] = close.ewm(span=20, adjust=False).mean()
        df["EMA50"] = close.ewm(span=50, adjust=False).mean()

        # Volume MA 20
        vol = df["Volume"].squeeze()
        df["Volume_MA"] = vol.rolling(20).mean()

        def safe_float(v):
            try:
                f = float(v)
                return None if (f != f) else round(f, 4)  # nan check
            except Exception:
                return None

        rows = []
        for _, row in df.iterrows():
            date = str(row["Date"])[:10] if "Date" in row else str(row.name)[:10]
            rows.append({
                "date": date,
                "open":   safe_float(row.get("Open")),
                "high":   safe_float(row.get("High")),
                "low":    safe_float(row.get("Low")),
                "close":  safe_float(row.get("Close")),
                "volume": safe_float(row.get("Volume")),
                "rsi":    safe_float(row.get("RSI")),
                "macd":   safe_float(row.get("MACD")),
                "macd_signal": safe_float(row.get("MACD_signal")),
                "macd_hist":   safe_float(row.get("MACD_hist")),
                "ema20":  safe_float(row.get("EMA20")),
                "ema50":  safe_float(row.get("EMA50")),
                "volume_ma": safe_float(row.get("Volume_MA")),
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
        # 1. Get universe tickers
        tickers = engine.get_small_cap_universe(use_finviz=True, max_tickers=200)
        logger.info(f"SmallCap universe: {len(tickers)} tickers")

        # 2. Download OHLCV data via yfinance
        data_dict: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                df = yf.download(ticker, period="60d", interval="1d",
                                 auto_adjust=True, progress=False, timeout=8)
                if df is not None and len(df) >= 20:
                    df = df.reset_index()
                    df.columns = [str(c) for c in df.columns]
                    data_dict[ticker] = df
            except Exception:
                pass

        if not data_dict:
            return {"signals": [], "stats": {"reason": "no_data"}, "market_regime": "UNKNOWN"}

        # 3. Scan universe
        signals = engine.scan_universe(
            tickers=list(data_dict.keys()),
            data_dict=data_dict,
            portfolio_value=body.portfolio_value,
        )

        # 4. Filter by quality
        filtered = [s for s in signals if s.get("quality_score", 0) >= body.min_quality]
        filtered = filtered[:body.top_n]

        stats = {
            "stocks_scanned": len(tickers),
            "stocks_with_data": len(data_dict),
            "raw_signals": len(signals),
            "filtered_signals": len(filtered),
            "reason": "success" if filtered else "no_qualifying",
        }
        return {"signals": filtered, "stats": stats, "market_regime": "RISK_ON"}

    except Exception as e:
        logger.exception("SmallCap scan failed")
        return {"signals": [], "stats": {"reason": "error", "error": str(e)}, "market_regime": "UNKNOWN", "error": str(e)}
