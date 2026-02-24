"""
Manual ticker lookup router.
POST /api/lookup   - analyze one or more tickers on demand
"""

import logging
import yfinance as yf
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from api.deps import get_smallcap_engine

router = APIRouter()
logger = logging.getLogger(__name__)


class LookupRequest(BaseModel):
    tickers: List[str]
    portfolio_value: float = 10000


@router.post("")
def lookup_tickers(body: LookupRequest):
    """Analyze specific tickers and return signals."""
    engine = get_smallcap_engine()
    results = []

    for ticker in body.tickers:
        ticker = ticker.upper().strip()
        try:
            # Fetch OHLCV
            df = yf.download(ticker, period="60d", interval="1d",
                             auto_adjust=True, progress=False, timeout=10)
            if df is None or len(df) < 20:
                results.append({"ticker": ticker, "error": "Insufficient data (need 20+ days)"})
                continue

            df = df.reset_index()
            df.columns = [str(c) for c in df.columns]

            # Get optional stock info
            info = {}
            try:
                info = yf.Ticker(ticker).info or {}
            except Exception:
                pass

            signal = engine.scan_stock(ticker, df, stock_info=info)
            if signal:
                results.append(signal)
            else:
                results.append({"ticker": ticker, "error": "No signal — entry criteria not met"})

        except Exception as e:
            logger.warning(f"Lookup failed for {ticker}: {e}")
            results.append({"ticker": ticker, "error": str(e)})

    return {"results": results, "count": len([r for r in results if "error" not in r])}
