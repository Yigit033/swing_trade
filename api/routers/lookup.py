"""
Manual ticker lookup router.
POST /api/lookup   - analyze one or more tickers on demand
"""

import logging
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


@router.post("")
def lookup_tickers(body: LookupRequest):
    """Analyze specific tickers and return signals."""
    engine = get_smallcap_engine()
    results = []

    for ticker in body.tickers:
        ticker = ticker.upper().strip()
        try:
            raw = yf.download(ticker, period="60d", interval="1d",
                              auto_adjust=True, progress=False)
            if raw is None or raw.empty or len(raw) < 20:
                results.append({"ticker": ticker, "error": "Insufficient data (need 20+ days)"})
                continue

            df = flatten_yf_df(raw)

            # Get optional stock info for better signal generation
            info = {}
            try:
                info = yf.Ticker(ticker).info or {}
            except Exception:
                pass

            signal = engine.scan_stock(ticker, df, stock_info=info)
            if signal:
                # Ensure portfolio_value-based position sizing
                if body.portfolio_value and body.portfolio_value != 10000:
                    # Recalculate shares if signal has position_size
                    if signal.get("entry_price") and body.portfolio_value:
                        alloc = body.portfolio_value * 0.1  # 10% per trade
                        signal["position_size"] = max(1, int(alloc / signal["entry_price"]))
                results.append(signal)
            else:
                results.append({"ticker": ticker, "error": "No signal — entry criteria not met"})

        except Exception as e:
            logger.warning(f"Lookup failed for {ticker}: {e}")
            results.append({"ticker": ticker, "error": str(e)})

    return {"results": results, "count": len([r for r in results if "error" not in r])}
