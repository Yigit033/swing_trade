"""
Manual ticker lookup router.
POST /api/lookup   - analyze one or more tickers
"""

import logging
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from api.deps import get_config, get_smallcap_engine

router = APIRouter()
logger = logging.getLogger(__name__)


class LookupRequest(BaseModel):
    tickers: List[str]
    portfolio_value: float = 10000


@router.post("")
def lookup_tickers(body: LookupRequest):
    """Analyze specific tickers and return signals."""
    config = get_config()
    engine = get_smallcap_engine()

    results = []
    for ticker in body.tickers:
        try:
            ticker = ticker.upper().strip()
            result = engine.analyze_ticker(ticker, body.portfolio_value)
            if result:
                results.append(result)
        except Exception as e:
            logger.warning(f"Lookup failed for {ticker}: {e}")
            results.append({"ticker": ticker, "error": str(e)})

    return {"results": results, "count": len(results)}
