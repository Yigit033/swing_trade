"""
SmallCap Scanner router.
POST /api/scanner/smallcap  - run SmallCap momentum scan
GET  /api/scanner/status    - scanner status / last run info
"""

import logging
from fastapi import APIRouter
from pydantic import BaseModel
from api.deps import get_smallcap_engine

router = APIRouter()
logger = logging.getLogger(__name__)


class ScanRequest(BaseModel):
    portfolio_value: float = 10000
    min_quality: int = 65
    top_n: int = 10


@router.post("/smallcap")
def run_smallcap_scan(body: ScanRequest):
    """Run the SmallCap Momentum Scanner."""
    engine = get_smallcap_engine()
    try:
        results = engine.scan(
            min_quality=body.min_quality,
            top_n=body.top_n,
            portfolio_value=body.portfolio_value,
        )
        return {
            "signals": results.get("signals", []),
            "stats": results.get("stats", {}),
            "market_regime": results.get("market_regime", "UNKNOWN"),
        }
    except Exception as e:
        logger.exception("SmallCap scan failed")
        return {"signals": [], "stats": {}, "error": str(e)}
