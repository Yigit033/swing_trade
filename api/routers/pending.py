"""
Pending trades router.
GET  /api/pending              - list pending trades
POST /api/pending/check        - auto-check all pending (by current price)
POST /api/pending/{id}/confirm - manually confirm one trade
"""

from fastapi import APIRouter, HTTPException
from api.deps import get_paper_storage, get_paper_tracker

router = APIRouter()


@router.get("")
def list_pending():
    storage = get_paper_storage()
    trades = storage.get_trades(status="PENDING")
    return {"pending": trades, "count": len(trades)}


@router.post("/check")
def check_pending():
    """Auto-check all pending trades against current market prices."""
    tracker = get_paper_tracker()
    confirmed = tracker.check_pending_confirmations()
    return {"confirmed": confirmed, "message": f"{len(confirmed)} trades confirmed"}


@router.post("/{trade_id}/confirm")
def confirm_trade(trade_id: int):
    """Manually confirm a single pending trade."""
    tracker = get_paper_tracker()
    result = tracker.confirm_trade(trade_id)
    if not result:
        raise HTTPException(status_code=404, detail="Trade not found or not in PENDING status")
    return {"message": "Trade confirmed", "trade": result}
