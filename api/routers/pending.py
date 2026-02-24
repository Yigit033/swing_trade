"""
Pending trades router.
GET  /api/pending              - list pending trades
POST /api/pending/check        - auto-confirm all pending (by next-day Open price)
POST /api/pending/{id}/confirm - manually move one trade from PENDING → OPEN
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
from api.deps import get_paper_storage, get_paper_tracker

router = APIRouter()


@router.get("")
def list_pending():
    storage = get_paper_storage()
    all_trades = storage.get_all_trades() or []
    pending = [t for t in all_trades if t.get("status") == "PENDING"]
    return {"pending": pending, "count": len(pending)}


@router.post("/check")
def check_pending():
    """Auto-confirm pending trades at next-day Open price (gap filter applied)."""
    tracker = get_paper_tracker()
    confirmed = tracker.confirm_pending_trades()
    return {"confirmed": confirmed or [], "message": f"{len(confirmed or [])} trades processed"}


@router.post("/{trade_id}/confirm")
def confirm_trade(trade_id: int):
    """Manually confirm a single pending trade → move to OPEN at entry_price."""
    storage = get_paper_storage()
    trade = storage.get_trade_by_id(trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    if trade.get("status") != "PENDING":
        raise HTTPException(status_code=400, detail=f"Trade is {trade.get('status')}, not PENDING")

    ok = storage.update_trade(trade_id, {
        "status": "OPEN",
        "updated_at": datetime.now().isoformat(),
    })
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to update trade")
    return {"message": "Trade confirmed and moved to OPEN", "trade": storage.get_trade_by_id(trade_id)}
