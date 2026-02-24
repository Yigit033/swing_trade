"""
Paper Trades CRUD router.
GET  /api/trades          - list all trades (filterable)
POST /api/trades          - add a new trade
GET  /api/trades/{id}     - single trade
PATCH /api/trades/{id}    - update fields
DELETE /api/trades/{id}   - remove trade
POST /api/trades/{id}/close - close a trade
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from api.deps import get_paper_storage, get_paper_tracker

router = APIRouter()


class TradeIn(BaseModel):
    ticker: str
    entry_date: str
    entry_price: float
    stop_loss: float
    target: float
    swing_type: str = "A"
    quality_score: float = 0
    position_size: int = 100
    max_hold_days: int = 7
    notes: str = ""
    trailing_stop: Optional[float] = None
    initial_stop: Optional[float] = None
    atr: Optional[float] = None
    signal_price: Optional[float] = None
    status: str = "OPEN"


class CloseTradeIn(BaseModel):
    exit_price: float
    exit_date: Optional[str] = None
    notes: str = ""


@router.get("")
def list_trades(status: Optional[str] = Query(None)):
    storage = get_paper_storage()
    if status:
        trades = storage.get_trades(status=status)
    else:
        trades = storage.get_trades()
    return {"trades": trades}


@router.get("/{trade_id}")
def get_trade(trade_id: int):
    storage = get_paper_storage()
    trade = storage.get_trade(trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    return trade


@router.post("", status_code=201)
def add_trade(body: TradeIn):
    storage = get_paper_storage()
    trade_dict = body.dict()
    if trade_dict["trailing_stop"] is None:
        trade_dict["trailing_stop"] = trade_dict["stop_loss"]
    if trade_dict["initial_stop"] is None:
        trade_dict["initial_stop"] = trade_dict["stop_loss"]
    if trade_dict["signal_price"] is None:
        trade_dict["signal_price"] = trade_dict["entry_price"]
    if trade_dict["atr"] is None:
        trade_dict["atr"] = 0.0
    trade_id = storage.add_trade(trade_dict)
    return {"id": trade_id, "message": "Trade added"}


@router.patch("/{trade_id}")
def update_trade(trade_id: int, updates: dict):
    storage = get_paper_storage()
    ok = storage.update_trade(trade_id, updates)
    if not ok:
        raise HTTPException(status_code=404, detail="Trade not found")
    return {"message": "Updated"}


@router.delete("/{trade_id}")
def delete_trade(trade_id: int):
    storage = get_paper_storage()
    ok = storage.delete_trade(trade_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Trade not found")
    return {"message": "Deleted"}


@router.post("/{trade_id}/close")
def close_trade(trade_id: int, body: CloseTradeIn):
    from datetime import datetime
    tracker = get_paper_tracker()
    exit_date = body.exit_date or datetime.now().strftime("%Y-%m-%d")
    result = tracker.close_trade(trade_id, body.exit_price, exit_date=exit_date, notes=body.notes)
    if not result:
        raise HTTPException(status_code=404, detail="Trade not found or already closed")
    return {"message": "Trade closed", "trade": result}


@router.post("/update-prices")
def update_prices():
    """Fetch latest prices and update all open trades."""
    tracker = get_paper_tracker()
    updated = tracker.update_open_trades()
    return {"message": f"Updated {updated} trades"}
