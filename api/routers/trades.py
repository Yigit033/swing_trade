"""
Paper Trades CRUD router.
GET  /api/trades          - list all trades (filterable by status)
POST /api/trades          - add a new trade
GET  /api/trades/{id}     - single trade
PATCH /api/trades/{id}    - update fields
DELETE /api/trades/{id}   - remove trade
POST /api/trades/{id}/close - close a trade
POST /api/trades/update-prices - fetch latest prices for all open trades
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from api.deps import get_paper_storage, get_paper_tracker
from api.auth import get_current_user_id

router = APIRouter()
logger = logging.getLogger(__name__)


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


# Short-TTL price cache so listing 40-50 trades (mostly closed) doesn't hammer
# yfinance on every page load. One batch request per cold ticker set / 5 min.
_PRICE_CACHE: dict = {}  # ticker -> (price, fetched_at_epoch)
_PRICE_CACHE_TTL_SEC = 300


def _enrich_open_trades_inline(trades: list) -> list:
    """
    Attach live current_price to ALL trades (open, pending AND closed) via one
    batched yfinance call + 5-min cache.

    - OPEN: also computes unrealized P&L vs entry.
    - CLOSED: current_price enables the "where is it trading now vs my exit?"
      post-trade review (exit quality feedback for the trader).
    Uses period='5d' so weekends/holidays still return the last trading day's close.
    """
    import time as _time

    all_tickers = list({t["ticker"] for t in trades if t.get("ticker")})
    if not all_tickers:
        return trades

    now = _time.time()
    prices = {
        tk: p for tk, (p, ts) in _PRICE_CACHE.items()
        if tk in all_tickers and now - ts < _PRICE_CACHE_TTL_SEC
    }
    missing = [tk for tk in all_tickers if tk not in prices]

    if missing:
        try:
            import yfinance as yf
            # yfinance 1.x: data["Close"] is always a DataFrame (even single ticker).
            data = yf.download(missing, period="5d", progress=False, auto_adjust=True)
            if not data.empty:
                close = data["Close"]  # DataFrame: columns = ticker names
                for tk in missing:
                    try:
                        col = close[tk] if tk in close.columns else close.iloc[:, 0]
                        val = col.dropna()
                        if not val.empty:
                            p = round(float(val.iloc[-1].item()), 4)
                            prices[tk] = p
                            _PRICE_CACHE[tk] = (p, now)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Inline price enrichment failed: {e}")

    for t in trades:
        cp = prices.get(t.get("ticker"))
        if not cp:
            continue
        status = t.get("status")
        t["current_price"] = cp
        if status == "OPEN":
            entry = t.get("entry_price") or 0
            size = t.get("position_size") or 100
            if entry:
                t["unrealized_pnl"] = round((cp - entry) * size, 2)
                t["unrealized_pnl_pct"] = round(((cp / entry) - 1) * 100, 2)
        elif status not in ("OPEN", "PENDING"):
            # Post-exit drift: how far has it moved since we exited?
            exit_px = t.get("exit_price") or 0
            if exit_px:
                t["since_exit_pct"] = round((cp / exit_px - 1) * 100, 2)
    return trades


@router.get("")
def list_trades(
    status: Optional[str] = Query(None),
    user_id: Optional[str] = Depends(get_current_user_id),
):
    storage = get_paper_storage()
    all_trades = storage.get_all_trades(user_id) or []
    if status:
        all_trades = [t for t in all_trades if t.get("status") == status]
    # Enrich OPEN trades that have no current_price yet
    all_trades = _enrich_open_trades_inline(all_trades)
    return {"trades": all_trades}


@router.get("/last-update")
def last_update(user_id: Optional[str] = Depends(get_current_user_id)):
    """
    Return the last time paper trades were updated (any field).
    
    This uses the MAX(updated_at) across all trades and is used by the
    frontend to show a 'Last data refresh' indicator on the Paper Trades page.
    """
    storage = get_paper_storage()
    ts = storage.get_last_update_timestamp()
    return {"last_update": ts}


@router.post("/update-prices")
def update_prices(user_id: Optional[str] = Depends(get_current_user_id)):
    """Fetch latest prices and update all open/pending trades."""
    tracker = get_paper_tracker()
    storage = get_paper_storage()
    # Resolve PENDING → OPEN/REJECTED first (same rules as /api/pending/check).
    # Without this, pending only advances when the Pending page is opened.
    tracker.confirm_pending_trades(user_id)
    updated = tracker.update_all_open_trades(user_id)
    # Record that a price refresh was triggered (even if there were no open trades)
    storage.touch_last_price_update()
    return {"message": f"Updated {len(updated) if updated else 0} trades", "trades": updated or []}


@router.get("/{trade_id}")
def get_trade(
    trade_id: int,
    user_id: Optional[str] = Depends(get_current_user_id),
):
    storage = get_paper_storage()
    trade = storage.get_trade_by_id(trade_id, user_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    return trade


@router.post("", status_code=201)
def add_trade(
    body: TradeIn,
    user_id: Optional[str] = Depends(get_current_user_id),
):
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
    trade_id = storage.add_trade(trade_dict, user_id)
    return {"id": trade_id, "message": "Trade added"}


@router.patch("/{trade_id}")
def update_trade(
    trade_id: int,
    updates: dict,
    user_id: Optional[str] = Depends(get_current_user_id),
):
    storage = get_paper_storage()
    ok = storage.update_trade(trade_id, updates, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Trade not found")
    return {"message": "Updated"}


@router.delete("/{trade_id}")
def delete_trade(
    trade_id: int,
    user_id: Optional[str] = Depends(get_current_user_id),
):
    storage = get_paper_storage()
    ok = storage.delete_trade(trade_id, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Trade not found")
    return {"message": "Deleted"}


@router.post("/{trade_id}/close")
def close_trade(
    trade_id: int,
    body: CloseTradeIn,
    user_id: Optional[str] = Depends(get_current_user_id),
):
    """Manually close a trade. Tracker sets status=MANUAL and calculates realized P&L."""
    storage = get_paper_storage()
    trade = storage.get_trade_by_id(trade_id, user_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    tracker = get_paper_tracker()
    ok = tracker.manual_close_trade(
        trade_id,
        exit_price=body.exit_price,
        notes=body.notes or "Manually closed",
        user_id=user_id,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Could not close trade")

    # Fetch the updated trade from storage after closing (tracker returns bool not dict)
    updated = storage.get_trade_by_id(trade_id)
    return {"message": "Trade closed", "trade": updated}
