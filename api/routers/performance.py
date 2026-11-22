"""
Performance analytics router.
GET /api/performance        - summary stats + open/closed trades (open trades enriched with live prices)
GET /api/performance/weekly-report - weekly report text

Closed trades definition: status NOT IN ('OPEN', 'PENDING')
"""

import logging
import math
from fastapi import APIRouter, Depends, Query
from typing import Optional
from api.deps import get_paper_storage
from api.auth import get_current_user_id
from api.utils import sanitize_for_json

router = APIRouter()
logger = logging.getLogger(__name__)


def is_win(trade: dict) -> bool:
    """A trade is a win if realized_pnl > 0."""
    pnl = trade.get("realized_pnl")
    return bool(pnl is not None and pnl > 0)


def _fetch_live_prices(tickers: list) -> dict:
    """Batch-fetch latest prices via yfinance. Returns {ticker: price}. Never raises."""
    if not tickers:
        return {}
    try:
        import yfinance as yf
        # yfinance 1.x: data["Close"] is always a DataFrame (even single ticker).
        # Pass list (not string join) and access each ticker uniformly.
        data = yf.download(tickers, period="5d", progress=False, auto_adjust=True)
        if data.empty:
            return {}
        prices = {}
        close = data["Close"]  # DataFrame: columns = ticker names
        for t in tickers:
            try:
                if t not in close.columns:
                    continue
                col = close[t]
                val = col.dropna()
                if not val.empty:
                    p = round(float(val.iloc[-1].item()), 4)
                    # NaN/Inf fiyatı ALMA — (cp - entry) hesabı NaN üretir ve
                    # JSON serileştirme 500'e düşer (trades.py'deki guard'ın eşi).
                    if math.isfinite(p) and p > 0:
                        prices[t] = p
            except Exception:
                pass
        return prices
    except Exception as e:
        logger.warning(f"yfinance live price fetch failed: {e}")
        return {}


def _enrich_open_trades(open_trades: list) -> list:
    """Add current_price, unrealized_pnl, unrealized_pnl_pct to open trades."""
    tickers = list({t["ticker"] for t in open_trades if t.get("ticker")})
    live = _fetch_live_prices(tickers)
    result = []
    for t in open_trades:
        trade = dict(t)
        cp = live.get(trade["ticker"])
        if cp:
            entry = trade.get("entry_price") or 0
            size = trade.get("position_size") or 100
            trade["current_price"] = cp
            if entry:
                from swing_trader.paper_trading.execution import mark_pnl
                pnl, pct = mark_pnl(trade, cp)
                trade["unrealized_pnl"] = round(pnl, 2)
                trade["unrealized_pnl_pct"] = round(pct, 2)
        result.append(trade)
    return result


@router.get("")
def get_performance(user_id: Optional[str] = Depends(get_current_user_id),
                    strategy: Optional[str] = Query(None, description="H6S | LEGACY | ALL (varsayılan: tümü)")):
    from swing_trader.paper_trading.storage import strategy_of
    storage = get_paper_storage()
    all_trades   = storage.get_all_trades(user_id)  or []

    closed_trades  = storage.get_closed_trades(limit=1000, user_id=user_id)
    want = (strategy or "ALL").upper()
    if want != "ALL":   # 2026-11-22: H6S ana strateji, LEGACY kontrol grubu — ayrı hesaplar
        all_trades = [t for t in all_trades if strategy_of(t) == want]
        closed_trades = [t for t in (closed_trades or []) if strategy_of(t) == want]
    open_trades    = [t for t in all_trades if t.get("status") == "OPEN"]
    pending_trades = [t for t in all_trades if t.get("status") == "PENDING"]

    # Filter out REJECTED trades for P&L calculations to avoid diluting averages
    # (Optional: check if user wants this. Usually in trading, rejected/canceled are ignored)
    valid_closed = [t for t in closed_trades if t.get("status") != "REJECTED"]
    
    wins   = [t for t in valid_closed if (t.get("realized_pnl") or 0) > 0]
    losses = [t for t in valid_closed if (t.get("realized_pnl") or 0) < 0]
    breakeven = [t for t in valid_closed if (t.get("realized_pnl") or 0) == 0]

    wins_pnl   = [t.get("realized_pnl") or 0 for t in wins]
    losses_pnl = [abs(t.get("realized_pnl") or 0) for t in losses]

    total_pnl   = sum((t.get("realized_pnl") or 0) for t in valid_closed)
    total_closed = len(valid_closed)
    total_decisive = len(wins) + len(losses)
    
    win_rate     = round(len(wins) / total_decisive * 100, 1) if total_decisive > 0 else 0
    avg_win      = sum(wins_pnl)  / len(wins_pnl)   if wins_pnl   else 0
    avg_loss     = -sum(losses_pnl) / len(losses_pnl) if losses_pnl else 0

    # P&L % metrics
    pnl_pcts = [t.get("realized_pnl_pct") or 0 for t in valid_closed if t.get("realized_pnl_pct") is not None]
    total_pnl_pct = round(sum(pnl_pcts), 2)
    avg_pnl_pct = round(sum(pnl_pcts) / len(pnl_pcts), 2) if pnl_pcts else 0

    # Maliyet sonrası (gerçek para) — paper fiyatları maliyetsiz; tek kaynak slippage_bps_per_side.
    from swing_trader.paper_trading.costs import net_of_cost, round_trip_cost_pct
    cost_pct = round_trip_cost_pct()
    net_rows = []
    for t in valid_closed:
        n_usd, n_pct = net_of_cost(t, cost_pct)
        if n_pct is not None:
            t["net_pnl"], t["net_pnl_pct"] = n_usd, n_pct
            net_rows.append((n_usd, n_pct))
    net_wins = sum(1 for _, p in net_rows if p > 0)
    net_losses = sum(1 for _, p in net_rows if p < 0)
    net_summary = {
        "cost_round_trip_pct": round(cost_pct, 3),
        "net_total_pnl": round(sum(u for u, _ in net_rows), 2),
        "net_total_pnl_pct": round(sum(p for _, p in net_rows), 2),
        "net_avg_pnl_pct": round(sum(p for _, p in net_rows) / len(net_rows), 2) if net_rows else 0,
        "net_wins": net_wins,
        "net_losses": net_losses,
        "net_win_rate": round(net_wins / (net_wins + net_losses) * 100, 1) if (net_wins + net_losses) else 0,
    }

    # 30 most-recent closed, sorted newest first
    recent = sorted(
        closed_trades,
        key=lambda x: x.get("exit_date") or "",
        reverse=True,
    )[:30]

    # Enrich open trades with live prices
    enriched_open = _enrich_open_trades(open_trades)

    # sanitize: eski kayıtlarda depolanmış NaN/Inf (ör. 2026-08-02 id=72/76) veya
    # numpy skalerleri JSON'a çevrilemez → endpoint 500 verirdi. Son bariyer.
    return sanitize_for_json({
        "summary": {
            "total_trades":   len(all_trades),
            "open_trades":    len(open_trades),
            "pending_trades": len(pending_trades),
            "closed_trades":  total_closed,
            "wins":   len(wins),
            "losses": len(losses),
            "breakeven": len(breakeven),
            "win_rate":     win_rate,
            "total_pnl":    round(total_pnl, 2),
            "total_pnl_pct": total_pnl_pct,
            "avg_pnl_pct":  avg_pnl_pct,
            "avg_win":      round(avg_win, 2),
            "avg_loss":     round(avg_loss, 2),
            **net_summary,
        },
        "recent_closed": recent,
        "open_trades":   enriched_open,
    })

