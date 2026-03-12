"""
Performance analytics router.
GET /api/performance        - summary stats + open/closed trades (open trades enriched with live prices)
GET /api/performance/weekly-report - weekly report text

Closed trades definition: status NOT IN ('OPEN', 'PENDING')
"""

import logging
from fastapi import APIRouter
from api.deps import get_paper_storage

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
        tickers_str = " ".join(tickers)
        data = yf.download(tickers_str, period="1d", interval="5m",
                           progress=False, auto_adjust=True)
        if data.empty:
            return {}
        prices = {}
        close = data["Close"]
        if len(tickers) == 1:
            # Single ticker — Close is a Series
            val = close.dropna()
            if not val.empty:
                prices[tickers[0]] = round(float(val.iloc[-1]), 4)
        else:
            for t in tickers:
                try:
                    val = close[t].dropna()
                    if not val.empty:
                        prices[t] = round(float(val.iloc[-1]), 4)
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
                trade["unrealized_pnl"] = round((cp - entry) * size, 2)
                trade["unrealized_pnl_pct"] = round(((cp / entry) - 1) * 100, 2)
        result.append(trade)
    return result


@router.get("")
def get_performance():
    storage = get_paper_storage()
    all_trades   = storage.get_all_trades()  or []

    closed_trades  = storage.get_closed_trades(limit=1000)
    open_trades    = [t for t in all_trades if t.get("status") == "OPEN"]
    pending_trades = [t for t in all_trades if t.get("status") == "PENDING"]

    # Filter out REJECTED trades for P&L calculations to avoid diluting averages
    # (Optional: check if user wants this. Usually in trading, rejected/canceled are ignored)
    valid_closed = [t for t in closed_trades if t.get("status") != "REJECTED"]
    
    wins   = [t for t in valid_closed if is_win(t)]
    losses = [t for t in valid_closed if not is_win(t)]

    wins_pnl   = [t.get("realized_pnl") or 0 for t in wins]
    losses_pnl = [abs(t.get("realized_pnl") or 0) for t in losses if (t.get("realized_pnl") or 0) < 0]

    total_pnl   = sum((t.get("realized_pnl") or 0) for t in valid_closed)
    total_closed = len(valid_closed)
    win_rate     = round(len(wins) / total_closed * 100, 1) if total_closed else 0
    avg_win      = sum(wins_pnl)  / len(wins_pnl)   if wins_pnl   else 0
    avg_loss     = -sum(losses_pnl) / len(losses_pnl) if losses_pnl else 0

    # P&L % metrics
    pnl_pcts = [t.get("realized_pnl_pct") or 0 for t in valid_closed if t.get("realized_pnl_pct") is not None]
    total_pnl_pct = round(sum(pnl_pcts), 2)
    avg_pnl_pct = round(sum(pnl_pcts) / len(pnl_pcts), 2) if pnl_pcts else 0

    # 30 most-recent closed, sorted newest first
    recent = sorted(
        closed_trades,
        key=lambda x: x.get("exit_date") or "",
        reverse=True,
    )[:30]

    # Enrich open trades with live prices
    enriched_open = _enrich_open_trades(open_trades)

    return {
        "summary": {
            "total_trades":   len(all_trades),
            "open_trades":    len(open_trades),
            "pending_trades": len(pending_trades),
            "closed_trades":  total_closed,
            "wins":   len(wins),
            "losses": len([t for t in losses if (t.get("realized_pnl") or 0) < 0]),
            "breakeven": len([t for t in losses if (t.get("realized_pnl") or 0) == 0]),
            "win_rate":     win_rate,
            "total_pnl":    round(total_pnl, 2),
            "total_pnl_pct": total_pnl_pct,
            "avg_pnl_pct":  avg_pnl_pct,
            "avg_win":      round(avg_win, 2),
            "avg_loss":     round(avg_loss, 2),
        },
        "recent_closed": recent,
        "open_trades":   enriched_open,
    }


@router.get("/weekly-report")
def weekly_report():
    storage = get_paper_storage()
    trades = storage.get_all_trades() or []
    try:
        from swing_trader.paper_trading.reporter import PaperTradeReporter
        reporter = PaperTradeReporter()
        report = reporter.generate_weekly_report(trades)
        return {"report": report}
    except Exception as e:
        closed = storage.get_closed_trades(limit=1000)
        wins = [t for t in closed if is_win(t)]
        wr = round(len(wins) / len(closed) * 100, 1) if closed else 0
        return {"report": f"Rapor üretilemedi ({e}). Özet: {len(closed)} kapalı trade, %{wr} win rate."}
