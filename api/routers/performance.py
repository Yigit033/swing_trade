"""
Performance analytics router.
GET /api/performance        - summary stats + open/closed trades
GET /api/performance/weekly-report - weekly report text

Closed trades definition: status NOT IN ('OPEN', 'PENDING')
This matches PaperTradeStorage.get_closed_trades() and Streamlit's reporter.
REJECTED trades ARE included (they were entered as PENDING, then rejected at
next-day confirmation — real trades that simply didn't execute at a favourable
gap, so they count in the win-rate denominator with pnl=0/negative).
"""

import logging
from fastapi import APIRouter
from api.deps import get_paper_storage

router = APIRouter()
logger = logging.getLogger(__name__)


def is_win(trade: dict) -> bool:
    """A trade is a win if realized_pnl > 0. No shortcuts."""
    pnl = trade.get("realized_pnl")
    return bool(pnl is not None and pnl > 0)


@router.get("")
def get_performance():
    storage = get_paper_storage()
    all_trades   = storage.get_all_trades()  or []

    # Use storage's own closed-trades query (NOT IN 'OPEN','PENDING')
    # so the count matches Streamlit's reporter.
    closed_trades  = storage.get_closed_trades(limit=1000)
    open_trades    = [t for t in all_trades if t.get("status") == "OPEN"]
    pending_trades = [t for t in all_trades if t.get("status") == "PENDING"]

    wins   = [t for t in closed_trades if is_win(t)]
    losses = [t for t in closed_trades if not is_win(t)]  # includes breakeven / pnl=0

    wins_pnl   = [t.get("realized_pnl") or 0 for t in wins]
    losses_pnl = [abs(t.get("realized_pnl") or 0) for t in losses if (t.get("realized_pnl") or 0) < 0]

    total_pnl = sum((t.get("realized_pnl") or 0) for t in closed_trades)
    total_closed = len(closed_trades)
    win_rate  = round(len(wins) / total_closed * 100, 1) if total_closed else 0
    avg_win   = sum(wins_pnl)  / len(wins_pnl)   if wins_pnl   else 0
    avg_loss  = -sum(losses_pnl) / len(losses_pnl) if losses_pnl else 0   # negative number

    # Maximum 30 most-recent closed, sorted newest first
    recent = sorted(
        closed_trades,
        key=lambda x: x.get("exit_date") or "",
        reverse=True,
    )[:30]

    return {
        "summary": {
            "total_trades":   len(all_trades),
            "open_trades":    len(open_trades),
            "pending_trades": len(pending_trades),
            "closed_trades":  total_closed,
            "wins":   len(wins),
            "losses": len([t for t in losses if (t.get("realized_pnl") or 0) < 0]),
            "breakeven": len([t for t in losses if (t.get("realized_pnl") or 0) == 0]),
            "win_rate":  win_rate,
            "total_pnl": round(total_pnl, 2),
            "avg_win":   round(avg_win, 2),
            "avg_loss":  round(avg_loss, 2),
        },
        "recent_closed": recent,
        "open_trades":   open_trades,
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
