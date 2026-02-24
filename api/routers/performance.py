"""
Performance analytics router.
GET /api/performance        - summary stats + open/closed trades
GET /api/performance/weekly-report - weekly report text
"""

from fastapi import APIRouter
from api.deps import get_paper_storage

router = APIRouter()

# Statuses that mean a trade is fully closed (system uses these actual values)
CLOSED_STATUSES = {"STOPPED", "TRAILED", "TARGET", "MANUAL", "WIN", "LOSS", "CLOSED"}
OPEN_STATUSES   = {"OPEN"}
PENDING_STATUSES = {"PENDING"}
# REJECTED trades don't contribute to P&L (entry was never confirmed)
REJECTED_STATUSES = {"REJECTED"}


def is_win(trade: dict) -> bool:
    pnl = trade.get("realized_pnl")
    if pnl is not None:
        return pnl > 0
    # Fallback for TARGET
    if trade.get("status") == "TARGET":
        return True
    return False


@router.get("")
def get_performance():
    storage = get_paper_storage()
    all_trades = storage.get_all_trades() or []

    open_trades    = [t for t in all_trades if t.get("status") in OPEN_STATUSES]
    pending_trades = [t for t in all_trades if t.get("status") in PENDING_STATUSES]
    closed_trades  = [t for t in all_trades if t.get("status") in CLOSED_STATUSES]
    wins   = [t for t in closed_trades if is_win(t)]
    losses = [t for t in closed_trades if not is_win(t)]

    total_pnl = sum((t.get("realized_pnl") or 0) for t in closed_trades)
    win_rate  = round(len(wins) / len(closed_trades) * 100, 1) if closed_trades else 0
    avg_win   = sum((t.get("realized_pnl") or 0) for t in wins) / len(wins) if wins else 0
    avg_loss  = sum((t.get("realized_pnl") or 0) for t in losses) / len(losses) if losses else 0

    # Recent 30 closed, sorted newest first
    recent = sorted(
        closed_trades,
        key=lambda x: x.get("exit_date") or "",
        reverse=True
    )[:30]

    return {
        "summary": {
            "total_trades":   len(all_trades),
            "open_trades":    len(open_trades),
            "pending_trades": len(pending_trades),
            "closed_trades":  len(closed_trades),
            "wins":   len(wins),
            "losses": len(losses),
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
        closed = [t for t in trades if t.get("status") in CLOSED_STATUSES]
        wins = [t for t in closed if is_win(t)]
        wr = round(len(wins) / len(closed) * 100, 1) if closed else 0
        return {"report": f"Rapor üretilemedi (reporter hatası: {e}). Özet: {len(closed)} kapalı trade, %{wr} win rate."}
