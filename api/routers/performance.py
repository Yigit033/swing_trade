"""
Performance analytics router.
GET /api/performance        - summary stats + closed trades
GET /api/performance/report - weekly HTML/text report
"""

from fastapi import APIRouter
from api.deps import get_paper_storage, get_paper_reporter

router = APIRouter()


@router.get("")
def get_performance():
    storage = get_paper_storage()
    reporter = get_paper_reporter()

    all_trades   = storage.get_trades() or []
    open_trades  = [t for t in all_trades if t.get("status") == "OPEN"]
    closed_trades = [t for t in all_trades if t.get("status") in ("WIN", "LOSS", "CLOSED")]
    pending_trades = [t for t in all_trades if t.get("status") == "PENDING"]

    # Basic statistics
    wins  = [t for t in closed_trades if t.get("realized_pnl", 0) > 0]
    losses = [t for t in closed_trades if t.get("realized_pnl", 0) <= 0]

    total_pnl = sum(t.get("realized_pnl", 0) or 0 for t in closed_trades)
    win_rate  = len(wins) / len(closed_trades) * 100 if closed_trades else 0
    avg_win   = sum(t.get("realized_pnl", 0) for t in wins) / len(wins) if wins else 0
    avg_loss  = sum(t.get("realized_pnl", 0) for t in losses) / len(losses) if losses else 0

    # Recent trades (last 20)
    recent = sorted(closed_trades, key=lambda x: x.get("exit_date", ""), reverse=True)[:20]

    return {
        "summary": {
            "total_trades": len(all_trades),
            "open_trades": len(open_trades),
            "pending_trades": len(pending_trades),
            "closed_trades": len(closed_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
        },
        "recent_closed": recent,
        "open_trades": open_trades,
    }


@router.get("/weekly-report")
def weekly_report():
    reporter = get_paper_reporter()
    storage  = get_paper_storage()
    trades   = storage.get_trades() or []
    try:
        report = reporter.generate_weekly_report(trades)
        return {"report": report}
    except Exception as e:
        return {"report": f"Rapor üretilemedi: {e}"}
