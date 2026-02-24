"""
GenAI router.
POST /api/genai/chat          - strategy chat (RAG-lite)
POST /api/genai/signal-brief  - signal briefing for a ticker
GET  /api/genai/weekly-report - weekly performance report (text)
"""

import logging
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from api.deps import get_paper_storage

router = APIRouter()
logger = logging.getLogger(__name__)

# Closed statuses — matches storage.get_closed_trades() logic
CLOSED_STATUSES = {"STOPPED", "TRAILED", "TARGET", "MANUAL", "WIN", "LOSS", "CLOSED", "REJECTED"}


class ChatRequest(BaseModel):
    message: str
    history: Optional[list] = []


class SignalBriefRequest(BaseModel):
    ticker: str
    signal: dict


@router.post("/chat")
async def strategy_chat(body: ChatRequest):
    """Free-form strategy chat using trade history as context."""
    storage = get_paper_storage()
    trades = storage.get_all_trades() or []   # FIX: was storage.get_trades() (non-existent)

    try:
        from swing_trader.genai.strategy_chat import StrategyChat
        chat = StrategyChat()
        answer = chat.answer(body.message, trades, history=body.history)
        return {"answer": answer}
    except Exception as e:
        logger.warning(f"GenAI chat error: {e}")
        # Deterministic fallback using correct closed-trades logic
        closed = storage.get_closed_trades(limit=1000)
        wins   = [t for t in closed if (t.get("realized_pnl") or 0) > 0]
        wr     = round(len(wins) / len(closed) * 100, 1) if closed else 0
        return {
            "answer": (
                f"GenAI şu an erişilemiyor. Mevcut istatistikler: "
                f"{len(closed)} kapalı işlem, %{wr} kazanma oranı."
            )
        }


@router.post("/signal-brief")
async def signal_brief(body: SignalBriefRequest):
    """Generate AI commentary for a specific signal."""
    try:
        from swing_trader.genai.signal_briefer import SignalBriefer
        briefer = SignalBriefer()
        brief = briefer.brief(body.ticker, body.signal)
        return {"brief": brief}
    except Exception as e:
        logger.warning(f"Signal brief error: {e}")
        sig = body.signal
        return {
            "brief": (
                f"{body.ticker}: Giriş ${sig.get('entry_price', 0):.2f}, "
                f"stop ${sig.get('stop_loss', 0):.2f}, "
                f"hedef ${sig.get('target', sig.get('target_1', 0)):.2f}. "
                f"Kalite skoru: {sig.get('quality_score', 0):.0f}/100."
            )
        }


@router.get("/weekly-report")
async def weekly_report():
    """Generate weekly performance report."""
    storage = get_paper_storage()
    trades  = storage.get_all_trades() or []   # FIX: was storage.get_trades() (non-existent)
    try:
        from swing_trader.paper_trading.reporter import PaperTradeReporter
        reporter = PaperTradeReporter()
        report = reporter.generate_weekly_report(trades)
        return {"report": report}
    except Exception as e:
        logger.error(f"Weekly report error: {e}")
        return {"report": f"Rapor üretilemedi: {e}"}
