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

    try:
        from swing_trader.genai.strategy_chat import StrategyChat
        chat = StrategyChat(storage=storage)

        # If not ready, force-reload .env and retry setup (uvicorn reload edge case)
        if not chat.client.is_ready():
            import os
            from pathlib import Path
            try:
                from dotenv import load_dotenv
                load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
                logger.info(f"Forced .env reload: GEMINI_API_KEY={'SET' if os.getenv('GEMINI_API_KEY') else 'MISSING'}")
            except ImportError:
                pass
            # Recreate with fresh env
            chat = StrategyChat(storage=storage)

        # Log LLM readiness for debugging
        logger.info(
            f"StrategyChat LLM status: provider={chat.client.provider}, "
            f"available={chat.client.available}, ready={chat.client.is_ready()}"
        )

        result = chat.ask(body.message)
        return {
            "answer": result.get("answer", "Cevap alınamadı"),
            "llm_available": result.get("llm_available", False),
            "success": result.get("success", False),
        }
    except Exception as e:
        logger.warning(f"GenAI chat error: {e}", exc_info=True)
        # Deterministic fallback using correct closed-trades logic
        closed = storage.get_closed_trades(limit=1000)
        wins   = [t for t in closed if (t.get("realized_pnl") or 0) > 0]
        wr     = round(len(wins) / len(closed) * 100, 1) if closed else 0
        return {
            "answer": (
                f"GenAI şu an erişilemiyor (hata: {e}). "
                f"Mevcut istatistikler: {len(closed)} kapalı işlem, %{wr} kazanma oranı."
            ),
            "llm_available": False,
            "success": False,
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
    try:
        from swing_trader.paper_trading.reporter import PaperTradeReporter
        reporter = PaperTradeReporter(storage)
        trades = storage.get_all_trades() or []
        report = reporter.generate_weekly_report(trades)
        return {"report": report}
    except Exception as e:
        logger.error(f"Weekly report error: {e}")
        return {"report": f"Rapor üretilemedi: {e}"}


@router.get("/weekly-report-ai")
async def weekly_report_ai():
    """AI-powered weekly performance report using WeeklyReporter."""
    storage = get_paper_storage()
    try:
        from swing_trader.genai.reporter import WeeklyReporter
        reporter = WeeklyReporter(storage, days=7)
        result = reporter.generate(force_refresh=True)
        return result
    except Exception as e:
        logger.error(f"AI Weekly report error: {e}")
        return {"success": False, "error": str(e), "report": None}


@router.get("/model-status")
async def model_status():
    """Get ML model status and metrics."""
    import json
    from pathlib import Path
    storage = get_paper_storage()

    META_PATH = Path("data/ml_models/signal_predictor_meta.json")
    MODEL_PATH = Path("data/ml_models/signal_predictor.pkl")

    # Count trades for training data info
    closed = storage.get_closed_trades(limit=9999)
    demo_count = len([t for t in closed if "[DEMO]" in (t.get("notes") or "")])
    real_trades = [t for t in closed
                   if "[DEMO]" not in (t.get("notes") or "")
                   and t.get("status") not in ("REJECTED", "PENDING")]
    real_count = len(real_trades)

    if MODEL_PATH.exists() and META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)
        return {
            "trained": True,
            "meta": meta,
            "real_count": real_count,
            "demo_count": demo_count,
        }
    return {
        "trained": False,
        "meta": None,
        "real_count": real_count,
        "demo_count": demo_count,
    }


@router.post("/train")
async def train_model():
    """Train/update the ML signal quality predictor model."""
    try:
        from swing_trader.ml.trainer import SignalTrainer
        trainer = SignalTrainer()
        result = trainer.run()
        return result
    except Exception as e:
        logger.error(f"Model train error: {e}")
        return {"success": False, "error": str(e)}


class PredictRequest(BaseModel):
    entry_price: float
    stop_loss: float
    target: float
    atr: float = 0.5
    quality_score: float = 7
    swing_type: str = "A"
    max_hold_days: int = 7


@router.post("/predict")
async def predict_signal(body: PredictRequest):
    """Predict signal quality using the trained ML model."""
    try:
        from swing_trader.ml.predictor import SignalPredictor
        from datetime import datetime

        predictor = SignalPredictor()
        if not predictor.is_ready:
            return {"success": False, "error": "Model henüz eğitilmedi. Önce modeli eğitin."}

        test_signal = {
            "entry_price": body.entry_price,
            "stop_loss": body.stop_loss,
            "target": body.target,
            "atr": body.atr,
            "quality_score": body.quality_score,
            "swing_type": body.swing_type,
            "max_hold_days": body.max_hold_days,
            "entry_date": datetime.now().strftime("%Y-%m-%d"),
        }
        pred = predictor.predict(test_signal)
        if pred:
            return {"success": True, **pred}
        return {"success": False, "error": "Tahmin başarısız"}
    except Exception as e:
        logger.error(f"Predict error: {e}")
        return {"success": False, "error": str(e)}
