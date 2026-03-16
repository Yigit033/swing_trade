"""
Market Regime API — current regime + history.
"""

import logging
from fastapi import APIRouter
from api.deps import get_regime_storage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/regime", tags=["regime"])


@router.get("/current")
def get_current_regime():
    """Get the latest regime from DB. If none, detect live."""
    storage = get_regime_storage()
    latest = storage.get_latest()

    if latest:
        return {
            "regime": latest["regime"],
            "confidence": latest["confidence"],
            "score_multiplier": latest["score_multiplier"],
            "spy_price": latest.get("spy_price"),
            "ma50": latest.get("ma50"),
            "ma200": latest.get("ma200"),
            "vix": latest.get("vix"),
            "spy_5d_return": latest.get("spy_5d_return"),
            "detected_at": latest.get("detected_at"),
        }

    # No history yet — detect live
    try:
        from swing_trader.small_cap.signals import SmallCapSignals
        signals = SmallCapSignals()
        result = signals.detect_market_regime()
        storage.save_regime(result)
        return {
            "regime": result["regime"],
            "confidence": result["confidence"],
            "score_multiplier": result["score_multiplier"],
            "spy_price": result.get("spy_price"),
            "ma50": result.get("ma50"),
            "ma200": result.get("ma200"),
            "vix": result.get("vix"),
            "spy_5d_return": result.get("spy_5d_return"),
            "detected_at": "just_detected",
        }
    except Exception as e:
        logger.error(f"Live regime detection failed: {e}")
        return {
            "regime": "UNKNOWN",
            "confidence": "TENTATIVE",
            "score_multiplier": 1.0,
            "detected_at": None,
        }


@router.get("/history")
def get_regime_history(limit: int = 30):
    """Get regime change history."""
    storage = get_regime_storage()
    history = storage.get_history(limit=limit)
    return {"history": history, "count": len(history)}
