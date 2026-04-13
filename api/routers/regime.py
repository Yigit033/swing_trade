"""
Market Regime API — current regime + history.
"""

import logging
from datetime import datetime

from fastapi import APIRouter
from api.deps import get_regime_storage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/regime", tags=["regime"])


@router.get("/current")
def get_current_regime():
    """
    Her istekte canlı SPY/VIX ile rejim hesaplanır (scanner ile aynı kaynak).

    Eski davranış: DB'deki son satır dönülüyordu; save_regime aynı rejimde insert
    atladığı için SPY/VIX haftalarca güncellenmiyordu — tarama logları ile header çelişiyordu.
    """
    storage = get_regime_storage()
    try:
        from swing_trader.small_cap.signals import SmallCapSignals

        result = SmallCapSignals().detect_market_regime()
    except Exception as e:
        logger.error("Live regime detection failed: %s", e)
        latest = storage.get_latest()
        if latest:
            out = {
                "regime": latest["regime"],
                "confidence": latest["confidence"],
                "spy_price": latest.get("spy_price"),
                "ma50": latest.get("ma50"),
                "ma200": latest.get("ma200"),
                "vix": latest.get("vix"),
                "spy_5d_return": latest.get("spy_5d_return"),
                "detected_at": latest.get("detected_at"),
                "stale_fallback": True,
                "fallback_reason": str(e)[:500],
            }
            de = latest.get("detect_error")
            if de:
                out["detect_error"] = de
            return out
        return {
            "regime": "UNKNOWN",
            "confidence": "TENTATIVE",
            "detected_at": None,
            "detect_error": str(e)[:500],
        }

    storage.save_regime(result)

    sampled_at = datetime.now().replace(microsecond=0).isoformat()
    out = {
        "regime": result["regime"],
        "confidence": result["confidence"],
        "spy_price": result.get("spy_price"),
        "ma50": result.get("ma50"),
        "ma200": result.get("ma200"),
        "vix": result.get("vix"),
        "spy_5d_return": result.get("spy_5d_return"),
        "detected_at": sampled_at,
        "live": True,
    }
    if result.get("detect_error"):
        out["detect_error"] = result["detect_error"]
    return out


@router.get("/history")
def get_regime_history(limit: int = 30):
    """Get regime change history."""
    storage = get_regime_storage()
    history = storage.get_history(limit=limit)
    return {"history": history, "count": len(history)}
