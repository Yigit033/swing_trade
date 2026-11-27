"""
Market Regime API — current regime + history.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from api.deps import get_regime_storage, get_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/regime", tags=["regime"])


def entry_status(strategy: str, market_regime: Optional[str], loop_policy) -> dict:
    """Otomatik alım şu an açık mı? Canlı auto-scan ile AYNI kurallar (saf fonksiyon).

    GRW (ana kural): yalnız SPY rejimi BULL iken alım (``api.auto_scan._auto_track_grw``); BEAR, CAUTION ya da
    UNKNOWN → BLOCKED. Eski kurallar: tarama (SPY) rejimi + ``auto_entry_capacity``.
    status: BLOCKED | LIMITED | OPEN.
    """
    from swing_trader.shadow.rules import PARAMS_GRW
    from swing_trader.small_cap.thresholds import auto_entry_capacity

    strategy = str(strategy or "LEGACY").upper()
    regime = str(market_regime or "UNKNOWN").upper()
    if strategy == "GRW":
        ok = regime == PARAMS_GRW["buy_regime"]
        return dict(strategy="GRW", index="SPY", regime=regime, status="OPEN" if ok else "BLOCKED")
    out = dict(strategy="LEGACY", index="SPY", regime=regime, status="OPEN")
    cap = auto_entry_capacity(regime, loop_policy, 0)
    if cap == 0 and regime == "BEAR":
        out["status"] = "BLOCKED"
    elif cap is not None:
        out.update(status="LIMITED", max_open=int(loop_policy.caution_max_concurrent))
    return out


def _entries(market_regime: Optional[str]) -> Optional[dict]:
    """Ana sayfadaki "alım açık mı" satırı. Hata rejim yanıtını bozmasın → None."""
    try:
        from swing_trader.small_cap.settings_config import load_settings

        settings = load_settings()
        out = entry_status(settings.auto_scan.strategy, market_regime, settings.backtest_loop)
        out["auto_scan_enabled"] = bool(settings.auto_scan.enabled)
        return out
    except Exception:
        logger.exception("Entry status unavailable")
        return None


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

        result = SmallCapSignals(get_config()).detect_market_regime()
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
                "entries": _entries(latest["regime"]),
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
        "entries": _entries(result["regime"]),
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
