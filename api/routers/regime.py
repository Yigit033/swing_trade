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

_small_cap_cache: dict = {}          # {seans: rejim} — IWM kapanışı seans içinde değişmez


def entry_status(strategy: str, small_cap_regime: Optional[str], market_regime: Optional[str],
                 loop_policy) -> dict:
    """Otomatik alım şu an açık mı? Canlı auto-scan ile AYNI kurallar (saf fonksiyon).

    H6S: küçük hisse (IWM) rejimi — BEAR giriş yok, CAUTION en çok N açık pozisyon
    (``api.auto_scan._auto_track_h6s``). Eski kurallar: tarama (SPY) rejimi +
    ``auto_entry_capacity``. status: BLOCKED | LIMITED | OPEN | UNKNOWN (rejim okunamadı).
    """
    from swing_trader.shadow.rules import PARAMS_H6S
    from swing_trader.small_cap.thresholds import auto_entry_capacity

    strategy = str(strategy or "LEGACY").upper()
    if strategy == "H6S":
        if not small_cap_regime:
            return dict(strategy="H6S", index="IWM", regime=None, status="UNKNOWN")
        regime = str(small_cap_regime).upper()
        out = dict(strategy="H6S", index="IWM", regime=regime, status="OPEN")
        if regime == "BEAR" and PARAMS_H6S["bear_block"]:
            out["status"] = "BLOCKED"
        elif regime == "CAUTION":
            out.update(status="LIMITED", max_open=int(PARAMS_H6S["caution_max_concurrent"]))
        return out
    regime = str(market_regime or "UNKNOWN").upper()
    out = dict(strategy="LEGACY", index="SPY", regime=regime, status="OPEN")
    cap = auto_entry_capacity(regime, loop_policy, 0)
    if cap == 0 and regime == "BEAR":
        out["status"] = "BLOCKED"
    elif cap is not None:
        out.update(status="LIMITED", max_open=int(loop_policy.caution_max_concurrent))
    return out


def _small_cap_regime_now() -> tuple:
    """(seans, rejim | None) — auto-scan'in kullanacağı küçük hisse rejimi; veri yoksa None."""
    from swing_trader.shadow import sources as src
    from swing_trader.shadow.runner import _small_cap_regime
    from swing_trader.utils.market_calendar import last_completed_session

    asof = last_completed_session()
    if asof not in _small_cap_cache:
        reg = _small_cap_regime(src, asof)
        if reg is None:
            return asof, None
        _small_cap_cache.clear()
        _small_cap_cache[asof] = str(reg.get("regime", "UNKNOWN")).upper()
    return asof, _small_cap_cache[asof]


def _entries(market_regime: Optional[str]) -> Optional[dict]:
    """Ana sayfadaki "alım açık mı" satırı. Hata rejim yanıtını bozmasın → None."""
    try:
        from swing_trader.small_cap.settings_config import load_settings

        settings = load_settings()
        strategy = str(settings.auto_scan.strategy or "LEGACY").upper()
        asof, small = _small_cap_regime_now() if strategy == "H6S" else (None, None)
        out = entry_status(strategy, small, market_regime, settings.backtest_loop)
        out["auto_scan_enabled"] = bool(settings.auto_scan.enabled)
        if asof:
            out["asof"] = asof.isoformat()
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
