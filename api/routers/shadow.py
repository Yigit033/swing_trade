"""
Gölge sinyaller (H4 içeriden alım, H5 bilanço sürprizi, H6/H6S/H6SR VCE, MOM lider rotasyonu,
GRW büyüme + kırılım) — ileri test, GERÇEK PARA YOK.

GET  /api/shadow            → son rapor (terfi kapısı durumu) + işlem listesi + koşu sağlığı
POST /api/shadow/daily-run  → günlük gölge koşusu (cron; X-Cron-Secret).
                              ?if_needed=true: en yeni seans için tam/temiz koşu varsa atlar
                              (yedek tetikleyiciler — GitHub cron'u + Daily Run zinciri).

Ana ürünün paper işlemlerinden, kapasitesinden ve raporlarından tamamen ayrıdır
(shadow_* tabloları). Kurallar: lab/PREREGISTRATION.md Ek A + Ek C (H6: Ek F, H6S: Ek J, H6SR: Ek Q, MOM: Ek U, GRW: Ek W).
"""
import hmac
import logging
import os

from fastapi import APIRouter, HTTPException, Query, Request

from api.utils import sanitize_for_json

router = APIRouter(prefix="/api/shadow", tags=["shadow"])
logger = logging.getLogger(__name__)

_TRADE_FIELDS = ("id", "strategy", "ticker", "event_date", "signal_date", "decided_at", "key", "status",
                 "reason", "entry_date", "entry_price", "stop", "exit_date", "exit_price", "exit_reason",
                 "bars", "ret_pct", "net_pct", "iwm_ret", "mark_price", "mark_date", "late_decision", "warmup")


def _reading(t):
    """Ek Y okuma notu (yalnız GRW): karar, gerekçeler, kırmızı bayraklar, geç mi, model."""
    d = t.get("details")
    r = (d.get("reading") if isinstance(d, dict) else None) or None
    if not r:
        return None
    keys = ("status", "verdict", "reasons", "red_flags", "late", "read_at", "provider", "model", "error")
    return {k: r.get(k) for k in keys if r.get(k) is not None}


@router.get("")
def get_shadow(limit: int = Query(300, ge=1, le=2000)):
    from swing_trader.shadow.ledger import ShadowLedger
    from swing_trader.shadow.runner import run_health

    ledger = ShadowLedger()
    trades = ledger.trades()
    trades.sort(key=lambda t: (t.get("signal_date") or "", t.get("id") or 0), reverse=True)
    try:
        health = run_health(ledger)
    except Exception:                       # sağlık özeti raporu asla düşürmesin
        logger.exception("Shadow health failed")
        health = None
    return sanitize_for_json({
        "report": ledger.get_meta("report"),
        "health": health,
        "last_run": ledger.get_meta("last_run"),
        "live_start": ledger.get_meta("live_start"),
        "h4_observation_start": ledger.get_meta("h4_observation_start"),
        "trades": [dict({k: t.get(k) for k in _TRADE_FIELDS}, reading=_reading(t)) for t in trades[:limit]],
        "total_trades": len(trades),
    })


@router.post("/daily-run")
def shadow_daily_run(request: Request, if_needed: bool = Query(False)):
    expected = (os.environ.get("CRON_SECRET") or "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="CRON_SECRET tanımlı değil — gölge koşusu devre dışı.")
    provided = (request.headers.get("X-Cron-Secret") or "").strip()
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Geçersiz cron secret")
    from swing_trader.shadow.runner import run_shadow_daily

    return sanitize_for_json(run_shadow_daily(if_needed=if_needed))
