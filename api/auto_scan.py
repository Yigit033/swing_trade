"""
Zamanlanmış (kullanıcı etkileşimi gerektirmeyen) günlük SmallCap taraması.

Manuel /api/scanner/smallcap/start ile AYNI motoru, AYNI eşik anlamını
kullanır — ayrı bir "otomatik mod" mantığı değil, tetikleyicisi saat olan
bir manuel tarama. Kapanış sonrası çalışır (varsayılan 16:30 ET) çünkü:
  - fetcher._drop_incomplete_last_bar zaten dünün TAMAMLANMIŞ barına göre
    karar veriyor (saatten bağımsız karar mantığı — bkz. market_calendar).
  - Universe artık Q5/Q5b'siz (2026-07-22 recall ölçümü) tamamen dünün
    kapanışına göre hesaplanan Finviz preset'lerine dayanıyor.
  - 16:30 ET, Finviz'in günlük kolonlarının (Change/Volume/20D-High) o
    günün kapanışını yansıtacak kadar sindiği an — hem manuel "kapanış
    sonrası tara" tavsiyesiyle hem edge ölçümünün varsaydığı pencereyle
    aynı zaman dilimi.

min_quality BİLEREK Scanner UI'daki "Auto-Track" slider'ından ayrı, ayarlar
dosyasındaki (auto_scan.min_quality) sabit bir eşik kullanır — gece kimse
izlemezken hangi eşiğin geçerli olduğu, o an ekranda ne görünüyor olduğuna
(unutulmuş bir slider konumuna) bağlı olmamalı.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Optional

from swing_trader.utils.market_calendar import _NYSE_TZ, is_trading_day

logger = logging.getLogger(__name__)

# Aynı gün ikinci kez tetiklenmeyi önler (döngü kayarsa / process restart
# olursa çift tarama riski). Bilerek process-local: tek makine, tek worker.
_last_run_date: Optional[date] = None


def _next_target_et(target_hour: int, target_minute: int, now: Optional[datetime] = None) -> datetime:
    """Bugün hedef saat geçtiyse yarının, geçmediyse bugünün hedef anı (ET)."""
    now_et = (now or datetime.now(tz=_NYSE_TZ)).astimezone(_NYSE_TZ)
    target_today = now_et.replace(
        hour=target_hour, minute=target_minute, second=0, microsecond=0
    )
    return target_today if now_et < target_today else target_today + timedelta(days=1)


def _run_auto_scan_once() -> None:
    """Senkron: mevcut manuel tarama koduyla AYNI fonksiyonu çağırır."""
    from swing_trader.small_cap.settings_config import load_settings
    from api.routers.scanner import ScanRequest, _execute_smallcap_scan
    from api.scanner_jobs import create_exclusive_scan_job, release_scan_slot
    from api.deps import get_paper_tracker

    # Manuel tarama slotunu paylaş — biri sürerken diğeri ASLA aynı anda
    # koşmasın (Finviz'e çift yük + kaynak çakışması). Slot doluysa bu
    # tetikleme sessizce atlanır; bir sonraki gün tekrar denenir.
    job_id = create_exclusive_scan_job()
    if not job_id:
        logger.info("Auto-scan skipped: a manual scan is already running")
        return

    try:
        us = load_settings().auto_scan
        body = ScanRequest(
            portfolio_value=us.portfolio_value,
            min_quality=us.min_quality,
            top_n=us.top_n,
        )

        logger.info(
            "Auto-scan starting (min_quality=%s top_n=%s)", us.min_quality, us.top_n
        )
        result = _execute_smallcap_scan(body, on_progress=None, job_id=job_id, user_id=None)
    finally:
        release_scan_slot(job_id)

    signals = result.get("signals", [])
    logger.info("Auto-scan complete: %d signal(s) at/above min_quality=%s", len(signals), us.min_quality)

    if not signals:
        return

    tracker = get_paper_tracker()
    added, skipped = [], []
    for signal in signals:
        try:
            trade_id = tracker.add_trade_from_signal(signal, None)
        except Exception:
            logger.exception("Auto-scan: failed to track %s", signal.get("ticker"))
            skipped.append(signal.get("ticker"))
            continue
        (added if trade_id > 0 else skipped).append(signal.get("ticker"))

    logger.info(
        "Auto-scan auto-track: %d added (PENDING), %d skipped (duplicate/cooldown/window): %s / %s",
        len(added), len(skipped), added, skipped,
    )


async def auto_scan_loop() -> None:
    """
    Günde bir kez, hedef ET saatinde tetikler. Hafta sonu/tatilde atlar.

    _execute_smallcap_scan senkron ve dakikalar sürebilir (yfinance batch
    fetch) — event loop'u kilitlemesin diye ayrı thread'de (asyncio.to_thread)
    çalıştırılır; tıpkı manuel taramanın kendi thread'inde koşması gibi.
    """
    global _last_run_date
    from swing_trader.small_cap.settings_config import load_settings

    while True:
        try:
            us = load_settings().auto_scan
            if not us.enabled:
                await asyncio.sleep(300)  # kapalıyken 5dk'da bir ayar değişmiş mi diye bak
                continue

            target = _next_target_et(us.target_hour_et, us.target_minute_et)
            wait_s = (target - datetime.now(tz=_NYSE_TZ)).total_seconds()
            if wait_s > 3600:
                # Hedefe uzun süre var — 1 saatlik dilimlerle bekle ki
                # arada auto_scan.enabled kapatılırsa/saat ayarı değişirse
                # döngü en geç 1 saat içinde yeni değeri görsün.
                await asyncio.sleep(3600)
                continue

            logger.info("Auto-scan scheduled for %s ET (in %.0f min)", target.strftime("%Y-%m-%d %H:%M"), max(wait_s, 0) / 60)
            await asyncio.sleep(max(wait_s, 0))

            today = datetime.now(tz=_NYSE_TZ).date()
            if _last_run_date == today:
                await asyncio.sleep(300)
                continue
            if not is_trading_day(today):
                logger.info("Auto-scan skipped: %s is not an NYSE trading day", today)
                _last_run_date = today
                continue

            _last_run_date = today
            await asyncio.to_thread(_run_auto_scan_once)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Auto-scan loop iteration failed")
            await asyncio.sleep(300)
