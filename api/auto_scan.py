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
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from swing_trader.utils.market_calendar import (
    _NYSE_TZ,
    last_completed_session,
    next_trading_day,
)

logger = logging.getLogger(__name__)

# Aynı seans ikinci kez tetiklenmeyi önler (döngü kayarsa / process restart
# olursa çift tarama riski). Bilerek process-local: tek makine, tek worker.
# Asıl tekillik DB'de (scan_source=auto + bar_session) — RAM Fly suspend'de
# sıfırlanır.
_last_run_session: Optional[date] = None

_AUTO_SOURCE = "auto"


def _next_target_et(target_hour: int, target_minute: int, now: Optional[datetime] = None) -> datetime:
    """Bugün hedef saat geçtiyse yarının, geçmediyse bugünün hedef anı (ET)."""
    now_et = (now or datetime.now(tz=_NYSE_TZ)).astimezone(_NYSE_TZ)
    target_today = now_et.replace(
        hour=target_hour, minute=target_minute, second=0, microsecond=0
    )
    return target_today if now_et < target_today else target_today + timedelta(days=1)


def _parse_created_at_utc(value: str) -> Optional[datetime]:
    """created_at is stored as UTC ISO (…Z). Naive values treated as UTC."""
    if not value:
        return None
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def run_covers_auto_session(
    stats: dict,
    created_at: str,
    session: date,
) -> bool:
    """
    True if this saved run is the cron/auto scan for `session`.

    Manuel taramalar (scan_source yok) sayılmaz — kapanış sonrası elle bakmak
    auto-track geçişini iptal etmesin. GitHub 21:30 + 22:30 + Cumartesi
    yakalama aynı auto taramayı üç kez koşturmasın diye yalnız auto kaydı bakılır.
    """
    if not isinstance(stats, dict):
        return False
    if stats.get("scan_source") != _AUTO_SOURCE:
        return False
    reason = stats.get("reason")
    if reason in ("stale_data", "data_quality", "rate_limited", "error"):
        return False
    bar = stats.get("bar_session")
    if bar:
        return str(bar)[:10] == session.isoformat()
    dt = _parse_created_at_utc(created_at)
    if dt is None:
        return False
    dt_et = dt.astimezone(_NYSE_TZ)
    close = datetime(session.year, session.month, session.day, 16, 0, tzinfo=_NYSE_TZ)
    nxt = next_trading_day(session)
    next_close = datetime(nxt.year, nxt.month, nxt.day, 16, 0, tzinfo=_NYSE_TZ)
    return close <= dt_et < next_close


def session_already_auto_scanned(session: date, lookback: int = 24) -> bool:
    """DB'de bu seans için başarılı bir auto tarama var mı?"""
    try:
        from api.deps import get_signal_history_storage

        rows = get_signal_history_storage().list_recent_stats(limit=lookback)
    except Exception:
        logger.exception("Could not read scan history for session skip")
        return False
    for row in rows:
        if run_covers_auto_session(row.get("stats") or {}, row.get("created_at") or "", session):
            return True
    return False


def run_daily_maintenance(force: bool = False) -> dict:
    """
    Günlük bakım — dışarıdan (cron) tetiklenir, in-process döngüye bağımlı değil.

    NEDEN DIŞARIDAN: fly.io makinesi boşta askıya alınıyor (min_machines_running=0)
    ve askıdayken hiçbir asyncio görevi çalışmıyor. Makine yalnız gelen HTTP
    isteğiyle uyanıyor. 7/24 makine (~$6/ay) yerine GitHub Actions cron bu
    endpoint'i çağırıyor: makine uyanır, iş biter, tekrar uyur — maliyet $0.

    NEDEN GÜNDE BİR YETERLİ: strateji tamamen GÜNLÜK BAR üzerinde karar veriyor
    (stop/T1/trailing/timeout hepsi günlük bar; VCE tetiği tamamlanmış bara
    bakıyor). Kapanış sonrası tek koşu yeni bilginin tamamını görür — 5 dakikada
    bir kontrol etmek ek bilgi üretmez.

    Sıra önemli: önce pending onayı (dünkü sinyaller OPEN'a döner), sonra çıkış
    kontrolü (yeni açılanlar da dahil), en son tarama (yarının adayları).

    Tarama takvimi TAKVİM GÜNÜ değil son tamamlanmış NYSE seansıdır. GitHub
    cron Cuma 21:30 UTC'yi Cumartesi sabaha kaydırsa veya Cumartesi yakalama
    koşsa, hâlâ Cuma kapanışını tarar — `is_trading_day(today)` Cumartesi
    `not_a_trading_day` deyip Cuma'yı sonsuza kaçırıyordu.
    """
    from api.deps import get_paper_tracker
    from swing_trader.small_cap.settings_config import load_settings

    global _last_run_session
    result: dict[str, Any] = {"pending_confirmed": 0, "trades_closed": 0, "scan": "skipped"}
    now_et = datetime.now(tz=_NYSE_TZ)
    today = now_et.date()
    session = last_completed_session(now_et)
    result["date_et"] = str(today)
    result["bar_session"] = session.isoformat()

    tracker = get_paper_tracker()

    # 1) PENDING → OPEN (t+1 açılış fiyatıyla; geç koşsa da tarihsel açılışı kullanır)
    try:
        processed = tracker.confirm_pending_trades(None)
        result["pending_confirmed"] = len(processed or [])
    except Exception:
        logger.exception("Daily maintenance: pending confirm failed")
        result["pending_error"] = True

    # 2) Açık pozisyonların çıkış kontrolü (stop/T1/trailing/timeout)
    try:
        updated = tracker.update_all_open_trades(None) or []
        closed = [t for t in updated if t.get("status") not in ("OPEN", "PENDING")]
        result["trades_closed"] = len(closed)
        result["closed_tickers"] = [t.get("ticker") for t in closed]
    except Exception:
        logger.exception("Daily maintenance: exit check failed")
        result["exit_error"] = True

    # 3) Günlük tarama — son tamamlanmış seans, işlem günü takvimi değil
    try:
        us = load_settings().auto_scan
        if not us.enabled:
            result["scan"] = "disabled"
        elif _last_run_session == session and not force:
            result["scan"] = "already_ran_session"
        elif session_already_auto_scanned(session) and not force:
            result["scan"] = "already_ran_session"
        else:
            status = _run_auto_scan_once()
            result["scan"] = status
            if status == "ran":
                _last_run_session = session
    except Exception:
        logger.exception("Daily maintenance: auto-scan failed")
        result["scan"] = "error"

    logger.info("Daily maintenance done: %s", result)
    return result


def _run_auto_scan_once() -> str:
    """Senkron: mevcut manuel tarama koduyla AYNI fonksiyonu çağırır. 'ran' | 'busy'."""
    from swing_trader.small_cap.settings_config import load_settings
    from api.routers.scanner import ScanRequest, _execute_smallcap_scan
    from api.scanner_jobs import create_exclusive_scan_job, release_scan_slot
    from api.deps import get_paper_tracker

    # Manuel tarama slotunu paylaş — biri sürerken diğeri ASLA aynı anda
    # koşmasın (Finviz'e çift yük + kaynak çakışması). Slot doluysa bu
    # tetikleme sessizce atlanır; bir sonraki cron tekrar dener.
    job_id = create_exclusive_scan_job()
    if not job_id:
        logger.info("Auto-scan skipped: a manual scan is already running")
        return "busy"

    try:
        us = load_settings().auto_scan
        body = ScanRequest(
            portfolio_value=us.portfolio_value,
            min_quality=us.min_quality,
            top_n=us.top_n,
            scan_source=_AUTO_SOURCE,
        )

        logger.info(
            "Auto-scan starting (min_quality=%s top_n=%s session=%s)",
            us.min_quality,
            us.top_n,
            last_completed_session().isoformat(),
        )
        result = _execute_smallcap_scan(body, on_progress=None, job_id=job_id, user_id=None)
    finally:
        release_scan_slot(job_id)

    signals = result.get("signals", [])
    logger.info("Auto-scan complete: %d signal(s) at/above min_quality=%s", len(signals), us.min_quality)

    if not signals:
        return "ran"

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
    return "ran"


async def auto_scan_loop() -> None:
    """
    Günde bir kez, hedef ET saatinde tetikler.

    Fly'da makine askıdayken bu döngü ÖLÜDÜR — asıl tetikleyici GitHub cron.
    Makine uyanık kalırsa (debug) hafta sonu da son tamamlanmış seansı tarar;
    DB skip aynı seansı iki kez koşturmaz.
    """
    global _last_run_session
    from swing_trader.small_cap.settings_config import load_settings

    while True:
        try:
            us = (await asyncio.to_thread(load_settings)).auto_scan
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

            session = last_completed_session()
            if _last_run_session == session:
                await asyncio.sleep(300)
                continue
            if await asyncio.to_thread(session_already_auto_scanned, session):
                _last_run_session = session
                await asyncio.sleep(300)
                continue

            status = await asyncio.to_thread(_run_auto_scan_once)
            if status == "ran":
                _last_run_session = session

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Auto-scan loop iteration failed")
            await asyncio.sleep(300)
