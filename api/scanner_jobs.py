"""
SmallCap scan background jobs — in-memory store + worker thread.
Client disconnect does not cancel the scan.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}

# Tek eşzamanlı scan — endpoint bu ID'yi thread başlamadan önce rezerve eder
_scan_slot_lock = threading.Lock()
_active_scan_job_id: Optional[str] = None

# ── ASILI KALAN İŞ BEKÇİSİ (2026-08-05) ──────────────────────────────────
# Canlı arıza: bir tarama %84'te ("Running momentum engine…") saatlerce takıldı.
# İstisna DEĞİLDİ — worker try/except/finally ile sarılı, istisna olsa "failed"
# olurdu. Thread bir ağ çağrısında (yfinance/Finviz) süresiz asılı kalmıştı ve
# hiçbir katmanda timeout yoktu. Sonuç ürünü kullanılamaz hale getiriyor:
# `create_exclusive_scan_job` tek slotu koruduğu için kullanıcı MAKİNE YENİDEN
# BAŞLAYANA KADAR bir daha tarama yapamıyor. Telefondaki "arka planda çalışıyor"
# kartı da bu yüzden hiç kaybolmuyordu.
#
# Kök nedeni tek tek avlamak yerine (asılma her ağ çağrısında olabilir) genel
# bir bekçi: iş İLERLEME KAYDETMEDEN bu süreyi aşarsa başarısız sayılır ve slot
# serbest bırakılır. Normal tarama saniyeler arayla ilerleme yazar, dolayısıyla
# meşru bir işi kesme riski yok.
STALE_JOB_SECONDS = 600     # 10 dk ilerleme yoksa asılı kabul et


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _monotonic() -> float:
    import time

    return time.monotonic()


def _reclaim_stale_scan_slot() -> Optional[str]:
    """
    Aktif iş ilerleme kaydetmeden STALE_JOB_SECONDS'ı aştıysa onu 'failed'
    işaretle ve slotu bırak. Reclaim edilen job_id'yi döndürür.

    Not: `_scan_slot_lock` TUTULMADAN çağrılmalı — kendi kilidini alır.
    """
    global _active_scan_job_id
    with _scan_slot_lock:
        jid = _active_scan_job_id
        if jid is None:
            return None
        with _lock:
            j = _jobs.get(jid)
            if not j or j.get("status") not in ("queued", "running"):
                _active_scan_job_id = None      # tutarsız durum — slotu bırak
                return jid
            last = j.get("last_progress_at")
            if last is None or (_monotonic() - last) < STALE_JOB_SECONDS:
                return None
            idle = int(_monotonic() - last)
            j.update(
                status="failed",
                phase="error",
                message=(
                    f"Tarama {idle // 60} dakikadır ilerlemedi — asılı kabul edilip "
                    "iptal edildi (büyük olasılıkla veri sağlayıcı yanıt vermedi). "
                    "Yeniden tarama başlatabilirsiniz."
                ),
                error="stalled",
            )
        _active_scan_job_id = None
        logger.warning("Asılı tarama işi geri alındı: %s (%d sn ilerleme yok)", jid, idle)
        return jid


def _prune_old_jobs() -> None:
    if len(_jobs) <= 40:
        return
    completed = [
        (jid, j.get("created_at", ""))
        for jid, j in _jobs.items()
        if j.get("status") in ("completed", "failed")
    ]
    completed.sort(key=lambda x: x[1])
    for jid, _ in completed[: max(0, len(_jobs) - 30)]:
        _jobs.pop(jid, None)


def create_exclusive_scan_job() -> Optional[str]:
    """
    Atomik: başka scan yoksa yeni job oluştur ve slotu rezerve et.
    Busy ise None döner.
    """
    global _active_scan_job_id
    _reclaim_stale_scan_slot()          # asılı iş varsa slotu kurtar
    with _scan_slot_lock:
        if _active_scan_job_id is not None:
            return None
        job_id = str(uuid.uuid4())
        with _lock:
            _jobs[job_id] = {
                "status": "queued",
                "progress": 0,
                "phase": "",
                "message": "",
                "result": None,
                "error": None,
                "created_at": _utc_now(),
                "last_progress_at": _monotonic(),
            }
            _prune_old_jobs()
        _active_scan_job_id = job_id
    return job_id


def update_job(job_id: str, **kwargs: Any) -> None:
    with _lock:
        j = _jobs.get(job_id)
        if j is None:
            return
        # Bekçi tarafından asılı ilan edilmiş bir iş, sonradan uyanan zombi
        # thread tarafından "completed"a çevrilemez — kullanıcı çoktan yeni
        # tarama başlatmış olabilir, eski sonucu geri getirmek yanıltıcı olur.
        if j.get("error") == "stalled":
            return
        j.update(kwargs)
        # Bekçinin ölçtüğü şey: son İLERLEME anı
        j["last_progress_at"] = _monotonic()


def get_job_public(job_id: str) -> Optional[dict[str, Any]]:
    # Kullanıcı zaten saniyede bir sorguyor — bekçiyi buraya bağlamak, asılı bir
    # işin ekranda sonsuza kadar "çalışıyor" görünmesini engeller.
    _reclaim_stale_scan_slot()
    with _lock:
        j = _jobs.get(job_id)
        if not j:
            return None
        out = {
            "status": j["status"],
            "progress": j["progress"],
            "phase": j.get("phase", ""),
            "message": j.get("message", ""),
            "error": j.get("error"),
        }
        if j["status"] == "completed" and j.get("result") is not None:
            out["result"] = j["result"]
        return out


def release_scan_slot(job_id: str) -> None:
    global _active_scan_job_id
    with _scan_slot_lock:
        if _active_scan_job_id == job_id:
            _active_scan_job_id = None


def current_scan_job_id() -> Optional[str]:
    with _scan_slot_lock:
        return _active_scan_job_id


ProgressCb = Callable[[int, str, str], None]


def run_scan_worker(
    job_id: str,
    body: Any,
    execute_fn: Callable[[Any, ProgressCb], dict],
) -> None:
    """Thread target — slot zaten rezerve; bitince release."""

    def progress(pct: int, phase: str, message: str) -> None:
        pct = max(0, min(100, int(pct)))
        update_job(job_id, progress=pct, phase=phase, message=message)

    try:
        update_job(job_id, status="running", progress=1, phase="starting", message="Starting…")
        result = execute_fn(body, progress)
        update_job(
            job_id,
            status="completed",
            progress=100,
            phase="done",
            message="Complete",
            result=result,
        )
    except Exception as e:
        logger.exception("Background scan job %s failed", job_id)
        update_job(
            job_id,
            status="failed",
            phase="error",
            message=str(e),
            error=str(e),
        )
    finally:
        release_scan_slot(job_id)
