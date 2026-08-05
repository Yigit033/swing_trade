"""
Asılı tarama işi bekçisi (2026-08-05).

CANLI ARIZA: kullanıcı telefondan tarama başlattı, iş %84'te
("Running momentum engine…") takıldı ve saatlerce orada kaldı. İstisna DEĞİLDİ
— worker try/except/finally ile sarılı, istisna olsaydı "failed" olurdu. Thread
bir ağ çağrısında (yfinance/Finviz) süresiz asılıydı ve hiçbir katmanda timeout
yoktu.

Kullanıcıya maliyeti iki katmanlı:
  1. Ekranda "arka planda çalışıyor" kartı hiç kaybolmuyor (tarayıcıyı kapatıp
     açmak fayda etmiyor — durum sunucuda).
  2. Daha kötüsü: `create_exclusive_scan_job` tek slotu koruduğu için kullanıcı
     MAKİNE YENİDEN BAŞLAYANA KADAR bir daha tarama yapamıyor. Ürün fiilen kilitli.

Kök nedeni tek tek avlamak yerine (asılma herhangi bir ağ çağrısında olabilir)
genel bekçi: ilerleme kaydetmeden STALE_JOB_SECONDS aşılırsa iş başarısız
sayılır ve slot bırakılır.
"""

import pytest

from api import scanner_jobs as sj


@pytest.fixture(autouse=True)
def _clean_store():
    sj._jobs.clear()
    sj._active_scan_job_id = None
    yield
    sj._jobs.clear()
    sj._active_scan_job_id = None


def _age_active_job(seconds: float):
    """Aktif işin son ilerlemesini `seconds` kadar geriye al."""
    jid = sj.current_scan_job_id()
    sj._jobs[jid]["last_progress_at"] = sj._monotonic() - seconds
    return jid


# ── Meşru işler kesilmemeli ──────────────────────────────────────────────

def test_running_job_blocks_second_scan():
    """Normal davranış korunuyor: tek eşzamanlı tarama."""
    first = sj.create_exclusive_scan_job()
    assert first
    assert sj.create_exclusive_scan_job() is None


def test_recent_progress_is_not_reclaimed():
    """İlerleme kaydeden iş asılı sayılmamalı — yavaş tarama kesilmez."""
    jid = sj.create_exclusive_scan_job()
    _age_active_job(sj.STALE_JOB_SECONDS - 30)
    sj.update_job(jid, progress=50, phase="scan")     # ilerleme geldi
    assert sj._reclaim_stale_scan_slot() is None
    assert sj.get_job_public(jid)["status"] in ("queued", "running")
    assert sj.create_exclusive_scan_job() is None, "meşru iş hâlâ slotu tutmalı"


# ── Asılı iş kurtarılmalı ────────────────────────────────────────────────

def test_stalled_job_is_marked_failed():
    jid = sj.create_exclusive_scan_job()
    sj.update_job(jid, status="running", progress=84, phase="scan")
    _age_active_job(sj.STALE_JOB_SECONDS + 60)

    pub = sj.get_job_public(jid)          # kullanıcının polling'i tetikler
    assert pub["status"] == "failed"
    assert pub["error"] == "stalled"
    assert "yeniden tarama" in pub["message"].lower()


def test_stalled_job_frees_the_scan_slot():
    """ASIL MESELE: kullanıcı yeniden tarama yapabilmeli."""
    jid = sj.create_exclusive_scan_job()
    sj.update_job(jid, status="running", progress=84)
    _age_active_job(sj.STALE_JOB_SECONDS + 60)

    new_id = sj.create_exclusive_scan_job()
    assert new_id and new_id != jid, "asılı iş slotu bırakmadı — ürün kilitli kalır"


def test_polling_alone_recovers_without_new_scan():
    """
    Kullanıcı sadece ekrana bakıyorsa bile kart kaybolmalı — bekçi polling
    yolunda da çalışıyor (canlıda kart hiç kaybolmuyordu).
    """
    jid = sj.create_exclusive_scan_job()
    sj.update_job(jid, status="running", progress=84)
    _age_active_job(sj.STALE_JOB_SECONDS + 60)

    assert sj.get_job_public(jid)["status"] == "failed"
    assert sj.current_scan_job_id() is None


# ── Zombi thread eski işi diriltmemeli ───────────────────────────────────

def test_zombie_thread_cannot_resurrect_reclaimed_job():
    """
    Asılı thread sonunda uyanırsa `update_job(status='completed')` çağırır.
    Kullanıcı çoktan yeni tarama başlatmış olabilir; eski sonucu geri getirmek
    yanıltıcı olur.
    """
    jid = sj.create_exclusive_scan_job()
    sj.update_job(jid, status="running", progress=84)
    _age_active_job(sj.STALE_JOB_SECONDS + 60)
    sj.get_job_public(jid)                                   # reclaim

    sj.update_job(jid, status="completed", progress=100, result={"signals": []})
    assert sj.get_job_public(jid)["status"] == "failed"


def test_zombie_release_does_not_steal_new_slot():
    """Geç biten eski thread'in release'i YENİ işin slotunu bırakmamalı."""
    old = sj.create_exclusive_scan_job()
    _age_active_job(sj.STALE_JOB_SECONDS + 60)
    new = sj.create_exclusive_scan_job()
    assert new and new != old

    sj.release_scan_slot(old)             # zombi thread'in finally bloğu
    assert sj.current_scan_job_id() == new, "yeni işin slotu çalındı"
