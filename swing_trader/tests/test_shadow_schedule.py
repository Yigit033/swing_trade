"""Gölge cron'unun saat penceresi kilidi (.github/workflows/shadow-run.yml).

2026-11-23 arızası: GitHub zamanlanmış koşuları 0.4–7.6 saat GEÇ başlatıyor (daily-run'ın son 100
koşusu). Eski 11:15/12:15 UTC cron'ları bu gecikmeyle ABD açılışından SONRA koşuyor, her karar
``late_decision`` olup terfi değerlendirmesinden düşüyordu — ileri test hiç geçerli işlem
biriktiremeyecekti. Bu testler cron'un iki mevsimde de güvenli pencerede kalmasını zorunlu kılar.
"""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from swing_trader.shadow import runner
from swing_trader.utils.market_calendar import last_completed_session, next_trading_day

yaml = pytest.importorskip("yaml")

NY = ZoneInfo("America/New_York")
WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "shadow-run.yml"
MAX_GITHUB_DELAY_H = 8.0          # ölçülen en kötü 7.6 s, yukarı yuvarlandı
NY_MIDNIGHT_UTC_WINTER = 5 * 60   # 00:00 EST = 05:00 UTC (yazın 04:00 — kış daha katı)
OPEN_UTC_SUMMER = 13 * 60 + 30    # 09:30 EDT = 13:30 UTC (kışın 14:30 — yaz daha katı)


def _workflow():
    d = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return d, d.get("on", d.get(True))          # YAML 1.1: çıplak `on` anahtarı True okunur


def _slots():
    _, on = _workflow()
    out = []
    for item in on["schedule"]:
        minute, hour, dom, month, dow = item["cron"].split()
        assert (dom, month) == ("*", "*")
        out.append((int(hour) * 60 + int(minute), dow))
    return sorted(out)


def _dow(spec: str) -> set:
    days = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        days |= set(range(int(a), int(b or a) + 1))
    return days


def test_every_slot_is_after_ny_midnight_in_both_seasons():
    # H4 yalnız today_et'den önceki dosyalama günlerini işler: NY gece yarısından önce koşarsa
    # F'nin Form 4'leri bir gün kayar ve girişi kaçırır.
    for t, _ in _slots():
        assert t >= NY_MIDNIGHT_UTC_WINTER, f"{t // 60:02d}:{t % 60:02d} UTC NY gece yarısından önce"


def test_primary_slot_beats_the_open_even_with_worst_github_delay():
    first = _slots()[0][0]
    assert first + MAX_GITHUB_DELAY_H * 60 <= OPEN_UTC_SUMMER


def test_runs_cover_every_entry_morning_and_are_serialized():
    d, _ = _workflow()
    for _, dow in _slots():
        # cron gün numarası UTC: Sal–Cmt 00:15 ET koşuları Pzt–Cum sinyallerinin girişinden önce
        assert {2, 3, 4, 5, 6} <= _dow(dow)
    assert d["concurrency"]["group"] == "shadow-run"
    assert d["concurrency"]["cancel-in-progress"] is False
    assert len(_slots()) >= 2, "yedek cron yok"


@pytest.mark.parametrize("utc_run, signal_day", [
    (datetime(2026, 11, 24, 5, 15, tzinfo=timezone.utc), date(2026, 11, 23)),   # kış, Sal → Pzt sinyali
    (datetime(2026, 7, 14, 5, 15, tzinfo=timezone.utc), date(2026, 7, 13)),     # yaz (EDT)
    (datetime(2026, 12, 5, 5, 15, tzinfo=timezone.utc), date(2026, 12, 4)),     # Cmt → Cuma sinyali
])
def test_runner_sees_the_right_day_and_is_on_time_with_worst_delay(utc_run, signal_day):
    for delay_h in (0.0, MAX_GITHUB_DELAY_H):
        now = (utc_run + timedelta(hours=delay_h)).astimezone(NY)
        assert last_completed_session(now) == signal_day          # H6/H6S/H6SR asof
        assert now.date() > signal_day                            # H4: F'nin dosyaları işlenir
        assert runner._late(now, signal_day) == 0, (utc_run, delay_h)
        assert next_trading_day(signal_day) > signal_day


def test_old_schedule_would_have_been_late():
    # Gerileme kanıtı: eski 11:15 UTC + ortanca ~4 s gecikme → açılıştan sonra → geç karar.
    now = datetime(2026, 11, 24, 15, 15, tzinfo=timezone.utc).astimezone(NY)
    assert runner._late(now, date(2026, 11, 23)) == 1


# ── İkinci tetik: Daily Run zinciri + NY gece yarısı beklemesi (2026-11-24) ──────────────────
# 2026-11-23 Pazartesi'nin iki gölge cron'u hiç tetiklenmedi; Daily Run 46/46 seans koştu.

def _window_mod():
    import importlib.util
    path = WORKFLOW.parents[1] / "scripts" / "shadow_window.py"
    spec = importlib.util.spec_from_file_location("shadow_window", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_chained_to_daily_run_completion():
    d, on = _workflow()
    daily = yaml.safe_load((WORKFLOW.parent / "daily-run.yml").read_text(encoding="utf-8"))
    assert on["workflow_run"]["workflows"] == [daily["name"]]
    assert on["workflow_run"]["types"] == ["completed"]


def test_job_timeout_covers_the_longest_wait():
    d, _ = _workflow()
    w = _window_mod()
    job = d["jobs"]["shadow-run"]
    assert job["timeout-minutes"] * 60 >= w.MAX_WAIT_S + 45 * 60     # + uyanış ve 3 POST denemesi
    assert job["timeout-minutes"] <= 360                              # GitHub iş sınırı


def test_backups_ask_if_needed_and_manual_runs_always():
    d, _ = _workflow()
    steps = d["jobs"]["shadow-run"]["steps"]
    call = next(s for s in steps if "daily-run" in s.get("run", ""))
    assert "if_needed=${IF_NEEDED}" in call["run"]
    assert call["env"]["IF_NEEDED"] == "${{ github.event_name == 'workflow_dispatch' && 'false' || 'true' }}"
    assert call["if"] == "steps.window.outputs.skip != '1'"
    waits = [s for s in steps if s.get("id") == "window"]
    assert waits and waits[0]["if"] == "github.event_name == 'workflow_run'"


@pytest.mark.parametrize("utc, expect", [
    (datetime(2026, 11, 24, 2, 30, tzinfo=timezone.utc), 2 * 3600 + 35 * 60),   # 21:30 EST → 00:05'e 2s35dk
    (datetime(2026, 7, 14, 1, 0, tzinfo=timezone.utc), 3 * 3600 + 5 * 60),      # 21:00 EDT (yaz)
    (datetime(2026, 11, 24, 5, 30, tzinfo=timezone.utc), 0),                    # 00:30 EST: pencere içinde
    (datetime(2026, 11, 21, 10, 34, tzinfo=timezone.utc), 0),                   # Cmt yakalama 05:34 ET
    (datetime(2026, 11, 23, 23, 0, tzinfo=timezone.utc), -1),                   # 18:00 EST: 6 s → sığmaz, atla
])
def test_window_wait(utc, expect):
    assert _window_mod().seconds_to_window(utc) == expect


@pytest.mark.parametrize("daily_done_utc, signal_day", [
    (datetime(2026, 11, 24, 2, 30, tzinfo=timezone.utc), date(2026, 11, 23)),
    (datetime(2026, 7, 14, 1, 0, tzinfo=timezone.utc), date(2026, 7, 13)),
    (datetime(2026, 12, 5, 3, 40, tzinfo=timezone.utc), date(2026, 12, 4)),    # Cuma → Cmt sabahı
])
def test_chained_run_after_wait_sees_the_right_day_and_is_on_time(daily_done_utc, signal_day):
    wait = _window_mod().seconds_to_window(daily_done_utc)
    assert wait >= 0
    now = (daily_done_utc + timedelta(seconds=wait, minutes=20)).astimezone(NY)   # + uyanış/koşu
    assert last_completed_session(now) == signal_day
    assert now.date() > signal_day                                    # H4: F'nin Form 4'leri işlenir
    assert runner._late(now, signal_day) == 0
