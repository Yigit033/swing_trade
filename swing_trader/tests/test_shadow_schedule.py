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
