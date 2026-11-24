"""Gölge koşusunun karar penceresine kaç saniye kaldı? (yalnız standart kütüphane)

Daily Run zinciri (``workflow_run``) ABD kapanışından sonra, ET akşamı tetiklenir. Gölge koşusu
NY gece yarısından ÖNCE koşarsa H4 o günün Form 4'lerini işleyemez (``runner.run_h4`` yalnız
today_et'den önceki günleri işler). Bu betik akşamsa sonraki 00:05 ET'ye kadar bekleme süresini,
pencere içindeyse 0'ı, bekleme GitHub iş süresine sığmıyorsa -1'i (atla — sabah cron'ları
karşılar) yazdırır. Kilitli: swing_trader/tests/test_shadow_schedule.py.
"""
from __future__ import annotations

import sys
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
MAX_WAIT_S = 16200                    # 4.5 s; iş zaman aşımı (shadow-run.yml) bunun üstünde
EVENING_FROM = time(16, 0)            # kapanıştan sonrası "akşam": gece yarısı beklenir
TARGET = time(0, 5)                   # NY gece yarısı + 5 dk


def seconds_to_window(now: datetime) -> int:
    et = now.astimezone(NY)
    if et.time() < EVENING_FROM:      # gece yarısından sonra (veya gündüz): hemen koş
        return 0
    nxt = datetime.combine(et.date() + timedelta(days=1), TARGET, tzinfo=NY)
    wait = int((nxt - et).total_seconds())
    return wait if wait <= MAX_WAIT_S else -1


if __name__ == "__main__":
    print(seconds_to_window(datetime.now(timezone.utc)))
    sys.exit(0)
