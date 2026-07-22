"""Otomatik tarama zamanlama mantığı — birim testleri (network/DB'siz)."""

from datetime import datetime

from swing_trader.utils.market_calendar import _NYSE_TZ
from api.auto_scan import _next_target_et


def _et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=_NYSE_TZ)


def test_next_target_before_today_target():
    # 15:00 ET'de, hedef 16:30 ET → bugünün hedefi (henüz geçmedi)
    now = _et(2026, 7, 22, 15, 0)
    target = _next_target_et(16, 30, now)
    assert target == _et(2026, 7, 22, 16, 30)


def test_next_target_after_today_target_rolls_to_tomorrow():
    # 18:00 ET'de, hedef 16:30 ET çoktan geçti → yarına kayar
    now = _et(2026, 7, 22, 18, 0)
    target = _next_target_et(16, 30, now)
    assert target == _et(2026, 7, 23, 16, 30)


def test_next_target_exact_moment_rolls_to_tomorrow():
    # Tam hedef anındaysak "henüz geçmedi" değil "geçti" sayılır (>= değil <)
    # — aynı saniyede sonsuz döngüye düşmesin.
    now = _et(2026, 7, 22, 16, 30)
    target = _next_target_et(16, 30, now)
    assert target == _et(2026, 7, 23, 16, 30)


def test_next_target_month_boundary_does_not_crash():
    # timedelta kullanımı ay sonunu doğru taşımalı (day=31 → day=32 patlardı)
    now = _et(2026, 7, 31, 20, 0)
    target = _next_target_et(16, 30, now)
    assert target == _et(2026, 8, 1, 16, 30)
