"""
Exit-parametre tutarlılık testleri (2026-07-26).

Exit ölçümü (scripts/exit_lab_vce_rvol.py) sonrası stop genişletildi. Kritik
mimari gereksinim: sinyal üretimi (risk.py, settings'ten okur) ile paper-trade
confirm/exit motoru (tracker.py, kendi sabitleri) AYNI stop cap'lerini
kullanmalı. Biri değişip diğeri unutulursa, sinyalde hesaplanan stop confirm'de
sessizce farklı bir değere kayar — canlıda fark edilmesi zor bir tutarsızlık.
Bu test o ikisini birbirine kilitler.
"""

from swing_trader.small_cap.settings_config import load_settings
from swing_trader.paper_trading import tracker


def test_stop_caps_consistent_between_settings_and_tracker():
    s = load_settings()
    assert dict(s.max_stop_by_type) == tracker.MAX_STOP_BY_TYPE, (
        "max_stop_by_type settings ile tracker arasında ayrıştı — "
        "sinyal stop'u confirm'de farklı kırpılır"
    )


def test_confirm_atr_matches_signal_stop_multiplier():
    # Confirm anındaki stop çarpanı, sinyal üretiminin stop çarpanıyla aynı
    # olmalı — aksi halde giriş stop'u ile confirm stop'u farklı genişlikte olur.
    s = load_settings()
    assert tracker.CONFIRM_ATR_MULTIPLIER == s.stop_atr_multiplier, (
        f"CONFIRM_ATR_MULTIPLIER ({tracker.CONFIRM_ATR_MULTIPLIER}) != "
        f"settings.stop_atr_multiplier ({s.stop_atr_multiplier})"
    )


def test_exit_stop_is_wide_enough_post_measurement():
    # Regresyon: ölçüm dar stop'un kazananları gürültüde kestiğini gösterdi.
    # Stop çarpanı >= 2.0 ve cap'ler >= %12 kalmalı (dar stop'a geri dönülmesin).
    s = load_settings()
    assert s.stop_atr_multiplier >= 2.0
    assert all(v >= 0.12 for v in s.max_stop_by_type.values())
    assert s.max_holding_days >= 15  # sinyaller 20g'de olgunlaşıyor
