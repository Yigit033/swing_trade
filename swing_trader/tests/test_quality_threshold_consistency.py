"""
Kalite eşiği tek-kaynak tutarlılık testleri (2026-07-27).

Eskiden eşik İKİ ayrı yerde tanımlıydı ve çelişebiliyordu:
  - engine.py: base 70 + {BULL:-10, CAUTION:0, BEAR:+5}  → BULL'da 60
  - thresholds.py: max(UI, regime_caps)                  → BULL'da 0 taban
Ölçüm (measure_score_edge.py) BULL Q60-70'in ~0% getiri verdiğini gösterdi.
İkisi de artık regime_thresholds'tan okur. Bu testler ikisini birbirine
kilitler — biri değişip diğeri unutulamaz.
"""

from swing_trader.small_cap.settings_config import load_settings
from swing_trader.small_cap.thresholds import effective_scan_thresholds


def test_bull_has_a_real_floor():
    # Asıl bug: BULL'un tabanı yoktu (0), değersiz Q60-70 sinyalleri geçiyordu.
    s = load_settings()
    assert s.regime_thresholds.bull_min_quality >= 75, (
        "BULL floor çok düşük — ölçüm Q78'i işaret etti (Q60-70 ~0% getiri)"
    )


def test_api_layer_applies_bull_floor():
    # Kullanıcı UI'da düşük seçse bile (65) BULL floor devreye girmeli.
    s = load_settings()
    eff_min, _ = effective_scan_thresholds("BULL", "CONFIRMED", 65, 10,
                                           regime_caps=s.regime_thresholds)
    assert eff_min == s.regime_thresholds.bull_min_quality
    assert eff_min >= 75


def test_user_can_go_stricter_than_floor():
    # UI floor'un ÜSTÜNE çıkabilmeli (max mantığı) — kullanıcı daha seçici olabilir.
    s = load_settings()
    eff_min, _ = effective_scan_thresholds("BULL", "CONFIRMED", 90, 10,
                                           regime_caps=s.regime_thresholds)
    assert eff_min == 90  # kullanıcının 90'ı floor'u (78) geçer


def test_engine_and_api_floors_match():
    # İki katman AYNI değerleri kullanmalı (tek kaynak: regime_thresholds).
    # engine.py'nin okuduğu floor mantığını burada yeniden kurup API ile kıyasla.
    s = load_settings()
    rt = s.regime_thresholds
    engine_floor = {
        "BULL": rt.bull_min_quality,
        "CAUTION": rt.caution_other_min_quality,
        "BEAR": rt.bear_tentative_min_quality,
    }
    for reg in ("BULL", "CAUTION", "BEAR"):
        conf = "TENTATIVE"
        api_min, _ = effective_scan_thresholds(reg, conf, 0, 10, regime_caps=rt)
        # API, UI=0 verildiğinde saf regime floor'u döndürür
        assert api_min == engine_floor[reg], (
            f"{reg}: engine floor {engine_floor[reg]} != API floor {api_min} — "
            "iki katman ayrıştı"
        )


def test_caution_is_defensive():
    # Ölçüm: CAUTION'da hiçbir eşik kârlı değil → çok yüksek olmalı (korunma).
    s = load_settings()
    assert s.regime_thresholds.caution_other_min_quality >= 78
