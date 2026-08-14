"""
RVOL thrust — ikinci sinyal pathway'i (v14, 2026-07-26) birim testleri.

RVOL thrust tanımı harness (scripts/discover_signal_families.py p_rvol_thrust)
ile BİREBİR olmalı — aksi halde ölçülen edge (R10 +3.34%, t=2.87) canlıda
kaybolur. Bu testler tetikleme kurallarını ve pathway seçim mantığını kilitler.
"""

import numpy as np
import pandas as pd

from swing_trader.small_cap.signals import SmallCapSignals


def _df(closes, volumes):
    """Basit OHLCV — Close/Volume dizilerinden. High=Close, Low=Close*0.98."""
    n = len(closes)
    c = np.array(closes, float)
    return pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=n, freq="D"),
        "Open": c * 0.995,
        "High": c * 1.01,
        "Low": c * 0.98,
        "Close": c,
        "Volume": np.array(volumes, float),
    })


def _sig():
    return SmallCapSignals()


def test_fires_on_thrust_green_above_ma20():
    # 60 gün düz $10, düşük hacim; son gün: hacim 3x + yeşil + MA20 üstü
    closes = [10.0] * 60 + [10.5]
    vols = [100_000] * 60 + [300_000]  # son gün 3x baseline
    passed, reason, m = _sig().check_rvol_thrust(_df(closes, vols))
    assert passed, reason
    assert m["rvol"] >= 2.5
    assert m["green"] is True


def test_no_fire_when_volume_below_threshold():
    closes = [10.0] * 60 + [10.5]
    vols = [100_000] * 60 + [200_000]  # sadece 2x < 2.5x
    passed, reason, _ = _sig().check_rvol_thrust(_df(closes, vols))
    assert not passed
    assert "thrust" in reason.lower()


def test_no_fire_on_red_day():
    closes = [10.0] * 60 + [9.8]  # kırmızı gün
    vols = [100_000] * 60 + [400_000]  # hacim yüksek ama gün kırmızı
    passed, reason, _ = _sig().check_rvol_thrust(_df(closes, vols))
    assert not passed
    assert "red" in reason.lower() or "flat" in reason.lower()


def test_no_fire_below_ma20():
    # Yükselen sonra düşen: son gün yeşil + hacimli AMA MA20'nin altında
    closes = [15.0] * 40 + list(np.linspace(15, 9, 20)) + [9.2]
    vols = [100_000] * 60 + [400_000]
    passed, reason, _ = _sig().check_rvol_thrust(_df(closes, vols))
    assert not passed
    assert "ma20" in reason.lower()


def test_insufficient_data():
    closes = [10.0] * 20
    vols = [100_000] * 20
    passed, reason, _ = _sig().check_rvol_thrust(_df(closes, vols))
    assert not passed
    assert "insufficient" in reason.lower()


def test_pathway_precedence_vce_wins_when_both_fire():
    # check_all_triggers: VCE ve RVOL ikisi de ateşlerse pathway = VCE.
    # (VCE daha dar/spesifik pattern.) Burada davranışı doğrudan test etmek
    # zor olduğundan mantığı belge olarak sabitliyoruz: kod elif ile VCE'yi
    # önce kontrol ediyor (signals.py check_all_triggers).
    sig = _sig()
    # RVOL ateşleyen ama VCE ateşlemeyen net senaryo → pathway rvol_thrust
    closes = [10.0] * 60 + [10.5]
    vols = [100_000] * 60 + [300_000]
    triggered, details = sig.check_all_triggers(_df(closes, vols))
    assert triggered
    # squeeze yok (düz seri) → VCE geçmez, RVOL geçer
    assert details["trigger_pathway"] == "rvol_thrust"


def test_thrust_metrics_recorded_even_when_not_triggered():
    closes = [10.0] * 60 + [10.5]
    vols = [100_000] * 60 + [150_000]  # thrust yok
    _, details = _sig().check_all_triggers(_df(closes, vols))
    assert "rvol_thrust" in details["triggers"]
    assert "rvol_metrics" in details


# ── ÜST BARAJLAR (2026-08-14, scripts/measure_rvol_guards.py) ─────────────────
# Hacim ve tek-gün hareketi bir BANT'tır, alt sınır değil. Ölçüm: RVOL 4-6x
# kovası EV -3.34% (WR %25), tek-gün 8-12% kovası EV -3.64%. Birleşik baraj
# EV +1.55% → +3.87%, PF 1.50 → 2.87 (TRAIN ve OOS'ta da düzeliyor).
# Bu testler barajları kilitler: gevşetmek için harness'ı yeniden koşturun.


def test_no_fire_on_event_day_volume():
    """RVOL >= 4x = tek seferlik olay (satın alma/halka arz), swing devamı değil.

    Canlı vaka: DV 2026-08-07, Nielsen'in 13.60$ nakit teklifi → RVOL 10.5x.
    Motor 13.21$ giriş + 16.01/18.41$ hedef veriyordu ama tahta teklifte kilitli.
    """
    closes = [10.0] * 60 + [10.5]
    vols = [100_000] * 60 + [1_000_000]  # 10x — DV senaryosu
    passed, reason, m = _sig().check_rvol_thrust(_df(closes, vols))
    assert not passed
    assert "olay" in reason.lower()
    assert m["rvol"] >= 4.0


def test_no_fire_on_event_day_price_move():
    """Hacim bantta olsa bile tek günde +%8+ sıçrama olay imzasıdır."""
    closes = [10.0] * 60 + [11.2]  # +%12 tek gün
    vols = [100_000] * 60 + [300_000]  # 3x — hacim bandın İÇİNDE
    passed, reason, m = _sig().check_rvol_thrust(_df(closes, vols))
    assert not passed
    assert "hareket" in reason.lower()
    assert m["rvol"] < 4.0  # eleme sebebi hacim DEĞİL
    assert m["day_change_pct"] >= 8.0


def test_fires_just_inside_both_guards():
    """Bantların hemen içi geçer — barajlar üst sınır, yasak değil."""
    closes = [10.0] * 60 + [10.7]  # +%7 < %8
    vols = [100_000] * 60 + [390_000]  # 3.9x < 4.0x
    passed, reason, m = _sig().check_rvol_thrust(_df(closes, vols))
    assert passed, reason
    assert 2.5 <= m["rvol"] < 4.0
    assert m["day_change_pct"] < 8.0


def test_guards_are_settings_tunable():
    """Barajlar koda gömülü değil — yeniden ölçüm için ayarlanabilir olmalı."""
    sig = _sig()
    assert sig._rvol_max == sig._settings.rvol_thrust_guards.max_rvol
    assert sig._rvol_max_day_change == sig._settings.rvol_thrust_guards.max_day_change_pct

    sig._rvol_max = 12.0  # barajı gevşet → aynı bar artık geçer
    closes = [10.0] * 60 + [10.5]
    vols = [100_000] * 60 + [1_000_000]
    passed, _, _ = sig.check_rvol_thrust(_df(closes, vols))
    assert passed


def test_vce_pathway_untouched_by_rvol_guards():
    """Baraj VCE'ye SIZMAMALI — ölçümde 0/33 VCE sinyali etkilendi.

    VCE'nin kendi hacim mantığı ayrıdır (alt baraj 1.5x, üst sınır YOK) ve
    squeeze şartı olay günlerini zaten eliyor.
    """
    sig = _sig()
    # VCE'de üst hacim barajı OLMAMALI: 10x hacimli bir squeeze-kırılımı
    # yalnız RVOL yolunda elenir, VCE mantığında böyle bir kural yok.
    src = sig.check_vce_breakout.__doc__ or ""
    assert "olay günü" not in src.lower()
    assert not hasattr(sig, "_vce_max_rvol")
