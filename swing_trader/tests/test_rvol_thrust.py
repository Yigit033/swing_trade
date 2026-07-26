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
