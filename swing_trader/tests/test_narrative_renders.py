"""
narrative.py duman testi (2026-08-05).

Neden gerekliydi: `generate_narrative` gövdeyi geniş bir `except Exception`
ile sarıyor ve hata halinde sessizce "Analiz hatası" döndürüyor. Bu tur içinde
tam olarak bu tuzağa düştük — `_generate_turkish`'e `signal` parametresi
geçilmediği halde gövdede `signal.get(...)` çağrıldı, `NameError` except'e
düştü ve TÜM anlatım tek satır hata metnine indi. 267 test geçmeye devam etti,
çünkü hiçbiri anlatımın İÇERİĞİNE bakmıyordu.

Bu test o boşluğu kapatıyor: hata yoluna düşmek = başarısızlık.
"""

import numpy as np
import pandas as pd
import pytest

from swing_trader.small_cap.narrative import generate_signal_narrative
from swing_trader.small_cap.technical_levels import calculate_technical_levels


def _bars(n=90):
    """Düşen trendden dönüp yükselen sentetik bar seti."""
    rng = np.random.default_rng(7)
    close = np.concatenate([np.linspace(30, 22, 50), np.linspace(22, 28, n - 50)])
    close = close * (1 + rng.normal(0, 0.008, n))
    return pd.DataFrame({
        "Open": close * 0.995, "High": close * 1.02, "Low": close * 0.98,
        "Close": close, "Volume": rng.integers(800_000, 3_000_000, n),
    })


def _signal(**over):
    df = _bars()
    px = float(df["Close"].iloc[-1])
    sig = {
        "ticker": "TEST", "swing_type": "A", "quality_score": 84.0,
        "entry_price": px, "stop_loss": px * 0.93,
        "target_1": px * 1.08, "target_2": px * 1.20,
        "hold_days_min": 7, "hold_days_max": 12,
        "rsi": 63.0, "volume_surge": 2.4, "atr_percent": 5.0,
        "five_day_return": 6.2, "float_millions": 18.0,
        "sector_rs_score": 18.0, "is_sector_leader": True,
        "macd_bullish": True, "higher_lows": True,
        "market_regime": "BULL",
        "technical_levels": calculate_technical_levels(df, px, 2.4),
    }
    sig.update(over)
    return sig


def test_narrative_does_not_fall_into_error_path():
    """Geniş except'in yuttuğu her hata bu testte görünür."""
    nar = generate_signal_narrative(_signal())
    assert "Analiz hatası" not in nar["headline"], (
        "Anlatım hata yoluna düştü — gövdede bir istisna var, "
        "geniş `except Exception` onu gizliyor."
    )
    assert "oluşturulamadı" not in nar["full_text"]


@pytest.mark.parametrize("section_marker", [
    "📌 **Setup:**",
    "📊 **Fiyat Seviyeleri:**",
    "📍 **Entry:**",
    "🛑 **Stop:**",
    "🎯 **T1:**",
    "⚖️ **Risk/Ödül:**",
    "🎯 **Öneri:**",
])
def test_narrative_has_all_core_sections(section_marker):
    assert section_marker in generate_signal_narrative(_signal())["full_text"]


def test_t1_fraction_comes_from_settings_not_hardcoded():
    """
    T1 kısmi satış oranı %50'den %33'e indi (ölçüldü). Metin sabit "yarısını"
    yazıyorsa kullanıcıya YANLIŞ talimat verir.
    """
    from swing_trader.small_cap.settings_config import load_settings
    pct = load_settings().partial_at_t1_fraction * 100
    text = generate_signal_narrative(_signal())["full_text"]
    assert f"%{pct:.0f}'ini sat" in text
    assert "yarısını sat" not in text


# ── VCE işaretleri: skoru +8/+5 kaydıran iki şey açıklanmalı ──────────────

def test_vce_flags_are_explained_when_present():
    """
    Bu iki işaret skorun neden yüksek olduğunun cevabı. Anlatımda yoklarsa
    kullanıcı sıralamayı anlayamaz. (Ölçüm: bonuslu skor bonussuzdan daha iyi
    sıralıyor — GATE_AUDIT.md "4. tur".)
    """
    text = generate_signal_narrative(
        _signal(vce_premium=True, vce_tight_coil=True))["full_text"]
    assert "Premium sıkışma" in text
    assert "Sıkı yay" in text


def test_vce_flags_absent_when_not_set():
    text = generate_signal_narrative(_signal())["full_text"]
    assert "Premium sıkışma" not in text
    assert "Sıkı yay" not in text


def test_removed_type_s_does_not_appear():
    """Type S 2026-08-04'te kaldırıldı — metinde izi kalmamalı."""
    text = generate_signal_narrative(_signal(swing_type="B"))["full_text"]
    assert "Hızlı hareketlere hazır ol" not in text
