"""
ÖLÜ AYAR / PARİTE testleri (2026-08-04).

Kullanıcı denetiminde üç "ölü ayar" bulundu — settings JSON'da bir değer yazıyor
ama canlı kod onu HİÇ okumuyor. Bu, ayarın yanlış olmasından daha tehlikelidir:
değeri değiştiren kişi davranışın değiştiğini SANIR.

  1. max_gap_down_pct = 4.0  →  tracker'da MAX_GAP_DOWN_PCT = 7.0 kod sabiti
     (canlı davranış doğruydu, ayar yanıltıcıydı)
  2. partial_at_t1_fraction = 0.5  →  tracker'da "T1 partial 50%" gömülü;
     AYRICA harness %33 ölçüyordu → PARİTE KIRIĞI. Ölçüm (measure_t1_fraction.py)
     %33'ün %50'den +0.42 puan iyi olduğunu gösterdi (monotonik, TRAIN+OOS aynı
     yön) → canlı %33'e hizalandı ve değer artık ayardan okunuyor.
  3. max_position_cost_portfolio_pct = 0.15  →  canlı sizing type_position_caps
     (C/A %25) kullanıyor; bu alan yalnız eski backtest modülünde. Değer canlı
     otoriteyle (%25) hizalandı ki iki yol aynı sayıyı görsün.

Bu testler ayar↔davranış bağının kopmasını yakalar.
"""

import pytest

from swing_trader.small_cap.settings_config import load_settings


def test_gap_limits_come_from_settings():
    """tracker gap limitlerini ayardan okumalı — kod sabiti OLMAMALI."""
    from swing_trader.paper_trading import tracker as T

    s = load_settings()
    assert T._gap_limits() == (s.max_gap_up_pct, s.max_gap_down_pct)


def test_gap_limits_are_not_frozen_at_import():
    """
    2026-08-05: limitler `MAX_GAP_UP_PCT = _gap_limits()` şeklinde MODÜL
    İMPORT ANINDA donuyordu — yani ölü-ayar tuzağının yarısı geri gelmişti.
    Kullanıcı UI'dan değeri değiştiriyor, DB'ye yazılıyor, ama çalışan süreç
    import anındaki değeri kullanmaya devam ediyordu. Modül seviyesinde
    donmuş bir kopya bir daha OLUŞMAMALI.
    """
    from pathlib import Path
    import re

    src = (Path(__file__).resolve().parents[1] / "paper_trading" / "tracker.py").read_text(
        encoding="utf-8"
    )
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert not re.search(r"^MAX_GAP_(UP|DOWN)_PCT\s*[,=]", code, re.M), (
        "gap limiti modül seviyesinde donduruluyor — çağrı anında okunmalı"
    )


def test_gap_limits_are_measured_values():
    """Ölçüm (measure_gap_filter.py): up 5 / down 7."""
    s = load_settings()
    assert s.max_gap_up_pct == 5.0
    assert s.max_gap_down_pct == 7.0


def test_gap_limits_respond_to_settings_change(monkeypatch):
    """
    Ayar değişince davranış DEĞİŞMELİ. Bu test tam olarak eski tuzağı yakalar:
    değeri değiştir, fonksiyonun yeni değeri döndürdüğünü doğrula.
    """
    from swing_trader.paper_trading import tracker as T

    class _S:
        max_gap_up_pct = 3.0
        max_gap_down_pct = 9.0

    monkeypatch.setattr(
        "swing_trader.small_cap.settings_config.load_settings", lambda *a, **k: _S()
    )
    up, down = T._gap_limits()
    assert (up, down) == (3.0, 9.0)


def test_t1_fraction_comes_from_settings():
    """T1 kısmi oranı ayardan okunmalı."""
    from swing_trader.paper_trading import tracker as T

    assert T._t1_partial_fraction() == load_settings().partial_at_t1_fraction


def test_t1_fraction_is_measured_value():
    """Ölçüm: %33 (monotonik eğride iç nokta; %25 sınırda olduğu için seçilmedi)."""
    assert load_settings().partial_at_t1_fraction == 0.33


def test_t1_fraction_parity_with_backtest_harness():
    """
    PARİTE: canlı T1 oranı == backtest harness'ının t1_frac'i.
    Ayrışırlarsa backtest artık ürünü ölçmez (bu tam olarak yaşanan hataydı:
    canlı %50 uygularken harness %33 ölçüyordu ve tüm EV sayıları yanlış
    varsayımla üretilmişti).
    """
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "scripts" / "backtest_live_replica.py").read_text(
        encoding="utf-8"
    )
    m = re.search(r"EXIT_NEW\s*=\s*dict\([^)]*t1_frac=([\d.]+)", src, re.S)
    assert m, "EXIT_NEW t1_frac bulunamadı"
    assert float(m.group(1)) == load_settings().partial_at_t1_fraction, (
        "canlı T1 oranı ile harness t1_frac ayrışmış — backtest ürünü ölçmüyor"
    )


def test_position_cost_setting_matches_live_authority():
    """
    max_position_cost_portfolio_pct canlıda kullanılmıyor (authority =
    type_position_caps). En azından SAYISAL olarak çelişmemeli — aksi halde
    eski backtest modülü canlıdan farklı boyutlandırır ve sonuçları kıyaslanamaz.
    """
    s = load_settings()
    assert s.max_position_cost_portfolio_pct == max(s.type_position_caps.values())


@pytest.mark.parametrize("field", [
    "max_gap_up_pct", "max_gap_down_pct", "partial_at_t1_fraction",
])
def test_no_hardcoded_duplicate_in_tracker(field):
    """
    tracker'da bu değerlerin sabit-sayı kopyası kalmamalı. Kopya kalırsa ayar
    yine sessizce ölür.
    """
    from pathlib import Path
    import re

    src = (Path(__file__).resolve().parents[1] / "paper_trading" / "tracker.py").read_text(
        encoding="utf-8"
    )
    # Yorum satırlarını çıkar (ölçüm notlarında sayılar geçiyor, onlar sorun değil)
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert "load_settings" in code, "tracker ayarı hiç okumuyor"
    # 'T1 partial 50%' gibi gömülü metin kalmamalı
    assert not re.search(r"T1 partial \d+%", code), "T1 oranı hâlâ metne gömülü"
