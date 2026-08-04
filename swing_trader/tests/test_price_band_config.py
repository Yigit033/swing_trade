"""
Maksimum fiyat tavanı testleri (2026-08-04).

BAĞLAM: max_price 1000 → 200. İki bağımsız gerekçe aynı noktada buluştu.

1) ÖLÇÜM (scripts/measure_price_band.py, 90 Q80+ sinyal / 21 ay, gerçek motor
   + gerçek exit + slippage):
       maks $1000 (eski): EV +2.36%  PF 1.71   OOS +2.51%
       maks $200        : EV +3.00%  PF 1.96   OOS +3.45%
       maks $100        : EV +3.18%  PF 2.05   OOS +3.87%
   Elenenler: $200 üstü n=11 EV -2.24%; $100 üstü n=22 EV -0.16%.
   $200 seçildi çünkü $100-200 bandı ~sıfır (net negatif değil) ve onu da kesmek
   11 sinyal daha götürüyor — sinyal kıtlığı varken bedeli faydasından büyük.

2) MEKANİK (veri deseninin ARKASINDAKİ sebep — desenin tesadüf olmadığının işareti):
   Pozisyon tavanı $10.000 portföyün %25'i = $2.500.
       $100 hisse → 25 adet, 1 adet = pozisyonun %4    (sağlam)
       $200 hisse → 12 adet, 1 adet = pozisyonun %8    (kaba)
       $400 hisse →  6 adet, 1 adet = pozisyonun %16   (kullanılamaz)
       $618 hisse →  4 adet, 1 adet = pozisyonun %25   (kullanılamaz)
   Pahalı hissede stop/hedef matematiği yuvarlamaya boğulur.

REDDEDİLEN: minimum fiyatı yükseltmek. Fiyat bandı deseni iki evrende TERS yön
gösterdi (likit $10-20 +5.37% vs illikit -0.57%) → gerçek bir fiyat etkisi yok.
Asıl ayırıcı LİKİDİTE: aynı $10-30 bandında likit +3.31% / illikit -2.14%.

⚠️ PORTFÖYE BAĞIMLI: mekanik gerekçe $10k portföy içindir. Portföy büyürse
(ör. $100k → pozisyon tavanı $25k) granülerlik sorunu kaybolur ve tavan
yeniden ölçülmelidir. Bu yüzden değer koda gömülü DEĞİL, ayarda.
"""

import pytest

from swing_trader.small_cap.settings_config import load_settings
from swing_trader.small_cap.filters import SmallCapFilters


def test_max_price_is_measured_value():
    s = load_settings()
    assert s.universe_filters.max_price == 200.0, \
        "maks fiyat ölçülen değerden saptı (measure_price_band.py)"


def test_post_filter_matches_universe_filter():
    """
    Evren post-filtresi motor filtresiyle AYNI tavanı kullanmalı. Ayrışırlarsa
    Finviz'den $200 üstü hisse çekilip motorda elenir — boşa fetch + kafa
    karıştıran 'filter_failed' sayacı.
    """
    s = load_settings()
    assert s.universe_scan.post_filter_price_max == s.universe_filters.max_price


def test_min_price_unchanged():
    """Minimum fiyat DEĞİŞMEDİ — fiyat bandı deseni iki evrende ters çıktı."""
    s = load_settings()
    assert s.universe_filters.min_price == 7.0


def test_filters_instance_picks_up_setting():
    """Ayar filters örneğine gerçekten yansıyor mu (ölü ayar olmasın)."""
    f = SmallCapFilters()
    assert f.MAX_PRICE == 200.0
    assert f.MIN_PRICE == 7.0


@pytest.mark.parametrize("price,ok", [
    (7.0, True), (50.0, True), (199.99, True), (200.0, True),
    (200.01, False), (618.0, False),
])
def test_price_gate_boundaries(price, ok):
    """Sınır davranışı net olsun: 200 dahil, üstü red."""
    f = SmallCapFilters()
    passed, _ = f.check_price(price)
    assert passed is ok, f"${price} için beklenen {ok}, gelen {passed}"


def test_position_granularity_rationale_holds():
    """
    Mekanik gerekçeyi kilitle: tavanın izin verdiği en pahalı hissede bile
    1 adet, pozisyonun %10'undan az olmalı. Bu şart bozulursa (ör. tavan
    yükseltilirse) sizing yuvarlamaya boğulur ve gerekçe çürür.
    """
    s = load_settings()
    max_px = s.universe_filters.max_price
    cap_pct = max(s.type_position_caps.values())        # en geniş tip tavanı
    position_usd = 10_000 * cap_pct
    one_share_pct = max_px / position_usd * 100
    assert one_share_pct < 10.0, (
        f"${max_px} hissede 1 adet pozisyonun %{one_share_pct:.1f}'i — "
        "granülerlik gerekçesi bozuldu"
    )
