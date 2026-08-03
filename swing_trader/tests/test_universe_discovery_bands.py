"""
Finviz keşif bandı + canlı/harness paritesi testleri (2026-08-03).

BAĞLAM: canlı üretim ~0.6 Q80 sinyal/ay veriyordu; profesyonel swing pratiği
4-12 işlem/ay. scripts/analyze_signal_lab.py (108 sinyal, 21 ay) sekiz kapıyı
tek tek gevşetti. Dolar-hacim ($5M), market cap ($250M) ve fiyat ($7) kapılarını
gevşetmek HİÇ ek sinyal getirmedi — yalnız Finviz "Average Volume" bandı getirdi:
+20 sinyal (%23), ek sinyallerin EV'si +2.16%, toplam EV seyrelmeden korundu
(+2.32% → +2.29%), OOS'ta da tuttu (train +2.34% / test +2.25%).

Bu testler iki şeyi kilitler:
  1. Bandın ölçülen değerde kalması (sessiz geri dönüş / elle kurcalama koruması).
  2. CANLI ile BACKTEST HARNESS'ının aynı bandı kullanması. Ayrışırlarsa backtest
     artık ürünü ölçmez — bu projede daha önce yaşanmış bir hata sınıfı
     (backtest_old_vs_new.py motoru atlıyordu ve silindi).
"""

import re
from pathlib import Path

import pytest

from swing_trader.small_cap import universe as U


REPO = Path(__file__).resolve().parents[2]


def test_discovery_bands_are_the_measured_values():
    """Ölçümle belirlenen bant: small 300K, mid 500K."""
    assert U.FINVIZ_MIN_AVG_VOLUME_SMALL == "Over 300K"
    assert U.FINVIZ_MIN_AVG_VOLUME_MID == "Over 500K"


def test_queries_use_the_constants_not_literals():
    """
    Q6/Q6b/Q7/Q7b sabitleri kullanmalı. Sorgu içine tekrar sabit-string yazılırsa
    tek-kaynak bozulur ve harness paritesi sessizce kayar.
    """
    src = (REPO / "swing_trader" / "small_cap" / "universe.py").read_text(encoding="utf-8")
    # Aktif sorgu bloklarını al (Q6/Q6b/Q7/Q7b — 'New High' veya 'Relative Volume' içerenler)
    blocks = re.findall(r"q(?:6b?|7b?)_filters = \{(.*?)\}", src, re.S)
    assert len(blocks) == 4, f"4 aktif sorgu bekleniyordu, {len(blocks)} bulundu"
    for i, b in enumerate(blocks):
        assert "FINVIZ_MIN_AVG_VOLUME_" in b, f"sorgu {i} sabiti kullanmıyor:\n{b}"
        assert "'Over 500K'" not in b and "'Over 1M'" not in b, \
            f"sorgu {i} hâlâ sabit-string bant içeriyor:\n{b}"


def _harness_bands():
    """backtest_live_replica.finviz_hit içindeki sayısal bantları çıkar."""
    src = (REPO / "scripts" / "backtest_live_replica.py").read_text(encoding="utf-8")
    # Fonksiyon gövdesi: 'def finviz_hit' → bir sonraki üst-seviye def/başlık
    body = re.search(r"def finviz_hit\(.*?(?=\n(?:def |#\s*═))", src, re.S)
    assert body, "finviz_hit gövdesi bulunamadı"
    bands = set(re.findall(r"av > (\d+(?:\.\d+)?)e(\d+)", body.group(0)))
    assert bands, "harness'ta 'av > NNNeM' kalıbı bulunamadı"
    return bands


def _to_number(band: str) -> float:
    """'Over 300K' -> 300_000.0"""
    m = re.match(r"Over \$?([\d.]+)([KMB]?)", band)
    assert m, f"bant ayrıştırılamadı: {band}"
    mult = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}[m.group(2)]
    return float(m.group(1)) * mult


def test_live_and_backtest_harness_use_same_bands():
    """
    PARİTE: canlı sorgu bandı == harness emülasyon bandı.
    Ayrışırlarsa backtest başka bir evreni ölçer ve sonuçları ürüne dair olmaktan çıkar.
    """
    live = {_to_number(U.FINVIZ_MIN_AVG_VOLUME_SMALL), _to_number(U.FINVIZ_MIN_AVG_VOLUME_MID)}
    harness = {float(m) * (10 ** int(e)) for m, e in _harness_bands()}
    assert live == harness, (
        f"canlı bantlar {sorted(live)} ile harness bantları {sorted(harness)} ayrışmış — "
        "backtest artık ürünü ölçmüyor"
    )


def test_widening_did_not_lower_the_liquidity_hard_gate():
    """
    Keşif genişledi ama KARAR katmanı sertliğini korumalı: motorun dolar-hacim
    hard-gate'i ($5M/gün) yerinde olmalı. Aksi halde gevşetme gerçek bir likidite
    riski olurdu.
    """
    from swing_trader.small_cap.filters import SmallCapFilters

    f = SmallCapFilters()
    assert f.MIN_DOLLAR_VOLUME >= 5_000_000, \
        f"dolar-hacim kapısı gevşemiş: {f.MIN_DOLLAR_VOLUME}"


@pytest.mark.parametrize("band", ["FINVIZ_MIN_AVG_VOLUME_SMALL", "FINVIZ_MIN_AVG_VOLUME_MID"])
def test_bands_are_valid_finviz_strings(band):
    """Finviz yalnız belirli hazır seçenekleri kabul eder; serbest metin sessizce boş döner."""
    value = getattr(U, band)
    assert value in {
        "Over 50K", "Over 100K", "Over 200K", "Over 300K", "Over 400K", "Over 500K",
        "Over 750K", "Over 1M", "Over 2M",
    }, f"{band}={value!r} geçerli bir Finviz seçeneği değil"
