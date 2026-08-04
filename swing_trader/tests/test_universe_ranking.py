"""
Evren sıralaması testleri (2026-08-04 basitleştirmesi).

ESKİ TASARIM: 4 ağırlıklı composite skor (rvol .30 / change .25 / volume .25 /
mcap .20) + kovalama cezası + erken-birikim bonusu = ~130 satır.

SORUN: bu ağırlıkların HİÇBİRİ ölçülmemişti. Üstelik sıralamanın etkisi de
doğrulanamıyordu çünkü tavan 15/15 taramada hiç bağlamadı (universe_cut_count=0).
Yani doğrulanmamış bir karar, "hangi hisseler taranacak" sorusuna cevap vermeye
hazır sessizce bekliyordu — ve evren 39 → 172'ye çıkmıştı (tavan 260), yakında
devreye girecekti.

YENİ TASARIM: tek ölçüt = DOLAR-HACİM. Gerekçe ÖLÇÜLDÜ — measure_price_band.py
kesişim testi (fiyat sabit, likidite değişken): likit grup +3.31% WR 62%,
illikit grup -2.14% WR 44%. Likidite elimizdeki en güçlü ayırıcı. Tavan
bağladığında en likit adayları korumak, doğrulanmamış 4 ağırlığa göre
sıralamaktan daha savunulabilir.

Bu testler sıralamanın likidite yönünde olduğunu ve tavanın sessiz kalmadığını
kilitler.
"""

import logging

import pandas as pd
import pytest

from swing_trader.small_cap.universe import SmallCapUniverse, build_rank_info


def _df(rows):
    """rows: (ticker, price, volume, change, rel_vol, mcap) listesi."""
    return pd.DataFrame([
        {"Ticker": t, "Price": p, "Volume": v, "Change": c,
         "Rel Volume": rv, "Market Cap": mc}
        for t, p, v, c, rv, mc in rows
    ])


@pytest.fixture
def univ():
    return SmallCapUniverse()


def test_ranking_is_dollar_volume(univ):
    """Sıralama anahtarı fiyat × hacim olmalı."""
    df = _df([
        ("LOW",  10.0,   400_000, "1.0%", 1.2, "500M"),   # $4M
        ("HIGH", 50.0, 1_000_000, "1.0%", 1.2, "500M"),   # $50M
        ("MID",  20.0,   500_000, "1.0%", 1.2, "500M"),   # $10M
    ])
    out = univ._calculate_composite_score(df.copy())
    ranked = out.sort_values("composite_score", ascending=False)["Ticker"].tolist()
    assert ranked == ["HIGH", "MID", "LOW"], ranked


def test_ranking_ignores_todays_change(univ):
    """
    Günlük değişim sıralamayı ETKİLEMEMELİ. Eski composite'te change %25 ağırlıklıydı
    ve kovalama cezası vardı — ikisi de ölçülmemişti. Artık likidite tek ölçüt.
    """
    df = _df([
        ("CALM",  20.0, 1_000_000, "0.5%",  1.0, "500M"),
        ("SPIKE", 20.0, 1_000_000, "18.0%", 5.0, "500M"),
    ])
    out = univ._calculate_composite_score(df.copy())
    scores = dict(zip(out["Ticker"], out["composite_score"]))
    assert scores["CALM"] == pytest.approx(scores["SPIKE"]), \
        "değişim/RVOL hâlâ sıralamayı etkiliyor"


def test_ranking_ignores_market_cap(univ):
    """Piyasa değeri de sıralamaya girmemeli (eski ağırlık .20, ölçülmemiş)."""
    df = _df([
        ("SMALL", 20.0, 1_000_000, "1.0%", 1.2, "350M"),
        ("BIG",   20.0, 1_000_000, "1.0%", 1.2, "8B"),
    ])
    out = univ._calculate_composite_score(df.copy())
    scores = dict(zip(out["Ticker"], out["composite_score"]))
    assert scores["SMALL"] == pytest.approx(scores["BIG"])


def test_dead_score_components_are_gone(univ):
    """
    2026-08-04: eski composite bileşenleri (rvol_score / change_score /
    mcap_score / early_bonus / vol_score) SİLİNDİ — hiçbiri ölçülmemişti ve
    sıralamaya girmiyorlardı. "Telemetri için tutalım" da yeterli gerekçe
    değildi: okunmayan kolon, okuyanı yanıltan yüktür.
    """
    df = _df([("A", 20.0, 1_000_000, "3.0%", 2.0, "500M")])
    out = univ._calculate_composite_score(df.copy())
    for col in ("rvol_score", "change_score", "vol_score", "mcap_score",
                "early_bonus", "rel_vol", "mcap_numeric", "change_pct"):
        assert col not in out.columns, f"{col} hâlâ hesaplanıyor (silinmeliydi)"


def test_only_needed_columns_remain(univ):
    """Sıralama için gereken minimum kolon seti."""
    df = _df([("A", 20.0, 1_000_000, "3.0%", 2.0, "500M")])
    out = univ._calculate_composite_score(df.copy())
    for col in ("vol_numeric", "price_numeric", "dollar_vol_numeric", "composite_score"):
        assert col in out.columns, f"{col} kayboldu"


# ── Tavan davranışı ──────────────────────────────────────────────────────

def test_cap_keeps_most_liquid():
    """Tavan bağladığında EN LİKİT adaylar korunmalı, illikitler kesilmeli."""
    df = pd.DataFrame({
        "Ticker": ["A", "B", "C", "D"],
        "composite_score": [100.0, 40.0, 80.0, 10.0],
    }).sort_values("composite_score", ascending=False)
    info = build_rank_info(df, cap=2)
    assert list(info["ranks"].keys())[:2] == ["A", "C"]
    assert info["cut_tickers"] == ["B", "D"]


def test_cap_not_binding_reports_zero_cut():
    """Tavan bağlamıyorsa kesilen olmamalı (bugünkü normal durum)."""
    df = pd.DataFrame({"Ticker": ["A", "B"], "composite_score": [2.0, 1.0]})
    info = build_rank_info(df, cap=260)
    assert info["cut_tickers"] == []
    assert info["ranked_total"] == 2


def test_cap_binding_logs_warning(caplog):
    """
    Tavan bağlaması ANORMAL — sessiz kalmamalı. 15/15 taramada hiç bağlamadı;
    bağladığı gün ya evren beklenmedik büyüdü ya sorgu bozuldu demektir.
    Bu test log seviyesinin WARNING'e yükseltildiğini kilitler.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "small_cap" / "universe.py").read_text(
        encoding="utf-8"
    )
    assert "Universe cap BAĞLADI" in src, "tavan uyarı mesajı değişmiş"
    idx = src.index("Universe cap BAĞLADI")
    window = src[max(0, idx - 400):idx]
    assert "logger.warning" in window, "tavan bağlaması hâlâ INFO seviyesinde loglanıyor"
