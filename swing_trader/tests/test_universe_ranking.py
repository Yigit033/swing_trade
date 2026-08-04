"""
Evren sıralaması testleri.

2026-08-04: eski 4-ağırlıklı composite skor (rvol .30 / change .25 / volume .25 /
mcap .20) + kovalama cezası + erken-birikim bonusu SİLİNDİ — metot dahil
(~130 satır). Hiçbir ağırlık ölçülmemişti ve etkisi de doğrulanamıyordu: tavan
15/15 taramada bağlamadı (cut=0).

Yeni: tek ölçüt DOLAR-HACİM, inline. Gerekçe ölçülmüş — measure_price_band.py
kesişim testi (fiyat SABİT, likidite değişken): likit +3.31% WR %62 / illikit
−2.14% WR %44. Sıralamanın tek işi, tavan bağladığında en likit adayları korumak.
"""

from pathlib import Path

import pandas as pd

from swing_trader.small_cap.universe import SmallCapUniverse, build_rank_info

SRC = Path(__file__).resolve().parents[1] / "small_cap" / "universe.py"


def test_composite_score_method_is_gone():
    """Metot tamamen kaldırıldı — inline tek ifade kaldı."""
    assert not hasattr(SmallCapUniverse, "_calculate_composite_score")


def test_dead_helpers_and_columns_are_gone():
    """
    Silinen parçalar geri sızmasın: 4 ağırlık, kovalama cezası, erken-birikim
    bonusu, ara skor kolonları ve artık kullanılmayan yüzde parser'ı.
    """
    src = SRC.read_text(encoding="utf-8")
    for dead in ("rank_weight", "chase_penalty", "early_bonus", "rvol_score",
                 "change_score", "mcap_score", "composite_score",
                 "_parse_percent"):
        assert dead not in src, f"{dead} hâlâ kodda"


def test_ranking_key_is_dollar_volume():
    """Sıralama fiyat × hacme göre, azalan."""
    src = SRC.read_text(encoding="utf-8")
    assert "df['dollar_volume']" in src, "dollar_volume hesabı yok"
    assert "sort_values('dollar_volume', ascending=False)" in src, \
        "dolar-hacme göre azalan sıralama yok"
    i = src.index("df['dollar_volume']")
    block = src[i:i + 260]
    assert "Volume" in block and "Price" in block, block


def test_no_query_flags_remain():
    """Kapatılmış Q1/Q2/Q3 sorgu bayrakları tamamen gitti."""
    assert "enable_finviz_query" not in SRC.read_text(encoding="utf-8")


# ── Tavan davranışı ──────────────────────────────────────────────────────

def test_cap_keeps_top_of_order():
    """Tavan bağladığında sıralamanın başı korunur, kuyruk kesilir."""
    info = build_rank_info(pd.DataFrame({"Ticker": ["A", "C", "B", "D"]}), cap=2)
    assert list(info["ranks"].keys())[:2] == ["A", "C"]
    assert info["cut_tickers"] == ["B", "D"]


def test_cap_not_binding_reports_zero_cut():
    """Tavan bağlamıyorsa kesilen olmamalı (bugünkü normal durum)."""
    info = build_rank_info(pd.DataFrame({"Ticker": ["A", "B"]}), cap=260)
    assert info["cut_tickers"] == []
    assert info["ranked_total"] == 2


def test_cap_binding_logs_warning():
    """
    Tavan bağlaması ANORMAL — 15/15 taramada bağlamadı. Bağladığı gün ya evren
    beklenmedik büyüdü ya sorgu bozuldu demektir; sessiz kalmamalı.
    """
    src = SRC.read_text(encoding="utf-8")
    idx = src.index("Universe cap BAĞLADI")
    assert "logger.warning" in src[max(0, idx - 300):idx], \
        "tavan bağlaması WARNING seviyesinde değil"
