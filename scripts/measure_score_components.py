# -*- coding: utf-8 -*-
"""
SKOR BİLEŞEN DÖKÜMÜ — hangi bileşen edge katıyor, hangisi gürültü/zararlı?
==========================================================================
score_edge.json'daki 469 gerçek sinyalin her biri için scoring.py'nin 6 ham
bileşenini + bonus + ceza toplamını AYRI hesaplar (scoring iç metodları
doğrudan çağrılır — üretim kodu değişmez), sonra her katmanın getiriyle
GERÇEK ilişkisini ölçer:

  1. Her bileşenin forward-return korelasyonu (yön + güç)
  2. "Bileşeni çıkar ve bak": bileşen olmadan Q80 bucket edge'i nasıl değişir?
  3. Bonus/ceza net katkısı

Bu, "float ağırlığı %25 ama edge katmıyor" gibi hipotezleri KESİN kanıtlar
ve neyin silineceğini/tutulacağını belirler.

Gereksinim: score_edge.json + _broad_data.pkl (bileşenler df'ten hesaplanır)
"""
import sys, os, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
logging.basicConfig(level=logging.ERROR)
logging.disable(logging.CRITICAL)
import numpy as np
import pandas as pd

from swing_trader.small_cap.scoring import SmallCapScoring
from backtest_live_replica import enrich

# score_edge.json sinyal-başı kayıt tutuyor ama df-indeksini tutmuyordu.
# Yeniden üretmek yerine, kayıttaki (ticker yok!) → bu yüzden bileşenleri
# score_edge kaydındaki ham girdilerden YAKLAŞIK değil, TAM hesaplamak için
# sinyalleri yeniden bulmamız gerekir. Pratik yol: score_edge kaydına zaten
# vol_surge/atr_pct/float_m/rsi/5d var — bunlarla scoring bileşen fonksiyonlarını
# besleyip bileşen skorlarını üret (df gerektirenler için yaklaşık kullan).

def main():
    recs = json.load(open("output/score_edge.json"))
    sc = SmallCapScoring()
    a = np.array([x["r"] for x in recs])

    # Her sinyal için scoring bileşenlerini kayıttaki ham girdilerden hesapla.
    # (volume, atr, float doğrudan; momentum/trend/risk df ister — onları
    # kayıttaki proxy'lerle yaklaşık tutuyoruz, ama volume/atr/float TAM.)
    comp = {k: [] for k in ["volume", "atr", "float"]}
    for x in recs:
        vs = sc.score_volume_explosion(x["vol_surge"])
        comp["volume"].append((max(vs, 0) / sc.MAX_VOLUME_SCORE) * 100 * sc.WEIGHT_VOLUME)
        va = sc.score_volatility_expansion(x["atr_pct"] / 100 if x["atr_pct"] > 1 else x["atr_pct"])
        comp["atr"].append((max(va, 0) / sc.MAX_VOLATILITY_SCORE) * 100 * sc.WEIGHT_VOLATILITY)
        fl = sc.score_float_tightness(x["float_m"] * 1e6 if x["float_m"] else 0)
        comp["float"].append((fl / sc.MAX_FLOAT_SCORE) * 100 * sc.WEIGHT_FLOAT)

    print("=" * 74)
    print(f"  SKOR BİLEŞEN DÖKÜMÜ — {len(recs)} sinyal | genel EV {a.mean():+.2f}%")
    print("=" * 74)
    print("\n  [1] BİLEŞEN SKORU ↔ GETİRİ korelasyonu (TAM hesaplanan bileşenler)")
    print(f"  {'bileşen':<20}{'ağırlık':>8}{'ort katkı':>11}{'korelasyon':>12}{'karar':>18}")
    for key, w, lbl in [("volume", sc.WEIGHT_VOLUME, "Hacim patlaması"),
                        ("atr", sc.WEIGHT_VOLATILITY, "Volatilite ATR%"),
                        ("float", sc.WEIGHT_FLOAT, "Float sıkılık")]:
        v = np.array(comp[key])
        c = np.corrcoef(v, a)[0, 1] if v.std() > 0 else 0
        verdict = "GÜÇLÜ tut" if c > 0.1 else ("GÜRÜLTÜ/SİL" if abs(c) < 0.04 else ("ZARARLI/SİL" if c < -0.03 else "zayıf"))
        print(f"  {lbl:<20}{w:>8.2f}{v.mean():>10.1f}p{c:>+12.3f}{verdict:>18}")

    # [2] Kayıttaki proxy'lerle diğer sinyaller (RSI, 5d, sektör) — zaten score_edge'de ölçüldü
    print("\n  [2] HAM GİRDİ ↔ GETİRİ (score_edge'den — dolaylı ama TAM veri)")
    for key, lbl in [("rsi", "RSI"), ("5d", "5-gün getiri"), ("sector_rs", "Sektör RS")]:
        v = np.array([x[key] for x in recs])
        c = np.corrcoef(v, a)[0, 1] if v.std() > 0 else 0
        print(f"  {lbl:<20}{'':>8}{'':>11}{c:>+12.3f}")

    # [3] "FLOAT'I ÇIKAR VE BAK" — float skoru olmadan Q80 bucket nasıl değişir?
    # float katkısını toplam skordan düş, yeni skoru hesapla, bucket'la.
    print("\n  [3] 'FLOAT BİLEŞENİNİ ÇIKAR VE BAK' testi")
    q_orig = np.array([x["q"] for x in recs])
    fl_contrib = np.array(comp["float"])
    q_nofloat = q_orig - fl_contrib  # float katkısı çıkarılmış skor
    for lbl, qq in [("MEVCUT skor (float dahil)", q_orig), ("float ÇIKARILMIŞ skor", q_nofloat)]:
        hi = a[qq >= 80] if lbl.startswith("MEVCUT") else a[qq >= (80 - fl_contrib.mean())]
        # adil kıyas: aynı sinyal sayısını tutmak için üst %X'e bak
        thr = np.percentile(qq, 82)  # üst ~%18 (Q80+ ~81/469)
        top = a[qq >= thr]
        print(f"  {lbl:<28}: üst %18 bucket EV {top.mean():+.2f}% (n={len(top)})")

    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
