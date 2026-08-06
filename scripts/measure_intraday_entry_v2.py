# -*- coding: utf-8 -*-
"""
GÜN İÇİ GİRİŞ ÖLÇÜMÜ v2 — v1'in İLERİYE-BAKIŞ YANLILIĞI düzeltildi
================================================================================
v1'in kusuru: "sinyal günü gün içi giriş" (C) yalnızca KAPANIŞTA GEÇERLİ çıkmış
sinyaller üzerinde ölçüldü. Gerçek hayatta saat 11:00'de o günün yeşil
kapanacağını, hacmin 1.5x'e ulaşacağını, kapanışın 20g zirve üstünde kalacağını
BİLEMEZSİN. Yani v1, sonucu bilinen günleri seçip "gün içinde girseydik" dedi —
tüm fakeout'lar ölçümden düştü ve C yapay olarak şişti (+7.58%).

v2 DÜRÜST KURGU — yalnız O AN bilinebilen bilgi:
  Gün içi tetik = (t−1 barında sıkışma) + (t−1 kapanışı MA50 üstünde)
                  + (gün içinde fiyat önceki 20g zirveyi aşıyor)
  Bu üç şart da giriş anında GÖZLENEBİLİR. Kapanışın nasıl olacağı bilinmez —
  dolayısıyla kapanışta geçersizleşen günler de İŞLEM olarak sayılır (fakeout).

KARŞILAŞTIRMA — hepsi AYNI 45 ticker, AYNI çıkış mantığı:
  A   t+1 açılış, yalnız TEYİTLİ sinyaller           → mevcut sistem
  C2  gün içi giriş, TÜM tetiklenebilir günler       → fakeout dahil
  C2c C2'nin yalnız sonradan teyitli çıkan alt kümesi → yanlılığın boyutunu gösterir
  B   t+1 dip limitleri, A ile EŞLEŞTİRİLMİŞ örneklem → doluluk yanlılığı yok

Sermaye kısıtı da ölçülür: daha çok işlem her zaman daha çok para demek değil,
slot doluysa zayıf işlem iyi işlemin yerini kapatır.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")

OOS_SPLIT = "2025-06-01"
DIP_KS = (0.25, 0.50, 1.00)
SQUEEZE_RATIO = 0.8        # signals.VCE_SQUEEZE_RATIO
BREAKOUT_LOOKBACK = 20     # signals.VCE_BREAKOUT_LOOKBACK


def stat(rs):
    if not rs:
        return dict(n=0, ev=0.0, wr=0.0, tot=0.0, oos=0.0)
    a = np.array([x["r"] for x in rs])
    o = np.array([x["r"] for x in rs if x["date"] >= OOS_SPLIT])
    return dict(n=len(a), ev=a.mean(), wr=(a > 0).mean() * 100, tot=a.sum(),
                oos=(o.mean() if len(o) else 0.0))


def line(label, s, base=None):
    d = ""
    if base and base["n"]:
        d = "  ΔEV %+.2f  ΔTOPLAM %+.0f" % (s["ev"] - base["ev"], s["tot"] - base["tot"])
    print("  %-42s%6d%+9.2f%%%7.0f%%%+10.0f%%%+9.2f%%%s"
          % (label, s["n"], s["ev"], s["wr"], s["tot"], s["oos"], d))


def main():
    from backtest_live_replica import enrich, EXIT_NEW, simulate_from_entry

    sigs = json.load(open("output/intraday_signals.json"))
    hourly = pickle.load(open("output/_hourly_bars.pkl", "rb"))
    raw = pickle.load(open("output/_broad_data.pkl", "rb"))
    tickers = sorted(hourly.keys())
    daily = {tk: enrich(raw[tk]) for tk in tickers if tk in raw}
    sig_key = {(s["tk"], s["date"]) for s in sigs}
    print("ticker %d | teyitli sinyal %d\n" % (len(daily), len(sig_key)))

    A, C2, C2c, Braw = [], [], [], {k: [] for k in DIP_KS}
    Amatch = {k: [] for k in DIP_KS}
    n_trigger_days = 0

    for tk, df in daily.items():
        hb = hourly.get(tk)
        if hb is None or len(df) < 80:
            continue
        dates = pd.to_datetime(df["Date"]).dt.date.values
        o = df["Open"].astype(float).values
        hi = df["High"].astype(float).values
        lo = df["Low"].astype(float).values
        cl = df["Close"].astype(float).values
        atr_abs = df["atr"].astype(float).values
        # ATR% serisi — sıkışma testi için
        atrp = np.where(cl > 0, atr_abs / cl * 100.0, np.nan)
        ma50 = pd.Series(cl).rolling(50).mean().values

        for t in range(60, len(df) - 21):
            dstr = str(dates[t])
            # ── Giriş anında BİLİNEN önkoşullar (hepsi t-1 ve öncesi) ──
            base = np.nanmean(atrp[t - 21:t - 6])
            if not np.isfinite(base) or base <= 0:
                continue
            squeezed = atrp[t - 1] < SQUEEZE_RATIO * base
            above50 = cl[t - 1] > ma50[t - 1] if np.isfinite(ma50[t - 1]) else False
            if not (squeezed and above50):
                continue
            hi20_prev = float(np.max(hi[t - BREAKOUT_LOOKBACK:t]))
            if not np.isfinite(hi20_prev) or hi20_prev <= 0:
                continue

            hd = hb[hb.index.date == dates[t]]
            if len(hd) == 0:
                continue
            cross = hd[hd["High"] > hi20_prev]
            if len(cross) == 0:
                continue
            n_trigger_days += 1
            entry_px = float(max(cross["Open"].iloc[0], hi20_prev))
            # Gün içi giriş: pozisyon t GÜNÜNDE açılıyor -> e=t icin t-1 ver
            r = simulate_from_entry(df, t - 1, EXIT_NEW, entry_px)
            if r is None:
                continue
            rec = {"r": r, "date": dstr}
            C2.append(rec)
            if (tk, dstr) in sig_key:
                C2c.append(rec)

        # ── A ve B: teyitli sinyaller uzerinde ──
        for t in range(60, len(df) - 21):
            dstr = str(dates[t])
            if (tk, dstr) not in sig_key:
                continue
            a = simulate_from_entry(df, t, EXIT_NEW, o[t + 1])
            if a is None:
                continue
            A.append({"r": a, "date": dstr})
            for k in DIP_KS:
                lim = o[t + 1] - k * atr_abs[t]
                if lo[t + 1] <= lim:
                    b = simulate_from_entry(df, t, EXIT_NEW, lim)
                    if b is not None:
                        Braw[k].append({"r": b, "date": dstr})
                        Amatch[k].append({"r": a, "date": dstr})   # ESLESTIRILMIS

    print("gün içi tetiklenebilir gün sayısı: %d" % n_trigger_days)
    print("  bunlardan kapanışta TEYİTLİ olan: %d  (fakeout oranı %%%.0f)"
          % (len(C2c), 100 * (1 - len(C2c) / max(len(C2), 1))))
    print()
    W = 104
    print("=" * W)
    print("  %-42s%6s%10s%8s%11s%10s" % ("kurgu", "işlem", "EV", "WR", "TOPLAM", "OOS EV"))
    print("=" * W)
    sA = stat(A)
    line("A   t+1 açılış — TEYİTLİ (MEVCUT SİSTEM)", sA)
    print("  " + "-" * (W - 4))
    line("C2  gün içi giriş — TÜM tetikler (DÜRÜST)", stat(C2), sA)
    line("C2c   ^ yalnız teyitli alt küme (YANLI)", stat(C2c), sA)
    print("  " + "-" * (W - 4))
    for k in DIP_KS:
        sB, sM = stat(Braw[k]), stat(Amatch[k])
        line("B   t+1 dip −%.2f ATR" % k, sB, sM)
        print("  %-42s%6d%+9.2f%%%7.0f%%%+10.0f%%%+9.2f%%   <- ayni islemlerde A"
              % ("      (eşleştirilmiş referans)", sM["n"], sM["ev"], sM["wr"], sM["tot"], sM["oos"]))

    # ── Sermaye kısıtı: slot sınırlı portföy ──
    print()
    print("  SERMAYE KISITI — eşzamanlı pozisyon sınırı (1 ay slot, bileşik)")
    print("  %-42s%12s%12s%12s" % ("", "3 slot", "5 slot", "8 slot"))

    def equity(rs, slots):
        from collections import deque
        rs = sorted(rs, key=lambda x: x["date"])
        eq, open_until = 1.0, deque()
        for r in rs:
            y, m = int(r["date"][:4]), int(r["date"][5:7])
            tm = y * 12 + m
            while open_until and open_until[0] <= tm:
                open_until.popleft()
            if len(open_until) >= slots:
                continue
            open_until.append(tm + 1)
            eq *= (1 + (1.0 / slots) * r["r"] / 100)
        return (eq - 1) * 100

    for label, rs in (("A   t+1 açılış (MEVCUT)", A), ("C2  gün içi (DÜRÜST)", C2)):
        print("  %-42s%11.1f%%%11.1f%%%11.1f%%"
              % (label, equity(rs, 3), equity(rs, 5), equity(rs, 8)))


if __name__ == "__main__":
    main()
