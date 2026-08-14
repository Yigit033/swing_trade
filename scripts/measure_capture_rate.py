# -*- coding: utf-8 -*-
"""
KAZANDIRAN HİSSE YAKALAMA ORANI — "Kazandıran hisseyi buluyor muyuz?"
=====================================================================
Kullanıcının net sorusu: kaç günde kazandırdığı ÖNEMLİ DEĞİL — kazandıran
hisseyi sistem yakaladı mı, kaçırdı mı? (Midas'ta yükselen hisse görüp "biz
neden bulamadık" endişesi.)

İki "kazandıran fırsat" tanımı:
  A) 15 işgününde %20+ yükseliş
  B) 30 işgününde %30+ yükseliş

METODOLOJİ DÜZELTMESİ (2026-08-14) — ÖNEMLİ:
Bu script eskiden fırsatın yalnızca BAŞLANGIÇ GÜNÜNDE sistemi koşturuyordu ve
"%0 yakalama" gibi anlamsız bir sonuç veriyordu. Hata şuydu: bizim tetiğimiz
KIRILIMDA girer, hareketin ilk gününde değil. Hareket taban/sıkışma günlerinde
"başlar", biz birkaç gün sonra kırılımda gireriz — yani doğru soru "hareketin
ilk gününde alabildik mi?" değil, "bu hareketi HİÇ yakalayabildik mi?"dir.

Şimdi EPİZOT bazlı ölçüyoruz: ardışık fırsat günleri tek bir HAREKET sayılır ve
o hareketin HERHANGİ bir gününde sinyal ürettiysek YAKALANDI kabul edilir.
Ayrıca yakaladıklarımızda "kaçıncı günde girdik" ve "girdiğimizde hareketin ne
kadarı KALMIŞTI" da raporlanır — geç yakalamak da bir kayıptır.

AŞAMALAR (nerede kaybediyoruz?):
  0 Finviz emülasyonuna (Q6/Q6b/Q7/Q7b) hiç girmedi  -> aday havuzuna girmedi
  1 Finviz'e girdi ama motor gate'leri eledi          -> ayar sorunu olabilir
  2 Motor sinyal üretti                               -> YAKALANDI

EVREN: varsayılan CANLI-TEMSİLİ evren (Finviz yapısal tarama, NASDAQ dahil).
Eski varsayılan (_broad_data = S&P 400+600) endeks üyeleriydi ve canlıda
avladığımız NASDAQ momentum isimlerini (KYMR, CLBK, SOUN...) içermiyordu.
Kıyas için: python scripts/measure_capture_rate.py sp
"""
import json
import logging
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.ERROR)
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd

from swing_trader.small_cap.engine import SmallCapEngine
from swing_trader.small_cap.settings_config import load_settings
from backtest_live_replica import build_regime_map, enrich, finviz_hit

DEFS = [("15g/%20", 15, 20.0), ("30g/%30", 30, 30.0)]

UNIVERSES = {
    "live": ("output/_live_data.pkl", "output/_shares_live.json",
             "Finviz canli evren"),
    "sp": ("output/_broad_data.pkl", "output/_shares_broad.json",
           "S&P 400+600 (eski)"),
}


def find_episodes(df, window, pct):
    """Ardışık fırsat günlerini TEK harekete grupla -> [(bas, son), ...].

    Fırsat günü = bugünün kapanışından sonraki `window` gün içinde high
    %pct+ yukarıda. Ardışık günleri tek epizot saymak şişirmeyi önler:
    10 gün süren bir yükseliş 10 ayrı "fırsat" değil, 1 harekettir.
    """
    c = df["Close"].astype(float).values
    h = df["High"].astype(float).values
    n = len(df)
    is_opp = np.zeros(n, dtype=bool)
    for t in range(60, n - 1):
        end = min(t + 1 + window, n)
        if end > t + 1:
            is_opp[t] = (h[t + 1:end].max() / c[t] - 1) * 100 >= pct

    eps, start = [], None
    for t in range(60, n - 1):
        if is_opp[t] and start is None:
            start = t
        elif not is_opp[t] and start is not None:
            eps.append((start, t - 1))
            start = None
    if start is not None:
        eps.append((start, n - 2))
    return eps


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "live"
    dpath, spath, uname = UNIVERSES[which]
    raw = pickle.load(open(dpath, "rb"))
    shares = json.load(open(spath))
    spy = pickle.load(open("output/_edge_spy.pkl", "rb")).copy()
    spy["_d"] = pd.to_datetime(spy["Date"]).dt.normalize()
    rmap = build_regime_map(spy)
    data = {t: enrich(df) for t, df in raw.items()}
    engine = SmallCapEngine()
    rt = load_settings().regime_thresholds
    floor = {"BULL": rt.bull_min_quality, "CAUTION": rt.caution_other_min_quality,
             "BEAR": rt.bear_tentative_min_quality}
    print(f"EVREN: {uname} - {len(data)} ticker", flush=True)

    for label, window, pct in DEFS:
        n_ep = 0
        caught = 0
        miss_finviz = 0     # epizot boyunca Finviz'e hiç girmedi
        miss_gate = 0       # girdi ama motor hep eledi
        offsets, remain, qs = [], [], []
        # Motorun eledigi epizotlarda EN ILERI asamaya kadar gelen ret sebebi.
        # Sadece "kac tane eledik" degil, HANGI KAPININ eledigi lazim — aksi
        # halde neyi ayarlayacagimizi bilemeyiz.
        STAGE_RANK = {"insufficient_data": 0, "scan_error": 0, "filter_failed": 1,
                      "no_trigger": 2, "swing_not_ready": 3, "rsi_gate": 4,
                      "stage_rejected": 4}
        reject_hist = {}

        for ti, (tk, df) in enumerate(data.items()):
            sh = shares.get(tk, {})
            sh_out, flt = sh.get("shares"), sh.get("float")
            c = df["Close"].astype(float).values
            h = df["High"].astype(float).values
            nb = len(df)

            for (a, b) in find_episodes(df, window, pct):
                n_ep += 1
                best = 0        # epizotta ulaşılan en ileri aşama
                best_reject, best_rank = None, -1
                for t in range(a, b + 1):
                    row = df.iloc[t]
                    close = float(row["Close"])
                    mcap = close * sh_out if sh_out else None
                    if not finviz_hit(row, mcap):
                        continue
                    best = max(best, 1)
                    day = pd.to_datetime(df["Date"].iloc[t]).normalize()
                    reg = rmap.get(day, "UNKNOWN")
                    spy_slice = spy[spy["_d"] <= day].tail(60)
                    info = {"ticker": tk, "marketCap": int(mcap) if mcap else 0,
                            "floatShares": int(flt) if flt else 0,
                            "shortName": tk, "sector": "Unknown"}
                    rc = {}
                    try:
                        sig = engine.scan_stock(
                            tk, df.iloc[:t + 1], stock_info=info, backtest_mode=True,
                            portfolio_value=10000,
                            spy_df_window=spy_slice if len(spy_slice) >= 6 else None,
                            regime=reg, reject_counts=rc)
                    except Exception:
                        sig = None
                    if not sig:
                        for key in rc:
                            rank = STAGE_RANK.get(key, 5)  # quality_type_* = 5 (en ileri)
                            if rank > best_rank:
                                best_rank, best_reject = rank, key
                        continue
                    q = sig.get("quality_score", 0) or 0
                    if q < floor.get(reg, 78):
                        if best_rank < 6:
                            best_rank, best_reject = 6, "regime_floor"
                        continue
                    # YAKALANDI — ilk sinyal gününü kaydet ve epizottan çık
                    best = 2
                    offsets.append(t - a)
                    qs.append(float(q))
                    end = min(t + 1 + window, nb)
                    if end > t + 1:
                        remain.append((h[t + 1:end].max() / c[t] - 1) * 100)
                    break
                if best == 2:
                    caught += 1
                elif best == 1:
                    miss_gate += 1
                    k = best_reject or "bilinmiyor"
                    reject_hist[k] = reject_hist.get(k, 0) + 1
                else:
                    miss_finviz += 1
            if (ti + 1) % 300 == 0:
                print(f"  ...{ti + 1}/{len(data)} ticker", flush=True)

        W = 78
        print("\n" + "=" * W)
        print(f"  KAZANDIRAN HAREKET YAKALAMA - tanim: {label}  (epizot bazli)")
        print("=" * W)
        print(f"  Toplam kazandiran HAREKET: {n_ep}")
        if n_ep == 0:
            continue
        print(f"  - YAKALANDI: {caught} (%{caught / n_ep * 100:.1f})")
        print(f"  - KACIRILDI: {n_ep - caught} (%{(n_ep - caught) / n_ep * 100:.1f})")
        print("\n  KACIRMA ASAMA DOKUMU (nerede kaybediyoruz?):")
        print(f"    Finviz'e hic girmedi:            {miss_finviz} "
              f"(%{miss_finviz / n_ep * 100:.0f})")
        print(f"    Finviz girdi, motor-gate eledi:  {miss_gate} "
              f"(%{miss_gate / n_ep * 100:.0f})")
        if reject_hist:
            print("\n  ELIMIZDEYDI AMA ELEDIK - HANGI KAPI? (en ileri asama):")
            for k, v in sorted(reject_hist.items(), key=lambda x: -x[1]):
                print(f"    {k:<24} {v:>5}  (%{v / max(miss_gate, 1) * 100:.0f})")
        if offsets:
            o = np.array(offsets)
            r = np.array(remain) if remain else np.array([0.0])
            print("\n  YAKALADIKLARIMIZDA GECIKME ve KALAN POTANSIYEL:")
            print(f"    Hareketin kacinci gununde girdik: medyan {np.median(o):.0f}. gun "
                  f"(p90 {np.percentile(o, 90):.0f}. gun)")
            print(f"    Girdigimizde ONUMUZDE kalan yukseliş: medyan {np.median(r):+.1f}%  "
                  f"ort {r.mean():+.1f}%")
            print(f"    Sinyal kalite skoru: medyan {np.median(qs):.1f}")

    print("\n" + "=" * 78)
    print("  YORUM: 'Finviz girdi ama gate eledi' YUKSEKSE -> sistemde ayar sorunu")
    print("         (elimizdeydi, biz eledik). 'Finviz'e hic girmedi' yuksekse ->")
    print("         o hareketler tezimize (sikisma-kirilim / hacim patlamasi)")
    print("         uymayan tipte; onlari yakalamak YENI bir tetik yolu gerektirir.")


if __name__ == "__main__":
    main()
