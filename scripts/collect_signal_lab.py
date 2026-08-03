# -*- coding: utf-8 -*-
"""
SİNYAL LABORATUVARI — TOPLAMA AŞAMASI
================================================================================
Motoru EN GEVŞEK kapılarla BİR KEZ koştur; her sinyalin ham bileşenlerini,
huni metadatasını ve gerçek exit getirisini kaydet. Bütün analiz (huni taraması,
ağırlık çarpıştırma, eşik taraması, OOS) bu tek dosya üzerinde OFFLINE yapılır —
motor bir daha koşmaz. Hem 10x hızlı hem de her varyant AYNI veriyi görür (adil).

NEDEN GEVŞEK TOPLA: dar toplayıp sonra genişletemezsin. Gevşek toplayıp
daraltmak ise sadece filtrelemedir. Bu yüzden kapılar toplama sırasında en
düşük değerlerine indirilir; gerçek (canlı) değerler analiz aşamasında
filtre olarak uygulanır.

MOTORUN GERÇEK HARD-GATE'LERİ (filters.apply_all_filters, 2026-08-03 denetimi):
    ✓ fiyat            (canlı $7-1000)
    ✓ market cap       (canlı $250M-10B)
    ✓ dolar-hacim      (canlı $5M/gün)   ← asıl likidite kapısı
    ⬜ float, ATR%     TAVSİYE — asla elemez (v13'ten beri)
Not: `min_avg_volume` ayarı apply_all_filters'ta KULLANILMIYOR (yerini
dolar-hacim aldı). Bu, ilk genişletme denemesinde 5 varyantın da tıpatıp
44 sinyal vermesinin sebebiydi — yanlış düğme çevriliyordu.

Cache girdisi : output/_broad_data.pkl, _shares_broad.json, _edge_spy.pkl
Çıktı         : output/signal_lab.json
"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
logging.basicConfig(level=logging.ERROR)
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd

from swing_trader.small_cap.engine import SmallCapEngine
from backtest_live_replica import enrich, simulate, EXIT_NEW, build_regime_map

SMALL = (300e6, 2e9)
MID = (2e9, 10e9)

# ── Toplama kapıları (canlıdan GEVŞEK; daraltma analiz aşamasında) ────────
COLLECT_MIN_PRICE = 3.0            # canlı 7.0
COLLECT_MIN_MCAP = 100e6           # canlı 250e6
COLLECT_MAX_MCAP = 20e9            # canlı 10e9
COLLECT_MIN_DOLLAR_VOL = 1e6       # canlı 5e6
FINVIZ_AV_SMALL = 300e3            # canlı 500e3
FINVIZ_AV_MID = 500e3              # canlı 1e6

# Bileşen maksimumları (scoring.py ile aynı) — normalize için
MAXES = dict(vol=30, atr=25, float=20, mom=15, risk=15, trend=25)


def finviz_hit(row, mcap, av_small=FINVIZ_AV_SMALL, av_mid=FINVIZ_AV_MID,
               min_price=COLLECT_MIN_PRICE):
    """universe.py Q6/Q6b/Q7/Q7b — gevşetilmiş bantlarla."""
    price = row["Close"]
    if price <= min_price:
        return False
    small = mcap is None or (SMALL[0] <= mcap < SMALL[1])
    mid = mcap is None or (MID[0] <= mcap <= MID[1])
    # Gevşek toplamada mcap bandı dışındakileri de al (analiz daraltacak)
    if mcap is not None and not (small or mid):
        small = COLLECT_MIN_MCAP <= mcap <= COLLECT_MAX_MCAP
    av = row["avgvol_liq"]
    if pd.isna(av):
        return False
    ma50, ma20, hi20p = row["ma50"], row["ma20"], row["hi20_prev"]
    new20 = (not pd.isna(hi20p)) and row["High"] > hi20p
    above50 = (not pd.isna(ma50)) and price > ma50
    above20 = (not pd.isna(ma20)) and price > ma20
    rvol = row["Volume"] / row["avgvol50"] if row["avgvol50"] > 0 else 0
    green = row["chg"] > 0

    if (small or mid) and av > (av_mid if mid and not small else av_small):
        if above50 and new20:
            return True
        if rvol > 2 and green and above20:
            return True
    return False


def main():
    print("Veri yukleniyor...", flush=True)
    raw = pickle.load(open("output/_broad_data.pkl", "rb"))
    shares = json.load(open("output/_shares_broad.json"))
    spy = pickle.load(open("output/_edge_spy.pkl", "rb")).copy()
    spy["_d"] = pd.to_datetime(spy["Date"]).dt.normalize()
    rmap = build_regime_map(spy)
    data = {t: enrich(df) for t, df in raw.items()}

    engine = SmallCapEngine()
    sc = engine.scoring
    f = engine.filters
    print(f"  Canli kapilar: fiyat>={f.MIN_PRICE} mcap>={f.MIN_MARKET_CAP/1e6:.0f}M "
          f"dolarvol>={f.MIN_DOLLAR_VOLUME/1e6:.1f}M", flush=True)

    # Kapıları toplama seviyesine indir
    f.MIN_PRICE = COLLECT_MIN_PRICE
    f.MAX_PRICE = 5000.0
    f.MIN_MARKET_CAP = COLLECT_MIN_MCAP
    f.MAX_MARKET_CAP = COLLECT_MAX_MCAP
    f.MIN_DOLLAR_VOLUME = COLLECT_MIN_DOLLAR_VOL
    print(f"  Toplama kapilari: fiyat>={COLLECT_MIN_PRICE} mcap>={COLLECT_MIN_MCAP/1e6:.0f}M "
          f"dolarvol>={COLLECT_MIN_DOLLAR_VOL/1e6:.1f}M", flush=True)

    recs = []
    tickers = list(data.keys())
    for ti, tk in enumerate(tickers):
        df = data[tk]
        sh = shares.get(tk, {})
        sh_out, flt = sh.get("shares"), sh.get("float")
        n = len(df)
        cl = df["Close"].astype(float)
        vol = df["Volume"].astype(float)
        dvol20 = (cl * vol).rolling(20).mean()

        for t in range(60, n - 21):
            row = df.iloc[t]
            close = float(row["Close"])
            mcap = close * sh_out if sh_out else None
            if not finviz_hit(row, mcap):
                continue
            day = pd.to_datetime(df["Date"].iloc[t]).normalize()
            reg = rmap.get(day, "UNKNOWN")
            df_win = df.iloc[:t + 1]
            spy_slice = spy[spy["_d"] <= day].tail(60)
            info = {"ticker": tk, "marketCap": int(mcap) if mcap else 0,
                    "floatShares": int(flt) if flt else 0,
                    "shortName": tk, "sector": "Unknown"}
            try:
                sig = engine.scan_stock(
                    tk, df_win, stock_info=info, backtest_mode=True,
                    portfolio_value=10000,
                    spy_df_window=spy_slice if len(spy_slice) >= 6 else None,
                    regime=reg)
            except Exception:
                sig = None
            if not sig:
                continue
            r = simulate(df, t, EXIT_NEW)
            if r is None:
                continue

            # Ham bileşenler (0-max, ÇARPILMAMIŞ) — ağırlık çarpıştırması için
            atr_p = sig.get("atr_percent", 0) or 0
            atr_frac = atr_p / 100 if atr_p > 1 else atr_p
            float_sh = (sig.get("float_millions", 0) or 0) * 1e6
            try:
                comps = dict(
                    vol=sc.score_volume_explosion(sig.get("volume_surge", 0) or 0),
                    atr=sc.score_volatility_expansion(atr_frac),
                    float=sc.score_float_tightness(float_sh),
                    mom=sc.score_momentum_continuity(df_win),
                    risk=sc.score_risk_control(df_win, atr_frac),
                    trend=sc.score_trend_quality(df_win, None),
                )
            except Exception:
                continue

            recs.append({
                "tk": tk, "date": str(day.date()), "r": float(r),
                "q": float(sig.get("quality_score", 0) or 0),
                "pw": sig.get("trigger_pathway", "?"),
                "reg": reg,
                "type": sig.get("swing_type", "?"),
                # huni metadatasi (analiz asamasinda kapi olarak uygulanacak)
                "price": close,
                "mcap_m": (mcap or 0) / 1e6,
                "dvol_m": float(dvol20.iloc[t]) / 1e6 if not pd.isna(dvol20.iloc[t]) else 0.0,
                "float_m": (flt or 0) / 1e6,
                "avgvol_k": float(row["avgvol_liq"]) / 1e3 if not pd.isna(row["avgvol_liq"]) else 0.0,
                # ham bilesenler
                **{f"c_{k}": float(v) for k, v in comps.items()},
                "vce_premium": bool(sig.get("vce_premium", False)),
                "vce_tight_coil": bool(sig.get("vce_tight_coil", False)),
            })

        if (ti + 1) % 100 == 0:
            print(f"  ...{ti+1}/{len(tickers)} ticker ({len(recs)} sinyal)", flush=True)

    json.dump(recs, open("output/signal_lab.json", "w"), default=str)
    print(f"\nTOPLANDI: {len(recs)} sinyal -> output/signal_lab.json")
    if recs:
        a = np.array([x["r"] for x in recs])
        print(f"  Genel: EV {a.mean():+.2f}%  WR {(a>0).mean()*100:.0f}%")
        print(f"  Tarih araligi: {min(x['date'] for x in recs)} -> {max(x['date'] for x in recs)}")


if __name__ == "__main__":
    main()
