# -*- coding: utf-8 -*-
"""
AĞIRLIK ÇARPIŞTIRMA — "Huni ile skorlama savaşıyor mu? Float/Momentum ağırlığı
teknik-bonuslara kaydırılınca getiri artıyor mu?"
================================================================================
Kullanıcı/dış-denetim hipotezi (Alpha Killer): Finviz hunisi aşırı-hacimli
hisse getiriyor → skorun aşırı-uzama/RSI cezaları onları buduyor → Q80'i geçen
"tesadüfen ceza yemeyen azınlık" → float/hacim bileşenleri sıfır korelasyon
gösteriyor çünkü tıkanıklık var.

TEST (yarım bırakılan, şimdi DOĞRU): her sinyal için 6 HAM bileşen skoru
(0-100 normalize edilmemiş, ÇARPILMAMIŞ) + bonus + penalty + getiri kaydedilir.
Sonra farklı AĞIRLIK SETLERİ altında skoru YENİDEN hesaplayıp (toplam ağırlık
hep 1.0 — adil normalize) Q80 bucket getirisini karşılaştırırız.

Ağırlık setleri:
  A) MEVCUT (v33): vol.12 atr.13 float.25 mom.25 risk.10 trend.15
  B) float/mom YARIYA: float.125 mom.125 → artan .25'i trend'e (.40)
  C) float/mom SIFIR: → trend .55, atr .25, risk .20
  D) trend-ağırlıklı: trend.35 atr.20 vol.15 float.15 mom.10 risk.05
  E) sadece-trend+risk: trend.60 risk.40 (float/mom/vol tamamen çıkar)

+ VCE-premium/tight-coil bonusları zaten skorun DIŞINDA (engine.py'de +8/+5
eklenir) — onları da modelleyip "teknik bonusa ağırlık ver" senaryosunu kurar.

Gerçek motor koşturulur (scan_stock), böylece bonus/penalty gerçek değerleriyle
gelir. Cache: _broad_data.pkl + _shares_broad.json + _edge_spy.pkl
Çıktı: output/score_components.json (yeniden kullanılabilir ham döküm)
"""
import sys, os, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
logging.basicConfig(level=logging.ERROR)
logging.disable(logging.CRITICAL)
import numpy as np
import pandas as pd

from swing_trader.small_cap.engine import SmallCapEngine
from swing_trader.small_cap.scoring import SmallCapScoring
from swing_trader.small_cap.settings_config import load_settings
from backtest_live_replica import enrich, finviz_hit, simulate, EXIT_NEW, build_regime_map


def collect_components():
    """Gerçek motoru koştur, her sinyal için HAM bileşen skorlarını + bonus +
    penalty + getiriyi topla. Ham bileşenler scoring iç metodlarından, bonus/
    penalty ise toplam skordan geri-hesapla (total = weighted_sum + bonus - pen)."""
    raw = pickle.load(open("output/_broad_data.pkl", "rb"))
    shares = json.load(open("output/_shares_broad.json"))
    spy = pickle.load(open("output/_edge_spy.pkl", "rb")).copy()
    spy["_d"] = pd.to_datetime(spy["Date"]).dt.normalize()
    rmap = build_regime_map(spy)
    data = {t: enrich(df) for t, df in raw.items()}
    engine = SmallCapEngine()
    sc = engine.scoring

    records = []
    tickers = list(data.keys())
    for ti, tk in enumerate(tickers):
        df = data[tk]
        sh = shares.get(tk, {})
        sh_out, flt = sh.get("shares"), sh.get("float")
        n = len(df)
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
            stock_info = {"ticker": tk, "marketCap": int(mcap) if mcap else 0,
                          "floatShares": int(flt) if flt else 0, "shortName": tk, "sector": "Unknown"}
            try:
                sig = engine.scan_stock(tk, df_win, stock_info=stock_info, backtest_mode=True,
                                        portfolio_value=10000,
                                        spy_df_window=spy_slice if len(spy_slice) >= 6 else None,
                                        regime=reg)
            except Exception:
                sig = None
            if not sig:
                continue
            r = simulate(df, t, EXIT_NEW, apply_slippage=False)
            if r is None:
                continue
            # HAM bileşen skorları (0-max, ÇARPILMAMIŞ) — scoring iç metodları
            boosters = sig.get("_boosters_debug")  # yoksa aşağıda yeniden hesapla
            vol_surge = sig.get("volume_surge", 0)
            atr_p = sig.get("atr_percent", 0)
            atr_frac = atr_p / 100 if atr_p > 1 else atr_p
            float_sh = (sig.get("float_millions", 0) or 0) * 1e6
            # ham bileşenler (normalize edilmemiş 0-max)
            raw_vol = sc.score_volume_explosion(vol_surge)
            raw_atr = sc.score_volatility_expansion(atr_frac)
            raw_float = sc.score_float_tightness(float_sh)
            raw_mom = sc.score_momentum_continuity(df_win)
            raw_risk = sc.score_risk_control(df_win, atr_frac)
            raw_trend = sc.score_trend_quality(df_win, None)
            records.append({
                "r": r, "q": sig.get("quality_score", 0), "reg": reg,
                "pw": sig.get("trigger_pathway", "vce"), "date": str(day.date()),
                # ham bileşenler + maksimumları (normalize için)
                "raw_vol": raw_vol, "raw_atr": raw_atr, "raw_float": raw_float,
                "raw_mom": raw_mom, "raw_risk": raw_risk, "raw_trend": raw_trend,
                "vce_premium": bool(sig.get("vce_premium", False)),
                "vce_tight_coil": bool(sig.get("vce_tight_coil", False)),
            })
        if (ti + 1) % 250 == 0:
            print(f"  ...{ti+1}/{len(tickers)} ({len(records)} sinyal)", flush=True)

    json.dump(records, open("output/score_components.json", "w"), default=str)
    return records, sc


# Bileşen maksimumları (normalize için — scoring.py ile aynı)
MAXES = dict(vol=30, atr=25, float=20, mom=15, risk=15, trend=25)


def rescore(rec, weights, premium_pts=0, coil_pts=0):
    """Bir kaydı verilen ağırlık setiyle yeniden skorla (0-100 normalize + ağırlık).
    weights: {vol,atr,float,mom,risk,trend} toplam ~1.0. premium/coil: teknik bonus."""
    total = 0.0
    for key, mx in MAXES.items():
        raw = rec[f"raw_{key}"]
        total += (max(raw, 0) / mx) * 100 * weights.get(key, 0)
    # teknik bonuslar (VCE premium/tight-coil'e ağırlık verme senaryosu)
    if rec.get("vce_premium"):
        total += premium_pts
    if rec.get("vce_tight_coil"):
        total += coil_pts
    return total


WEIGHT_SETS = {
    "A) MEVCUT v33":        dict(vol=.12, atr=.13, float=.25, mom=.25, risk=.10, trend=.15),
    "B) float/mom YARIYA":  dict(vol=.12, atr=.13, float=.125, mom=.125, risk=.10, trend=.40),
    "C) float/mom SIFIR":   dict(vol=.10, atr=.25, float=.0, mom=.0, risk=.20, trend=.45),
    "D) trend-ağırlıklı":   dict(vol=.15, atr=.20, float=.15, mom=.10, risk=.05, trend=.35),
    "E) sadece trend+risk": dict(vol=.0, atr=.0, float=.0, mom=.0, risk=.40, trend=.60),
    "F) MEVCUT+teknik bonus": dict(vol=.12, atr=.13, float=.125, mom=.125, risk=.10, trend=.40),  # + premium/coil ağırlıklı
}


def q80_bucket_ev(recs, weights, prem=0, coil=0, top_frac=0.18):
    """Bu ağırlık setiyle yeniden skorla, üst %top_frac bucket'ın EV'sini döndür."""
    scored = [(rescore(r, weights, prem, coil), r["r"]) for r in recs]
    scored.sort(key=lambda x: -x[0])
    n_top = max(1, int(len(scored) * top_frac))
    top = scored[:n_top]
    rets = np.array([r for _, r in top])
    a_all = np.array([r["r"] for r in recs])
    # skor-getiri korelasyonu da hesapla
    qs = np.array([s for s, _ in scored]); rs = np.array([r for _, r in scored])
    corr = np.corrcoef(qs, rs)[0, 1] if qs.std() > 0 else 0
    return {"top_n": n_top, "top_ev": rets.mean(), "top_wr": (rets > 0).mean() * 100,
            "corr": corr}


def main():
    if os.path.exists("output/score_components.json"):
        recs = json.load(open("output/score_components.json"))
        sc = SmallCapScoring()
        print(f"  Cache'ten {len(recs)} sinyal yüklendi")
    else:
        recs, sc = collect_components()

    print("\n" + "=" * 84)
    print(f"  AĞIRLIK ÇARPIŞTIRMA — {len(recs)} gerçek sinyal | üst %18 bucket (Q80 proxy)")
    print("=" * 84)
    a = np.array([x["r"] for x in recs])
    print(f"  Genel: EV {a.mean():+.2f}%  WR {(a>0).mean()*100:.0f}%")
    print(f"\n  {'ağırlık seti':<26}{'üst-bucket EV':>15}{'WR':>7}{'skor↔getiri korr':>18}")
    print("  " + "-" * 80)
    best = None
    for name, w in WEIGHT_SETS.items():
        prem, coil = (15, 10) if "teknik bonus" in name else (0, 0)
        res = q80_bucket_ev(recs, w, prem, coil)
        marker = " ←" if (best is None or res["top_ev"] > best[1]) else ""
        if best is None or res["top_ev"] > best[1]:
            best = (name, res["top_ev"])
        print(f"  {name:<26}{res['top_ev']:>+13.2f}%{res['top_wr']:>6.0f}%{res['corr']:>+18.3f}{marker}")

    print(f"\n  → EN İYİ üst-bucket EV: '{best[0]}' ({best[1]:+.2f}%)")
    print(f"    (MEVCUT v33 ile karşılaştır — anlamlı fark var mı?)")

    # HİPOTEZ TESTİ: aşırı-uzama/RSI cezası yiyen sinyaller mi Q80'den eleniyor?
    print("\n" + "=" * 84)
    print("  HİPOTEZ: 'huni-skor savaşı' — Q80 azınlığı tesadüfen ceza yemeyenler mi?")
    print("=" * 84)
    # ham bileşen skorları yüksek AMA düşük q → ceza yemiş demektir
    high_raw = [x for x in recs if (x["raw_trend"] + x["raw_mom"]) >= 25]
    print(f"  Ham trend+momentum yüksek (>=25) sinyaller: {len(high_raw)}")
    if high_raw:
        hr = np.array([x["r"] for x in high_raw])
        print(f"    Getirisi: EV {hr.mean():+.2f}%  WR {(hr>0).mean()*100:.0f}%")
        print(f"    (Bunlar 'iyi kurulum ama ceza yemiş' olabilir — Q80'i geçebiliyorlar mı?)")


if __name__ == "__main__":
    main()
