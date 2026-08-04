# -*- coding: utf-8 -*-
"""
SKOR BONUS/CEZA DENETİMİ — 34 değiştiricinin her biri işe yarıyor mu?
================================================================================
Kalite skorunun 6 ANA BİLEŞENİ ölçülmüştü (measure_weight_reallocation.py +
analyze_signal_lab.py: hiçbiri çıkarılamaz). Ama skorun üstüne binen ~17 bonus
ve ~21 ceza HİÇ ölçülmemişti. 2026-08-04'te öğrendik ki ölçülmemiş bir skor
bileşeni gerçek para kaybettirir: katalizör bonusları eşiği sessizce 6 puan
kaydırıyordu ve silindiler.

TASARIM (38 ayrı motor koşusu yerine 1 koşu):
Skorlama saf hesaptır — I/O yok, ucuz. Bu yüzden motoru BİR KEZ koşturup, her
sinyal skorlanırken o anda 34 varyantı da hesaplıyoruz: ilgili ayarı 0'a çekip
skoru yeniden hesapla, fark = o bileşenin O SİNYALDEKİ katkısı.

Sonra tamamı OFFLINE analiz edilir:
  1. Bileşen kaç sinyalde ateşledi? (hiç ateşlemiyorsa → SİL)
  2. Ateşlediği sinyallerin getirisi, ateşlemediklerinden farklı mı?
  3. KARAR TESTİ: bileşeni skordan çıkarınca Q80 seçimi değişir mi, EV artar mı?
     (gate denetimindeki bırak-birini-çıkar mantığının aynısı)

Not: bonus toplamı `bonus_cap` ile sınırlı, ceza sınırsız. Tek bir bileşeni
sıfırlamak cap bağlıyken toplamı değiştirmeyebilir — bu ZATEN doğru atıf:
bizi ilgilendiren marjinal etki.

KARAR KURALI (önceden yazıldı):
  Hiç ateşlemiyor                        → SİL
  Çıkarınca Q80 EV'si ARTIYOR (TRAIN+OOS) → SİL (zararlı)
  Çıkarınca EV DÜŞÜYOR (TRAIN+OOS)        → KAL (işe yarıyor)
  Yön tutarsız / n<5                      → KAL (kanıt yok)

================================================================================
SONUÇ (2026-08-05, 95 sinyal / 21 ay) — uygulandı
================================================================================
BONUSLAR (14): hepsinin marjinal etkisi tam 0.00. Sebep bir tuzak: bonus_cap
  sinyallerin %100'ünde bağlıyor (ham toplam ~60 vs tavan 30), yani tek bonusu
  sıfırlamak toplamı değiştirmiyor. "Hiç ateşlemiyor" diye yorumlamak YANLIŞ
  ATIF olurdu — doğru yorum: 14 koşulun net çıktısı herkese sabit +30, ayırt
  etme gücü sıfır. Kod tek satıra indi (bonus = st.bonus_cap), davranış BİREBİR.

CEZALAR (21): 10'u hiç ateşlemedi (girdilerini VCE/RVOL tetiği garanti ediyor),
  6'sı ateşledi ama cezaladığı sinyaller ortalamanın 2-3 katı kazandı (yön ters:
  5g>25% → +13.24%, 5g>40% → +20.36%, tek gün>25% → +21.12%) → silindi.
  KALAN 5: pen_a_rsi_gt_70/65, pen_c_rsi_gt_65/60, pen_today_gt_10.

pen_spread_risk: iki koşulu (dvol<7M VE ATR>%8) 21 ayda hiç birlikte olmadı.
  Silmeden önce saf ATR>%8 cezası ölçüldü: 6-8% bandı TRAIN +9.09% / OOS −9.63%
  (n=16, işaret değişiyor) → eğri uydurma olurdu, tavan EKLENMEDİ.

PARİTE: aynı 95 bar eski/yeni kodla skorlandı → 84 aynı, 11 arttı, 0 azaldı;
  Q80 seçimi birebir (n=78, EV +3.11%, WR 56%), sıralama birebir.
Kilit test: swing_trader/tests/test_score_modifiers_measured.py
"""
import sys, os, json, pickle, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
logging.basicConfig(level=logging.ERROR)
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd

from swing_trader.small_cap.engine import SmallCapEngine
from backtest_live_replica import enrich, simulate, EXIT_NEW, build_regime_map
from collect_signal_lab import finviz_hit

OOS_SPLIT = "2025-06-01"
Q_LIVE = 80.0
PRICE_MAX = 200.0
CACHE = "output/score_modifiers.json"

# Eşik/bant tanımlayan alanlar ATLANIR — onları 0'a çekmek "bileşeni kaldırmak"
# değil, "bandı değiştirmek" olur (yanlış atıf). Yalnız PUAN alanları ölçülür.
SKIP = {
    "bonus_early_entry_lo", "bonus_early_entry_hi", "bonus_very_early_hi",
}


def component_names():
    src = open("swing_trader/small_cap/scoring.py", encoding="utf-8").read()
    names = sorted(set(re.findall(r"st\.((?:bonus|pen)_\w+)", src)))
    return [n for n in names if n not in SKIP and n != "bonus_cap"]


def stats(rows, key="r"):
    if not rows:
        return dict(n=0, ev=0.0, wr=0.0)
    a = np.array([r[key] for r in rows])
    return dict(n=len(a), ev=float(a.mean()), wr=float((a > 0).mean() * 100))


def collect():
    print("Veri yukleniyor...", flush=True)
    raw = pickle.load(open("output/_broad_data.pkl", "rb"))
    shares = json.load(open("output/_shares_broad.json"))
    spy = pickle.load(open("output/_edge_spy.pkl", "rb")).copy()
    spy["_d"] = pd.to_datetime(spy["Date"]).dt.normalize()
    rmap = build_regime_map(spy)
    data = {t: enrich(df) for t, df in raw.items()}

    engine = SmallCapEngine()
    sc = engine.scoring
    st = sc._settings if hasattr(sc, "_settings") else None
    # scoring ayar nesnesini bul (scoring_tuning)
    tuning = engine.settings.scoring_tuning
    comps = component_names()
    print(f"  {len(comps)} bilesen olculecek", flush=True)

    orig = sc.calculate_quality_score
    pending = {}

    def patched(df, volume_surge, atr_percent, float_shares, boosters=None):
        base = orig(df, volume_surge, atr_percent, float_shares, boosters)
        deltas = {}
        for attr in comps:
            old = getattr(tuning, attr)
            try:
                setattr(tuning, attr, 0)
            except Exception:
                continue
            try:
                without = orig(df, volume_surge, atr_percent, float_shares, boosters)
                deltas[attr] = round(base - without, 3)
            finally:
                setattr(tuning, attr, old)
        pending["deltas"] = deltas
        return base

    sc.calculate_quality_score = patched

    recs = []
    tickers = list(data.keys())
    for ti, tk in enumerate(tickers):
        df = data[tk]
        sh = shares.get(tk, {})
        sh_out, flt = sh.get("shares"), sh.get("float")
        n = len(df)
        for t in range(60, n - 21):
            row = df.iloc[t]
            close = float(row["Close"])
            if close > PRICE_MAX:
                continue
            mcap = close * sh_out if sh_out else None
            if not finviz_hit(row, mcap):
                continue
            day = pd.to_datetime(df["Date"].iloc[t]).normalize()
            reg = rmap.get(day, "UNKNOWN")
            spy_slice = spy[spy["_d"] <= day].tail(60)
            info = {"ticker": tk, "marketCap": int(mcap) if mcap else 0,
                    "floatShares": int(flt) if flt else 0,
                    "shortName": tk, "sector": "Unknown"}
            pending.clear()
            try:
                s = engine.scan_stock(
                    tk, df.iloc[:t + 1], stock_info=info, backtest_mode=True,
                    portfolio_value=10000,
                    spy_df_window=spy_slice if len(spy_slice) >= 6 else None,
                    regime=reg)
            except Exception:
                s = None
            if not s or "deltas" not in pending:
                continue
            r = simulate(df, t, EXIT_NEW)
            if r is None:
                continue
            recs.append({
                "tk": tk, "date": str(day.date()), "r": float(r),
                "q": float(s.get("quality_score", 0) or 0),
                "d": pending["deltas"],
            })
        if (ti + 1) % 200 == 0:
            print(f"  ...{ti+1}/{len(tickers)} ({len(recs)} sinyal)", flush=True)

    sc.calculate_quality_score = orig
    json.dump(recs, open(CACHE, "w"), default=str)
    return recs


def main():
    recs = json.load(open(CACHE)) if os.path.exists(CACHE) else collect()
    q80 = [r for r in recs if r["q"] >= Q_LIVE]
    base = stats(q80)
    b_tr = stats([r for r in q80 if r["date"] < OOS_SPLIT])
    b_te = stats([r for r in q80 if r["date"] >= OOS_SPLIT])
    comps = component_names()

    W = 118
    print("\n" + "=" * W)
    print(f"  SKOR DEĞİŞTİRİCİ DENETİMİ — {len(recs)} sinyal | Q{Q_LIVE:.0f}+ taban: "
          f"n={base['n']} EV {base['ev']:+.2f}% WR {base['wr']:.0f}%")
    print("=" * W)
    print(f"  {'bileşen':<26}{'ateşleme':>9}{'ateşleyenler':>22}{'ateşlemeyenler':>22}"
          f"{'ΔEV (çıkarınca)':>17}   karar")
    print("  " + "-" * (W - 4))

    rows_out = []
    for attr in comps:
        fired_all = [r for r in recs if abs(r["d"].get(attr, 0)) > 1e-9]
        # Karar testi: bileşeni skordan çıkar, Q80 seçimini yeniden yap
        without = [r for r in recs if (r["q"] - r["d"].get(attr, 0)) >= Q_LIVE]
        s_wo = stats(without)
        d_ev = s_wo["ev"] - base["ev"] if s_wo["n"] else 0.0
        tr_wo = stats([r for r in without if r["date"] < OOS_SPLIT])
        te_wo = stats([r for r in without if r["date"] >= OOS_SPLIT])
        d_tr = tr_wo["ev"] - b_tr["ev"] if tr_wo["n"] else 0.0
        d_te = te_wo["ev"] - b_te["ev"] if te_wo["n"] else 0.0

        f_q80 = [r for r in q80 if abs(r["d"].get(attr, 0)) > 1e-9]
        nf_q80 = [r for r in q80 if abs(r["d"].get(attr, 0)) <= 1e-9]
        sf, snf = stats(f_q80), stats(nf_q80)

        if not fired_all:
            verdict = "SİL — hiç ateşlemiyor"
        elif len(f_q80) < 5:
            verdict = f"kanıt yok (Q80'de n={len(f_q80)})"
        elif d_ev > 0.3 and d_tr > 0 and d_te > 0:
            verdict = "SİL — çıkarınca EV artıyor"
        elif d_ev < -0.3 and d_tr < 0 and d_te < 0:
            verdict = "KAL — çıkarınca EV düşüyor"
        elif abs(d_ev) <= 0.3:
            verdict = "nötr — seçimi değiştirmiyor"
        else:
            verdict = f"yön tutarsız (TR {d_tr:+.1f}/OOS {d_te:+.1f})"

        rows_out.append((attr, len(fired_all), sf, snf, d_ev, d_tr, d_te, verdict))
        f_s = f"n={sf['n']:<3} EV {sf['ev']:+6.2f}%" if sf["n"] else "—"
        nf_s = f"n={snf['n']:<3} EV {snf['ev']:+6.2f}%" if snf["n"] else "—"
        print(f"  {attr:<26}{len(fired_all):>9}{f_s:>22}{nf_s:>22}{d_ev:>+17.2f}   {verdict}")

    print("\n" + "=" * W)
    print("  ÖZET")
    print("=" * W)
    groups = {}
    for r in rows_out:
        key = r[7].split(" —")[0].split(" (")[0]
        groups.setdefault(key, []).append(r)
    for key in ("SİL", "KAL", "nötr", "kanıt yok", "yön tutarsız"):
        g = [v for k, v in groups.items() if k.startswith(key) for v in v]
        if not g:
            continue
        print(f"\n  {key}: {len(g)}")
        for attr, nf, sf, snf, d_ev, d_tr, d_te, verdict in sorted(g, key=lambda x: -abs(x[4])):
            print(f"    · {attr:<26} ateşleme {nf:<5} ΔEV {d_ev:+6.2f} "
                  f"(TR {d_tr:+5.2f} / OOS {d_te:+5.2f})")


if __name__ == "__main__":
    main()
