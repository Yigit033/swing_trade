# -*- coding: utf-8 -*-
"""
TAVAN DIŞI SKOR EKLEMELERİ — +8 premium-VCE ve +5 tight-coil işe yarıyor mu?
================================================================================
2026-08-05 skor denetiminde 14 bonus + 21 ceza ölçüldü. Ama iki ekleme
`calculate_quality_score` DIŞINDA, engine.py'de skora doğrudan biniyor:

    engine.py:564   if vce_metrics['is_premium']:      quality_score += 8
    engine.py:575   if 0 < squeeze_ratio < 0.65:       quality_score += 5

Bunlar bonus tavanına TABİ DEĞİL — yani gerçekten skoru kaydırıyorlar ve
Q80 seçimini değiştirebiliyorlar. Kurulum edge'i ölçülmüştü (premium R10
+4.84% vs +2.42%; tight-coil OOS +1.92% vs +1.31%) ama +8/+5'in SEÇİME
etkisi hiç ölçülmedi. Ölçülmemiş bir eşik kaydırıcısı tam olarak katalizör
bonuslarının yaptığı hatadır (eşiği sessizce 6 puan kaydırıyordu).

YÖNTEM: motoru canlı-replika evrende koşturup her sinyalin skorunu VE
`trigger_details.vce_metrics`'ini (is_premium, squeeze_ratio) kaydediyoruz.
Sonra eklemeyi geri alıp (q − 8·premium − 5·coil) Q80 seçimini yeniden
yapıyoruz. TRAIN/OOS ayrı.

DİKKAT — burada bir tuzağa düştük, kayda geçsin: ilk deneme bayrakları
`output/signal_lab.json`'daki `vce_premium` / `vce_tight_coil` alanlarından
okudu ve "ikisi de hiç ateşlemiyor" çıktı. Yanlıştı: engine.py bu bayrakları
`boosters`'a yazıyor ama SİNYAL SÖZLÜĞÜNE hiç koymuyor, dolayısıyla
`sig.get("vce_premium")` her zaman False dönüyordu. Bayrak yokluğunu
"özellik ateşlemiyor" diye okumak YANLIŞ ATIF. Doğru kaynak vce_metrics.

KARAR KURALI (önceden yazıldı):
  Çıkarınca EV DÜŞÜYOR (TRAIN+OOS aynı yön) → KAL (işe yarıyor)
  Çıkarınca EV ARTIYOR (TRAIN+OOS aynı yön) → SİL (zararlı)
  Seçim hiç değişmiyor                      → SİL (etkisiz, sadece karmaşa)
  Yön tutarsız                              → KAL (kanıt yok, dokunma)
"""
import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.CRITICAL)

OOS_SPLIT = "2025-06-01"
Q_LIVE = 80.0
PREMIUM_PTS = 8.0
COIL_PTS = 5.0
COIL_RATIO_MAX = 0.65
PRICE_MAX = 200.0
CACHE = "output/post_cap_bonuses.json"


def collect():
    """Motoru koştur; skoru ve vce_metrics'i (asıl kaynak) kaydet."""
    from swing_trader.small_cap.engine import SmallCapEngine
    from backtest_live_replica import build_regime_map, enrich, simulate, EXIT_NEW
    from collect_signal_lab import finviz_hit

    print("Veri yukleniyor...", flush=True)
    raw = pickle.load(open("output/_broad_data.pkl", "rb"))
    shares = json.load(open("output/_shares_broad.json"))
    spy = pickle.load(open("output/_edge_spy.pkl", "rb")).copy()
    spy["_d"] = pd.to_datetime(spy["Date"]).dt.normalize()
    rmap = build_regime_map(spy)

    engine = SmallCapEngine()
    recs = []
    tickers = list(raw.keys())
    for ti, tk in enumerate(tickers):
        df = enrich(raw[tk])
        sh = shares.get(tk, {})
        so, fl = sh.get("shares"), sh.get("float")
        for t in range(60, len(df) - 21):
            row = df.iloc[t]
            close = float(row["Close"])
            if close > PRICE_MAX:
                continue
            mcap = close * so if so else None
            if not finviz_hit(row, mcap):
                continue
            day = pd.to_datetime(df["Date"].iloc[t]).normalize()
            sl = spy[spy["_d"] <= day].tail(60)
            try:
                s = engine.scan_stock(
                    tk, df.iloc[:t + 1],
                    stock_info={"ticker": tk, "marketCap": int(mcap) if mcap else 0,
                                "floatShares": int(fl) if fl else 0,
                                "shortName": tk, "sector": "Unknown"},
                    backtest_mode=True, portfolio_value=10000,
                    spy_df_window=sl if len(sl) >= 6 else None,
                    regime=rmap.get(day, "UNKNOWN"))
            except Exception:
                s = None
            if not s:
                continue
            r = simulate(df, t, EXIT_NEW)
            if r is None:
                continue
            vm = (s.get("trigger_details") or {}).get("vce_metrics", {}) or {}
            recs.append({
                "tk": tk, "date": str(day.date()), "r": float(r),
                "q": float(s.get("quality_score", 0) or 0),
                "is_premium": bool(vm.get("is_premium", False)),
                "squeeze_ratio": float(vm.get("squeeze_ratio", 0) or 0),
                "pathway": s.get("trigger_pathway", ""),
            })
        if (ti + 1) % 200 == 0:
            print(f"  ...{ti+1}/{len(tickers)} ({len(recs)} sinyal)", flush=True)

    json.dump(recs, open(CACHE, "w"))
    return recs


def stats(rows):
    if not rows:
        return dict(n=0, ev=0.0, wr=0.0)
    a = np.array([r["r"] for r in rows], dtype=float)
    return dict(n=len(a), ev=float(a.mean()), wr=float((a > 0).mean() * 100))


def fmt(s):
    return f"n={s['n']:<3} EV {s['ev']:+6.2f}% WR {s['wr']:3.0f}%" if s["n"] else "n=0"


def main():
    recs = json.load(open(CACHE)) if os.path.exists(CACHE) else collect()
    for r in recs:
        r["prem"] = bool(r.get("is_premium"))
        sq = float(r.get("squeeze_ratio", 0) or 0)
        r["coil"] = bool(0 < sq < COIL_RATIO_MAX)
        r["q_raw"] = float(r["q"]) - (PREMIUM_PTS if r["prem"] else 0) - (COIL_PTS if r["coil"] else 0)

    n_prem = sum(r["prem"] for r in recs)
    n_coil = sum(r["coil"] for r in recs)
    print("=" * 96)
    print(f"  TAVAN DIŞI EKLEMELER — {len(recs)} kayıt | premium {n_prem} | tight-coil {n_coil}")
    print("=" * 96)

    if not n_prem and not n_coil:
        print("  İkisi de hiç ateşlemiyor → ikisi de SİL.")
        return

    # Bayrak taşıyanların ham getirisi (kurulum edge'i hâlâ geçerli mi?)
    print("\n  KURULUM EDGE'İ (tüm kayıtlar, eşik öncesi)")
    for name, key in (("premium-VCE", "prem"), ("tight-coil", "coil")):
        yes = [r for r in recs if r[key]]
        no = [r for r in recs if not r[key]]
        print(f"    {name:<12} VAR  {fmt(stats(yes))}   YOK  {fmt(stats(no))}")

    # Karar testi: eklemeyi kaldır, Q80 seçimini yeniden yap
    print(f"\n  KARAR TESTİ — Q{Q_LIVE:.0f} seçimi")
    variants = {
        "MEVCUT (+8 premium, +5 coil)": lambda r: r["q"],
        "premium YOK (+0, +5)": lambda r: r["q_raw"] + (COIL_PTS if r["coil"] else 0),
        "coil YOK (+8, +0)": lambda r: r["q_raw"] + (PREMIUM_PTS if r["prem"] else 0),
        "İKİSİ DE YOK": lambda r: r["q_raw"],
    }
    base = None
    for label, score_fn in variants.items():
        sel = [r for r in recs if score_fn(r) >= Q_LIVE]
        s = stats(sel)
        tr = stats([r for r in sel if r["date"] < OOS_SPLIT])
        te = stats([r for r in sel if r["date"] >= OOS_SPLIT])
        if base is None:
            base = (s, tr, te)
            print(f"    {label:<30} {fmt(s)} | TR {fmt(tr)} | OOS {fmt(te)}")
            continue
        b, btr, bte = base
        d = s["ev"] - b["ev"]
        dtr = tr["ev"] - btr["ev"] if tr["n"] else 0.0
        dte = te["ev"] - bte["ev"] if te["n"] else 0.0
        print(f"    {label:<30} {fmt(s)} | TR {fmt(tr)} | OOS {fmt(te)}")
        print(f"    {'':<30} ΔEV {d:+.2f} (TR {dtr:+.2f} / OOS {dte:+.2f})  Δn {s['n']-b['n']:+d}")

    # Eklemenin gerçekten eşiği geçirdiği sinyaller — asıl soru bunlar kazandı mı?
    print("\n  EKLEME SAYESİNDE Q80'İ GEÇENLER (asıl soru: bunlar kazandı mı?)")
    for name, pts, key in (("premium +8", PREMIUM_PTS, "prem"), ("tight-coil +5", COIL_PTS, "coil")):
        crossed = [r for r in recs if r[key] and r["q"] >= Q_LIVE and (r["q"] - pts) < Q_LIVE]
        s = stats(crossed)
        print(f"    {name:<15} {fmt(s)}" + (f"   {[r['tk'] + ' ' + format(r['r'], '+.1f') + '%' for r in crossed]}" if crossed else ""))

    # ── ÇELDİRİCİ KONTROLÜ ────────────────────────────────────────────────
    # "Eklemeyi kaldırınca EV arttı" TEK BAŞINA kanıt DEĞİL: kaldırmak barajı da
    # yükseltiyor ve HERHANGİ bir baraj yükseltmesi EV'yi artırır (Q80→Q88 de
    # artırır). Doğru soru: aynı sinyal SAYISINA düz eşik yükseltmesiyle inince
    # EV ne oluyor? Ekleme ancak düz eşikten DAHA KÖTÜ ise gerçekten zararlıdır.
    print("\n  ÇELDİRİCİ KONTROLÜ — aynı n'de düz eşik yükseltmesi ne veriyor?")

    def sel_at_n(score_fn, target_n):
        """score_fn'e göre en yüksek target_n sinyali seç (eşiği yukarı kaydırmaya eşdeğer)."""
        ranked = sorted(recs, key=lambda r: -score_fn(r))
        return ranked[:target_n]

    cur = lambda r: r["q"]
    for label, variant_fn in (("premium YOK", lambda r: r["q"] - (PREMIUM_PTS if r["prem"] else 0)),
                              ("İKİSİ DE YOK", lambda r: r["q_raw"])):
        v_sel = [r for r in recs if variant_fn(r) >= Q_LIVE]
        n = len(v_sel)
        v, flat = stats(v_sel), stats(sel_at_n(cur, n))
        print(f"    n={n:<4} {label:<14} {fmt(v)}")
        print(f"    {'':<5} {'düz eşik (Q↑)':<14} {fmt(flat)}   fark {v['ev']-flat['ev']:+.2f}")
        if v["ev"] - flat["ev"] < -0.3:
            print(f"    {'':<5} -> Ekleme düz eşikten İYİ: mis-rank etmiyor, KALSIN")
        elif v["ev"] - flat["ev"] > 0.3:
            print(f"    {'':<5} -> Ekleme düz eşikten KÖTÜ: gerçekten mis-rank ediyor, SİL")
        else:
            print(f"    {'':<5} -> Fark yok: etki sadece baraj kaydırması, ayırt etmiyor")

    # ── ASIL SORU: bonus SIRALAMA özelliği mi, EŞİK özelliği mi? ──────────
    # Çeldirici kontrolü gösterdi ki bonuslu skor İYİ sıralıyor. Ama Q80
    # karşılaştırmasında da kullanılınca 34 sinyali barajın üstüne TAŞIYOR ve
    # onların EV'si +0.89% (taban +4.19%). Yani doğru bilgi YANLIŞ yerde
    # kullanılıyor. Ayrılabilir mi: barajı HAM skora uygula, sıralamayı
    # bonuslu skorla yap?
    print("\n  AYRIŞTIRMA — baraj ham skora, sıralama bonuslu skora")
    print(f"    {'kurgu':<34}{'n':>5}{'EV':>9}{'WR':>6}   {'TRAIN':>9}{'OOS':>9}")
    setups = {
        "MEVCUT: baraj+sıralama bonuslu": [r for r in recs if r["q"] >= Q_LIVE],
        "baraj HAM, sıralama bonuslu": [r for r in recs if r["q_raw"] >= Q_LIVE],
    }
    for label, sel in setups.items():
        s, tr, te = (stats(sel),
                     stats([r for r in sel if r["date"] < OOS_SPLIT]),
                     stats([r for r in sel if r["date"] >= OOS_SPLIT]))
        print(f"    {label:<34}{s['n']:>5}{s['ev']:>+8.2f}%{s['wr']:>5.0f}%   "
              f"{tr['ev']:>+8.2f}%{te['ev']:>+8.2f}%")

    # Eşik eğrisi — barajın kendisi doğru yerde mi? (bonuslar dahil skorla)
    print("\n  EŞİK EĞRİSİ (mevcut skor, bonuslar dahil)")
    print(f"    {'eşik':<8}{'n':>5}{'ay/sinyal':>11}{'EV':>9}{'WR':>6}   {'TRAIN':>9}{'OOS':>9}")
    months = 21.0
    for q in (78, 80, 82, 84, 86, 88, 90):
        sel = [r for r in recs if r["q"] >= q]
        if not sel:
            continue
        s, tr, te = (stats(sel),
                     stats([r for r in sel if r["date"] < OOS_SPLIT]),
                     stats([r for r in sel if r["date"] >= OOS_SPLIT]))
        print(f"    Q{q:<7}{s['n']:>5}{s['n']/months:>10.1f}{s['ev']:>+9.2f}%{s['wr']:>5.0f}%   "
              f"{tr['ev']:>+8.2f}%{te['ev']:>+8.2f}%")


if __name__ == "__main__":
    main()
