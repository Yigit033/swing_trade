# -*- coding: utf-8 -*-
"""
GATE DENETİMİ — "Bu kadar kapıya gerçekten gerek var mı?"
================================================================================
Kullanıcının haklı itirazı: sistem katman katman büyümüş, hangisinin işe
yaradığı belli değil. Skor bileşenlerine uyguladığımız BIRAK-BİRİNİ-ÇIKAR
testinin aynısını KAPILARA uyguluyoruz.

YÖNTEM: her kapıyı tek tek DEVRE DIŞI bırak, gerçek motoru koştur, sonucu
mevcut (tüm kapılar açık) taban ile karşılaştır.
  - Kapıyı kaldırınca EV DÜŞÜYORSA → kapı işe yarıyor, KALIR
  - EV DEĞİŞMİYORSA (ve hiç sinyal eklemiyorsa) → kapı İNERT, silinebilir
  - EV ARTIYORSA → kapı ZARARLI, kaldırılmalı

Kapılar iki yolla kapatılır (ikisi de motoru DEĞİŞTİRMEDEN):
  a) Ayar eşiğini imkânsız değere çek (rsi, late_entry, distribution, stage, rr)
  b) check_boosters çıktısındaki bayrağı zorla (obv_distribution, swing_ready,
     trend_phase) — kapı o bayrağı okuyor, biz bayrağı nötrleyip kapıyı susturuyoruz

KARAR KURALI (önceden yazıldı): bir kapı ancak
  (a) kaldırılınca EV düşmüyor VE (b) TRAIN+OOS'ta aynı yön VE (c) eklediği
  sinyal ya 0 ya da pozitif EV'li ise SİLİNMEYE aday olur. Tek koşul bile
  sağlanmazsa kapı KALIR — şüphede kapıyı koru (asimetrik risk: yanlış silinen
  bir kapı gerçek para kaybettirir, gereksiz duran bir kapı sadece sinyal azaltır).
"""
import sys, os, json, pickle, copy
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
SPAN = 21.0
CACHE = "output/gate_value.json"


# ── Kapı kapatıcıları ────────────────────────────────────────────────────
def _off_rsi(s, b):
    s.max_entry_rsi = 1000

def _off_late_entry(s, b):
    s.scan_gates.late_entry_five_day_total_gt = 1e9

def _off_distribution(s, b):
    s.scan_gates.distribution_day_min_vol = 1e9

def _off_stage(s, b):
    s.scan_gates.reject_stage3 = False
    s.scan_gates.reject_stage4 = False

def _off_rr(s, b):
    s.min_rr_at_entry = 0.0
    s.min_rr_type_c = 0.0

def _off_parabolic(s, b):
    s.scan_gates.parabolic_five_day_return_gt = 1e9
    s.scan_gates.extreme_five_day_return_gt = 1e9
    s.scan_gates.extreme_rsi_gt = 1e9

def _off_obv(s, b):
    b["obv_distribution"] = False

def _off_swing_ready(s, b):
    b["swing_ready"] = True

def _off_overext(s, b):
    s.signal_confirmation.overext_today_change_max = 1e9
    s.signal_confirmation.overext_single_day_max = 1e9
    s.signal_confirmation.overext_five_day_total_max = 1e9


GATES = [
    ("TABAN (tüm kapılar açık)", None),
    ("− RSI kapısı (>70)", _off_rsi),
    ("− Geç giriş (5g>%30 & RSI>65)", _off_late_entry),
    ("− Dağıtım günü", _off_distribution),
    ("− Weinstein Stage 3/4", _off_stage),
    ("− R:R ≥1.8", _off_rr),
    ("− Parabolik/ekstrem", _off_parabolic),
    ("− OBV dağıtım", _off_obv),
    ("− Swing onayı", _off_swing_ready),
    ("− Aşırı uzama", _off_overext),
]


def stats(rows):
    if not rows:
        return dict(n=0, ev=0.0, wr=0.0, pf=0.0)
    a = np.array([r["r"] for r in rows])
    w, l = a[a > 0], a[a <= 0]
    pf = (w.sum() / abs(l.sum())) if l.size and l.sum() != 0 else float("inf")
    return dict(n=len(a), ev=float(a.mean()), wr=float((a > 0).mean() * 100), pf=float(pf))


def fmt(s):
    if s["n"] == 0:
        return "n=0"
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return f"n={s['n']:<4} EV {s['ev']:+6.2f}%  WR {s['wr']:3.0f}%  PF {pf:>5}"


def run(data, shares, rmap, spy, disabler, label):
    """Taze motor kur (ayarlar izole olsun), kapıyı kapat, koştur."""
    engine = SmallCapEngine()
    booster_override = {}
    if disabler is not None:
        disabler(engine.settings, booster_override)

    if booster_override:
        sig = engine.signals
        orig = sig.check_boosters

        def patched(df, _o=orig, _ov=dict(booster_override)):
            b = _o(df)
            b.update(_ov)
            return b

        sig.check_boosters = patched

    out = []
    for tk, df in data.items():
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
            try:
                s = engine.scan_stock(
                    tk, df.iloc[:t + 1], stock_info=info, backtest_mode=True,
                    portfolio_value=10000,
                    spy_df_window=spy_slice if len(spy_slice) >= 6 else None,
                    regime=reg)
            except Exception:
                s = None
            if not s:
                continue
            r = simulate(df, t, EXIT_NEW)
            if r is None:
                continue
            out.append({"key": f"{tk}|{day.date()}", "date": str(day.date()),
                        "r": float(r), "q": float(s.get("quality_score", 0) or 0)})
    return out


def main():
    if os.path.exists(CACHE):
        results = json.load(open(CACHE))
        print(f"Cache'ten {len(results)} varyant yuklendi")
    else:
        print("Veri yukleniyor...", flush=True)
        raw = pickle.load(open("output/_broad_data.pkl", "rb"))
        shares = json.load(open("output/_shares_broad.json"))
        spy = pickle.load(open("output/_edge_spy.pkl", "rb")).copy()
        spy["_d"] = pd.to_datetime(spy["Date"]).dt.normalize()
        rmap = build_regime_map(spy)
        data = {t: enrich(df) for t, df in raw.items()}
        results = {}
        for label, fn in GATES:
            print(f">>> {label} ...", flush=True)
            rows = run(data, shares, rmap, spy, fn, label)
            results[label] = rows
            print(f"    {len(rows)} sinyal ({len([r for r in rows if r['q']>=Q_LIVE])} Q80+)", flush=True)
        json.dump(results, open(CACHE, "w"), default=str)

    base_rows = [r for r in results["TABAN (tüm kapılar açık)"] if r["q"] >= Q_LIVE]
    base = stats(base_rows)
    base_keys = {r["key"] for r in base_rows}
    b_tr = stats([r for r in base_rows if r["date"] < OOS_SPLIT])
    b_te = stats([r for r in base_rows if r["date"] >= OOS_SPLIT])

    W = 116
    print("\n" + "=" * W)
    print(f"  GATE DENETİMİ — Q{Q_LIVE:.0f}+ sinyaller | kapıyı KALDIRINCA ne oluyor?")
    print("=" * W)
    print(f"  {'kapı':<32}{'sonuç':<40}{'Δ EV':>8}{'EK sinyal':>12}{'EK EV':>9}   karar")
    print("  " + "-" * (W - 4))
    verdicts = []
    for label, _ in GATES:
        rows = [r for r in results[label] if r["q"] >= Q_LIVE]
        s = stats(rows)
        if label.startswith("TABAN"):
            print(f"  {label:<32}{fmt(s):<40}{'—':>8}{'—':>12}{'—':>9}   —")
            continue
        d = s["ev"] - base["ev"]
        extra = [r for r in rows if r["key"] not in base_keys]
        se = stats(extra)
        tr = stats([r for r in rows if r["date"] < OOS_SPLIT])
        te = stats([r for r in rows if r["date"] >= OOS_SPLIT])
        d_tr, d_te = tr["ev"] - b_tr["ev"], te["ev"] - b_te["ev"]

        if se["n"] == 0 and abs(d) < 0.01:
            verdict = "İNERT — hiç etkisi yok"
        elif d < -0.3:
            verdict = "KALIR — kaldırınca EV düşüyor"
        elif d > 0.3 and d_tr > 0 and d_te > 0:
            verdict = "ZARARLI — kaldırılmalı"
        elif d > 0.3:
            verdict = f"belirsiz (TRAIN {d_tr:+.2f} / OOS {d_te:+.2f})"
        else:
            verdict = "nötr — EV'yi değiştirmiyor"
        verdicts.append((label, s, d, se, d_tr, d_te, verdict))
        ex_n = f"+{se['n']}" if se["n"] else "0"
        ex_ev = f"{se['ev']:+.2f}%" if se["n"] else "—"
        print(f"  {label:<32}{fmt(s):<40}{d:>+8.2f}{ex_n:>12}{ex_ev:>9}   {verdict}")

    print("\n" + "=" * W)
    print("  ÖZET — silinmeye aday kapılar (şüphede kapıyı KORU ilkesi)")
    print("=" * W)
    inert = [v for v in verdicts if v[6].startswith("İNERT")]
    harmful = [v for v in verdicts if v[6].startswith("ZARARLI")]
    keep = [v for v in verdicts if v[6].startswith("KALIR")]
    neutral = [v for v in verdicts if v[6].startswith("nötr") or v[6].startswith("belirsiz")]
    for title, group in (("İNERT (güvenle silinebilir)", inert),
                         ("ZARARLI (kaldırılmalı)", harmful),
                         ("İŞE YARIYOR (kalır)", keep),
                         ("NÖTR/BELİRSİZ (kalır — şüphede koru)", neutral)):
        print(f"\n  {title}: {len(group)}")
        for v in group:
            print(f"    · {v[0]:<32} ΔEV {v[2]:+.2f}  (TRAIN {v[4]:+.2f} / OOS {v[5]:+.2f})")


if __name__ == "__main__":
    main()
