# -*- coding: utf-8 -*-
"""
KARŞILAŞTIRMALI BACKTEST — ESKİ SİSTEM vs YENİ SİSTEM
=====================================================
Kullanıcının sorusu: "Bu ürün gerçekten para kazandırma canavarına dönüştü mü?
Yaptığımız her değişikliğin toplam farkı ne?"

Aynı geniş & tarafsız evren (S&P 400+600, ~1000 ticker, batmışlar/düşenler
dahil → hayatta-kalan yanılgısı azaltılmış), aynı dönem (2024-06→2026-05),
iki sistem YAN YANA:

  ESKİ:  sinyal = yalnız VCE (Variant B)
         exit   = stop 1.5 ATR, cap %8-12, T2 cap %28, hold 10, dar trail
  YENİ:  sinyal = VCE + RVOL thrust (2. pathway)
         exit   = stop 2.5 ATR, cap %14-18, T2 cap YOK, hold 20, chandelier trail

Her ikisi: t+1 açılış girişi (lookahead yok), gerçekçi stop dolumu (gap-down'da
açılıştan), bar-bar simülasyon. Sinyal tanımları signals.py / discover
harness'larıyla tutarlı.

Metrikler (portföy değil, PER-TRADE — saf sinyal+exit edge'i):
  - İşlem sayısı, EV/trade, WR, ort kazanç/kayıp, medyan, en kötü %5 (kuyruk),
    toplam getiri (ardışık compound değil — bağımsız trade toplamı), Sharpe~
  - Rejim kırılımı (BULL/CAUTION/BEAR) + OOS split (2025-06)

Cache: output/_broad_data.pkl (scripts/_fetch_broad_data.py) + _edge_spy.pkl
"""
import sys, os, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
logging.basicConfig(level=logging.ERROR)
import numpy as np
import pandas as pd

from swing_trader.small_cap.regime_logic import regime_from_spy_close

SPLIT = pd.Timestamp("2025-06-01")
MIN_MCAP_BARS = 55  # VCE ihtiyacı


# ══════════════════════════════════════════════════════════════════════
# İNDİKATÖRLER + SİNYAL TANIMLARI (canlı sistemle birebir)
# ══════════════════════════════════════════════════════════════════════
def enrich(df):
    df = df.copy()
    c = df["Close"].astype(float); h = df["High"].astype(float); l = df["Low"].astype(float)
    v = df["Volume"].astype(float)
    df["ma20"] = c.rolling(20).mean(); df["ma50"] = c.rolling(50).mean()
    df["hi20"] = h.rolling(20).max()
    df["vol50"] = v.rolling(50).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean(); df["atr_pct"] = df["atr"] / c * 100
    return df


def sig_vce(df, t):
    """VCE Variant B: squeeze + breakout + green + MA50 + ZORUNLU RVOL>=1.5x.
    2026-07-27: hacim barajı eklendi (signals.check_vce_breakout ile aynı —
    hacimsiz kırılım fakeout riski, ölçüm: EV +1.55→+3.64% PF 1.43→2.13).
    NOT: bu ham-sinyal harness'ı artık aşıldı; canlı-birebir için
    backtest_live_replica.py gerçek scan_stock'u çağırır (bu gate dahil)."""
    a = df["atr_pct"].iloc[t - 1]; b = df["atr_pct"].iloc[t - 20:t - 5].mean()
    if pd.isna(a) or pd.isna(b) or b <= 0 or a >= b * 0.8:
        return False
    c = df["Close"].iloc[t]; hi = df["hi20"].iloc[t - 1]; ma50 = df["ma50"].iloc[t]
    if pd.isna(hi) or pd.isna(ma50):
        return False
    vol50 = df["vol50"].iloc[t]
    rvol = df["Volume"].iloc[t] / vol50 if vol50 > 0 else 0
    return c > hi > 0 and c > df["Close"].iloc[t - 1] and c > ma50 and rvol >= 1.5


def sig_rvol(df, t):
    """RVOL thrust: RVOL>=2.5x(50g) + yeşil + MA20 üstü (signals.py ile aynı)."""
    vol50 = df["vol50"].iloc[t]
    if pd.isna(vol50) or vol50 <= 0:
        return False
    rvol = df["Volume"].iloc[t] / vol50
    c = df["Close"].iloc[t]; ma20 = df["ma20"].iloc[t]
    if pd.isna(ma20):
        return False
    return rvol >= 2.5 and c > df["Close"].iloc[t - 1] and c > ma20


# ══════════════════════════════════════════════════════════════════════
# EXIT SİMÜLASYONU (bar-bar, gerçekçi stop dolumu)
# ══════════════════════════════════════════════════════════════════════
def simulate(df, t, exit_cfg):
    o = df["Open"].astype(float).values; c = df["Close"].astype(float).values
    h = df["High"].astype(float).values; low = df["Low"].astype(float).values
    n = len(df); e = t + 1
    if e >= n:
        return None
    entry = o[e]; atr = float(df["atr"].iloc[t])
    if entry <= 0 or atr <= 0:
        return None

    # stop = entry - stop_atr*ATR, ama type-cap ile kırpılır (canlı davranış)
    raw_stop = entry - exit_cfg["stop_atr"] * atr
    cap_stop = entry * (1 - exit_cfg["max_stop_pct"])
    stop = max(raw_stop, cap_stop)
    t1 = entry * (1 + exit_cfg["t1_pct"]) if exit_cfg.get("t1_pct") else None
    t2 = entry * (1 + exit_cfg["t2_pct"]) if exit_cfg.get("t2_pct") else None
    trail_mult = exit_cfg.get("trail_atr")
    trail_act = exit_cfg.get("trail_after", 1.5)
    hold = exit_cfg["hold"]

    pos = 1.0; realized = 0.0; peak = entry; t1_done = False
    last = min(e + hold, n - 1)
    for j in range(e, last + 1):
        if low[j] <= stop:
            px = min(o[j], stop) if o[j] < stop else stop
            realized += pos * (px / entry - 1); pos = 0.0; break
        if t1 and not t1_done and h[j] >= t1:
            realized += exit_cfg["t1_frac"] * (t1 / entry - 1)
            pos -= exit_cfg["t1_frac"]; t1_done = True
            if exit_cfg.get("be_after_t1"):
                stop = max(stop, entry)
        if t2 and h[j] >= t2:
            realized += pos * (t2 / entry - 1); pos = 0.0; break
        if h[j] > peak:
            peak = h[j]
        if trail_mult and (peak - entry) / atr >= trail_act:
            new_trail = peak - trail_mult * atr
            if new_trail > stop:
                stop = new_trail
    if pos > 0:
        realized += pos * (c[last] / entry - 1)
    return realized * 100


EXIT_OLD = dict(stop_atr=1.5, max_stop_pct=0.10, t1_pct=0.10, t1_frac=0.5,
                be_after_t1=True, t2_pct=0.28, trail_atr=2.5, trail_after=2.0, hold=10)
EXIT_NEW = dict(stop_atr=2.5, max_stop_pct=0.15, t1_pct=0.10, t1_frac=0.33,
                be_after_t1=True, t2_pct=None, trail_atr=3.0, trail_after=1.5, hold=20)


# ══════════════════════════════════════════════════════════════════════
def build_regime_map(spy):
    rmap = {}
    closes = spy["Close"].astype(float).reset_index(drop=True)
    dates = pd.to_datetime(spy["Date"]).reset_index(drop=True)
    for i in range(len(spy)):
        if i < 50:
            rmap[dates[i].normalize()] = "UNKNOWN"; continue
        try:
            rmap[dates[i].normalize()] = regime_from_spy_close(
                closes.iloc[max(0, i - 251):i + 1], None).get("regime", "UNKNOWN")
        except Exception:
            rmap[dates[i].normalize()] = "UNKNOWN"
    return rmap


def collect(data, rmap):
    """Her sistem için trade listesi. Sinyal→exit→sonuç + rejim + tarih."""
    old_trades, new_trades = [], []
    for tk, df in data.items():
        n = len(df)
        for t in range(60, n - 21):
            day = pd.to_datetime(df["Date"].iloc[t]).normalize()
            reg = rmap.get(day, "UNKNOWN")
            v = sig_vce(df, t)
            r = sig_rvol(df, t)
            # ESKİ: yalnız VCE
            if v:
                res = simulate(df, t, EXIT_OLD)
                if res is not None:
                    old_trades.append({"r": res, "date": day, "reg": reg, "tk": tk, "pw": "VCE"})
            # YENİ: VCE veya RVOL (VCE önce; ikisi de ateşlerse VCE)
            if v or r:
                res = simulate(df, t, EXIT_NEW)
                if res is not None:
                    new_trades.append({"r": res, "date": day, "reg": reg, "tk": tk,
                                       "pw": "VCE" if v else "RVOL"})
    return old_trades, new_trades


def stats(trades):
    if not trades:
        return None
    a = np.array([x["r"] for x in trades])
    wins = a[a > 0]; losses = a[a <= 0]
    return {
        "n": len(a), "ev": a.mean(), "wr": (a > 0).mean() * 100,
        "avg_win": wins.mean() if len(wins) else 0,
        "avg_loss": losses.mean() if len(losses) else 0,
        "median": np.median(a), "worst5": np.percentile(a, 5),
        "total": a.sum(), "sharpe": a.mean() / a.std() if a.std() > 0 else 0,
    }


def print_block(label, s):
    if not s:
        print(f"  {label}: veri yok"); return
    print(f"  {label}")
    print(f"    İşlem: {s['n']}  |  EV/trade: {s['ev']:+.2f}%  |  WR: {s['wr']:.0f}%  |  medyan: {s['median']:+.1f}%")
    print(f"    Ort kazanç: {s['avg_win']:+.1f}%  |  ort kayıp: {s['avg_loss']:+.1f}%  |  en kötü %5: {s['worst5']:+.1f}%")
    print(f"    Toplam getiri (bağımsız Σ): {s['total']:+.0f}%  |  Sharpe~: {s['sharpe']:.3f}")


def main():
    with open("output/_broad_data.pkl", "rb") as f:
        data = pickle.load(f)
    data = {t: enrich(df) for t, df in data.items()}
    with open("output/_edge_spy.pkl", "rb") as f:
        spy = pickle.load(f)
    rmap = build_regime_map(spy)

    old, new = collect(data, rmap)

    print("╔" + "═" * 78 + "╗")
    print(f"║  BACKTEST: ESKİ vs YENİ SİSTEM — {len(data)} ticker (S&P400+600), 2024-06→2026-05" + " " * 5 + "║")
    print("╚" + "═" * 78 + "╝")

    print("\n" + "═" * 80)
    print("  GENEL")
    print("═" * 80)
    so, sn = stats(old), stats(new)
    print_block("ESKİ (yalnız VCE + dar exit)", so)
    print()
    print_block("YENİ (VCE+RVOL + geniş exit)", sn)
    if so and sn:
        print(f"\n  → FARK: EV/trade {sn['ev']-so['ev']:+.2f}%  |  işlem {sn['n']-so['n']:+d}  |  "
              f"WR {sn['wr']-so['wr']:+.0f}%  |  toplam getiri {sn['total']-so['total']:+.0f}%")

    # OOS
    print("\n" + "═" * 80)
    print("  OUT-OF-SAMPLE (2025-06 sonrası — 'gelecek' verisi)")
    print("═" * 80)
    print_block("ESKİ (OOS)", stats([x for x in old if x["date"] >= SPLIT]))
    print()
    print_block("YENİ (OOS)", stats([x for x in new if x["date"] >= SPLIT]))

    # Rejim kırılımı (yeni sistem)
    print("\n" + "═" * 80)
    print("  YENİ SİSTEM — REJİM KIRILIMI (kötü piyasada ayakta mı?)")
    print("═" * 80)
    for reg in ["BULL", "CAUTION", "BEAR", "UNKNOWN"]:
        s = stats([x for x in new if x["reg"] == reg])
        if s:
            print(f"  {reg:<9}: n={s['n']:<5} EV {s['ev']:+.2f}%  WR {s['wr']:.0f}%  toplam {s['total']:+.0f}%")

    # Pathway kırılımı (yeni sistem: VCE vs RVOL ayrı)
    print("\n" + "═" * 80)
    print("  YENİ SİSTEM — PATHWAY KIRILIMI (VCE vs RVOL katkısı)")
    print("═" * 80)
    for pw in ["VCE", "RVOL"]:
        s = stats([x for x in new if x["pw"] == pw])
        if s:
            print(f"  {pw:<6}: n={s['n']:<5} EV {s['ev']:+.2f}%  WR {s['wr']:.0f}%  toplam {s['total']:+.0f}%")

    print("\n" + "═" * 80)
    # Özet dosyası
    out = {"old": so, "new": sn}
    json.dump(out, open("output/backtest_old_vs_new.json", "w"), indent=2, default=str)
    print("  📁 output/backtest_old_vs_new.json")


if __name__ == "__main__":
    main()
