# -*- coding: utf-8 -*-
"""
CANLI-BİREBİR BACKTEST — ürünü geçmişte GERÇEKTE nasıl kullanıyorsak öyle koştur
================================================================================
Kullanıcının haklı itirazı: önceki backtest (backtest_old_vs_new.py) canlı
sistemi taklit ETMİYORDU — (a) 995 hisseyi filtresiz tarıyordu (canlıda Finviz
ön-filtreliyor), (b) VCE/RVOL sinyalini elle yeniden kodluyordu (canlıda GERÇEK
motor scan_stock tüm gate'leriyle karar veriyor). Bu harness ikisini de düzeltir.

CANLI AKIŞ (birebir):
  1. Finviz evreni EMÜLE et (Q6/Q6b/Q7/Q7b — universe.py'deki canlı sorgularla
     BİREBİR: 20g yeni zirve+SMA50 / RVOL>2+yeşil+SMA20, likidite, mcap band).
     → o gün Finviz'in döndüreceği aday havuzu (point-in-time, survivorship yok).
  2. GERÇEK motoru koştur: engine.scan_stock(backtest_mode=True) — TÜM gate'ler
     (dolar-hacim, RSI, OBV, Weinstein, VCP, quality≥eşik, R:R). Elle sinyal YOK.
  3. Point-in-time mcap/float ENJEKTE et (stock_info) — motorun mcap filtresi
     gerçek(-e yakın) veriyle çalışsın (yaklaşık: güncel shares × o günkü close).
  4. Sinyal → t+1 açılış girişi → gerçekçi exit sim (eski dar vs yeni geniş).
  5. Regime point-in-time SPY'den (canlı ile aynı fonksiyon).

SINIRLAR (dürüstlük — %100 birebir imkansız):
  - mcap YAKLAŞIK (güncel shares × geçmiş close; dilüsyon hatası).
  - Type S (short squeeze) backtest'te oluşamaz (geçmiş short-interest yok).
  - Katalizör/haber/insider bonusları backtest'te 0 (geçmiş veri yok).
  - Sektör-RS gerçek ETF yerine SPY proxy.
  Bunlar canlı ile kaçınılmaz küçük farklar; sonuç yine de ~%95 birebir.

Cache: output/_broad_data.pkl + _shares_broad.json + _edge_spy.pkl
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
from swing_trader.small_cap.regime_logic import regime_from_spy_close

SPLIT = pd.Timestamp("2025-06-01")
SMALL = (300e6, 2e9)
MID = (2e9, 10e9)


# ══════════════════════════════════════════════════════════════════════
# FINVIZ EVREN EMÜLATÖRÜ (universe.py Q6/Q6b/Q7/Q7b ile BİREBİR)
# ══════════════════════════════════════════════════════════════════════
def enrich(df):
    df = df.copy()
    c = df["Close"].astype(float); h = df["High"].astype(float); l = df["Low"].astype(float)
    v = df["Volume"].astype(float)
    df["ma20"] = c.rolling(20).mean(); df["ma50"] = c.rolling(50).mean()
    df["avgvol50"] = v.rolling(50).mean()   # RVOL için (signals.py check_rvol_thrust ile aynı)
    df["avgvol_liq"] = v.rolling(50).mean() # Finviz "Average Volume" ~ ortalama günlük hacim
    df["hi20_prev"] = h.rolling(20).max().shift()
    df["chg"] = (c / c.shift() - 1) * 100
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean(); df["atr_pct"] = df["atr"] / c * 100
    return df


def finviz_hit(row, mcap):
    """Bu (ticker, gün) canlı Finviz sorgularının HERHANGİ birine takılır mı?
    universe.py Q6/Q6b/Q7/Q7b kriterleriyle birebir. mcap None ise band-bilinmez
    → iyimser (küçük ya da mid say)."""
    price = row["Close"]
    if price <= 7:
        return False
    small = mcap is None or (SMALL[0] <= mcap < SMALL[1])
    mid = mcap is None or (MID[0] <= mcap <= MID[1])
    av = row["avgvol_liq"]
    if pd.isna(av):
        return False

    ma50 = row["ma50"]; ma20 = row["ma20"]; hi20p = row["hi20_prev"]
    new20 = (not pd.isna(hi20p)) and row["High"] > hi20p
    above50 = (not pd.isna(ma50)) and price > ma50
    above20 = (not pd.isna(ma20)) and price > ma20
    rvol = row["Volume"] / row["avgvol50"] if row["avgvol50"] > 0 else 0
    green = row["chg"] > 0

    # Q6 (small): 20g yeni zirve + SMA50 üstü + avgvol>500K
    if small and av > 500e3 and above50 and new20:
        return True
    # Q6b (mid): avgvol>1M
    if mid and av > 1e6 and above50 and new20:
        return True
    # Q7 (small): RVOL>2 + yeşil + SMA20 üstü + avgvol>500K
    if small and av > 500e3 and rvol > 2 and green and above20:
        return True
    # Q7b (mid): avgvol>1M
    if mid and av > 1e6 and rvol > 2 and green and above20:
        return True
    return False


# ══════════════════════════════════════════════════════════════════════
# EXIT SİMÜLASYONU (eski dar vs yeni geniş) + SENTETİK SLIPPAGE (S1)
# ══════════════════════════════════════════════════════════════════════
def _slippage_bps(df, t):
    """S1 (2026-07-27): gerçekçi kayma tahmini (spread verisi yok → proxy).
    Small-cap'te spread ≈ likidite azlığı + oynaklık. Dolar-hacim düşük ve/veya
    ATR yüksekse giriş/çıkışta daha çok kayarsın. Backtest'i canlıya yaklaştırır
    (kâğıt-üstü kârın slippage'te erimesini modeller). Tek yönlü bps döndürür."""
    try:
        dvol = float((df["Volume"].tail(20) * df["Close"].tail(20)).mean())
        atrp = float(df["atr_pct"].iloc[t])
    except Exception:
        return 15.0  # bilinmiyorsa temkinli varsayım
    slip = 8.0  # taban: likit small-cap ~8bps tek yön
    if dvol < 3_000_000: slip += 25
    elif dvol < 7_000_000: slip += 12
    elif dvol < 15_000_000: slip += 5
    if atrp > 8: slip += 20
    elif atrp > 5: slip += 8
    return slip


def simulate(df, t, cfg, apply_slippage=True):
    o = df["Open"].astype(float).values; c = df["Close"].astype(float).values
    h = df["High"].astype(float).values; low = df["Low"].astype(float).values
    n = len(df); e = t + 1
    if e >= n:
        return None
    entry = o[e]; atr = float(df["atr"].iloc[t])
    if entry <= 0 or atr <= 0:
        return None
    # Slippage: giriş + çıkış = 2 yön. Toplam getiriden düş (gerçekçilik).
    slip_pct = (2 * _slippage_bps(df, t) / 10000.0) if apply_slippage else 0.0
    raw_stop = entry - cfg["stop_atr"] * atr
    cap_stop = entry * (1 - cfg["max_stop_pct"])
    stop = max(raw_stop, cap_stop)
    t1 = entry * (1 + cfg["t1_pct"]) if cfg.get("t1_pct") else None
    t2 = entry * (1 + cfg["t2_pct"]) if cfg.get("t2_pct") else None
    pos = 1.0; realized = 0.0; peak = entry; t1_done = False
    last = min(e + cfg["hold"], n - 1)
    for j in range(e, last + 1):
        if low[j] <= stop:
            px = min(o[j], stop) if o[j] < stop else stop
            realized += pos * (px / entry - 1); pos = 0.0; break
        if t1 and not t1_done and h[j] >= t1:
            realized += cfg["t1_frac"] * (t1 / entry - 1)
            pos -= cfg["t1_frac"]; t1_done = True
            if cfg.get("be_after_t1"):
                stop = max(stop, entry)
        if t2 and h[j] >= t2:
            realized += pos * (t2 / entry - 1); pos = 0.0; break
        if h[j] > peak:
            peak = h[j]
        if cfg.get("trail_atr") and (peak - entry) / atr >= cfg.get("trail_after", 1.5):
            nt = peak - cfg["trail_atr"] * atr
            if nt > stop:
                stop = nt
    if pos > 0:
        realized += pos * (c[last] / entry - 1)
    return (realized - slip_pct) * 100  # slippage cezasını düş


EXIT_OLD = dict(stop_atr=1.5, max_stop_pct=0.10, t1_pct=0.10, t1_frac=0.5,
                be_after_t1=True, t2_pct=0.28, trail_atr=2.5, trail_after=2.0, hold=10)
EXIT_NEW = dict(stop_atr=2.5, max_stop_pct=0.15, t1_pct=0.10, t1_frac=0.33,
                be_after_t1=True, t2_pct=None, trail_atr=3.0, trail_after=1.5, hold=20)


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


def main():
    raw = pickle.load(open("output/_broad_data.pkl", "rb"))
    shares = json.load(open("output/_shares_broad.json"))
    spy_raw = pickle.load(open("output/_edge_spy.pkl", "rb"))
    spy = spy_raw.copy()
    spy_win = spy.tail(0)  # placeholder
    rmap = build_regime_map(spy)

    data = {t: enrich(df) for t, df in raw.items()}
    engine = SmallCapEngine()

    # SPY penceresi hızlı erişim için tarih→index
    spy["_d"] = pd.to_datetime(spy["Date"]).dt.normalize()

    old_trades, new_trades = [], []
    scanned_days = 0
    finviz_pass = 0
    engine_signals = 0

    tickers = list(data.keys())
    for ti, tk in enumerate(tickers):
        df = data[tk]
        sh = shares.get(tk, {})
        sh_out = sh.get("shares")
        flt = sh.get("float")
        n = len(df)
        for t in range(60, n - 21):
            row = df.iloc[t]
            close = float(row["Close"])
            mcap = close * sh_out if sh_out else None
            # 1. FINVIZ EVREN EMÜLASYONU — o gün aday havuzuna girer mi?
            if not finviz_hit(row, mcap):
                continue
            finviz_pass += 1
            # 2. GERÇEK MOTOR — scan_stock, tüm gate'ler, point-in-time
            day = pd.to_datetime(df["Date"].iloc[t]).normalize()
            reg = rmap.get(day, "UNKNOWN")
            df_win = df.iloc[:t + 1]
            # point-in-time SPY penceresi (RS için)
            spy_slice = spy[spy["_d"] <= day].tail(60)
            stock_info = {
                "ticker": tk,
                "marketCap": int(mcap) if mcap else 0,
                "floatShares": int(flt) if flt else 0,
                "shortName": tk, "sector": "Unknown",
            }
            try:
                sig = engine.scan_stock(
                    tk, df_win, stock_info=stock_info, backtest_mode=True,
                    portfolio_value=10000,
                    spy_df_window=spy_slice if len(spy_slice) >= 6 else None,
                    regime=reg,
                )
            except Exception:
                sig = None
            if not sig:
                continue
            engine_signals += 1
            pw = sig.get("trigger_pathway", "vce_breakout")
            r_old = simulate(df, t, EXIT_OLD)
            r_new = simulate(df, t, EXIT_NEW)
            if r_old is not None:
                old_trades.append({"r": r_old, "date": day, "reg": reg, "pw": pw})
            if r_new is not None:
                new_trades.append({"r": r_new, "date": day, "reg": reg, "pw": pw})
        if (ti + 1) % 200 == 0:
            print(f"  ...{ti+1}/{len(tickers)} ticker tarandı (motor sinyali: {engine_signals})", flush=True)

    def stats(tr):
        if not tr:
            return None
        a = np.array([x["r"] for x in tr])
        w = a[a > 0]; l = a[a <= 0]
        return dict(n=len(a), ev=a.mean(), wr=(a > 0).mean() * 100,
                    aw=w.mean() if len(w) else 0, al=l.mean() if len(l) else 0,
                    med=np.median(a), worst5=np.percentile(a, 5), total=a.sum())

    def show(lbl, s):
        if not s:
            print(f"  {lbl}: sinyal yok"); return
        print(f"  {lbl}")
        print(f"    İşlem: {s['n']}  EV/trade: {s['ev']:+.2f}%  WR: {s['wr']:.0f}%  medyan: {s['med']:+.1f}%")
        print(f"    Ort kazanç: {s['aw']:+.1f}%  ort kayıp: {s['al']:+.1f}%  en kötü %5: {s['worst5']:+.1f}%  toplam Σ: {s['total']:+.0f}%")

    print("\n╔" + "═" * 76 + "╗")
    print("║  CANLI-BİREBİR BACKTEST — gerçek motor + Finviz-emüle evren + point-in-time" + " " + "║")
    print("╚" + "═" * 76 + "╝")
    print(f"\n  Finviz emülasyonu geçen (ticker,gün): {finviz_pass:,}  →  motor sinyali üretti: {engine_signals:,}")
    print(f"  (huni: {finviz_pass:,} aday → {engine_signals:,} sinyal = motor gate'leri %{100*(1-engine_signals/max(finviz_pass,1)):.0f} eledi)")

    print("\n" + "═" * 78)
    print("  GENEL — aynı sinyaller, eski dar exit vs yeni geniş exit")
    print("═" * 78)
    so, sn = stats(old_trades), stats(new_trades)
    show("ESKİ EXIT (stop1.5, cap%10, T2cap28, hold10)", so)
    print()
    show("YENİ EXIT (stop2.5, cap%15, cap yok, hold20)", sn)
    if so and sn:
        print(f"\n  → EXIT FARKI: EV {sn['ev']-so['ev']:+.2f}%  WR {sn['wr']-so['wr']:+.0f}%  toplam {sn['total']-so['total']:+.0f}%")

    print("\n" + "═" * 78)
    print("  OUT-OF-SAMPLE (2025-06+) — yeni exit")
    print("═" * 78)
    show("YENİ (OOS)", stats([x for x in new_trades if x["date"] >= SPLIT]))

    print("\n" + "═" * 78)
    print("  PATHWAY KIRILIMI (yeni exit) — VCE vs RVOL, GERÇEK motordan geçmiş")
    print("═" * 78)
    for pw in ["vce_breakout", "rvol_thrust"]:
        show(pw, stats([x for x in new_trades if x["pw"] == pw]))
        print()

    print("═" * 78)
    print("  REJİM KIRILIMI (yeni exit)")
    print("═" * 78)
    for reg in ["BULL", "CAUTION", "BEAR", "UNKNOWN"]:
        s = stats([x for x in new_trades if x["reg"] == reg])
        if s:
            print(f"  {reg:<9}: n={s['n']:<4} EV {s['ev']:+.2f}%  WR {s['wr']:.0f}%  toplam {s['total']:+.0f}%")

    json.dump({"old": so, "new": sn}, open("output/backtest_live_replica.json", "w"),
              indent=2, default=str)
    print("\n  📁 output/backtest_live_replica.json")


if __name__ == "__main__":
    main()
