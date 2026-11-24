"""Lab v3 shared data layer: OHLCV panels, as-of shares, Finviz-emulated pool.

Veri kökü: varsayılan output/lab (2021–26). Ek R (dokunulmamış dönem sınavı) aynı betikleri
``LAB_DATA=output/lab_oos LAB_EVAL_START=2016-01-04 LAB_EVAL_END=2020-12-31`` ile ayrı bir kökte koşar.
Kökü izleyen betikler: fetch_ohlcv, h0_signals, h6_signals, h6r, evaluate, challengers.OUT, oos_r.
Diğerleri (d1/d2/core_n/…) 2021–26 köküne sabittir — LAB_DATA ile çalıştırılmamalı.

DİKKAT (Ek R): output/lab kökü 2026-11-21'de bölünme dosyası OLMADAN indirildi; o kökte evren ve havuz,
sonradan bölünen/birleşen hisselerde o günün fiyatını ve piyasa değerini yanlış hesaplar (H6S'nin +66%'sı
bu kusurdandı). O kök tarihsel kayıttır. Yeni ölçümler bölünme dosyası olan köklerde yapılmalı:
output/lab_is_split (2021–26, düzeltilmiş) ve output/lab_oos (2016–2020).
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/lab"                       # 2021–26 kökü; güncel (tarihsiz) anlık görüntüler burada
LAB = ROOT / os.environ.get("LAB_DATA", "output/lab")
OHLCV = LAB / "ohlcv"
FACTS = LAB / "facts"
EARN = LAB / "earnings"
EVAL_START = pd.Timestamp(os.environ.get("LAB_EVAL_START", "2021-01-04"))
EVAL_END = pd.Timestamp(os.environ["LAB_EVAL_END"]) if os.environ.get("LAB_EVAL_END") else None   # dahil
BENCH = ("SPY", "^VIX", "IWM", "IJR", "MDY")
FIELDS = ("Open", "High", "Low", "Close", "Volume")


@lru_cache(maxsize=1)
def long_frame() -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(f) for f in sorted(OHLCV.glob("batch_*.parquet"))])
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    df = df.drop_duplicates(["ticker", "Date"], keep="last").sort_values(["ticker", "Date"])
    df["Volume"] = df["Volume"].fillna(0)
    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def panels():
    """dict field -> DataFrame[dates x tickers] on the SPY trading calendar."""
    df = long_frame()
    dates = pd.DatetimeIndex(sorted(df.loc[df.ticker == "SPY", "Date"].unique()))
    out = {}
    for f in FIELDS:
        p = df.pivot(index="Date", columns="ticker", values=f).reindex(dates)
        out[f] = p
    return out


@lru_cache(maxsize=1)
def frames() -> dict:
    """ticker -> production-shaped DataFrame (Date column, RangeIndex)."""
    df = long_frame()
    return {t: g.drop(columns="ticker").reset_index(drop=True) for t, g in df.groupby("ticker")}


@lru_cache(maxsize=1)
def nasdaq_screener() -> pd.DataFrame:
    """Nasdaq screener snapshot (2026-11-21): current cap/country/sector fallback."""
    f = BASE / "nasdaq/screener.json"
    if not f.exists():
        return pd.DataFrame()
    rows = json.loads(f.read_text())["data"]["rows"]
    df = pd.DataFrame(rows)
    df["ticker"] = df.symbol.str.replace("/", "-", regex=False).str.strip()
    px = pd.to_numeric(df.lastsale.str.replace("$", "", regex=False), errors="coerce")
    cap = pd.to_numeric(df.marketCap, errors="coerce")
    df["shares_now"] = (cap / px).where((cap > 0) & (px > 0))
    return df.set_index("ticker")[["country", "sector", "shares_now"]]


def load_facts(t):
    f = FACTS / f"{t}.json"
    return json.loads(f.read_text()) if f.exists() else None


@lru_cache(maxsize=1)
def shares_panel() -> pd.DataFrame:
    """As-of shares outstanding [dates x tickers].

    yfinance share series is noisy (class aggregation glitches), so a rolling
    median of the last 5 observations is used. Before the first observation the
    earliest value is back-filled (documented minor look-ahead, cap-band only).
    Tickers without a series fall back to CURRENT sharesOutstanding (flagged).

    Ek R (kökte bölünme dosyası varsa): Yahoo serisi o günün ham sayısıdır; gözlemler önce bugünün hisse
    birimine çevrilir (gözlem tarihinden sonraki bölünmeler), medyan ve ileri/geri taşıma o birimde yapılır
    (bölünme gününde medyan gecikmesi olmasın), sonra o günün birimine dönülür (÷ F). Güncel sayı yedeği de
    bugünün birimindedir → ÷ F. Böylece her yerde piyasa değeri = actual_close × shares_panel.
    """
    P = panels()
    dates = P["Close"].index
    cols = {}
    nq = nasdaq_screener()
    ev = split_events()
    SF = split_factor() if ev else None                 # None → 2021–26 kökü: eski davranış aynen
    for t in P["Close"].columns:
        rec = load_facts(t) or {}
        s = rec.get("shares") or {}
        if not s and not (rec.get("info") or {}).get("sharesOutstanding"):
            if t in nq.index and pd.notna(nq.at[t, "shares_now"]):
                cols[t] = pd.Series(float(nq.at[t, "shares_now"]), index=dates)
            continue
        if s:
            ser = pd.Series(s, dtype=float)
            ser.index = pd.to_datetime(ser.index)
            ser = ser[ser > 0].sort_index()
            if t in ev:                                  # gözlemleri bugünün birimine çevir
                ser = ser * np.array([factor_after(ev[t], d) for d in ser.index])
            ser = ser.rolling(5, min_periods=1).median()
            ser = ser[~ser.index.duplicated(keep="last")]
            cols[t] = ser.reindex(dates.union(ser.index)).ffill().bfill().reindex(dates)
        else:
            cur = (rec.get("info") or {}).get("sharesOutstanding")
            if cur:
                cols[t] = pd.Series(float(cur), index=dates)
    out = pd.DataFrame(cols, index=dates)
    if SF is not None:                                   # bugünün birimi → o günün birimi
        out = out / SF.reindex(columns=out.columns)
    return out


@lru_cache(maxsize=1)
def split_events() -> dict:
    """ticker -> [(tarih, oran)] (Yahoo "Stock Splits": 4.0 = 4'e bölünme, 0.1 = 10→1 birleşme). Dosya yoksa {}."""
    files = sorted((LAB / "splits").glob("batch_*.parquet"))
    if not files:
        return {}
    sp = pd.concat([pd.read_parquet(f) for f in files])
    sp = sp[sp.ratio > 0].sort_values(["ticker", "Date"])
    return {t: [(pd.Timestamp(d), float(r)) for d, r in zip(g.Date, g.ratio)] for t, g in sp.groupby("ticker")}


def factor_after(events, d) -> float:
    """d gününden SONRAKİ bölünme oranlarının çarpımı (d günü bölünme varsa o gün zaten bölünmüş sayılır)."""
    f = 1.0
    for when, ratio in events:
        if when > d:
            f *= ratio
    return f


@lru_cache(maxsize=1)
def split_factor() -> pd.DataFrame:
    """F[dates x tickers] = t'den SONRAKİ bölünme oranlarının çarpımı (Ek R bölünme düzeltmesi).

    Yahoo fiyat ve hacmi bugüne kadarki bölünmelere göre düzeltilmiş; hisse sayısı serisi ise o günün ham
    sayısı. İkisinin çarpımı sonradan bölünen hissede piyasa değerini oran kadar küçültür (Tesla 2016:
    gerçek ~30 milyar $ → ~2 milyar $), ters bölünende 7 $ altı kâğıdı 7 $ üstü gösterir. O günün gerçek
    fiyatı = düzeltilmiş × F, gerçek hacmi = düzeltilmiş ÷ F. Getiriler düzeltilmiş fiyatla hesaplanır.
    <kök>/splits yoksa (2021–26 kökü) F ≡ 1 → o kökteki sonuçlar bit düzeyinde aynı kalır.
    """
    C = panels()["Close"]
    F = pd.DataFrame(1.0, index=C.index, columns=C.columns)
    for t, events in split_events().items():
        if t in F.columns:
            for when, ratio in events:
                F.loc[F.index < when, t] *= ratio
    return F


def actual_close() -> pd.DataFrame:
    """O gün ekranda görülen (sonraki bölünmelerden önceki) kapanış — fiyat bandı ve piyasa değeri için."""
    return panels()["Close"] * split_factor()


@lru_cache(maxsize=1)
def info_frame() -> pd.DataFrame:
    rows = []
    for f in FACTS.glob("*.json"):
        rec = json.loads(f.read_text())
        rows.append({"ticker": rec["ticker"], **(rec.get("info") or {})})
    yf_info = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame(
        columns=["country", "sector", "floatShares", "sharesOutstanding"])
    nq = nasdaq_screener()
    if len(nq):   # fill country/sector gaps from Nasdaq (current snapshot)
        idx = yf_info.index.union(nq.index)
        yf_info = yf_info.reindex(idx)
        for c in ("country", "sector"):
            yf_info[c] = yf_info[c].where(yf_info[c].notna(), nq[c].reindex(idx))
    return yf_info


@lru_cache(maxsize=1)
def indicators():
    P = panels()
    c, h, v = P["Close"], P["High"], P["Volume"]
    F = split_factor()
    av = v.rolling(63, min_periods=40).mean()
    ind = {
        "avgvol63": av / F,                     # o günün gerçek hisse adedi (eşik karşılaştırmaları için)
        "dv20": (c * v).rolling(20, min_periods=15).mean(),     # dolar: bölünmeden bağımsız
        "sma20": c.rolling(20).mean(),
        "sma50": c.rolling(50).mean(),
        "high20": h.rolling(20).max(),
    }
    ind["relvol"] = v / av.shift(1)             # oran: düzeltilmiş hacimlerle, bölünmeden bağımsız
    ind["cap"] = c * F * shares_panel().reindex(columns=c.columns)
    return ind


@lru_cache(maxsize=1)
def pool_mask() -> pd.DataFrame:
    """Pre-registered broad pool (Section 0): price 7-250, avgvol63>300K, dv20>5M, cap 300M-10B."""
    I = indicators()
    c = actual_close()
    m = (c >= 7) & (c <= 250) & (I["avgvol63"] > 300_000) & (I["dv20"] > 5e6) \
        & (I["cap"] >= 3e8) & (I["cap"] <= 1e10)
    return m.drop(columns=[x for x in BENCH if x in m.columns])


@lru_cache(maxsize=1)
def pool_mask_large() -> pd.DataFrame:
    """Ek D2 pool L: price ≥ 7 (no ceiling), avgvol63>300K, dv20>5M, cap ≥ 10B (no ceiling)."""
    I = indicators()
    c = actual_close()
    m = (c >= 7) & (I["avgvol63"] > 300_000) & (I["dv20"] > 5e6) & (I["cap"] >= 1e10)
    return m.drop(columns=[x for x in BENCH if x in m.columns])


@lru_cache(maxsize=1)
def finviz_members(cap_n: int = 300) -> dict:
    """date -> list of tickers the live Finviz Q6/Q6b/Q7/Q7b union would return,
    ranked by dollar volume (Volume × Price) and cut to `cap_n` (live: 300)."""
    from swing_trader.small_cap.universe import SmallCapUniverse
    P, I = panels(), indicators()
    c, h, v = P["Close"], P["High"], P["Volume"]
    info = info_frame()
    usa = pd.Series(False, index=c.columns)
    if len(info):
        usa = usa | pd.Series(info["country"].eq("United States"), dtype=bool).reindex(c.columns, fill_value=False)
    excluded = set(getattr(SmallCapUniverse, "EXCLUDED_TICKERS", set()) or set())
    cap = I["cap"]
    small = (cap >= 3e8) & (cap < 2e9) & (I["avgvol63"] > 300_000)
    mid = (cap >= 2e9) & (cap <= 1e10) & (I["avgvol63"] > 500_000)
    ca = actual_close()                          # fiyat bandı o günün gerçek fiyatıyla (Ek R)
    base = (small | mid) & (ca > 7) & (ca >= 6) & (ca <= 250)
    base = base & usa.reindex(c.columns, fill_value=False).values[None, :]
    q6 = (c > I["sma50"]) & (h >= I["high20"])
    q7 = (I["relvol"] > 2) & (c > c.shift(1)) & (c > I["sma20"])
    member = base & (q6 | q7)
    dvol = v * c
    out = {}
    for d in c.index:
        row = member.loc[d]
        tick = [t for t in row.index[row.values] if t not in excluded and t not in BENCH]
        if not tick:
            out[d] = []
            continue
        ranked = dvol.loc[d, tick].sort_values(ascending=False)
        out[d] = list(ranked.index[:cap_n])
    return out


def window_6mo(df: pd.DataFrame, d: pd.Timestamp) -> pd.DataFrame:
    """Production scan window: yf.download(period='6mo') as of EOD d."""
    start = d - pd.DateOffset(months=6) + pd.Timedelta(days=1)
    w = df[(df.Date >= start) & (df.Date <= d)]
    return w.reset_index(drop=True)


def asof_window(df: pd.DataFrame, d: pd.Timestamp, f: float = 1.0) -> pd.DataFrame:
    """Canlının d günü göreceği 6 aylık pencere: düzeltilmiş barlar × F(d), hacim ÷ F(d) (f = split_factor).

    f == 1 (2021–26 kökü ve sonradan bölünmemiş hisseler) → window_6mo ile birebir aynı.
    """
    w = window_6mo(df, d)
    if f != 1.0 and len(w):
        w = w.copy()
        cols = ["Open", "High", "Low", "Close"]
        w[cols] = w[cols] * f
        w["Volume"] = w["Volume"] / f
    return w


def _earnings_full_dates(t):
    """Same yfinance call (get_earnings_dates(limit=40)) stored with timestamps (H5 data)."""
    f = LAB / "earnings_full" / f"{t}.json"
    if not f.exists():
        return None
    rows = json.loads(f.read_text()).get("rows") or []
    days = []
    for r in rows:
        ts = pd.Timestamp(r["ts"])
        days.append(pd.Timestamp(ts.tz_convert("America/New_York").date() if ts.tzinfo else ts.date()))
    return pd.DatetimeIndex(sorted(set(days))) if days else None


def earnings_dates(t):
    """Earnings dates for the live gate replica.

    Ek D1 notu (2026-11-22): 130 dosya indirme sırasında Yahoo kısıtlamasıyla BOŞ yazılmıştı
    ("No earnings dates found"); H0 sinyallerinin 168/539'u bilanço kapısından veri yokluğu
    yüzünden geçiyordu. Boş/yoksa earnings_full'a düşülür — iki kaynak ortak 1.281 hissede
    örtüşen dönemde birebir aynı tarihleri veriyor (0 uyuşmazlık).
    """
    f = EARN / f"{t}.json"
    rec = json.loads(f.read_text()) if f.exists() else {}
    dates = rec.get("dates") or []
    if dates:
        return pd.DatetimeIndex(pd.to_datetime(dates))
    return _earnings_full_dates(t)


def earnings_calendar_known(ed, d, max_gap: int = 155) -> bool:
    """Ek R bilanço veri kalitesi kuralı (canlıdaki "bilinmiyorsa ret" ilkesinin geçmiş veri karşılığı).

    d günü için takvim güvenilirdir ancak d'den önceki son kayıt ile d ve sonrasındaki ilk kayıt mevcutsa VE
    araları ≤ max_gap günse (arada eksik çeyrek yok: ardışık bilançolar ~60–150 gün arayla gelir, bir çeyrek
    eksikse ara ~180 gün olur). Aksi halde kapı "yakında bilanço yok" diyemez → sinyal reddedilir.
    """
    ed = pd.DatetimeIndex(ed).sort_values()
    i = int(ed.searchsorted(pd.Timestamp(d), side="left"))        # ed[i] >= d
    if i == 0 or i == len(ed):
        return False
    return (ed[i] - ed[i - 1]).days <= max_gap
