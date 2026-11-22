"""Lab v3 shared data layer: OHLCV panels, as-of shares, Finviz-emulated pool."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "output/lab"
OHLCV = LAB / "ohlcv"
FACTS = LAB / "facts"
EARN = LAB / "earnings"
EVAL_START = pd.Timestamp("2021-01-04")
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
    f = LAB / "nasdaq/screener.json"
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
    """
    P = panels()
    dates = P["Close"].index
    cols = {}
    nq = nasdaq_screener()
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
            ser = ser[ser > 0].sort_index().rolling(5, min_periods=1).median()
            ser = ser[~ser.index.duplicated(keep="last")]
            cols[t] = ser.reindex(dates.union(ser.index)).ffill().bfill().reindex(dates)
        else:
            cur = (rec.get("info") or {}).get("sharesOutstanding")
            if cur:
                cols[t] = pd.Series(float(cur), index=dates)
    return pd.DataFrame(cols, index=dates)


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
    ind = {
        "avgvol63": v.rolling(63, min_periods=40).mean(),
        "dv20": (c * v).rolling(20, min_periods=15).mean(),
        "sma20": c.rolling(20).mean(),
        "sma50": c.rolling(50).mean(),
        "high20": h.rolling(20).max(),
    }
    ind["relvol"] = v / ind["avgvol63"].shift(1)
    ind["cap"] = c * shares_panel().reindex(columns=c.columns)
    return ind


@lru_cache(maxsize=1)
def pool_mask() -> pd.DataFrame:
    """Pre-registered broad pool (Section 0): price 7-250, avgvol63>300K, dv20>5M, cap 300M-10B."""
    P, I = panels(), indicators()
    c = P["Close"]
    m = (c >= 7) & (c <= 250) & (I["avgvol63"] > 300_000) & (I["dv20"] > 5e6) \
        & (I["cap"] >= 3e8) & (I["cap"] <= 1e10)
    return m.drop(columns=[x for x in BENCH if x in m.columns])


@lru_cache(maxsize=1)
def pool_mask_large() -> pd.DataFrame:
    """Ek D2 pool L: price ≥ 7 (no ceiling), avgvol63>300K, dv20>5M, cap ≥ 10B (no ceiling)."""
    P, I = panels(), indicators()
    c = P["Close"]
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
    base = (small | mid) & (c > 7) & (c >= 6) & (c <= 250)
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
