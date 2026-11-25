"""Fetch daily OHLCV for the research universe (Lab v3).

Universe: Discovery V2 Phase 1 snapshot CANDIDATE records (current NASDAQ/NYSE
common stocks + ADRs, observed 2026-10-10). This is NOT point-in-time: stocks
delisted before 2026-10-10 are missing (survivorship bias, documented).

Prices use yfinance auto_adjust=True — the same adjustment production gets
from ``yf.Ticker(t).history()`` (default auto_adjust=True).

Output: <kök>/ohlcv/batch_XXX.parquet (long format) + manifest.json; <kök>/splits/batch_XXX.parquet.
Resumable: existing batches are skipped. Kök: LAB_DATA (varsayılan output/lab).

Bölünme olayları (Yahoo "Stock Splits") HER ZAMAN indirme gününe kadar toplanır (LAB_SPLITS_END ile
sabitlenebilir): fiyatlar indirme günündeki bölünmelere göre düzeltilmiş gelir, hisse sayısı serisi ise o
günün ham sayısıdır — lab/common.split_factor bu dosya olmadan o günün fiyatını/piyasa değerini yanlış
hesaplar (Lab v3 Ek R: AMC 2021'de 3 $ yerine 31 $ göründü, sahte +303% işlem). OHLCV yine
LAB_OHLCV_END'de kesilir. Ek R kökü:
    LAB_DATA=output/lab_oos LAB_OHLCV_START=2014-10-01 LAB_OHLCV_END=2021-03-06 LAB_SPLITS_END=2026-11-26
Not: 2026-11-21'de indirilen output/lab kökünde bölünme dosyası YOK (tarihsel, düzeltilmemiş kayıt).
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "output/discovery_v2/final_snapshots/20261010T201236851614Z/universe.json"
LAB = ROOT / os.environ.get("LAB_DATA", "output/lab")
OUT = LAB / "ohlcv"
START = os.environ.get("LAB_OHLCV_START", "2020-06-01")
END = os.environ.get("LAB_OHLCV_END", "2026-11-21")   # END exclusive → last bar 2026-11-20
# exclusive; varsayılan yarın → bölünmeler indirme gününe kadar toplanır (eksikse evren emülasyonu yanlış: Ek R)
SPLITS_END = os.environ.get("LAB_SPLITS_END") or (pd.Timestamp.now().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
BATCH = 100
BENCH = ["SPY", "^VIX", "IWM", "IJR", "MDY"]


def universe():
    override = os.environ.get("LAB_TICKERS")       # hazır hisse listesi (Ek T sürekli kökü); yoksa snapshot adayları
    if override:
        return BENCH + [t for t in (ROOT / override).read_text().split() if t not in BENCH]
    recs = json.loads(SNAP.read_text())["records"]
    syms = sorted({r["yfinance_symbol_candidate"] or r["symbol"]
                   for r in recs if r["discovery_status"] == "CANDIDATE"})
    return BENCH + syms


def fetch(batch):
    """→ (OHLCV long frame, split events frame). Splits are empty unless LAB_SPLITS_END is set."""
    empty = pd.DataFrame(columns=["ticker", "Date", "ratio"])
    for attempt in range(4):
        try:
            raw = yf.download(batch, start=START, end=SPLITS_END or END, auto_adjust=True,
                              group_by="ticker", threads=True, progress=False, actions=bool(SPLITS_END))
            break
        except Exception as e:  # network hiccup → backoff
            print("retry", attempt, e, flush=True)
            time.sleep(10 * (attempt + 1))
    else:
        return pd.DataFrame(), empty
    rows, splits = [], []
    for t in batch:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if df.empty:
            continue
        df = df.reset_index().rename(columns={"index": "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
        df["ticker"] = t
        if SPLITS_END and "Stock Splits" in df.columns:
            sp = df[df["Stock Splits"].fillna(0) > 0]
            if len(sp):
                splits.append(sp[["ticker", "Date"]].assign(ratio=sp["Stock Splits"].astype(float)))
        df = df[df["Date"] < pd.Timestamp(END)]
        if df.empty:
            continue
        rows.append(df[["ticker", "Date", "Open", "High", "Low", "Close", "Volume"]])
    return (pd.concat(rows) if rows else pd.DataFrame()), (pd.concat(splits) if splits else empty)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    syms = universe()
    batches = [syms[i:i + BATCH] for i in range(0, len(syms), BATCH)]
    print(f"{len(syms)} symbols, {len(batches)} batches", flush=True)
    for i, b in enumerate(batches):
        f = OUT / f"batch_{i:03d}.parquet"
        if f.exists():
            continue
        df, sp = fetch(b)
        if SPLITS_END:                       # önce bölünmeler: OHLCV dosyası varsa bölünmesi de vardır
            (LAB / "splits").mkdir(parents=True, exist_ok=True)
            sp.to_parquet(LAB / "splits" / f"batch_{i:03d}.parquet", index=False)
        df.to_parquet(f, index=False)
        print(f"batch {i}: {df['ticker'].nunique() if len(df) else 0}/{len(b)} tickers, {len(sp)} splits", flush=True)
        time.sleep(1.5)
    (OUT / "manifest.json").write_text(json.dumps(dict(
        start=START, end_exclusive=END, splits_end_exclusive=SPLITS_END, auto_adjust=True, snapshot=str(SNAP.name),
        fetched_at=pd.Timestamp.utcnow().isoformat(), n_symbols=len(syms)), indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
