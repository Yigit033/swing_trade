"""Fetch daily OHLCV for the research universe (Lab v3).

Universe: Discovery V2 Phase 1 snapshot CANDIDATE records (current NASDAQ/NYSE
common stocks + ADRs, observed 2026-10-10). This is NOT point-in-time: stocks
delisted before 2026-10-10 are missing (survivorship bias, documented).

Prices use yfinance auto_adjust=True — the same adjustment production gets
from ``yf.Ticker(t).history()`` (default auto_adjust=True).

Output: output/lab/ohlcv/batch_XXX.parquet (long format) + manifest.json.
Resumable: existing batches are skipped.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "output/discovery_v2/final_snapshots/20261010T201236851614Z/universe.json"
OUT = ROOT / "output/lab/ohlcv"
START, END = "2020-06-01", "2026-11-21"   # END exclusive → last bar 2026-11-20
BATCH = 100
BENCH = ["SPY", "^VIX", "IWM", "IJR", "MDY"]


def universe():
    recs = json.loads(SNAP.read_text())["records"]
    syms = sorted({r["yfinance_symbol_candidate"] or r["symbol"]
                   for r in recs if r["discovery_status"] == "CANDIDATE"})
    return BENCH + syms


def fetch(batch):
    for attempt in range(4):
        try:
            raw = yf.download(batch, start=START, end=END, auto_adjust=True,
                              group_by="ticker", threads=True, progress=False)
            break
        except Exception as e:  # network hiccup → backoff
            print("retry", attempt, e, flush=True)
            time.sleep(10 * (attempt + 1))
    else:
        return pd.DataFrame()
    rows = []
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
        rows.append(df[["ticker", "Date", "Open", "High", "Low", "Close", "Volume"]])
    return pd.concat(rows) if rows else pd.DataFrame()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    syms = universe()
    batches = [syms[i:i + BATCH] for i in range(0, len(syms), BATCH)]
    print(f"{len(syms)} symbols, {len(batches)} batches", flush=True)
    for i, b in enumerate(batches):
        f = OUT / f"batch_{i:03d}.parquet"
        if f.exists():
            continue
        df = fetch(b)
        df.to_parquet(f, index=False)
        print(f"batch {i}: {df['ticker'].nunique() if len(df) else 0}/{len(b)} tickers", flush=True)
        time.sleep(1.5)
    (OUT / "manifest.json").write_text(json.dumps(dict(
        start=START, end_exclusive=END, auto_adjust=True, snapshot=str(SNAP.name),
        fetched_at=pd.Timestamp.utcnow().isoformat(), n_symbols=len(syms)), indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
