"""yfinance earnings history WITH timestamps and surprise (H5). Resumable, rate-limit safe.
Usage: python lab/fetch_earnings_full.py tickers.txt → output/lab/earnings_full/<T>.json"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yfinance as yf

OUT = Path(__file__).resolve().parents[1] / "output/lab/earnings_full"


def one(t):
    f = OUT / f"{t}.json"
    if f.exists():
        return
    for attempt in range(5):
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=40)
            rows = [] if e is None else [
                dict(ts=str(k), est=None if v.iloc[0] != v.iloc[0] else float(v.iloc[0]),
                     rep=None if v.iloc[1] != v.iloc[1] else float(v.iloc[1]),
                     surprise=None if v.iloc[2] != v.iloc[2] else float(v.iloc[2]))
                for k, v in e[["EPS Estimate", "Reported EPS", "Surprise(%)"]].iterrows()]
            f.write_text(json.dumps({"ticker": t, "rows": rows}))
            time.sleep(0.3)
            return
        except Exception as ex:
            msg = str(ex)
            if "Rate" in msg or "Too Many" in msg or "NoneType" in msg:
                time.sleep(30 * (attempt + 1))
                continue
            f.write_text(json.dumps({"ticker": t, "rows": [], "error": msg[:200]}))
            return


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ticks = Path(sys.argv[1]).read_text().split()
    with ThreadPoolExecutor(3) as ex:
        for i, _ in enumerate(ex.map(one, ticks), 1):
            if i % 200 == 0:
                print(f"{i}/{len(ticks)}", flush=True)
    print("EARN_FULL_DONE", flush=True)


if __name__ == "__main__":
    main()
