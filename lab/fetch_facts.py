"""Per-ticker facts for Finviz emulation (Lab v3).

  shares   : yfinance get_shares_full() — dated shares-outstanding history
             (point-in-time-ish market cap = shares_t × Close_t)
  info     : country / sector / industry / floatShares / sharesOutstanding (CURRENT only)
  earnings : get_earnings_dates() — past + scheduled dates (only with --earnings)

Usage: python lab/fetch_facts.py tickers.txt [--earnings]
Output: output/lab/facts/<T>.json  (resumable; existing files skipped)
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/lab/facts"
EOUT = ROOT / "output/lab/earnings"
INFO_KEYS = ("country", "sector", "industry", "floatShares", "sharesOutstanding",
             "impliedSharesOutstanding", "quoteType", "marketCap")


def _call(fn, tries=5):
    """Retry with long backoff on rate limits; raise the last error otherwise."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            limited = "Rate" in msg or "Too Many" in msg or "NoneType" in msg
            if attempt == tries - 1 or not limited:
                raise
            time.sleep(30 * (attempt + 1))


def facts(t):
    f = OUT / f"{t}.json"
    if f.exists():
        return "skip"
    rec = {"ticker": t}
    tk = yf.Ticker(t)
    try:
        s = _call(lambda: tk.get_shares_full(start="2019-06-01"))
        if s is not None and len(s):
            s = s[~s.index.duplicated(keep="last")].sort_index()
            rec["shares"] = {str(k.date()): float(v) for k, v in s.items()}
        info = _call(lambda: tk.info) or {}
        rec["info"] = {k: info.get(k) for k in INFO_KEYS}
    except Exception as e:
        if "Rate" in str(e) or "Too Many" in str(e) or "NoneType" in str(e):
            return "rate_limited"          # do NOT persist; a rerun will retry
        rec["error"] = str(e)[:200]
    f.write_text(json.dumps(rec))
    time.sleep(0.3)
    return "ok"


def earnings(t):
    f = EOUT / f"{t}.json"
    if f.exists():
        return "skip"
    rec = {"ticker": t}
    for attempt in range(3):
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=40)
            rec["dates"] = [] if e is None else sorted({str(k.date()) for k in e.index})
            break
        except Exception as ex:
            rec["error"] = str(ex)[:200]
            time.sleep(3 * (attempt + 1))
    f.write_text(json.dumps(rec))
    return "ok"


def main():
    tickers = [x.strip() for x in Path(sys.argv[1]).read_text().split() if x.strip()]
    fn = earnings if "--earnings" in sys.argv else facts
    (EOUT if fn is earnings else OUT).mkdir(parents=True, exist_ok=True)
    done = 0
    with ThreadPoolExecutor(3) as ex:
        for _ in ex.map(fn, tickers):
            done += 1
            if done % 200 == 0:
                print(f"{done}/{len(tickers)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
