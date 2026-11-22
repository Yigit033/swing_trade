"""Ek H verisi: SEC EDGAR çeyreklik indeks (13D + seyreltme formları), CIK→ticker, FINRA açığa satış.

Usage: python lab/fetch_altdata.py [edgar] [finra]
Output: output/lab/altdata/edgar_events.parquet, cik_tickers.json, finra_short.parquet
SEC adil kullanım: ≤ 8 istek/sn, kişisel bilgi içermeyen tanımlayıcı User-Agent.
"""
import gzip
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output/lab/altdata"
UA = "swing-trade-research-client admin@research-client.invalid"
FORMS_13D = {"SC 13D", "SCHEDULE 13D"}
FORMS_DILUTION = {"S-1", "S-3", "S-3ASR", "F-1", "F-3", "F-3ASR",
                  "424B1", "424B2", "424B3", "424B4", "424B5", "424B7"}
QUARTERS = [(y, q) for y in range(2020, 2027) for q in (1, 2, 3, 4) if (y, q) >= (2020, 3)]


def _get(url, ua=UA, tries=4):
    for k in range(tries):
        try:
            with urlopen(Request(url, headers={"User-Agent": ua, "Accept-Encoding": "identity"}), timeout=60) as r:
                return r.read()
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def edgar():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for y, q in QUARTERS:
        try:
            raw = _get(f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/master.gz")
        except Exception as e:
            print(f"{y}Q{q}: {e}", flush=True)       # henüz yayımlanmamış çeyrek
            continue
        text = gzip.decompress(raw).decode("latin-1")
        n = 0
        for line in text.splitlines():
            parts = line.split("|")
            if len(parts) != 5 or not parts[0].isdigit():
                continue
            cik, company, form, date, fname = parts
            if form in FORMS_13D or form in FORMS_DILUTION:
                rows.append((form, int(cik), company, date, fname))
                n += 1
        print(f"{y}Q{q}: {n} rows", flush=True)
        time.sleep(0.2)
    ev = pd.DataFrame(rows, columns=["form", "cik", "company", "date", "filename"])
    ev["date"] = pd.to_datetime(ev["date"])
    ev.to_parquet(OUT / "edgar_events.parquet", index=False)
    tick = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
    m = {}
    for v in tick.values():
        m.setdefault(str(int(v["cik_str"])), []).append(str(v["ticker"]).upper().replace(".", "-"))
    (OUT / "cik_tickers.json").write_text(json.dumps(m))
    print("edgar events", len(ev), ev.form.value_counts().to_dict(), "| ciks", len(m), flush=True)


def finra():
    from lab.common import panels, BENCH
    OUT.mkdir(parents=True, exist_ok=True)
    P = panels()
    days = [d for d in P["Close"].index if d >= pd.Timestamp("2020-12-01")]
    universe = set(P["Close"].columns) - set(BENCH)

    def one(d):
        url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d:%Y%m%d}.txt"
        try:
            df = pd.read_csv(io.BytesIO(_get(url, ua="Mozilla/5.0 research-client")), sep="|",
                             usecols=["Date", "Symbol", "ShortVolume", "TotalVolume"], dtype={"Symbol": str})
        except Exception as e:
            return d, None, str(e)[:80]
        df = df[df.Symbol.notna()]
        df["Symbol"] = df.Symbol.str.upper().str.replace(".", "-", regex=False)
        df = df[df.Symbol.isin(universe)]
        df["Date"] = d
        return d, df, None

    frames, errs = [], []
    with ThreadPoolExecutor(6) as ex:
        for i, (d, df, e) in enumerate(ex.map(one, days), 1):
            if df is None:
                errs.append((str(d.date()), e))
            else:
                frames.append(df)
            if i % 200 == 0:
                print(f"finra {i}/{len(days)} errors={len(errs)}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT / "finra_short.parquet", index=False)
    print(f"finra rows {len(out)} days {out.Date.nunique()} errors {len(errs)} {errs[:3]}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["edgar", "finra"]
    if "edgar" in which:
        edgar()
    if "finra" in which:
        finra()
