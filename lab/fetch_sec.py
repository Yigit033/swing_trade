"""Download SEC DERA insider transaction data sets (Form 3/4/5), 2020Q3 → latest published.
Generic client User-Agent (no personal data). Polite: sequential, 1 req/sec."""
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "output/lab/sec"
UA = "swing-trade-research-client admin@research-client.invalid"
URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{}q{}_form345.zip"

OUT.mkdir(parents=True, exist_ok=True)
for y in range(2020, 2027):
    for q in range(1, 5):
        if (y, q) < (2020, 3):
            continue
        f = OUT / f"{y}q{q}.zip"
        if f.exists() and f.stat().st_size > 1000:
            continue
        try:
            req = urllib.request.Request(URL.format(y, q), headers={"User-Agent": UA})
            data = urllib.request.urlopen(req, timeout=120).read()
            f.write_bytes(data)
            print(y, q, len(data), flush=True)
        except Exception as e:
            print(y, q, "MISSING", e, flush=True)
        time.sleep(1)
print("SEC_DONE", flush=True)
