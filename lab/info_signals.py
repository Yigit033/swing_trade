"""Pre-registered information-based swing signals H4 (insider clusters) and H5 (PEAD).
See lab/PREREGISTRATION.md, Ek A. Parameters are FIXED.

Usage: python lab/info_signals.py [H4] [H5] [--pool large]
Output: output/lab/challengers/trades_H4.parquet / trades_H5.parquet (+ events_*.parquet)
        --pool large (Ek D2): trades_H4L / trades_H5L (+ events_H4L / events_H5L), havuz L
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.common import panels, pool_mask, pool_mask_large, EVAL_END, EVAL_START, LAB  # noqa: E402
from lab.challengers import simulate, walk, atr14, OUT  # noqa: E402


def _norm(sym):
    return str(sym).strip().upper().replace(".", "-")


def insider_purchases() -> pd.DataFrame:
    rows = []
    for z in sorted((LAB / "sec").glob("*.zip")):
        zf = zipfile.ZipFile(z)
        rd = lambda n, cols: pd.read_csv(zf.open(n), sep="\t", dtype=str, usecols=cols,
                                         on_bad_lines="skip", quoting=3)
        sub = rd("SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE",
                                    "ISSUERTRADINGSYMBOL"])
        own = rd("REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
        tr = rd("NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES",
                                       "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
        sub = sub[sub.DOCUMENT_TYPE.str.strip() == "4"]
        tr = tr[(tr.TRANS_CODE == "P") & (tr.TRANS_ACQUIRED_DISP_CD == "A")].copy()
        tr["val"] = pd.to_numeric(tr.TRANS_SHARES, errors="coerce") * \
            pd.to_numeric(tr.TRANS_PRICEPERSHARE, errors="coerce")
        tr = tr[tr.val > 0].groupby("ACCESSION_NUMBER", as_index=False).val.sum()
        rel = own.RPTOWNER_RELATIONSHIP.fillna("")
        own = own[rel.str.contains("Director") | rel.str.contains("Officer")]
        m = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER")
        rows.append(m)
    df = pd.concat(rows).drop_duplicates(["ACCESSION_NUMBER", "RPTOWNERCIK"])
    df["F"] = pd.to_datetime(df.FILING_DATE, format="%d-%b-%Y", errors="coerce")
    df["ticker"] = df.ISSUERTRADINGSYMBOL.map(_norm)
    # A joint filing lists several owners for ONE purchase: count the value once per accession.
    df["val_once"] = np.where(df.duplicated("ACCESSION_NUMBER"), 0.0, df.val)
    return df.dropna(subset=["F"])


def h4_events(purch: pd.DataFrame) -> pd.DataFrame:
    ev = []
    for t, g in purch.groupby("ticker"):
        g = g.sort_values("F")
        last = None
        for F in sorted(g.F.unique()):
            F = pd.Timestamp(F)
            w = g[(g.F > F - pd.Timedelta(days=30)) & (g.F <= F)]
            if w.RPTOWNERCIK.nunique() >= 2 and w.val_once.sum() >= 100_000:
                if last is None or (F - last).days > 60:
                    ev.append(dict(ticker=t, F=F, insiders=w.RPTOWNERCIK.nunique(),
                                   value=float(w.val_once.sum())))
                    last = F
    return pd.DataFrame(ev)


def to_candidates(ev, date_col, key_col, P, pool):
    """Map an information timestamp to signal bar t (= last session ≤ date) inside the pool."""
    cal = P["Close"].index
    a = atr14(P)
    rows = []
    for r in ev.itertuples():
        t = r.ticker
        if t not in P["Close"].columns:
            continue
        i = cal.searchsorted(getattr(r, date_col), side="right") - 1
        if i < 0:
            continue
        d = cal[i]
        if d < EVAL_START or (EVAL_END is not None and d > EVAL_END) or not bool(pool.at[d, t]):
            continue
        rows.append(dict(date=d, ticker=t, key=float(getattr(r, key_col)), atr=float(a.at[d, t])))
    c = pd.DataFrame(rows)
    return c.sort_values(["date", "key"], ascending=[True, False]) if len(c) else c


def atr_exit(ie, arr, r):
    if not np.isfinite(r.atr) or r.atr <= 0:
        return None
    return walk(ie, arr, arr[0][ie] - 2.5 * r.atr, 20)


def run_h4(P, pool, name="H4"):
    ev = h4_events(insider_purchases())
    ev.to_parquet(OUT / f"events_{name}.parquet", index=False)
    cands = to_candidates(ev, "F", "value", P, pool)
    print(f"{name} events {len(ev)}, in-pool candidates {len(cands)}", flush=True)
    return simulate(cands, P, atr_exit, name)


def h5_events(P):
    cal = P["Close"].index
    C = P["Close"]
    ev = []
    for f in (LAB / "earnings_full").glob("*.json"):
        rec = json.loads(f.read_text())
        t = rec["ticker"]
        if t not in C.columns:
            continue
        for x in rec.get("rows", []):
            s, est, rep = x.get("surprise"), x.get("est"), x.get("rep")
            if s is None or est is None or rep is None:
                continue
            if not (s >= 10 and rep > est and abs(est) >= 0.02):
                continue
            ts = pd.Timestamp(x["ts"])
            ts = ts.tz_convert("America/New_York") if ts.tzinfo else ts.tz_localize("America/New_York")
            day = pd.Timestamp(ts.date())
            before_open = (ts.hour, ts.minute) < (9, 30)
            j = cal.searchsorted(day, side="left" if before_open else "right")
            if j <= 0 or j >= len(cal):
                continue
            R = cal[j]
            if not (C.at[R, t] > C.at[cal[j - 1], t]):
                continue
            ev.append(dict(ticker=t, R=R, surprise=s, hour=ts.hour + ts.minute / 60))   # hour: Ek S veri kalitesi
    return pd.DataFrame(ev)


def run_h5(P, pool, name="H5"):
    ev = h5_events(P)
    ev.to_parquet(OUT / f"events_{name}.parquet", index=False)
    cands = to_candidates(ev, "R", "surprise", P, pool)
    print(f"{name} events {len(ev)}, in-pool candidates {len(cands)}", flush=True)
    return simulate(cands, P, atr_exit, name)


def main():
    args = sys.argv[1:]
    large = "--pool" in args and args[args.index("--pool") + 1] == "large"
    args = [a for a in args if a not in ("--pool", "large")]
    P = panels()
    pool = (pool_mask_large() if large else pool_mask()).reindex(columns=P["Close"].columns, fill_value=False)
    OUT.mkdir(parents=True, exist_ok=True)
    which = args or ["H4", "H5"]
    for name, fn in (("H4", run_h4), ("H5", run_h5)):
        if name not in which:
            continue
        name = name + ("L" if large else "")
        T = fn(P, pool, name)
        T.to_parquet(OUT / f"trades_{name}.parquet", index=False)
        print(name, len(T), f"mean {T.ret_pct.mean():.2f}% median {T.ret_pct.median():.2f}%", flush=True)


if __name__ == "__main__":
    main()
