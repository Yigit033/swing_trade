"""Ek H — H7 (aktivist 13D), H8 (seyreltme filtresi), H10 (yoğun açığa satış filtresi).

Usage:
  python lab/info_edgar.py build      # dışlama listeleri + H7 işlemleri
  python lab/h0_sim.py --exclude output/lab/altdata/ex_dilution.parquet
  python lab/h0_sim.py --exclude output/lab/altdata/ex_shortvol.parquet
  python lab/info_edgar.py evaluate   # ön kayıtlı kararlar
"""
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.challengers import OUT as CH, atr14, simulate, walk  # noqa: E402
from lab.common import EVAL_START, earnings_dates, indicators, panels, pool_mask  # noqa: E402
from lab.info_signals import atr_exit  # noqa: E402

ALT = ROOT / "output/lab/altdata"
H0 = ROOT / "output/lab/h0"
UA = "swing-trade-research-client admin@research-client.invalid"
FORMS_13D = {"SC 13D", "SCHEDULE 13D"}


def _universe_map():
    m = json.loads((ALT / "cik_tickers.json").read_text())
    cols = set(panels()["Close"].columns)
    return {int(k): [t for t in v if t in cols] for k, v in m.items() if any(t in cols for t in v)}


def _subject_from_header(filename):
    """İki taraf da listeli ise başvuru başlığındaki SUBJECT COMPANY CIK'i."""
    url = "https://www.sec.gov/Archives/" + filename
    for k in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                head = r.read(30000).decode("latin-1", "ignore")
            break
        except Exception:
            time.sleep(1 + k)
    else:
        return None
    time.sleep(0.15)
    i = head.find("SUBJECT COMPANY:")
    if i < 0:
        return None
    j = head.find("CENTRAL INDEX KEY:", i)
    return int(head[j + 18: j + 40].split()[0]) if j > 0 else None


def h7_events(ev, umap):
    d = ev[ev.form.isin(FORMS_13D)]
    out, ambiguous = [], 0
    for fname, g in d.groupby("filename"):
        listed = [c for c in g.cik.unique() if c in umap]
        if not listed:
            continue
        if len(listed) > 1:
            ambiguous += 1
            subj = _subject_from_header(fname)
            if subj not in listed:
                continue
            listed = [subj]
        out.append(dict(ticker=umap[listed[0]][0], F=g.date.iloc[0], filename=fname))
    e = pd.DataFrame(out).sort_values(["ticker", "F"])
    keep, last = [], {}
    for r in e.itertuples():                      # aynı hissede 60 gün içinde yeni olay yok
        prev = last.get(r.ticker)
        if prev is None or (r.F - prev).days > 60:
            keep.append(True)
            last[r.ticker] = r.F
        else:
            keep.append(False)
    print(f"H7: 13D filings mapped {len(e)} (ambiguous resolved via header {ambiguous}), events {sum(keep)}")
    return e[keep]


def h7_trades(ev7):
    P, pool = panels(), pool_mask()
    pool = pool.reindex(columns=P["Close"].columns, fill_value=False)
    cal, a, dv = P["Close"].index, atr14(P), indicators()["dv20"]
    rows = []
    for r in ev7.itertuples():
        i = cal.searchsorted(r.F, side="right") - 1
        if i < 0:
            continue
        t = cal[i]
        if t < EVAL_START or not bool(pool.at[t, r.ticker]):
            continue
        rows.append(dict(date=t, ticker=r.ticker, key=float(dv.at[t, r.ticker]), atr=float(a.at[t, r.ticker])))
    c = pd.DataFrame(rows).sort_values(["date", "key"], ascending=[True, False])
    print(f"H7 in-pool candidates {len(c)}")
    return simulate(c, P, atr_exit, "H7")


def candidate_signals():
    """H0 (bilanço kapısından geçen) ∪ H6 adayları — (date, ticker) tekil."""
    from swing_trader.small_cap.filters import SmallCapFilters
    from swing_trader.small_cap.settings_config import SmallCapSettings
    filt = SmallCapFilters(None, SmallCapSettings())
    s = pd.read_parquet(H0 / "signals_emulated.parquet")
    ok = []
    for r in s.itertuples():
        ed = earnings_dates(r.ticker)
        ok.append(ed is not None and len(ed) > 0 and filt.check_earnings(r.ticker, r.date.to_pydatetime(), list(ed))[0])
    h0 = s[ok][["date", "ticker"]]
    h6 = pd.read_parquet(CH / "h6_candidates.parquet")[["date", "ticker"]]
    u = pd.concat([h0, h6]).drop_duplicates().reset_index(drop=True)
    u["date"] = pd.DatetimeIndex(u.date)
    return u


def dilution_exclusions(ev, umap, sig):
    d = ev[~ev.form.isin(FORMS_13D)]
    d = d[d.cik.isin(umap)]
    by = {}
    for c, g in d.groupby("cik"):
        for t in umap[c]:
            by.setdefault(t, []).extend(g.date.tolist())
    by = {t: np.array(sorted(v), dtype="datetime64[ns]") for t, v in by.items()}
    drop = []
    for r in sig.itertuples():
        arr = by.get(r.ticker)
        if arr is None:
            continue
        lo, hi = np.datetime64(r.date - pd.Timedelta(days=30)), np.datetime64(r.date)
        k = np.searchsorted(arr, lo, side="left")
        if k < len(arr) and arr[k] <= hi:
            drop.append((r.date, r.ticker))
    return pd.DataFrame(drop, columns=["date", "ticker"])


def shortvol_exclusions(sig):
    f = pd.read_parquet(ALT / "finra_short.parquet")
    f = f[f.TotalVolume > 0]
    f["ratio"] = f.ShortVolume / f.TotalVolume
    P = panels()
    R = f.pivot_table(index="Date", columns="Symbol", values="ratio", aggfunc="last")
    R = R.reindex(index=P["Close"].index)
    svr5 = R.rolling(5, min_periods=3).mean()
    pool = pool_mask().reindex(index=svr5.index, columns=svr5.columns, fill_value=False)
    thr = svr5.where(pool).quantile(0.80, axis=1)
    drop, nodata = [], 0
    for r in sig.itertuples():
        v = svr5.at[r.date, r.ticker] if (r.date in svr5.index and r.ticker in svr5.columns) else np.nan
        if not np.isfinite(v):
            nodata += 1
            continue
        if v >= thr.get(r.date, np.inf):
            drop.append((r.date, r.ticker))
    print(f"H10: signals without FINRA data {nodata}")
    return pd.DataFrame(drop, columns=["date", "ticker"])


def build(part="all"):
    if part in ("all", "edgar"):
        ev = pd.read_parquet(ALT / "edgar_events.parquet")
        umap = _universe_map()
        ev7 = h7_events(ev, umap)
        ev7.to_parquet(ALT / "events_H7.parquet", index=False)
        T7 = h7_trades(ev7)
        T7.to_parquet(CH / "trades_H7.parquet", index=False)
        sig = candidate_signals()
        sig.to_parquet(ALT / "candidate_signals.parquet", index=False)
        exd = dilution_exclusions(ev, umap, sig)
        exd.to_parquet(ALT / "ex_dilution.parquet", index=False)
        print(f"candidates {len(sig)} | dilution excluded {len(exd)} | H7 trades {len(T7)}")
    if part in ("all", "short"):
        sig = pd.read_parquet(ALT / "candidate_signals.parquet")
        exq = shortvol_exclusions(sig)
        exq.to_parquet(ALT / "ex_shortvol.parquet", index=False)
        print(f"candidates {len(sig)} | shortvol excluded {len(exq)}")


def _drop_eval(sig, ex):
    from lab.pipeline_audit import attach, boot, forward
    P = panels()
    fwd = forward(P, 8)
    pool = pool_mask().reindex(columns=P["Close"].columns, fill_value=False)
    s = attach(sig, fwd, fwd.where(pool).mean(axis=1))
    key = set(zip(pd.DatetimeIndex(ex.date), ex.ticker))
    s["dropped"] = [(pd.Timestamp(d), t) in key for d, t in zip(s.date, s.ticker)]
    out = {}
    for flag, g in s.groupby("dropped"):
        lo, hi = boot(g.excess.values, g.date.values)
        out["dropped" if flag else "kept"] = dict(n=int(g.excess.notna().sum()), excess8=round(float(g.excess.mean()), 3),
                                                  ci=[round(lo, 3), round(hi, 3)])
    return out


def evaluate_all():
    from lab.d1_eval import load
    from lab.evaluate import evaluate
    res = {}
    T7 = pd.read_parquet(CH / "trades_H7.parquet")
    r7, _, _ = evaluate("H7", T7)
    res["H7"] = r7
    p = r7["portfolio_10pct"]
    print(f"H7 n={r7['n']} months={r7['months']} gross={r7['gross_mean']} net50={r7['net_mean']} "
          f"excess={r7['excess_mean']} CI{r7['excess_ci95']} win={r7['win_rate']} port={p['total_pct']} "
          f"dd={p['max_dd_pct']} PROMOTED={r7['promoted']} {r7['criteria']}")
    base, _, _ = evaluate("H0", load(""))
    bp = base["portfolio_10pct"]
    print(f"H0  n={base['n']} net50={base['net_mean']} port={bp['total_pct']} dd={bp['max_dd_pct']}")
    sig = pd.read_parquet(ALT / "candidate_signals.parquet")
    P = panels()
    h6c = pd.read_parquet(CH / "h6_candidates.parquet")
    for name, ex_file, tag in (("H8 seyreltme", "ex_dilution", "_exex_dilution"),
                               ("H10 açığa satış", "ex_shortvol", "_exex_shortvol")):
        ex = pd.read_parquet(ALT / f"{ex_file}.parquet")
        rf, _, _ = evaluate(name, load(tag))
        fp = rf["portfolio_10pct"]
        key = set(zip(pd.DatetimeIndex(ex.date), ex.ticker))
        c6 = h6c[[(pd.Timestamp(d), t) not in key for d, t in zip(h6c.date, h6c.ticker)]]
        T6 = simulate(c6, P, lambda ie, arr, row: walk(ie, arr, None, 8), "H6f")
        r6, _, _ = evaluate("H6f", T6)
        de = _drop_eval(sig, ex)
        a = rf["net_mean"] > base["net_mean"] and fp["total_pct"] > bp["total_pct"]
        d = de.get("dropped", {})
        b = bool(d and d["excess8"] < 0 and d["ci"][1] < 0)
        res[name] = dict(h0_filtered=dict(n=rf["n"], net=rf["net_mean"], port=fp["total_pct"], dd=fp["max_dd_pct"]),
                         h6_filtered=dict(n=r6["n"], net=r6["net_mean"], port=r6["portfolio_10pct"]["total_pct"],
                                          dd=r6["portfolio_10pct"]["max_dd_pct"]),
                         dropped_vs_kept=de, a=a, b=b, decision="PRODUCTION_GATE" if (a and b) else "REJECTED")
        print(f"{name}: H0f n={rf['n']} net50={rf['net_mean']} port={fp['total_pct']} dd={fp['max_dd_pct']} | "
              f"H6f n={r6['n']} net50={r6['net_mean']} port={r6['portfolio_10pct']['total_pct']} "
              f"dd={r6['portfolio_10pct']['max_dd_pct']} | elenen {d} kalan {de.get('kept')} | a={a} b={b} "
              f"→ {res[name]['decision']}")
    (ROOT / "output/lab/h_report.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2] if len(sys.argv) > 2 else "all")
    else:
        evaluate_all()
