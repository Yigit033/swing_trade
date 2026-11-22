"""Canlı pipeline'ın katman katman tanısı (TANI — strateji değişikliği gerekçesi DEĞİL).

Lab v3 H0 replikasının kayıtları (2021-01 → 2026-11, canlı Finviz evreni emülasyonu):
  outcomes_emulated : her hisse-gün için motorun İLK eleme aşaması (veya SIGNAL)
  signals/selected/trades_emulated : motor sonrası canlı seçim, giriş ve çıkış (bilanço onarılmış)

Ölçü: sinyal günü d için fwd = Close[d+H] / Open[d+1] − 1 (H = 8 seans = canlı ortanca tutuş),
fark = fwd − aynı gün havuz (Bölüm 0) ortalaması. Bir katman para kazandırıyorsa ELEDİKLERİ,
GEÇİRDİKLERİNDEN kötü gitmeli. Güven aralığı: gün-kümeli bootstrap.

Usage: python lab/pipeline_audit.py
Output: output/lab/pipeline_audit.json (+ yazdırılan tablolar)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.common import EVAL_START, earnings_dates, panels, pool_mask  # noqa: E402

H0 = ROOT / "output/lab/h0"
H = 8
COST = 0.50
ENGINE_ORDER = [("filter_failed", "Sert filtreler (fiyat/cap/dolar hacim)"),
                ("no_trigger", "Tetik (VCE / RVOL)"),
                ("swing_not_ready", "Swing onayı (MA20 + 5g momentum)"),
                ("rsi_gate", "RSI kapısı"),
                ("stage_rejected", "Weinstein aşama kapısı"),
                ("quality", "Kalite eşiği (tip tabanları)")]


def forward(P, h):
    C, O = P["Close"], P["Open"]
    return (C.shift(-h) / O.shift(-1) - 1) * 100


def boot(values, days, n=1000, seed=3):
    df = pd.DataFrame({"v": values, "d": days}).dropna()
    if len(df) < 5:
        return (np.nan, np.nan)
    g = df.groupby("d")["v"].agg(["sum", "count"])
    s, c, k = g["sum"].values, g["count"].values, len(g)
    rng = np.random.default_rng(seed)
    m = [s[i].sum() / c[i].sum() for i in (rng.integers(0, k, k) for _ in range(n))]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(df, col="excess"):
    lo, hi = boot(df[col].values, df["date"].values)
    return dict(n=int(df[col].notna().sum()), mean=round(float(df[col].mean()), 3),
                ci=[round(lo, 3), round(hi, 3)], fwd=round(float(df["fwd"].mean()), 3))


def attach(df, fwd, pool_mean):
    dates, cols = fwd.index, fwd.columns
    di = dates.get_indexer(pd.DatetimeIndex(df["date"]))
    ci = cols.get_indexer(df["ticker"])
    ok = (di >= 0) & (ci >= 0)
    arr = fwd.values
    f = np.full(len(df), np.nan)
    f[ok] = arr[di[ok], ci[ok]]
    df = df.copy()
    df["fwd"] = f
    df["excess"] = f - pool_mean.reindex(pd.DatetimeIndex(df["date"])).values
    return df


def main():
    P = panels()
    pool = pool_mask().reindex(columns=P["Close"].columns, fill_value=False)
    fwd = forward(P, H)
    pool_mean = fwd.where(pool).mean(axis=1)
    out = {"horizon_sessions": H}

    # ── evren: canlı Finviz emülasyonu üyeleri vs havuz ──
    o = pd.read_parquet(H0 / "outcomes_emulated.parquet")
    o = o[pd.DatetimeIndex(o.date) >= EVAL_START]
    o = attach(o, fwd, pool_mean)
    o["stage"] = o.reject.where(~o.reject.str.startswith("quality_type"), "quality")
    pool_rows = fwd.where(pool).stack()
    out["pool_fwd_mean"] = round(float(pool_rows[pool_rows.index.get_level_values(0) >= EVAL_START].mean()), 3)
    out["universe"] = summarize(o)

    # ── motor içi huni ──
    funnel, alive = [], o
    for code, label in ENGINE_ORDER:
        dropped = alive[alive.stage == code]
        alive = alive[alive.stage != code]
        funnel.append(dict(stage=label, kept=summarize(alive), dropped=summarize(dropped)))

    # ── motor sonrası: bilanço → rejim eşiği + top N → kapasite/cooldown → gap → giriş ──
    from swing_trader.small_cap.filters import SmallCapFilters
    from swing_trader.small_cap.settings_config import SmallCapSettings
    filt = SmallCapFilters(None, SmallCapSettings())
    sig = attach(pd.read_parquet(H0 / "signals_emulated.parquet"), fwd, pool_mean)
    keep = []
    for r in sig.itertuples():
        ed = earnings_dates(r.ticker)
        keep.append(ed is not None and len(ed) > 0 and
                    filt.check_earnings(r.ticker, r.date.to_pydatetime(), list(ed))[0])
    sig["earn_ok"] = keep
    sel = pd.read_parquet(H0 / "selected_emulated.parquet")
    sel_key = set(zip(pd.DatetimeIndex(sel.date), sel.ticker))
    T = pd.read_parquet(H0 / "trades_emulated.parquet")
    t_key = {(pd.Timestamp(a), b): s for a, b, s in zip(T.signal_date, T.ticker, T.status)}
    sig["selected"] = [(pd.Timestamp(d), t) in sel_key for d, t in zip(sig.date, sig.ticker)]
    sig["tracked"] = [(pd.Timestamp(d), t) in t_key for d, t in zip(sig.date, sig.ticker)]
    sig["opened"] = [t_key.get((pd.Timestamp(d), t)) not in (None, "REJECTED")
                     for d, t in zip(sig.date, sig.ticker)]
    post = [("Bilanço kapısı", sig, sig.earn_ok)]
    s1 = sig[sig.earn_ok]
    post.append(("Rejim eşiği + top N", s1, s1.selected))
    s2 = s1[s1.selected]
    post.append(("Kapasite / aynı hisse / cooldown", s2, s2.tracked))
    s3 = s2[s2.tracked]
    post.append(("Gap filtresi (+5 / −7)", s3, s3.opened))
    for label, base, mask in post:
        funnel.append(dict(stage=label, kept=summarize(base[mask]), dropped=summarize(base[~mask])))
    out["funnel"] = funnel

    # ── çıkış motoru: aynı girişler, gerçekleşen çıkış vs sabit tutuş ──
    tr = T[~T.status.isin(["REJECTED", "OPEN", "REVIEW"])].copy()
    tr = tr.rename(columns={"signal_date": "date"})
    tr = attach(tr, fwd, pool_mean)
    exit_rows = {"Gerçekleşen (canlı çıkış motoru)": float(tr.ret_pct.mean()),
                 f"Sabit {H} seans tut (stop/hedef yok)": float(tr.fwd.mean())}
    hm = []
    C = P["Close"]
    idx = {d: i for i, d in enumerate(C.index)}
    for r in tr.itertuples():
        i = idx.get(pd.Timestamp(r.date))
        j = None if i is None else i + int(r.hold)
        hm.append(np.nan if j is None or j >= len(C.index) else
                  (C[r.ticker].iloc[j] / r.entry_price - 1) * 100)
    tr["fwd_holdmax"] = hm
    exit_rows["Tip süresi kadar tut (stop/hedef yok)"] = float(np.nanmean(hm))
    by_status = tr.groupby("status").agg(n=("ret_pct", "size"), realized=("ret_pct", "mean"),
                                         fixed=("fwd", "mean")).round(2)
    out["exit"] = dict(mean_gross=dict((k, round(v, 3)) for k, v in exit_rows.items()),
                       by_status=by_status.reset_index().to_dict("records"), n=len(tr))

    # ── kalite skoru: yüksek skor daha iyi mi? (motorun skorladığı her şey) ──
    q = o[o.quality.notna()].copy()
    q["bin"] = pd.cut(q.quality, [0, 60, 70, 78, 82, 86, 101], right=False)
    out["quality_bins"] = [dict(bin=str(b), **summarize(g)) for b, g in q.groupby("bin", observed=True)]

    # ── sinyaller: tetik yolu, rejim, yıl ──
    for key, col in (("pathway", "pathway"), ("regime", "regime")):
        out[key] = [dict(k=str(k), **summarize(g)) for k, g in sig.groupby(col)]
    sig["year"] = pd.DatetimeIndex(sig.date).year
    out["signal_year"] = [dict(k=int(k), **summarize(g)) for k, g in sig.groupby("year")]

    (ROOT / "output/lab/pipeline_audit.json").write_text(json.dumps(out, indent=1, default=str))
    _print(out)


def _print(out):
    u = out["universe"]
    print(f"Havuz ort. {H}-seans getiri: {out['pool_fwd_mean']:+.2f}% | canlı evren (Finviz emülasyonu) "
          f"n={u['n']} fark {u['mean']:+.2f} CI{u['ci']}")
    print(f"\n{'Katman':40s} {'geçen n':>8s} {'geçen fark':>11s} {'elenen n':>9s} {'elenen fark':>12s}  CI(elenen)")
    for f in out["funnel"]:
        k, d = f["kept"], f["dropped"]
        print(f"{f['stage']:40s} {k['n']:8d} {k['mean']:+11.2f} {d['n']:9d} {d['mean']:+12.2f}  {d['ci']}")
    print("\nÇıkış:", out["exit"]["mean_gross"], "n=", out["exit"]["n"])
    print(pd.DataFrame(out["exit"]["by_status"]).to_string(index=False))
    for key in ("quality_bins", "pathway", "regime", "signal_year"):
        print(f"\n{key}:")
        for r in out[key]:
            name = r.get("bin", r.get("k"))
            print(f"  {str(name):14s} n={r['n']:6d} fark {r['mean']:+.2f} CI{r['ci']} brüt {r['fwd']:+.2f}")


if __name__ == "__main__":
    main()
