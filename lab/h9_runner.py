"""Ek P — H9 "Büyük Koşu Avcısı": 35 gösterge → büyük koşu (+%50 / 20 seans) olasılığı, görülmemiş yıllarda sınav.
H9b (Ek P2): aynı her şey, hedef = gerçek işlemin getirisi (2.5×ATR stop, ≤20 seans) → regresyon.

Eğitim 2021-01-04 → 2023-11-30, test 2024-01-02 → son−20. Model/eşik test yarısına bakmadan sabit.

Usage: python lab/h9_runner.py [run|money]
Output: output/lab/p_h9{,b}.json, output/lab/h9{,b}_test_candidates.parquet
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.challengers import OUT, rsi_wilder, walk  # noqa: E402
from lab.common import EVAL_START, earnings_dates, indicators, panels, pool_mask, shares_panel  # noqa: E402
from lab.core_n import iwm_hold  # noqa: E402
from lab.evaluate import attach_benchmark, boot_ci  # noqa: E402
from lab.h6r import STOP_ATR, regime_series  # noqa: E402
from lab.swing_o import portfolio, simulate as simulate_h6s  # noqa: E402
import lab.h0_signals as h0  # noqa: E402

W, RUN, CRASH = 20, 0.50, -0.30
TRAIN_END, TEST_START = pd.Timestamp("2023-11-30"), pd.Timestamp("2024-01-02")
FRAC, COST = 0.15, 0.50


def features(P, pool, regimes, emit):
    """Her göstergeyi hesaplayıp hemen emit(ad, tablo) ile çıkarır (bellek: aynı anda tek tablo)."""
    O, H, L, C, V = (P[f] for f in ("Open", "High", "Low", "Close", "Volume"))
    I = indicators()
    iwm = C["IWM"]
    r1 = C.pct_change(fill_method=None)
    R = {n: C / C.shift(n) - 1 for n in (1, 5, 20, 60, 120)}
    for n, f in R.items():
        emit(f"r{n}", f)
    sma200 = C.rolling(200, min_periods=150).mean()
    for k, s in (("sma20", I["sma20"]), ("sma50", I["sma50"]), ("sma200", sma200)):
        emit(f"c_{k}", C / s - 1)
    hi252 = H.rolling(252, min_periods=120).max()
    lo252 = L.rolling(252, min_periods=120).min()
    emit("dist_hi252", C / hi252 - 1)
    emit("dist_lo252", C / lo252 - 1)
    newhi = (H >= hi252).values
    rows = np.arange(len(C.index), dtype=float)[:, None]
    last = pd.DataFrame(np.where(newhi, rows, np.nan), index=C.index, columns=C.columns).ffill()
    emit("days_since_hi252", (pd.DataFrame(np.broadcast_to(rows, C.shape), index=C.index, columns=C.columns)
                             - last).clip(upper=252))
    emit("rsi14", rsi_wilder(C, 14))
    pc = C.shift(1)
    tr = np.maximum(H - L, np.maximum((H - pc).abs(), (L - pc).abs()))
    A14 = tr.rolling(14).mean()
    emit("atr_pct", A14 / C)
    emit("atr5_50", tr.rolling(5).mean() / tr.rolling(50).mean())
    bbw = C.rolling(20).std() / I["sma20"]
    emit("bbw_rel", bbw / bbw.rolling(120, min_periods=60).mean())
    emit("range_pct", (H - L) / C)
    emit("close_loc", ((C - L) / (H - L)).where(H > L, 0.5))
    emit("gap", O / pc - 1)
    emit("max_r1_20", r1.rolling(20).max())
    av63 = I["avgvol63"]
    emit("rvol", I["relvol"])
    emit("rvol5", V.rolling(5).mean() / av63.shift(1))
    emit("vol20_63", V.rolling(20).mean() / av63)
    emit("log_dv20", np.log(I["dv20"].where(I["dv20"] > 0)))
    emit("upvol20", (V * (r1 > 0)).rolling(20).sum() / V.rolling(20).sum())
    emit("log_cap", np.log(I["cap"].where(I["cap"] > 0)))
    emit("log_price", np.log(C))
    sh = shares_panel().reindex(columns=C.columns)
    emit("shares_chg252", sh / sh.shift(252) - 1)
    ir20, ir60 = iwm / iwm.shift(20) - 1, iwm / iwm.shift(60) - 1
    emit("rs20", R[20].sub(ir20, axis=0))
    emit("rs60", R[60].sub(ir60, axis=0))
    bc = lambda s: pd.DataFrame(np.broadcast_to(s.values[:, None], C.shape), index=C.index, columns=C.columns)  # noqa: E731
    emit("iwm_r20", bc(ir20))
    emit("vix", bc(C["^VIX"].ffill()))
    code = {"BULL": 2.0, "CAUTION": 1.0, "BEAR": 0.0}
    emit("regime", bc(pd.Series([code.get(regimes.get(d, ("?", ""))[0], np.nan) for d in C.index], index=C.index)))
    above = (C > I["sma50"]).where(pool)
    emit("breadth50", bc(above.mean(axis=1)))
    # bilanço takvimi: sonraki / önceki bilançoya seans (60'ta kesik)
    nxt = np.full(C.shape, np.nan, dtype=np.float32)
    prv = np.full(C.shape, np.nan, dtype=np.float32)
    ridx = np.arange(len(C.index))
    for j, t in enumerate(C.columns):
        if not pool[t].any():
            continue
        ed = earnings_dates(t)
        if ed is None or not len(ed):
            continue
        pos = np.unique(C.index.searchsorted(pd.DatetimeIndex(ed)))
        k = np.searchsorted(pos, ridx, side="left")
        ok = k < len(pos)
        nxt[ok, j] = np.minimum(pos[k[ok]] - ridx[ok], 60)
        k2 = np.searchsorted(pos, ridx, side="right") - 1
        ok2 = k2 >= 0
        prv[ok2, j] = np.minimum(ridx[ok2] - pos[k2[ok2]], 60)
    emit("earn_next", pd.DataFrame(nxt, index=C.index, columns=C.columns))
    emit("earn_prev", pd.DataFrame(prv, index=C.index, columns=C.columns))
    return A14


def build():
    P = panels()
    O, C = P["Open"], P["Close"]
    pool = pool_mask().reindex(columns=C.columns, fill_value=False).astype(bool)
    regimes = regime_series("IWM")
    entry = O.shift(-1)
    fmax = C[::-1].rolling(W, min_periods=1).max()[::-1].shift(-1) / entry - 1
    fmin = C[::-1].rolling(W, min_periods=1).min()[::-1].shift(-1) / entry - 1
    end = C.index[-W - 1]
    inwin = (C.index >= EVAL_START) & (C.index <= end)
    M = pool.values & inwin[:, None] & np.isfinite(entry.values) & (entry.values > 0) & np.isfinite(fmax.values)
    ri, ci = np.nonzero(M)
    df = pd.DataFrame({"date": C.index[ri], "ticker": C.columns[ci],
                       "run": (fmax.values[M] >= RUN), "crash": (fmin.values[M] <= CRASH)})
    cols = []

    def emit(k, f):
        x = np.asarray(f.values[M], dtype=np.float32)
        x[~np.isfinite(x)] = np.nan
        df[k] = x
        cols.append(k)

    A14 = features(P, pool, regimes, emit)
    assert len(cols) == 35, len(cols)
    df["money"] = trade_label(P, A14)[M].astype(np.float32)
    return df, cols, regimes, A14


def trade_label(P, A14):
    """Her hisse-günü için gerçek işlem getirisi (%): t+1 açılış, 2.5×ATR stop (stop önce), 20. kapanış."""
    O, L, C = P["Open"], P["Low"], P["Close"]
    entry = O.shift(-1).values
    stop = entry - STOP_ATR * A14.values
    out = np.full(entry.shape, np.nan)
    done = np.zeros(entry.shape, dtype=bool)
    for k in range(1, W + 1):
        Lk, Ok = L.shift(-k).values, O.shift(-k).values
        hit = ~done & np.isfinite(Lk) & (Lk <= stop)
        out[hit] = np.minimum(Ok[hit], stop[hit])
        done |= hit
    Ck = C.shift(-W).values
    last = ~done & np.isfinite(Ck)
    out[last] = Ck[last]
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * (out / entry - 1)


def simulate_h9(cands, P, regimes, caution_max, A14, hold=W):
    O, H, L, C = (P[f] for f in ("Open", "High", "Low", "Close"))
    idx = {d: i for i, d in enumerate(C.index)}
    pos, rows = [], []
    for r in cands.itertuples():
        i = idx[r.date]
        if i + 1 >= len(C.index):
            continue
        regime = regimes.get(r.date, ("UNKNOWN", ""))[0]
        active = [p for p in pos if p[2] > i]
        if regime == "BEAR":
            continue
        if regime == "CAUTION" and len(active) >= caution_max:
            continue
        t = r.ticker
        if any(p[0] == t for p in active):
            continue
        a = A14.at[r.date, t]
        arr = (O[t].values, H[t].values, L[t].values, C[t].values)
        o = arr[0][i + 1]
        if not np.isfinite(a) or a <= 0 or not np.isfinite(o) or o <= 0:
            continue
        xi, xp, why = walk(i + 1, arr, o - STOP_ATR * a, hold)
        at_open = bool(np.isfinite(arr[0][xi]) and xp == arr[0][xi] and xi > i + 1)
        pos.append((t, i + 1, xi))
        rows.append(dict(ticker=t, signal_date=r.date, entry_day=C.index[i + 1], exit_day=C.index[xi],
                         entry_price=o, exit_price=xp, ret_pct=100 * (xp / o - 1), reason=why, key=r.key,
                         regime=regime, at_open=at_open, bars=xi - i))
    return pd.DataFrame(rows)


def main(mode="run"):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score

    S = h0.live_settings()
    cm = int(S.backtest_loop.caution_max_concurrent)
    P = panels()
    df, cols, regimes, A14 = build()
    if mode == "money":
        df = df[np.isfinite(df.money)]
    tr, te = df[df.date <= TRAIN_END], df[df.date >= TEST_START]
    print(f"satır: eğitim {len(tr):,} (koşu %{100*tr.run.mean():.2f}) | test {len(te):,} (koşu %{100*te.run.mean():.2f})")
    hp = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
              l2_regularization=1.0, early_stopping=False, random_state=7)
    if mode == "money":
        model = HistGradientBoostingRegressor(loss="squared_error", **hp)
        model.fit(tr[cols].values, tr.money.clip(-60, 150).values)
        s_tr, s_te = model.predict(tr[cols].values), model.predict(te[cols].values)
    else:
        model = HistGradientBoostingClassifier(**hp)
        model.fit(tr[cols].values, tr.run.values)
        s_tr = model.predict_proba(tr[cols].values)[:, 1]
        s_te = model.predict_proba(te[cols].values)[:, 1]
    tau = float(np.quantile(s_tr, 0.99))
    te = te.assign(score=s_te)
    sel = te[te.score >= tau]
    res = {"prereg_sha256": hashlib.sha256((ROOT / "lab/PREREGISTRATION.md").read_bytes()).hexdigest(),
           "rows_train": len(tr), "rows_test": len(te), "tau": tau,
           "mode": mode, "auc_train_run": round(float(roc_auc_score(tr.run, s_tr)), 3),
           "auc_test_run": round(float(roc_auc_score(te.run, s_te)), 3),
           "test_base": dict(run=round(100 * te.run.mean(), 2), crash=round(100 * te.crash.mean(), 2),
                             money=round(float(np.nanmean(te.money)), 3)),
           "test_selected": dict(n=len(sel), run=round(100 * sel.run.mean(), 2), crash=round(100 * sel.crash.mean(), 2),
                                 money=round(float(np.nanmean(sel.money)), 3)),
           "test_rank_corr": round(float(pd.Series(s_te).corr(te.money.reset_index(drop=True), method="spearman")), 4)}
    print(res)

    cands = sel[["date", "ticker", "score"]].rename(columns={"score": "key"}).sort_values(
        ["date", "key"], ascending=[True, False])
    sfx = "b" if mode == "money" else ""
    cands.to_parquet(ROOT / f"output/lab/h9{sfx}_test_candidates.parquet", index=False)
    T = simulate_h9(cands, P, regimes, cm, A14)
    T = attach_benchmark(T, pool_mask(), "IWM")
    T["net"] = T.ret_pct - COST
    T["excess"] = T.ret_pct - T.pool_ret
    lo, hi = boot_ci(T, "excess")
    yearly = T.assign(y=T.entry_day.dt.year).groupby("y").agg(
        n=("net", "size"), net=("net", "mean"), excess=("excess", "mean"),
        win=("net", lambda x: 100 * (x > 0).mean())).round(2)
    port = portfolio(T, FRAC, COST)
    port20 = portfolio(T, FRAC, 0.20)
    d0, d1 = T.entry_day.min(), T.exit_day.max()
    iwm = iwm_hold(d0, d1)
    cands_h6 = pd.read_parquet(OUT / "h6_candidates.parquet").sort_values(["date", "key"], ascending=[True, False])
    h6 = simulate_h6s(cands_h6, P, regimes, cm, True)
    h6 = h6[(h6.entry_day >= TEST_START) & (h6.entry_day <= d1)]
    h6p = portfolio(h6, FRAC, COST)
    crit = dict(P1_n=bool(len(T) >= 100),
                P2_net_excess=bool(T.net.mean() > 0 and T.excess.mean() > 0 and lo > 0),
                P3_vs_iwm_h6s=bool(port["total_pct"] > iwm["total_pct"] and port["total_pct"] > h6p["total_pct"]),
                P4_dd=bool(port["max_dd_pct"] > -35),
                P5_years=bool((yearly.excess > 0).sum() >= 2))
    res.update(trades=len(T), gross=round(T.ret_pct.mean(), 3), net=round(T.net.mean(), 3),
               net_median=round(T.net.median(), 3), win=round(100 * (T.net > 0).mean(), 1),
               excess=round(T.excess.mean(), 3), excess_ci95=[round(lo, 3), round(hi, 3)],
               exits=T.reason.value_counts().to_dict(), yearly=yearly.reset_index().to_dict("records"),
               portfolio_50bps=port, portfolio_20bps=port20, iwm_hold=iwm, h6s_15pct_same_period=h6p,
               window=[str(d0.date()), str(d1.date())], criteria=crit, PROMOTED=all(crit.values()))

    # Betimleyici: hangi göstergeler öğrenildi (eğitim alt örneği, permütasyon)
    smp = tr.sample(150_000, random_state=7)
    pi = permutation_importance(model, smp[cols].values,
                                smp.money.clip(-60, 150).values if mode == "money" else smp.run.values,
                                scoring="r2" if mode == "money" else "roc_auc", n_repeats=3, random_state=7)
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
    res["importance_top15"] = imp.head(15).round(4).to_dict()
    for k, v in res.items():
        print(f"{k}: {v}")
    (ROOT / f"output/lab/p_h9{sfx}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str),
                                               encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run")
