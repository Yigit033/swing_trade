"""Ek S — H4 (içeriden alım kümesi) ve H5 (bilanço sürprizi) iki dönemde, düzeltilmiş evrende.

Ön kayıt: lab/PREREGISTRATION.md Ek S (veri indirilmeden önce yazıldı). Kurallar lab/info_signals.py
(Ek A, donuk); bu dosya yalnız veri çekmeyi, iki dönemin koşusunu ve ön kayıtlı ölçütleri içerir.

  python lab/oos_s.py sec        # 2016–2020: SEC DERA Form 3/4/5 veri setleri (2015Q3–2020Q4)
  python lab/oos_s.py earnings   # 2016–2020: havuz hisselerinin bilanço kayıtları (limit=100)
  python lab/oos_s.py run        # etkin kökte H4 ve H5 → <kök>/s_result.json, s_trades_H{4,5}.parquet
  python lab/oos_s.py verdict    # iki kökün sonuçlarından ön kayıtlı hüküm → output/lab_oos/s_verdict.json

`run` iki kökte koşulur: varsayılan (2016–2020, output/lab_oos) ve
  LAB_DATA=output/lab_is_split LAB_EVAL_START=2021-01-04 LAB_EVAL_END=2026-11-20 python lab/oos_s.py run
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.oos_r import _fetch_all, _limited, _retry  # noqa: E402  (varsayılan kök: output/lab_oos)
from lab.common import EVAL_END, EVAL_START, LAB  # noqa: E402

OOS, IS = ROOT / "output/lab_oos", ROOT / "output/lab_is_split"
SEC_UA = "swing-trade-research-client admin@research-client.invalid"     # lab/fetch_sec.py ile aynı, kişisel veri yok
SEC_URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{}q{}_form345.zip"
EARN_LIMIT = 100
MIN_N = {"H4": 50, "H5": 150}
OK_HOURS = ((5.0, 9.5), (16.0, 20.0))          # ET; dışındaki saat damgası "varsayılan/güvenilmez" sayılır


def prereg_sha256() -> str:
    s = (ROOT / "lab/PREREGISTRATION.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    sec = s[s.index("# Ek S — "):]
    nxt = sec.find("\n# ", 1)
    if nxt != -1:
        sec = sec[:nxt]
    return hashlib.sha256((sec.rstrip("\n") + "\n").encode("utf-8")).hexdigest()


# ── Veri ─────────────────────────────────────────────────────────────────────

def cmd_sec():
    import urllib.request
    out = LAB / "sec"
    out.mkdir(parents=True, exist_ok=True)
    for y in range(2015, 2021):
        for q in range(1, 5):
            if (y, q) < (2015, 3):
                continue
            f = out / f"{y}q{q}.zip"
            if f.exists() and f.stat().st_size > 1000:
                continue
            try:
                req = urllib.request.Request(SEC_URL.format(y, q), headers={"User-Agent": SEC_UA})
                data = urllib.request.urlopen(req, timeout=180).read()
                f.write_bytes(data)
                print(y, q, len(data), flush=True)
            except Exception as e:
                print(y, q, "MISSING", e, flush=True)
            time.sleep(1)
    print("files:", len(list(out.glob("*.zip"))), flush=True)


def _earn_full_one(t, allow_empty):
    """lab/fetch_earnings_full.one ile aynı kayıt biçimi; limit=100 ve boş/kısıtlı yanıtı ilk geçişte yazmaz."""
    import yfinance as yf
    f = LAB / "earnings_full" / f"{t}.json"
    if f.exists():
        return "skip"
    rec = {"ticker": t, "rows": []}
    try:
        e = _retry(lambda: yf.Ticker(t).get_earnings_dates(limit=EARN_LIMIT))
    except Exception as ex:
        if _limited(ex):
            return "rate_limited"
        if not allow_empty:
            return "error"
        rec["error"], e = str(ex)[:200], None
    if e is not None and len(e):
        nan = lambda x: None if x != x else float(x)          # noqa: E731
        rec["rows"] = [dict(ts=str(k), est=nan(v.iloc[0]), rep=nan(v.iloc[1]), surprise=nan(v.iloc[2]))
                       for k, v in e[["EPS Estimate", "Reported EPS", "Surprise(%)"]].iterrows()]
    elif not allow_empty:
        return "empty"
    f.write_text(json.dumps(rec))
    time.sleep(0.3)
    return "ok"


def cmd_earnings():
    _fetch_all(_earn_full_one, (LAB / "pool_tickers.txt").read_text().split(), LAB / "earnings_full")


# ── Koşu ve ölçütler ─────────────────────────────────────────────────────────

def _iwm_ret(a, b) -> float:
    from lab.common import panels
    c = panels()["Close"]["IWM"].dropna()
    return float(c.loc[:b].iloc[-1] / c.loc[:a].iloc[-1] - 1) * 100


def cmd_run():
    import lab.info_signals as inf
    from lab.challengers import OUT
    from lab.common import panels, pool_mask
    from lab.evaluate import evaluate
    P = panels()
    pool = pool_mask().reindex(columns=P["Close"].columns, fill_value=False)
    OUT.mkdir(parents=True, exist_ok=True)
    cal = P["Close"].index
    start_prev = cal[cal < EVAL_START][-1]                       # dönem başından önceki son kapanış
    res = dict(prereg_ek_s_sha256=prereg_sha256(), kok=LAB.name,
               donem=[str(EVAL_START.date()), str((EVAL_END or cal[-1]).date())],
               havuz_medyan=int(pool.loc[EVAL_START:(EVAL_END or cal[-1])].sum(axis=1).median()))
    for name, fn in (("H4", inf.run_h4), ("H5", inf.run_h5)):
        T = fn(P, pool, name)
        T.to_parquet(OUT / f"trades_{name}.parquet", index=False)
        ev, TT, _ = evaluate(name, T, cost=0.50)
        TT.to_parquet(LAB / f"s_trades_{name}.parquet", index=False)
        port = ev["portfolio_10pct"]
        last = T.exit_day.max()
        iwm = round(_iwm_ret(start_prev, last), 1)
        S = {"S1_yeterlilik": bool(ev["n"] >= MIN_N[name] and ev["months"] >= 30),
             "S2_net": bool(ev["net_mean"] > 0),
             "S3_havuz_farki_ci": bool(ev["excess_mean"] > 0 and ev["excess_ci95"][0] > 0),
             "S4_portfoy_iwm_ustu": bool(port["total_pct"] > iwm),
             "S5_max_dusus": bool(port["max_dd_pct"] > -35)}
        r = dict(n=ev["n"], ay=ev["months"], brut=ev["gross_mean"], net_50bps=ev["net_mean"],
                 net_20bps=ev["net_20bps"], medyan=ev["net_median"], kazanma=ev["win_rate"], pf=ev["profit_factor"],
                 havuz=ev["pool_mean"], havuz_farki=ev["excess_mean"], havuz_farki_ci95=ev["excess_ci95"],
                 iwm_islem_basi=ev["iwm_mean"], en_iyi_yuzde5_haric=ev["net_trim_top5"], yillik=ev["yearly"],
                 portfoy=port, iwm_al_tut=iwm, pencere=[str(start_prev.date()), str(last.date())],
                 p_olcutleri=ev["criteria"], cikis=T.reason.value_counts().to_dict(), olcutler=S)
        if name == "H5":                                         # saat damgası kalitesi (havuz içi olaylar)
            E = pd.read_parquet(OUT / "events_H5.parquet")
            E = E.merge(T[["ticker", "signal_date"]], left_on=["ticker", "R"], right_on=["ticker", "signal_date"])
            ok = np.zeros(len(E), dtype=bool)
            for lo, hi in OK_HOURS:
                ok |= (E.hour >= lo) & (E.hour < hi)
            r["saat"] = dict(olay=len(E), aralik_disi_yuzde=round(100 * float((~ok).mean()), 1) if len(E) else None,
                             acilis_oncesi_yuzde=round(100 * float((E.hour < 9.5).mean()), 1) if len(E) else None,
                             en_sik=E.hour.round(2).value_counts().head(5).to_dict())
            r["zamanlama_belirsiz"] = bool(len(E) and (~ok).mean() >= 0.30)
        res[name] = r
        print(f"\n== {name} [{LAB.name}] n={r['n']} ({r['ay']} ay) brüt {r['brut']} net50 {r['net_50bps']} "
              f"net20 {r['net_20bps']} kazanma {r['kazanma']} PF {r['pf']} | havuz {r['havuz']} fark {r['havuz_farki']} "
              f"CI{r['havuz_farki_ci95']} | %5 hariç {r['en_iyi_yuzde5_haric']}")
        print(f"portföy %10 {port} | IWM al-tut {iwm}% {r['pencere']}")
        print(pd.DataFrame(r["yillik"]).to_string(index=False))
        print("çıkış", r["cikis"], "| ölçütler", S, "| saat", r.get("saat"))
    (LAB / "s_result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def cmd_verdict():
    from lab.evaluate import boot_ci
    out = dict(prereg_ek_s_sha256=prereg_sha256())
    A, B = (json.loads((r / "s_result.json").read_text(encoding="utf-8")) for r in (OOS, IS))
    for name in ("H4", "H5"):
        a, b = A[name], B[name]                                  # a: 2016–2020, b: düzeltilmiş 2021–26
        T = pd.concat([pd.read_parquet(r / f"s_trades_{name}.parquet") for r in (OOS, IS)])
        lo, hi = boot_ci(T, "excess")
        sa, sb = a["olcutler"], b["olcutler"]
        passed = all(sa.values()) and sb["S2_net"] and sb["S4_portfoy_iwm_ustu"] and b["havuz_farki"] > 0
        partial = a["havuz_farki"] > 0 and b["havuz_farki"] > 0 and lo > 0
        if not sa["S1_yeterlilik"]:
            v = "YETERSİZ VERİ"
        elif passed:
            v = "GEÇTİ"
        elif partial:
            v = "KISMEN"
        else:
            v = "KALDI"
        out[name] = dict(hukum=v, birlesik=dict(n=len(T), havuz_farki=round(float(T.excess.mean()), 3),
                                                ci95=[round(lo, 3), round(hi, 3)],
                                                net_50bps=round(float((T.ret_pct - 0.5).mean()), 3)),
                         d2016_2020=dict(olcutler=sa, n=a["n"], net=a["net_50bps"], havuz_farki=a["havuz_farki"],
                                         ci95=a["havuz_farki_ci95"], portfoy=a["portfoy"]["total_pct"],
                                         dusus=a["portfoy"]["max_dd_pct"], iwm=a["iwm_al_tut"],
                                         zamanlama_belirsiz=a.get("zamanlama_belirsiz")),
                         d2021_2026=dict(olcutler=sb, n=b["n"], net=b["net_50bps"], havuz_farki=b["havuz_farki"],
                                         ci95=b["havuz_farki_ci95"], portfoy=b["portfoy"]["total_pct"],
                                         dusus=b["portfoy"]["max_dd_pct"], iwm=b["iwm_al_tut"],
                                         zamanlama_belirsiz=b.get("zamanlama_belirsiz")))
        print(f"\n{name}: HÜKÜM {v}")
        for k in ("d2016_2020", "d2021_2026", "birlesik"):
            print("  ", k, out[name][k])
    (OOS / "s_verdict.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8", errors="replace")
    cmds = {k[4:]: v for k, v in list(globals().items()) if k.startswith("cmd_")}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(f"usage: python lab/oos_s.py [{'|'.join(cmds)}]")
    cmds[sys.argv[1]]()
