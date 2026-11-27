"""Ek X — TRD: büyüyen şirket + yükselen eğilim, haftalık karar (kırılım kalıbı yok), iki dönemde.

Ön kayıt: lab/PREREGISTRATION.md Ek X (sonuç görülmeden yazıldı). Veri Ek V ile aynı (output/lab_liq fiyat +
sec_facts/quarters.parquet); hiçbir şey indirilmez.

  python lab/trend_x.py run      # TRD + betimleyici kıyaslar × iki dönem → output/lab_liq/x_result.json + hüküm
"""
import os

os.environ.setdefault("LAB_DATA", "output/lab_liq")
os.environ.setdefault("LAB_EVAL_START", "2016-01-04")
os.environ.setdefault("LAB_EVAL_END", "")

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.common import BENCH, LAB  # noqa: E402  (LAB_DATA yukarıda sabitlendi)
from lab import pro_rules as pr  # noqa: E402
from lab import trd_rules as tr  # noqa: E402

assert LAB.name == "lab_liq", f"Ek X sürekli kökte koşar (output/lab_liq), etkin kök: {LAB}"

SEC = LAB / "sec_facts"
PERIODS = {"A": ("2016-01-04", "2020-12-31"), "B": ("2021-01-04", "2026-11-20")}
COST, FEE, CAP0 = 0.20, 1.5, 10_000.0                  # 20 bps gidiş-dönüş + 1.5$/emir
MIN_TRADES = 60


def prereg_sha256() -> str:
    s = (ROOT / "lab/PREREGISTRATION.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    sec = s[s.index("# Ek X — "):]
    nxt = sec.find("\n# ", 1)
    if nxt != -1:
        sec = sec[:nxt]
    return hashlib.sha256((sec.rstrip("\n") + "\n").encode("utf-8")).hexdigest()


def growth_panels(cal, columns):
    q = pd.read_parquet(SEC / "quarters.parquet")
    ok = pd.DataFrame(False, index=cal, columns=columns)
    rate = pd.DataFrame(np.nan, index=cal, columns=columns)
    for t, g in q.groupby("ticker"):
        if t in ok.columns:
            ok[t] = pr.growth_asof(g, cal)[1]
            rate[t] = tr.growth_rate_asof(g, cal)
    return ok, rate


def buy_hold(P, sym, start, end) -> float:
    c = P["Close"][sym].dropna()
    return float(c.loc[:end].iloc[-1] / c.loc[:pd.Timestamp(start) - pd.Timedelta(days=1)].iloc[-1] - 1) * 100


def summarize(port, eq, T, P, start, end, cost=COST):
    from lab.evaluate import boot_ci
    s = dict(n=len(T), portfoy=port, spy=round(buy_hold(P, "SPY", start, end), 1),
             iwm=round(buy_hold(P, "IWM", start, end), 1))
    if len(T) == 0:
        return s
    lo, hi = boot_ci(T, "excess") if len(T) > 10 else (np.nan, np.nan)
    yrs = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    y = T.assign(year=T.entry_day.dt.year, net=T.ret_pct - cost)
    yearly = y.groupby("year").agg(n=("net", "size"), net=("net", "mean"), excess=("excess", "mean")).round(2)
    last = eq.groupby(eq.index.year).last()
    yr_port = (last / last.shift(1).fillna(CAP0) - 1) * 100
    top5 = T.ret_pct.nlargest(5)
    s.update(
        brut=round(float(T.ret_pct.mean()), 3), net=round(float((T.ret_pct - cost).mean()), 3),
        medyan=round(float((T.ret_pct - cost).median()), 3), kazanma=round(100 * float((T.ret_pct > cost).mean()), 1),
        ort_kazanc=round(float(T.ret_pct[T.ret_pct > cost].mean()), 2),
        ort_kayip=round(float(T.ret_pct[T.ret_pct <= cost].mean()), 2),
        havuz=round(float(T.pool_ret.mean()), 3), havuz_farki=round(float(T.excess.mean()), 3),
        havuz_farki_ci95=[round(lo, 3), round(hi, 3)], ortalama_seans=round(float(T.bars.mean()), 1),
        cikis=T.reason.value_counts().to_dict(),
        en_iyi_5=[(r.ticker, str(r.entry_day.date()), round(r.ret_pct, 1)) for r in T.loc[top5.index].itertuples()],
        en_iyi_5_payi=round(100 * float(top5.sum() / T.ret_pct.sum()), 1) if T.ret_pct.sum() > 0 else None,
        yillik=yearly.reset_index().to_dict("records"),
        yillik_portfoy={int(k): round(float(v), 1) for k, v in yr_port.items()},
        yillik_emir=round(2 * port["taken"] / yrs, 1))
    return s


def cmd_run():
    import lab.h0_signals as h0
    from lab.common import actual_close, indicators, panels
    from lab.evaluate import boot_ci
    from lab.rotation import attach_pool
    P = panels()
    C = P["Close"]
    cal = C.index
    I, ca = indicators(), actual_close()
    pool = (ca >= 10) & (I["avgvol63"] > 300_000) & (I["dv20"] > 10e6) & (I["cap"] >= 1e9)
    pool = pool.reindex(columns=C.columns, fill_value=False).fillna(False)
    pool[[x for x in BENCH if x in pool.columns]] = False
    regimes = h0.regime_series()
    bull = pd.Series({d: v[0] == "BULL" for d, v in regimes.items()}).reindex(cal, fill_value=False).astype(bool)
    always = pd.Series(True, index=cal)
    trend, rs = tr.trend_mask(P), pr.rs_score(P)
    grow, rate = growth_panels(cal, C.columns)
    elig = pool & grow & trend
    # ad → (uygunluk, sıra anahtarı, piyasa şartı)
    variants = {"TRD": (elig, rs, bull), "buyumesiz": (pool & trend, rs, bull),
                "piyasa_sartsiz": (elig, rs, always), "buyume_sirali": (elig, rate, bull)}
    res = dict(prereg_ek_x_sha256=prereg_sha256(), maliyet=dict(bps=COST * 100, emir_ucreti=FEE, hesap=CAP0),
               donemler={}, TRD={}, betimleyici={})
    wl = pd.Series(tr.week_last(cal), index=cal)
    trd_all = []
    for per, (a, b) in PERIODS.items():
        dec = wl.loc[a:b] & bull.loc[a:b]
        res["donemler"][per] = dict(
            aralik=[a, b], havuz_medyan=int(pool.loc[a:b].sum(axis=1).median()),
            bull_gun_orani=round(100 * float(bull.loc[a:b].mean()), 1),
            karar_gunu=int(wl.loc[a:b].sum()), alim_yapilabilen_karar_gunu=int(dec.sum()),
            uygun_hisse_medyan=int(elig.loc[a:b].sum(axis=1).median()),
            buyume_sarti_saglayan_medyan=int((pool & grow).loc[a:b].sum(axis=1).median()))
        print(f"\n######## dönem {per} [{a} → {b}] {res['donemler'][per]}", flush=True)
        for name, (el, key, bl) in variants.items():
            port, eq, T = tr.simulate(P, el, key, bl, a, b, cost=COST, fee=FEE, cap0=CAP0)
            T = attach_pool(T, P, pool) if len(T) else T
            s = summarize(port, eq, T, P, a, b)
            _show(name, s)
            if name == "TRD":
                T.to_parquet(LAB / f"x_trades_TRD_{per}.parquet", index=False)
                eq.to_frame("equity").to_parquet(LAB / f"x_equity_TRD_{per}.parquet")
                trd_all.append(T)
                s["olcutler"] = {"X0_yeterlilik": bool(s["n"] >= MIN_TRADES),
                                 "X1_spy_ustu": bool(port["total_pct"] > s["spy"]),
                                 "X2_max_dusus": bool(port["max_dd_pct"] > -35)}
                s["duyarlilik"] = {
                    "bps10": tr.simulate(P, el, key, bl, a, b, cost=0.10, fee=FEE, cap0=CAP0)[0],
                    "bps50": tr.simulate(P, el, key, bl, a, b, cost=0.50, fee=FEE, cap0=CAP0)[0],
                    "hesap_25k": tr.simulate(P, el, key, bl, a, b, cost=COST, fee=FEE, cap0=25_000.0)[0]}
                print("   ölçütler", s["olcutler"], "| duyarlılık",
                      {k: (v["total_pct"], v["max_dd_pct"]) for k, v in s["duyarlilik"].items()}, flush=True)
                res["TRD"][per] = s
            else:
                res["betimleyici"].setdefault(name, {})[per] = s
    A, B = res["TRD"]["A"], res["TRD"]["B"]
    allT = pd.concat(trd_all)
    lo, hi = boot_ci(allT, "excess") if len(allT) > 10 else (np.nan, np.nan)
    x0 = A["olcutler"]["X0_yeterlilik"] and B["olcutler"]["X0_yeterlilik"]
    x1 = A["olcutler"]["X1_spy_ustu"] and B["olcutler"]["X1_spy_ustu"]
    x2 = A["olcutler"]["X2_max_dusus"] and B["olcutler"]["X2_max_dusus"]
    x3 = bool(x0 and A.get("havuz_farki", -1) > 0 and B.get("havuz_farki", -1) > 0 and lo > 0)
    verdict = "YETERSİZ VERİ" if not x0 else ("GEÇTİ" if (x1 and x2 and x3) else ("KISMEN" if x1 else "KALDI"))
    live = "TRD" if (x0 and x1 and x2) else "GRW"
    res["birlesik"] = dict(n=len(allT), havuz_farki=round(float(allT.excess.mean()), 3) if len(allT) else None,
                           ci95=[round(lo, 3), round(hi, 3)], X0=bool(x0), X1=bool(x1), X2=bool(x2), X3=x3,
                           hukum=verdict, canli_sanal_ana_kural=live)
    print(f"\nTRD: birlikte n={len(allT)} havuz farkı {res['birlesik']['havuz_farki']} CI[{lo:.3f}, {hi:.3f}] | "
          f"X0 {x0} X1 {x1} X2 {x2} X3 {x3} → HÜKÜM: {verdict} | canlı sanal ana kural: {live}")
    (LAB / "x_result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def _show(name, s):
    p = s["portfoy"]
    if s["n"] == 0:
        print(f"== {name}: işlem yok", flush=True)
        return
    print(f"== {name}: n={s['n']} brüt {s['brut']} net {s['net']} medyan {s['medyan']} kazanma %{s['kazanma']} "
          f"(ort. kazanç {s['ort_kazanc']} / kayıp {s['ort_kayip']}) seans {s['ortalama_seans']} "
          f"| havuz {s['havuz']} fark {s['havuz_farki']} CI{s['havuz_farki_ci95']}")
    print(f"   portföy {p} | SPY {s['spy']}% IWM {s['iwm']}% | yıllık emir {s['yillik_emir']} | çıkış {s['cikis']}")
    print(f"   yıllık portföy {s['yillik_portfoy']} | en iyi 5 {s['en_iyi_5']} pay %{s['en_iyi_5_payi']}", flush=True)


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8", errors="replace")
    cmds = {k[4:]: v for k, v in list(globals().items()) if k.startswith("cmd_")}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(f"usage: python lab/trend_x.py [{'|'.join(cmds)}]")
    cmds[sys.argv[1]]()
