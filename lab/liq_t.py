"""Ek T — MOM (lider rotasyonu) ve DIP (panik dipleri), büyük ve likit hisselerde, iki dönemde.

Ön kayıt: lab/PREREGISTRATION.md Ek T (sonuç görülmeden yazıldı). Tek bir sürekli veri kökü
(output/lab_liq, 2014-10 → 2026-11); iki dönem ayrı ayrı, her biri 10.000$'dan.

  LAB_DATA=output/lab_liq LAB_OHLCV_START=2014-10-01 LAB_OHLCV_END=2026-11-21 LAB_SPLITS_END=2026-11-26 \\
      LAB_TICKERS=output/lab_liq/tickers.txt python lab/fetch_ohlcv.py        # fiyat + bölünmeler
  python lab/liq_t.py shares     # hisse sayısı serileri (Yahoo, 2015'ten)
  python lab/liq_t.py run        # MOM ve DIP × iki dönem → output/lab_liq/t_result.json + hüküm
"""
import os

os.environ.setdefault("LAB_DATA", "output/lab_liq")
os.environ.setdefault("LAB_EVAL_START", "2016-01-04")
os.environ.setdefault("LAB_EVAL_END", "")            # boş = yok; dönemleri bu betik kendisi böler

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.oos_r import _fetch_all, _shares_one  # noqa: E402  (LAB_DATA yukarıda sabitlendi)
from lab.common import FACTS, LAB  # noqa: E402
from lab.rotation import attach_pool, mom_trades  # noqa: E402

assert LAB.name == "lab_liq", f"Ek T sürekli kökte koşar (output/lab_liq), etkin kök: {LAB}"

PERIODS = {"A": ("2016-01-04", "2020-12-31"), "B": ("2021-01-04", "2026-11-20")}
COST, FEE, CAP0, FRAC = 0.10, 1.5, 10_000.0, 0.10      # 10 bps gidiş-dönüş + 1.5$/emir, %10/pozisyon


def prereg_sha256() -> str:
    s = (ROOT / "lab/PREREGISTRATION.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    sec = s[s.index("# Ek T — "):]
    nxt = sec.find("\n# ", 1)
    if nxt != -1:
        sec = sec[:nxt]
    return hashlib.sha256((sec.rstrip("\n") + "\n").encode("utf-8")).hexdigest()


def cmd_shares():
    _fetch_all(_shares_one, (LAB / "tickers.txt").read_text().split(), FACTS)


# ── DIP: Ek §2 H2 kuralı, havuz L ────────────────────────────────────────────

def dip_trades(P, pool) -> pd.DataFrame:
    from lab.challengers import h2
    T = h2(P, pool).copy()
    T["strategy"] = "DIP"
    O = P["Open"]
    idx = {d: i for i, d in enumerate(O.index)}
    op = np.array([O.at[r.exit_day, r.ticker] for r in T.itertuples()])
    T["at_open"] = (T.exit_price.values == op) & (T.exit_day.values > T.entry_day.values)
    T["bars"] = [idx[r.exit_day] - idx[r.entry_day] + 1 for r in T.itertuples()]
    return T.sort_values(["entry_day", "key"], ascending=[True, False]).reset_index(drop=True)


# ── Değerlendirme ────────────────────────────────────────────────────────────

def buy_hold(P, sym, start, end) -> float:
    c = P["Close"][sym].dropna()
    return float(c.loc[:end].iloc[-1] / c.loc[:pd.Timestamp(start) - pd.Timedelta(days=1)].iloc[-1] - 1) * 100


def summarize(T, P, start, end, cost=COST, cap0=CAP0):
    from lab import swing_o as so
    from lab.evaluate import boot_ci
    port, eq = so.portfolio(T, FRAC, cost, fee=FEE, cap0=cap0, brake=False, return_curve=True)
    lo, hi = boot_ci(T, "excess") if len(T) > 10 else (np.nan, np.nan)
    yrs = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    y = T.assign(year=T.entry_day.dt.year, net=T.ret_pct - cost)
    yearly = y.groupby("year").agg(n=("net", "size"), net=("net", "mean"), excess=("excess", "mean")).round(2)
    cal = P["Close"].index
    days = cal[(cal >= cal[cal < pd.Timestamp(start)][-1]) & (cal <= pd.Timestamp(end))]
    e = eq.reindex(days).ffill().fillna(cap0)                          # ilk işleme kadar nakit
    yr_port = (e.groupby(e.index.year).last() / e.groupby(e.index.year).last().shift(1).fillna(cap0) - 1) * 100
    return dict(
        n=len(T), ay=int(T.entry_day.dt.to_period("M").nunique()),
        brut=round(float(T.ret_pct.mean()), 3), net=round(float((T.ret_pct - cost).mean()), 3),
        medyan=round(float((T.ret_pct - cost).median()), 3), kazanma=round(100 * float((T.ret_pct > cost).mean()), 1),
        havuz=round(float(T.pool_ret.mean()), 3), havuz_farki=round(float(T.excess.mean()), 3),
        havuz_farki_ci95=[round(lo, 3), round(hi, 3)], ortalama_seans=round(float(T.bars.mean()), 1),
        cikis=T.reason.value_counts().to_dict(), yillik=yearly.reset_index().to_dict("records"),
        portfoy=port, yillik_portfoy={int(k): round(float(v), 1) for k, v in yr_port.items() if k >= pd.Timestamp(start).year},
        yillik_emir=round(2 * port["taken"] / yrs, 1),
        spy=round(buy_hold(P, "SPY", start, end), 1), iwm=round(buy_hold(P, "IWM", start, end), 1)), e


def cmd_run():
    import lab.h0_signals as h0
    from lab import swing_o as so
    from lab.common import panels, pool_mask_large
    from lab.evaluate import boot_ci
    P = panels()
    pool = pool_mask_large().reindex(columns=P["Close"].columns, fill_value=False)
    regimes = h0.regime_series()                                        # canlı rejim: SPY 1 yıl + VIX
    dip_all = dip_trades(P, pool)
    res = dict(prereg_ek_t_sha256=prereg_sha256(), maliyet=dict(bps=COST * 100, emir_ucreti=FEE, hesap=CAP0), fikirler={})
    trades, curves = {"MOM": [], "DIP": []}, {}
    for per, (a, b) in PERIODS.items():
        ps = pool.loc[a:b].sum(axis=1)
        rg = pd.Series({d: v[0] for d, v in regimes.items()}).loc[a:b].value_counts().to_dict()
        res.setdefault("donemler", {})[per] = dict(aralik=[a, b], havuz_medyan=int(ps.median()), rejim_gunleri=rg)
        sets = {"MOM": mom_trades(P, pool, regimes, a, b),
                "DIP": dip_all[(dip_all.signal_date >= a) & (dip_all.signal_date <= b)].reset_index(drop=True)}
        for name, T in sets.items():
            T = attach_pool(T, P, pool)
            T.to_parquet(LAB / f"t_trades_{name}_{per}.parquet", index=False)
            s, eq = summarize(T, P, a, b)
            s["olcutler"] = {"T0_yeterlilik": bool(s["n"] >= 100),
                             "T1_spy_ustu": bool(s["portfoy"]["total_pct"] > s["spy"]),
                             "T2_max_dusus": bool(s["portfoy"]["max_dd_pct"] > -35)}
            s["duyarlilik"] = {
                "bps20": so.portfolio(T, FRAC, 0.20, fee=FEE, cap0=CAP0, brake=False),
                "hesap_25k": so.portfolio(T, FRAC, COST, fee=FEE, cap0=25_000.0, brake=False),
                "hesap_50k": so.portfolio(T, FRAC, COST, fee=FEE, cap0=50_000.0, brake=False)}
            if name == "MOM":
                Tn = attach_pool(mom_trades(P, pool, regimes, a, b, use_regime=False), P, pool)
                s["duyarlilik"]["rejim_kuralsiz"] = dict(n=len(Tn), havuz_farki=round(float(Tn.excess.mean()), 3),
                                                         portfoy=so.portfolio(Tn, FRAC, COST, fee=FEE, cap0=CAP0, brake=False))
            res["fikirler"].setdefault(name, {})[per] = s
            trades[name].append(T)
            curves[(name, per)] = so.portfolio(T, FRAC, COST, fee=FEE, cap0=CAP0 / 2, brake=False, return_curve=True)[1]
            p = s["portfoy"]
            print(f"\n== {name} dönem {per} [{a} → {b}] n={s['n']} ({s['ay']} ay) brüt {s['brut']} net {s['net']} "
                  f"kazanma {s['kazanma']} ort. seans {s['ortalama_seans']} | havuz {s['havuz']} fark {s['havuz_farki']} "
                  f"CI{s['havuz_farki_ci95']}")
            print(f"portföy {p} | SPY {s['spy']}% IWM {s['iwm']}% | yıllık emir {s['yillik_emir']}")
            print("yıllık portföy", s["yillik_portfoy"], "| çıkış", s["cikis"])
            print(pd.DataFrame(s["yillik"]).to_string(index=False))
            print("ölçütler", s["olcutler"], "| duyarlılık", {k: (v.get("total_pct") if "total_pct" in v else v["portfoy"]["total_pct"])
                                                             for k, v in s["duyarlilik"].items()})
    for name in ("MOM", "DIP"):                                         # hüküm: iki dönem birlikte
        A, B = res["fikirler"][name]["A"], res["fikirler"][name]["B"]
        allT = pd.concat(trades[name])
        lo, hi = boot_ci(allT, "excess")
        t0 = A["olcutler"]["T0_yeterlilik"] and B["olcutler"]["T0_yeterlilik"]
        t1 = A["olcutler"]["T1_spy_ustu"] and B["olcutler"]["T1_spy_ustu"]
        t2 = A["olcutler"]["T2_max_dusus"] and B["olcutler"]["T2_max_dusus"]
        t3 = bool(A["havuz_farki"] > 0 and B["havuz_farki"] > 0 and lo > 0)
        v = "YETERSİZ VERİ" if not t0 else ("GEÇTİ" if (t1 and t2 and t3) else ("KISMEN" if t1 else "KALDI"))
        res["fikirler"][name]["birlesik"] = dict(n=len(allT), havuz_farki=round(float(allT.excess.mean()), 3),
                                                 ci95=[round(lo, 3), round(hi, 3)], T0=bool(t0), T1=bool(t1),
                                                 T2=bool(t2), T3=t3, hukum=v)
        print(f"\n{name}: birlikte n={len(allT)} havuz farkı {allT.excess.mean():.3f} CI[{lo:.3f}, {hi:.3f}] | "
              f"T0 {t0} T1 {t1} T2 {t2} T3 {t3} → HÜKÜM: {v}")
    combo = {}                                                          # betimleyici: yarı yarıya iki alt hesap
    for per, (a, b) in PERIODS.items():
        cal = P["Close"].index
        days = cal[(cal >= cal[cal < pd.Timestamp(a)][-1]) & (cal <= pd.Timestamp(b))]
        e = sum(curves[(n, per)].reindex(days).ffill().fillna(CAP0 / 2) for n in ("MOM", "DIP"))
        combo[per] = dict(total_pct=round(float(e.iloc[-1] / CAP0 - 1) * 100, 1),
                          max_dd_pct=round(float((e / e.cummax() - 1).min() * 100), 1),
                          spy=res["fikirler"]["MOM"][per]["spy"])
    res["yari_yariya_iki_alt_hesap"] = combo
    print("\nyarı yarıya iki alt hesap (betimleyici):", combo)
    (LAB / "t_result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8", errors="replace")
    cmds = {k[4:]: v for k, v in list(globals().items()) if k.startswith("cmd_")}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(f"usage: python lab/liq_t.py [{'|'.join(cmds)}]")
    cmds[sys.argv[1]]()
