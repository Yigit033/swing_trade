"""Ek V — PRO: büyüyen lider hisse + VCE kırılımı + dar stop + kazananı tutma + değişken boyut, iki dönemde.

Ön kayıt: lab/PREREGISTRATION.md Ek V (sonuç görülmeden ve bilanço verisi indirilmeden yazıldı). Fiyat verisi
Ek T'nin sürekli kökü (output/lab_liq); büyüme verisi SEC companyfacts.zip.

  curl -A "<genel UA>" -o output/lab_liq/sec_facts/company_tickers.json https://www.sec.gov/files/company_tickers.json
  curl -A "<genel UA>" -o output/lab_liq/sec_facts/companyfacts.zip \\
       https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip
  python lab/pro_v.py facts      # zip → output/lab_liq/sec_facts/quarters.parquet
  python lab/pro_v.py run        # PRO + betimleyici kıyaslar × iki dönem → output/lab_liq/v_result.json + hüküm
"""
import os

os.environ.setdefault("LAB_DATA", "output/lab_liq")
os.environ.setdefault("LAB_EVAL_START", "2016-01-04")
os.environ.setdefault("LAB_EVAL_END", "")

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import zipfile  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.common import BENCH, LAB  # noqa: E402  (LAB_DATA yukarıda sabitlendi)
from lab import pro_rules as pr  # noqa: E402

assert LAB.name == "lab_liq", f"Ek V sürekli kökte koşar (output/lab_liq), etkin kök: {LAB}"

SEC = LAB / "sec_facts"
PERIODS = {"A": ("2016-01-04", "2020-12-31"), "B": ("2021-01-04", "2026-11-20")}
COST, FEE, CAP0 = 0.20, 1.5, 10_000.0                  # 20 bps gidiş-dönüş + 1.5$/emir
MIN_TRADES = 60


def prereg_sha256() -> str:
    s = (ROOT / "lab/PREREGISTRATION.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    sec = s[s.index("# Ek V — "):]
    nxt = sec.find("\n# ", 1)
    if nxt != -1:
        sec = sec[:nxt]
    return hashlib.sha256((sec.rstrip("\n") + "\n").encode("utf-8")).hexdigest()


# ── Büyüme verisi: zip → çeyrek tablosu ──────────────────────────────────────

def _extract(chunk):
    out = []
    with zipfile.ZipFile(SEC / "companyfacts.zip") as z:
        names = set(z.namelist())
        for t, cik in chunk:
            name = f"CIK{int(cik):010d}.json"
            if name not in names:
                continue
            try:
                q = pr.quarters_table(json.loads(z.read(name)))
            except Exception as e:                      # bozuk kayıt: hisse "veri yok" sayılır
                print("HATA", t, e, flush=True)
                continue
            if len(q):
                out.append(q.assign(ticker=t))
            if len(out) % 500 == 0 and len(q):
                print(f"  {len(out)} hisse", flush=True)
    return pd.concat(out) if out else pd.DataFrame()


def cmd_facts():
    cik = {v["ticker"]: v["cik_str"] for v in json.loads((SEC / "company_tickers.json").read_text()).values()}
    ticks = (LAB / "tickers.txt").read_text().split()
    pairs = [(t, cik[t]) for t in ticks if t in cik]
    print(f"hisse {len(ticks)} | CIK eşleşen {len(pairs)}", flush=True)
    n = int(os.environ.get("LAB_WORKERS", "1"))
    if n <= 1:                                          # büyük JSON'lar: 13.7 GB RAM'de çok süreç çöküyor
        parts = [_extract(pairs)]
    else:
        with ProcessPoolExecutor(n) as ex:
            parts = list(ex.map(_extract, [pairs[i::n] for i in range(n)]))
    q = pd.concat([p for p in parts if len(p)]).sort_values(["ticker", "end"]).reset_index(drop=True)
    q.to_parquet(SEC / "quarters.parquet", index=False)
    print(f"çeyrek satırı {len(q)} | hisse {q.ticker.nunique()} | dönem sonu {q.end.min().date()} → {q.end.max().date()}")


def growth_panels(cal, columns):
    q = pd.read_parquet(SEC / "quarters.parquet")
    known = pd.DataFrame(False, index=cal, columns=columns)
    ok = pd.DataFrame(False, index=cal, columns=columns)
    for t, g in q.groupby("ticker"):
        if t in known.columns:
            k, o = pr.growth_asof(g, cal)
            known[t], ok[t] = k, o
    return known, ok


# ── Değerlendirme ────────────────────────────────────────────────────────────

def buy_hold(P, sym, start, end) -> float:
    c = P["Close"][sym].dropna()
    return float(c.loc[:end].iloc[-1] / c.loc[:pd.Timestamp(start) - pd.Timedelta(days=1)].iloc[-1] - 1) * 100


def summarize(T, P, start, end, cost=COST, cap0=CAP0, fixed_frac=None):
    from lab.evaluate import boot_ci
    port, eq, Tk = pr.portfolio_risk(T, P["Close"], cost=cost, fee=FEE, cap0=cap0, fixed_frac=fixed_frac)
    s = dict(n=len(T), portfoy=port, spy=round(buy_hold(P, "SPY", start, end), 1),
             iwm=round(buy_hold(P, "IWM", start, end), 1))
    if len(T) == 0:
        return s, eq
    lo, hi = boot_ci(T, "excess") if len(T) > 10 else (np.nan, np.nan)
    yrs = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    y = T.assign(year=T.entry_day.dt.year, net=T.ret_pct - cost)
    yearly = y.groupby("year").agg(n=("net", "size"), net=("net", "mean"), excess=("excess", "mean")).round(2)
    cal = P["Close"].index
    days = cal[(cal >= cal[cal < pd.Timestamp(start)][-1]) & (cal <= pd.Timestamp(end))]
    e = eq.reindex(days).ffill().fillna(cap0)
    last = e.groupby(e.index.year).last()
    yr_port = (last / last.shift(1).fillna(cap0) - 1) * 100
    top5 = T.ret_pct.nlargest(5)
    s.update(
        brut=round(float(T.ret_pct.mean()), 3), net=round(float((T.ret_pct - cost).mean()), 3),
        medyan=round(float((T.ret_pct - cost).median()), 3), kazanma=round(100 * float((T.ret_pct > cost).mean()), 1),
        ort_kazanc=round(float(T.ret_pct[T.ret_pct > cost].mean()), 2), ort_kayip=round(float(T.ret_pct[T.ret_pct <= cost].mean()), 2),
        havuz=round(float(T.pool_ret.mean()), 3), havuz_farki=round(float(T.excess.mean()), 3),
        havuz_farki_ci95=[round(lo, 3), round(hi, 3)], ortalama_seans=round(float(T.bars.mean()), 1),
        ort_stop_uzakligi=round(100 * float(T.stop_dist.mean()), 2), cikis=T.reason.value_counts().to_dict(),
        en_iyi_5=[(r.ticker, str(r.entry_day.date()), round(r.ret_pct, 1)) for r in T.loc[top5.index].itertuples()],
        en_iyi_5_payi=round(100 * float(top5.sum() / T.ret_pct.sum()), 1) if T.ret_pct.sum() > 0 else None,
        yillik=yearly.reset_index().to_dict("records"),
        yillik_portfoy={int(k): round(float(v), 1) for k, v in yr_port.items() if k >= pd.Timestamp(start).year},
        yillik_emir=round(2 * port["taken"] / yrs, 1))
    return s, e


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
    vce, lead, rs = pr.vce_mask(P), pr.leader_mask(P, pool), pr.rs_score(P)
    known, grow = growth_panels(cal, C.columns)
    bullm = pd.DataFrame(np.repeat(bull.values[:, None], C.shape[1], axis=1), index=cal, columns=C.columns)
    base = pool & lead & vce
    signals = {"PRO": base & grow & bullm, "buyumesiz": base & bullm, "lidersiz": pool & vce & grow & bullm,
               "piyasa_sartsiz": base & grow}
    res = dict(prereg_ek_v_sha256=prereg_sha256(), maliyet=dict(bps=COST * 100, emir_ucreti=FEE, hesap=CAP0),
               donemler={}, PRO={}, betimleyici={})
    pro_all = []
    for per, (a, b) in PERIODS.items():
        hits = (base & bullm).loc[a:b]
        res["donemler"][per] = dict(
            aralik=[a, b], havuz_medyan=int(pool.loc[a:b].sum(axis=1).median()),
            bull_gun_orani=round(100 * float(bull.loc[a:b].mean()), 1),
            lider_kirilim_sinyali=int(hits.values.sum()),
            buyume_verisi_olan=int((hits & known.loc[a:b]).values.sum()),
            buyume_sarti_saglayan=int((hits & grow.loc[a:b]).values.sum()))
        print(f"\n######## dönem {per} [{a} → {b}] {res['donemler'][per]}")
        for name, sig in signals.items():
            T = attach_pool(pr.pro_trades(P, sig, rs, a, b), P, pool)
            s, _ = summarize(T, P, a, b)
            _show(name, s)
            if name == "PRO":
                T.to_parquet(LAB / f"v_trades_PRO_{per}.parquet", index=False)
                pro_all.append(T)
                s["olcutler"] = {"V0_yeterlilik": bool(s["n"] >= MIN_TRADES),
                                 "V1_spy_ustu": bool(s["portfoy"]["total_pct"] > s["spy"]),
                                 "V2_max_dusus": bool(s["portfoy"]["max_dd_pct"] > -35)}
                s["duyarlilik"] = {
                    "bps10": pr.portfolio_risk(T, C, cost=0.10, fee=FEE, cap0=CAP0)[0],
                    "bps50": pr.portfolio_risk(T, C, cost=0.50, fee=FEE, cap0=CAP0)[0],
                    "hesap_25k": pr.portfolio_risk(T, C, cost=COST, fee=FEE, cap0=25_000.0)[0]}
                print("   ölçütler", s["olcutler"], "| duyarlılık",
                      {k: (v["total_pct"], v["max_dd_pct"]) for k, v in s["duyarlilik"].items()})
                res["PRO"][per] = s
                s10, _ = summarize(T, P, a, b, fixed_frac=0.10)
                _show("sabit_%10", s10)
                res["betimleyici"].setdefault("sabit_yuzde10", {})[per] = s10
                T8 = attach_pool(pr.pro_trades(P, sig, rs, a, b, exit_mode="time8"), P, pool)
                s8, _ = summarize(T8, P, a, b)
                _show("8_seans_cikis", s8)
                res["betimleyici"].setdefault("sekiz_seans_cikis", {})[per] = s8
            else:
                res["betimleyici"].setdefault(name, {})[per] = s
    A, B = res["PRO"]["A"], res["PRO"]["B"]
    allT = pd.concat(pro_all)
    lo, hi = boot_ci(allT, "excess") if len(allT) > 10 else (np.nan, np.nan)
    v0 = A["olcutler"]["V0_yeterlilik"] and B["olcutler"]["V0_yeterlilik"]
    v1 = A["olcutler"]["V1_spy_ustu"] and B["olcutler"]["V1_spy_ustu"]
    v2 = A["olcutler"]["V2_max_dusus"] and B["olcutler"]["V2_max_dusus"]
    v3 = bool(v0 and A["havuz_farki"] > 0 and B["havuz_farki"] > 0 and lo > 0)
    verdict = "YETERSİZ VERİ" if not v0 else ("GEÇTİ" if (v1 and v2 and v3) else ("KISMEN" if v1 else "KALDI"))
    res["birlesik"] = dict(n=len(allT), havuz_farki=round(float(allT.excess.mean()), 3) if len(allT) else None,
                           ci95=[round(lo, 3), round(hi, 3)], V0=bool(v0), V1=bool(v1), V2=bool(v2), V3=v3,
                           hukum=verdict)
    print(f"\nPRO: birlikte n={len(allT)} havuz farkı {res['birlesik']['havuz_farki']} CI[{lo:.3f}, {hi:.3f}] | "
          f"V0 {v0} V1 {v1} V2 {v2} V3 {v3} → HÜKÜM: {verdict}")
    (LAB / "v_result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def _show(name, s):
    p = s["portfoy"]
    if s["n"] == 0:
        print(f"== {name}: işlem yok")
        return
    print(f"== {name}: n={s['n']} brüt {s['brut']} net {s['net']} medyan {s['medyan']} kazanma %{s['kazanma']} "
          f"(ort. kazanç {s['ort_kazanc']} / kayıp {s['ort_kayip']}) seans {s['ortalama_seans']} stop %{s['ort_stop_uzakligi']} "
          f"| havuz {s['havuz']} fark {s['havuz_farki']} CI{s['havuz_farki_ci95']}")
    print(f"   portföy {p} | SPY {s['spy']}% IWM {s['iwm']}% | yıllık emir {s['yillik_emir']} | çıkış {s['cikis']}")
    print(f"   yıllık portföy {s['yillik_portfoy']} | en iyi 5 {s['en_iyi_5']} pay %{s['en_iyi_5_payi']}")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8", errors="replace")
    cmds = {k[4:]: v for k, v in list(globals().items()) if k.startswith("cmd_")}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(f"usage: python lab/pro_v.py [{'|'.join(cmds)}]")
    cmds[sys.argv[1]]()
