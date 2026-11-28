"""Ek Z — ERN: GRW'de bilanço açıklamasından hemen önce alım yapmama kuralı, iki dönem.

Ön kayıt: lab/PREREGISTRATION.md Ek Z (bilanço verisi indirilmeden ve sonuç görülmeden yazıldı).
Bilanço günü: SEC 8-K madde 2.02 dosyalamaları (acceptanceDateTime UTC → New York).

  python lab/earnings_z.py run     # GRW sinyalleri → eksik 8-K takvimleri SEC'ten → kurallı/kuralsız portföy
                                   # → output/lab_liq/z_result.json + hüküm
"""
import os

os.environ.setdefault("LAB_DATA", "output/lab_liq")
os.environ.setdefault("LAB_EVAL_START", "2016-01-04")
os.environ.setdefault("LAB_EVAL_END", "")

import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from urllib.error import HTTPError  # noqa: E402
from urllib.request import Request, urlopen  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab import pro_rules as pr  # noqa: E402
from lab.ern_rules import WINDOW, reaction_index, skip_mask  # noqa: E402
from lab import pro_v as pv  # noqa: E402
from lab.common import BENCH, LAB  # noqa: E402

assert LAB.name == "lab_liq", f"Ek Z sürekli kökte koşar (output/lab_liq), etkin kök: {LAB}"

EARN = LAB / "earn8k"
UA = "swing-trade-research-client admin@research-client.invalid"     # genel UA (kişisel bilgi yok)
COVER_DAYS, MIN_COVER = 120, 0.85
MAX_DD_SLACK = 3.0


def prereg_sha256() -> str:
    s = (ROOT / "lab/PREREGISTRATION.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    sec = s[s.index("# Ek Z — "):]
    nxt = sec.find("\n# ", 1)
    if nxt != -1:
        sec = sec[:nxt]
    return hashlib.sha256((sec.rstrip("\n") + "\n").encode("utf-8")).hexdigest()


# ── SEC 8-K 2.02 takvimi ─────────────────────────────────────────────────────

def _get(url: str) -> dict:
    for attempt in range(5):
        try:
            req = Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
            with urlopen(req, timeout=30) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
            time.sleep(0.12)                                   # SEC: en çok 10 istek/sn
            return json.loads(raw.decode("utf-8"))
        except HTTPError as e:
            if e.code == 404:
                return {}
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"SEC unreachable: {url}")


def _rows(block: dict) -> list:
    forms, items = block.get("form") or [], block.get("items") or []
    acc, fd = block.get("acceptanceDateTime") or [], block.get("filingDate") or []
    return [dict(filed=fd[i], accepted=acc[i]) for i in range(len(forms))
            if forms[i] == "8-K" and "2.02" in str(items[i] if i < len(items) else "")]


def fetch_calendar(ticker: str, cik: int) -> list:
    f = EARN / f"{ticker}.json"
    if f.exists():
        return json.loads(f.read_text())["rows"]
    d = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    fl = d.get("filings") or {}
    rows = _rows(fl.get("recent") or {})
    for extra in fl.get("files") or []:
        if str(extra.get("filingTo", "9999")) < "2015-06-01":
            continue
        rows += _rows(_get(f"https://data.sec.gov/submissions/{extra['name']}"))
    f.write_text(json.dumps(dict(ticker=ticker, cik=cik, rows=rows)))
    return rows


# ── Koşu ─────────────────────────────────────────────────────────────────────

def cmd_run():
    import lab.h0_signals as h0
    from lab.common import actual_close, indicators, panels
    from lab.rotation import attach_pool
    P = panels()
    C, O = P["Close"], P["Open"]
    cal = C.index
    I, ca = indicators(), actual_close()
    pool = (ca >= 10) & (I["avgvol63"] > 300_000) & (I["dv20"] > 10e6) & (I["cap"] >= 1e9)
    pool = pool.reindex(columns=C.columns, fill_value=False).fillna(False)
    pool[[x for x in BENCH if x in pool.columns]] = False
    regimes = h0.regime_series()
    bull = pd.Series({d: v[0] == "BULL" for d, v in regimes.items()}).reindex(cal, fill_value=False).astype(bool)
    vce, rs = pr.vce_mask(P), pr.rs_score(P)
    _, grow = pv.growth_panels(cal, C.columns)
    bullm = pd.DataFrame(np.repeat(bull.values[:, None], C.shape[1], axis=1), index=cal, columns=C.columns)
    sig = pool & vce & grow & bullm                                       # GRW (Ek V "lidersiz")

    tickers = sorted(C.columns[sig.loc["2015-12-01":].any(axis=0).values])
    cik = {v["ticker"]: int(v["cik_str"])
           for v in json.loads((pv.SEC / "company_tickers.json").read_text()).values()}
    EARN.mkdir(exist_ok=True)
    print(f"GRW sinyali olan hisse {len(tickers)} | CIK eşleşen {sum(t in cik for t in tickers)}", flush=True)
    cal_rows = {}
    for n, t in enumerate(tickers):
        if t in cik:
            cal_rows[t] = fetch_calendar(t, cik[t])
        if n % 50 == 0:
            print(f"  takvim {n}/{len(tickers)}", flush=True)
    reactions = {t: sorted({reaction_index(r["accepted"], cal) for r in rows if r.get("accepted")})
                 for t, rows in cal_rows.items()}
    filed = {t: pd.DatetimeIndex(sorted({pd.Timestamp(r["filed"]) for r in rows})) for t, rows in cal_rows.items()}
    skip = skip_mask(cal, C.columns, reactions)
    sig_rule = sig & ~skip

    res = dict(prereg_ek_z_sha256=prereg_sha256(), window_sessions=WINDOW,
               maliyet=dict(bps=pv.COST * 100, emir_ucreti=pv.FEE, hesap=pv.CAP0), donemler={})
    ref = json.loads((LAB / "v_result.json").read_text(encoding="utf-8"))["betimleyici"]["lidersiz"]
    pooled_skip, pooled_keep = [], []
    for per, (a, b) in pv.PERIODS.items():
        T0 = attach_pool(pr.pro_trades(P, sig, rs, a, b), P, pool)
        T1 = attach_pool(pr.pro_trades(P, sig_rule, rs, a, b), P, pool)
        p0, _, Tk0 = pr.portfolio_risk(T0, C, cost=pv.COST, fee=pv.FEE, cap0=pv.CAP0)
        p1, _, Tk1 = pr.portfolio_risk(T1, C, cost=pv.COST, fee=pv.FEE, cap0=pv.CAP0)
        assert (p0["total_pct"], p0["max_dd_pct"]) == (ref[per]["portfoy"]["total_pct"], ref[per]["portfoy"]["max_dd_pct"]), \
            f"GRW yeniden üretilemedi: {p0} vs {ref[per]['portfoy']}"
        # Z0: takvim kapsaması (GRW'nin aldığı işlemler)
        def covered(r):
            f = filed.get(r.ticker)
            return bool(f is not None and len(f) and (abs((f - r.entry_day).days) <= COVER_DAYS).any())
        cov = float(np.mean([covered(r) for r in Tk0.itertuples()])) if len(Tk0) else 0.0
        # GRW'nin aldığı işlemlerden kuralın eleyecekleri (sinyal günü maskede)
        sk = np.array([bool(skip.at[r.signal_date, r.ticker]) for r in Tk0.itertuples()])
        net0 = (Tk0.ret_pct - pv.COST).to_numpy()
        pooled_skip += list(net0[sk])
        pooled_keep += list(net0[~sk])
        # betimleyici: tepki seansını içinde taşıyan GRW işlemleri — tepki günü açılış boşluğu
        gaps, through = [], 0
        for r in Tk0.itertuples():
            ie, xe = cal.get_loc(r.entry_day), cal.get_loc(r.exit_day)
            stop = r.entry_price * (1 - r.stop_dist)
            for k in reactions.get(r.ticker, []):
                if ie < k <= xe:
                    pc, op = C.iat[k - 1, C.columns.get_loc(r.ticker)], O.iat[k, O.columns.get_loc(r.ticker)]
                    if np.isfinite(pc) and np.isfinite(op) and pc > 0:
                        gaps.append(100 * (op / pc - 1))
                        through += int(op <= stop)
                    break
        g = np.array(gaps)
        res["donemler"][per] = dict(
            aralik=[a, b], spy=round(pv.buy_hold(P, "SPY", a, b), 1), iwm=round(pv.buy_hold(P, "IWM", a, b), 1),
            GRW=p0, ERN=p1, sinyal_GRW=len(T0), sinyal_ERN=len(T1), takvim_kapsamasi=round(100 * cov, 1),
            elenen_alinmis=int(sk.sum()), alinmis=len(Tk0),
            elenen_ort_net=round(float(net0[sk].mean()), 3) if sk.any() else None,
            kalan_ort_net=round(float(net0[~sk].mean()), 3) if (~sk).any() else None,
            tepki_tasiyan=len(g), tepki_bosluk_medyan=round(float(np.median(g)), 2) if len(g) else None,
            tepki_bosluk_p10=round(float(np.percentile(g, 10)), 2) if len(g) else None,
            tepki_stop_alti_acilis=through,
            duyarlilik={"bps10": pr.portfolio_risk(T1, C, cost=0.10, fee=pv.FEE, cap0=pv.CAP0)[0],
                        "bps50": pr.portfolio_risk(T1, C, cost=0.50, fee=pv.FEE, cap0=pv.CAP0)[0]})
        d = res["donemler"][per]
        print(f"\n## {per} [{a} → {b}] SPY {d['spy']} IWM {d['iwm']} | kapsama %{d['takvim_kapsamasi']}")
        print(f"   GRW {p0['total_pct']:+.1f}% (düşüş {p0['max_dd_pct']}) alınan {p0['taken']}"
              f" | ERN {p1['total_pct']:+.1f}% (düşüş {p1['max_dd_pct']}) alınan {p1['taken']}")
        print(f"   elenen (GRW'de alınmış) {d['elenen_alinmis']}/{d['alinmis']} ort. net {d['elenen_ort_net']}"
              f" | kalan ort. net {d['kalan_ort_net']}")
        print(f"   tepki taşıyan {d['tepki_tasiyan']} | boşluk medyan {d['tepki_bosluk_medyan']} p10 {d['tepki_bosluk_p10']}"
              f" | stopun altında açılış {through}")
    A, B = res["donemler"]["A"], res["donemler"]["B"]
    z0 = A["takvim_kapsamasi"] >= 100 * MIN_COVER and B["takvim_kapsamasi"] >= 100 * MIN_COVER
    z1 = all(x["ERN"]["total_pct"] > x["GRW"]["total_pct"] for x in (A, B))
    z2 = all(x["ERN"]["max_dd_pct"] >= x["GRW"]["max_dd_pct"] - MAX_DD_SLACK for x in (A, B))
    ms, mk = (float(np.mean(pooled_skip)) if pooled_skip else np.nan), float(np.mean(pooled_keep))
    z3 = bool(pooled_skip) and ms < mk
    verdict = "YETERSİZ VERİ" if not z0 else ("GEÇTİ" if (z1 and z2 and z3) else "KALDI")
    res["birlesik"] = dict(elenen_n=len(pooled_skip), elenen_ort_net=round(ms, 3), kalan_ort_net=round(mk, 3),
                           Z0=bool(z0), Z1=bool(z1), Z2=bool(z2), Z3=bool(z3), hukum=verdict)
    print(f"\nERN: elenen n={len(pooled_skip)} ort. net {ms:.3f} vs kalan {mk:.3f} | "
          f"Z0 {z0} Z1 {z1} Z2 {z2} Z3 {z3} → HÜKÜM: {verdict}")
    (LAB / "z_result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    {"run": cmd_run}[sys.argv[1]]()
