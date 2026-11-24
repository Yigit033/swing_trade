"""Ek R — H6S'nin dokunulmamış yıllarda sınavı (2016–2020, kurallar donuk).

Ön kayıt: lab/PREREGISTRATION.md Ek R (veri indirilmeden önce yazıldı). Aynı lab betikleri ayrı bir veri
kökünde (output/lab_oos) koşar; bu dosya yalnız sırayı, veri çekmeyi ve ön kayıtlı ölçütleri içerir —
sinyal, çıkış ve portföy mantığı lab/h0_signals, lab/h6_signals, lab/swing_o ve lab/evaluate'tedir.

Sıra:
  python lab/oos_r.py ohlcv      # fiyat + bölünme olayları (lab/fetch_ohlcv)
  python lab/oos_r.py tickers    # hisse sayısı gereken hisseler → fact_tickers.txt
  python lab/oos_r.py shares     # Yahoo hisse sayısı serileri (2015'ten)
  python lab/oos_r.py h0         # canlı motor, emüle evren (lab/h0_signals)
  python lab/oos_r.py vce        # canlı VCE isabetleri → vce_hits.parquet, earn_tickers.txt
  python lab/oos_r.py earnings   # Yahoo bilanço takvimi (limit=100)
  python lab/oos_r.py run        # bilanço kapısı → H6S → ölçütler → hüküm → r_oos.json
"""
import os

os.environ.setdefault("LAB_DATA", "output/lab_oos")
os.environ.setdefault("LAB_EVAL_START", "2016-01-04")
os.environ.setdefault("LAB_EVAL_END", "2020-12-31")
os.environ.setdefault("LAB_OHLCV_START", "2014-10-01")
os.environ.setdefault("LAB_OHLCV_END", "2021-03-06")
os.environ.setdefault("LAB_SPLITS_END", "2026-11-26")

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.common import BASE, BENCH, EARN, EVAL_END, EVAL_START, FACTS, LAB  # noqa: E402
from lab.common import earnings_calendar_known as calendar_known  # noqa: E402  (ön kayıt: ≤155 gün)

assert LAB.resolve() != BASE.resolve(), "Ek R 2021–26 kökünde koşmaz (LAB_DATA=output/lab_oos)"

SHARES_START = "2015-01-01"
EARN_LIMIT = 100
FRAC_LIVE, FEE, CAP0 = 0.15, 1.5, 10_000.0
IWM_FROM = pd.Timestamp("2015-12-31")
STRESS = (("2015-12-31", "2016-02-11"), ("2018-08-31", "2018-12-24"), ("2020-02-19", "2020-03-18"))


# ── Veri çekme ───────────────────────────────────────────────────────────────

def _limited(e) -> bool:
    m = str(e)
    return "Rate" in m or "Too Many" in m or "NoneType" in m


def _retry(fn, tries=4):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if attempt == tries - 1 or not _limited(e):
                raise
            time.sleep(30 * (attempt + 1))


def _shares_one(t, allow_empty):
    import yfinance as yf
    f = FACTS / f"{t}.json"
    if f.exists():
        return "skip"
    rec = {"ticker": t}
    base = BASE / "facts" / f"{t}.json"            # ülke/sektör/güncel sayı: güncel anlık görüntü (in-sample ile aynı)
    if base.exists():
        info = json.loads(base.read_text()).get("info")
        if info:
            rec["info"] = info
    try:
        s = _retry(lambda: yf.Ticker(t).get_shares_full(start=SHARES_START))
    except Exception as e:
        if _limited(e):
            return "rate_limited"                    # yazma; sonraki geçiş yeniden dener
        if not allow_empty:
            return "error"
        rec["error"], s = str(e)[:200], None
    if s is not None and len(s):
        s = s[~s.index.duplicated(keep="last")].sort_index()
        rec["shares"] = {str(k.date()): float(v) for k, v in s.items()}
    elif not allow_empty:
        return "empty"                               # boş = çoğu kez kısıtlama; ilk geçişte yazma
    f.write_text(json.dumps(rec))
    time.sleep(0.3)
    return "ok"


def _earn_one(t, allow_empty):
    import yfinance as yf
    f = EARN / f"{t}.json"
    if f.exists():
        return "skip"
    rec = {"ticker": t}
    try:
        e = _retry(lambda: yf.Ticker(t).get_earnings_dates(limit=EARN_LIMIT))
    except Exception as ex:
        if _limited(ex):
            return "rate_limited"
        if not allow_empty:
            return "error"
        rec["error"], e = str(ex)[:200], None
    dates = [] if e is None else sorted({str(k.date()) for k in e.index})
    if not dates and not allow_empty:
        return "empty"
    rec["dates"] = dates
    f.write_text(json.dumps(rec))
    time.sleep(0.3)
    return "ok"


def _fetch_all(fn, tickers, out_dir, passes=3):
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in range(passes):
        final = p == passes - 1
        counts = {}
        with ThreadPoolExecutor(3) as ex:
            for i, r in enumerate(ex.map(lambda t: fn(t, final), tickers), 1):
                counts[r] = counts.get(r, 0) + 1
                if i % 250 == 0:
                    print(f"  pass {p + 1}: {i}/{len(tickers)} {counts}", flush=True)
        print(f"pass {p + 1} done: {counts}", flush=True)
        if not any(counts.get(k) for k in ("rate_limited", "error", "empty")):
            break
        time.sleep(60)
    print("files:", sum(1 for _ in out_dir.glob("*.json")), "of", len(tickers), flush=True)


def cmd_ohlcv():
    import lab.fetch_ohlcv as f
    f.main()


def cmd_tickers():
    """Değerlendirme döneminde bir gün bile evren tabanına (gerçek fiyat 6–250 $, ort. hacim > 300K) girebilen hisseler."""
    from lab.common import panels, split_factor
    P, F = panels(), split_factor()
    ca = P["Close"] * F
    av = P["Volume"].rolling(63, min_periods=40).mean() / F
    win = (ca.index >= EVAL_START - pd.Timedelta(days=10)) & (ca.index <= EVAL_END)
    m = ((ca > 6) & (ca <= 250) & (av > 300_000)).loc[win].any()
    ticks = sorted(t for t in m.index[m.values] if t not in BENCH)
    (LAB / "fact_tickers.txt").write_text("\n".join(ticks))
    print(len(ticks), "tickers →", LAB / "fact_tickers.txt")


def cmd_shares():
    _fetch_all(_shares_one, (LAB / "fact_tickers.txt").read_text().split(), FACTS)


def cmd_h0():
    import lab.h0_signals as h0
    # Varsayılan (çekirdek − 2) işçi × çok iş parçacıklı BLAS bu makinede belleği tüketti (BrokenProcessPool);
    # işçi başına tek BLAS iş parçacığı sonucu değiştirmez (motor pandas hesapları), yalnız belleği sınırlar.
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    sys.argv = ["h0_signals", "--universe", "emulated", "--workers", os.environ.get("LAB_WORKERS", "8")]
    h0.main()


def cmd_vce():
    import lab.h0_signals as h0
    import lab.h6_signals as h6
    h0._init()
    o = h6.trigger_rows()
    hits = h6.vce_hits(o)
    H = pd.DataFrame(hits, columns=["date", "ticker"])
    H.to_parquet(LAB / "vce_hits.parquet", index=False)
    ticks = sorted(H.ticker.unique())
    (LAB / "earn_tickers.txt").write_text("\n".join(ticks))
    print(f"trigger rows {len(o)} → VCE hits {len(H)} on {H.date.nunique()} days, {len(ticks)} tickers")


def cmd_earnings():
    _fetch_all(_earn_one, (LAB / "earn_tickers.txt").read_text().split(), EARN)


# ── Ön kayıtlı kurallar ──────────────────────────────────────────────────────

def prereg_sha256() -> str:
    """Ön kayıt Ek R bölümünün SHA256'sı: "# Ek R — " başlığından bir sonraki üst başlığa ("# ") kadar, LF satır
    sonlu, sonda tek satır sonu. Veri inmeden önceki değer: 31fe5e5e…632f4 (sonradan eklenen bölümler etkilemez)."""
    s = (ROOT / "lab/PREREGISTRATION.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    sec = s[s.index("# Ek R — "):]
    nxt = sec.find("\n# ", 1)
    if nxt != -1:
        sec = sec[:nxt]
    return hashlib.sha256((sec.rstrip("\n") + "\n").encode("utf-8")).hexdigest()


def _curve(T, frac, cost, cap0=CAP0, brake=True):
    """lab/swing_o.portfolio ile aynı hesap + günlük özsermaye eğrisi (yıllık tablo ve düşüş pencereleri için)."""
    from lab import swing_o as so
    return so.portfolio(T, frac, cost, fee=FEE, cap0=cap0, brake=brake, return_curve=True)


def _iwm_ret(a, b) -> float:
    from lab.common import panels
    c = panels()["Close"]["IWM"].dropna()
    return float(c.loc[:b].iloc[-1] / c.loc[:a].iloc[-1] - 1) * 100


def _yearly(eq, cap0=CAP0):
    """Takvim yılı getirileri: portföy (ilk işleme kadar nakitte) ve IWM, yıl sonu kapanışından yıl sonu kapanışına."""
    from lab.common import panels
    iwm = panels()["Close"]["IWM"].dropna()
    days = iwm.index[(iwm.index >= IWM_FROM) & (iwm.index <= eq.index[-1])]
    e = eq.reindex(days).ffill().fillna(cap0)
    rows = []
    for y in sorted(set(days.year) - {IWM_FROM.year}):
        a, b = days[days.year < y][-1], days[days.year == y][-1]
        rows.append(dict(yil=int(y), portfoy=round(float(e[b] / e[a] - 1) * 100, 1),
                         iwm=round(float(iwm[b] / iwm[a] - 1) * 100, 1)))
    return rows, e


def cmd_run():
    import lab.h0_signals as h0
    import lab.h6_signals as h6
    from lab import swing_o as so
    from lab.common import earnings_dates, load_facts, panels, split_events, split_factor
    from lab.evaluate import evaluate
    from lab.h6r import regime_series

    h0._init()
    S, P = h0.live_settings(), panels()
    H = pd.read_parquet(LAB / "vce_hits.parquet")
    hits = list(zip(H.date, H.ticker))

    # Bilanço kapısı: ön kayıtlı (veri kalitesi kurallı) ve betimleyici (kuralsız)
    cands, rej = h6.earnings_gate(hits, known=calendar_known)
    cands_raw, rej_raw = h6.earnings_gate(hits)
    no_cal = sum(1 for d, t in hits if (lambda e: e is None or len(e) == 0)(earnings_dates(t)))
    gap = sum(1 for d, t in hits
              if (lambda e: e is not None and len(e) > 0 and not calendar_known(e, d))(earnings_dates(t)))
    cands.to_parquet(LAB / "h6_candidates.parquet", index=False)

    regimes = regime_series("IWM")
    cm, bb = int(S.backtest_loop.caution_max_concurrent), bool(S.backtest_loop.bear_block_new_entries)
    T = so.simulate(cands, P, regimes, cm, bb)                              # H6S (donuk kurallar)
    T_free = so.simulate(cands, P, regimes, 10 ** 6, False)                 # R5: frensiz
    T_run = so.simulate(cands, P, regimes, cm, bb, runner=True)             # betimleyici: H6SR çıkışı
    T_raw = so.simulate(cands_raw, P, regimes, cm, bb)                      # betimleyici: bilanço kuralı yok
    T.to_parquet(LAB / "trades_H6S.parquet", index=False)

    ev, TT, _ = evaluate("H6S_OOS", T, cost=0.50)
    live, eq = _curve(T, FRAC_LIVE, 0.50)
    free, _ = _curve(T_free, FRAC_LIVE, 0.50)
    last = T.exit_day.max()
    iwm_total = round(_iwm_ret(IWM_FROM, last), 1)
    yearly, e = _yearly(eq)

    R = {
        "R0_yeterlilik": bool(ev["n"] >= 150 and ev["months"] >= 30),
        "R1_net_ve_havuz_farki": bool(ev["net_mean"] > 0 and ev["excess_mean"] > 0),
        "R2_havuz_farki_ci_alt_sinir": bool(ev["excess_ci95"][0] > 0),
        "R3_portfoy_iwm_ustu": bool(live["total_pct"] > iwm_total),
        "R4_max_dusus": bool(live["max_dd_pct"] > -35),
        "R5_fren_degeri": bool(live["total_pct"] >= free["total_pct"] and live["max_dd_pct"] > free["max_dd_pct"]),
    }
    if not R["R0_yeterlilik"]:
        verdict = "YETERSİZ VERİ"
    elif R["R1_net_ve_havuz_farki"] and R["R2_havuz_farki_ci_alt_sinir"] and R["R3_portfoy_iwm_ustu"] \
            and R["R4_max_dusus"]:
        verdict = "GEÇTİ"
    elif R["R1_net_ve_havuz_farki"] and live["total_pct"] > 0:
        verdict = "KISMEN"
    else:
        verdict = "KALDI"

    # Betimleyici okumalar
    reg_days = pd.Series({d: v[0] for d, v in regimes.items()})
    reg_days = reg_days[(reg_days.index >= EVAL_START) & (reg_days.index <= EVAL_END)]
    stress = []
    iwm = P["Close"]["IWM"].dropna()
    for a, b in STRESS:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        stress.append(dict(pencere=f"{a.date()}→{b.date()}",
                           portfoy=round(float(e.loc[:b].iloc[-1] / e.loc[:a].iloc[-1] - 1) * 100, 1),
                           iwm=round(float(iwm.loc[:b].iloc[-1] / iwm.loc[:a].iloc[-1] - 1) * 100, 1)))
    SF, ev_sp = split_factor(), split_events()
    sig_t = sorted(T.ticker.unique())
    kinds = {"seri": 0, "guncel_sayi_yedegi": 0}
    for t in sig_t:
        kinds["seri" if (load_facts(t) or {}).get("shares") else "guncel_sayi_yedegi"] += 1
    fb = {t for t in sig_t if not (load_facts(t) or {}).get("shares")}
    first_obs = {t: pd.Timestamp(min((load_facts(t) or {}).get("shares") or {"2100-01-01": 0})) for t in sig_t}
    backfilled = int(sum(r.signal_date < first_obs[r.ticker] for r in T.itertuples()))   # ilk gözlem sinyalden sonra
    runner = so.portfolio(T_run, FRAC_LIVE, 0.50, fee=FEE, cap0=CAP0, brake=True)
    ev_free, _, _ = evaluate("H6S_frensiz", T_free, cost=0.50)
    ev_raw, _, _ = evaluate("H6S_bilanco_kuralsiz", T_raw, cost=0.50)
    res = dict(
        prereg_ek_r_sha256=prereg_sha256(),
        donem=[str(EVAL_START.date()), str(EVAL_END.date())],
        huni=dict(vce_isabet=len(hits), takvim_yok=no_cal, takvim_boslugu=gap, bilanco_reddi_toplam=rej,
                  aday=len(cands), aday_kuralsiz=len(cands_raw), bilanco_reddi_kuralsiz=rej_raw),
        h6s=dict(n=ev["n"], ay=ev["months"], brut=ev["gross_mean"], net_50bps=ev["net_mean"],
                 net_20bps=ev["net_20bps"], medyan=ev["net_median"], kazanma=ev["win_rate"], pf=ev["profit_factor"],
                 havuz=ev["pool_mean"], havuz_farki=ev["excess_mean"], havuz_farki_ci95=ev["excess_ci95"],
                 iwm_islem_basi=ev["iwm_mean"], en_iyi_yuzde5_haric=ev["net_trim_top5"], yillik=ev["yearly"],
                 p_olcutleri=ev["criteria"], cikis=T.reason.value_counts().to_dict(),
                 rejim=T.groupby("regime").ret_pct.agg(["size", "mean"]).round(2).to_dict()),
        portfoy_canli_ayar=live, iwm_al_tut=iwm_total, pencere=[str(IWM_FROM.date()), str(last.date())],
        frensiz=dict(portfoy=free, n=ev_free["n"], net_50bps=ev_free["net_mean"], havuz_farki=ev_free["excess_mean"]),
        olcutler=R, hukum=verdict,
        betimleyici=dict(
            yuzde10=so.portfolio(T, 0.10, 0.50, fee=FEE, cap0=CAP0, brake=True),
            bps20_yuzde15=so.portfolio(T, FRAC_LIVE, 0.20, fee=FEE, cap0=CAP0, brake=True),
            hesap_25k=so.portfolio(T, FRAC_LIVE, 0.50, fee=FEE, cap0=25_000.0, brake=True),
            yillik_portfoy_iwm=yearly, dusus_pencereleri=stress,
            rejim_gun_dagilimi=reg_days.value_counts().to_dict(),
            h6sr_cikisi=dict(portfoy=runner, net_50bps=round(float((T_run.ret_pct - 0.5).mean()), 3), n=len(T_run)),
            bilanco_kuralsiz=dict(n=ev_raw["n"], net_50bps=ev_raw["net_mean"], havuz_farki=ev_raw["excess_mean"],
                                  portfoy=so.portfolio(T_raw, FRAC_LIVE, 0.50, fee=FEE, cap0=CAP0, brake=True)),
            veri=dict(islem_hissesi=len(sig_t), hisse_sayisi_kaynagi=kinds,
                      yedekli_hisse_islem_sayisi=int(T.ticker.isin(fb).sum()),
                      hisse_sayisi_geriye_tasinan_islem=backfilled,
                      bolunme_duzeltmeli_islem=int(sum(float(SF.at[r.signal_date, r.ticker]) != 1.0
                                                       for r in T.itertuples())),
                      bolunme_olayi_olan_hisse=len(ev_sp))),
    )
    (LAB / "r_oos.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    h = res["h6s"]
    print(f"huni {res['huni']}")
    print(f"H6S n={h['n']} ({h['ay']} ay) brüt {h['brut']} net50 {h['net_50bps']} net20 {h['net_20bps']} "
          f"kazanma {h['kazanma']} PF {h['pf']} | havuz {h['havuz']} fark {h['havuz_farki']} "
          f"CI{h['havuz_farki_ci95']} | IWM/işlem {h['iwm_islem_basi']} | %5 hariç {h['en_iyi_yuzde5_haric']}")
    print(pd.DataFrame(h["yillik"]).to_string(index=False))
    print(f"canlı ayar portföy {live} | IWM al-tut {iwm_total}% ({res['pencere']})")
    print(f"frensiz {res['frensiz']}")
    print(pd.DataFrame(yearly).to_string(index=False))
    print("düşüş pencereleri", stress)
    for k, v in res["betimleyici"].items():
        if k not in ("yillik_portfoy_iwm", "dusus_pencereleri"):
            print(k, v)
    print("ÖLÇÜTLER", R)
    print("HÜKÜM:", verdict)


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):                  # Windows konsolu "→" yazamıyor (cp1254)
        _s.reconfigure(encoding="utf-8", errors="replace")
    cmds = {k[4:]: v for k, v in list(globals().items()) if k.startswith("cmd_")}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(f"usage: python lab/oos_r.py [{'|'.join(cmds)}]")
    cmds[sys.argv[1]]()
