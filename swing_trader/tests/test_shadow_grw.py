"""Ek W — GRW "büyüme + kırılım" gölge ileri testi.

Kural lab'da ölçülen kuralın aynısı olmalı (Ek V "lider şartsız" kıyası, lab/pro_rules.py): havuz G'de VCE
kırılımı + SEC büyüme şartı + SPY rejimi BULL; giriş ertesi açılış; stop = sinyal günü dibi (en çok %8);
çıkış = Close < SMA50 → ertesi açılış. Ön kayıtlı paket geçmiş sınavda KALDI — burada yalnız ileri veride izlenir.
"""
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from swing_trader.shadow import growth, report, rules, runner
from swing_trader.shadow.ledger import ShadowLedger
from swing_trader.tests.test_shadow_runner import FakeSrc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab import pro_rules as lab_pro  # noqa: E402

NY = ZoneInfo("America/New_York")


# ── Kilit ───────────────────────────────────────────────────────────────────

def test_params_frozen_to_preregistration():
    """Ek W parametreleri. Değiştirmek yeni ön kayıt gerektirir — bu hash'i güncellemeden önce
    PREREGISTRATION.md'ye yeni bir ek yaz."""
    p = rules.PARAMS_GRW
    assert (p["price_min"], p["avgvol63_min"], p["dv20_min"], p["cap_min"]) == (10.0, 300_000, 1e7, 1e9)
    assert (p["min_sales_growth"], p["max_stale_days"], tuple(p["yoy_days"])) == (0.25, 200, (350, 380))
    assert (p["buy_regime"], p["regime_index"], p["max_stop"], p["sma"]) == ("BULL", "SPY", 0.08, 50)
    assert (p["risk"], p["risk_cold"], p["cold_streak"], p["max_frac"]) == (0.01, 0.005, 2, 0.25)
    assert (p["cost_pct"], p["fee_per_order"], p["min_closed"], p["min_months"], p["max_months"]) == (0.20, 1.5, 60, 24, 36)
    assert rules.PARAMS_GRW_SHA256 == "ca1d88da4f387ff773911638ed449d3ae0380eed61a95c3f7c5d629f93cc7198"
    assert "GRW" not in rules.EXITS and "GRW" not in rules.RUNNERS      # çıkışı SMA50'ye bağlı (grw_plan)
    # diğer kilitler değişmedi
    assert rules.PARAMS_SHA256.startswith("178c293e") and rules.PARAMS_H6_SHA256.startswith("86a8947e")
    assert rules.PARAMS_H6S_SHA256.startswith("47dfd9d0") and rules.PARAMS_H6SR_SHA256.startswith("6d9c5a98")
    assert rules.PARAMS_MOM_SHA256.startswith("ac313589")


def test_production_constants_equal_the_lab_constants():
    p = rules.PARAMS_GRW
    assert (growth.REV_TAGS, growth.NI_TAGS) == (lab_pro.REV_TAGS, lab_pro.NI_TAGS)
    assert (growth.Q_DAYS, growth.M9_DAYS, growth.FY_DAYS, growth.YOY_DAYS) == (
        lab_pro.Q_DAYS, lab_pro.M9_DAYS, lab_pro.FY_DAYS, lab_pro.YOY_DAYS)
    assert (growth.MIN_SALES_GROWTH, growth.MAX_STALE_DAYS) == (lab_pro.MIN_SALES_GROWTH, lab_pro.MAX_STALE_DAYS)
    assert (p["min_sales_growth"], p["max_stale_days"]) == (lab_pro.MIN_SALES_GROWTH, lab_pro.MAX_STALE_DAYS)
    assert (p["max_stop"], p["risk"], p["risk_cold"], p["cold_streak"], p["max_frac"]) == (
        lab_pro.MAX_STOP, lab_pro.RISK, lab_pro.RISK_COLD, lab_pro.COLD_STREAK, lab_pro.MAX_FRAC)


# ── Büyüme verisi: lab ile birebir ───────────────────────────────────────────

def _random_facts(seed):
    """Çeyreklik + dokuz aylık + yıllık kayıtlar; bazıları sonradan düzeltilmiş, iki satış etiketi karışık."""
    rng = np.random.default_rng(seed)
    rev, alt, ni = [], [], []
    level, profit = 100.0, 5.0
    for y in range(2017, 2022):
        q_rev, q_ni = [], []
        for k, (ms, me, filed_m) in enumerate((("01-01", "03-31", "05"), ("04-01", "06-30", "08"), ("07-01", "09-30", "11"))):
            level *= 1 + rng.normal(0.06, 0.08)
            profit += rng.normal(0.5, 2.0)
            q_rev.append(level), q_ni.append(profit)
            filed = f"{y}-{filed_m}-{int(rng.integers(1, 20)):02d}"
            (rev if rng.random() < 0.7 else alt).append(dict(start=f"{y}-{ms}", end=f"{y}-{me}", val=level, filed=filed))
            ni.append(dict(start=f"{y}-{ms}", end=f"{y}-{me}", val=profit, filed=filed))
            if rng.random() < 0.5:                       # ertesi yıl karşılaştırma sütununda düzeltilmiş değer
                rev.append(dict(start=f"{y}-{ms}", end=f"{y}-{me}", val=level * 0.9, filed=f"{y + 1}-{filed_m}-21"))
            if k == 2:
                rev.append(dict(start=f"{y}-01-01", end=f"{y}-09-30", val=sum(q_rev), filed=filed))
                ni.append(dict(start=f"{y}-01-01", end=f"{y}-09-30", val=sum(q_ni), filed=filed))
        level *= 1 + rng.normal(0.06, 0.08)
        profit += rng.normal(0.5, 2.0)
        fy = f"{y + 1}-02-{int(rng.integers(10, 27)):02d}"
        rev.append(dict(start=f"{y}-01-01", end=f"{y}-12-31", val=sum(q_rev) + level, filed=fy))
        ni.append(dict(start=f"{y}-01-01", end=f"{y}-12-31", val=sum(q_ni) + profit, filed=fy))
    gaap = {"Revenues": {"units": {"USD": rev}}, "SalesRevenueNet": {"units": {"USD": alt}},
            "NetIncomeLoss": {"units": {"USD": ni}}}
    return {"facts": {"us-gaap": gaap}}


def test_growth_module_reproduces_the_lab_rules():
    cal = pd.bdate_range("2018-01-02", "2022-06-30")
    seen_ok = 0
    for seed in range(12):
        f = _random_facts(seed)
        q, ql = growth.quarters_table(f), lab_pro.quarters_table(f)
        pd.testing.assert_frame_equal(q, ql)
        assert len(q) >= 16                               # yıl sonu çeyrekleri de türetildi
        (k, o), (kl, ol) = growth.growth_asof(q, cal), lab_pro.growth_asof(ql, cal)
        assert (k == kl).all() and (o == ol).all()
        seen_ok += int(o.any())
        for i in (60, 400, 900):                          # tek günlük canlı sorgu = aynı günün panel değeri
            g = growth.growth_on(q, cal[i].date())
            assert (g["known"], g["ok"]) == (bool(k[i]), bool(o[i]))
    assert seen_ok >= 3                                   # test gerçekten şartı sağlayan dönemler içeriyor


def test_growth_detail_reports_the_quarter_and_the_growth_rate():
    f = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [dict(start="2018-01-01", end="2018-03-31", val=100.0, filed="2018-05-05"),
                                       dict(start="2019-01-01", end="2019-03-31", val=130.0, filed="2019-05-04")]}},
        "NetIncomeLoss": {"units": {"USD": [dict(start="2018-01-01", end="2018-03-31", val=5.0, filed="2018-05-05"),
                                            dict(start="2019-01-01", end="2019-03-31", val=9.0, filed="2019-05-04")]}}}}}
    q = growth.quarters_table(f)
    assert growth.growth_on(q, date(2019, 5, 3)) == dict(known=False, ok=False, quarter_end="2018-03-31")
    g = growth.growth_on(q, date(2019, 5, 6))
    assert (g["known"], g["ok"], g["quarter_end"], g["sales_growth_pct"]) == (True, True, "2019-03-31", 30.0)
    assert growth.growth_on(growth.quarters_table({}), date(2019, 5, 6)) == dict(known=False, ok=False)


# ── Havuz ve işlem planı ─────────────────────────────────────────────────────

def _df(n=130, price=50.0, volume=1e6):
    return pd.DataFrame(dict(Close=np.full(n, price), Volume=np.full(n, volume)))


@pytest.mark.parametrize("price,volume,cap,want", [
    (50.0, 1e6, 2e9, True),
    (9.9, 5e6, 2e9, False),                               # fiyat < 10
    (50.0, 2.9e5, 2e9, False),                            # ortalama hacim ≤ 300K (dolar hacim 14.5M)
    (12.0, 8e5, 2e9, False),                              # dolar hacim 9.6M ≤ 10M
    (50.0, 1e6, 9.9e8, False),                            # piyasa değeri < 1 mlr
    (50.0, 1e6, None, False),
])
def test_pool_g_membership(price, volume, cap, want):
    assert rules.grw_pool(_df(price=price, volume=volume), cap)[0] is want


def _lab_and_live(seed, n=260, n_signals=14):
    """Aynı rastgele fiyat yolunda lab işlemleri ve canlı planlar."""
    rng = np.random.default_rng(seed)
    days = rules.sessions_between(date(2024, 1, 2), date(2025, 6, 30))[:n]
    cal = pd.DatetimeIndex(pd.to_datetime(days))
    c = 50 * np.exp(np.cumsum(rng.normal(0.0008, 0.02, n)))
    o = c * (1 + rng.normal(0, 0.006, n))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.008, n)))
    P = {k: pd.DataFrame({"X": v}, index=cal) for k, v in (("Open", o), ("Low", lo), ("Close", c), ("High", np.maximum(o, c)))}
    sig = pd.DataFrame(False, index=cal, columns=["X"])
    for i in sorted(rng.choice(np.arange(70, n - 2), n_signals, replace=False)):
        sig.iat[int(i), 0] = True
    key = pd.DataFrame(1.0, index=cal, columns=["X"])
    T = lab_pro.pro_trades(P, sig, key, cal[60], cal[-1])
    frame = pd.DataFrame({k: P[k]["X"] for k in ("Open", "High", "Low", "Close")})
    frame["Volume"] = 1e6
    return T, frame, [d.date() for d in cal[sig["X"].values]]


def test_production_plan_reproduces_the_lab_trades():
    reasons = set()
    for seed in range(8):
        T, frame, signal_days = _lab_and_live(seed)
        assert len(T) >= 3
        for r in T.itertuples():
            plan = rules.grw_plan(frame, r.signal_date.date())
            assert plan["entry_day"] == r.entry_day.date() and plan["entry_price"] == pytest.approx(r.entry_price)
            assert plan["stop"] == pytest.approx(r.entry_price * (1 - r.stop_dist))
            assert plan["mark_day"] == r.exit_day.date() and plan["mark_price"] == pytest.approx(r.exit_price)
            assert plan["ret_pct"] == pytest.approx(r.ret_pct) and plan["bars"] == r.bars
            if r.reason == "END":                         # lab dönem sonunda kapatır; canlıda pozisyon hâlâ açık
                assert plan["status"] == "OPEN" and plan["exit_reason"] == "OPEN"
            else:
                assert (plan["status"], plan["exit_reason"], plan["at_open"]) == ("CLOSED", r.reason, bool(r.at_open))
            reasons.add(r.reason)
    assert {"STOP", "TRAIL"} <= reasons


def test_plan_edge_cases():
    days = rules.sessions_between(date(2025, 1, 2), date(2025, 6, 30))
    idx = pd.DatetimeIndex(pd.to_datetime(days))
    c = np.full(len(days), 100.0)
    f = pd.DataFrame(dict(Open=c, High=c, Low=c * 0.97, Close=c, Volume=1e6), index=idx)
    assert rules.grw_plan(f, date(2025, 1, 1))["reason"] == "NO_SIGNAL_BAR"
    assert rules.grw_plan(f, days[-1])["status"] == "PENDING"                # giriş barı henüz yok
    g = f.copy()
    g.iloc[61, g.columns.get_loc("Open")] = 96.5                              # giriş açılışı sinyal dibinin (97) altında
    assert rules.grw_plan(g, days[60])["reason"] == "GAP_BELOW_STOP"
    g = f.copy()
    g.iloc[61, g.columns.get_loc("Open")] = np.nan
    assert rules.grw_plan(g, days[60])["reason"] == "NO_ENTRY_BAR"
    g = f.copy()
    g.iloc[60, g.columns.get_loc("Low")] = 80.0                               # dip uzak → stop girişin %8 altı
    g.iloc[61:, g.columns.get_loc("Low")] = 99.0
    plan = rules.grw_plan(g, days[60])
    assert plan["stop"] == pytest.approx(92.0) and plan["status"] == "OPEN" and not plan["sell_pending"]


# ── Koşucu ───────────────────────────────────────────────────────────────────

OK = dict(known=True, ok=True, quarter_end="2025-03-31", sales_growth_pct=40.0)
ASOF = date(2025, 6, 13)


def _hit(t, key=0.5):
    return dict(ticker=t, cap=5e9, close=22.0, low=21.8, dv20=2e7, key=key)


def _run(L, src, y, m, d, hour=7, **kw):
    return runner.run_shadow_daily(now=datetime(y, m, d, hour, 0, tzinfo=NY), ledger=L, src=src, **kw)


def _ledger(tmp_path):
    return ShadowLedger(str(tmp_path / "s.db"))


def test_breakout_with_growth_in_bull_market_is_recorded_and_others_are_skipped_with_reasons(tmp_path):
    L, src = _ledger(tmp_path), FakeSrc()
    src.grw = {ASOF: [_hit("ABC", 0.9), _hit("XYZ", 0.4), _hit("NOD", 0.2)]}
    src.growth = {"ABC": OK, "XYZ": dict(known=True, ok=False, sales_growth_pct=3.0)}       # NOD: veri yok
    out = _run(L, src, 2025, 6, 16)
    g = out["GRW"]
    assert (g["hits"], g["regime"], g["added"], g["PENDING"], g["GROWTH_FAIL"], g["NO_GROWTH_DATA"]) == (
        3, "BULL", 3, 1, 1, 1)
    rows = {t["ticker"]: t for t in L.trades("GRW")}
    assert (rows["ABC"]["status"], rows["ABC"]["signal_date"], rows["ABC"]["late_decision"]) == ("PENDING", "2025-06-13", 0)
    assert rows["ABC"]["details"]["growth"]["sales_growth_pct"] == 40.0 and rows["ABC"]["key"] == 0.9
    assert (rows["XYZ"]["status"], rows["XYZ"]["reason"]) == ("SKIPPED", "GROWTH_FAIL")
    assert (rows["NOD"]["status"], rows["NOD"]["reason"]) == ("SKIPPED", "NO_GROWTH_DATA")
    assert L.get_meta("params_grw_sha256") == rules.PARAMS_GRW_SHA256 and L.get_meta("grw_live_start") == "2025-06-13"
    assert not [i for i in runner.run_health(L, now=datetime(2025, 6, 16, 7, 5, tzinfo=NY))["issues"] if "GRW" in i]


def test_no_buys_and_no_sec_calls_when_the_market_is_not_bull(tmp_path, monkeypatch):
    L, src = _ledger(tmp_path), FakeSrc()
    src.grw, src.growth = {ASOF: [_hit("ABC")]}, {"ABC": OK}
    monkeypatch.setattr(runner, "_regime_on", lambda bars, index, day: dict(regime="CAUTION"))
    g = _run(L, src, 2025, 6, 16)["GRW"]
    assert (g["regime"], g["NOT_BULL"]) == ("CAUTION", 1) and src.growth_calls == []
    assert [(t["status"], t["reason"]) for t in L.trades("GRW")] == [("SKIPPED", "NOT_BULL")]


def test_open_position_blocks_a_new_signal_in_the_same_ticker(tmp_path):
    L, src = _ledger(tmp_path), FakeSrc()
    src.grw, src.growth = {ASOF: [_hit("ABC")], date(2025, 6, 18): [_hit("ABC")]}, {"ABC": OK}
    _run(L, src, 2025, 6, 16)
    g = _run(L, src, 2025, 6, 19)["GRW"]
    assert g["TICKER_BUSY"] == 1
    assert sorted((t["signal_date"], t["status"], t["reason"]) for t in L.trades("GRW")) == [
        ("2025-06-13", "OPEN", None), ("2025-06-18", "SKIPPED", "TICKER_BUSY")]


def test_unreachable_sources_defer_instead_of_recording_a_wrong_answer(tmp_path, monkeypatch):
    L, src = _ledger(tmp_path), FakeSrc()
    src.grw, src.growth = {ASOF: [_hit("ABC")]}, {"ABC": OK}

    def boom(ticker, day):
        raise OSError("SEC down")
    monkeypatch.setattr(src, "growth_check", boom)
    out = _run(L, src, 2025, 6, 16)
    assert out["GRW"]["deferred"] == "SEC_UNAVAILABLE" and L.trades("GRW") == []
    assert "GRW: SEC_UNAVAILABLE" in runner.run_health(L, now=datetime(2025, 6, 16, 7, 5, tzinfo=NY))["issues"]
    monkeypatch.undo()                                                      # yedek tetikleyici: SEC geri geldi
    out = _run(L, src, 2025, 6, 16, hour=8, if_needed=True)
    assert out["GRW"]["PENDING"] == 1 and L.trades("GRW")[0]["late_decision"] == 0

    L2, src2 = ShadowLedger(str(tmp_path / "t.db")), FakeSrc()
    monkeypatch.setattr(src2, "grw_hits", lambda asof: dict(hits=[], stats=dict(candidates=0, error="UNIVERSE_UNAVAILABLE")))
    assert _run(L2, src2, 2025, 6, 16)["GRW"]["deferred"] == "UNIVERSE_UNAVAILABLE"
    L3, src3 = ShadowLedger(str(tmp_path / "u.db")), FakeSrc()
    src3.grw = {ASOF: [_hit("ABC")]}
    src3.bars["SPY"] = src3.bars["SPY"][src3.bars["SPY"].Date < pd.Timestamp(ASOF)]
    assert _run(L3, src3, 2025, 6, 16)["GRW"]["deferred"] == "SPY_BAR_MISSING"


def test_decision_after_the_open_is_flagged_late_and_params_change_blocks_only_grw(tmp_path):
    L, src = _ledger(tmp_path), FakeSrc()
    src.grw, src.growth = {ASOF: [_hit("ABC")]}, {"ABC": OK}
    _run(L, src, 2025, 6, 16, hour=11)
    assert L.trades("GRW")[0]["late_decision"] == 1
    L.set_meta("params_grw_sha256", "eski")
    out = _run(L, src, 2025, 6, 17)
    assert out["GRW"] == dict(error="GRW_PARAMS_CHANGED_WITHOUT_PREREGISTRATION")
    assert "error" not in out["H6"] and "error" not in out["update"]


class TrailSrc(FakeSrc):
    """ABC 20 Haziran'a kadar yükselir, sonra düşer: kapanış 50 günlük ortalamanın altına iner."""

    def __init__(self, drop=0.985, wide_stop=True):
        super().__init__()
        b = self.bars["ABC"].copy()
        i0 = int(np.where(b.Date == pd.Timestamp("2025-06-20"))[0][0])
        c = b.Close.to_numpy().copy()
        c[i0 + 1:] = c[i0] * drop ** np.arange(1, len(c) - i0)
        low = c * 0.995
        if wide_stop:                                    # sinyal günü dibi uzak → stop girişin %8 altı
            low[int(np.where(b.Date == pd.Timestamp(ASOF))[0][0])] *= 0.90
        self.bars["ABC"] = pd.DataFrame(dict(Date=b.Date, Open=c * 0.999, High=c * 1.01, Low=low, Close=c, Volume=1e6))
        self.grw, self.growth = {ASOF: [_hit("ABC")]}, {"ABC": OK}


def test_lifecycle_enter_hold_and_exit_at_the_open_after_the_close_below_sma50(tmp_path):
    L, src = _ledger(tmp_path), TrailSrc()
    _run(L, src, 2025, 6, 16)
    assert L.trades("GRW")[0]["status"] == "PENDING"
    _run(L, src, 2025, 6, 17)
    t = L.trades("GRW")[0]
    b = src.bars["ABC"].set_index("Date")
    low_signal, entry = float(b.loc["2025-06-13", "Low"]), float(b.loc["2025-06-16", "Open"])
    assert (t["status"], t["entry_date"]) == ("OPEN", "2025-06-16") and t["entry_price"] == pytest.approx(entry, abs=1e-3)
    assert t["stop"] == pytest.approx(max(low_signal, 0.92 * entry), abs=1e-3)

    _run(L, src, 2025, 7, 25)
    t = L.trades("GRW")[0]
    s50 = b.Close.rolling(50).mean()
    after = b.loc["2025-06-16":]
    first_below = after.index[(after.Close < s50.loc[after.index]).to_numpy()][0]
    exit_day = b.index[b.index.get_loc(first_below) + 1]
    assert after.loc[:first_below, "Low"].min() > t["stop"]                 # stop çalışmadı
    assert (t["status"], t["exit_reason"], t["exit_date"]) == ("CLOSED", "TRAIL", exit_day.date().isoformat())
    assert t["exit_price"] == pytest.approx(float(b.loc[exit_day, "Open"]), abs=1e-3)
    assert t["net_pct"] == pytest.approx(t["ret_pct"] - 0.20, abs=1e-6)
    spy = src.bars["SPY"].set_index("Date")
    want = 100 * (float(spy.loc[exit_day, "Open"]) / float(spy.loc["2025-06-16", "Open"]) - 1)
    assert t["iwm_ret"] == pytest.approx(want, abs=1e-3)                    # kıyas SPY, aynı açılış → açılış
    rep = L.get_meta("report")
    r = rep["strategies"]["GRW"]
    assert (r["benchmark"], r["cost_pct"], r["fee_per_order"], r["closed"]) == ("SPY", 0.20, 1.5, 1)
    assert r["sizing"] == dict(risk=0.01, cold=0.005, streak=2, max_frac=0.25)
    assert (r["gate"]["status"], r["gate"]["need_closed"], r["gate"]["need_months"]) == ("TOPLANIYOR", 60, 24)
    assert rep["params_grw_sha256"] == rules.PARAMS_GRW_SHA256


def test_stop_exit_and_gap_below_the_signal_low(tmp_path):
    L, src = _ledger(tmp_path), TrailSrc(wide_stop=False)                   # stop sinyal günü dibinde: önce o çalışır
    _run(L, src, 2025, 6, 16)
    _run(L, src, 2025, 7, 25)
    t = L.trades("GRW")[0]
    assert (t["status"], t["exit_reason"]) == ("CLOSED", "STOP") and t["exit_price"] <= t["stop"] + 1e-6

    L2, src2 = ShadowLedger(str(tmp_path / "t.db")), FakeSrc()
    src2.grw, src2.growth = {ASOF: [_hit("ABC")]}, {"ABC": OK}
    b = src2.bars["ABC"]
    b.loc[b.Date == pd.Timestamp("2025-06-16"), "Open"] = float(b.loc[b.Date == pd.Timestamp("2025-06-13"), "Low"].iloc[0]) - 0.01
    _run(L2, src2, 2025, 6, 16)
    _run(L2, src2, 2025, 6, 17)
    assert [(t["status"], t["reason"]) for t in L2.trades("GRW")] == [("SKIPPED", "GAP_BELOW_STOP")]


def test_missing_grw_block_counts_as_incomplete(tmp_path):
    L, src = _ledger(tmp_path), FakeSrc()
    _run(L, src, 2025, 6, 16)
    lr = L.get_meta("last_run")
    lr.pop("GRW")
    L.set_meta("last_run", lr)
    h = runner.run_health(L, now=datetime(2025, 6, 16, 7, 5, tzinfo=NY))
    assert "GRW: MISSING" in h["issues"] and not h["complete"]


# ── Rapor: risk tabanlı boyut ────────────────────────────────────────────────

def _trade(ticker, entry_d, exit_d, entry, exit_, stop, key=1.0):
    return dict(strategy="GRW", ticker=ticker, status="CLOSED", entry_date=entry_d, exit_date=exit_d, entry_price=entry,
                exit_price=exit_, stop=stop, key=key, ret_pct=100 * (exit_ / entry - 1), details=dict(path=[]))


def test_report_portfolio_sizes_by_risk_like_the_lab():
    days = rules.sessions_between(date(2025, 6, 2), date(2025, 7, 31))
    iso = [d.isoformat() for d in days]
    sizing = report.SIZING_BY["GRW"]
    # %8 stop → %12.5 pozisyon; %2 stop → tavan %25
    rows = [_trade("A", iso[1], iso[5], 100.0, 100.0, 92.0), _trade("B", iso[1], iso[5], 100.0, 100.0, 98.0)]
    p = report.portfolio(rows, days, cost=0.0, fee=0.0, sizing=sizing)
    assert p["taken"] == 2 and p["total_pct"] == 0.0
    rows = [_trade("A", iso[1], iso[5], 100.0, 110.0, 92.0)]
    assert report.portfolio(rows, days, cost=0.0, fee=0.0, sizing=sizing)["total_pct"] == pytest.approx(1.25, abs=0.01)
    rows = [_trade("B", iso[1], iso[5], 100.0, 110.0, 98.0)]
    assert report.portfolio(rows, days, cost=0.0, fee=0.0, sizing=sizing)["total_pct"] == pytest.approx(2.5, abs=0.01)
    # iki zarar üst üste → üçüncü işlemde risk yarıya iner (%10 stop: %10 yerine %5 pozisyon)
    rows = [_trade("A", iso[1], iso[3], 100.0, 95.0, 90.0), _trade("B", iso[5], iso[7], 100.0, 95.0, 90.0),
            _trade("C", iso[10], iso[12], 100.0, 110.0, 90.0)]
    p = report.portfolio(rows, days, cost=0.0, fee=0.0, sizing=sizing)
    assert p["total_pct"] == pytest.approx(100 * (0.995 ** 2 * 1.005 - 1), abs=0.01)
    # lab formülüyle aynı sonuç (aynı işlemler, 20 bps + 1.5$/emir)
    T = pd.DataFrame([dict(ticker=r["ticker"], entry_day=pd.Timestamp(r["entry_date"]), exit_day=pd.Timestamp(r["exit_date"]),
                           entry_price=r["entry_price"], exit_price=r["exit_price"], ret_pct=r["ret_pct"], key=1.0,
                           at_open=True, bars=3, stop_dist=(r["entry_price"] - r["stop"]) / r["entry_price"]) for r in rows])
    C = pd.DataFrame(100.0, index=pd.DatetimeIndex(pd.to_datetime(days)), columns=list("ABC"))
    lab = lab_pro.portfolio_risk(T, C, cost=0.20, fee=1.5)[0]
    live = report.portfolio(rows, days, cost=0.20, fee=1.5, sizing=sizing)
    assert live["total_pct"] == pytest.approx(lab["total_pct"], abs=0.06) and live["taken"] == lab["taken"] == 3


def test_other_strategies_keep_fixed_ten_percent_sizing():
    assert set(report.SIZING_BY) == {"GRW"}
    days = rules.sessions_between(date(2025, 6, 2), date(2025, 6, 30))
    iso = [d.isoformat() for d in days]
    row = dict(_trade("A", iso[1], iso[5], 100.0, 110.0, 98.0), strategy="H6S")
    assert report.portfolio([row], days, cost=0.0, fee=0.0)["total_pct"] == pytest.approx(1.0, abs=0.01)
