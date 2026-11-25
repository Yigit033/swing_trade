"""Ek U — MOM "lider rotasyonu" gölge ileri testi.

Kural lab'da sınanan kuralın aynısı olmalı (Ek T, lab/rotation.py): ay sonu seansında karar, ertesi açılışta
işlem; uygun = büyük hisse havuzunda ve 200 günlük ortalamanın üstünde; sıra > 20 ya da uygun değil → sat;
yalnız BULL'da al; en çok 10 pozisyon. Geçmiş sınavda KALDI — burada yalnız ileri veride izlenir.
"""
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from swing_trader.shadow import report, rules, runner
from swing_trader.shadow.ledger import ShadowLedger
from swing_trader.tests.test_shadow_runner import FakeSrc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab import rotation as lab_rotation  # noqa: E402

NY = ZoneInfo("America/New_York")
TICKERS = [f"T{i:02d}" for i in range(25)]             # T00 en hızlı yükselen … T24 en yavaş
CAP = 2e10


def _closes(days, mutate=None) -> pd.DataFrame:
    n = len(days)
    c = pd.DataFrame({t: 50.0 * (1 + 0.0010 - 0.00003 * i) ** np.arange(n) for i, t in enumerate(TICKERS)},
                     index=pd.DatetimeIndex(pd.to_datetime(days)))
    if mutate:
        mutate(c)
    return c


def _bars(close: pd.Series, volume=2e6) -> pd.DataFrame:
    return pd.DataFrame(dict(Date=close.index, Open=close.to_numpy() * 0.999, High=close.to_numpy() * 1.01,
                             Low=close.to_numpy() * 0.99, Close=close.to_numpy(), Volume=volume))


# ── Kilit ve takvim ─────────────────────────────────────────────────────────

def test_params_frozen_to_preregistration():
    """Ek U parametreleri. Değiştirmek yeni ön kayıt gerektirir — bu hash'i güncellemeden önce
    PREREGISTRATION.md'ye yeni bir ek yaz."""
    p = rules.PARAMS_MOM
    assert (p["lookback"], p["skip"], p["sma"], p["max_positions"], p["sell_rank"]) == (252, 21, 200, 10, 20)
    assert (p["price_min"], p["avgvol63_min"], p["dv20_min"], p["cap_min"]) == (7.0, 300_000, 5e6, 1e10)
    assert (p["buy_regime"], p["regime_index"], p["cost_pct"], p["fee_per_order"]) == ("BULL", "SPY", 0.10, 1.5)
    assert (p["min_closed"], p["min_months"], p["max_months"]) == (40, 24, 36)
    assert rules.PARAMS_MOM_SHA256 == "ac313589e336573aa32fcfb0c2bbe2042fe88c4de315051be1ebc0dbc3477235"
    assert "MOM" not in rules.EXITS and "MOM" not in rules.RUNNERS      # çıkışı aylık sıraya bağlı
    # diğer kilitler değişmedi
    assert rules.PARAMS_SHA256.startswith("178c293e") and rules.PARAMS_H6S_SHA256.startswith("47dfd9d0")


def test_month_end_sessions_follow_the_exchange_calendar():
    assert rules.is_month_end_session(date(2026, 11, 30)) and not rules.is_month_end_session(date(2026, 11, 27))
    assert rules.is_month_end_session(date(2026, 1, 30))                # 31 Ocak Cumartesi
    assert rules.is_month_end_session(date(2025, 12, 31))
    assert not rules.is_month_end_session(date(2026, 5, 31))            # Pazar: seans değil
    assert rules.last_month_end_session(date(2026, 5, 31)) == date(2026, 5, 29)
    assert rules.last_month_end_session(date(2026, 12, 15)) == date(2026, 11, 30)
    assert rules.last_month_end_session(date(2026, 11, 30)) == date(2026, 11, 30)


# ── Saf kurallar ────────────────────────────────────────────────────────────

def _frame(n=300, price=50.0, growth=0.001, volume=2e6):
    days = rules.sessions_between(date(2024, 1, 2), date(2026, 3, 31))[:n]
    c = pd.Series(price * (1 + growth) ** np.arange(n), index=pd.DatetimeIndex(pd.to_datetime(days)))
    return _bars(c, volume).set_index("Date")


def test_facts_score_pool_and_trend():
    f = _frame()
    i = len(f) - 1
    x = rules.mom_facts(f, i, CAP)
    c = f["Close"]
    assert x["score"] == pytest.approx(c.iloc[i - 21] / c.iloc[i - 252] - 1)       # son ay atlanır
    assert x["in_pool"] and x["above"]
    assert not rules.mom_facts(f, i, 9.9e9)["in_pool"]                              # değer < 10 mlr $
    assert not rules.mom_facts(f, i, None)["in_pool"]
    assert not rules.mom_facts(_frame(price=3.0), i, CAP)["in_pool"]                # fiyat < 7 $
    assert not rules.mom_facts(_frame(volume=1e5), i, CAP)["in_pool"]               # ort. hacim < 300K
    assert not rules.mom_facts(_frame(price=8.0, growth=0.0, volume=4e5), i, CAP)["in_pool"]   # dolar hacim < 5M
    assert not rules.mom_facts(_frame(growth=-0.001), i, CAP)["above"]              # 200 günlük ortalamanın altı
    assert np.isnan(rules.mom_facts(f.iloc[:252], 251, CAP)["score"])               # 253 seans gerekir
    short = rules.mom_facts(f.iloc[:150], 149, CAP)
    assert np.isnan(short["score"]) and not short["above"]                          # SMA200 için de geçmiş yetmiyor


def _facts(scores, pool=True, above=True):
    return {t: dict(in_pool=pool, above=above, score=s) for t, s in scores.items()}


def test_decide_buys_top_ten_only_in_bull_and_keeps_within_twenty():
    scores = {f"S{i:02d}": 1.0 - i * 0.01 for i in range(30)}                       # S00 en güçlü
    d = rules.mom_decide(_facts(scores), [], "BULL")
    assert d["buys"] == [f"S{i:02d}" for i in range(10)] and d["sells"] == [] and d["eligible"] == 30
    for regime in ("CAUTION", "BEAR", "UNKNOWN"):
        assert rules.mom_decide(_facts(scores), [], regime)["buys"] == []
    held = ["S00", "S15", "S19", "S20", "S25"]                                      # sıra 1, 16, 20, 21, 26
    d = rules.mom_decide(_facts(scores), held, "BULL")
    assert d["sells"] == [("S20", "RANK"), ("S25", "RANK")]                         # sıra > 20 → sat
    assert d["buys"] == ["S01", "S02", "S03", "S04", "S05", "S06", "S07"]           # 3 kalan + 7 = 10
    d = rules.mom_decide(_facts(scores), held, "CAUTION")
    assert d["sells"] == [("S20", "RANK"), ("S25", "RANK")] and d["buys"] == []     # satış rejime bakmaz


def test_decide_sell_reasons():
    f = _facts({f"S{i:02d}": 1.0 - i * 0.01 for i in range(12)})
    f["OUT"] = dict(in_pool=False, above=True, score=5.0)
    f["DOWN"] = dict(in_pool=True, above=False, score=5.0)
    f["NOSCORE"] = dict(in_pool=True, above=True, score=float("nan"))
    d = rules.mom_decide(f, ["OUT", "DOWN", "NOSCORE", "GONE", "S03"], "BULL")
    assert d["sells"] == [("OUT", "POOL"), ("DOWN", "TREND"), ("NOSCORE", "NA"), ("GONE", "POOL")]
    assert "S03" not in d["buys"] and len(d["buys"]) == 9                           # elde 1 + 9 alım


# ── Lab paritesi: production kuralı lab/rotation.mom_trades ile aynı işlemleri üretir ──────────

def test_production_rule_reproduces_the_lab_rotation():
    days = rules.sessions_between(date(2014, 10, 1), date(2016, 6, 30))             # gerçek NYSE takvimi
    idx = pd.DatetimeIndex(pd.to_datetime(days))
    TICKERS = [f"T{i:02d}" for i in range(40)]                                       # sık aralıklı 40 hisse
    C = pd.DataFrame({t: 50.0 * (1 + 0.0012 - 0.00002 * i) ** np.arange(len(days)) for i, t in enumerate(TICKERS)},
                     index=idx)
    C.iloc[idx.get_loc(pd.Timestamp("2016-01-15")):, C.columns.get_loc("T00")] *= 0.5        # trend bozulur → TREND
    j = idx.get_loc(pd.Timestamp("2016-01-05"))
    C.iloc[j:, C.columns.get_loc("T01")] = C.iloc[j, C.columns.get_loc("T01")]               # yerinde sayar → sırası düşer → RANK
    P = {"Open": C * 0.999, "Close": C, "High": C * 1.01, "Low": C * 0.99}
    cap = pd.DataFrame(CAP, index=idx, columns=TICKERS)
    cap.loc["2016-03-31", "T03"] = 5e9                                              # havuzdan çıkar → POOL
    pool = cap >= 1e10
    regimes = {d: ("BULL", "CONFIRMED") for d in idx}
    regimes[pd.Timestamp("2016-01-29")] = ("CAUTION", "CONFIRMED")                  # o ay sonunda alım yok
    start, end = "2016-01-04", "2016-06-30"
    lab = lab_rotation.mom_trades(P, pool, regimes, start, end)

    V = pd.DataFrame(2e6, index=idx, columns=TICKERS)
    decision_days = [d for d in days if rules.is_month_end_session(d)
                     and date(2015, 12, 31) <= d < date(2016, 6, 30)]
    held, got = {}, []
    for t in decision_days:
        i = idx.get_loc(pd.Timestamp(t))
        d1 = idx[i + 1]
        facts = {}
        for tk in TICKERS:
            fr = pd.DataFrame(dict(Open=P["Open"][tk], High=P["High"][tk], Low=P["Low"][tk],
                                   Close=C[tk], Volume=V[tk])).iloc[: i + 1]
            facts[tk] = rules.mom_facts(fr, i, float(cap.at[pd.Timestamp(t), tk]))
        d = rules.mom_decide(facts, list(held), regimes[pd.Timestamp(t)][0])
        for tk, why in d["sells"]:
            got.append((tk, held.pop(tk), d1, why))
        for tk in d["buys"]:
            held[tk] = d1
    want = sorted((r.ticker, r.entry_day, r.exit_day, r.reason) for r in lab.itertuples() if r.reason != "END")
    assert sorted(got) == want and len(want) >= 3
    assert {r for _, _, _, r in got} == {"TREND", "RANK", "POOL"}                   # üç satış nedeni de sınandı
    assert sorted(held.items()) == sorted((r.ticker, r.entry_day) for r in lab.itertuples() if r.reason == "END")


# ── Koşucu: karar → giriş → satış kararı → çıkış ────────────────────────────

class MomSrc(FakeSrc):
    """25 büyük hisse, 2024-01 → 2026-03 gerçek seans takviminde; SPY yükseliyor (BULL)."""

    def __init__(self, mutate=None):
        super().__init__()
        days = rules.sessions_between(date(2024, 1, 2), date(2026, 3, 31))
        C = _closes(days, mutate)
        self.bars = {t: _bars(C[t]) for t in TICKERS}
        base = pd.Series(100.0 * 1.0006 ** np.arange(len(days)), index=C.index)
        for sym, mult in (("SPY", 4.0), ("IWM", 2.0)):
            self.bars[sym] = _bars(base * mult, 5e7)
        self.bars["^VIX"] = pd.DataFrame(dict(Date=C.index, Open=15.0, High=15.5, Low=14.5, Close=15.0, Volume=0.0))
        self.caps = {t: CAP for t in TICKERS}

    def mom_universe(self):
        return dict(self.caps)

    def candidate_universe(self):
        return {"T00"}

    def earnings_reporters(self, start, end):
        return pd.DataFrame()


def _run(L, src, y, m, d, hour=7, **kw):
    return runner.run_shadow_daily(now=datetime(y, m, d, hour, 0, tzinfo=NY), ledger=L, src=src, **kw)


def test_rotation_lifecycle_decide_enter_sell_exit(tmp_path):
    def crash(c):                                         # T00 Şubat ortasında çöker → SMA200 altı
        i = c.index.get_loc(pd.Timestamp("2026-02-13"))
        c.iloc[i:, c.columns.get_loc("T00")] *= 0.5
    L, src = ShadowLedger(str(tmp_path / "s.db")), MomSrc(crash)

    out = _run(L, src, 2026, 1, 30)                       # Cuma sabahı: son seans Perşembe 29 Ocak
    assert out["MOM"] == dict(rebalance=None, last_rebalance=None)
    assert L.get_meta("mom_live_start") == "2026-01-29" and L.get_meta("params_mom_sha256") == rules.PARAMS_MOM_SHA256

    out = _run(L, src, 2026, 2, 2)                        # Pazartesi sabahı: 30 Ocak ayın son seansı
    m = out["MOM"]
    assert (m["rebalance"], m["regime"], m["universe"], m["eligible"], m["buys"], m["sells"]) == \
        ("2026-01-30", "BULL", 25, 25, 10, 0)
    rows = L.trades("MOM")
    assert sorted(r["ticker"] for r in rows) == TICKERS[:10]
    assert all(r["status"] == "PENDING" and r["signal_date"] == "2026-01-30" and r["late_decision"] == 0
               for r in rows)
    assert L.get_meta("mom_last_rebalance") == "2026-01-30"
    assert _run(L, src, 2026, 2, 2, hour=8)["MOM"]["rebalance"] is None            # aynı karar yeniden verilmez

    _run(L, src, 2026, 2, 3)                              # giriş: 2 Şubat açılışı
    t00 = next(r for r in L.trades("MOM") if r["ticker"] == "T00")
    want_open = float(src.bars["T00"].set_index("Date").at[pd.Timestamp("2026-02-02"), "Open"])
    assert t00["status"] == "OPEN" and t00["entry_date"] == "2026-02-02" and t00["stop"] is None
    assert t00["entry_price"] == pytest.approx(want_open, abs=1e-3)

    out = _run(L, src, 2026, 3, 2)                        # 27 Şubat ayın son seansı: T00 trendi bozuldu
    assert (out["MOM"]["rebalance"], out["MOM"]["sells"], out["MOM"]["buys"]) == ("2026-02-27", 1, 1)
    t00 = next(r for r in L.trades("MOM") if r["ticker"] == "T00")
    assert t00["status"] == "OPEN" and t00["details"]["sell_signal_date"] == "2026-02-27"
    assert t00["details"]["sell_reason"] == "TREND" and t00["details"]["sell_late"] == 0
    t10 = next(r for r in L.trades("MOM") if r["ticker"] == "T10")
    assert t10["status"] == "PENDING" and t10["signal_date"] == "2026-02-27"

    _run(L, src, 2026, 3, 3)                              # çıkış ve yeni giriş: 2 Mart açılışı
    t00 = next(r for r in L.trades("MOM") if r["ticker"] == "T00")
    bars = src.bars["T00"].set_index("Date")
    assert (t00["status"], t00["exit_date"], t00["exit_reason"]) == ("CLOSED", "2026-03-02", "TREND")
    assert t00["exit_price"] == pytest.approx(float(bars.at[pd.Timestamp("2026-03-02"), "Open"]), abs=1e-3)
    assert t00["ret_pct"] < -40 and t00["net_pct"] == pytest.approx(t00["ret_pct"] - 0.10, abs=1e-6)
    spy = src.bars["SPY"].set_index("Date")
    assert t00["iwm_ret"] == pytest.approx(100 * (spy.at[pd.Timestamp("2026-03-02"), "Open"]
                                                  / spy.at[pd.Timestamp("2026-02-02"), "Open"] - 1), abs=1e-3)
    assert t00["bars"] == len(rules.sessions_between(date(2026, 2, 2), date(2026, 3, 2))) - 1
    assert t00["details"]["path"][-1] == ["2026-03-02", pytest.approx(t00["exit_price"], abs=1e-3)]
    t10 = next(r for r in L.trades("MOM") if r["ticker"] == "T10")
    assert t10["status"] == "OPEN" and t10["entry_date"] == "2026-03-02"
    active = [r for r in L.trades("MOM") if r["status"] == "OPEN"]
    assert len(active) == 10                              # 9 eski + 1 yeni

    rep = L.get_meta("report")
    mom = rep["strategies"]["MOM"]
    assert (mom["benchmark"], mom["cost_pct"], mom["fee_per_order"]) == ("SPY", 0.10, 1.5)
    assert (mom["gate"]["status"], mom["gate"]["need_closed"], mom["gate"]["need_months"]) == ("TOPLANIYOR", 40, 24)
    assert mom["closed"] == 1 and mom["portfolio_10pct"]["taken"] == 10             # nakit 10 pozisyona yeter
    assert rep["params_mom_sha256"] == rules.PARAMS_MOM_SHA256
    assert rep["strategies"]["H5"]["benchmark"] == "IWM" and rep["strategies"]["H5"]["cost_pct"] == 0.5
    issues = runner.run_health(L, datetime(2026, 3, 3, 8, 0, tzinfo=NY))["issues"]
    assert not [i for i in issues if i.startswith(("MOM", "update"))], issues      # (H4 birikimi bu testte beklenir)


def test_decision_after_the_open_is_flagged_late(tmp_path):
    L, src = ShadowLedger(str(tmp_path / "s.db")), MomSrc()
    _run(L, src, 2026, 1, 30)
    _run(L, src, 2026, 2, 2, hour=11)                     # açılıştan sonra karar
    assert all(r["late_decision"] == 1 for r in L.trades("MOM"))


def test_missed_month_end_is_processed_late_not_lost(tmp_path):
    L, src = ShadowLedger(str(tmp_path / "s.db")), MomSrc()
    _run(L, src, 2026, 1, 30)
    out = _run(L, src, 2026, 2, 4)                        # 2 ve 3 Şubat koşuları kaçtı
    assert out["MOM"]["rebalance"] == "2026-01-30" and out["MOM"]["buys"] == 10
    rows = L.trades("MOM")
    assert all(r["late_decision"] == 1 and r["status"] == "OPEN" and r["entry_date"] == "2026-02-02" for r in rows)
    assert L.get_meta("report")["strategies"]["MOM"]["late_decisions"] == 10       # değerlendirmeye girmez


def test_unavailable_universe_or_stale_bars_defer_the_decision(tmp_path):
    L, src = ShadowLedger(str(tmp_path / "s.db")), MomSrc()
    _run(L, src, 2026, 1, 30)
    caps, src.caps = src.caps, {}
    out = _run(L, src, 2026, 2, 2)
    assert out["MOM"] == dict(rebalance="2026-01-30", deferred="UNIVERSE_UNAVAILABLE") and L.trades("MOM") == []
    h = runner.run_health(L, datetime(2026, 2, 2, 7, 30, tzinfo=NY))
    assert h["complete"] is False and "MOM: UNIVERSE_UNAVAILABLE" in h["issues"]
    src.caps = caps
    stale = {t: b[b.Date < pd.Timestamp("2026-01-30")] for t, b in src.bars.items() if t in TICKERS}
    full = dict(src.bars)
    src.bars.update(stale)                                # karar günü barları henüz gelmedi
    out = _run(L, src, 2026, 2, 2, hour=8, if_needed=True)
    assert out["MOM"]["deferred"] == "NO_CURRENT_BARS" and L.trades("MOM") == []
    src.bars = full
    out = _run(L, src, 2026, 2, 2, hour=9, if_needed=True)                          # yedek tetikleyici yeniden dener
    assert out["MOM"]["buys"] == 10 and all(r["late_decision"] == 0 for r in L.trades("MOM"))
    assert _run(L, src, 2026, 2, 2, hour=9, if_needed=True)["skipped"] == "ALREADY_COMPLETE"


def test_spy_bar_missing_defers(tmp_path):
    L, src = ShadowLedger(str(tmp_path / "s.db")), MomSrc()
    _run(L, src, 2026, 1, 30)
    b = src.bars["SPY"]
    src.bars["SPY"] = b[b.Date < pd.Timestamp("2026-01-30")]
    assert _run(L, src, 2026, 2, 2)["MOM"]["deferred"] == "SPY_BAR_MISSING"


def test_no_buys_when_regime_is_not_bull(tmp_path, monkeypatch):
    L, src = ShadowLedger(str(tmp_path / "s.db")), MomSrc()
    _run(L, src, 2026, 1, 30)
    monkeypatch.setattr(runner, "_regime_on", lambda bars, index, day: {"regime": "CAUTION"})
    out = _run(L, src, 2026, 2, 2)
    assert (out["MOM"]["regime"], out["MOM"]["buys"]) == ("CAUTION", 0) and L.trades("MOM") == []
    assert L.get_meta("mom_last_rebalance") == "2026-01-30"                         # karar verildi: alım yok


def test_params_change_blocks_only_mom(tmp_path):
    L, src = ShadowLedger(str(tmp_path / "s.db")), MomSrc()
    L.set_meta("params_mom_sha256", "eski-kurallar")
    out = _run(L, src, 2026, 2, 2)
    assert out["MOM"]["error"] == "MOM_PARAMS_CHANGED_WITHOUT_PREREGISTRATION" and L.trades("MOM") == []
    assert "error" not in out["H6"]


def test_missing_entry_bar_skips_the_buy(tmp_path):
    L, src = ShadowLedger(str(tmp_path / "s.db")), MomSrc()
    _run(L, src, 2026, 1, 30)
    _run(L, src, 2026, 2, 2)
    b = src.bars["T05"]
    src.bars["T05"] = b[b.Date != pd.Timestamp("2026-02-02")]                       # giriş günü barı hiç gelmedi
    _run(L, src, 2026, 2, 5)
    t05 = next(r for r in L.trades("MOM") if r["ticker"] == "T05")
    assert (t05["status"], t05["reason"]) == ("SKIPPED", "NO_ENTRY_BAR")


def test_missing_mom_block_counts_as_incomplete(tmp_path):
    L, src = ShadowLedger(str(tmp_path / "s.db")), MomSrc()
    _run(L, src, 2026, 1, 30)
    lr = L.get_meta("last_run")
    del lr["MOM"]                                         # MOM eklenmeden önceki sürümün koşusu
    L.set_meta("last_run", lr)
    assert "MOM: MISSING" in runner.run_health(L, datetime(2026, 1, 30, 8, 0, tzinfo=NY))["issues"]


# ── Rapor: maliyet, sabit ücret, kıyas ──────────────────────────────────────

def test_portfolio_fee_matches_the_lab_formula():
    days = rules.sessions_between(date(2026, 2, 2), date(2026, 2, 6))
    t = dict(strategy="MOM", ticker="A", entry_date="2026-02-02", entry_price=100.0, exit_date="2026-02-05",
             exit_price=110.0, status="CLOSED", key=1.0,
             details=dict(path=[["2026-02-02", 101.0], ["2026-02-03", 104.0], ["2026-02-04", 108.0],
                                ["2026-02-05", 110.0]]))
    p = report.portfolio([t], days, cost=0.10, fee=1.5)
    sh = (1000.0 - 1.5) / (100.0 * 1.0005)                 # lab/swing_o.portfolio: (alloc − ücret) / (fiyat × (1 + maliyet/2))
    want = 10_000.0 - 1000.0 + sh * 110.0 * (1 - 0.0005) - 1.5
    assert p["total_pct"] == pytest.approx((want / 10_000.0 - 1) * 100, abs=0.01) and p["taken"] == 1
    free = report.portfolio([t], days, cost=0.10)          # ücretsiz hali (diğer stratejiler) değişmedi
    assert free["total_pct"] > p["total_pct"]


def test_strategy_report_uses_mom_cost_and_gate():
    assert report.GATE["MOM"] == dict(min_closed=40, min_months=24, max_months=36)
    days = rules.sessions_between(date(2026, 2, 2), date(2026, 3, 31))
    rows = [dict(strategy="MOM", ticker=f"A{i}", status="CLOSED", entry_date="2026-02-02", entry_price=100.0,
                 exit_date="2026-03-02", exit_price=105.0, ret_pct=5.0, iwm_ret=2.0, key=1.0, late_decision=0,
                 warmup=0, details=dict(path=[["2026-03-02", 105.0]])) for i in range(3)]
    r = report.strategy_report("MOM", rows, days, date(2026, 2, 2), date(2026, 3, 31), 3.0)
    assert r["net_mean"] == pytest.approx(4.9) and r["excess_iwm_mean"] == pytest.approx(3.0)
    assert (r["benchmark"], r["cost_pct"], r["iwm_buy_hold_pct"]) == ("SPY", 0.10, 3.0)
    assert r["gate"]["status"] == "TOPLANIYOR" and r["gate"]["F3_beats_iwm"] is False   # 3 işlem portföyü +%1.4 < %3
