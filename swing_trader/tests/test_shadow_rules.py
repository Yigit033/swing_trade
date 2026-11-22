"""Gölge sinyal kuralları (H4/H5): birim testleri + lab paritesi.

Lab paritesi testleri ``output/lab`` verisi varsa koşar (yerel). Production kural
fonksiyonlarının, ön kayıtlı lab sonuçlarını (events_H4/H5, trades_H4/H5) birebir
ürettiğini doğrular — gölge mod, ölçülen stratejinin aynısını izlemeli.
"""
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swing_trader.shadow import rules
from swing_trader.shadow.ledger import ShadowLedger
from swing_trader.shadow.report import boot_ci, gate, portfolio

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "output" / "lab"


# ── Parametre kilidi ────────────────────────────────────────────────────────

def test_params_frozen_to_preregistration():
    """Ek A/C parametreleri. Değiştirmek yeni ön kayıt gerektirir — bu hash'i
    güncellemeden önce PREREGISTRATION.md'ye yeni bir ek yaz."""
    assert rules.PARAMS["H4"]["window_days"] == 30 and rules.PARAMS["H4"]["min_insiders"] == 2
    assert rules.PARAMS["H4"]["min_value"] == 100_000 and rules.PARAMS["H4"]["gap_days"] == 60
    assert rules.PARAMS["H5"]["min_surprise"] == 10.0 and rules.PARAMS["H5"]["min_abs_estimate"] == 0.02
    assert rules.C["stop_atr"] == 2.5 and rules.C["max_bars"] == 20 and rules.C["cost_pct"] == 0.5
    assert rules.PARAMS_SHA256 == "178c293e6718" + rules.PARAMS_SHA256[12:]


# ── Çıkış ───────────────────────────────────────────────────────────────────

def _frame(rows, start="2025-03-03"):
    idx = pd.DatetimeIndex(rules.sessions_between(date.fromisoformat(start), date(2025, 12, 31))[:len(rows)])
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=idx)


def test_walk_stop_first_and_gap_fill_at_open():
    f = _frame([(10, 10.5, 9.5, 10, 1e6), (9, 9.2, 8.5, 9, 1e6)])
    assert rules.walk(f, 0, 9.4) == (1, 9.0, "STOP")          # gap altı açılış → Open'dan
    f = _frame([(10, 12, 9.3, 11.5, 1e6)])
    assert rules.walk(f, 0, 9.4) == (0, 9.4, "STOP")          # aynı barda stop önce


def test_walk_time_exit_on_20th_session_and_open_mark():
    rows = [(10, 10.2, 9.9, 10 + k * 0.01, 1e6) for k in range(25)]
    f = _frame(rows)
    xi, xp, why = rules.walk(f, 1, 5.0)
    assert (xi, why) == (20, "TIME") and xp == pytest.approx(10.20)
    xi, xp, why = rules.walk(_frame(rows[:10]), 1, 5.0)
    assert why == "OPEN" and xi == 9


def test_walk_holds_through_missing_bar():
    rows = [(10, 10.2, 9.9, 10, 1e6)] * 3 + [(np.nan,) * 5] + [(10, 10.2, 9.9, 10, 1e6)] * 30
    xi, _, why = rules.walk(_frame(rows), 1, 5.0)
    assert why == "TIME" and xi == 20                          # eksik seans da sayılır (lab)


def test_atr14_simple_mean_and_plan_trade():
    rows = [(10, 11, 9, 10, 1e6)] * 30
    f = _frame(rows)
    assert rules.atr14(f).iloc[-1] == pytest.approx(2.0)
    plan = rules.plan_trade(f, f.index[20].date())
    assert plan["entry_price"] == 10 and plan["stop"] == pytest.approx(5.0)
    assert plan["status"] == "OPEN" and plan["bars"] == 9


# ── H5 ──────────────────────────────────────────────────────────────────────

def test_h5_rule_and_reaction_session():
    assert rules.h5_qualifies(12.0, 0.50, 0.56)
    assert not rules.h5_qualifies(9.9, 0.50, 0.55)
    assert not rules.h5_qualifies(12.0, 0.01, 0.02)            # |tahmin| < 0.02
    assert not rules.h5_qualifies(12.0, -0.50, -0.56)          # açıklanan ≤ tahmin
    assert not rules.h5_qualifies(None, 0.5, 0.6)
    # 08:00 ET → aynı gün; 16:05 ET → ertesi seans; Cuma akşamı → Pazartesi
    assert rules.h5_reaction_session("2025-11-04 08:00:00-05:00") == date(2025, 11, 4)
    assert rules.h5_reaction_session("2025-11-04 16:05:00-05:00") == date(2025, 11, 5)
    assert rules.h5_reaction_session("2025-11-07 16:05:00-05:00") == date(2025, 11, 10)
    # Tatil sabahı (Şükran Günü) → sonraki seans
    assert rules.h5_reaction_session("2025-11-27 07:00:00-05:00") == date(2025, 11, 28)


# ── H4 ──────────────────────────────────────────────────────────────────────

def _filing(acc, owners, txs, form="4", sym="ABC", d=date(2025, 5, 1)):
    return dict(form=form, accession=acc, filing_date=d, symbol=sym, owners=owners, transactions=txs)


def test_h4_purchase_rows_match_lab_semantics():
    buy = [dict(code="P", acquired="A", shares="1000", price="60")]
    f = _filing("a1", [dict(cik="0001", relationship="Director"), dict(cik="0002", relationship="Officer")], buy)
    rows = rules.h4_purchase_rows(f)
    assert [r["value_once"] for r in rows] == [60000.0, 0.0]   # ortak dosya: değer bir kez
    assert [r["owner_cik"] for r in rows] == ["1", "2"]
    assert rules.h4_purchase_rows(_filing("a2", [dict(cik="3", relationship="TenPercentOwner")], buy)) == []
    assert rules.h4_purchase_rows(_filing("a3", [dict(cik="3", relationship="Director")],
                                          [dict(code="S", acquired="D", shares="10", price="5")])) == []
    assert rules.h4_purchase_rows(_filing("a4", [dict(cik="3", relationship="Director")], buy, form="4/A")) == []


def test_h4_cluster_window_value_and_gap():
    P = pd.DataFrame([
        dict(ticker="ABC", filing_date="2025-05-01", owner_cik="1", value_once=60_000),
        dict(ticker="ABC", filing_date="2025-05-20", owner_cik="2", value_once=50_000),   # küme → olay
        dict(ticker="ABC", filing_date="2025-06-10", owner_cik="3", value_once=90_000),   # 60 gün içinde
        dict(ticker="ABC", filing_date="2025-08-15", owner_cik="4", value_once=70_000),
        dict(ticker="ABC", filing_date="2025-08-16", owner_cik="5", value_once=40_000),   # >60 g → olay
        dict(ticker="XYZ", filing_date="2025-05-01", owner_cik="1", value_once=500_000),  # tek kişi
    ])
    ev = rules.h4_events(P)
    assert [(e["ticker"], e["event_date"]) for e in ev] == [("ABC", date(2025, 5, 20)), ("ABC", date(2025, 8, 16))]
    # Önceki olay 1 Nisan: 20 Mayıs 60 gün içinde → yok; 10 Haziran penceresi (2+3) → olay.
    assert [e["event_date"] for e in rules.h4_events(P, prior_event_dates={"ABC": date(2025, 4, 1)})] ==         [date(2025, 6, 10), date(2025, 8, 16)]
    w = rules.h4_events(P, observation_start=date(2025, 5, 1))
    assert w[0]["warmup"] and not w[1]["warmup"]


# ── Rapor / kapı ────────────────────────────────────────────────────────────

def test_gate_states():
    base = dict(months_live=7, eligible_closed=200, portfolio_10pct=dict(total_pct=12, max_dd_pct=-10),
                iwm_buy_hold_pct=5.0)
    assert gate("H5", dict(base, months_live=2))["status"] == "TOPLANIYOR"
    assert gate("H5", dict(base, net_mean=0.8, net_ci95=[0.1, 1.5]))["status"] == "TERFİ ADAYI"
    assert gate("H5", dict(base, net_mean=-0.2, net_ci95=[-0.9, 0.5]))["status"] == "KALDI"
    assert gate("H5", dict(base, net_mean=0.3, net_ci95=[-0.2, 0.8]))["status"] == "BELİRSİZ"
    assert gate("H5", dict(base, months_live=12.5, net_mean=0.3, net_ci95=[-0.2, 0.8]))["status"] == "KALDI"


def test_portfolio_marks_and_costs():
    s = rules.sessions_between(date(2025, 3, 3), date(2025, 3, 7))
    t = dict(entry_date="2025-03-04", entry_price=10.0, exit_date="2025-03-06", exit_price=11.0, key=1,
             details=dict(path=[["2025-03-04", 10.5], ["2025-03-05", 9.0], ["2025-03-06", 11.0]]))
    r = portfolio([t], s, cost=0.0)
    assert r["taken"] == 1 and r["total_pct"] == pytest.approx(1.0)
    assert r["max_dd_pct"] == pytest.approx(100 * (9900 / 10050 - 1), abs=0.01)   # 10.5 → 9.0 düşüşü


def test_ledger_idempotent_and_isolated(tmp_path):
    L = ShadowLedger(str(tmp_path / "s.db"))
    row = dict(strategy="H5", ticker="ABC", event_date="2025-05-01", signal_date="2025-05-01",
               decided_at="2025-05-02T10:00:00Z", key=12.0, status="PENDING")
    assert L.add_trade(row) and not L.add_trade(row)
    assert len(L.trades()) == 1
    L.set_meta("x", {"a": 1})
    assert L.get_meta("x") == {"a": 1}
    assert L.add_purchases([dict(accession="a", owner_cik="1", ticker="ABC", filing_date=date(2025, 5, 1),
                                 value_once=1.0)]) == 1
    assert len(L.purchases(["ABC"], "2025-04-01", "2025-05-01")) == 1


def test_ledger_postgres_ddl_closes_rest_api():
    """Supabase public tabloları REST'e açar: gölge tabloları RLS açık + politikasız olmalı."""
    from unittest.mock import MagicMock, patch
    from swing_trader.shadow import ledger as L
    conn = MagicMock()
    with patch.object(L._st, "_MODE", "pg"), patch.object(L._st, "_connect", return_value=conn):
        L.ShadowLedger()
    sqls = [c.args[0] for c in conn.cursor.return_value.execute.call_args_list]
    for t in ("shadow_trades", "shadow_insider_purchases", "shadow_meta"):
        assert f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY" in sqls
    assert not any("paper_trades" in q for q in sqls), "gölge defteri ana tabloya dokunmamalı"


# ── Lab paritesi (yerel veri varsa) ─────────────────────────────────────────

needs_lab = pytest.mark.skipif(not (LAB / "challengers" / "trades_H5.parquet").exists(),
                               reason="lab verisi yok (output/lab)")


def _lab_bars(tickers):
    frames = []
    for f in sorted((LAB / "ohlcv").glob("batch_*.parquet")):
        d = pd.read_parquet(f)
        d = d[d.ticker.isin(tickers)]
        if len(d):
            frames.append(d)
    df = pd.concat(frames).dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    df["Volume"] = df["Volume"].fillna(0)
    df = df.drop_duplicates(["ticker", "Date"], keep="last").sort_values(["ticker", "Date"])
    return {t: g.drop(columns="ticker") for t, g in df.groupby("ticker")}


LAB_END = date(2026, 11, 20)
SESS = None


def _sessions():
    global SESS
    if SESS is None:
        SESS = rules.sessions_between(date(2020, 6, 1), LAB_END)
    return SESS


@needs_lab
@pytest.mark.parametrize("strategy,n", [("H5", 400), ("H4", 250)])
def test_trades_match_lab(strategy, n):
    T = pd.read_parquet(LAB / f"challengers/trades_{strategy}.parquet")
    T = T.sample(min(n, len(T)), random_state=11)
    bars = _lab_bars(set(T.ticker))
    bad = []
    for r in T.itertuples():
        frame = rules.align(bars[r.ticker], _sessions())
        plan = rules.plan_trade(frame, pd.Timestamp(r.signal_date).date())
        exp_reason = {"END_OF_DATA": "OPEN"}.get(r.reason, r.reason)
        got_day = plan.get("exit_day") or plan.get("mark_day")
        if not (plan.get("entry_day") == pd.Timestamp(r.entry_day).date()
                and abs(plan["entry_price"] - r.entry_price) < 1e-6
                and got_day == pd.Timestamp(r.exit_day).date()
                and abs(plan.get("exit_price", plan.get("mark_price")) - r.exit_price) < 1e-6
                and plan["exit_reason"] == exp_reason):
            bad.append((r.ticker, str(r.signal_date)[:10], plan.get("exit_reason"), r.reason))
    assert not bad, f"{len(bad)}/{len(T)} işlem lab'dan farklı: {bad[:5]}"


@needs_lab
def test_h5_events_match_lab():
    ev = pd.read_parquet(LAB / "challengers/events_H5.parquet")
    rng = np.random.default_rng(5)
    files = sorted((LAB / "earnings_full").glob("*.json"))
    files = [files[i] for i in rng.choice(len(files), 300, replace=False)]
    recs = [json.loads(f.read_text()) for f in files]
    tickers = {r["ticker"] for r in recs}
    bars = _lab_bars(tickers)
    sess = _sessions()
    got = []
    for rec in recs:
        t = rec["ticker"]
        if t not in bars:
            continue
        frame = rules.align(bars[t], sess)
        dates = list(frame.index.date)
        pos = {d: i for i, d in enumerate(dates)}
        for x in rec.get("rows", []):
            if not rules.h5_qualifies(x.get("surprise"), x.get("est"), x.get("rep")):
                continue
            R = rules.h5_reaction_session(x["ts"])
            i = pos.get(R)
            if i is None or i == 0:
                continue
            c = frame["Close"]
            if c.iloc[i] > c.iloc[i - 1]:
                got.append((t, R))
    exp = [(r.ticker, pd.Timestamp(r.R).date()) for r in ev[ev.ticker.isin(tickers)].itertuples()]
    assert sorted(got) == sorted(exp)


@needs_lab
def test_h4_events_match_lab():
    if not list((LAB / "sec").glob("*.zip")):
        pytest.skip("SEC DERA verisi yok")
    import sys
    sys.path.insert(0, str(ROOT))
    from lab.info_signals import insider_purchases
    p = insider_purchases()
    P = pd.DataFrame(dict(ticker=p.ticker, filing_date=p.F.dt.date.astype(str),
                          owner_cik=p.RPTOWNERCIK, value_once=p.val_once))
    got = sorted((e["ticker"], e["event_date"]) for e in rules.h4_events(P))
    lab = pd.read_parquet(LAB / "challengers/events_H4.parquet")
    exp = sorted((r.ticker, pd.Timestamp(r.F).date()) for r in lab.itertuples())
    assert got == exp
