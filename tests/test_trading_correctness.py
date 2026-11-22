"""Offline correctness regressions; no provider or production database access."""
from datetime import date
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import json
import sqlite3
from datetime import datetime

from swing_trader.paper_trading.tracker import PaperTradeTracker
from swing_trader.paper_trading.execution import evaluate, mark_pnl
from swing_trader.paper_trading.storage import PaperTradeStorage, _CREATE_TABLE_SQLITE, _migrate_execution_columns


def trade(**changes):
    row = dict(id=1, ticker="TEST", status="OPEN", entry_date="2026-10-01T13:30:00Z",
               entry_price=100., initial_stop=90., stop_loss=90., target=110.,
               target_2=125., position_size=100, atr=0., max_hold_days=20)
    row.update(changes)
    return row


def bars(values):
    return pd.DataFrame([dict(Date=date.fromisoformat(d), Open=o, High=h, Low=l, Close=c)
                         for d, o, h, l, c in values])


def test_entry_day_stop_not_skipped():
    tape = bars([("2026-10-01", 100, 101, 85, 95)])
    result = PaperTradeTracker(Mock()).check_exit_conditions(trade(), tape)
    assert result[:2] == ("STOPPED", 90.)


def test_t1_cannot_retroactively_stop_before_fill():
    # Existing partial without its event time is not safe to replay.
    tape = bars([("2026-10-01", 100, 104, 97, 102),
                 ("2026-10-02", 104, 112, 103, 111)])
    result = PaperTradeTracker(Mock()).check_exit_conditions(
        trade(initial_stop=100., stop_loss=100., target=125.,
              partial_exit_price=110., partial_exit_pct=33.), tape)
    assert result[0] == "OPEN"
    assert "REVIEW_REQUIRED" in result[3]


def test_chronological_t1_then_be_preserves_risk():
    tape = bars([('2026-10-01', 100, 104, 97, 102),
                 ('2026-10-02', 104, 112, 103, 111),
                 ('2026-10-05', 105, 108, 99, 100)])
    t = trade(initial_risk=1000.)
    first, outcome = evaluate(t, tape.iloc[:2], .33)
    assert outcome[0] == 'OPEN'
    assert first['partial_exit_date'][:10] == '2026-10-02'
    assert 'initial_stop' not in first and 'initial_risk' not in first
    final, outcome = evaluate(dict(t, **first), tape, .99)  # executed ratio is frozen
    assert outcome[:3] == ('TRAILED', 100., '2026-10-05T20:00:00Z')
    assert final['realized_pnl'] == 330.
    assert final['r_multiple'] == .33
    state = json.loads(final['execution_state'])
    assert [e['kind'] for e in state['events']] == ['T1', 'TRAILED']
    assert [e['shares'] for e in state['events']] == [33, 67]


@pytest.mark.parametrize('opening,low,expected', [(100,85,90), (85,80,85)])
def test_stop_first_and_gap_down(opening, low, expected):
    updates, result = evaluate(trade(), bars([('2026-10-01',opening,130,low,110)]), .33)
    assert result[:2] == ('STOPPED', expected)
    assert 'partial_exit_price' not in updates


def test_gap_up_target_conservative_fill_and_integer_shares():
    tape = bars([('2026-10-01',100,105,99,104), ('2026-10-02',130,132,128,129)])
    updates, result = evaluate(trade(position_size=101, initial_risk=1010), tape, .33)
    assert result[:2] == ('TARGET',125.)
    assert updates['realized_pnl'] == 33*10+68*25
    assert updates['r_multiple'] == pytest.approx(2030/1010)


def test_ambiguous_t1_be_t2_same_bar_is_pessimistic():
    u, r = evaluate(trade(initial_risk=1000), bars([('2026-10-01',100,130,99,120)]), .33)
    assert r[:2] == ('TRAILED',100.)
    assert u['realized_pnl'] == 330
    assert 'Ambiguous' in r[3]


def test_t1_and_timeout_same_bar_accounts_for_remainder():
    u, r = evaluate(trade(max_hold_days=2), bars([
        ('2026-10-01',100,104,99,103), ('2026-10-02',104,112,103,111)]), .33)
    assert r[:2] == ('TIMEOUT',111.)
    assert u['realized_pnl'] == 33*10+67*11


def test_trailing_activates_only_after_bar_stop_check():
    t = trade(atr=2.,target=150.,target_2=170.)
    u,r = evaluate(t,bars([('2026-10-01',100,110,99,108)]),.33)
    assert r[0] == 'OPEN' and u['trailing_stop'] == 104.
    u2,r2 = evaluate(dict(t,**u),bars([
        ('2026-10-01',100,110,99,108),('2026-10-02',108,109,103,104)]),.33)
    assert r2[:2] == ('TRAILED',104.)


@pytest.mark.parametrize('entry', ['2026-10-01', '2026-10-01T18:00:00Z'])
def test_unknown_or_intraday_entry_does_not_invent_daily_fills(entry):
    u,r = evaluate(trade(entry_date=entry), bars([('2026-10-01',100,130,80,110)]), .33)
    assert not u and 'REVIEW_REQUIRED' in r[3]


def test_missing_or_revised_bar_is_not_silently_replayed():
    tape = bars([('2026-10-01',100,105,99,104), ('2026-10-02',104,108,102,105)])
    t=trade();u,_=evaluate(t,tape,.33)
    for bad in [tape.iloc[1:], tape.assign(High=[106,108])]:
        updates,r=evaluate(dict(t,**u),bad,.33)
        assert not updates and 'REVIEW_REQUIRED' in r[3]


def test_storage_checkpoint_atomic_idempotent_and_immutable_risk(tmp_path):
    store=PaperTradeStorage(str(tmp_path/'trades.db'))
    identity=store.add_trade(trade())
    original=store.get_trade_by_id(identity)
    assert original['initial_risk'] == 1000.
    tape=bars([('2026-10-01',100,105,99,104),('2026-10-02',104,112,103,111)])
    updates,_=evaluate(original,tape,.33)
    assert store.apply_execution_updates(original,updates)
    assert not store.apply_execution_updates(original,updates)  # racing stale writer
    fresh=store.get_trade_by_id(identity)
    again,_=evaluate(fresh,tape,.33)
    assert json.loads(again['execution_state'])['events'] == json.loads(updates['execution_state'])['events']
    assert fresh['initial_stop']==90. and fresh['initial_risk']==1000.
    assert not store.update_trade(identity,{'initial_stop':100.})
    assert store.close_trade(identity,105.,'2026-10-05T20:00:00Z','CLOSED')
    closed=store.get_trade_by_id(identity)
    assert closed['realized_pnl']==665. and closed['r_multiple']==.665
    assert not store.close_trade(identity,999.,'2026-10-06','CLOSED')
    assert store.get_trade_by_id(identity)==closed


def test_tracker_processes_t1_and_final_exit_in_one_update_and_no_double_sale(tmp_path):
    store=PaperTradeStorage(str(tmp_path/'trades.db'));identity=store.add_trade(trade())
    tracker=PaperTradeTracker(store);original=store.get_trade_by_id(identity)
    tape=bars([('2026-10-01',100,105,99,104),('2026-10-02',104,112,103,111),
               ('2026-10-05',105,126,104,125)])
    with patch.object(tracker,'fetch_price_history',return_value=tape), patch(
        'swing_trader.paper_trading.tracker._t1_partial_fraction',return_value=.33):
        result=tracker.update_trade_status(original)
        assert result['status']=='TARGET' and result['realized_pnl']==2005.
        repeated=tracker.update_trade_status(original)
        assert repeated['realized_pnl']==2005.
        assert len(json.loads(repeated['execution_state'])['events'])==2


def test_additive_sqlite_migration_twice_never_backfills_legacy(tmp_path):
    path=str(tmp_path/'legacy.db');conn=sqlite3.connect(path);conn.execute(_CREATE_TABLE_SQLITE)
    conn.execute("INSERT INTO paper_trades(ticker,entry_date,entry_price,stop_loss,target,initial_stop,status,realized_pnl) VALUES ('OLD','2026-05-19',100,100,125,100,'TARGET',330)")
    before=conn.execute('SELECT * FROM paper_trades').fetchall();oldcols=[r[1] for r in conn.execute('PRAGMA table_info(paper_trades)')];conn.commit();conn.close()
    PaperTradeStorage(path);PaperTradeStorage(path)
    conn=sqlite3.connect(path)
    assert conn.execute('SELECT '+','.join(oldcols)+' FROM paper_trades').fetchall()==before
    assert conn.execute('SELECT initial_risk,r_multiple,partial_exit_date,execution_state FROM paper_trades').fetchone()==(None,)*4
    conn.close()


def test_postgres_migration_is_additive_and_fails_closed():
    cursor=Mock();_migrate_execution_columns(cursor,True)
    from swing_trader.paper_trading.storage import EXECUTION_COLUMNS
    assert len(cursor.execute.call_args_list)==len(EXECUTION_COLUMNS)==6   # +strategy (2026-11-22)
    assert all(c.args[0].startswith('ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS') for c in cursor.execute.call_args_list)
    cursor.execute.side_effect=RuntimeError('migration failure')
    with pytest.raises(RuntimeError):_migrate_execution_columns(cursor,True)


def backtester(tape):
    from swing_trader.small_cap.smallcap_backtest import SmallCapBacktester
    bt=SmallCapBacktester();bt._reset_state()
    bt._data={'TEST':tape};bt._end=pd.Timestamp('2026-12-31')
    bt.pending_signals=[dict(signal=dict(ticker='TEST',swing_type='A',quality_score=85,
        hold_days_max=20,position_size=10,stop_loss=92.,target_1=110.,target_2=125.),
        signal_close=100.,signal_date=pd.Timestamp('2026-10-01'),atr=4.)]
    return bt


def test_next_open_atr_stop_targets_sizing_invariant_to_future_ohlcv():
    dates=pd.bdate_range(end='2026-10-02',periods=80)
    tape=pd.DataFrame(dict(Date=dates,Open=100.,High=102.,Low=98.,Close=100.,Volume=1000000.))
    original=backtester(tape);original._confirm_pending(pd.Timestamp('2026-10-02'))
    changed=tape.copy();changed.loc[79,['High','Low','Close','Volume']]=[1000.,1.,500.,999999999.]
    modified=backtester(changed);modified._confirm_pending(pd.Timestamp('2026-10-02'))
    keys=('entry_price','size','stop','target','target_2','initial_risk')
    assert len(original.open_trades)==len(modified.open_trades)==1
    assert [original.open_trades[0][k] for k in keys]==[modified.open_trades[0][k] for k in keys]
    # Canlı onay matematiği: açılış − 2.5×ATR(4) = 90, tip A tavanı %15 → 85'in üstünde kalır.
    assert original.open_trades[0]['stop']==90.


def test_no_stale_open_fill_when_entry_bar_missing():
    tape=pd.DataFrame(dict(Date=pd.bdate_range(end='2026-10-01',periods=80),Open=100.,High=102.,Low=98.,Close=100.,Volume=1000000.))
    bt=backtester(tape);bt._confirm_pending(pd.Timestamp('2026-10-02'))
    assert bt.open_trades==[]
    assert len(bt.pending_signals)==1, "canlı tracker bar gelene kadar bekler (14 gün sonra reddeder)"


def test_partial_mark_to_market_uses_remaining_shares():
    assert mark_pnl(trade(partial_exit_price=110.,partial_exit_pct=33.),105.) == (665.,6.65)


@pytest.mark.parametrize('gap,expected', [(104.,'OPEN'), (106.,'REJECTED'),(92.,'REJECTED')])
def test_confirmation_gap_filters_and_initial_risk(tmp_path,gap,expected):
    store=PaperTradeStorage(str(tmp_path/'pending.db'))
    tid=store.add_trade(trade(status='PENDING',entry_date='2026-10-01 16:30',signal_price=100.))
    assert store.get_trade_by_id(tid)['initial_risk'] is None
    tracker=PaperTradeTracker(store)
    tape=bars([('2026-10-01',100,102,99,100),('2026-10-02',gap,gap+2,gap-2,gap)])
    with patch.object(tracker,'fetch_price_history',return_value=tape),patch(
        'swing_trader.paper_trading.tracker._gap_limits',return_value=(5.,7.)):
        tracker.confirm_pending_trades()
    row=store.get_trade_by_id(tid)
    assert row['status']==expected
    if expected=='OPEN':
        assert row['initial_risk']==pytest.approx((row['entry_price']-row['initial_stop'])*100)
        assert row['entry_date']=='2026-10-02T13:30:00Z'


def test_no_incomplete_entry_bar_is_executed(tmp_path):
    store=PaperTradeStorage(str(tmp_path/'incomplete.db'));tid=store.add_trade(trade())
    tracker=PaperTradeTracker(store)
    tape=bars([('2026-10-01',100,130,85,100)])
    with patch.object(tracker,'fetch_price_history',return_value=tape),patch(
        'swing_trader.utils.market_calendar.last_completed_session',return_value=date(2026,9,30)):
        result=tracker.update_trade_status(store.get_trade_by_id(tid))
    assert result['status']=='OPEN' and result.get('execution_state') is None
    assert 'ENTRY_BAR_MISSING' in result['execution_warning']


def test_policy_edit_intraday_cannot_use_pre_edit_low():
    t=trade(policy_effective_at='2026-10-01T18:00:00Z',stop_loss=98.)
    u,r=evaluate(t,bars([('2026-10-01',100,110,97,105)]),.33)
    assert not u and 'INTRADAY_POLICY_EDIT' in r[3]


def test_backtest_follows_live_daily_order(monkeypatch):
    """Canlı günlük bakım sırası: PENDING onayı → çıkış → (kapanış) rejim + tarama."""
    from swing_trader.small_cap.smallcap_backtest import SmallCapBacktester
    bt=SmallCapBacktester()
    tape=pd.DataFrame(dict(Date=pd.bdate_range(end='2026-10-02',periods=80),Open=100.,High=101.,
                           Low=99.,Close=100.,Volume=10000.))
    def load(*a):
        bt._data={'TEST':tape}
    monkeypatch.setattr(bt,'_load_data',load)
    monkeypatch.setattr(bt,'_fetch_benchmarks',lambda *a:(None,None))
    calls=[]
    monkeypatch.setattr(bt,'_confirm_pending',lambda day:calls.append(('confirm',day)))
    monkeypatch.setattr(bt,'_settle_exits',lambda day:calls.append(('exit',day)))
    def regime(day):
        calls.append(('regime',day))
        return {'regime':'BULL','confidence':'CONFIRMED'},78,10
    monkeypatch.setattr(bt,'_daily_regime_bundle',regime)
    monkeypatch.setattr(bt,'_scan_and_track',lambda day,*a:calls.append(('scan',day)))
    bt.run_backtest(['TEST'],'2026-10-01','2026-10-02')
    day=pd.Timestamp('2026-10-02')
    order=[c for c in calls if c[1]==day]
    assert order==[('confirm',day),('exit',day),('regime',day),('scan',day)]


def test_backtest_end_uses_no_future_bar_or_stop_floor():
    from swing_trader.small_cap.smallcap_backtest import SmallCapBacktester
    bt=SmallCapBacktester()
    tape=bars([('2026-10-01',100,101,99,100),('2026-10-02',80,85,75,80),
               ('2026-10-05',1000,1001,999,1000)])
    tape.Date=pd.to_datetime(tape.Date)
    bt._data={'TEST':tape}
    bt.open_trades=[dict(ticker='TEST',swing_type='A',pathway='VCE',quality_score=80,rank_score=80,
        entry_day=pd.Timestamp('2026-10-01'),entry_price=100.,size=10,stop=90.,target=110.,
        target_2=125.,max_hold_days=20,initial_risk=100.,status='OPEN',reason='',exit_day=None,
        exit_price=0.,bars=2,sold=0,partial_price=0.,partial_day=None,partial_realized=False)]
    bt._force_close_all(pd.Timestamp('2026-10-02'))
    assert bt.trades[0]['exit_price']==bt._exit_fill_price(80.)
    assert bt.trades[0]['status']=='FORCED'


def test_full_partial_fraction_sells_no_extra_remaining_shares():
    u,r=evaluate(trade(),bars([('2026-10-01',100,112,99,110)]),1.)
    assert r[0]=='TARGET' and u['realized_pnl']==1000.
    assert sum(e['shares'] for e in json.loads(u['execution_state'])['events'])==100


def test_single_share_t1_is_full_exit_not_zero_share_sale():
    u,r=evaluate(trade(position_size=1),bars([('2026-10-01',100,112,99,110)]),.33)
    assert r[0]=='TARGET' and u['realized_pnl']==10.
    assert json.loads(u['execution_state'])['sold']==1


def test_backtest_exit_is_live_execution_evaluate():
    """Backtest çıkışı canlı execution.evaluate'in sonucunu birebir taşır (kısmi T1 + BE aynı bar)."""
    from swing_trader.small_cap.smallcap_backtest import SmallCapBacktester
    bt=SmallCapBacktester()
    bt.settings=bt.settings.model_copy(deep=True)
    bt.settings.slippage_bps_per_side=0  # isolate accounting, not a production edit
    bt.settings.commission_usd_per_side=0.0  # çıkış paritesi; ücret ayrı testte
    tape=bars([('2026-10-01',100,130,99,120)]);tape.Date=pd.to_datetime(tape.Date)
    bt._data={'TEST':tape};bt._end=pd.Timestamp('2026-10-01')
    sig=dict(ticker='TEST',swing_type='A',quality_score=85,rank_score=85,hold_days_max=20)
    bt._open_trade(sig,pd.Timestamp('2026-10-01'),100.,100,90.,110.,125.,0.)
    live,(st,px,_,_,_)=evaluate(trade(stop_loss=90.,initial_stop=90.,target=110.,target_2=125.,
                                     initial_risk=1000.,atr=0.),bars([('2026-10-01',100,130,99,120)]),
                                bt.settings.partial_at_t1_fraction)
    bt._settle_exits(pd.Timestamp('2026-10-01'))
    assert len(bt.trades)==1 and bt.open_trades==[]
    assert bt.trades[0]['status']==st=='TRAILED'
    assert bt.trades[0]['pnl_dollar']==live['realized_pnl']==330.
    assert bt.capital==10330.


def test_backtest_regime_uses_one_year_spy_history(monkeypatch):
    """Canlı rejim SPY'nin 1 yıllık geçmişini kullanır (MA200). Backtest SPY'yi yalnız fiyat
    ısınmasıyla (7 ay) indirince dönem başında rejim canlıdan farklı çıkıyordu (ASST 2026-08-24)."""
    from swing_trader.small_cap.smallcap_backtest import SmallCapBacktester
    bt=SmallCapBacktester()
    monkeypatch.setattr(bt,'_load_data',lambda *a:setattr(bt,'_data',{'T':pd.DataFrame(
        dict(Date=pd.bdate_range(end='2026-10-02',periods=80),Open=1.,High=1.,Low=1.,Close=1.,Volume=1.))}))
    seen={}
    def bench(start,end):
        seen['start']=start
        return None,None
    monkeypatch.setattr(bt,'_fetch_benchmarks',bench)
    monkeypatch.setattr(bt,'_scan_and_track',lambda *a:None)
    bt.run_backtest(['T'],'2026-10-01','2026-10-02')
    assert (pd.Timestamp('2026-10-01')-pd.Timestamp(seen['start'])).days>=365


def test_backtest_respects_cash_account_like_live(monkeypatch):
    """Canlı auto-scan ile aynı: açık+bekleyen+yeni tutar ≤ hesap (kaldıraçsız nakit hesap)."""
    from swing_trader.small_cap.smallcap_backtest import SmallCapBacktester
    bt=SmallCapBacktester();bt._reset_state();bt.initial_capital=10000
    tape=pd.DataFrame(dict(Date=pd.bdate_range(end='2026-10-02',periods=80),Open=50.,High=51.,
                           Low=49.,Close=50.,Volume=1e6))
    bt._data={'AAA':tape,'BBB':tape,'CCC':tape};bt._earnings={t:[pd.Timestamp('2026-08-05')] for t in bt._data}
    bt.open_trades=[dict(ticker='OLD',entry_price=20.,size=150,sold=0,partial_realized=False)]   # 3.000$
    sizes={'AAA':100,'BBB':80,'CCC':20}          # 5.000$ / 4.000$ / 1.000$ @ 50$
    monkeypatch.setattr(bt.engine,'scan_stock',lambda t,w,**k:dict(ticker=t,quality_score=90,
        rank_score={'AAA':3,'BBB':2,'CCC':1}[t],position_size=sizes[t]))
    bt._scan_and_track(pd.Timestamp('2026-10-02'),{'regime':'BULL','confidence':'CONFIRMED'},60,10)
    assert [p['signal']['ticker'] for p in bt.pending_signals]==['AAA','CCC']
    assert bt._diag['skip_cash']==1


# ── H6S "Profesyonel VCE" canlı paper stratejisi (2026-11-22) ────────────────

def _h6s_days(n, start="2026-10-01"):
    from swing_trader.utils.market_calendar import next_trading_day
    d, out = date.fromisoformat(start), []
    while len(out) < n:
        out.append(d.isoformat())
        d = next_trading_day(d)
    return out


def test_h6s_exit_is_fixed_hold_without_targets_or_trailing():
    days = _h6s_days(10)
    tape = bars([(d, 100 + 5 * i, 106 + 5 * i, 99 + 5 * i, 105 + 5 * i) for i, d in enumerate(days)])
    t = trade(strategy='H6S', target=0., target_2=0., max_hold_days=8, atr=4.)
    upd, (status, xp, xd, reason, stop) = evaluate(t, tape, .33)
    assert status == 'TIMEOUT' and xp == 140. and xd.startswith(days[7])          # 8. seans kapanışı
    assert stop == 90. and 'partial_exit_price' not in upd                        # trail/kısmi yok
    legacy = evaluate(trade(atr=4.), tape, .33)[1]
    assert legacy[0] != 'TIMEOUT'                                                  # eski kurallar değişmedi


def test_h6s_stop_first_with_gap_fill():
    days = _h6s_days(3)
    tape = bars([(days[0], 100, 101, 95, 96), (days[1], 85, 88, 84, 86), (days[2], 90, 91, 89, 90)])
    _, (status, xp, *_r) = evaluate(trade(strategy='H6S', target=0., target_2=0., max_hold_days=8, atr=4.), tape, .33)
    assert (status, xp) == ('STOPPED', 85.)                                        # gap'te Open'dan


def test_h6s_confirm_has_no_gap_filter_and_atr_stop(tmp_path):
    store = PaperTradeStorage(str(tmp_path / 'h6s.db'))
    tracker = PaperTradeTracker(store)
    with patch('swing_trader.utils.market_calendar.entry_window_open', return_value=True):
        tid = tracker.add_trade_from_signal(dict(ticker='HHH', entry_price=50., date='2026-10-01',
                                                 position_size=20, atr14=2., strategy='H6S', regime='BULL'))
    row = store.get_trade_by_id(tid)
    assert (row['status'], row['strategy'], row['max_hold_days'], row['target']) == ('PENDING', 'H6S', 8, 0.)
    tape = bars([('2026-10-01', 50, 51, 49, 50), ('2026-10-02', 55, 56, 54, 55)])   # +%10 gap
    with patch.object(tracker, 'fetch_price_history', return_value=tape):
        tracker.confirm_pending_trades()
    row = store.get_trade_by_id(tid)
    assert row['status'] == 'OPEN' and row['entry_price'] == 55. and row['stop_loss'] == 50.   # 55 − 2.5×2


def test_strategies_have_separate_duplicates_and_cash(tmp_path):
    store = PaperTradeStorage(str(tmp_path / 's.db'))
    base = dict(entry_date='2026-10-01', stop_loss=9., target=12., swing_type='A', status='OPEN',
                entry_price=10., position_size=100)
    store.add_trade(dict(base, ticker='AAA', strategy='H6S'))
    store.add_trade(dict(base, ticker='BBB'))                                       # eski satır (NULL)
    assert store.check_duplicate('AAA', strategy='H6S') and not store.check_duplicate('AAA', strategy='LEGACY')
    assert store.check_duplicate('BBB', strategy='LEGACY') and not store.check_duplicate('BBB', strategy='H6S')
    assert store.active_notional_for_owner(None, strategy='H6S') == 1000.
    assert store.count_active_trades_for_owner(None, strategy='LEGACY') == 1
