"""Offline API contract checks: no production DB, provider calls or scheduler."""
from unittest.mock import Mock
import ast
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.auth import get_current_user_id
from api.routers import lookup, performance, trades


@pytest.mark.parametrize('route', ['trades', 'performance'])
def test_missing_quote_never_uses_another_ticker(monkeypatch, route):
    import yfinance
    frame = pd.DataFrame({('Close', 'AAA'): [110.0]})
    monkeypatch.setattr(yfinance, 'download', lambda *a, **kw: frame)
    trades._PRICE_CACHE.clear()
    rows = [dict(ticker=t, status='OPEN', entry_price=100, position_size=10)
            for t in ['AAA', 'MISSING']]
    enrich = trades._enrich_open_trades_inline if route == 'trades' else performance._enrich_open_trades
    result = enrich(rows)
    assert result[0]['current_price'] == 110
    assert result[0]['unrealized_pnl'] == 100
    assert 'current_price' not in result[1]
    assert 'unrealized_pnl' not in result[1]
    trades._PRICE_CACHE.clear()


def client_for(router, prefix):
    app = FastAPI()
    app.include_router(router, prefix=prefix)
    app.dependency_overrides[get_current_user_id] = lambda: 'test-owner'
    return TestClient(app)


def test_trades_and_performance_share_partial_marked_pnl(monkeypatch):
    import yfinance
    monkeypatch.setattr(yfinance, 'download', lambda *a, **kw:
                        pd.DataFrame({('Close', 'AAA'): [105.0]}))
    trades._PRICE_CACHE.clear()
    row = dict(id=1, ticker='AAA', status='OPEN', entry_price=100,
               position_size=100, partial_exit_price=110, partial_exit_pct=33,
               initial_stop=90, stop_loss=100, initial_risk=1000, r_multiple=None)
    storage = Mock()
    storage.get_all_trades.return_value = [row]
    storage.get_closed_trades.return_value = []
    monkeypatch.setattr(trades, 'get_paper_storage', lambda: storage)
    monkeypatch.setattr(performance, 'get_paper_storage', lambda: storage)
    a = client_for(trades.router, '/api/trades').get('/api/trades').json()['trades'][0]
    b = client_for(performance.router, '/api/performance').get('/api/performance').json()['open_trades'][0]
    for output in (a, b):
        assert output['unrealized_pnl'] == 665
        assert output['unrealized_pnl_pct'] == 6.65
        assert output['initial_stop'] == 90 and output['stop_loss'] == 100
        assert output['initial_risk'] == 1000 and output['r_multiple'] is None
    trades._PRICE_CACHE.clear()


def test_performance_reports_net_of_trading_cost(monkeypatch):
    """Paper fiyatları maliyetsiz; ekrandaki para maliyet sonrası olmalı (tek kaynak: ayar)."""
    from swing_trader.small_cap.settings_config import SmallCapSettings
    settings = SmallCapSettings().model_copy(update={'slippage_bps_per_side': 25})   # 0.50 gidiş-dönüş
    monkeypatch.setattr('swing_trader.small_cap.settings_config.load_settings', lambda *a, **k: settings)
    closed = [dict(id=1, ticker='WIN', status='TIMEOUT', entry_price=100, position_size=10,
                   realized_pnl=3.0, realized_pnl_pct=0.30, exit_date='2026-10-02'),
              dict(id=2, ticker='BIG', status='TARGET', entry_price=50, position_size=20,
                   realized_pnl=100.0, realized_pnl_pct=10.0, exit_date='2026-10-03'),
              dict(id=3, ticker='REJ', status='REJECTED', entry_price=10, position_size=1,
                   realized_pnl=0.0, realized_pnl_pct=0.0, exit_date='2026-10-04')]
    storage = Mock()
    storage.get_all_trades.return_value = closed
    storage.get_closed_trades.return_value = closed
    monkeypatch.setattr(performance, 'get_paper_storage', lambda: storage)
    body = client_for(performance.router, '/api/performance').get('/api/performance').json()
    s = body['summary']
    assert s['cost_round_trip_pct'] == 0.5
    assert s['win_rate'] == 100.0 and s['total_pnl_pct'] == 10.3          # brüt değişmedi
    assert s['net_total_pnl_pct'] == pytest.approx(9.3)                    # 0.30−0.50 + 10.0−0.50
    assert (s['net_wins'], s['net_losses'], s['net_win_rate']) == (1, 1, 50.0)
    assert s['net_total_pnl'] == pytest.approx(3.0 - 5.0 + 100.0 - 5.0)    # tutar × %0.50
    rows = {t['ticker']: t for t in body['recent_closed']}
    assert rows['WIN']['net_pnl_pct'] == pytest.approx(-0.2) and 'net_pnl' not in rows['REJ']


def test_execution_warning_survives_update_and_patch_contract(monkeypatch):
    row = dict(id=1, ticker='AAA', status='OPEN', execution_warning='REVIEW_REQUIRED: INTRADAY_ENTRY')
    tracker, storage = Mock(), Mock()
    tracker.update_all_open_trades.return_value = [row]
    tracker.update_trade_status.return_value = row
    storage.get_trade_by_id.return_value = row
    storage.update_trade.return_value = True
    monkeypatch.setattr(trades, 'get_paper_tracker', lambda: tracker)
    monkeypatch.setattr(trades, 'get_paper_storage', lambda: storage)
    client = client_for(trades.router, '/api/trades')
    refreshed = client.post('/api/trades/update-prices')
    edited = client.patch('/api/trades/1', json={'stop_loss': 101})
    assert refreshed.status_code == edited.status_code == 200
    assert refreshed.json()['trades'][0]['execution_warning'] == row['execution_warning']
    assert edited.json()['trade']['execution_warning'] == row['execution_warning']


def test_lookup_exposes_existing_risk_atr_without_recomputing(monkeypatch):
    engine = Mock()
    engine.filters.apply_all_filters.return_value = (True, {})
    engine.signals.describe_setup.return_value = {}
    engine.signals.check_all_triggers.return_value = (True, {})
    engine.signals.check_boosters.return_value = {'swing_ready': True}
    engine.scoring.calculate_quality_score.return_value = 81
    engine._classify_swing_type.return_value = ('A', (2, 15), 'test')
    engine.risk.add_risk_management.return_value = dict(
        stop_loss=90, target_1=110, target_2=120, atr=4.25, position_size=22)
    frame = pd.DataFrame({'High': [102], 'Low': [98], 'Close': [100]})
    result = lookup._analyze_smallcap_ticker('AAA', frame, {}, engine, 25000)
    assert result['atr'] == 4.25 and result['position_size'] == 22
    assert result['hold_days'] == (2, 15)
    assert engine.risk.add_risk_management.call_args.args[2] == 25000


def test_production_modules_do_not_import_discovery_research():
    root = Path(__file__).resolve().parents[1]
    for folder in ('api', 'swing_trader/small_cap', 'swing_trader/data', 'swing_trader/paper_trading'):
        for source in (root / folder).rglob('*.py'):
            if 'tests' in source.parts:
                continue
            tree = ast.parse(source.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                names = [n.name for n in node.names] if isinstance(node, ast.Import) else (
                    [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                assert not any('discovery' in name.split('.') for name in names), source


def test_performance_filters_by_strategy(monkeypatch):
    """H6S ana strateji ve eski kurallar (NULL dahil) ayrı hesaplar olarak raporlanır."""
    rows = [dict(id=1, ticker='NEW', status='TIMEOUT', strategy='H6S', entry_price=10, position_size=100,
                 realized_pnl=50.0, realized_pnl_pct=5.0, exit_date='2026-10-02'),
            dict(id=2, ticker='OLD', status='STOPPED', strategy=None, entry_price=10, position_size=100,
                 realized_pnl=-30.0, realized_pnl_pct=-3.0, exit_date='2026-10-03')]
    storage = Mock()
    storage.get_all_trades.return_value = rows
    storage.get_closed_trades.return_value = rows
    monkeypatch.setattr(performance, 'get_paper_storage', lambda: storage)
    c = client_for(performance.router, '/api/performance')
    assert c.get('/api/performance', params={'strategy': 'H6S'}).json()['summary']['closed_trades'] == 1
    legacy = c.get('/api/performance', params={'strategy': 'LEGACY'}).json()
    assert [t['ticker'] for t in legacy['recent_closed']] == ['OLD']
    assert c.get('/api/performance').json()['summary']['closed_trades'] == 2


@pytest.mark.parametrize('method,path', [('put', '/api/settings'), ('post', '/api/settings/reset'),
                                         ('post', '/api/genai/chat'), ('post', '/api/genai/train'),
                                         ('post', '/api/backtest/smallcap'), ('post', '/api/lookup')])
def test_write_and_costly_endpoints_require_login(monkeypatch, method, path):
    """2026-11-22: bu uçlar girişsiz çağrılabiliyordu (canlı ayarları değiştirme / LLM kredisi)."""
    import api.auth as auth
    from api.routers import settings as settings_router, genai, backtest, lookup
    monkeypatch.setattr(auth, 'SUPABASE_JWT_SECRET', 'configured')       # production: auth açık
    app = FastAPI()                                                          # api/main.py ile aynı önekler
    app.include_router(settings_router.router)
    app.include_router(genai.router, prefix='/api/genai')
    app.include_router(backtest.router, prefix='/api/backtest')
    app.include_router(lookup.router, prefix='/api/lookup')
    c = TestClient(app)
    assert getattr(c, method)(path, json={}).status_code == 401
    assert c.get('/api/settings').status_code == 200                       # okuma açık kalır


def test_admin_allowlist_blocks_other_accounts(monkeypatch):
    import api.auth as auth
    from api.routers import settings as settings_router
    monkeypatch.setattr(auth, 'ADMIN_EMAILS', {'owner@example.com'})
    app = FastAPI()
    app.include_router(settings_router.router)
    app.dependency_overrides[auth.verify_token] = lambda: {'sub': 'u2', 'email': 'stranger@example.com'}
    assert TestClient(app).put('/api/settings', json={}).status_code == 403


def test_midas_orders_buy_sell_hold():
    from datetime import date as _d
    from api.routers.orders import build_orders
    rows = [dict(id=1, ticker='NEW', status='PENDING', strategy='H6S', entry_date='2026-11-20 16:40',
                 signal_price=20.0, entry_price=20.0, position_size=50, atr=1.0),
            dict(id=2, ticker='EXIT', status='OPEN', strategy='H6S', entry_date='2026-11-13T14:30:00Z',
                 entry_price=30.0, position_size=33, stop_loss=27.0, atr=1.2),
            dict(id=3, ticker='KEEP', status='OPEN', strategy='H6S', entry_date='2026-11-20T14:30:00Z',
                 entry_price=10.0, position_size=100, stop_loss=9.0, atr=0.4),
            dict(id=4, ticker='OLD', status='OPEN', strategy=None, entry_date='2026-11-20T14:30:00Z',
                 entry_price=5.0, position_size=10, stop_loss=4.0),
            dict(id=5, ticker='TODAY', status='OPEN', strategy='H6S', entry_date='2026-11-12T14:30:00Z',
                 entry_price=40.0, position_size=25, stop_loss=36.0, atr=1.6)]
    o = build_orders(rows, _d(2026, 11, 23), 10000.0)                     # Pazartesi
    assert o['session'] == '2026-11-23'
    assert o['buy'] == [dict(ticker='NEW', shares=50, est_price=20.0, est_cost=1000.0, stop_distance=2.5,
                             est_stop=17.5, buy_day='2026-11-23', sell_day='2026-12-03', overdue=False, trade_id=1)]
    # 12 Kas girişi: 12,13,16,17,18,19,20,23 → 8. seans bugün → kapanışta SAT
    assert [(r['ticker'], r['sell_day']) for r in o['sell']] == [('TODAY', '2026-11-23')]
    # 13 Kas girişi → 8. seans 24 Kas (TUT); 20 Kas girişi → 2 Ara (TUT; 26 Kas Şükran Günü tatil)
    assert [(r['ticker'], r['sell_day']) for r in o['hold']] == [('EXIT', '2026-11-24'), ('KEEP', '2026-12-02')]
    assert 'OLD' not in {r['ticker'] for k in ('buy', 'sell', 'hold') for r in o[k]}
    assert o['held'] == 1000 + 990 + 1000 + 1000 and o['free'] == 10000 - 3990
