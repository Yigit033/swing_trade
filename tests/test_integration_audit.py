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
