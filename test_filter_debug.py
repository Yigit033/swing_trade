"""Detailed debug of each filter stage"""
from swing_trader.small_cap import SmallCapEngine
from swing_trader.data.fetcher import DataFetcher
import pandas as pd

engine = SmallCapEngine()
fetcher = DataFetcher('yfinance')

# Use known working tickers
tickers = ['IONQ', 'SMCI', 'CRDO', 'SOUN', 'JOBY']
print('Detailed Filter Debug')
print('=' * 70)

for t in tickers:
    print(f'\n{t}:')
    
    df = fetcher.fetch_stock_data(t, period='3mo')
    if df is None or len(df) < 21:
        print('  NO DATA')
        continue
    
    info = engine.get_stock_info(t)
    
    # 1. Check filters
    from datetime import datetime
    filter_ok, filter_res = engine.filters.apply_all_filters(t, df, info, datetime.now())
    print(f'  Filters: {"PASS" if filter_ok else "FAIL"}')
    if not filter_ok:
        print(f'    {filter_res}')
        continue
    
    # 2. Check triggers
    trig_ok, trig_res = engine.signals.check_all_triggers(df)
    vol_surge = trig_res.get('volume_surge', 0)
    atr_pct = trig_res.get('atr_percent', 0)
    print(f'  Triggers: {"PASS" if trig_ok else "FAIL"} | Vol={vol_surge:.1f}x ATR={atr_pct*100:.1f}%')
    if not trig_ok:
        continue
    
    # 3. Check swing confirmation
    boosters = engine.signals.check_boosters(df)
    swing_ready = boosters.get('swing_ready', False)
    swing_det = boosters.get('swing_details', {})
    fiveD = swing_det.get('five_day_momentum', {})
    ma20 = swing_det.get('above_ma20', {})
    print(f'  Swing Ready: {"PASS" if swing_ready else "FAIL"} | 5d={fiveD.get("return",0):+.0f}% | MA20={ma20.get("passed")}')
    
    if swing_ready:
        print(f'  --> SIGNAL CANDIDATE!')
