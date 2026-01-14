"""Debug SmallCap filter failures"""
from swing_trader.small_cap import SmallCapEngine
from swing_trader.data.fetcher import DataFetcher

engine = SmallCapEngine()
fetcher = DataFetcher('yfinance')

tickers = ['BEAM', 'NTLA', 'SRPT', 'PLUG', 'ASTS']
print('Debugging filter failures...')
print('=' * 60)

for t in tickers:
    df = fetcher.fetch_stock_data(t, period='3mo')
    if df is None or len(df) < 21:
        print(f'{t}: No data')
        continue
    
    info = engine.get_stock_info(t)
    float_shares = info.get('floatShares', 0)
    float_m = float_shares / 1e6 if float_shares else 0
    mcap = info.get('marketCap', 0) / 1e6
    
    # Check volume surge
    vol_surge = engine.signals.calculate_volume_surge(df)
    atr_pct = engine.filters.calculate_atr_percent(df) * 100
    
    print(f'{t}: Float={float_m:.0f}M | MCap=${mcap:.0f}M | Vol={vol_surge:.1f}x | ATR={atr_pct:.1f}%')
    
    # Current filter limits
    max_float = 75  # New threshold
    min_vol = 1.5   # New threshold
    
    issues = []
    if float_m > max_float:
        issues.append(f'Float {float_m:.0f}M > {max_float}M')
    if vol_surge < min_vol:
        issues.append(f'VolSurge {vol_surge:.1f}x < {min_vol}x')
    if atr_pct < 3.5:
        issues.append(f'ATR {atr_pct:.1f}% < 3.5%')
    
    if issues:
        print(f'   BLOCKED: {", ".join(issues)}')
    else:
        print(f'   OK - passes filters')
    print()
