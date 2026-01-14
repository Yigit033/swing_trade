"""Test swing confirmation system"""
from swing_trader.small_cap import SmallCapEngine
from swing_trader.data.fetcher import DataFetcher

engine = SmallCapEngine()
fetcher = DataFetcher('yfinance')

# Test with a few stocks
test_tickers = ['GLUE', 'VTYX', 'AAPL', 'TSLA', 'PLUG']
print('Testing swing confirmation system...')
print('=' * 70)

for ticker in test_tickers:
    df = fetcher.fetch_stock_data(ticker, period='3mo')
    if df is not None and len(df) >= 21:
        swing_ready, details = engine.signals.check_swing_confirmation(df)
        five_d = details['five_day_momentum']
        ma20 = details['above_ma20']
        ext = details['overextension']
        rsi = details['rsi']
        
        print(f"{ticker}:")
        print(f"  Swing Ready: {swing_ready}")
        print(f"  5-Day Return: {five_d['return']:+.1f}% (passed={five_d['passed']})")
        print(f"  Above MA20: {ma20['passed']} (distance={ma20['distance']:+.1f}%)")
        print(f"  RSI: {rsi:.0f}")
        print(f"  Safe to enter: {ext['safe']}")
        if ext.get('details'):
            d = ext['details']
            print(f"    Today: {d.get('today_change', 0):+.1f}%")
            print(f"    Max single day: {d.get('max_single_day', 0):.1f}%")
            print(f"    5-day total: {d.get('five_day_total', 0):+.1f}%")
        print()
