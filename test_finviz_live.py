"""Test optimized system with LIVE Finviz data"""
from swing_trader.small_cap import SmallCapEngine
from swing_trader.data.fetcher import DataFetcher

engine = SmallCapEngine()
fetcher = DataFetcher('yfinance')

print('Testing with LIVE Finviz Data')
print('=' * 70)

# Get Finviz universe (real small-caps)
tickers = engine.get_small_cap_universe(use_finviz=True, max_tickers=50)
print(f'Got {len(tickers)} tickers from Finviz')
print(f'Top 10: {tickers[:10]}')

# Fetch data for top 30
data_dict = {}
for t in tickers[:30]:
    df = fetcher.fetch_stock_data(t, period='3mo')
    if df is not None and len(df) >= 21:
        data_dict[t] = df

print(f'Fetched data for {len(data_dict)} tickers')

# Scan
signals = engine.scan_universe(list(data_dict.keys()), data_dict, 10000)
print(f'\nSignals found: {len(signals)}')
print('=' * 70)

type_a = [s for s in signals if s['swing_type'] == 'A']
type_b = [s for s in signals if s['swing_type'] == 'B']
type_c = [s for s in signals if s['swing_type'] == 'C']

print(f"Type C (Early Stage, 2-4d): {len(type_c)}")
print(f"Type A (Continuation, 4-8d): {len(type_a)}")
print(f"Type B (Momentum, 2-5d): {len(type_b)}")
print()

for s in signals[:5]:
    t = s['swing_type']
    label = {'A': 'Cont', 'B': 'Mom', 'C': 'Early'}.get(t, t)
    print(f"{s['ticker']}: Type {t} ({label}) | {s['hold_days_min']}-{s['hold_days_max']}d | Q={s['quality_score']:.0f}")
    print(f"   5d={s['five_day_return']:+.0f}% | RSI={s['rsi']:.0f} | Vol={s['volume_surge']:.1f}x")
