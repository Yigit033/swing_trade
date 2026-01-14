"""Test Swing Type Classification"""
from swing_trader.small_cap import SmallCapEngine
from swing_trader.data.fetcher import DataFetcher

engine = SmallCapEngine()
fetcher = DataFetcher('yfinance')

# Get universe
tickers = engine.get_small_cap_universe(use_finviz=False)
print(f"Testing with {len(tickers)} tickers")

# Fetch data
data_dict = {}
for ticker in tickers[:50]:
    df = fetcher.fetch_stock_data(ticker, period='3mo')
    if df is not None and len(df) >= 21:
        data_dict[ticker] = df

print(f"Fetched data for {len(data_dict)} tickers\n")

# Scan
signals = engine.scan_universe(list(data_dict.keys()), data_dict, 10000)

print(f"\n{'='*70}")
print(f"SWING TRADE SIGNALS: {len(signals)}")
print(f"{'='*70}\n")

type_a = [s for s in signals if s['swing_type'] == 'A']
type_b = [s for s in signals if s['swing_type'] == 'B']

print(f"🐢 TYPE A (Continuation, 5-7 days): {len(type_a)}")
for s in type_a:
    print(f"   {s['ticker']}: Q={s['quality_score']:.0f} | 5d:{s['five_day_return']:+.0f}% | RSI:{s['rsi']:.0f}")
    print(f"      └─ {s['type_reason']}")

print(f"\n🚀 TYPE B (Momentum, 1-3 days): {len(type_b)}")
for s in type_b:
    print(f"   {s['ticker']}: Q={s['quality_score']:.0f} | 5d:{s['five_day_return']:+.0f}% | RSI:{s['rsi']:.0f}")
    print(f"      └─ {s['type_reason']}")
