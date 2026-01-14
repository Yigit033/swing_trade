"""Full SmallCap Swing scan test"""
from swing_trader.small_cap import SmallCapEngine
from swing_trader.data.fetcher import DataFetcher

engine = SmallCapEngine()
fetcher = DataFetcher('yfinance')

# Get universe (use static for speed)
tickers = engine.get_small_cap_universe(use_finviz=False)
print(f"Testing with {len(tickers)} tickers from static list")

# Fetch data
data_dict = {}
for ticker in tickers[:50]:
    df = fetcher.fetch_stock_data(ticker, period='3mo')
    if df is not None and len(df) >= 21:
        data_dict[ticker] = df

print(f"Fetched data for {len(data_dict)} tickers")

# Scan
signals = engine.scan_universe(list(data_dict.keys()), data_dict, 10000)

print(f"\n{'='*70}")
print(f"SWING TRADE SIGNALS: {len(signals)}")
print(f"{'='*70}")

for i, s in enumerate(signals[:10]):
    print(f"{i+1}. {s['ticker']}")
    print(f"   Q={s['quality_score']:.0f} | 5d={s['five_day_return']:+.0f}% | RSI={s['rsi']:.0f}")
    print(f"   Entry=${s['entry_price']:.2f} | MCap=${s['market_cap_millions']:.0f}M")
    print()
