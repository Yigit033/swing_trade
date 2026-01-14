"""Quick SmallCap scan test"""
from swing_trader.small_cap import SmallCapEngine
from swing_trader.data.fetcher import DataFetcher

engine = SmallCapEngine()
fetcher = DataFetcher('yfinance')

# Get tickers
tickers = engine.get_small_cap_universe()[:20]  # Test with first 20
print(f"Testing with {len(tickers)} tickers...")

# Fetch data
data_dict = {}
for ticker in tickers:
    try:
        df = fetcher.fetch_stock_data(ticker, period='3mo')
        if df is not None and len(df) >= 20:
            data_dict[ticker] = df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")

print(f"Fetched data for {len(data_dict)} tickers")

# Scan
signals = engine.scan_universe(tickers, data_dict, 10000)

print(f"\n=== RESULTS ===")
print(f"Signals found: {len(signals)}")

for s in signals[:5]:
    print(f"  {s['ticker']}: Q={s['quality_score']:.0f}, Entry=${s['entry_price']:.2f}, Stop=${s['stop_loss']:.2f}")
