"""Test SmallCap scan with finvizfinance"""
from swing_trader.small_cap import SmallCapEngine
from swing_trader.data.fetcher import DataFetcher

engine = SmallCapEngine()
fetcher = DataFetcher('yfinance')

# Get REAL small-cap tickers from Finviz
print("Fetching small-cap universe from Finviz...")
tickers = engine.get_small_cap_universe(max_tickers=50)  # Limit for testing
print(f"Got {len(tickers)} tickers")
print(f"First 10: {tickers[:10]}")

# Fetch data for these tickers
print("\nFetching price data...")
data_dict = {}
for i, ticker in enumerate(tickers[:30]):  # Limit for speed
    try:
        df = fetcher.fetch_stock_data(ticker, period='3mo')
        if df is not None and len(df) >= 20:
            data_dict[ticker] = df
            print(f"  {i+1}. {ticker}: {len(df)} rows")
    except Exception as e:
        print(f"  {ticker}: Error - {e}")

print(f"\nFetched data for {len(data_dict)} tickers")

# Scan
print("\nScanning for signals...")
signals = engine.scan_universe(list(data_dict.keys()), data_dict, 10000)

print(f"\n=== RESULTS ===")
print(f"Signals found: {len(signals)}")

for i, s in enumerate(signals[:10]):
    print(f"  {i+1}. {s['ticker']}: Q={s['quality_score']:.0f}, Entry=${s['entry_price']:.2f}, MCap=${s.get('market_cap_millions', 0):.0f}M")
