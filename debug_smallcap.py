"""Debug script for SmallCap signal generation"""

from swing_trader.small_cap import SmallCapEngine
from swing_trader.data.fetcher import DataFetcher

engine = SmallCapEngine()
fetcher = DataFetcher('yfinance')

# Test with HIMS (known volatile stock)
df = fetcher.fetch_stock_data('HIMS', period='3mo')
print(f'HIMS data: {len(df)} rows')

# Check filters
info = engine.get_stock_info('HIMS')
print(f"Market Cap: ${info.get('marketCap', 0)/1e6:.0f}M")
print(f"Float: {info.get('floatShares', 0)/1e6:.0f}M")

# Check trigger values
vol_surge = engine.signals.calculate_volume_surge(df)
atr_pct = engine.signals.calculate_atr_percent(df)
print(f'Volume Surge: {vol_surge:.2f}x (need >= 2.0x)')
print(f'ATR%: {atr_pct*100:.1f}% (need >= 6%)')

# Check breakout
passed, reason = engine.signals.check_breakout(df)
print(f'Breakout: {passed} - {reason}')

# Check all triggers
triggered, details = engine.signals.check_all_triggers(df)
print(f'\nAll Triggers Passed: {triggered}')
for trigger, info in details.get('triggers', {}).items():
    print(f"  {trigger}: {info.get('passed')} - {info.get('reason')}")
