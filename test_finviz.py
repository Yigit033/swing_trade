"""Test finvizfinance to find correct filter names"""
from finvizfinance.screener.overview import Overview

# First let's see what filters are available
foverview = Overview()

# Get available filter options
try:
    # Try to get filter options
    print("Testing finvizfinance...")
    
    # Use simple filter first
    filters_dict = {
        'Market Cap.': '+Small (over $300mln)',  # + prefix for positive
        'Average Volume': 'Over 1M'
    }
    
    foverview.set_filter(filters_dict=filters_dict)
    df = foverview.screener_view()
    
    print(f"Found {len(df)} stocks")
    print(df['Ticker'].head(20).tolist())
    
except Exception as e:
    print(f"Error: {e}")
    
    # Try without market cap filter
    print("\nTrying without market cap filter...")
    foverview2 = Overview()
    filters_dict2 = {
        'Average Volume': 'Over 1M',
        'Price': 'Under $50'
    }
    foverview2.set_filter(filters_dict=filters_dict2)
    df2 = foverview2.screener_view()
    print(f"Found {len(df2)} stocks")
    if len(df2) > 0:
        print(df2.head())
