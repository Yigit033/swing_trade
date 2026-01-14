"""
Database Integrity Audit Script
Checks for corrupted stock data by comparing DB prices vs live yfinance prices
"""

import sqlite3
import yfinance as yf
import pandas as pd
import random
import sys
import io

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def run_audit():
    # Get all tickers and their latest prices from DB
    conn = sqlite3.connect('data/stocks.db')
    query = '''
    SELECT ticker, close as db_close, date 
    FROM stock_data 
    WHERE date = (SELECT MAX(date) FROM stock_data)
    ORDER BY ticker
    '''
    db_data = pd.read_sql_query(query, conn)
    conn.close()

    print(f'Checking {len(db_data)} tickers for data corruption...')
    print('=' * 60)

    # Check ALL tickers (not just sample) - may take a while
    all_tickers = list(db_data['ticker'])
    
    # Download in batches of 100
    batch_size = 100
    corrupted = []
    checked = 0
    
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        
        try:
            live_data = yf.download(batch, period='1d', progress=False)
            
            if live_data.empty:
                continue
                
            if isinstance(live_data.columns, pd.MultiIndex):
                live_prices = live_data['Close'].iloc[-1] if len(live_data) > 0 else pd.Series()
            else:
                live_prices = live_data['Close'] if 'Close' in live_data.columns else pd.Series()
            
            for ticker in batch:
                db_row = db_data[db_data['ticker'] == ticker]
                if db_row.empty:
                    continue
                db_price = db_row['db_close'].values[0]
                
                if ticker in live_prices.index:
                    live_price = live_prices[ticker]
                    if pd.notna(live_price) and db_price > 0 and live_price > 0:
                        diff_pct = abs(db_price - live_price) / live_price * 100
                        if diff_pct > 5:  # More than 5% difference
                            corrupted.append({
                                'ticker': ticker,
                                'db_price': db_price,
                                'live_price': live_price,
                                'diff_pct': diff_pct
                            })
                checked += 1
                
        except Exception as e:
            print(f'Error in batch {i}: {e}')
            continue
        
        print(f'Checked {min(i+batch_size, len(all_tickers))}/{len(all_tickers)} tickers...')

    print('=' * 60)
    
    if corrupted:
        print(f'\n!!! CORRUPTED TICKERS FOUND: {len(corrupted)} !!!\n')
        for c in sorted(corrupted, key=lambda x: x['diff_pct'], reverse=True):
            print(f"  {c['ticker']}: DB=${c['db_price']:.2f} vs Live=${c['live_price']:.2f} ({c['diff_pct']:.1f}% diff)")
        
        # Return list of corrupted tickers
        return [c['ticker'] for c in corrupted]
    else:
        print('\n[OK] No corrupted tickers found!')
        return []

if __name__ == '__main__':
    corrupted_tickers = run_audit()
    
    if corrupted_tickers:
        print(f'\n\nTickers to re-download: {corrupted_tickers}')
