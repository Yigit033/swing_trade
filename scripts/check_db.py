import sqlite3
conn = sqlite3.connect('data/stocks.db')
cursor = conn.cursor()

# stock_data - ana veri tablosu
print("="*70)
print("STOCK DATA TABLE")
print("="*70)
cursor.execute('SELECT COUNT(*) FROM stock_data')
print(f"Total rows: {cursor.fetchone()[0]}")

cursor.execute('SELECT COUNT(DISTINCT ticker) FROM stock_data')
print(f"Unique tickers: {cursor.fetchone()[0]}")

cursor.execute('SELECT MIN(date), MAX(date) FROM stock_data')
dates = cursor.fetchone()
print(f"Date range: {dates[0]} to {dates[1]}")

cursor.execute('SELECT ticker, COUNT(*) as cnt FROM stock_data GROUP BY ticker ORDER BY cnt DESC LIMIT 20')
print(f"\nTop 20 tickers by data count:")
for row in cursor.fetchall():
    print(f"  {row[0]:6s}: {row[1]} bars")

# trading_signals
print("\n" + "="*70)
print("TRADING SIGNALS TABLE")
print("="*70)
cursor.execute('SELECT * FROM sqlite_master WHERE type="table" AND name="trading_signals"')
if cursor.fetchone():
    cursor.execute('SELECT COUNT(*) FROM trading_signals')
    print(f"Total signals: {cursor.fetchone()[0]}")
    cursor.execute('SELECT * FROM trading_signals ORDER BY created_at DESC LIMIT 5')
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    print(f"Columns: {cols}")
    for row in rows:
        print(f"  {row}")
else:
    print("Table not found")

# stock_info icindeki small-cap bilgileri
print("\n" + "="*70)
print("STOCK INFO TABLE")
print("="*70)
cursor.execute('SELECT * FROM sqlite_master WHERE type="table" AND name="stock_info"')
if cursor.fetchone():
    cursor.execute('SELECT COUNT(*) FROM stock_info')
    print(f"Total entries: {cursor.fetchone()[0]}")
    cursor.execute('SELECT * FROM stock_info LIMIT 3')
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    print(f"Columns: {cols}")
    for row in rows:
        print(f"  {row}")

# Small-cap related tickers
print("\n" + "="*70)
print("SMALL-CAP TICKERS IN DATABASE")
print("="*70)
small_caps = ['RKLB', 'LUNR', 'ASTS', 'SOUN', 'BBAI', 'IONQ', 'OKLO', 'SMR', 
              'NNE', 'AVXL', 'HIMS', 'APLD', 'CAVA', 'TOST', 'QS', 'BLNK',
              'JOBY', 'AEHR', 'ACHR', 'CRDO', 'RAMP']
for ticker in small_caps:
    cursor.execute('SELECT COUNT(*), MIN(date), MAX(date) FROM stock_data WHERE ticker=?', (ticker,))
    row = cursor.fetchone()
    if row[0] > 0:
        print(f"  {ticker:5s}: {row[0]:4d} bars | {row[1]} to {row[2]}")
    else:
        print(f"  {ticker:5s}: NOT FOUND")

conn.close()
