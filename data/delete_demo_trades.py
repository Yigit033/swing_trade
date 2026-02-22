import sqlite3
from pathlib import Path

db = Path(r'c:\swing_trade\data\paper_trades.db')
conn = sqlite3.connect(str(db))
cur = conn.cursor()

cur.execute("SELECT id, ticker, status FROM paper_trades WHERE notes LIKE '%[DEMO]%'")
rows = cur.fetchall()
print(f"Silinecek demo trade sayisi: {len(rows)}")
for r in rows:
    print(f"  id={r[0]} {r[1]} {r[2]}")

cur.execute("DELETE FROM paper_trades WHERE notes LIKE '%[DEMO]%'")
conn.commit()
conn.close()
print("Demo tradeler silindi!")
