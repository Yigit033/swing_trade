"""
Mevcut SQLite veritabanındaki tüm paper trade'leri Supabase PostgreSQL'e taşır.

Kullanım:
    set DATABASE_URL=postgresql://postgres:SIFRE@db.PROJE_ID.supabase.co:5432/postgres
    python migrate_to_postgres.py
"""

import os
import sqlite3
import sys
from pathlib import Path

SQLITE_PATH = Path(__file__).parent / "data" / "paper_trades.db"


def migrate():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("HATA: DATABASE_URL env variable ayarlanmamış.")
        print("Örnek: set DATABASE_URL=postgresql://postgres:SIFRE@db.ID.supabase.co:5432/postgres")
        sys.exit(1)

    if not SQLITE_PATH.exists():
        print(f"HATA: SQLite DB bulunamadı: {SQLITE_PATH}")
        sys.exit(1)

    import psycopg2

    # SQLite'tan oku
    sqlite_conn = sqlite3.connect(str(SQLITE_PATH))
    sqlite_conn.row_factory = sqlite3.Row
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT * FROM paper_trades ORDER BY id")
    trades = [dict(row) for row in cursor.fetchall()]
    sqlite_conn.close()

    print(f"SQLite'tan {len(trades)} trade okundu.")

    if not trades:
        print("Taşınacak trade yok.")
        return

    # PostgreSQL'e bağlan
    pg_conn = psycopg2.connect(database_url)
    pg_cursor = pg_conn.cursor()

    # Tablo oluştur (yoksa)
    pg_cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id              SERIAL PRIMARY KEY,
            ticker          TEXT NOT NULL,
            entry_date      TEXT NOT NULL,
            entry_price     REAL NOT NULL,
            stop_loss       REAL NOT NULL,
            target          REAL NOT NULL,
            swing_type      TEXT,
            quality_score   REAL,
            position_size   INTEGER DEFAULT 100,
            max_hold_days   INTEGER DEFAULT 7,
            status          TEXT DEFAULT 'OPEN',
            exit_date       TEXT,
            exit_price      REAL,
            realized_pnl    REAL,
            realized_pnl_pct REAL,
            notes           TEXT,
            trailing_stop   REAL,
            initial_stop    REAL,
            atr             REAL,
            signal_price    REAL,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    inserted = 0
    skipped = 0

    for t in trades:
        try:
            pg_cursor.execute("""
                INSERT INTO paper_trades
                (ticker, entry_date, entry_price, stop_loss, target,
                 swing_type, quality_score, position_size, max_hold_days,
                 status, exit_date, exit_price, realized_pnl, realized_pnl_pct,
                 notes, trailing_stop, initial_stop, atr, signal_price,
                 created_at, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT DO NOTHING
            """, (
                t['ticker'], t['entry_date'], t['entry_price'],
                t['stop_loss'], t['target'],
                t.get('swing_type', 'A'), t.get('quality_score', 0),
                t.get('position_size', 100), t.get('max_hold_days', 7),
                t.get('status', 'OPEN'),
                t.get('exit_date'), t.get('exit_price'),
                t.get('realized_pnl'), t.get('realized_pnl_pct'),
                t.get('notes', ''),
                t.get('trailing_stop'), t.get('initial_stop'),
                t.get('atr'), t.get('signal_price'),
                t.get('created_at'), t.get('updated_at'),
            ))
            inserted += 1
            print(f"  ✅ {t['ticker']} ({t['status']}) — ID {t['id']}")
        except Exception as e:
            skipped += 1
            print(f"  ⚠️  {t['ticker']} atlandı: {e}")

    pg_conn.commit()
    pg_conn.close()
    print(f"\n✅ Migrasyon tamamlandı: {inserted} eklendi, {skipped} atlandı.")


if __name__ == "__main__":
    migrate()
