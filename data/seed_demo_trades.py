"""
seed_demo_trades.py — Demo (Sentetik) Trade Ekleyici

Bu script sadece model eğitmek için gerekli minimum veriye ulaşmak amacıyla
gerçekçi ama SAHTE trade kayıtları ekler.

NOT: Bunlar gerçek trade değildir. İstersen sonradan:
     DELETE FROM paper_trades WHERE notes LIKE '%[DEMO]%'
     komutuyla silebilirsin.

Çalıştırmak için:
    python data/seed_demo_trades.py
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import random

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "paper_trades.db"

# ─────────────────────────────────────────────────
# Sentetik trade tanımları
# Gerçekçi çeşitlilik: farklı tipler, farklı sonuçlar
# ─────────────────────────────────────────────────
DEMO_TRADES = [
    # (ticker, entry, stop, target, atr, quality, swing_type, max_hold, exit_price, exit_status, days)
    # WIN'ler — TARGET'a ulaşanlar
    ("NVDA",  440.0, 420.0, 480.0, 8.5,  8.5, "A", 7,  479.5, "TARGET",  5),
    ("SMCI",   82.0,  77.5,  96.0, 2.1,  9.0, "B", 8,   96.2, "TARGET",  6),
    ("MSTR",  320.0, 300.0, 370.0, 12.0, 7.5, "A", 7,  368.0, "TARGET",  4),
    ("CLSK",   12.5,  11.5,  15.0, 0.45, 8.0, "C", 10,  14.9, "TARGET",  7),
    ("APP",    72.0,  67.0,  83.0, 2.8,  8.8, "B", 7,   83.1, "TARGET",  5),
    ("LUNR",    7.2,   6.5,   9.0, 0.35, 7.0, "S", 8,    8.8, "TRAILED", 6),
    ("HOOD",   17.5,  16.0,  21.0, 0.65, 7.5, "A", 7,   20.8, "TARGET",  4),

    # LOSS'lar — STOPPED çıkışlar
    ("PLUG",    3.8,   3.4,   4.8, 0.18, 5.5, "C", 7,    3.3, "STOPPED", 2),
    ("NKLA",    1.2,   1.05,  1.6, 0.08, 5.0, "S", 5,    1.0, "STOPPED", 1),
    ("MVST",    4.5,   4.1,   5.6, 0.22, 6.0, "B", 8,    4.0, "STOPPED", 3),

    # TIMEOUT — süre doldu
    ("CLOV",    2.8,   2.5,   3.5, 0.12, 6.5, "A", 7,    2.75,"TIMEOUT", 7),
]

def seed_demo_trades():
    """Demo trade'leri SQLite'a ekle."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    today = datetime.now()
    added = 0

    for (ticker, entry, stop, target, atr, quality, swing_type,
         max_hold, exit_price, status, days) in DEMO_TRADES:

        # Giriş ve çıkış tarihleri (geçmişte)
        entry_date = (today - timedelta(days=days + random.randint(5, 60))).strftime("%Y-%m-%d")
        exit_date  = (datetime.strptime(entry_date, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")

        # Gerçekleşen P/L hesabı
        realized_pnl_pct = round((exit_price / entry - 1) * 100, 2)
        realized_pnl     = round((exit_price - entry) * 100, 2)  # 100 hisse varsayımı

        cursor.execute("""
            INSERT INTO paper_trades
              (ticker, entry_date, entry_price, stop_loss, target,
               swing_type, quality_score, position_size, max_hold_days,
               status, exit_date, exit_price, realized_pnl, realized_pnl_pct,
               atr, initial_stop, trailing_stop, notes,
               created_at, updated_at)
            VALUES (?,?,?,?,?, ?,?,?,?, ?,?,?,?,?, ?,?,?,?, ?,?)
        """, (
            ticker, entry_date, entry, stop, target,
            swing_type, quality, 100, max_hold,
            status, exit_date, exit_price, realized_pnl, realized_pnl_pct,
            atr, stop, stop, f"[DEMO] Sentetik eğitim verisi",
            today.isoformat(), today.isoformat()
        ))
        added += 1
        print(f"  ✅ {ticker} | {status} | {realized_pnl_pct:+.2f}%")

    conn.commit()
    conn.close()
    print(f"\nToplam {added} demo trade eklendi.")
    print("Silmek için: DELETE FROM paper_trades WHERE notes LIKE '%[DEMO]%'")


if __name__ == "__main__":
    print("Demo trade'ler ekleniyor...\n")
    seed_demo_trades()
