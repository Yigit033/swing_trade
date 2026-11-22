"""Gölge defteri — ana paper_trades tablosundan TAMAMEN ayrı tablolar (shadow_*).

Ana ürünün işlemleri, kapasitesi, cooldown'u ve raporları bu tabloları görmez.
PostgreSQL (DATABASE_URL) veya SQLite (yerel / test) — paper_trading.storage ile aynı mod.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from ..paper_trading import storage as _st

TRADE_COLUMNS = (
    "strategy", "ticker", "event_date", "signal_date", "decided_at", "key", "status", "reason",
    "entry_date", "entry_price", "atr", "stop", "exit_date", "exit_price", "exit_reason", "bars",
    "ret_pct", "net_pct", "iwm_ret", "mark_price", "mark_date", "late_decision", "warmup",
    "details", "updated_at",
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ShadowLedger:
    def __init__(self, db_path: Optional[str] = None):
        self._path = db_path
        self._pg = db_path is None and _st._MODE == "pg"
        self.init()

    # ── bağlantı ──
    def _conn(self):
        if self._path:
            c = sqlite3.connect(self._path)
            c.row_factory = sqlite3.Row
            return c
        return _st._connect()

    @property
    def ph(self) -> str:
        return "%s" if self._pg else "?"

    def _rows(self, cur, rows) -> List[Dict]:
        if self._pg:
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in rows]
        return [dict(r) for r in rows]

    def init(self) -> None:
        pk = "SERIAL PRIMARY KEY" if self._pg else "INTEGER PRIMARY KEY AUTOINCREMENT"
        ddl = [
            f"""CREATE TABLE IF NOT EXISTS shadow_trades (
                id {pk},
                strategy TEXT NOT NULL, ticker TEXT NOT NULL, event_date TEXT NOT NULL,
                signal_date TEXT NOT NULL, decided_at TEXT NOT NULL, key REAL,
                status TEXT NOT NULL, reason TEXT,
                entry_date TEXT, entry_price REAL, atr REAL, stop REAL,
                exit_date TEXT, exit_price REAL, exit_reason TEXT, bars INTEGER,
                ret_pct REAL, net_pct REAL, iwm_ret REAL, mark_price REAL, mark_date TEXT,
                late_decision INTEGER DEFAULT 0, warmup INTEGER DEFAULT 0,
                details TEXT, updated_at TEXT,
                UNIQUE (strategy, ticker, event_date))""",
            """CREATE TABLE IF NOT EXISTS shadow_insider_purchases (
                accession TEXT NOT NULL, owner_cik TEXT NOT NULL, ticker TEXT,
                filing_date TEXT NOT NULL, value_once REAL,
                PRIMARY KEY (accession, owner_cik))""",
            """CREATE TABLE IF NOT EXISTS shadow_meta (
                key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)""",
            "CREATE INDEX IF NOT EXISTS idx_shadow_trades_status ON shadow_trades(status)",
            "CREATE INDEX IF NOT EXISTS idx_shadow_purch_ticker ON shadow_insider_purchases(ticker, filing_date)",
        ]
        if self._pg:
            # Supabase public şemasını REST API'ye açar; backend-only tablolar RLS açık +
            # politika yok (scripts/supabase_rls_fix.sql düzeni). Backend superuser → bypass.
            ddl += [f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY"
                    for t in ("shadow_trades", "shadow_insider_purchases", "shadow_meta")]
        conn = self._conn()
        try:
            cur = conn.cursor()
            for q in ddl:
                cur.execute(q)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _exec(self, sql: str, args: Iterable = (), fetch: bool = False):
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(args))
            out = self._rows(cur, cur.fetchall()) if fetch else cur.rowcount
            conn.commit()
            return out
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── meta ──
    def get_meta(self, key: str, default=None):
        rows = self._exec(f"SELECT value FROM shadow_meta WHERE key = {self.ph}", (key,), fetch=True)
        if not rows or rows[0]["value"] is None:
            return default
        try:
            return json.loads(rows[0]["value"])
        except ValueError:
            return rows[0]["value"]

    def set_meta(self, key: str, value) -> None:
        p = self.ph
        self._exec(f"INSERT INTO shadow_meta (key, value, updated_at) VALUES ({p}, {p}, {p}) "
                   f"ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at",
                   (key, json.dumps(value, default=str), _now()))

    # ── işlemler ──
    def add_trade(self, row: Dict) -> bool:
        """Yeni gölge kaydı; (strategy, ticker, event_date) zaten varsa dokunmaz (idempotent)."""
        row = dict(row)
        row.setdefault("updated_at", _now())
        if isinstance(row.get("details"), (dict, list)):
            row["details"] = json.dumps(row["details"], default=str)
        cols = [c for c in TRADE_COLUMNS if c in row]
        p = self.ph
        n = self._exec(f"INSERT INTO shadow_trades ({', '.join(cols)}) VALUES ({', '.join([p] * len(cols))}) "
                       f"ON CONFLICT (strategy, ticker, event_date) DO NOTHING",
                       [row[c] for c in cols])
        return bool(n)

    def update_trade(self, trade_id: int, updates: Dict) -> None:
        updates = dict(updates, updated_at=_now())
        if isinstance(updates.get("details"), (dict, list)):
            updates["details"] = json.dumps(updates["details"], default=str)
        cols = [c for c in TRADE_COLUMNS if c in updates]
        p = self.ph
        self._exec(f"UPDATE shadow_trades SET {', '.join(f'{c} = {p}' for c in cols)} WHERE id = {p}",
                   [updates[c] for c in cols] + [trade_id])

    def trades(self, strategy: Optional[str] = None, statuses: Optional[Iterable[str]] = None) -> List[Dict]:
        q, args = "SELECT * FROM shadow_trades WHERE 1=1", []
        if strategy:
            q += f" AND strategy = {self.ph}"
            args.append(strategy)
        if statuses:
            st = list(statuses)
            q += f" AND status IN ({', '.join([self.ph] * len(st))})"
            args += st
        rows = self._exec(q + " ORDER BY signal_date, key DESC, id", args, fetch=True)
        for r in rows:
            if r.get("details"):
                try:
                    r["details"] = json.loads(r["details"])
                except ValueError:
                    pass
        return rows

    def last_event_dates(self, strategy: str) -> Dict[str, str]:
        rows = self._exec(f"SELECT ticker, MAX(event_date) AS d FROM shadow_trades WHERE strategy = {self.ph} "
                          f"GROUP BY ticker", (strategy,), fetch=True)
        return {r["ticker"]: r["d"] for r in rows}

    # ── H4 alım deposu ──
    def add_purchases(self, rows: List[Dict]) -> int:
        """Tek bağlantı/işlemde toplu ekleme (satır başına bağlantı Supabase'de yavaş)."""
        if not rows:
            return 0
        p, n = self.ph, 0
        sql = (f"INSERT INTO shadow_insider_purchases (accession, owner_cik, ticker, filing_date, value_once) "
               f"VALUES ({p}, {p}, {p}, {p}, {p}) ON CONFLICT (accession, owner_cik) DO NOTHING")
        conn = self._conn()
        try:
            cur = conn.cursor()
            for r in rows:
                cur.execute(sql, (r["accession"], r["owner_cik"], r["ticker"], str(r["filing_date"]),
                                  float(r["value_once"])))
                n += max(cur.rowcount, 0)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return n

    def purchases(self, tickers: Iterable[str], since: str, until: str) -> List[Dict]:
        tickers = sorted(set(tickers))
        if not tickers:
            return []
        p = self.ph
        return self._exec(
            f"SELECT accession, owner_cik, ticker, filing_date, value_once FROM shadow_insider_purchases "
            f"WHERE ticker IN ({', '.join([p] * len(tickers))}) AND filing_date > {p} AND filing_date <= {p}",
            tickers + [since, until], fetch=True)
