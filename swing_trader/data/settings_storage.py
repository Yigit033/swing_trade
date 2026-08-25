"""
Kalıcı ayar depolaması (DB destekli overlay).

NEDEN VAR (2026-08-03 canlı bug):
fly.toml'da mount YOK → container diski geçici. `data/smallcap_settings.json`
imaja git'ten kopyalanıyor; UI'dan yapılan ayar değişikliği o dosyaya yazılıyor
ve **her deploy'da siliniyor**. Kullanıcı auto-scan'i UI'dan açtı, çalıştı, sonra
bir deploy geldi ve ayar sessizce `enabled=False` varsayılanına döndü — hiçbir
hata mesajı olmadan. Bu sadece auto_scan'i değil TÜM UI ayarlarını etkiliyordu.

Çözüm: ayar YAMASI (tam ayar değil, yalnız kullanıcının değiştirdiği alanlar)
Postgres'te tutulur ve dosya katmanının ÜSTÜNE bindirilir:

    varsayılanlar (kod)  →  data/smallcap_settings.json (git)  →  DB yaması (UI)

Yalnız yamayı saklamak bilinçli: kod varsayılanları veya git JSON'u ileride
değişirse (ör. ölçümle yükselttiğimiz eşikler), kullanıcının dokunmadığı alanlar
yeni değeri otomatik alır. Tam anlık görüntü saklasaydık, eski değerler DB'de
donup kalıcı olarak yeni kalibrasyonu ezerdi — sessiz gerileme.

DATABASE_URL yoksa (yerel geliştirme) DB katmanı devre dışıdır; dosya tek
kaynaktır ve davranış eskisi gibi kalır.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass

_SETTINGS_KEY = "smallcap"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    patch      TEXT NOT NULL,
    updated_at TEXT
)
"""


def _database_url() -> Optional[str]:
    """Her çağrıda tazeden oku — testler monkeypatch edebilsin."""
    url = os.environ.get("DATABASE_URL")
    if url and url.startswith(("postgresql", "postgres")):
        return url
    return None


def is_enabled() -> bool:
    """DB destekli kalıcılık aktif mi (yerelde genelde değil)."""
    return _database_url() is not None


def _connect():
    url = _database_url()
    if not url:
        return None
    from swing_trader.utils.pg_connect import connect as pg_connect

    return pg_connect(url)


def load_patch() -> Dict[str, Any]:
    """
    DB'de saklı kullanıcı yamasını döndür. DB yok/erişilemiyor/boşsa {} döner.

    ASLA fırlatmaz: ayar okuması ürünü kilitlememeli — DB'ye ulaşılamazsa
    dosya katmanıyla (git değerleri) çalışmaya devam ederiz.
    """
    conn = None
    try:
        conn = _connect()
        if conn is None:
            return {}
        cur = conn.cursor()
        cur.execute(_CREATE_SQL)
        conn.commit()
        cur.execute("SELECT patch FROM app_settings WHERE key = %s", (_SETTINGS_KEY,))
        row = cur.fetchone()
        if not row or not row[0]:
            return {}
        patch = json.loads(row[0])
        return patch if isinstance(patch, dict) else {}
    except Exception as e:
        logger.warning("Settings DB overlay okunamadı (%s) — dosya katmanı kullanılıyor: %s",
                       e.__class__.__name__, e)
        return {}
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def save_patch(patch: Dict[str, Any]) -> bool:
    """
    Kullanıcı yamasını DB'ye yaz (upsert). Başarılıysa True.

    DB yoksa False döner — çağıran taraf bunu "yalnız dosyaya yazıldı, deploy'da
    kaybolabilir" olarak yorumlamalı ve loglamalı.
    """
    if not isinstance(patch, dict):
        raise TypeError("patch must be a dict")
    conn = None
    try:
        conn = _connect()
        if conn is None:
            return False
        cur = conn.cursor()
        cur.execute(_CREATE_SQL)
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        cur.execute(
            """
            INSERT INTO app_settings (key, patch, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (key) DO UPDATE
                SET patch = EXCLUDED.patch, updated_at = EXCLUDED.updated_at
            """,
            (_SETTINGS_KEY, json.dumps(patch, ensure_ascii=False), now),
        )
        conn.commit()
        logger.info("Ayar yaması DB'ye kaydedildi (%d üst-seviye alan)", len(patch))
        return True
    except Exception as e:
        logger.error("Ayar yaması DB'ye yazılamadı (%s): %s", e.__class__.__name__, e)
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def clear_patch() -> bool:
    """Kullanıcı yamasını sil (reset → git/kod varsayılanlarına dön)."""
    conn = None
    try:
        conn = _connect()
        if conn is None:
            return False
        cur = conn.cursor()
        cur.execute(_CREATE_SQL)
        cur.execute("DELETE FROM app_settings WHERE key = %s", (_SETTINGS_KEY,))
        conn.commit()
        logger.info("Ayar yaması DB'den silindi (reset)")
        return True
    except Exception as e:
        logger.error("Ayar yaması silinemedi (%s): %s", e.__class__.__name__, e)
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
