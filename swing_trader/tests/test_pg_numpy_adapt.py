"""numpy skalerleri Postgres parametresi olarak güvenli mi (2026-11-24 gölge arızası).

numpy 2 + psycopg2: ``np.float64`` float'ın alt sınıfı → psycopg2 float adaptörü ``repr`` ile
``np.float64(1.5)`` yazıyordu → Postgres ``schema "np" does not exist``. İlk gölge
güncellemesi (açık işlemler) bu yüzden düştü. Testler SQLite kullandığı için görünmedi; burada
psycopg2'nin gerçek alıntılama yolu (``adapt``) sunucusuz çalıştırılır.
"""
from datetime import datetime

import numpy as np
import pytest

psycopg2 = pytest.importorskip("psycopg2")
from psycopg2.extensions import ISQLQuote, adapt, adapters  # noqa: E402

from swing_trader.shadow import runner  # noqa: E402
from swing_trader.shadow.ledger import ShadowLedger  # noqa: E402
from swing_trader.tests.test_shadow_runner import NY, FakeSrc  # noqa: E402
from swing_trader.utils import pg_connect  # noqa: E402

NUMPY2 = int(np.__version__.split(".")[0]) >= 2
KEY = (np.generic, ISQLQuote)


@pytest.fixture
def bare_psycopg2():
    """Adaptör kayıtsız başla; testten sonra önceki duruma dön (kayıt süreç geneli)."""
    saved = adapters.pop(KEY, None)
    yield
    adapters.pop(KEY, None)
    if saved is not None:
        adapters[KEY] = saved


def _quote(v) -> bytes:
    try:
        return adapt(v).getquoted()
    except psycopg2.ProgrammingError:          # np.int64 vb.: "can't adapt type"
        return b"<cannot adapt>"


@pytest.mark.skipif(not NUMPY2, reason="repr hatası numpy 2'ye özgü")
def test_bug_reproduces_without_adapter(bare_psycopg2):
    assert _quote(round(np.float64(1.23456), 4)) == b"np.float64(1.2346)"
    assert _quote(np.int64(7)) == b"<cannot adapt>"


def test_connect_registers_the_adapter_before_any_query(bare_psycopg2, monkeypatch):
    monkeypatch.setattr(psycopg2, "connect", lambda *a, **k: "conn")
    assert pg_connect.connect("postgresql://u@h/db") == "conn"
    assert _quote(round(np.float64(1.23456), 4)) == b"1.2346"


@pytest.mark.parametrize("value, sql", [
    (round(np.float64(1.23456), 4), b"1.2346"),     # runner.update_trades: round(np.float64) → np.float64
    (np.float32(0.5), b"0.5"),
    (np.int64(7), b"7"),
    (np.bool_(True), b"true"),
    (np.float64("nan"), b"'NaN'::float"),
    (1.5, b"1.5"),                                 # yerli tipler değişmez
    (None, b"NULL"),
])
def test_numpy_scalars_quote_as_native(bare_psycopg2, value, sql):
    pg_connect.register_numpy_adapters()
    assert _quote(value) == sql


class _PgQuotingLedger(ShadowLedger):
    """SQLite'a yazar ama her parametreyi önce Postgres yolundaki gibi alıntılar."""

    def __init__(self, path):
        self.quoted = []
        super().__init__(path)

    def _exec(self, sql, args=(), fetch=False):
        args = tuple(args)
        self.quoted += [_quote(a) for a in args]
        return super()._exec(sql, args, fetch)


def _lifecycle(tmp_path) -> list:
    L, src = _PgQuotingLedger(str(tmp_path / "s.db")), FakeSrc()
    for day in (16, 17):                                   # PENDING → OPEN (update_trade yazar)
        runner.run_shadow_daily(now=datetime(2025, 6, day, 7, 0, tzinfo=NY), ledger=L, src=src)
    runner.run_shadow_daily(now=datetime(2025, 7, 25, 7, 0, tzinfo=NY), ledger=L, src=src)   # CLOSED
    assert L.trades("H5")[0]["status"] == "CLOSED"
    return L.quoted


@pytest.mark.skipif(not NUMPY2, reason="repr hatası numpy 2'ye özgü")
def test_shadow_lifecycle_hits_the_bug_without_adapter(bare_psycopg2, tmp_path):
    # Test gerçekten arızalı yolu çalıştırıyor mu: adaptörsüz Postgres alıntısı bozuk olmalı.
    assert any(q.startswith(b"np.") or q == b"<cannot adapt>" for q in _lifecycle(tmp_path))


def test_shadow_lifecycle_params_are_postgres_safe(bare_psycopg2, tmp_path):
    pg_connect.register_numpy_adapters()
    bad = [q for q in _lifecycle(tmp_path) if q.startswith(b"np.") or q == b"<cannot adapt>"]
    assert not bad, bad[:5]
