"""Fail-fast PostgreSQL connect.

A paused or firewalled Supabase host must not block a worker for 10s × N DNS
records — that freezes health checks and Fly marks the machine unhealthy.

Every PostgreSQL connection in the app goes through ``connect`` — so this is also
the one place where numpy scalars are made safe to pass as query parameters.
"""

from __future__ import annotations

CONNECT_TIMEOUT_SEC = 3


def register_numpy_adapters() -> None:
    """Quote numpy scalars as their native Python values.

    numpy 2 changed ``repr(np.float64(1.5))`` to ``'np.float64(1.5)'``. ``np.float64``
    subclasses ``float``, so psycopg2 picks its float adapter, which quotes with
    ``repr`` — the SQL then reads ``ret_pct = np.float64(1.5)`` and Postgres fails with
    ``schema "np" does not exist`` (2026-11-24, first shadow update with open trades).
    ``np.int64``/``np.bool_`` are not int/bool subclasses and fail with "can't adapt".
    psycopg2 walks the type's MRO and ``np.generic`` comes before ``float``, so one
    adapter covers every numpy scalar. Idempotent.
    """
    try:
        import numpy as np
        from psycopg2.extensions import adapt, register_adapter
    except ImportError:
        return
    register_adapter(np.generic, lambda v: adapt(v.item()))


def connect(url: str):
    """Open a psycopg2 connection with a short TCP timeout."""
    import psycopg2

    register_numpy_adapters()
    return psycopg2.connect(url, connect_timeout=CONNECT_TIMEOUT_SEC)
