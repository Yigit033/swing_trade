"""Fail-fast PostgreSQL connect.

A paused or firewalled Supabase host must not block a worker for 10s × N DNS
records — that freezes health checks and Fly marks the machine unhealthy.
"""

from __future__ import annotations

CONNECT_TIMEOUT_SEC = 3


def connect(url: str):
    """Open a psycopg2 connection with a short TCP timeout."""
    import psycopg2

    return psycopg2.connect(url, connect_timeout=CONNECT_TIMEOUT_SEC)
