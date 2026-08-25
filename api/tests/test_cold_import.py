"""Importing api.main must not load pandas/numpy/yfinance.

Fly's edge proxy gives ~8s for TCP on 0.0.0.0:8000 after a cold start.
The scientific stack takes longer than that on shared-cpu-1x.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_import_api_main_does_not_load_pandas_numpy_yfinance():
    script = (
        "import sys\n"
        "from api.main import app\n"
        "heavy = [m for m in ('pandas', 'numpy', 'yfinance') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
        "assert any(getattr(r, 'path', None) == '/api/health' for r in app.routes)\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_pg_connect_timeout_is_fail_fast():
    from swing_trader.utils.pg_connect import CONNECT_TIMEOUT_SEC

    assert CONNECT_TIMEOUT_SEC <= 3
