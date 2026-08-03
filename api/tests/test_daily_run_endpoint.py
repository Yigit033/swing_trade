"""
POST /api/scanner/daily-run — günlük tetikleyici testleri (2026-08-04).

BAĞLAM: fly.io makinesi boştayken askıya alınıyor (min_machines_running=0) ve
askıdayken hiçbir asyncio görevi çalışmıyor — canlı logda görüldü:
    18:32:59 autosuspending machine
    18:58:35 Starting machine        (yalnız gelen HTTP isteğiyle)
Bu, auto_scan_loop'u ve pending/exit döngüsünü pratikte hiç çalıştırmıyordu.
7/24 makine (~$6/ay) yerine ücretsiz GitHub Actions cron bu endpoint'i çağırıyor.

Endpoint dışarıya açık olduğu için kimlik doğrulaması kritik: CRON_SECRET yoksa
kapalı (503), yanlışsa reddedilir (401). Bu testler o sözleşmeyi kilitler.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app, raise_server_exceptions=False)
URL = "/api/scanner/daily-run"


@pytest.fixture
def secret(monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "test-secret-123")
    return "test-secret-123"


@pytest.fixture
def stub_maintenance(monkeypatch):
    """Gerçek bakım işini (yfinance/DB) çalıştırmadan endpoint'i test et."""
    calls = []

    def _fake(force=False):
        calls.append(force)
        return {"pending_confirmed": 0, "trades_closed": 0, "scan": "ran"}

    import api.auto_scan as auto_scan
    monkeypatch.setattr(auto_scan, "run_daily_maintenance", _fake)
    return calls


def test_disabled_when_secret_not_configured(monkeypatch):
    """CRON_SECRET yoksa endpoint kapalı olmalı — kazara açık kalmasın."""
    monkeypatch.delenv("CRON_SECRET", raising=False)
    r = client.post(URL, headers={"X-Cron-Secret": "anything"})
    assert r.status_code == 503


def test_rejects_missing_header(secret, stub_maintenance):
    r = client.post(URL)
    assert r.status_code == 401
    assert not stub_maintenance, "kimlik doğrulanmadan bakım çalıştı"


def test_rejects_wrong_secret(secret, stub_maintenance):
    r = client.post(URL, headers={"X-Cron-Secret": "yanlis"})
    assert r.status_code == 401
    assert not stub_maintenance


def test_runs_with_correct_secret(secret, stub_maintenance):
    r = client.post(URL, headers={"X-Cron-Secret": secret})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scan"] == "ran"
    assert stub_maintenance == [False]


def test_force_flag_is_passed_through(secret, stub_maintenance):
    """?force=true aynı gün ikinci taramayı zorlar (elle tetikleme senaryosu)."""
    r = client.post(f"{URL}?force=true", headers={"X-Cron-Secret": secret})
    assert r.status_code == 200
    assert stub_maintenance == [True]


def test_response_is_json_safe(secret, monkeypatch):
    """
    Bakım sonucu NaN/numpy içerebilir (fiyat hesapları). sanitize_for_json'dan
    geçmeli — aksi halde endpoint 500 verir ve cron başarısız görünür.
    (Bu proje bu hata sınıfını /api/trades'te canlıda yaşadı.)
    """
    import numpy as np
    import api.auto_scan as auto_scan
    monkeypatch.setattr(
        auto_scan, "run_daily_maintenance",
        lambda force=False: {"scan": "ran", "pnl": float("nan"),
                             "n": np.int64(3), "ratio": np.float64(0.5)},
    )
    r = client.post(URL, headers={"X-Cron-Secret": secret})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pnl"] is None      # NaN → None
    assert body["n"] == 3
    assert body["ratio"] == 0.5
