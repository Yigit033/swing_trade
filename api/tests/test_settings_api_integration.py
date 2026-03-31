"""
Integration tests: GET/PUT /api/settings + disk persistence + SmallCapEngine cache refresh.

Bu, arayüzdeki «Kaydet» ile aynı sözleşmeyi doğrular; tam tarayıcı E2E yerine
API katmanında deterministik ve hızlı kalır.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def isolated_settings_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from swing_trader.small_cap.settings_config import default_settings, save_settings

    p = tmp_path / "smallcap_settings.json"
    save_settings(default_settings(), path=p)
    monkeypatch.setattr("swing_trader.small_cap.settings_config.DEFAULT_SETTINGS_PATH", p)
    return p


@pytest.fixture()
def api_client(isolated_settings_file: Path) -> TestClient:
    from api.deps import invalidate_smallcap_engine_cache

    invalidate_smallcap_engine_cache()
    from api.main import app

    return TestClient(app)


def test_api_settings_get_returns_schema(api_client: TestClient) -> None:
    r = api_client.get("/api/settings")
    assert r.status_code == 200
    data = r.json()
    assert data.get("max_entry_rsi") == 70
    assert "scoring_tuning" in data
    st = data["scoring_tuning"]
    assert isinstance(st.get("volume_surge_tiers"), list)
    assert len(st["volume_surge_tiers"]) >= 1


def test_api_settings_put_merges_persists_and_roundtrips(
    api_client: TestClient, isolated_settings_file: Path
) -> None:
    r = api_client.put("/api/settings", json={"max_entry_rsi": 66})
    assert r.status_code == 200
    payload = r.json()
    assert payload.get("ok") is True
    assert payload["settings"]["max_entry_rsi"] == 66

    disk = json.loads(isolated_settings_file.read_text(encoding="utf-8"))
    assert disk["max_entry_rsi"] == 66

    r2 = api_client.get("/api/settings")
    assert r2.status_code == 200
    assert r2.json()["max_entry_rsi"] == 66


def test_api_settings_put_invalidates_engine_cache(api_client: TestClient) -> None:
    """Kayıt sonrası bir sonraki get_smallcap_engine() dosyadan okunan değeri kullanır."""
    from api.deps import get_smallcap_engine, invalidate_smallcap_engine_cache

    invalidate_smallcap_engine_cache()
    api_client.put("/api/settings", json={"max_entry_rsi": 61})
    eng = get_smallcap_engine()
    assert eng.settings.max_entry_rsi == 61


def test_api_settings_put_validation_error(api_client: TestClient) -> None:
    r = api_client.put("/api/settings", json={"max_entry_rsi": 999})
    assert r.status_code == 422


def test_api_settings_trailing_slash_no_redirect(api_client: TestClient) -> None:
    """Strict clients (follow_redirects=False) must get 200, not 307 Temporary Redirect."""
    r_get = api_client.get("/api/settings/", follow_redirects=False)
    assert r_get.status_code == 200
    r_put = api_client.put("/api/settings/", json={"max_entry_rsi": 69}, follow_redirects=False)
    assert r_put.status_code == 200
    assert r_put.json()["settings"]["max_entry_rsi"] == 69
    r_rst = api_client.post("/api/settings/reset/", follow_redirects=False)
    assert r_rst.status_code == 200
    assert r_rst.json()["settings"]["max_entry_rsi"] == 70
