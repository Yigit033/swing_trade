"""
Ayar kalıcılığı regresyon testleri (2026-08-03).

Canlı bug: kullanıcı auto-scan'i UI'dan AÇTI, çalıştı, sonra bir deploy geldi ve
ayar sessizce `enabled=False`'a döndü. Sebep: fly.toml'da mount yok → container
diski geçici; `data/smallcap_settings.json` imaja git'ten kopyalanıyor, UI'nın
yazdığı dosya deploy'da siliniyor. Hiçbir hata görünmüyordu — ayar "kaydedildi"
diyordu ve gerçekten kaydedilmişti, sadece geçici bir diske.

Bu testler kalıcılık sözleşmesini kilitler:
  1. Katman sırası: kod varsayılanları → JSON dosyası → DB yaması (en güçlü)
  2. DB'ye SADECE kullanıcının değiştirdiği alanlar (yama) yazılır — tam anlık
     görüntü değil. Yoksa ileride ölçümle yükseltilen bir eşik, DB'de donmuş
     eski değer tarafından sessizce ezilirdi.
  3. DB yoksa davranış eskisi gibi (dosya tek kaynak) — yerel geliştirme kırılmaz.
"""

import json

import pytest

from swing_trader.small_cap import settings_config as sc


@pytest.fixture
def no_db(monkeypatch):
    """DB katmanını kapat — dosya-tek-kaynak davranışı."""
    monkeypatch.setattr(sc, "_db_overlay", lambda: {})
    return None


@pytest.fixture
def fake_db(monkeypatch):
    """Bellekte yaşayan sahte DB yaması + is_enabled/save/load kancaları."""
    store = {"patch": {}}

    monkeypatch.setattr(sc, "_db_overlay", lambda: dict(store["patch"]))

    fake_mod = type("_M", (), {
        "is_enabled": staticmethod(lambda: True),
        "load_patch": staticmethod(lambda: dict(store["patch"])),
        "save_patch": staticmethod(lambda p: (store.__setitem__("patch", dict(p)), True)[1]),
        "clear_patch": staticmethod(lambda: (store.__setitem__("patch", {}), True)[1]),
    })
    import sys
    monkeypatch.setitem(sys.modules, "swing_trader.data.settings_storage", fake_mod)
    return store


# ── Katman sırası ────────────────────────────────────────────────────────

def test_db_overlay_wins_over_file(fake_db):
    """DB yaması dosya/varsayılan değerini ezmeli — asıl bug buydu."""
    file_value = sc.load_settings().auto_scan.enabled
    assert file_value is False, "beklenen başlangıç: auto-scan kapalı"

    fake_db["patch"] = {"auto_scan": {"enabled": True}}
    assert sc.load_settings().auto_scan.enabled is True


def test_db_overlay_is_partial_not_snapshot(fake_db):
    """Yama yalnız bir alanı taşısa da diğer alanlar dosya/koddan gelmeli."""
    fake_db["patch"] = {"auto_scan": {"enabled": True}}
    s = sc.load_settings()
    assert s.auto_scan.enabled is True
    # Dokunulmayan alan varsayılanını korur (yama tam görüntü DEĞİL)
    assert s.auto_scan.target_hour_et == 16
    # Tamamen ilgisiz bölüm de etkilenmez
    assert s.regime_thresholds.caution_other_min_quality == 80


def test_no_db_falls_back_to_file(no_db):
    """DATABASE_URL yoksa eski davranış: dosya tek kaynak, hata yok."""
    s = sc.load_settings()
    assert s.auto_scan.enabled is False
    assert s.regime_thresholds.caution_other_min_quality == 80


def test_explicit_path_ignores_db_overlay(fake_db, tmp_path):
    """Açık path verilen çağrı (testler/geçici dosya) DB katmanını bindirmez."""
    fake_db["patch"] = {"auto_scan": {"enabled": True}}
    p = tmp_path / "isolated.json"
    p.write_text(json.dumps({"auto_scan": {"enabled": False}}), encoding="utf-8")
    assert sc.load_settings(path=p).auto_scan.enabled is False


# ── Yazma yolu ───────────────────────────────────────────────────────────

def test_patch_is_persisted_to_db(fake_db, tmp_path, monkeypatch):
    """apply_settings_patch DB'ye SADECE yamayı yazmalı."""
    monkeypatch.setattr(sc, "DEFAULT_SETTINGS_PATH", tmp_path / "s.json")
    sc.apply_settings_patch({"auto_scan": {"enabled": True}})
    assert fake_db["patch"] == {"auto_scan": {"enabled": True}}, \
        f"DB'ye tam görüntü yazıldı: {list(fake_db['patch'])[:5]}"


def test_patches_accumulate(fake_db, tmp_path, monkeypatch):
    """İkinci yama birincisini silmemeli (birikimli)."""
    monkeypatch.setattr(sc, "DEFAULT_SETTINGS_PATH", tmp_path / "s.json")
    sc.apply_settings_patch({"auto_scan": {"enabled": True}})
    sc.apply_settings_patch({"auto_scan": {"top_n": 20}})
    assert fake_db["patch"]["auto_scan"] == {"enabled": True, "top_n": 20}


def test_survives_simulated_redeploy(fake_db, tmp_path, monkeypatch):
    """
    Deploy senaryosu: dosya git sürümüne döner (UI yazdığı dosya silinir),
    DB yaması sağ kalır → ayar KORUNUR. Bug'ın tam tersi davranış.
    """
    file_path = tmp_path / "s.json"
    monkeypatch.setattr(sc, "DEFAULT_SETTINGS_PATH", file_path)

    sc.apply_settings_patch({"auto_scan": {"enabled": True}})
    assert sc.load_settings().auto_scan.enabled is True

    # ── deploy: container diski yenilenir, dosya git haline döner ──
    file_path.unlink(missing_ok=True)

    assert sc.load_settings().auto_scan.enabled is True, \
        "deploy sonrası ayar kayboldu — kalıcılık çalışmıyor"


def test_corrupt_db_patch_does_not_break_loading(monkeypatch):
    """DB yaması bozuksa (geçersiz alan) ürün açılmaya devam etmeli."""
    monkeypatch.setattr(sc, "_db_overlay", lambda: {"auto_scan": {"enabled": "evet-mi-hayır-mı"}})
    s = sc.load_settings()          # fırlatmamalı
    assert s.auto_scan.enabled in (True, False)
