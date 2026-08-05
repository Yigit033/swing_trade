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


@pytest.fixture(autouse=True)
def _fresh_settings_cache():
    """
    load_settings 30 sn önbellekli (2026-08-05 — her çağrı JSON okuyup DB turu
    atıyordu, ~2.5 sn; tracker'ın çıkış döngüsünde ve her sinyalde çağrılıyor).
    Bu testler DB katmanını doğrudan değiştirdiği için önbelleği atlamaları
    gerekir; aksi halde önceki testin değerini görürler.
    """
    sc.invalidate_settings_cache()
    yield
    sc.invalidate_settings_cache()


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

def _file_layer_value():
    """DB katmanı olmadan dosya/koddan gelen auto_scan.enabled değeri."""
    import json as _json
    from pathlib import Path as _Path
    p = _Path(sc.DEFAULT_SETTINGS_PATH)
    if p.exists():
        raw = _json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "auto_scan" in raw and "enabled" in raw["auto_scan"]:
            return bool(raw["auto_scan"]["enabled"])
    return bool(sc.SmallCapSettings().auto_scan.enabled)


def test_db_overlay_wins_over_file(fake_db):
    """
    DB yaması dosya/varsayılan değerini ezmeli — asıl bug buydu.
    Belirli bir varsayılana bağlanmıyoruz: dosya katmanının TERSİNİ yamalayıp
    üstün gelip gelmediğine bakıyoruz (varsayılan ileride değişse de geçerli).
    """
    base = _file_layer_value()
    fake_db["patch"] = {"auto_scan": {"enabled": not base}}
    assert sc.load_settings().auto_scan.enabled is (not base)


def test_db_overlay_is_partial_not_snapshot(fake_db):
    """Yama yalnız bir alanı taşısa da diğer alanlar dosya/koddan gelmeli."""
    base = _file_layer_value()
    fake_db["patch"] = {"auto_scan": {"enabled": not base}}
    s = sc.load_settings()
    assert s.auto_scan.enabled is (not base)
    # Dokunulmayan alan değerini korur (yama tam görüntü DEĞİL)
    assert s.auto_scan.target_hour_et == 16
    # Tamamen ilgisiz bölüm de etkilenmez
    assert s.regime_thresholds.caution_other_min_quality == 80


def test_no_db_falls_back_to_file(no_db):
    """DATABASE_URL yoksa eski davranış: dosya tek kaynak, hata yok."""
    s = sc.load_settings()
    assert s.auto_scan.enabled is _file_layer_value()
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


# ── Kademeli geri çekilme (2026-08-04) ───────────────────────────────────
# RİSK: DB yaması katmanı eklendiğinde şu senaryo doğdu — bir ayar alanı koddan
# kaldırılırsa (model extra="forbid") ama DB'deki eski yama o alanı taşırsa,
# doğrulama patlar ve load_settings TÜM kalibrasyonu (eşikler, exit, evren
# filtreleri) sessizce varsayılana düşürür. Ölçülmüş her parametre bir anda
# kaybolur ve kimse fark etmez. Çözüm: katmanları tek tek geri çek.

def test_invalid_db_patch_falls_back_to_file_not_defaults(monkeypatch, tmp_path):
    """
    Bozuk DB yaması yalnız KENDİSİ yok sayılmalı; dosya katmanındaki ölçülmüş
    değerler KORUNMALI. Eskiden hepsi birden varsayılana düşüyordu.
    """
    # Dosya katmanında ölçülmüş, varsayılandan FARKLI bir değer olsun
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"max_holding_days": 17}), encoding="utf-8")
    monkeypatch.setattr(sc, "DEFAULT_SETTINGS_PATH", p)
    # DB yaması geçersiz (modelde olmayan alan → extra="forbid" patlatır)
    monkeypatch.setattr(sc, "_db_overlay", lambda: {"bu_alan_yok_artik": 123})

    s = sc.load_settings()
    assert s.max_holding_days == 17, (
        "bozuk DB yaması dosya katmanındaki ölçülmüş değeri de sildi — "
        "kademeli geri çekilme çalışmıyor"
    )


def test_invalid_file_and_db_falls_back_to_defaults(monkeypatch, tmp_path):
    """İki katman da bozuksa varsayılana düşer (son çare) ama ÇÖKMEZ."""
    p = tmp_path / "s.json"
    p.write_text('{"bu_alan_yok_artik": 1}', encoding="utf-8")
    monkeypatch.setattr(sc, "DEFAULT_SETTINGS_PATH", p)
    monkeypatch.setattr(sc, "_db_overlay", lambda: {"baska_olmayan_alan": 2})
    s = sc.load_settings()          # fırlatmamalı
    assert s.max_holding_days == sc.SmallCapSettings().max_holding_days


# ── Önbellek sözleşmesi (2026-08-05) ─────────────────────────────────────

def test_cache_serves_repeat_reads_without_hitting_db(monkeypatch):
    """
    Önbellek gerçekten çalışmalı: TTL içinde ikinci okuma DB'ye GİTMEMELİ.
    Bu, ölçülen performans sorununun düzeltildiğini bağlar — 5 çağrı 12.7 sn
    sürüyordu, çünkü her çağrı JSON okuyup üstüne bir DB turu atıyordu ve bu
    fonksiyon sıcak yollarda (her sinyal, her çıkış kontrolü) çağrılıyor.
    """
    calls = {"n": 0}

    def _counting_overlay():
        calls["n"] += 1
        return {}

    monkeypatch.setattr(sc, "_db_overlay", _counting_overlay)
    sc.invalidate_settings_cache()

    sc.load_settings()
    sc.load_settings()
    sc.load_settings()
    assert calls["n"] == 1, f"DB {calls['n']} kez okundu — önbellek çalışmıyor"


def test_invalidate_forces_fresh_read(monkeypatch):
    """
    UI'dan kayıt yapılınca ayar ANINDA uygulanmalı — önbellek gecikmesi
    kullanıcıya "değiştirdim ama bir şey olmadı" yaşatmamalı.
    """
    store = {"patch": {}}
    monkeypatch.setattr(sc, "_db_overlay", lambda: dict(store["patch"]))
    sc.invalidate_settings_cache()

    base = sc.load_settings().auto_scan.enabled
    store["patch"] = {"auto_scan": {"enabled": not base}}

    assert sc.load_settings().auto_scan.enabled is base, "önbellek beklendiği gibi tutmuyor"
    sc.invalidate_settings_cache()
    assert sc.load_settings().auto_scan.enabled is (not base), "invalidate sonrası taze okumadı"


def test_explicit_path_never_uses_cache(tmp_path, monkeypatch):
    """Açık `path` verilen çağrılar (testler, geçici dosyalar) önbelleğe girmez."""
    monkeypatch.setattr(sc, "_db_overlay", lambda: {})
    sc.invalidate_settings_cache()
    sc.load_settings()   # önbelleği doldur

    p = tmp_path / "s.json"
    p.write_text(json.dumps({"max_holding_days": 11}), encoding="utf-8")
    assert sc.load_settings(path=p).max_holding_days == 11, (
        "açık path önbellekten dönmüş — izole dosya okuması bozulur"
    )
