# Final Correctness Report

11 Ekim 2026. **Yerel correctness kabulü: PASS. Production deployment yapılmadı.**
Bu çalışma kârlılık artışı iddiası veya yeni strateji içermez.

## Değişen dosyalar

| Dosya | Değişiklik |
|---|---|
| `swing_trader/paper_trading/execution.py` — yeni | I/O’suz, deterministik günlük-bar state transition ve ortak PnL hesabı |
| `swing_trader/paper_trading/tracker.py` | Kronolojik yürütme, tamamlanmış seans kontrolü, confirmation başlangıç riski, eski/kapalı kaydı yeniden yazmama |
| `swing_trader/paper_trading/storage.py` | Nullable migration, değişmez başlangıç riski, atomik checkpoint/PnL ve stale-writer kontrolü |
| `swing_trader/small_cap/smallcap_backtest.py` | Open kararlarında önceki barlar/rejim/equity; giriş günü exit; T1 kronolojisi; dönem sonundan sonraki fiyatı kullanmama |
| `api/routers/trades.py`, `api/routers/performance.py` | Partial sonrası ekran/backend marked PnL’sinde ortak kalan-adet hesabı |
| `scripts/backtest_live_replica.py` | Yalnız LEGACY/UNVERIFIED uyarısı; eski motor yeniden yazılmadı |
| `tests/test_trading_correctness.py` — yeni | A–I vakaları, migration ve sınır durumları |
| `api/tests/test_manual_stop_exit.py` | Stop düzenleme zamanını açık kaydeden, eksiksiz bar fixture’ı |
| `swing_trader/tests/test_backtest_exit_parity.py`, `test_tracker_trail_replay.py`, `test_tracker_nan_guard.py` | Yeni execution sözleşmesine uyum; mevcut davranış assertion’ları korundu |
| `.gitignore` | Yeni correctness testi ve mevcut trail regression dosyasının yanlışlıkla dağıtım dışında kalmasını önleyen dar istisnalar |

Önceden mevcut, görev dışı çalışma ağacı değişiklikleri korunmuştur; tablo yalnız bu görevde dokunulan dosyaları gösterir. Frontend, scheduler ve deployment değiştirilmedi.

## Düzeltilen hatalar ve kabul sonucu

| Kontrol | Sonuç |
|---|---|
| A — Entry günü stop | **PASS:** bilinen 09:30 New York girişinin ilk barı dahil |
| B/C — T1→BE ve imkânsız geçmiş stop | **PASS:** T1 olayı zamanlı checkpoint’te; BE önceki bara uygulanmıyor |
| D — Partial/kalan adet/PnL | **PASS:** 100 adet, giriş 100, T1 110/%33, kalan çıkış 105 → **$665**, tekrar çalıştırmada aynı |
| E — Orijinal risk | **PASS:** başlangıç stop 90 ve risk **$1.000** korunuyor; yukarıdaki işlem **0,665R** |
| F — Aynı bar stop+hedef | **PASS:** eski aktif stop önce; T1 sonrası BE/T2 sırası belirsizse muhafazakâr BE-first |
| G — Gap | **PASS:** stop altı açılışta Open; gap-up hedefte muhafazakâr hedef fiyatı; confirmation +%5/−%7 sınırları değişmedi |
| H — Open look-ahead | **PASS:** giriş gününün High/Low/Close/Volume’u aşırı değiştirilince entry ATR/stop/target/adet aynı; rejim/equity önceki tamamlanmış veriden |
| I — Tekrar/eşzamanlı çalışma | **PASS:** checkpoint, partial ve terminal PnL tek transaction; stale ikinci yazım reddediliyor |
| Strateji parametreleri | **PASS:** VCE/RVOL, Discovery V2, Quality, filtreler, risk/target/settings dosyaları değiştirilmedi |
| Tarihsel sonuçların korunması | **PASS:** kapalı kayıtlar otomatik replay edilmiyor; risk/fill backfill yok; arşiv backtestler yeniden çalıştırılmadı/yazılmadı |

Tam adet muhasebesinde yeni partial satış en az 1 ve en fazla mevcut adet kadar; 1 adetlik pozisyon T1’de tamamen kapanır. Gerçek satılan adet checkpoint’te saklanır; oran veya ayar sonradan değişse de satış adedi yeniden hesaplanmaz. Eski kayıtlarda mevcut yüzde/adet hesabı korunur. Hedef fiyatları, partial oranı ve strateji eşikleri değiştirilmemiştir.

## Çalıştırılmış testler

Önce iki kritik vaka eski kodda **2 FAILED** ile sabitlendi. Düzeltme sonrası son kapsamlı koşu:

```text
python -m pytest tests swing_trader/tests api/tests swing_trader/discovery -q -p no:cacheprovider --basetemp output/final_correctness_final --junitxml output/final_correctness_final.xml
533 passed, 3 warnings in 14.27s
```

[JUnit sonucu](output/final_correctness_final.xml). Discovery testleri dahil. Üç uyarı protobuf/NumPy deprecation; pytest ayrıca async fixture scope için deprecation uyarısı yazdı. `git diff --check`: PASS. Kök `test_backtest.py` import sırasında dış veri/backtest çalıştıran standalone script olduğundan unit suite’e alınmadı; eski performans dosyalarını yeniden üretmedik.

Migration: `initial_risk`, `r_multiple`, `partial_exit_date`, `execution_state`, `policy_effective_at` **nullable** eklendi; hiçbir tarihsel alan tahmin edilerek doldurulmadı. SQLite’ta ikinci çalıştırma/idempotency, eski satırların korunması ve DDL hata rollback’i geçti. PostgreSQL DDL sözleşmesi ve hata yayılımı test double ile geçti; gerçek PostgreSQL entegrasyonu aşağıda açıkça ayrılmıştır.

İlk genel koşudaki eski default-storage testleri yerel `data/paper_trades.db` üzerinde additive şema migration’ını tetikledi; sonrasında bu fixture’lar Mock storage’a alındı. Yerel **21 eski kaydın** yeni risk/state/fill-date/R alanları hâlâ **NULL**; geçmiş fill veya PnL güncellemesi yapılmadı. Bu görevde uzak production DB bağlantısı açılmadı.

## Bilinen sınırlamalar / BLOCKED durumları

- **BLOCKED — gerçek PostgreSQL staging migration doğrulaması:** canlı veya staging DB’de çalıştırılmadı. Yerel başarı production’a doğrudan onay değildir.
- **BLOCKED — intraday veya saati bilinmeyen girişin günlük OHLC ile replay’i:** `REVIEW_REQUIRED` döner; giriş öncesi High/Low pozisyona uygulanmaz. Günlük engine yalnız açıkça 09:30 ET girişini tam barla işleyebilir. Gün içi stop/target düzenlemesinin sırası da günlük barla uydurulmaz.
- **BLOCKED — eski partial kaydın eksik olay zamanı/orijinal riski:** otomatik replay/backfill yok. Bilinen beş uyuşmazlık **PRCT, PAR, BRUN, PUBM, MITK** ayrı veri sorunu olarak kalıyor; hiçbirine düzeltme uygulanmadı.
- Eksik/duplicate/geçmişi revize edilmiş barlar checkpoint’i ilerletmez. Corporate-action kaynaklı geçmiş fiyat revizyonu da inceleme gerektirir. `execution_warning` API yanıtında döner; bunun için yeni frontend geliştirilmedi.
- OHLC içindeki gerçek sıra ve dolum saati bilinmiyor. İntraday olaylar için 16:00 timestamp’i **proxy**, stop-first/BE-first **muhafazakâr varsayım**; broker execution kanıtı değildir.
- Tarihsel backtest ile production’ın tüm giriş/risk/fundamentals varsayımları hâlâ eşdeğer değildir. Legacy replica’nın gelecekteki likiditeyi kullanan slippage yolu düzeltilmiş performans diye sunulmadı. Eski raporların getirileri bu düzeltmeyle kendiliğinden doğrulanmış olmaz.

## Production’a güvenli aktarım için gerekli kontroller

1. Ayrı staging PostgreSQL kopyasında yedek al; additive migration’ı iki kez çalıştır; eski satır sayısı ve mevcut alan checksum’larının değişmediğini, yeni alanların NULL kaldığını doğrula. DDL hata rollback ve eşzamanlı stale-writer senaryosunu gerçek PostgreSQL’de de geçir. **Bu kontrol tamamlanana kadar aktarım BLOCKED.**
2. Açık eski kayıtların giriş saati/T1 geçmişi belirsizse otomatik güncellemeyi açma; veri uyuşmazlıklarını ayrı tut. Güncel bar sağlayıcısının adjustment/kalender uyumunu ve `REVIEW_REQUIRED` yanıtını kontrol et.
3. Deploy edilecek diff’in yalnız bu rapordaki dosyaları kapsadığını doğrula; mevcut ilgisiz kullanıcı değişikliklerini dahil etme. Dondurulmuş tek örnekte entry→T1→BE/T2/timeout ve tekrar çağrı smoke kontrolünü yap. Tarihsel paper veya arşiv backtest sonucunu overwrite etme.

**Yerel düzeltme ve doğrulama sona erdi. Kârlılık artışı ölçülmedi/iddia edilmedi; production’a aktarım yapılmadı.**
