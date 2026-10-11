# Final Full-Stack Integration Audit

11 Ekim 2026; incelenen baseline: `a293b2f` ve önceki `f590f3a`, `b486676`, `6936f4d`.
**Doğrulanmış entegrasyon hataları bulundu; küçük düzeltmeler yalnız yerelde yapıldı. Commit/push/deploy yapılmadı.**
Strateji, hedef/stop parametreleri, position sizing, scheduler ve production DB kayıtları değiştirilmedi.

## Sonuç

PASS yalnız aşağıda belirtilen yerel test kapsamını ifade eder; authenticated canlı browser E2E anlamına gelmez.
`FAIL → PASS` baseline hatasının yerelde düzeltilip test edildiğini gösterir. Production hücreleri yeni düzeltmeler için UNKNOWN'dur.

| Feature/Change | Backend | API | Frontend | Production | Status |
|---|---|---|---|---|---|
| Scanner T1/T2 metni ve track payload | PASS | PASS | PASS | UNKNOWN | FAIL → PASS |
| Manual Lookup → track tarih/portföy/ATR/adet/hold | PASS | PASS | PASS | UNKNOWN | FAIL → PASS |
| Paper Trades fiyat, partial PnL, execution uyarısı | PASS | PASS | PASS | UNKNOWN | FAIL → PASS |
| Performance fiyat/PnL ve boş-hata ayrımı | PASS | PASS | PASS | UNKNOWN | FAIL → PASS |
| Settings hata ekranı ve auto-entry açıklamaları | PASS | PASS | PASS | UNKNOWN | FAIL → PASS |
| Edge policy cohort/types ve tarih etiketi | PASS | PASS | PASS | UNKNOWN | FAIL → PASS |
| Internal initial_risk / r_multiple / checkpoint | PASS | PASS | NOT_APPLICABLE | UNKNOWN | PASS (yerel sözleşme) |
| Discovery V2 izolasyonu | PASS | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | PASS (statik import kontrolü) |
| Backend `a293b2f` deployment | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | PASS | PASS (workflow + health) |
| Frontend `a293b2f` deployment | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | PASS | PASS (Vercel deployment kaydı) |
| Giriş yapılmış canlı kullanıcı E2E | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |

## FAIL bulguları ve düzeltmeler

1. **Eksik ticker'a başka hissenin fiyatı:** `api/routers/trades.py::_enrich_open_trades_inline` ve `performance.py::_fetch_live_prices`, eksik sütunda `close.iloc[:, 0]` kullanıyordu. Fallback kaldırıldı; eksik fiyat eksik kalıyor. İki endpoint için provider-missing-symbol testleri geçti.
2. **Lookup takip isteği:** `frontend/src/app/lookup/page.tsx::handleAdd`, `data_as_of`, seçilen `portfolioValue`, `position_size` ve `hold_days` bilgilerini kaybediyor, ATR'yi sıfır gönderiyordu. Mevcut bar tarihi ve risk çıktıları `trackSignal` üzerinden taşınıyor. `api/routers/lookup.py::_analyze_smallcap_ticker` mevcut risk motorunun `atr` sonucunu response'a ekliyor; yeniden ATR hesabı yok. Eksik bar tarihiyle ekleme engellendi. Render-handler, gerçek serializer ve API risk-output testleri geçti.
3. **Yanlış sabit T1 oranı:** `scanner/page.tsx::SignalCard` yüzde 50 diyordu; ayar `partial_at_t1_fraction` ile değişebiliyor (kod default'u 0.33). Sabit oran yerine ayarlardaki kısmi çıkış oranını belirten metin kullanıldı; hiçbir hedef/strateji değişmedi.
4. **API hatasının boş sonuç gibi görünmesi:** `trades/page.tsx` ve `performance/page.tsx`, hook `isError` alanını kullanmıyordu. Açık hata/tekrar-dene görünümü eklendi. `settings/page.tsx` içindeki `!draft` loading guard'ı hata ekranını erişilemez yapıyordu; hata önceliği düzeltildi. Settings hatası yeni render testinde önce yeniden üretildi, düzeltme sonrası geçti.
5. **Partial PnL ve güncel fiyat uydurulması:** `trades/page.tsx` null backend PnL'sini tam pozisyon adediyle tekrar hesaplıyor, eksik güncel fiyatı giriş fiyatıyla gösteriyordu. Bu fallback'ler kaldırıldı; bilinmeyen değer `—`. API'lerin partial toplamı aynı: 100 adet/100 giriş, 33 adet/110 T1, kalan/105 mark → $665 / %6.65. Orijinal stop 90, güncel stop 100 ve risk 1000 ayrı kalıyor.
6. **Execution uyarısının gizlenmesi/stale ekranlar:** API'nin `execution_warning` alanı TS `Trade` tipine eklendi; PATCH ve update-prices sonuçları tiplenip uyarılar ekranda gösteriliyor. Price-refresh sonrası Performance/Pending query cache'leri de invalid ediliyor. Checkpoint JSON veya internal risk alanları için gereksiz yeni UI yapılmadı. R-multiple ekranda hesaplanmıyor; eski/null riskten sahte R üretilmedi.
7. **Performance yanlış kapsam/format:** best/worst hesabındaki `||` sıfırı kaybediyordu; `??` kullanıldı. Reddedilmiş veya PnL'siz kayıtlar grafiklerden çıkarıldı. Negatif best return artık `+-5%` görünmüyor. Yüzdelerin toplamı portföy getirisi değildir; etiket düzeltildi. Eğri/best-worst son 30 API kaydıyla sınırlı olarak açıklandı, full-history gibi sunulmadı.
8. **Auto-entry açıklaması:** `settings/page.tsx` kalite eşiğini geçen her sinyalin PENDING olacağını ve BEAR ayarının yalnız simülasyona ait olduğunu söylüyordu. Mevcut `api/auto_scan.py::auto_entry_capacity` davranışıyla uyumlu BEAR/CAUTION ve owner OPEN+PENDING kapasite açıklamaları yazıldı. Kullanıcı Scanner Auto-Track ve zamanlanmış auto-scan ayrı akışlar olarak kaldı; kurallar birleştirilmedi/değiştirilmedi.
9. **Edge tarih bağımsızlığı iddiası:** `edge/page.tsx` farklı tarihleri “bağımsız” sayıyordu; outcome pencerelerinin örtüşebileceği açıklandı. Backend'in confidence/verdict algoritması değiştirilmedi; bu audit istatistiksel edge doğrulaması değildir.

## Testler ve kapsam sınırı

- `python -m pytest tests swing_trader/tests api/tests swing_trader/discovery -q -p no:cacheprovider --basetemp output/fullstack_integration_final --junitxml output/fullstack_integration_final.xml`: **539 passed**, 3 deprecation warning; 7.77 saniye. Yeni `tests/test_integration_audit.py`: 6 test; SQLite/mocked bağımlılıklar, production DB yok.
- `cd frontend; node --test integration-audit.test.cjs`: **10 passed**. Gerçek TSX render, lookup handler, gerçek API serializer ve query invalidation testleri; provider/query bağımlılıkları mock. Browser hydration/auth/network E2E değildir.
- `node node_modules/typescript/bin/tsc --noEmit --incremental false`: PASS. `npm run build`: PASS, 17 statik sayfa; mevcut middleware deprecation uyarısı devam ediyor.
- `git diff --check`: PASS. `.gitignore` yalnız yeni backend regression testini izlenebilir kılacak şekilde güncellendi; üretilmiş test/build çıktıları dışarıda.
- Production modüllerinde Discovery import'ları için AST testi + UI kaynaklarında `5245`/`5.245`/`5,245` araması temiz. Discovery/research CLI paketleri aktif scanner'a bağlanmadı. Bu audit yeni broad-market tarama çalıştırmadı.
- Pydantic request sözleşmeleri korunuyor. Trades/Performance response'ları dinamik dict/sanitize_for_json yolunda; kapsamlı response_model doğrulaması yok. TestClient testleri belirtilen alanları gerçek JSON response üzerinden kontrol ediyor; bütün olası null/provider/auth varyantları test edilmiş sayılmadı.
- Gerçek PostgreSQL migration/stale-writer staging kontrolü ve authenticated canlı mutation/browser E2E **UNKNOWN**. Eski audit'teki bu engeller deploy başarısıyla çözülmüş sayılmıyor.

## Production kanıtı — sadece okuma

- GitHub `fly-deploy.yml` workflow kaydı: `a293b2fea1b0b471fe1e1d11ed5156bdc345c13c`, completed/success, `2026-10-11T11:36:07Z`. Backend `/api/health`: HTTP 200, status ok, version 2.1.0. Health SHA yayınlamıyor: **çalışan process SHA'sı bağımsız olarak UNKNOWN**, başarılı deploy commit'i biliniyor.
- GitHub Vercel status success; Production deployment ID `6994953180`, aynı SHA, success `2026-10-11T11:36:40Z`. `https://swingtrade.vercel.app`: HTTP 200 HTML/Vercel. Canonical alias'ın runtime SHA'sı ayrıca yayımlanmıyor; deploy kaydı ile runtime kimliği aynı kanıt değildir.
- Backend CORS preflight: HTTP 200, `Access-Control-Allow-Origin: https://swingtrade.vercel.app`.
- Bu turdaki düzeltmeler production'a gönderilmedi; canlı ürünün kusursuz olduğu iddia edilmiyor.
