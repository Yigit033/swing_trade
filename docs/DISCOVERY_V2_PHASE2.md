# Discovery Engine V2 — Phase 2

## Kısa sonuç

**Bağımsız full-universe OHLCV hazırlama ve shadow tarama çalıştı; production'a bağlanmadı.**
9 Ekim EOD'da5.245 adaydan5.221'inde güncel veri,5.141'inde yeterli geçmiş var.
19 unique trigger,2 hard-gate PASS,**0 final PASS**. 9 Ekim legacy karşılaştırması
UNKNOWN;8 Ekim retrospektif karşılaştırmasında19 ek trigger var, kâr kanıtı yok.
**481 test geçti; tam tarama offline replay'de birebir aynı sonucu verdi.**

## Kapsam ve sınır

Bu çalışma bağımsız araştırma/ölçüm altyapısıdır; işlem tavsiyesi veya kârlılık kanıtı değildir.
Production scanner, Finviz, risk planner, T1/T2, stop/trailing, paper trading, scheduler,
migration, frontend ve deployment değiştirilmedi. Başlangıçtaki kirli çalışma ağacındaki
production değişiklikleri bu çalışmanın değişiklikleri değildir.

Birincil EOD seansı **2026-10-09**. İşlem öncesi gözlem zamanı 10 Ekim UTC / 11 Ekim
Türkiye zamanıdır. Varsayımsal sonraki açılış **2026-10-12 09:30 America/New_York**;
Columbus Day hisse piyasası tatili değildir. Henüz gerçekleşmemiş giriş/getiri üretilmez.

Phase 1 evreni: `output/discovery_v2/final_snapshots/20261010T201236851614Z`.
Evren SHA256: `fc9814a51b15f3424df87903869c5801f930803626be4fde0a23a905b6e88758`.
5.245 common-stock/ADR adayı: NASDAQ 3.382, NYSE 1.863. Market cap burada bir eligibility
ölçümü değildir; büyük şirketler de bulunur, güncel cap kapısı sonradan uygulanır.
Bu liste geçmiş tarihler için point-in-time evren değildir; delisted evreni değildir.

## 1. Mimari ve dosyalar

Yeni bağımsız dosyalar, `swing_trader/discovery/` altında:

| Dosya / fonksiyon | Sorumluluk |
|---|---|
| `calendar.py::last_completed`, `is_session` | 2025–2028 doğrulanmış cash-equity takvimi; ET/DST, erken kapanış, 30 dk veri yayın tamponu |
| `bars.py::pilot_records` | Borsa × eski likidite grubu; sabit hash sırası, UNKNOWN grubu, rastgele 300 kesmesi yok |
| `bars.py::YahooProvider`, `prepare_one`, `prepare` | Ayrı ilk yükleme/incremental güncelleme; en çok 2 worker; checkpoint/resume; kontrollü retry ve circuit breaker |
| `bars.py::normalize`, `rows_frame` | Seans/duplicate/NaN/OHLC zarfı/adjustment kontrolü ve production uyumlu DataFrame |
| `bars.py::store_object`, `read_object` | İçerik-hash kimlikli, silinmeyen/değiştirilmeyen JSON OHLCV revizyonları |
| `shadow.py::evaluate`, `scan`, `save_scan` | Ağ/DB kullanmadan champion tetikleri, aşama bazlı PASS/FAIL/UNKNOWN, karşılaştırma ve replay girdileri |
| `facts.py::collect`, `interpret_calendar` | Yalnız güncel tetikleyiciler için isteğe bağlı, sınırlı güncel cap/float/earnings gözlemi |
| `opportunity.py::register`, `measure`, `forward` | Değişmez gözlemler; örtüşme kümeleri; 5/10/20/30 seans next-open ölçüm altyapısı |
| `phase2.py::main` | Opt-in CLI; production import/otomatik job kaydı yok |
| `test_phase2.py` | Ağsız deterministik testler ve production parity kontrolleri |

Phase 1 `master.py`, `legacy.py`, `comparison.py`, `__main__.py` yeniden yazılmadı.
Yeni dependency eklenmedi. `dateutil`, pandas/numpy/yfinance zaten requirements içinde.

### Veri sözleşmesi

- Hisse/SPY ilk isteği 400 takvim günü: en büyük sonlu ihtiyaç SPY MA200 + 5 günlük
  rejim doğrulaması = **204 bar**. Hisse minimumu VCE 55, RVOL 52; ayarlanabilir hacim
  ve ATR lookback'i de minimum hesaplamasına katılır. Referans VIX için 30 takvim günü.
- EMA/MACD'nin matematiksel olarak sonlu bir maksimum geçmişi yoktur. Hisse taramasında
  production `period='6mo'` yaklaşımına karşılık **EOD+1 gün eksi 6 takvim ayı** penceresi
  sabitlenir. Fonksiyonlar aynıdır; başka gün indirilen farklı uzunluktaki bir production
  penceresiyle bit-bit aynı EMA çıktısı olduğu ileri sürülmez.
- Normal güncelleme eksik tail/iç boşluk + **2 overlap barı** ister. Bu iki bar gereksiz
  altı aylık tekrar değil, geçmiş düzeltmesini saptama kontrolüdür. Overlap değişirse veya
  yeni split/temettü gelirse tam revizyon indirilir. Önceki nesne korunur.
- Kayıt: symbol/vendor symbol, kaynak, fetch timestamp, istenen seans/başlangıç,
  son geçerli seans, adjustment modu, kurumsal aksiyonlar, kimlik metadata'sı ve barlar.
- `auto_adjust=False` yanıtındaki OHLC + Adj Close korunur; taramada `Adj Close / Close`
  oranı yalnız OHLC'ye uygulanır. Volume dividend-adjust edilmez. Bu, yfinance
  `auto_adjust` ile deterministik testte karşılaştırılmıştır. Yahoo'nun geçmiş OHLC/Volume
  split-adjustment'ı bağımsız bir kurumsal aksiyon kaynağıyla sertifikalanmış değildir.
- Duplicate veya sırasız seans, sonlu olmayan fiyat/hacim, negatif hacim, bozuk OHLC,
  şüpheli adjustment katsayısı karantinaya neden olur. Gerekli pencerede eksik seans
  varsa trigger `UNKNOWN` kalır. Sıfır hacim ise gerçek işlem yokluğu da olabilir:
  trigger aynı algoritmayla hesaplanır, ayrı `data_quality=UNKNOWN` uyarısı taşınır.
  Böylece ek bir sıfır-hacim trading filtresi icat edilmez.
- Yalnız tamamlanmış EOD barları kullanılır; kapanış + 30 dakika öncesinde son seans
  bir önceki seanstır. Bu tampon sağlayıcının veriyi kesinleştirdiğine dair SLA değildir.
- VIX bir hisse değildir. VIX'in cash-equity tatiline düşen referans satırları açıkça
  listelenerek hizalanır; NASDAQ/NYSE hisselerinde böyle satır kabul edilmez.
- Güvenli eşleştirme yapılamayan sınıf hisseleri silinmez: `MAPPING_REVIEW_REQUIRED`.
  Yahoo symbol/exchange/EQUITY/USD doğrulaması kalıcı security ID değildir. Ticker
  değişimi, aynı sembolün yeniden kullanılması, merger/SPAC geçmişi hâlâ tam çözülemez.
- Checkpoint'ler symbol sıralamasına ve plana bağlıdır; farklı planla resume reddedilir.
  Tamamlanmış bir run aynen döner; başarısız run yeniden denenmek istenirse yeni run dizini
  ve `--previous` kullanılır. 403/429 sonrası aynı run içinde devam/engel aşma yapılmaz.

## 2. Champion ile gerçek ilişki

### Değişmeyen strateji

`small_cap/signals.py::check_all_triggers`, `check_vce_breakout`, `check_rvol_thrust`
doğrudan çağrılır. Yeni trigger veya gevşetilmiş eşik yok.

- VCE: ATR14% squeeze oranı <0,8; önceki 20 günlük tepe üstü; yeşil kapanış; MA50
  üstü; bugünü içeren 50 günlük hacim ortalamasına göre RVOL ≥1,5. Premium/tight-coil
  işaretleri strateji kapısını gevşetmez.
  **Kod/doküman çelişkisi:** `signals.py::check_vce_breakout` docstring'i hacmin
  sinyali engellemediğini söylüyor; aynı fonksiyonun aktif `rvol50 < VCE_MIN_RVOL_GATE`
  koşulu gerçekten reddediyor. Bu rapor docstring'i değil çalışan koşulu esas alır.
- RVOL: 2,5 ≤ RVOL <4, pozitif günlük değişim <%8, MA20 üstü; VCE varsa o önceliklidir.
  Trigger RVOL'u bugünü içeren **ortalama50** hacme dayanır; scoring'deki
  `calculate_volume_surge` önceki50 günün **medyanını** kullanır. Aynı metrik sayılmaz.
  52–54 barlık IPO'da RVOL değerlendirilebilir; VCE'nin yeterli veri durumu UNKNOWN
  kalır. VCE'nin55 bar şartı tüm hisseye ortak bir ret olarak uygulanmaz.
- `check_boosters` / `check_swing_confirmation`, RSI (VCE muaf),
  `patterns.py::detect_weinstein_stage`, `engine.py::_classify_swing_type`,
  `scoring.py::calculate_quality_score` aynı production fonksiyonlarıdır.
- Rank = Quality + premium VCE için 8 + 0<squeeze<0,65 için 5. Kalite kapısında Rank kullanılmaz.
- Rejim `regime_logic.py::regime_from_spy_close`; SPY204 ve tamamlanmış VIX gerekir.
  Yetersiz rejimde floor sıfıra indirilmez, kalite `UNKNOWN` olur.
- Kalite minimumu `thresholds.py::effective_scan_thresholds`; explicit patch'teki
  `auto_scan.min_quality=60`, `top_n=15` ve mevcut regime floors kullanılır; scheduler
  çağrılmaz. Manuel API body varsayılanı 65/top10'dur, aynı ayar sanılmaz. Mevcut rejim
  tabanları bunların üstündedir: 78/80/82; BULL efektif Q78/top10.
  Broad discovery çıktısı top10 ile kesilmez; top-N bir üretim/gösterim kısıtı olarak
  ayrıca kayıtlıdır, risk/portföy veya işlem seçimi burada yapılmaz.

### Bilinmeyen veri = uygun hisse değildir

`filters.py::check_market_cap` bilinmeyeni "pre-screened" varsayımıyla production'da
kabul eder. `check_earnings` hata/boş sonuçta kabul edebilir. Finviz'siz evrende bu
varsayımlar taşınmaz: market cap/earnings ayrı `UNKNOWN`; trigger hesaplanabilir ama
tam trade eligibility ileri sürülemez.

Float da kritiktir: production `score_float_tightness` bilinmeyene sayısal varsayılan
verir. Shadow aynı formülle **diagnostic Quality/Rank** üretir; float bilinmiyorsa
kalite kapısı `UNKNOWN` kalır. Tetikleyici adaylarda olası float bantlarının verdiği
skor aralığı ayrıca kayıtlıdır. Bu belirsizlik aralığı yeni optimize edilmiş strateji
değil, mevcut formülün eksik girdiye duyarlılığıdır.

Hard gates `filters.py`: fiyat $7–$200, 20 günlük `Close*Volume` ortalaması ≥$5M,
market cap $250M–$10B, bilanço signal day dahil sonraki 0..3 takvim gününde olmamalı.
Float ve ATR bir hard gate değildir. Metadata yalnız gözlem timestamp'i karar
timestamp'inden geç değilse ve aynı seansa açıkça bağlanmışsa kullanılabilir.

Etkin patch DB'den salt-okunur alındı: son güncelleme `2026-09-09T16:43:19.942155Z`.
Patch içindeki kaldırılmış `universe_filters.max_float_shares` alanı, production'ın
`settings_config.py::_prune_removed_keys` fonksiyonuyla temizlenir; tüm ayarların
sessizce varsayılanlara dönmesine izin verilmez. Temizlenmiş tam config snapshot'a girer.

### Legacy preselection hâlâ kapsamı daraltıyor

`universe.py::get_finviz_universe`: USA, small $300M–$2B / mid $2B–$10B,
avg share volume small300K/mid500K, price>$7; iki VCE sorgusu MA50 + yeni20g tepe,
iki RVOL sorgusu RelVol>2 + yeşil + MA20. Ardından dedup → 33 hardcoded exclusion
→ $6–$250 post-price → güncel fiyat×hacim sıralaması → 300 tavanı. Dar sonuçta static
merge/fallback vardır. Broad bu discovery kısıtlarını kullanmaz; trading gates korunur.

Bu fark, yeni evrendeki her ek sembolün kaliteli fırsat olduğu anlamına gelmez.
Kayıtlarda query öncesi tüm-popülasyon sayıları olmadığından her Finviz predicate'in
kaç gerçek fırsat kaybettirdiği bu turda ölçülemez.

## 3. Gerçek çalıştırmalar ve ölçümler

### Veri toplama: pilot → 800 → 5.245

Tüm sayılar **2026-10-09 EOD** içindir; evren paydası 5.245 hisse, SPY/VIX dahil değildir.
Kaynaklar `output/discovery_v2/phase2/` altındaki plan/preparation/scan snapshot'larıdır.

| Aşama | İstenen hisse | Güncel verili | Yeterli/kesintisiz | Benzersiz trigger | Veri hazırlama süresi |
|---|---:|---:|---:|---:|---:|
| `pilot_network` / `pilot_verified` | 240 | 239 | 238 | 0 | 246,0 sn |
| `expansion` / `expansion_verified` | 800 | 797 | 794 | 1 | 280,4 sn |
| `full` / `full_raw_scan` | 5.245 | 5.221 | 5.141 | 19 | 2.289,4 sn |

Kademeler birbiriyle örtüşür; sayılar toplanarak farklı hisseler gibi gösterilmez.
800 aşamasındaki tek tetikleyici SNOW idi; fiyatı $200 tavanının üstündeydi.
Pilot sonucu tüm piyasaya genellenmedi; tam çalışma ayrıca gerçekleştirildi.

| Tam yükleme sonucu | Sayı / açıklama |
|---|---|
| Master candidates | 5.245: NASDAQ3.382 / NYSE1.863 |
| Kademeler boyunca history isteği denenmiş unique hisse | 5.233; mapping12 için istek yapılmadı |
| Veri mevcut ve son seans güncel | 5.221; **%99,54** |
| Trigger için yeterli/kesintisiz OHLCV | 5.141; **%98,02** |
| Yeterli veri, borsa kırılımı | NASDAQ3.303 / NYSE1.838 |
| Barı kısa | 77; yeni IPO/az geçmiş olabilir, otomatik kötü hisse değil |
| Gerekli lookback'te boşluk | 3 |
| Cache nesnesi yok | 24 = 12 mapping + 12 başarısız/karantina |
| VCE / RVOL / unique | 6 / 14 / **19**; bir hissede iki tetik var, çift sayılmadı |
| Raw scanner süresi | 125,656 sn; metadata çağrısı ve indirme hariç |
| İlk full genişletme | 4.448 mantıksal history çağrısı, 12 retry; erişim circuit'i açılmadı |
| İlk iki aşamadan tekrar kullanılan | 799 cache hit: 797 hisse + 2 referans |
| Full yeniden çalışma (`warm_cache`) | **5.218 cache hit**, yalnız 29 history çağrısı, 12 retry; 93,625 sn |
| Nakit veri ücreti | $0; internet/CPU/disk maliyeti ölçülmedi |

Güncel verili kümenin borsa dağılımı NASDAQ3.370 / NYSE1.851; yeterli-geçmişli
altküme tablodaki3.303 /1.838'dir. Common/ADR etiket dağılımı4.927 /318'dir.

Warm run'da 5 kısmi seri ve 12 veri-hatalı kayıt tekrar denendi; kapsam artmadı.
Mapping belirsiz12 kayıt için istek yapılmadı. 5.218 hit'in 2'si referans seridir.
Bu gerçek ölçüm, aynı tamamlanmış seans için bütün evrenin altı aylık verisinin tekrar
indirilmediğini gösterir. **Ertesi yeni seans incremental akışı ise henüz takvimde
gerçekleşmedi**; eksik gün/overlap/split senaryoları deterministik testte doğrulandı.

Veri karantinası: NONFINITE_OHLCV **LRHC, GTBP, BTLN, IMMP, TNMG, ALP, UCAR, BRTX,
SFWL, WCT**; NONPOSITIVE_PRICE_OR_NEGATIVE_VOLUME **GMEX**; YFTzMissingError **SVA**.
Son hata delisting kanıtı değildir. PARTIAL: **WORX, ATLQ, FISV, ATTT, NFE**;
kısmi5 serinin tamamı aynı düzeyde kullanılamaz değildir: scanner gerekli pencereye bakar.
Mapping: **CRD.A, CRD.B, HEI.A, HVT.A, BRK.A, BRK.B, UHAL.B, CIG.C, AGM.A, MKC.V,
AKO.A, AKO.B**; noktalı ticker'ı körlemesine tireye çevirerek başka security indirilmedi.

`fresh_by_exchange` JSON alanının bu sürümdeki anlamı **yeterli/kesintisiz OHLCV**
kırılımıdır; sadece son barı bulunan daha geniş5.221 kümenin borsa sayısı değildir.
Stage retleri birbirinden bağımsız/nonexclusive'dir; toplanıp toplam elenen denmez.

Kontrollü pilot **240 aday + SPY/VIX**; NASDAQ128 / NYSE112. Seçim: eski Mayıs cache'i
üzerinden <5M / 5–25M / ≥25M günlük dolar-hacim ve UNKNOWN grupları. Eski likidite güncel
eligibility değildir. Strata sayıları `phase2/pilot_plan.json` içinde; pilot piyasayı
temsil eden bir kârlılık örneklemi değildir.

İlk sandbox denemesi: 5 sembolde bağlantı/timezone hatası, kalan237 kayıt denenmedi.
Bu çevresel başarısızlık başarılı provider denemesiyle birleştirilmedi. Ağ izinli pilot
ayrı `pilot_network/` altında. VIX takvim hizalama kontrolü bu pilotta yakalanıp yalnız
research referans veri katmanında giderildi; eski başarısız artefakt silinmedi.

### Nihai 9 Ekim sonucu: metadata zenginleştirilmiş snapshot

Kaynak: `full_final/scan.json`, karar zamanı **2026-10-10T22:08:47.095848Z**;
tarama120,156 sn. Sinyal tarihi sütunu kullanılan EOD bardır; sinyalin gerçekten
9 Ekim kapanışında üretildiği iddia edilmez. Şirket bilgisi10 Ekim'de gözlendi,
o günkü karar için geçerlidir; 9 Ekim kapanışında biliniyormuş gibi PIT geri-test yapılmaz.

- 19 unique trigger; market cap/float19'unda biliniyor, ileri earnings7'sinde biliniyor.
  `current_facts.json`: **38 mantıksal metadata isteği**, kaydedilmiş exception0,
  boş/eski earnings12, retry0. Boş veri HTTP başarı olsa bile tam bilgi değildir.
- **Hard gate'lerin dördünü birlikte geçen2: MNSO ve PBT.** İkisi de Weinstein stage3
  nedeniyle reddedildi. MNSO ayrıca swing confirmation ve Quality kapılarını geçmedi.
- **Nihai PASS0.** Tetikleyen19'un tamamında en az bir kanıtlanmış ret var.
  Earnings UNKNOWN olan12 hisse yalnız eksik bilgiden dolayı beklemiyor;
  zaten bilinen başka kapılardan eleniyor. Hiçbiri işlem önerisi değil.
- **Tam aşama değerlendirmesi7/5.245 = %0,1335**; tüm aşamaları gözlenen tetikleyiciler.
  Tetiklemeyen5.122 hisseye binlerce ek fundamentals isteği yapılmadı; bu nedenle
  %99,54 OHLCV freshness, %99,54 tam trade eligibility coverage demek değildir.
  5.141 conclusive FAIL / 104 veri nedeniyle UNKNOWN; PASS yok.
- 19 tetikleyicinin15'i swing/RSI/stage/quality kapılarından en az birinde FAIL.
  Quality tek başına10 ret; hard gate retleriyle örtüşür, sayılar toplanmaz.
- Rejim BULL/CONFIRMED: SPY778,57, MA50=766,28, MA200=719,66, VIX14,84.
  Efektif Q tabanı78; threshold değişmedi. Tam config `full_final/inputs.json` içinde.

| Aşama | PASS | FAIL | UNKNOWN |
|---|---:|---:|---:|
| Universe | 5.245 | 0 | 0 |
| OHLCV yeterlilik/güncellik | 5.141 | 0 | 104 |
| Veri kalitesi ek kontrolü | 4.709 | 0 | 536 |
| Provider kimlik eşleşmesi | 5.140 | 1 | 104 |
| Fiyat | 3.121 | 2.020 | 104 |
| 20g dolar hacim | 2.942 | 2.199 | 104 |
| Market cap | 7 | 12 | 5.226 |
| Earnings | 7 | 0 | 5.238 |
| VCE | 6 | 5.129 | 110 |
| RVOL | 14 | 5.127 | 104 |
| Swing confirmation | 1.159 | 3.982 | 104 |
| RSI | 4.808 | 333 | 104 |
| Weinstein | 1.287 | 3.854 | 104 |
| Quality | 9 | 10 | 5.226 |

Data-quality UNKNOWN536 = değerlendirilemeyen104 + kullanılabilir barları olup
ek uyarı taşıyan432. Bu uyarılar trigger formülünü sessizce değiştirmedi.
Kimlik çelişkisi **LUMN**: master resmî kaydıNASDAQ, Yahoo `exchangeName=NYQ`.
İkisinden biri doğru varsayılıp geçilmedi; olası borsa geçişi/provider gecikmesi
bu turda kesinleştirilemedi. Bu bir veri kimliği bulgusu, fiyat yönü tahmini değil.

### 9 Ekim'deki 19 trigger gözlemi — tavsiye değildir

Tümü `2026-10-09` barında; **legacy üyeliği tümünde UNKNOWN**. Yeni/legacy-dışı olduğu
kanıtlanmış19 hisse listesi değildir. Q/Rank güncel gözlenen float ile, değişmeyen
production formülüyle hesaplandı. AUC Q100,60 geçerlidir: efektif `final_score_max=140`,
`bonus_cap=30`; skor bir yüzde veya kazanma olasılığı değildir.

Kısaltmalar: F=fiyat, DV=dolar hacim, MC=market cap, S=swing, R=RSI, W=Weinstein,
Q=quality. Ret sütunu bütün bilinen retleri; U sütunu bütün ek eligibility
belirsizliklerini gösterir (E=earnings, D=veri kalitesi). Listelenmeyen kapılar PASS.
VCE/RVOL alternatifleri OR'dur; seçilmeyen trigger'ın FAIL'i ret değildir.

| Ticker | Setup | Ref.close $ | Q / Rank | DV M$ | MC M$ | Bilinen ret | U |
|---|---|---:|---:|---:|---:|---|---|
| PSIG | VCE + RVOL | 5,10 | 96,87 / 109,87 | 7,40 | 78,44 | F, MC | E |
| AUC | RVOL | 9,19 | 100,60 / 100,60 | 0,05 | 203,36 | DV, MC | D, E |
| BRBI | RVOL | 15,34 | 90,15 / 90,15 | 0,01 | 402,66 | DV, R | D, E |
| SVAQ | VCE | 10,18 | 82,15 / 90,15 | 1,35 | 298,49 | DV | D, E |
| BID | RVOL | 9,92 | 83,60 / 88,60 | 0,37 | 191,22 | DV, MC, W | D, E |
| PAMT | RVOL | 12,78 | 84,71 / 84,71 | 0,30 | 267,64 | DV | E |
| SLSN | VCE | 1,02 | 70,83 / 83,83 | 0,03 | 72,06 | F, DV, MC, W, Q | — |
| NDLS | RVOL | 14,22 | 82,37 / 82,37 | 0,46 | 84,85 | DV, MC, S, W | — |
| PBT | RVOL | 34,86 | 82,10 / 82,10 | 7,06 | 1.624,78 | W | — |
| XBIT | VCE | 2,29 | 73,75 / 81,75 | 0,08 | 69,82 | F, DV, MC, W, Q | E |
| BIII | RVOL | 10,07 | 74,75 / 79,75 | 0,42 | 231,61 | DV, MC, R, W, Q | D, E |
| HCAC | RVOL | 10,15 | 79,15 / 79,15 | 0,85 | 296,35 | DV, R | D, E |
| SYPR | VCE | 2,06 | 68,51 / 76,51 | 0,04 | 47,43 | F, DV, MC, Q | — |
| GTERA | RVOL | 10,50 | 70,95 / 70,95 | 0,82 | 249,13 | DV, MC, R, Q | D, E |
| LPCV | RVOL | 10,08 | 65,35 / 70,35 | 0,39 | 289,80 | DV, S, Q | D, E |
| TLSA | RVOL | 1,19 | 63,88 / 63,88 | 0,81 | 151,44 | F, DV, MC, R, Q | E |
| SNOW | VCE | 368,89 | 50,47 / 58,47 | 1.417,26 | 130.144,40 | F, MC, Q | — |
| TEVA | RVOL | 41,10 | 56,60 / 56,60 | 231,65 | 47.934,62 | MC, Q | — |
| MNSO | RVOL | 9,60 | 53,20 / 53,20 | 8,28 | 2.806,05 | S, W, Q | — |

Security-type uyarısı: BRBI/MNSO name-evidence ADR; SVAQ/BID/BIII/HCAC/GTERA/LPCV
name-evidence suspected SPAC/classA. IPO/SPAC/ADR bilgisi kanıt yoksa UNKNOWN
kalır; yanlış biçimde "kesin değil" yapılmaz. Tüm sınıflandırmalar name heuristic.
Resmî dosya PBT'yi "Permian Basin Royalty Trust Common Stock" diye adlandırıyor:
common-stock etiketi işletme şirketiyle ekonomik eşdeğerlik garantisi değildir.
Bu tur Phase 1 sınıflandırması veya production eligibility değiştirilmedi.
Tam security flags, trigger metrikleri, bütün aşamalar ve bar nesnesi kimliği
her satırda `full_final/scan.json` içinde mevcuttur.

## 4. Same-session comparison ve production 403 kanıtı

Salt-okunur son60 production run sorgusunda en son run **131**, oluşturma
`2026-10-09T02:29:07.198174Z`, **bar_session=2026-10-08**, seçilen57 ve verili57.
9 Ekim EOD legacy listesi yok. Dolayısıyla 9 Ekim broad-only / overlap / legacy-only
kesin sayıları **UNKNOWN_INCOMPLETE**, sıfır değildir. Static liste Finviz yerine konmaz.

8 Ekim için ayrı **HISTORICAL_RESEARCH** karşılaştırması yapılabilir: aynı bar kesimi ve
kayıtlı57 membership. Ancak bugünkü evren / sonradan indirilmiş fiyatlar kullanılır;
PIT veya gerçek zamanlı kaçırılmış kazanç kanıtı değildir. Eksik OHLCV olan legacy
sembolü "legacy-only sinyal" diye etiketlenmez; `legacy_data_unknown` grubuna girer.

Bu karşılaştırma çalıştırıldı: `full_oct08_research/scan.json`, 121,625 sn.
57 legacy üyenin **57'sinde de** yeterli veri vardı; toplam master'da104 aday
değerlendirilemediğinden global durum `INCOMPLETE_DATA_COVERAGE` olarak kaldı.
24 broad trigger = **19 broad-only + 5 overlap**. Overlap: **DXC, HNRG, LW, NTLA, RXO**.
Legacy üyelerinin52'si bu offline EOD rekonstrüksiyonunda trigger vermedi; bunlar
"52 legacy-only başarılı sinyal" değildir. Legacy listesi preselection, broad listesi
trigger listesidir; iki farklı aşamanın karşılaştırıldığı özellikle kaydedilir.

### 8 Ekim'de legacy dışında bulunan 19 tarihsel gözlem

Hepsinin seansı **2026-10-08**, kaydı sonradan oluşturulmuş araştırma cohort'udur.
**Hepsinde tarihsel market cap, earnings ve float UNKNOWN**; Q/Rank float varsayılanlı
diagnostic skordur, trade eligibility değildir. Bugünkü şirket bilgisi geçmişe yazılmadı.
Tabloda yazılmayan fiyat/DV/RSI/swing/stage kontrolleri PASS; veri uyarısı ayrıca gösterilir.
Tüm satırlarda universe/OHLCV/provider identity PASS, karşılanmayan diğer kapılar JSON'da.

| Ticker | Setup | Close $ | Q / Rank | Dolar hacim M$ | Bilinen ret / ek belirsizlik |
|---|---|---:|---:|---:|---|
| CD | RVOL | 10,91 | 91,89 / 91,89 | 1,50 | DV, RSI |
| SMXT | RVOL | 4,66 | 91,61 / 91,61 | 1,05 | Fiyat, DV, RSI |
| JDZG | RVOL | 5,63 | 76,93 / 81,93 | 5,23 | Fiyat, RSI; veri kalitesi UNKNOWN |
| AFRI | VCE | 12,50 | 78,25 / 78,25 | 0,35 | DV |
| OTAI | RVOL | 10,09 | 72,25 / 77,25 | 0,05 | DV, RSI |
| PTC | RVOL | 193,73 | 75,49 / 75,49 | 671,79 | RSI |
| OLB | VCE | 0,55 | 70,25 / 75,25 | 28,76 | Fiyat |
| GFR | RVOL | 6,26 | 74,29 / 74,29 | 8,08 | Fiyat, stage |
| BAK | RVOL | 1,82 | 73,25 / 73,25 | 2,96 | Fiyat, DV, swing, stage |
| PSO | VCE | 16,63 | 65,25 / 73,25 | 11,15 | Stage |
| ACGC | RVOL | 10,07 | 71,65 / 71,65 | 0,49 | DV; veri kalitesi UNKNOWN |
| AUC | RVOL | 9,15 | 71,45 / 71,45 | 0,04 | DV; veri kalitesi UNKNOWN |
| QNBC | RVOL | 46,15 | 67,65 / 67,65 | 0,86 | DV; veri kalitesi UNKNOWN |
| UTZ | VCE | 14,32 | 59,45 / 67,45 | 23,90 | Bilinen ret yok; cap/earnings/quality UNKNOWN |
| RCMT | RVOL | 41,85 | 63,52 / 63,52 | 3,30 | DV, swing, RSI |
| FLYE | VCE | 2,32 | 63,29 / 63,29 | 15,58 | Fiyat, stage |
| FACT | RVOL | 10,74 | 62,85 / 62,85 | 0,20 | DV, RSI; veri kalitesi UNKNOWN |
| GFS | RVOL | 49,36 | 61,16 / 61,16 | 196,06 | Bilinen ret yok; cap/earnings/quality UNKNOWN |
| XCBE | RVOL | 10,03 | 57,25 / 57,25 | 0,24 | DV, swing, RSI, stage; veri kalitesi UNKNOWN |

UTZ/GFS'nin eksik float'a bağlı Q aralıkları sırasıyla43,20–78,20 ve44,91–79,91;
efektif floor78. Varsayılan skoru düşük diye otomatik kesin ret yazılmadı;
aynı nedenle bu ikisinin gerçekten kalite kapısını geçtiği de söylenemez.
Bu liste sıralama/coverage örneğidir; gelecekteki kazanca göre seçilmedi.

Phase 1'in aynı gün yerel ağdan dört Finviz sorgusu `HTTPError` ile başarısızdı;
`legacy_20261010.json` içinde HTTP status alanları boş. Ayrı Phase 1 HTTPS probe'unda
403 görülmüştü. Dört sorgunun tamamının ölçülmüş status'unu 403 diye göstermiyoruz.
Production run istatistikleri query-başına HTTP durumunu saklamıyor. Son başarılı
rank telemetry/membership daha eski seansa ait. **Yerelde 403 kanıtlı; production'da
aynı 403'ün gerçekleştiği kanıtlanamadı.** Yeni bir production scan tetiklenmedi.

## 5. Opportunity tracking — getiriler henüz yok

`opportunity.py` tüm broad trigger gözlemlerini (reddedilenler ve UNKNOWN dahil)
kayda alır; yalnız kazananlar seçilmez. Legacy membership UNKNOWN ise broad-only
olduğu iddia edilmez. Aynı ticker/seans/cohort tekrar kaydı çoğaltmaz. Aynı ticker'ın
30 seanslık birincil gözlem penceresindeki tekrarlar aynı bağımlılık kümesine alınır.

- Signal day: EOD; referans giriş: **bir sonraki planlı cash seansının Open'ı**.
  Eksik giriş barı varsa bir sonraki mevcut bara atlanmaz.
- 5/10/20/30 horizon: giriş seansı gün1, ilgili günün Close'ı / giriş Open'ı −1.
- MFE: o aralıktaki en yüksek High / giriş Open −1 (en az0); MAE: en düşük Low /
  giriş Open −1 (en çok0). Bunlar uygulanmış exit / realized PnL değildir.
- Barların tamamı aynı corporate-action-adjusted cache revizyonundan alınır.
  Dividend-adjusted oran bir getiri proxy'sidir; broker nakit akışı değildir.
- Maliyet senaryoları toplam20/50/100 bps; yarısı entry, yarısı exit fiyatına
  olumsuz uygulanır. Gerçek spread, market impact, komisyon tarifesi veya manuel
  fill garantisi değildir. Likidite bilgisi gözlemde saklanır.
- Sonraki açılıştan sonra eklenmek istenen kayıt `PROSPECTIVE_SHADOW` kabul edilmez.
  Eski araştırma cohort'u ve ileriye dönük cohort ayrı tutulur.
- Henüz olgunlaşmamış horizon `PENDING`; eksik seans `MISSING_DATA`; getiri uydurulmaz.

Gerçek komut çalıştırması: `opportunities_current.json` içinde **19 PROSPECTIVE_SHADOW**
gözlemi, `forward_current.json` içinde **76 PENDING / 0 MATURE** horizon.
İlk olgunlaşma seansları sırasıyla **16 Ekim, 23 Ekim, 6 Kasım, 20 Kasım 2026**.
`opportunities_oct08_research.json` ayrı24 HISTORICAL_RESEARCH gözlemi içeriyor.
İki cohort birleştirilmedi. Örtüşme kontrolü kod/test düzeyinde; tek günün19 ayrı
ticker'ı gelecek günlerle ilişkilendirilene kadar19 bağımsız başlangıç gözlemidir.

Coverage benchmark A: tamamlanmış seans için güncel barlı / master5.245 ve
yeterli-geçmişli / master5.245 oranları birlikte raporlanır. Metadata tamlığı ve
sonuçlanan eligibility ayrı paydalardır; yalnız sinyal üretenleri paydaya almak yoktur.
Opportunity benchmark B: kayıt anında sabitlenmiş legacy-NO cohort'unda bütün iyi/kötü
sonuçlar, aynı next-open ve maliyet koşullarında5/10/20/30 seans ölçülür. Örtüşen kayıtlar
bağımsız örnek sayılmaz; ticker/karar-seansı kümelenmesi korunur. Bugün prospective
legacy-NO doğrulanamadığı için **B benchmark'ının kazanç sonucu henüz yoktur**.

## 6. Ücretsiz kaynaklar ve ölçek sınırları

Yahoo/yfinance yalnız bu izole araştırma için kullanıldı; ücretli endpoint/fallback
ve API anahtarı kullanılmadı. Kod en çok2 worker ve global ≥0,5s history-call pacing
uygular; bu **Yahoo tarafından ilan edilmiş bir quota/SLA iddiası değildir**. İç auth/
timezone istekleri yfinance içinde olabilir; rapordaki çağrı sayısı mantıksal history
çağrısıdır, toplam HTTP sayısı değildir. 401/403/429 durdurur, diğer hatalarda en çok bir
retry/5s backoff; peş peşe5 çağrı başarısızlığında tarama durur. Proxy rotasyonu veya
kısıt aşma yoktur. [yfinance kullanım uyarısı](https://github.com/ranaroussi/yfinance),
[history API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html).

Tiingo starter/trial koşullarının §1.6(a) metni kalıcı veri ve türev sonuç depolamayı
kısıtlıyor; bu nedenle kalıcı ücretsiz cache için kullanılmadı.
[Tiingo güncel koşullar](https://app.tiingo.com/tos/).
Finnhub ücretsiz daily-candle erişim/retention hakkı bu ortamda doğrulanamadı;
production kodundaki eski 60req/min varsayımı erişim yetkisi kanıtı sayılmadı.
[Finnhub fiyatlandırma](https://finnhub.io/pricing).

Resmî takvim dayanağı: [NYSE](https://www.nyse.com/trade/hours-calendars),
[Nasdaq Trader](https://www.nasdaqtrader.com/Trader.aspx?id=Calendar).
2025 ulusal yas günü ([Nasdaq resmî bildirimi](https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2025-1))
ve 2027-12-31 New Year istisnası research takviminde ayrıca
ele alındı. 2028 sonrası takvim otomatik tahmin edilmez; doğrulanmış aralığın dışı hata.

Ücretsiz kaynaklarla bu turda güvenilir biçimde çözülemeyenler: delisted/PIT evren,
tarihsel float/market cap ve earnings announcement timestamp'i, eksiksiz rename/security
ID zinciri, kurumsal aksiyonların bağımsız doğrulanması, historical spread/depth/impact,
manuel gerçek fill. Yahoo'ya bugün erişebilmek, ticari üründe yeniden dağıtım veya
kalıcı ücretsiz tam-piyasa hizmet garantisi vermez.

Ölçülen disk: warm kontrolünden sonra5.228 immutable bar nesnesi, **394,80 MiB**
(scan/input/source paketleri hariç). Her değişen seri tam yeni JSON revizyonudur;
snapshot replay korunur fakat günlük disk büyümesi henüz uzun dönem ölçülmedi.
Tüm5.245 hissede her yeni gün en az bir history çağrısı gerekirse0,5s pacing tek başına
yaklaşık44 dakika alt-sınır demektir; incremental bar indirmek request sayısını
kendi başına azaltmaz. Bu geleceğe ait süre garantisi değil, mevcut istemcinin
matematiksel ölçek sınırıdır. İzinli toplu EOD kaynağı/lisansı doğrulanmadan gerçek
ürün SLA'sı vaat edilmemeli. Tek full scanner sürecinde yaklaşık1 GiB RAM görüldü;
bu bir peak bellek benchmark'ı değildir. Persistent cache hit ve immutable replay
çalıştı; günlük yeni seans, uzun dönem disk/sağlayıcı istikrarı henüz saha testine muhtaç.

Production `data/fetcher.py` ve Tiingo/Finnhub/cache yolları değiştirilmedi. Yeni katman
aynı pandas OHLCV arayüzünü ve yfinance'ı kullanır; eski pickle formatını üretim cache'ine
yazmaz. Pickle'ın adjustment/fetch/PIT metadata'sı yeterli olmadığı için yeni kalıcı
nesne sözleşmesi ayrı tutuldu. Eski cache yalnız opt-in pilot likidite örneklemesinde okundu.
İki overlap barından uzakta kalan sessiz tarihsel fiyat düzeltmesi bu tasarımla her zaman
saptanamaz; periyodik bağımsız reconciliation henüz yapılmış/garantilenmiş değildir.

## 7. Testler ve tekrar üretme

Çalıştırıldı: **481 test geçti** = 55 Phase 2 + 33 Phase 1 + 393 diğer mevcut
backend/trading testi. Komut:

```powershell
python -m pytest tests swing_trader/discovery swing_trader/tests api/tests -q -p no:cacheprovider --basetemp output/discovery_v2/phase2/tests-complete --junitxml output/discovery_v2/phase2/test_results_complete.xml
python -m compileall -q swing_trader/discovery
```

JUnit kanıtı `output/discovery_v2/phase2/test_results_complete.xml`. Protobuf/joblib ve pytest
async fixture ayarı için mevcut deprecation uyarıları var; başarısız test yok.
Bu sayı frontend/e2e veya canlı broker entegrasyon testi anlamına gelmez.
Son teslim kontrolü aynı481 testi tekrar çalıştırdı: **481 passed, 3 warnings,
9,14 sn**; `test_results_handoff.xml` (basetemp `tests-handoff`). Compileall da başarılı.

Yeni testler: DST/erken kapanış; tamamlanmamış bar; NaN/duplicate/OHLC doğrulaması;
adjustment ve değişmeyen dividend-volume; split/reverse-split rebase; incremental gap
repair; cache hit; yarıda kalmış checkpoint resume; 403/rate-limit circuit; secret
maskeleme; 52 barlık RVOL; VCE/RVOL birlikte tetiklenme ve öncelik; RSI muafiyeti;
gerçek engine ile Quality/Rank/type parity; UNKNOWN float/cap/earnings; gelecekteki
bar ve metadata'nın dışlanması; checksum/source bundle; next-open/horizon/maliyet;
örtüşen gözlemlerin tek bağımsız kümeye alınması.

Pilot ve800-aday snapshot'ları kendi kaydedildikleri kod sürümünde frozen inputs ile
ağsız tekrar çalıştırıldı, sonuç eşitliği doğrulandı. Sonraki401 hata sınıflandırma
korumasından sonra nihai tam-evren snapshot'ı güncel kodla üretildi ve **tam replay
birebir eşit** bulundu. Eski pilot kod hash'i güncelmiş gibi gösterilmiyor.

```text
full_final scan_id:
1e6246a98f90665e8412e7d0b253ac164e4e8b0a150e4097d933a1f4dd95e48c
replay identical: true
result_sha256:
dd46588df911a7ca38390f4c4e2d719a0e931c239f2c5561c7fadfb794cc7366
```

Tam ilk yükleme işlemi başladıktan sonra research tarafında hata-maskeleme ve
cache-integrity korumaları güçlendirildi; süreç yüklendiği sürümle tamamlandı.
Güncel sürüm ayrıca `warm_cache` gerçek çalışması ve481 testle doğrulandı. Yüklemede
HTTP401/403/429 yaşanmadığından bu branch'lerin gerçek sağlayıcıda tetiklendiği
iddia edilmez; circuit davranışı deterministik testlerle sınandı.

Runtime: Python3.12.6, pandas2.3.3, numpy2.5.3, yfinance1.3.0; tam sürüm listesi
snapshot `scan.json → inputs.runtime` alanında. Snapshot kaynak-kod kopyaları `source/` altında;
manifest tüm girdileri ve kod kopyalarını hash'ler. API ayar anahtarları/credentials
bu pakete alınmaz. Artifact'lar gitignore edilen `output/` altında yereldir.

Örnek CLI (paths çalışma köküne göre; çıktı dizinleri yeni olmalı):

```powershell
python -m swing_trader.discovery.phase2 --help
python -m swing_trader.discovery.phase2 plan --snapshot output/discovery_v2/final_snapshots/20261010T201236851614Z --count 0 --output output/discovery_v2/next_plan.json
python -m swing_trader.discovery.phase2 prepare --plan output/discovery_v2/next_plan.json --cache output/discovery_v2/phase2/cache --run output/discovery_v2/next_run --previous output/discovery_v2/phase2/full/preparation.json --workers 2 --min-interval 0.5
```

`scan` ağsızdır; explicit settings/legacy/facts alır. `replay` frozen inputs, immutable
bar nesneleri, config/code/runtime sürümlerini doğrular. Kod/girdi değişirse mismatch
verir; bugünkü fiyat sağlayıcısına tekrar bağlanıp eski sonucu benzetmeye çalışmaz.

Bu çalışmayı ağsız doğrulama:

```powershell
python -m swing_trader.discovery.phase2 replay --scan output/discovery_v2/phase2/full_final/scan.json --inputs output/discovery_v2/phase2/full_final/inputs.json --cache output/discovery_v2/phase2/cache
```

Gözlem oluşturma ve gelecekte ölçme komutları (aynı çıktı adı tekrar yazılmaz):

```powershell
python -m swing_trader.discovery.phase2 track --scan output/discovery_v2/phase2/full_final/scan.json --cohort PROSPECTIVE_SHADOW --output output/discovery_v2/phase2/another_observation_book.json
python -m swing_trader.discovery.phase2 forward --opportunities output/discovery_v2/phase2/opportunities_current.json --preparation output/discovery_v2/phase2/full/preparation.json --cache output/discovery_v2/phase2/cache --output output/discovery_v2/phase2/another_forward_measurement.json
```

Prospective kayıt komutu12 Ekim açılışından sonra bu seans için bilerek reddedilecektir;
mevcut kayıt kitabı korunmalı, sonraki seanslar `--previous` ile append edilmelidir.
Gelecek sonuç ölçümünde `--preparation` yeni tamamlanmış seansa ait olmalıdır;
buradaki9 Ekim preparation'ıyla her zaman PENDING kalması doğrudur.

## 8. Kanıt / hipotez / eksik veri ve sonraki adım

### Kanıtlananlar

1. Finviz'siz5.245 adayın5.141'inde mevcut VCE/RVOL algoritmaları gerçekten çalıştırıldı.
   300 kesmesi yok; %98,02 kullanılabilir OHLCV coverage ölçüldü. Kaynak:
   `bars.py::prepare`, `shadow.py::scan`,9 Ekim full snapshot.
2. Güncel19 trigger'dan **0 nihai aday** çıktı. Daha büyük evren otomatik daha kaliteli
   veya daha kârlı sinyal değildir. PBT/MNSO hard gate PASS, downstream FAIL; kaynak
   `shadow.py::evaluate`, `patterns.py::detect_weinstein_stage`,9 Ekim satır kayıtları.
3. 8 Ekim'de legacy preselection dışında19 ek trigger var; bunların17'sinde bilinen
   ret,2'sinde metadata/score belirsizliği var. Kaynak `full_oct08_research`; bu dönem
   yalnız retrospektif araştırma, PIT/OOS kazanç kanıtı değil.
4. Cap/float/earnings için eski pre-screened varsayımlarını bütün evrene taşımak
   doğru olmaz:19 tetikleyicinin12'sinin market cap'i hedef bandın dışında;12'sinde
   earnings belirsiz. `filters.py::check_market_cap/check_earnings`, `facts.py::collect`.
5. Sınıf-hisse mapping12, hatalı veri12, kısa geçmiş77 ve gerekli-lookback boşluğu3
   doğrudan ölçülmüş kapsama engelleridir. LUMN'da resmî borsa/provider borsa çelişkisi
   var. Hiçbiri sessizce "sinyal vermedi" sayılmadı.
6. Kalıcı cache, checkpoint/resume testleri, full offline replay ve ayrı shadow
   kayıtları çalışıyor. Aynı seans warm run29 çağrı; full replay birebir eşit.

### Hipotezler / henüz kanıtlanmayanlar

- Geniş evrende bulunan ek trigger'ların maliyet sonrası daha iyi fırsatlar olması:
  **kanıt yok**. Eski capture yüzdeleri veya MFE değerleri güncel executable recall
  yerine kullanılmadı. Bu tur filtre gevşetme/score optimizasyonu yapılmadı.
- LUMN uyuşmazlığının transferden mi provider gecikmesinden mi kaynaklandığı bilinmiyor.
- Düşük işlem sıklığı/zero-volume gözlemlerindeki hareketlerin manuel next-open
  alımda uygulanabilirliği ölçülmedi; yüksek Quality bunu tek başına çözmez.
- Yerel Finviz erişim sorununun production'da aynen oluşması kanıtlanamadı;
  production per-query HTTP audit kaydı yok. Remote deployment SHA doğrulanmadı:
  parity iddiası mevcut yerel production fonksiyonları/testlerine aittir.

### Eksik veri ve araştırma sınırları

9 Ekim legacy listesi; gerçek10–30g forward sonuçlar; geçmiş earnings/cap/float
timestamp'leri; split/rename kalıcı-ID zinciri; delisted/PIT evren; gerçek spread,
gap sonrası manuel fill ve market impact eksik. 8 Ekim listesi seçilmiş57 ticker'ı
kanıtlar, bütün Finviz sorgularının o anki tam yanıtını veya final trade listesini değil.
Free Yahoo erişimi bu tek çalışmada başarılı; sürekli erişim/lisans/SLA kanıtı değil.
Yıllık getiri, portföy CAGR/drawdown, expectancy veya kâr artışı hesaplanmadı.

### En küçük sonraki adım — bu turda uygulanmadı

Production'a bağlamadan, bir sonraki tamamlanmış seansta **aynı izolasyonla incremental
veri hazırlama + broad scan + gerçek aynı-seans legacy membership snapshot** üretmek;
metadata'yı tetikleyici kısa listesinde tamamlayıp mevcut shadow kitabına append etmek.
9 Ekim cohort'u ilk kez16 Ekim'de5-seans ölçümüne ulaşacak. Bir tek olgunlaşmış
gün/cohort istatistiksel üstünlük göstermeye yetmez.

En fazla üç sonraki araştırma hipotezi, parametre değiştirmeden:

| Öncelik | Hipotez / test | Başarı ölçütü | Başarısızlık / durdurma | Zorluk |
|---|---|---|---|---|
| 1 | Birden fazla ardışık EOD'da ücretsiz incremental hazırlama sürdürülebilir | ≥%98 kullanılabilir OHLCV, her eksik için neden, aynı-girdi replay eşitliği | Erişim kısıtı, açıklanamayan stale/partial, kontrolsüz çağrı/disk büyümesi | Düşük-orta; çağrı süresi her gün ölçülmeli |
| 2 | Metadata/mapping doğrulaması belirsizliği azaltır | Aynı-seans kanıtlı earnings/cap/float coverage artar, kimlik çelişkisi PASS yapılmaz | Lisans/PIT kanıtı yok veya yanlış security eşleşmesi | Orta; ücretsiz kaynak sınırı var |
| 3 | Broad-only cohort net fırsat kalitesini artırır | Önceden sabitlenmiş next-open,20/50/100bps senaryolarıyla olgunlaşmış bağımsız cohort'ta legacy'ye göre net sonuç farkı; ticker/tarih-kümeli belirsizlik aralığı raporlanır | İyileşme maliyetle kaybolur, birkaç outlier'a dayanır veya aralık üstünlüğü desteklemez | Orta-yüksek; en az30 işlem günü ve yeterli bağımsız gözlem beklenmeli |

Bunlar başarı vaadi değil, ölçülebilir araştırma sorularıdır. Production trading
kuralları, T1/T2/stop/scheduler/deployment ve DB migration değişmedi. Phase 2 sonunda
ağ/veri toplama işlemleri tamamlandı; otomatik çalışan job veya entegrasyon eklenmedi.
