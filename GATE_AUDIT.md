# Gate Denetimi

> **Ölçüm:** 2026-08-04 · `scripts/measure_gate_value.py` + `measure_remaining_gates.py`
> 78 Q80+ sinyal / 21 ay / 995 ticker · gerçek motor (`scan_stock`) + gerçek exit
> (2.5×ATR, T1 %33, chandelier trail) + slippage
> **Taban:** EV **+3.11%** · WR %56 · PF 2.00

Her kapı tek tek devre dışı bırakıldı, sonuç tabanla karşılaştırıldı
(**bırak-birini-çıkar**, skor bileşenlerine uygulanan yöntemin aynısı).

## Silme ilkesi

```
Ölçüldü + etkisi yok        → SİL   (zararlı olması gerekmez)
Ölçüldü + işe yarıyor       → KAL
Ölçüldü + zararlı           → SİL
Ölçüm bozuk / kanıt yok     → KAL   → sonra DOĞRU ölç
Hiçbir kod okumuyor         → SİL   (ölçüme bile gerek yok)
```

---

## 🟢 KALAN KAPILAR — 6

### Ölçüldü, koruyucu değeri kanıtlı (3)

| Kapı | ΔEV | TRAIN | OOS | Engellediği sinyaller |
|---|---|---|---|---|
| **Weinstein Stage 3/4** | **−0.98** | −1.46 | −0.54 | 18 @ **−2.10%** |
| **Swing onayı** (MA20 üstü + 5g momentum) | **−0.50** | −1.00 | −0.10 | 4 @ **−7.16%** |
| **RSI > 70** (VCE ve Type S muaf) | **−0.44** | −0.75 | −0.27 | 23 @ +1.16% |

Üçü de **TRAIN ve OOS'ta aynı yönde.**

- **Weinstein en güçlü:** engellediği 18 sinyalin EV'si −2.10%. Dağıtım (Stage 3)
  ve düşüş (Stage 4) fazına girmek doğrudan para kaybı.
- **Swing onayı en keskin:** yalnız 4 sinyal engelliyor ama EV'leri −7.16%.
- **RSI nüanslı:** engellediği sinyaller pozitif (+1.16%) ama kaliteyi
  seyreltiyor → toplam EV −0.44 düşüyor. `measure_rsi_gate.py`: eşiği 75/80/100'e
  çekmek her seferinde negatif-EV sinyal ekliyor.

### Çekirdek (ölçüldü, kapı değil filtre/eşik) (3)

| Katman | Kanıt |
|---|---|
| Evren filtresi (fiyat / mcap / **$5M dolar-hacim**) | Kesişim testi: likit +3.31% / illikit −2.14% |
| Tetik (VCE + RVOL thrust) | VCE R10 +5.2% t=2.6 OOS ✓ · RVOL EV +0.96% OOS +2.10% |
| Kalite eşiği (Q78/80/82) | Q80 +2.41% vs Q73 +1.18% |

*(Ayrıca `insufficient_data` ve `scan_error` — teknik zorunluluk, kaldırılamaz.)*

---

## 🔴 SİLİNENLER

### Gate'ler — ölçüldü, ΔEV tam 0.00, 21 ayda hiç ateşlenmedi (5)

| Kapı | Neden hiç ateşlenmiyordu |
|---|---|
| **Geç giriş** (5g>%30 & RSI>65) | VCE muafiyeti + Weinstein + swing onayı bu vakaları zaten eliyordu |
| **Dağıtım günü** (hacim≥2× & değişim≤−%5) | VCE **ve** RVOL **ikisi de** yeşil kapanış istiyor → "hacimli düşüş günü" sinyal olarak hiç oluşamıyor |
| **R:R** (rejime göre 1.0-2.0) | 2.5×ATR stop + T1 %10 + T2 tavanları matematiksel olarak neredeyse hep R:R(T2) > 2.0 üretiyor |
| **Zayıf trend** (markdown & strength<10) | VCE/RVOL + swing onayından geçen sinyal tanımı gereği yükseliş trendinde; "markdown" fazı hiç birlikte oluşmuyor |
| **OBV dağıtım** | 21 ayda **1 kez** ateşlendi — engellediği sinyal **+%14.79 kazandı.** Koruyucu değeri sıfır ölçüldü (n=1 olduğu için "zararlı" kanıtlanamaz, ama "faydalı" kesinlikle çürütüldü) |

⚠️ **R:R notu:** İlk ölçümüm **geçersizdi** — kapıyı `min_rr_at_entry=0` ile
kapatmaya çalıştım ama gate rejime göre koda gömülü değerleri kullanıyordu, ayar
yalnız *bilinmeyen rejim* dalını etkiliyordu. Değerler ayara taşındıktan sonra
gate gerçekten kapatıldı ve ΔEV 0.00 doğrulandı.
**Exit parametreleri (stop çarpanı / T2 tavanları) materyal olarak değişirse bu
kapı yeniden ölçülmeli** — o zaman bağlayabilir.

### Hiçbir kodun okumadığı ayarlar (grep ile doğrulandı)

```
scan_gates.parabolic_five_day_return_gt
scan_gates.extreme_five_day_return_gt
scan_gates.extreme_rsi_gt
```

UI'da ayarlanabiliyorlardı ama **hiçbir etkileri yoktu.**

### Aynı turda silinen diğerleri

| Ne | Neden |
|---|---|
| Composite sıralama: metot + 4 ağırlık + kovalama cezası + erken-birikim bonusu (~130 satır) | Hiçbiri ölçülmemişti; tavan 15/15 taramada bağlamadı. Yerine ölçülmüş tek satır: dolar-hacim |
| Finviz Q1/Q2/Q3 (73 satır) | Recall ölçümü: %0.5-2 katkı |
| `_parse_percent` | Composite gidince kullanılmaz kaldı |
| ~52 satır mezar taşı yorumu + bayat docstring | Silinen özellikleri anlatıyorlardı |

**Toplam silinen:** 5 gate · 21 ayar · 15 UI alanı · ~350 satır kod.

---

## Neden bu kadar kapı hiç ateşlenmiyordu?

Tek bir yapısal sebep: **VCE ve RVOL thrust ikisi de "yeşil kapanış + trend
üstü" istiyor.** Bu iki şart, aşağı yönlü/dağıtım/geç-giriş senaryolarını
tetik aşamasında zaten eliyor. Sonraki kapılar aynı şeyi tekrar kontrol
ediyordu — biri yeterliydi.

Kapılar zamanla eklendi ama **hangi tetiğin ne garanti ettiği hiç ölçülmedi.**
Bu denetim o boşluğu kapattı.

---

## 3. tur — SKOR DEĞİŞTİRİCİLERİ (2026-08-05)

Kapılar bittikten sonra geriye skorun kendi içi kalmıştı: **14 bonus + 21 ceza.**
Hiçbiri hiç ölçülmemişti. Harness: `scripts/measure_score_modifiers.py`
(95 sinyal / 21 ay, bırak-birini-çıkar + Q80 seçim testi + TRAIN/OOS).

### Bulgu 1 — 14 bonusun ayırt etme gücü SIFIR

| Ölçüm | Sonuç |
|---|---|
| Bonus tavanına (30) dayanan sinyal oranı | **%100** |
| Tavan üstü aşılıp atılan puan | ortalama **+29.7**, maks +48 (ham toplam ~60) |
| Tek bonusu sıfırlamanın ΔEV'si | 14/14'ünde **tam 0.00** |

Sebep: bonuslar toplanıp `min(bonus, 30)` ile kırpılıyordu. Ham toplam her zaman
tavanın iki katıydı, dolayısıyla **hangi bonus açık olursa olsun skora giden net
sayı aynıydı: +30.** 40 satır kod, 14 ayarlanabilir parametre ve 14 UI slider'ı
tek bir sabitin işini yapıyordu — dahası kullanıcıya "bunu kısarsam skor değişir"
izlenimi veriyordu, halbuki değişmiyordu.

→ Kod tek satıra indi: `bonus = st.bonus_cap`. **Davranış birebir aynı.**

Alternatif (tavanı kaldırıp bonusların gerçekten ayırt etmesini sağlamak)
kasıtlı olarak yapılmadı: ham toplam 40-78 bandına çıkar, bu ÖLÇÜLMEMİŞ bir
davranış değişikliğidir ve Q78/80/82 eşiklerinin tümünün yeniden kalibrasyonunu
gerektirir. Sonraki turun konusu.

### Bulgu 2 — 21 cezanın 16'sı işe yaramıyordu

**Hiç ateşlemeyen (10):** `pen_5d_gt_30`, `pen_b_rsi_gt_75`, `pen_b_rsi_gt_80`,
`pen_below_ma50`, `pen_not_swing_ready`, `pen_obv_distribution`, `pen_parabolic`,
`pen_spread_risk`, `pen_today_gt_15`, `pen_weak_trend_phase`
— kapı denetimindeki aynı yapısal sebep: girdilerini VCE/RVOL tetiği zaten
garanti ediyor (yeşil kapanış + MA50/MA20 üstü + swing_ready hard gate).

**Ateşleyip ΔEV 0.00 verenler (6) — ceza YÖNÜ TERSTİ:**

| Ceza | Ateşleme | Cezalananların EV'si |
|---|---|---|
| `pen_5d_gt_40` | 3 | **+20.36%** |
| `pen_ext_day_gt_25` | 1 | **+21.12%** |
| `pen_ext_day_gt_20` | 1 | **+15.16%** |
| `pen_5d_gt_25` | 5 | **+13.24%** |
| `pen_ma20_falling` | 2 | **+11.24%** |
| `pen_b_rsi_gt_85` | 1 | +8.75% |

Yani "çok koşmuş, cezalandır" sezgisi bu sistemde **yanlış**: cezalanan
sinyaller ortalamanın iki-üç katı kazandı. Sadece cezalar Q80 seçimini
değiştirecek kadar büyük olmadığı için zarar görünmüyordu — ama gerekçesi
olmayan bir kural her an büyütülebilir, o yüzden silindi.

**Kalanlar (5) — çıkarınca EV DÜŞÜYOR, yani koruyorlar:**

| Ceza | Ateşleme | ΔEV (çıkarınca) |
|---|---|---|
| `pen_c_rsi_gt_60` | 12 | −0.20 |
| `pen_c_rsi_gt_65` | 17 | −0.16 |
| `pen_today_gt_10` | 3 | −0.10 |
| `pen_a_rsi_gt_65` | 4 | −0.09 |
| `pen_a_rsi_gt_70` | 28 | −0.07 |

RSI merdiveni tutuyor; aşırı-uzama merdivenleri tutmuyor. Tip B'de RSI cezası
yok — Type B'nin tanımı zaten "yüksek RSI ile koşuyor".

### `pen_spread_risk` — 9 gün önce eklenmişti, ölüydü

S1 (2026-07-27) düşük dolar-hacim **VE** ATR>%8 birleşimini cezalıyordu.
21 ayda bu iki koşul **hiç birlikte gerçekleşmedi** (dvol<7M: 3 sinyal — üstelik
hepsi kazandı; ATR>%8: 5 sinyal). Kapı fiilen ölüydü.

Silmeden önce **saf ATR>%8 cezası** da ölçüldü (koruma sessizce kaybolmasın):

| ATR% bandı | n | TÜM EV | TRAIN | OOS |
|---|---|---|---|---|
| 0-4% | 306 | +0.50% | −0.45% | +0.95% |
| 4-6% | 142 | **+3.72%** | +5.59% | +2.32% |
| 6-8% | 16 | −2.61% | **+9.09%** | **−9.63%** |
| 8%+ | 5 | −6.90% | −8.25% | −4.86% |

6-8% bandı dönemler arası **işaret değiştiriyor** (n=16). Q80'de ATR tavanı tek
bir işlemi eliyor → tek gözleme eğri uydurmak olurdu. **Tavan eklenmedi, ceza
silindi, bulgu kayda geçti.**

### Parite doğrulaması (asıl güvence)

Aynı 95 barı eski ve yeni kodla skorlayıp karşılaştırdık:

```
SKOR FARKI (yeni - eski):  değişmeyen 84/95 · artan 11 (ort +12.4) · azalan 0
Q80 SEÇİMİ:  ESKİ n=78 EV +3.11% WR 56%  ->  YENİ n=78 EV +3.11% WR 56%
             Q80'e yeni giren 0 · Q80'den düşen 0
SIRALAMA  :  8 sinyalli tek kalabalık gün (2024-09-20) — top_3/4/10 AYNI
```

11 sinyalin skoru yükseldi (silinen ceza kadar) ama **hiçbiri kararı
değiştirmedi**: ne eşiği geçen küme, ne sıralama, ne EV. Ölçümün "ΔEV 0.00"
iddiası doğru sebepten doğruydu.

**Bu turda silinen:** 14 bonus · 16 ceza · 33 ayar · 31 UI slider'ı · scoring.py
571 → 434 satır.

---

## Ölçülmemiş kalanlar (sonraki tur)

| Katman | Not |
|---|---|
| Bonus tavanının kendisi (30) | Tavanı kaldırıp bonusların ayırt etmesini sağlamak ölçülmeli — eşik rekalibrasyonu gerektirir |
| Tip sınıflaması (C/A/B) | Boyut/stop/hedef tavanı belirliyor, sinyal üretmiyor |
| Rejim tespiti | Eşiği belirliyor; eşik ölçüldü, rejim mantığı ölçülmedi |
| Cooldown 5 gün · Ticker 2-zarar | Portföy seviyesi |
| `signal_confirmation.overext_*` | Yalnız bir log karakterini (✓/⚠) belirliyor; skor cezaları kendi sabit eşiklerini kullanıyor |

---

## Bilanço

| | Önce | Sonra |
|---|---|---|
| Hard gate | 11 | **6** |
| Ölçülmüş gate oranı | %27 | **%100** |
| Gerekçesi belirsiz gate | 8 | **0** |
| Skor bonusu (koşullu) | 14 | **0** (sabit +30) |
| Skor cezası | 21 | **5** (hepsi ölçülmüş) |
| Ölçülmüş skor-değiştirici oranı | %0 | **%100** |

Kalan her kapının ve her cezanın ya **ölçülmüş koruyucu değeri** var ya da
teknik zorunluluk. Ölçülmemiş hiçbir kural sinyal seçimine karışmıyor.
