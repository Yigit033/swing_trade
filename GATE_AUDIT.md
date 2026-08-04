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

## Ölçülmemiş kalanlar (sonraki tur)

| Katman | Not |
|---|---|
| ~17 bonus + ~21 ceza (skor içi) | Bırak-birini-çıkar mümkün, örneklem gerekir |
| Tip sınıflaması (C/A/B/S) | Boyut/stop/hedef tavanı belirliyor, sinyal üretmiyor |
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

Kalan her kapının ya **ölçülmüş koruyucu değeri** var ya da teknik zorunluluk.
