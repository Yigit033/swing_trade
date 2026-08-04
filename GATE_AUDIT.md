# Gate Denetimi — hangi kapı neden var?

> **Ölçüm:** 2026-08-04 · `scripts/measure_gate_value.py`
> 78 Q80+ sinyal / 21 ay / 995 ticker · gerçek motor (`scan_stock`) + gerçek exit
> (2.5×ATR, T1 %33, chandelier trail) + slippage
> **Taban:** EV **+3.11%** · WR %56 · PF 2.00

Yöntem: skor bileşenlerine uygulanan **bırak-birini-çıkar** testinin aynısı.
Her kapı tek tek devre dışı bırakılır, sonuç tabanla karşılaştırılır.

---

## 🟢 İŞE YARIYOR — 3 kapı (KALDI)

| Kapı | ΔEV | TRAIN | OOS | Eklediği sinyaller |
|---|---|---|---|---|
| **Weinstein Stage 3/4** | **−0.98** | −1.46 | −0.54 | +18 @ **−2.10%** |
| **Swing onayı** (MA20 üstü + 5g momentum) | **−0.50** | −1.00 | −0.10 | +4 @ **−7.16%** |
| **RSI > 70** (VCE ve Type S muaf) | **−0.44** | −0.75 | −0.27 | +23 @ +1.16% |

Üçü de **TRAIN ve OOS'ta aynı yönde** — kaldırılırlarsa EV düşer.

- **Weinstein en güçlü:** eklediği 18 sinyalin EV'si **−2.10%**. Stage 3 (dağıtım)
  ve Stage 4 (düşüş) fazına girmek doğrudan para kaybı.
- **Swing onayı en keskin:** sadece 4 sinyal ekliyor ama EV'leri **−7.16%**.
  Az sayıda ama felaket işlem engelliyor.
- **RSI kapısı nüanslı:** eklediği sinyaller pozitif (+1.16%) ama kaliteyi
  seyreltiyor → toplam EV −0.44 düşüyor. Ayrıca `measure_rsi_gate.py`: eşiği
  75/80/100'e çekmek her seferinde negatif-EV sinyal ekliyor.

## 🟡 KALDI — 2 kapı (ölçülemedi / yetersiz veri)

| Kapı | Durum |
|---|---|
| **OBV dağıtım** | ΔEV +0.15 ama etki tek sinyalden (n=1) → karar verilemez, izlemede |
| **R:R (rejime göre)** | ⚠️ **Ölçümüm GEÇERSİZDİ** — bkz. aşağıda |
| **Zayıf trend** (markdown fazı) | Ölçüme dahil edilmedi |

### R:R ölçümündeki hata (dürüstlük kaydı)

Kapıyı `min_rr_at_entry=0` yaparak kapatmaya çalıştım ama gate rejime göre
**koda gömülü** değerler kullanıyordu:

```python
_regime_rr = {"BULL": 1.0, "CAUTION": 1.5, "BEAR": 2.0}
min_rr = _regime_rr.get(regime, self.settings.min_rr_at_entry)
```

Yani ayarı sıfırlamak yalnız *bilinmeyen rejim* dalını etkiledi; kapı hiç
kapanmadı. ΔEV 0.00 sonucu **kapının inert olduğunu göstermiyor** — ölçümün
çalışmadığını gösteriyor. **Kapı silinmedi.**

Bunun yerine gerçek sorun düzeltildi: değerler `regime_thresholds`'a taşındı
(`bull_min_rr` / `caution_min_rr` / `bear_min_rr`). Eskiden ayar "1.8" diyor,
gerçek uygulanan 1.0-2.0 arasıydı — klasik **yanıltıcı ayar** tuzağı.

---

## 🔴 SİLİNDİ — 2 gate + 3 tamamen ölü ayar

### Gate'ler (ölçüldü, ΔEV tam 0.00 — hiç ateşlenmiyorlardı)

| Kapı | Neden hiç ateşlenmiyordu |
|---|---|
| **Geç giriş** (5g>%30 & RSI>65) | VCE muafiyeti + Weinstein + swing onayı bu vakaları zaten eliyordu |
| **Dağıtım günü** (hacim≥2× & değişim≤−%5) | VCE **ve** RVOL **ikisi de** yeşil kapanış istiyor → "hacimli düşüş günü" bir sinyal olarak hiç oluşamıyor |

### Hiçbir kodun okumadığı ayarlar (grep ile doğrulandı)

```
scan_gates.parabolic_five_day_return_gt
scan_gates.extreme_five_day_return_gt
scan_gates.extreme_rsi_gt
```

Bunlar UI'da ayarlanabiliyordu ama **hiçbir kod okumuyordu.** "Tarama geçitleri"
bölümünün tamamı (5 alan) UI'dan da kaldırıldı.

### Ayrıca silinen (aynı turda)

| Ne | Neden |
|---|---|
| Composite sıralama: 4 ağırlık + kovalama cezası + erken-birikim bonusu (~90 satır) | Hiçbiri ölçülmemişti; tavan 15/15 taramada bağlamadı |
| `rank_weight_*` · `chase_penalty_*` (8 ayar + 8 UI alanı) | Yukarıdakine bağlı |
| Finviz Q1/Q2/Q3 (73 satır) + 3 ayar + 3 UI toggle | Recall ölçümü: %0.5-2 katkı, kapatılmışlardı |

---

## Silme ilkesi

**"İşe yaramıyorsa sil"** — zararlı olması gerekmez. Ama silmenin şartı **ölçüm**:

```
Ölçüldü + etkisi yok        → SİL
Ölçüldü + işe yarıyor       → KAL
Ölçülemedi / ölçüm bozuk    → KAL (silmek için kanıt yok)
Hiçbir kod okumuyor         → SİL (ölçüme bile gerek yok)
```

R:R kapısı son kategoriden kurtuldu: ölçümüm bozuktu, o yüzden kalmaya devam
ediyor. Bir sonraki turda doğru ölçülecek.

---

## Ölçülmemiş kalanlar (sonraki tur)

| Katman | Not |
|---|---|
| ~17 bonus + ~21 ceza (skor içi) | Bırak-birini-çıkar mümkün, örneklem gerekir |
| Tip sınıflaması (C/A/B/S) | Boyut/stop/hedef tavanı belirliyor, sinyal üretmiyor |
| Rejim tespiti | Eşiği belirliyor; eşik ölçüldü, rejim mantığı ölçülmedi |
| Zayıf trend (markdown) gate | Ölçüme dahil edilmedi |
| R:R kapısı | Ölçüm bozuktu — doğru kapatıcıyla tekrar |
| Cooldown 5 gün · Ticker 2-zarar | Portföy seviyesi |
| `signal_confirmation.overext_*` | Sadece bir log karakterini (✓/⚠) belirliyor; skorlama penaltıları kendi sabit eşiklerini kullanıyor |

---

## Bilanço

| Durum | Sayı |
|---|---|
| 🟢 Ölçüldü, işe yarıyor → kaldı | 3 |
| 🟡 Ölçülemedi → kaldı | 3 |
| 🔴 Ölçüldü, etkisi yok → **silindi** | 2 gate |
| 🔴 Hiç okunmuyordu → **silindi** | 3 ayar |

**Silinen toplam:** 2 gate + 14 ayar + 8 UI alanı + ~165 satır kod.
Kalan her parçanın ya ölçülmüş gerekçesi var ya da neden ölçülemediği yazılı.
