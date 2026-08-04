# Gate Denetimi — hangi kapı neden var?

> **Ölçüm tarihi:** 2026-08-04 · **Harness:** `scripts/measure_gate_value.py`
> **Örneklem:** 78 Q80+ sinyal / 21 ay / 995 ticker · gerçek motor (`scan_stock`) +
> gerçek exit (2.5×ATR, T1 %33, chandelier trail) + slippage
> **Taban:** EV **+3.11%** · WR %56 · PF 2.00

Bu belge tek bir soruya cevap verir: **"Bu kapıya gerçekten gerek var mı?"**
Yöntem, skor bileşenlerine uygulanan **bırak-birini-çıkar** testinin aynısıdır:
her kapı tek tek devre dışı bırakılır, sonuç tabanla karşılaştırılır.

```
Kaldırınca EV DÜŞER    → kapı işe yarıyor        → KALIR
Kaldırınca EV DEĞİŞMEZ → kapı hiç ateşlenmiyor   → DORMANT
Kaldırınca EV ARTAR    → kapı zararlı            → KALDIRILIR
```

**Karar kuralı (ölçümden ÖNCE yazıldı): şüphede kapıyı KORU.** Risk asimetrik —
yanlış silinen bir kapı gerçek para kaybettirir, gereksiz duran bir kapı sadece
sinyal sayısını azaltır.

---

## 🟢 İŞE YARIYOR — 3 kapı (kanıtlı)

| Kapı | ΔEV | TRAIN | OOS | Eklediği sinyaller |
|---|---|---|---|---|
| **Weinstein Stage 3/4** | **−0.98** | −1.46 | −0.54 | +18 @ **−2.10%** |
| **Swing onayı** (MA20 üstü, 5g momentum) | **−0.50** | −1.00 | −0.10 | +4 @ **−7.16%** |
| **RSI > 70** (VCE ve Type S muaf) | **−0.44** | −0.75 | −0.27 | +23 @ +1.16% |

Üçü de **TRAIN ve OOS'ta aynı yönde** — kaldırılırlarsa EV düşer.

**Weinstein en güçlü kapı:** eklediği 18 sinyalin EV'si **−2.10%**. Stage 3
(dağıtım) ve Stage 4 (düşüş) fazındaki hisselere girmek doğrudan para kaybı.

**Swing onayı en keskin:** sadece 4 sinyal ekliyor ama onların EV'si **−7.16%**.
Az sayıda ama felaket işlemleri engelliyor.

**RSI kapısı bir nüans taşıyor:** eklediği sinyallerin EV'si +1.16% (pozitif!)
ama toplam EV −0.44 düşüyor. Sebep: kaliteyi seyreltiyor — 23 vasat sinyal
ortalamayı aşağı çekiyor. Sinyal sayısı ile kalite arasındaki takas.
*(Ayrıca `measure_rsi_gate.py`: eşiği 75/80/100'e çekmek her seferinde
negatif-EV sinyal ekliyor.)*

---

## ⚪ DORMANT — 5 kapı (mevcut popülasyonda hiç ateşlenmiyor)

| Kapı | ΔEV | Eklediği sinyal |
|---|---|---|
| Geç giriş (5g>%30 **&** RSI>65) | 0.00 | 0 |
| Dağıtım günü (hacim≥2× **&** değişim≤−%5) | 0.00 | 0 |
| R:R ≥ 1.8 | 0.00 | 0 |
| Parabolik / ekstrem (5g>%70, RSI>85) | 0.00 | 0 |
| Aşırı uzama (bugün>%15, tek gün>%25, 5g>%40) | 0.00 | 0 |

**Neden hiç ateşlenmiyorlar?** Yapısal olarak daha önceki kapılar tarafından
kapsanıyorlar:

- **Geç giriş / parabolik / aşırı uzama:** VCE muafiyeti + Weinstein Stage 3/4
  reddi + swing onayı bu vakaları zaten eliyor
- **Dağıtım günü:** VCE ve RVOL thrust ikisi de **yeşil kapanış** şartı koyuyor;
  "hacimli düşüş günü" tetiği hiç geçemiyor
- **R:R ≥ 1.8:** stop 2.5×ATR + T1 %10 kombinasyonu matematiksel olarak neredeyse
  her zaman R:R > 1.8 üretiyor

### Neden SİLİNMİYORLAR?

Bu kapılar **bugünkü sinyal popülasyonunda** ateşlenmiyor. Üçüncü bir pathway
eklenirse (ör. sıkı konsolidasyon kırılımı — `measure_third_pathway.py`'de aday
olarak geçti) popülasyon değişir ve bu kapılar **koruyucu hale gelebilir.**

Maliyeti sıfır (hiç ateşlenmiyorlar), faydası opsiyon değeri. Karar kuralı gereği
**korunuyorlar** — ama artık *dormant* olduğu **belgeli**, gizemli değil.

🔧 **İzleme:** Tarama `stats.reject_counts` her kapının kaç kez ateşlendiğini
sayıyor. Bir dormant kapı saymaya başlarsa popülasyon değişmiş demektir → yeniden
ölç.

---

## 🟡 NÖTR — 1 kapı

| Kapı | ΔEV | TRAIN | OOS | Not |
|---|---|---|---|---|
| OBV dağıtım | +0.15 | +0.00 | +0.24 | +1 sinyal @ +14.79% |

Kaldırınca EV **hafif artıyor** ama etki tek bir sinyalden geliyor (n=1).
n=1 ile karar verilmez → **korunuyor** (şüphede kapıyı koru).

👁️ **İzleme listesi:** örneklem büyürse yeniden ölç. Eğer OBV kapısı tutarlı
şekilde kârlı sinyalleri engelliyorsa kaldırılmalı.

---

## Ölçülmemiş kalanlar

Bu denetim `scan_stock` içindeki **hard gate'leri** kapsıyor. Hâlâ ölçülmemiş
olanlar:

| Katman | Neden ölçülmedi |
|---|---|
| ~17 bonus + ~21 ceza (skor içi) | Bırak-birini-çıkar mümkün ama örneklem gerekir |
| Tip sınıflaması (C/A/B/S) | Boyut/stop/hedef tavanlarını belirliyor, sinyal üretmiyor |
| Rejim tespiti | Eşiği belirliyor; eşik ölçüldü, rejim mantığı ölçülmedi |
| Cooldown 5 gün · Ticker 2-zarar | Portföy seviyesi, sinyal seviyesi değil |
| Kazanç raporu ±3 gün | Backtest'te point-in-time veri yok |

Bunlar **bir sonraki denetim turunun** konusu.

---

## Özet

| Durum | Sayı | Aksiyon |
|---|---|---|
| 🟢 İşe yarıyor | 3 | Kalır — dokunma |
| ⚪ Dormant | 5 | Kalır — belgelendi, izlemede |
| 🟡 Nötr | 1 | Kalır — n=1, izlemede |
| 🔴 Zararlı | **0** | — |

**Hiçbir kapı zararlı çıkmadı.** Sistem "gereksiz karmaşık" değil ama
**5 kapının uykuda olduğu** artık belgeli — bir sonraki geliştirici (veya
6 ay sonraki biz) neyin neden orada olduğunu bilecek.
