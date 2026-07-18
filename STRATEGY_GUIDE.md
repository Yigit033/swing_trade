# 📚 Strateji Rehberi — SmallCap VCE Swing Sistemi

> **Senkron notu:** Bu rehber 2026-07-18 itibarıyla kod ve ölçüm gerçeğiyle birebir senkrondur.
> Buradaki **her sayı** repo'daki bir ölçüm harness'ından gelir (bkz. §9). Bir sabiti
> değiştirmeden önce ilgili harness yeniden koşulur — "hissiyatla kalibrasyon" yasaktır.

---

## 1. Tek cümlelik tez

**Küçük/orta ölçekli (mcap $300M–$10B) ABD hisselerinde, volatilite sıkışması sonrası
gelen kırılım gününü (VCE) tamamlanmış günlük barda tespit et; ertesi sabah açılışta gir;
riski ATR ile sınırla.** Sistemin kanıtlanmış tek edge'i budur — geri kalan her katman bu
edge'i korumak için vardır.

## 2. Neden VCE? — Ölçüm hikayesi

2026-06'da tüm sinyal yolları 57 hisse × 2 yıl (24k bar) benchmark'ına karşı ölçüldü
(`scripts/measure_signal_edge.py` + `test_edge_hypotheses.py`):

| Sinyal yolu (eski sistem) | Ölçülen edge | Karar |
|---|---|---|
| volume_ignition / erken birikim | R5 **−1.17%** (t=−1.83) | ❌ kaldırıldı |
| technical_breakout (5-bar high) | ölçülebilir edge yok | ❌ kaldırıldı |
| trend_continuation | R5 +0.29% (t=0.65, gürültü) | ❌ kaldırıldı |
| **volsqueeze_breakout (VCE)** | **R10 +2.42% (n=408, t=2.75); OOS +2.62% (t=2.17)** | ✅ TEK birincil trigger |

Sonuç (v13 tez değişimi): VCE tek sinyal üreticisidir; eski metrikler yalnız
gösterim/skorlama bağlamı için hesaplanır.

## 3. Sinyal kuralı — "Variant B" (kod: `signals.check_vce_breakout`)

Dört **sert kapı**, hepsi tamamlanmış günlük barda:

1. **Sıkışma:** ATR%(14, SMA) dün < **0.8 ×** taban çizgisi (t−20…t−6 barlarının ATR% ortalaması)
2. **Kırılım:** Bugünün kapanışı > önceki **20 günün en yüksek** fiyatı (bugün hariç)
3. **Yeşil gün:** Kapanış > dünkü kapanış
4. **Trend:** Kapanış > **MA50** (kapanış bazlı)

**Sert kapı OLMAYANLAR (bilinçli):** hacim ≥1.5× ve kapanış-pozisyonu ≥0.6 yalnız
**premium tier** skorlamasıdır. Sert kapı yapıldıklarında ("Variant D") sinyal başına edge
+5.16'ya çıkıyor ama frekans 3.3× düşüyor ve **toplam** edge azalıyordu (sistem bir hafta
0 sinyal üretti — yaşandı). Ayrıca eski `min_atr ≥ %3` kapısı VCE'ye uygulanmaz: sıkışmış
hisse tanımı gereği düşük ATR'lidir.

## 4. Zamanlama disiplini — edge'in yaşadığı yer

Ölçüm konvansiyonu: sinyal barı `t` → giriş **`t+1` açılışı**. Canlı sistem bunu üç
mekanizmayla korur:

- **Incomplete-bar guard:** ET 16:00'dan önce bugünün yarım barı atılır — hacim-göreli
  kurallar yalnız tamamlanmış barda karar verebilir.
- **Entry-window guard:** Sinyal barının ertesi seans açılışı GEÇTİYSE trade açılmaz
  (`entry window missed`) — t+2 girişi ölçülmemiştir, ölçülmemiş girişe sermaye bağlanmaz.
- **Kanonik tarama penceresi: ABD kapanışı sonrası (23:05+ TR).** Seans içi taramanın
  sinyalleri dünün barına aittir (takip edilemez); pre-market taramada evren sorguları boş
  döner (UI uyarır).

## 5. Evren (huni) — adaylar nereden gelir?

**4 Finviz sorgusu** (2026-07-18'den beri; hepsi USA, fiyat >$7):

| Sorgu | Bant | Ayırt edici |
|---|---|---|
| VCE BREAKOUT DAY (small) | $300M–2B, AvgVol>500K | Change ≥2%, RelVol>1.5 |
| VCE BREAKOUT DAY (mid) | $2–10B, AvgVol>1M | Change ≥2%, RelVol>1.5 |
| 20D NEW HIGH (small) | $300M–2B, AvgVol>500K | SMA50 üstü + 20g yeni zirve |
| 20D NEW HIGH (mid) | $2–10B, AvgVol>1M | SMA50 üstü + 20g yeni zirve |

20D-high sorguları VCE'nin **zorunlu koşulunu** tarar → yapısal catch-all. Post-filter:
fiyat **$7–1000** (recall ölçümü iki ölü bölgeyi kapattı: $7-8 bandı RKLB +31/JOBY +22'yi,
$200 tavanı WING koşusunun 5 sinyalini kesiyordu). Sonra composite momentum skoru ile
sıralanıp **top 260** taranır; tavanın kestiği isimler telemetriye yazılır (`universe_cut_tickers`).

**Recall ölçümü bulgusu** (`measure_universe_recall.py`, 408 sinyal): huni sinyallerin
%46.6'sını yakalar AMA kaçanların edge'i **+0.32 (t=0.37) ≈ SIFIR**; yakalananların edge'i
**+4.82 (t=3.07)**. Huni bir **edge yoğunlaştırıcıdır** — recall'ü kovalamak için likidite
tabanını gevşetme (ölçüldü: para etmiyor). Eski Q1-Q4 sorguları (%0-2 katkı) kapatıldı/silindi.

## 6. Rejim katmanı

SPY vs MA50/MA200 + VIX → **BULL / CAUTION / BEAR** (+ CONFIRMED/TENTATIVE güveni).
Kalite eşiği rejim-farkındadır: taban 70 → efektif **BULL 60 / CAUTION 70 / BEAR 75**.
Rejim çarpanı skora uygulanmaz; bilgi olarak gösterilir.

## 7. Giriş onayı — PENDING → OPEN (kod: `tracker.confirm_pending_trades`)

Sinyal akşam **PENDING** yazılır; ertesi seans açılışında onaylanır:

- **Gap filtresi (2026-07-17'de yeniden ölçüldü):** gap-up > **+5%** → RED (pump-open
  koruması; ölçülen maliyeti ~0). Gap-down < **−7%** → RED (eski −5 limiti "shakeout"
  girişlerini kesiyordu — reddedilen gap-down grubu R10 +25.97 ile örneklemin en iyisiydi;
  −7'ye gevşetme edge'i +1.86→+2.24 yükseltti).
- Stop/hedef gerçek açılış fiyatından yeniden hesaplanır (aşağıda).
- Tatil-farkında takvim: onay günü hafta sonu/tatil atlanarak bulunur.

## 8. Risk ve çıkışlar (kod: `risk.py` + `tracker.py`; yol simülasyonuyla doğrulandı)

| Kural | Değer | Kanıt/Not |
|---|---|---|
| Stop | **1.5 × ATR** (girişten) | Yol simülasyonunda EN İYİ: EV +2.87%/trade, WR %59 |
| Stop tavanı (tip bazlı) | C %8 · A/B %10 · S %12 | Aşırı geniş ATR stoplarına fren |
| T1 | ATR × tip çarpanı, taban 1.5R, cap ~%8-10 | %67 ulaşım oranı ölçüldü |
| T2 | rejim-ayarlı, cap ~%28 | MFE p75'e denk gelir |
| Trailing stop | Hold süresinin %50'si + 2 ATR kazanç sonrası | Kâr koruması |
| Timeout | Tip bazlı max hold (örn. C 3-8 gün) | Ölü sermayeyi boşalt |
| Cooldown | Kapanan trade'den sonra 5 gün aynı hisseye girme | Revenge-trade freni |
| Pozisyon | Risk bazlı boyutlama (~%1.5 risk/trade) | |

## 9. Kalibrasyon disiplini — harness envanteri

**Kural: sabit değiştirmeden önce ilgili harness koşulur; PR/commit mesajına sayı yazılır.**

| Harness | Neyi ölçer | Ne zaman koş |
|---|---|---|
| `scripts/measure_signal_edge.py` | Tüm sinyal yollarının ham edge'i (+ veri cache'i üretir) | Trigger kuralı değişince |
| `scripts/validate_volsqueeze.py` | VCE varyantları, train/test, ticker konsantrasyonu, rejim | VCE sabitleri değişince |
| `scripts/measure_gap_filter.py` | Gap limitlerinin edge maliyeti | `MAX_GAP_*` değişince |
| `scripts/measure_universe_recall.py` | Finviz sorgularının sinyal yakalama oranı + kaçanların otopsisi | Sorgu/filtre/fiyat bandı değişince |
| `scripts/exit_strategy_lab.py` | Stop/T1/T2 alternatiflerinin EV'si | Çıkış kuralları değişince |
| `test_backtest.py` | All-season stress (BEAR/SIDEWAYS/BULL/CURRENT) | Büyük değişiklik sonrası |

## 10. Beklenti yönetimi — "makine" nasıl görünür?

- **Günde 0-2 sinyal normaldir; 0 sinyallik günler arıza değildir** (Variant B bilinçli
  olarak frekans×edge toplamını maksimize eder). Sinyal yoksa sistem "bugün ölçülü fırsat
  yok" demektedir — bu bir çıktıdır.
- Sinyal başına beklenen R10 edge'i: **~+2.2–2.4%** (huniden geçen altkümede ~+4.8%).
  Bunlar ORTALAMALARDIR; tek tek trade'ler geniş dağılır, WR ~%55-60 hedefi gerçekçidir.
- Canlı öğrenme döngüsü: her sinyalin R3/R5/R10 forward return'ü otomatik takip edilir
  (`ForwardReturnTracker`), tarama geçmişi + evren telemetrisi kaydedilir — kalibrasyon
  kararları bu birikimle verilir.

---

*Değişiklik geçmişi ve gerekçeler: proje hafızasındaki kalibrasyon notları (v7→v14) ve
`docs/BACKTEST_LIVE_PARITY.md`. Eski çok-faktörlü (EMA/RSI/MACD) rehber, ölçümde edge'i
sıfır çıkan v1 sistemiyle birlikte 2026-07'de kaldırıldı.*
