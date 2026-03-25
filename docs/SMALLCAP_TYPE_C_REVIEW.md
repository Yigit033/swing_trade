# Tip C (Early Stage): sınıflandırma ve deney fikirleri

Backtest özetlerinde trade sayısının büyük kısmı **Tip C** olabiliyor; bu, motorun öncelik sırası ve piyasa koşullarının birleşimidir. Tip C “kötü” değildir — tasarımda “erken aşama, daha geniş R/R” hedefi vardır — ama **düşük kaliteli C** girişleri veya **sık stop** dönemlerinde toplam P/L’ye negatif katkı görülebilir.

## Nerede tanımlı?

`SmallCapEngine._classify_swing_type` içinde (`swing_trader/small_cap/engine.py`).

**Öncelik sırası:** S → C → B → A. Tip C, kısa sıkışma (S) yoksa ve skor yeterliyse atanır.

## Tip C skoru (özet)

Puana katkı veren örnek bileşenler:

- **5 günlük getiri** −5% … +15% (özellikle 0…10% “sweet spot” bonusu)
- **RSI** 40–60 (ideal), 60–65 hâlâ kısmi puan
- **Hacim patlaması** 1.8x … 4.0x
- **MA20 mesafesi** −3% … +8%
- **Kapanış konumu** üst yarı (≥ 0.55)
- **RSI divergence** (+3), **MACD bullish**, **higher lows** bonusları

**Eşik:** `type_c_score >= 8` ise Tip C; aksi halde B/A yoluna düşülür.

## Neden örneklemde ağırlıklı görünebilir?

- S ve B için daha dar bantlar (sıkışma, yüksek momentum) genelde daha az hisseyi yakalar.
- C bandı (5d, RSI, hacim) **daha geniş**; `min_quality` ve rejim geçtikten sonra kalan adayların çoğu C olabilir.
- Backtestte katalizör / sektör RS sıfırlandığı için canlıya göre sınıflandırma da hafifçe kayabilir (parite: `BACKTEST_LIVE_PARITY.md`).

## Anlamlı deneyler (A/B önerisi)

1. **Skor eşiği:** `type_c_score >= 9` veya `10` (kodda sabit) — daha az C, potansiyel olarak daha seçici giriş.
2. **`min_quality`:** Aynı dönemde 65 → 70 veya 75; funnel’da `diagnostics` ile birlikte okuyun.
3. **Rejim:** Sadece belirli rejimlerde C’ye izin (ör. yüksek volatilitede C’yi daraltmak) — ek mantık gerektirir.
4. **Sadece C ile koşu:** Aynı ticker/dönemde trade tablosunu `swing_type == 'C'` ile filtreleyip getiri / `exit_stats` dağılımını ayrı not edin; “C mi sızıntı yoksa genel stop yapısı mı?” sorusuna cevap verir.

## Dokümantasyon ile yetinme

Bu dosya, plandaki “Tip C için ayrı deney veya dokümantasyon” maddesini **dokümantasyon** tarafıyla kapatır. Sayısal cevap için yukarıdaki parametre veya kod değişiklikleriyle ikinci bir backtest JSON’u üretip [BACKTEST_AB_BASELINE.md](./BACKTEST_AB_BASELINE.md) akışıyla kıyaslayın.
