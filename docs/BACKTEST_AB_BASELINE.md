# Backtest A/B baseline karşılaştırması

Aynı hisse listesi, dönem ve parametrelerle **eski kod** vs **yeni kod** sonuçlarını sayısal kıyaslamak için kullanın. Kayıtlı eski JSON yoksa “ne kadar iyileşti?” sorusuna tek koşuda cevap verilemez.

## Adımlar

1. **A koşusu (baseline):** İstediğiniz commit’e geçin veya eski sabitleri geçici geri alın.
2. UI veya `POST /api/backtest/smallcap` ile backtest çalıştırın.
3. Tarayıcıda yanıtı veya Network sekmesinden **tam JSON** gövdesini kopyalayıp `run_a.json` olarak kaydedin.
4. **B koşusu (güncel):** `main` / güncel dalda aynı `period_days`, `tickers`, `min_quality`, `top_n`, `max_concurrent`, `initial_capital` ile tekrar çalıştırın.
5. `run_b.json` kaydedin.
6. Karşılaştırma:

```bash
python scripts/compare_backtest_json.py run_a.json run_b.json
```

## Script ne karşılaştırır?

- `metrics`: `total_return`, `total_pnl_dollar`, `win_rate`, `profit_factor`, `max_drawdown`, `total_trades`, `avg_hold_days`
- `diagnostics` (varsa): tüm anahtarlar fark fark
- `exit_stats` (varsa): her `status` için `count` ve ortalama `avg_pnl`
- Trade sayısı

## Notlar

- yfinance verisi zamanla hafif değişebilir; mümkünse aynı gün içinde ardışık koşular tercih edin.
- Finviz evreni yerine **sabit ticker listesi** kullanmak tekrarlanabilirliği artırır.
- Hata dönen JSON (`error` alanı) script tarafından reddedilir.

İlgili: [BACKTEST_LIVE_PARITY.md](./BACKTEST_LIVE_PARITY.md), [SMALLCAP_TYPE_C_REVIEW.md](./SMALLCAP_TYPE_C_REVIEW.md).
