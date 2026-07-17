# 📈 Swing Trading System — AI-Powered

Kişisel kullanım için geliştirilmiş, üretim kalitesinde Python tabanlı swing trade sistemi. Teknik analiz, risk yönetimi, paper trading, backtesting ve **çok katmanlı yapay zeka** (XGBoost + Generative AI) içerir.

---

## 🧠 AI Mimarisi — İki Katmanlı

Bu sistem, iki farklı AI yaklaşımını **hibrid bir mimariyle** birleştirir:

| Katman | Teknoloji | Ne Yapar |
|--------|-----------|----------|
| **Katman 1** | XGBoost (Klasik ML) | Geçmiş trade verilerinden win probability tahmini (**%72 kazanma ihtimali** gibi) |
| **Katman 2** | LLM / GenAI | Sayıları Türkçe yoruma çevirir — haftalık rapor, sinyal brifingi, strateji Q&A |

> **Hibrid Prensip:** Tüm hesaplamalar (R/R, ATR%, profit factor, risk%) Python'da yapılır. LLM sadece yorumlama ve raporlama yapar — asla sayısal karar vermez.

---

## 🎯 Özellikler

### 🔬 Klasik Teknik Analiz
- **Otomatik Hisse Tarama**: 200+ hisse, 15 saniyede
- **15+ Gösterge**: RSI, MACD, EMA (20/50/200), ADX, OBV, VWAP, ATR, Bollinger
- **Çoklu Faktörlü Skorlama**: 0–100+ skala
- **Risk Yönetimi**: Position sizing, stop-loss, take-profit otomasyonu
- **Backtesting Motoru**: Geçmiş verilerle strateji testi

### 🚀 SmallCap Momentum Sistemi (Senior Trader v2.1)
- **4 Tip Sınıflandırma**: S (Squeeze), C (Erken), B (Momentum), A (Devam)
- **Float Tiering**: Atomic (≤15M), Micro (15-30M), Small (30-50M), Tight (50-60M)
- **RSI Bullish Divergence**: Erken dönüş tespiti
- **Kısa Sıkışma Tespiti**: SI ≥ %20, Days-to-Cover ≥ 5
- **Sektor RS Analizi**: Sektör liderlerini öne çıkarır (+12 bonus)
- **Finviz Entegrasyonu**: Canlı momentum evreni

### 🤖 Generative AI Özellikleri
| Özellik | Tetiklenme | Çıktı |
|---------|-----------|-------|
| **📝 Haftalık Rapor** | Performance sekmesi → "Rapor Oluştur" | Trader tarzı Türkçe özet + stratejik tavsiye |
| **🤖 Sinyal Brifingi** | Manual Lookup → hisse tarandığında | 2-3 cümle setup yorumu ("R/R güçlü, volatilite yüksek...") |
| **💬 Strateji Danışmanı** | Performance sekmesi → serbest soru | RAG-lite Q&A — tüm trade geçmişi context olarak LLM'e verilir |

> API key olmadan tüm GenAI özellikleri **deterministik fallback** modunda çalışır.

### 🧮 XGBoost ML Sinyal Tahmini
- Geçmiş kapalı trade'lerden eğitilir (`ml/trainer.py`)
- Her sinyal için **win probability** hesaplar (0–100%)
- **SHAP** ile feature importance açıklaması
- Manual Lookup'ta XGBoost badge olarak gösterilir: `🤖 AI Tahmin: %72 — High Confidence`
- 50+ kapalı trade sonrası aktif hale gelir

### 📊 Paper Trading Sistemi
- **Ertesi Gün Onay Mekanizması**: Sinyaller PENDING olarak girer, ertesi gün Open fiyatında onaylanır
- **Gap Filtresi**: Gap-up > +5% veya Gap-down > -3% → Otomatik REJECT
- **Modern Kart Arayüzü**: Her bekleyen sinyal için Entry / Stop / Target / R/R metrikleri
- **Manuel Onay/Ret Butonları**: Tek tıkla ✅ Onayla veya ❌ İptal Et
- **Trailing Stop**: ATR bazlı, pozisyon olgunlaşınca aktif
- **Otomatik Kapanma**: Target hit, Stop hit, Timeout, Trailing stop

---

## 🚀 Kurulum

### 1. Gereksinimler
- Python 3.8+
- Internet bağlantısı (piyasa verisi için)

### 2. Kurulum
```bash
git clone https://github.com/Yigit033/swing_trade.git
cd swing_trade

python -m venv venv
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### 3. `.env` Dosyası (Opsiyonel — GenAI için)
```env
# AI Provider (ikisinden birini seç)
LLM_PROVIDER=gemini          # veya: openai

# API Keyler
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
```
> API key olmadan sistem çalışmaya devam eder — GenAI özellikleri deterministik fallback kullanır.

### 4. Uygulamayı Başlat
```bash
# Backend (FastAPI)
uvicorn api.main:app --reload --port 8000

# Frontend (Next.js) — ayrı terminalde
cd frontend && npm run dev
```
Tarayıcıda `http://localhost:5000` açılır. (Windows'ta `SwingTrade_Dashboard.bat` ikisini birden başlatır.)

---

## 🖥️ Dashboard Sayfaları (Next.js)

| Sayfa | İçerik |
|-------|--------|
| **🚀 Scanner** | SmallCap momentum tarama (arka plan job + canlı ilerleme) |
| **🗂 Scanner History** | Geçmiş taramalar + forward-return takibi |
| **📝 Manual Lookup** | Tek hisse, aşama aşama tanı (filtre → tetik → skor) |
| **📊 Paper Trades** | Aktif/Bekleyen/Kapalı trade takibi, pending onay akışı |
| **📉 Performance** | Win rate, profit factor, haftalık rapor |
| **⚙️ Settings** | Motor parametreleri (JSON tabanlı, UI'dan düzenlenir) |
| **🤖 AI / Chat** | XGBoost eğitim-tahmin + strateji sohbeti |

---

## 📁 Proje Yapısı

```
swing_trade/
├── api/                   # FastAPI backend (9 router)
├── frontend/              # Next.js dashboard
├── swing_trader/
│   ├── genai/             # Generative AI modülleri
│   │   ├── llm_client.py      # OpenAI/Gemini provider-agnostic client
│   │   ├── prompts.py         # Tüm prompt builder'lar
│   │   ├── reporter.py        # Haftalık rapor orchestrator
│   │   ├── signal_briefer.py  # Sinyal brifingi orchestrator
│   │   ├── strategy_chat.py   # Strateji Q&A orchestrator
│   │   └── data_collector.py  # DB'den deterministik veri toplama
│   ├── ml/                # Klasik ML
│   │   ├── trainer.py         # XGBoost eğitim
│   │   ├── predictor.py       # Win probability tahmini
│   │   └── features.py        # Feature engineering
│   ├── small_cap/         # SmallCap Momentum motoru
│   │   ├── engine.py          # Ana tarama motoru
│   │   ├── scoring.py         # Kalite skoru (0-100+)
│   │   ├── narrative.py       # Metin analiz üretimi
│   │   └── risk.py            # ATR bazlı risk hesaplamaları
│   ├── paper_trading/     # Paper Trade sistemi
│   │   ├── tracker.py         # Trade takibi, gap filtresi, trailing stop
│   │   ├── storage.py         # SQLite CRUD operasyonları
│   │   └── reporter.py        # Performans özeti
│   └── data/              # Fetcher + tarama geçmişi storage'ları
├── data/
│   └── paper_trades.db    # SQLite veritabanı
├── config.yaml            # Ayarlar
├── requirements.txt
└── .env                   # API keyleri (git'e commit edilmez)
```

---

## 📋 Günlük İş Akışı

```bash
# 1. Dashboard'u başlat (backend + frontend)
SwingTrade_Dashboard.bat   # veya: uvicorn api.main:app --port 8000  +  cd frontend && npm run dev

# 2. SmallCap veya Manual Lookup'tan sinyal tara
# → Beğendiğin sinyalli "📌 Track" ile PENDING'e ekle

# 3. Ertesi gün Update Prices'a bas
# → Sistem ertesi gün açılış fiyatını çeker, gap filtresini uygular
# → Onaylananlar OPEN olur, reddedilenler REJECTED olarak kayıt düşer

# 4. Aktif trade'leri takip et
# → Stop/Target/Timeout otomatik kapanma
# → Trailing stop devreye girer (ATR bazlı, pozisyon olgunlaşınca)

# 5. Performance sekmesinde haftalık rapor al
# → "Rapor Oluştur" → LLM, trade geçmişini analiz eder
# → Strateji Danışmanı'na serbest soru sor
```

---

## 🔧 ML Model Eğitimi

XGBoost modeli 50+ kapalı trade'den sonra eğitilebilir:

```bash
# Dashboard'da → 🤖 AI Model sekmesi → "Modeli Eğit"
# Veya:
python -c "from swing_trader.ml.trainer import ModelTrainer; ModelTrainer().train()"
```

Eğitim sonrası Manual Lookup'ta `🤖 AI Tahmin: %XX — Confidence` badge'i görünür.

---

## 🛡️ Risk Parametreleri

| Parametre | Large Cap | SmallCap |
|-----------|-----------|---------|
| Max risk / trade | %2 | %0.5 |
| Max pozisyon | %20 portföy | %5 portföy |
| Stop loss | ATR × 2.0 | ATR × 1.0 (cap %12) |
| Gap-up reject | — | > %5 |
| Gap-down reject | — | < -%3 |
| Trailing stop | — | ATR bazlı (2+ ATR kârda) |

---

## ⚠️ Risk Uyarısı

> Hisse senedi alım satımı önemli kayıp riski taşır. Geçmiş performans gelecekteki sonuçları garanti etmez. Bu yazılım yalnızca eğitim amaçlıdır. Yatırım kararları almadan önce bir finansal danışmana başvurun.

**Gerçek para kullanmadan önce en az 3 ay paper trading yapın.**

---

## 📄 Sürüm Geçmişi

| Versiyon | İçerik |
|----------|--------|
| v3.0 | GenAI özellikleri (Signal Briefer, Strategy Chat, Weekly Reporter), modern Pending UI, kalite skoru düzeltmeleri |
| v2.1 | SmallCap Senior Trader: sektor RS, insider bonus, kısa sıkışma, trailing stop |
| v2.0 | Paper Trading sistemi, gap filtresi, ertesi gün onay |
| v1.5 | XGBoost ML sinyal tahmini, SHAP feature importance |
| v1.0 | Temel tarama, teknik analiz, backtesting, ilk web dashboard |
