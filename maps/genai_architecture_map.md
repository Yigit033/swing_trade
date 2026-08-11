# 🧠 Swing Trade AI — GenAI Mimari Haritası

> Bu doküman senin GenAI sisteminin nasıl çalıştığını tek sayfada anlatır.
> Unuttuğunda buraya bak.

---

## Büyük Resim: Hybrid Architecture

```
╔══════════════════════════════════════════════════════════════╗
║                     SENİN SİSTEMİN                          ║
║                                                              ║
║   [Deterministik Katman]  →  [LLM Katman]  →  [Kullanıcı]   ║
║   Python hesaplar            Gemini yorumlar    Dashboard     ║
║   (kesin sayılar)            (insan dili)       (UI)          ║
╚══════════════════════════════════════════════════════════════╝
```

**Temel Kural:** LLM hiçbir zaman hesaplama yapmaz. Python hesaplar, LLM sadece yorumlar.

---

## Dosya Haritası — Kim Ne Yapıyor?

```
swing_trader/genai/
├── llm_client.py      → 🔌 LLM bağlantısı (Gemini veya OpenAI)
├── prompts.py          → 📝 LLM'e gönderilen prompt şablonları  
├── data_collector.py   → 📊 SQLite'dan veri toplayan katman
├── reporter.py         → 📰 Haftalık rapor orchestrator
├── signal_briefer.py   → ⚡ Tek sinyal yorumu (2-3 cümle)
└── strategy_chat.py    → 💬 Soru-cevap chat orchestrator

swing_trader/ml/
├── features.py         → 🔧 Ham veriyi ML özelliklerine çevirir
├── trainer.py          → 🎓 XGBoost modelini eğitir
├── predictor.py        → 🎯 Eğitilmiş modelle tahmin yapar
└── explainer.py        → 🔍 SHAP ile "neden bu tahmin?" açıklar

api/routers/genai.py    → 🌐 Frontend'in çağırdığı API endpoint'leri
frontend/src/app/chat/  → 💻 Chat sayfası UI (4 tab)
```

---

## 3 Ana Özellik — Veri Akışı

### 1️⃣ Strategy Chat (AI ile sohbet)

**Sen soruyorsun → AI trade verilerine bakıp cevaplıyor**

```
[Kullanıcı sorusu]
        ↓
[strategy_chat.py] → StrategyChat.ask(soru)
        ↓
[data_collector.py] → SQLite'dan tüm trade'leri topla
        ↓                  (win rate, P/L, tip bazında istatistik)
[prompts.py] → build_strategy_chat_prompt(soru, context)
        ↓         Soru + trade verileri tek bir string'e birleşir
[llm_client.py] → client.complete(prompt)
        ↓            Gemini'ye gönder, cevabı al
[Frontend] → Chat baloncuğunda göster
```

**Önemli:** LLM'e gönderilen veri sadece ilk 10 trade (token limiti için).
Bu "RAG-lite" — gerçek RAG'da vektör DB kullanılır, burada veri küçük olduğu için direkt prompt'a koyuyorsun.

---

### 2️⃣ Haftalık Rapor

**Butona bas → AI son 7 günü analiz etsin**

```
[Butona tıkla]
        ↓
[reporter.py] → WeeklyReporter.generate()
        ↓
  Önbellek var mı? → Evet → Önbellekten döndür (aynı gün tekrar API çağırmaz)
        ↓ Hayır
[data_collector.py] → Son 7 günün trade verilerini topla
        ↓
[prompts.py] → build_weekly_report_prompt(context)
        ↓
[llm_client.py] → client.complete(prompt) — Gemini'ye gönder
        ↓
  Raporu önbelleğe kaydet (data/genai_cache/weekly_report.json)
        ↓
[Frontend] → Markdown rapor göster
```

**Önemli:** Önbellek günlük bazlı — aynı gün tekrar butona basarsan cache'den gelir.
"Raporu Yenile" → `force_refresh=True` → cache'i atlar.

---

### 3️⃣ Sinyal Yorumu (Signal Briefing)

**Scanner bir sinyal buldu → AI 2-3 cümle yorum yazsın**

```
[Scanner sinyali] → {entry: $15, stop: $13.50, target: $19, quality: 8}
        ↓
[signal_briefer.py] → SignalBriefer.brief(signal)
        ↓
[prompts.py] → build_signal_briefing_prompt(signal)
        ↓       Python R/R, risk%, ATR% hesaplar → LLM'e hazır verir
[llm_client.py] → client.complete(prompt) — maks 200 token, kısa yorum
        ↓
[Frontend] → Sinyal kartında "AI Yorum" kutusu
```

**Önemli:** Cache YOK — her sinyal anlık değerlendirme ister.

---

## LLM Client Nasıl Çalışıyor?

```python
# llm_client.py — Tek satırda özetlersek:
client = LLMClient()                    # .env'den provider + key okur
response = client.complete(prompt)      # Gemini'ye gönder, string al

# Desteklenen providerlar:
#   LLM_PROVIDER=gemini  → gemini-2.5-flash kullanır
#   LLM_PROVIDER=openai  → gpt-4o-mini kullanır
```

**Şu anki config:** Gemini 2.5 Flash ✅

---

## ML Modeli (XGBoost) — Bonus

Bu GenAI değil, klasik ML. LLM ile alakası yok.

```
[Geçmiş trade'ler] → [features.py] → özellik çıkar (R/R, ATR%, kalite...)
                              ↓
                      [trainer.py] → XGBoost eğit, .pkl kaydet
                              ↓
                      [predictor.py] → Yeni sinyal geldi → %72 WIN tahmini
                              ↓
                      [explainer.py] → "Neden? R/R oranı yüksek olduğu için"
```

**Durum:** 15+ gerçek trade kapattığında model eğitilebilir hale gelir.

---

## 🐛 Tespit Edilen Sorunlar

### Sorun #1: Trade sıralama hatası
- **Dosya:** [data_collector.py](file:///c:/active_projects/swing_trade/swing_trader/genai/data_collector.py#L127) satır 127
- **Ne oluyor:** Trade'ler P/L'ye göre sıralanıyor (en karlı üstte)
- **Sonuç:** Prompt'ta "SON TRADELER" deniyor ama aslında "EN KARLI TRADELER" gidiyor
- **AI etkisi:** "Son 10 trade'iniz hep karlı!" diyor çünkü sadece kazançlıları görüyor

### Sorun #2: Prompt'ta sadece 10 trade
- **Dosya:** [prompts.py](file:///c:/active_projects/swing_trade/swing_trader/genai/prompts.py#L223) satır 223
- **Ne oluyor:** `trades[:10]` — 32 trade'den sadece 10'u LLM'e gidiyor
- **Sonuç:** AI eksik veriyle yorum yapıyor

---

## API Endpoint'leri

| Endpoint | Method | Ne Yapıyor |
|----------|--------|------------|
| `/api/genai/chat` | POST | Strategy chat — soru sor, cevap al |
| `/api/genai/weekly-report-ai` | GET | Haftalık AI raporu üret |
| `/api/genai/signal-brief` | POST | Tek sinyal için AI yorum |
| `/api/genai/model-status` | GET | ML model durumu |
| `/api/genai/train` | POST | ML modeli eğit |
| `/api/genai/predict` | POST | Sinyal tahmini yap |
