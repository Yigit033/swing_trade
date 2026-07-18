# 🚀 SmallCap Swing Trade — Günlük Kullanım Akışı

Modern stack: **Next.js dashboard** + **FastAPI backend**. İstersen **Supabase ile giriş**; yapılandırılmadıysa uygulama girişsiz de çalışır (`docs/AUTH_SETUP.md`).

---

## Hızlı başlatma (lokal)

```bash
# 1) Backend (ayrı terminal)
cd swing_trade
venv\Scripts\activate          # Windows
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 2) Frontend (ayrı terminal)
cd frontend
npm install   # ilk sefer
npm run dev   # http://localhost:5000
```

`.env` / `frontend/.env.local` içinde **`NEXT_PUBLIC_API_URL`** (örn. `http://localhost:8000`) tanımlı olmalı. Üretimde Vercel + Fly benzeri URL’ler kullanılır.

---

## 📋 Günlük akış (5 adım)

### 1. 📥 Veri
- Ayrı bir veri indirme adımı **yok** — tarama sırasında güncel OHLCV doğrudan çekilir (yfinance → Tiingo → Finnhub fallback, tamamlanmamış gün barı otomatik atılır).

### 2. 🔍 Tarama yap
- Soldan **Scanner** → **Run Scan** (veya eşdeğer tarama butonu).
- Tarama **arka planda job** olarak çalışır; üstte **ScannerScanBanner** ilerlemeyi gösterir.
- **Job id** `localStorage`’da tutulur: tarama sürerken **yeni sekmede** aynı siteyi açsan da polling devam eder (aynı tarayıcı / origin).
- Akış: Finviz → filtreler → sinyaller (backend’deki scanner pipeline).
- **⏰ Tarama penceresi (kritik):** Günlük bar kuralları yalnız **tamamlanmış** barda karar verir:
  - ✅ **Kanonik pencere: ABD kapanışı sonrası (23:05+ TR saati)** — sinyal bugünün barına aittir; giriş, ölçülen disiplinle **yarınki açılışta** (t+1 open) onaylanır.
  - ⚠️ **Seans içi** tarama: sinyaller DÜNÜN barına aittir; ölçülen giriş (bugünkü açılış) geçtiği için Track edilemez (UI: `entry window missed`).
  - ⚠️ **Pre-market** tarama: giriş penceresi geçerlidir ama Finviz RelVol/Change sorguları boş döner → evren zayıf kalır (UI uyarı gösterir).

### 3. 📊 Sinyalleri incele
Her sinyalde tipik olarak:
- **Swing type** (A/B/C/S) + kalite skoru  
- **Entry / Stop / Target**  
- **Hold days** (min–max)

### 4. 📌 Paper trading’e ekle
- Beğendiğin sinyalde **Track** (veya listedeki ekleme aksiyonu).
- Trade **Pending** olarak kaydolur → **Pending** sayfasından yönet.
- Ertesi gün **fiyat güncelleme** akışı (gap kuralları):
  - ✅ Gap makul → açılıştan **OPEN**
  - ❌ Gap çok yukarı/aşağı → **REJECTED** (eşikler backend stratejisine göre)
- Stop/target ATR ile bir sonraki oturuma göre yeniden hesaplanabilir.

### 5. 📈 Takip et
- **Paper Trades**: açık pozisyonlar ve geçmiş.
- Sistem mantığı (özet):
  - 🔴 **Stop** → kapanış (gap-down slippage dahil edilebilir)
  - 🔒 **Trailing stop** → ATR bazlı kâr koruması
  - 🎯 **Target** → hedefe ulaşınca kapanış
  - ⏰ **Timeout** → max gün dolunca kapanış
- İstersen **manuel kapatma** (UI’deki close aksiyonu).

---

## 📊 Performans ve görünürlük

| Ne | Nerede |
|----|--------|
| Özet metrikler (win rate, P/L, vb.) | **Performance** |
| Günlük özet / portföy özeti | **Dashboard** (`/`) |
| Grafikler | **Charts** |

---

## 🔧 Opsiyonel (aynı sidebar)

| Özellik | Sayfa |
|---------|--------|
| Tek hisse hızlı analiz | **Manual Lookup** |
| Strateji geçmişi testi | **Backtest** |
| Strateji / piyasa sohbeti | **AI Chat** |

CLI backtest: `python test_backtest.py` (projede varsa).

---

## 🔐 Giriş (opsiyonel)

Supabase açıksa: `/login` → oturum; **Paper Trades** RLS ile kullanıcıya özel olabilir. Auth yoksa eski davranış: tek kullanıcı / ortak veri (kurulumuna bağlı).

