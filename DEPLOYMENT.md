# 🚀 Swing Trade — Production Deployment Roadmap

**Temiz kurulum** — Sıfırdan deploy rehberi: FastAPI (Fly.io) + Next.js (Vercel).

---

## ☁️ Bulut Kullanımı — Cold Start ve Limitler

### Uygulama tam olarak lokaldeki gibi çalışır mı?
**Evet.** Backend (Fly.io), frontend (Vercel) ve veritabanı (Supabase) birlikte lokaldekiyle aynı işlevselliği sunar.

### Cold start (ilk istek gecikmesi)
`fly.toml`'da `auto_stop_machines = 'suspend'` kullanılır (2026-07-18): makine boşta
RAM anlık görüntüsüyle dondurulur, ilk istekte **~1–2 saniyede** geri döner — pratikte
cold start hissedilmez, ek maliyet yok.

| Durum | Beklenen süre |
|-------|----------------|
| Suspend'den uyanış (normal günlük kullanım) | ~1–2 saniye |
| Deploy sonrası İLK soğuk boot (Python import zinciri) | ~9–15 sn (health check `grace_period=30s` karşılar, kullanıcı görmez) |
| Sonraki istekler | Normal (~100–300 ms) |

**⚠️ `'stop'` moduna geri dönme:** stop'ta her uyanış tam Python boot maliyeti
(pandas+numpy+yfinance import zinciri) öder; fly proxy ~8 sn'de pes edip kullanıcıya
"Fly proxy waited too much" hatası gösterir (PM05). Bu yaşandı ve suspend ile çözüldü.
Alternatif: `min_machines_running = 1` (makine hep açık, ~$5–7/ay) — suspend varken gereksiz.

### Maliyet ve "Dolmaması" İçin
Fly.io artık geleneksel ücretsiz plan sunmuyor; **pay-as-you-go** kullanılıyor. Ancak:

- **$5 altı faturalar iptal edilir** — Aylık kullanım $5'ın altındaysa ücret alınmaz.
- **Senin kullanımın (min_machines_running = 0):**
  - Makine boşta: ~$0.15/ay (rootfs)
  - Çalışırken: shared-cpu-1x 1GB ≈ $0.0082/saat
  - Örnek: Günde 10 dk kullanım ≈ 5 saat/ay ≈ $0.04
  - **Toplam: ~$0.20–0.50/ay** → Fatura iptal, ücret yok

**Dolmaması için:**
- `min_machines_running = 0` kalsın (auto-stop açık)
- `min_machines_running = 1` yapma (sürekli çalışır, ~$6/ay)
- Gereksiz makine/volume oluşturma
- **Vercel:** Ücretsiz plan yeterli
- **Supabase:** Ücretsiz plan yeterli

### Özet
| Bileşen | Cold start | Not |
|---------|------------|-----|
| Backend (Fly.io) | ~1–2 sn (suspend'den uyanış) | Yalnız deploy sonrası ilk boot yavaş, o da health check arkasında |
| Frontend (Vercel) | Yok | Statik/SSR, hızlı |
| Supabase | Yok | Her zaman açık |

---

## 📌 Venv Ne Zaman Gerekli?

**Deploy için hiçbir zaman.** Fly.io kendi Docker image'ını build eder; Vercel Node kullanır.
Venv yalnız **lokal geliştirme** içindir (backend'i lokal çalıştırmak, `pytest`, `scripts/` altındaki
ölçüm harness'ları).

> Not: Eski SQLite→Supabase göçü (`migrate_to_postgres.py`) tamamlandı ve script silindi.
> Temiz kurulumda tabloları uygulama ilk bağlantıda kendisi oluşturur.

---

## 🗺️ Adım Adım Roadmap

### ADIM 0: Hazırlık (Bir kez)

1. **Fly CLI** kur (Windows PowerShell):
   ```powershell
   iwr https://fly.io/install.ps1 -useb | iex
   ```
   Kurulumdan sonra **PowerShell'i kapatıp yeniden aç**. Sonra `fly version` ile test et.
2. **Supabase** hesabı: https://supabase.com
3. **Vercel** hesabı: https://vercel.com (frontend için)

---

### ADIM 1: Supabase — Veritabanı Oluştur

1. [Supabase Dashboard](https://supabase.com/dashboard) → **New Project**
2. Proje adı, şifre belirle
3. **Settings → Database** → **Connection string** → **URI** kopyala  
   Örnek: `postgresql://postgres:XXXXX@db.xxxxx.supabase.co:5432/postgres`
4. **Not:** Tabloları sen oluşturmazsın — uygulama ilk bağlandığında otomatik oluşturur.

---

### ADIM 2: Fly.io — Backend Deploy

**Tüm komutlar proje kökünde (`C:\active_projects\swing_trade`). Venv'e girmene gerek yok.**

```powershell
cd C:\active_projects\swing_trade

# 1. Fly.io'ya giriş (tarayıcı açılır)
fly auth login

# 2. Uygulama oluştur
#    - "Use existing app" veya "Create new app" → swing-trade
#    - fly.toml zaten var, config sorulursa "Yes" de
fly launch --no-deploy

# 3. Secrets ayarla (Supabase connection string'ini yapıştır)
fly secrets set DATABASE_URL="postgresql://postgres:SIFRE@db.PROJE_ID.supabase.co:5432/postgres"

# 4. Auth (login kullanacaksan — Supabase Dashboard → Settings → API)
fly secrets set SUPABASE_URL="https://PROJE_ID.supabase.co"
fly secrets set SUPABASE_ANON_KEY="eyJhbG..."
fly secrets set SUPABASE_JWT_SECRET="jwt-secret-değeri"

# 5. CORS — Vercel deploy sonrası frontend URL'ini ekle (ADIM 4'te)
# fly secrets set CORS_ORIGINS="https://swing-trade-xxx.vercel.app"

# 6. Opsiyonel: GenAI için
fly secrets set OPENAI_API_KEY="sk-..."

# 7. Deploy
fly deploy
```

**Sonuç:** Backend `https://swing-trade.fly.dev` adresinde çalışır.

**Test:**
```powershell
curl https://swing-trade.fly.dev/api/health
```

---

### ⚙️ ADIM 2.5: Otomatik Deploy — CI (günlük yol, önerilen)

İlk kurulumdan sonra elle `fly deploy` gerekmez: **`main`'e her push,**
`.github/workflows/fly-deploy.yml` üzerinden **otomatik deploy olur**
(`flyctl deploy --remote-only`).

Tek ön koşul: GitHub repo secret'ı **`FLY_API_TOKEN`**:

```powershell
# 1. Deploy token üret (çıkan "FlyV1 ..." satırının TAMAMINI kopyala)
fly tokens create deploy -a swing-trade
```

2. GitHub → repo → **Settings → Secrets and variables → Actions → New repository secret**
   → Name: `FLY_API_TOKEN`, Value: kopyaladığın token.

**Belirti tanıma:** Actions'ta deploy `no access token available. Please login with
'flyctl auth login'` ile düşüyorsa bu secret eksik/bozuk demektir — aynı adımlarla yenile
(2026-07-18'de yaşandı, çözüm buydu).

---

### ADIM 3: Vercel — Frontend Deploy

**Frontend klasöründe. Node/npm kullanılır, venv yok.**

```powershell
cd C:\active_projects\swing_trade\frontend

# 1. Bağımlılıklar
npm install

# 2. Vercel CLI ile deploy (veya Dashboard'dan)
npx vercel --prod
```

**Vercel Dashboard kullanıyorsan:**
1. **New Project** → Repo'yu import et (`Yigit033/swing_trade`)
2. **Root Directory:** `frontend` seç (zorunlu — repo kökünde Next.js yok)
3. **Framework Preset:** Next.js (otomatik algılanır)
4. **Environment Variables:**
   - `NEXT_PUBLIC_API_URL` = `https://swing-trade.fly.dev`
5. **Deploy**

---

### ADIM 4: CORS — Backend'e Frontend URL Ekle

Frontend Vercel'de yayına alındıktan sonra (örn: `https://swing-trade-xxx.vercel.app`):

```powershell
cd C:\active_projects\swing_trade

fly secrets set CORS_ORIGINS="https://swing-trade-xxx.vercel.app"
```

Birden fazla domain varsa virgülle ayır:
```
fly secrets set CORS_ORIGINS="https://app.vercel.app,https://swing-trade.com"
```

---

## 📋 Hızlı Referans

| Komut | Nerede | Venv |
|-------|--------|------|
| `git push` (→ otomatik deploy, CI) | proje kökü | ❌ |
| `fly auth login` | proje kökü | ❌ |
| `fly tokens create deploy -a swing-trade` | proje kökü | ❌ |
| `fly secrets set DATABASE_URL="..."` | proje kökü | ❌ |
| `fly deploy` (manuel, gerekirse) | proje kökü | ❌ |
| `fly logs` | proje kökü | ❌ |
| `npm install` / `npx vercel --prod` | `frontend/` | ❌ |
| `pytest`, `scripts/` harness'ları (lokal) | proje kökü | ✅ |

---

## 🔧 Sorun Giderme

### "Database connection failed"
- `fly secrets list` ile DATABASE_URL'in set olduğunu kontrol et
- Supabase: **Settings → Database** → Connection string'de **Direct connection** kullan (5432 portu)

### CORS hatası (tarayıcıda)
- `CORS_ORIGINS` Vercel URL'ini içermeli (örn: `https://swing-trade-abc123.vercel.app`)
- Trailing slash olmasın

### 502 Bad Gateway
- `fly logs` ile logları incele
- `fly status` ile uygulama durumunu kontrol et

### Vercel build hatası (useSearchParams / Suspense)
- Bu hata düzeltildi (charts sayfası Suspense ile sarıldı)
- Yine de hata alırsan: Root Directory `frontend` olmalı
