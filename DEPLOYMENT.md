# 🚀 Swing Trade — Production Deployment Roadmap

**Temiz kurulum** — Sıfırdan deploy rehberi. Önceki Streamlit deploy silindi, şimdi FastAPI + Next.js.

---

## ☁️ Bulut Kullanımı — Cold Start ve Limitler

### Uygulama tam olarak lokaldeki gibi çalışır mı?
**Evet.** Backend (Fly.io), frontend (Vercel) ve veritabanı (Supabase) birlikte lokaldekiyle aynı işlevselliği sunar.

### Cold start (ilk istek gecikmesi)
Fly.io'da `min_machines_running = 0` kullanıldığı için makine boşta kalınca durur. İlk istek geldiğinde yeniden başlar:

| Durum | Beklenen süre |
|-------|----------------|
| Makine uyandığında ilk API isteği | ~8–15 saniye |
| Sonraki istekler | Normal (~100–300 ms) |

**Çözüm (cold start’ı kaldırmak):** `fly.toml` içinde `min_machines_running = 1` yap. Bir makine sürekli açık kalır, ek maliyet ~$5–7/ay.

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
| Backend (Fly.io) | Var (~8–15 sn) | `min_machines_running = 1` ile kaldırılabilir |
| Frontend (Vercel) | Yok | Statik/SSR, hızlı |
| Supabase | Yok | Her zaman açık |

**Cold start'ı kaldırmak için:** `fly.toml` içinde `min_machines_running = 1` yap, sonra `fly deploy`.

---

## 📌 Venv Ne Zaman Gerekli?

| Adım | Venv gerekli mi? | Neden |
|------|------------------|-------|
| `fly deploy` | ❌ Hayır | Fly.io kendi Docker image'ını build eder, senin venv'in kullanılmaz |
| `fly auth login` | ❌ Hayır | Fly CLI komutu, Python değil |
| `fly secrets set` | ❌ Hayır | Fly CLI komutu |
| `python migrate_to_postgres.py` | ✅ Evet | Lokal Python script, `psycopg2` vb. için venv gerekir |
| `vercel --prod` (frontend) | ❌ Hayır | Node/npm komutu, Python değil |

**Özet:** Sadece `migrate_to_postgres.py` çalıştırırken venv aktif olmalı. `fly deploy` için venv'e girmene gerek yok.

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

**Tüm komutlar proje kökünde (`C:\swing_trade`). Venv'e girmene gerek yok.**

```powershell
cd C:\swing_trade

# 1. Fly.io'ya giriş (tarayıcı açılır)
fly auth login

# 2. Uygulama oluştur
#    - "Use existing app" veya "Create new app" → swing-trade
#    - fly.toml zaten var, config sorulursa "Yes" de
fly launch --no-deploy

# 3. Secrets ayarla (Supabase connection string'ini yapıştır)
fly secrets set DATABASE_URL="postgresql://postgres:SIFRE@db.PROJE_ID.supabase.co:5432/postgres"

# 4. Opsiyonel: GenAI için
fly secrets set OPENAI_API_KEY="sk-..."

# 5. Deploy
fly deploy
```

**Sonuç:** Backend `https://swing-trade.fly.dev` adresinde çalışır.

**Test:**
```powershell
curl https://swing-trade.fly.dev/api/health
```

---

### ADIM 3: Vercel — Frontend Deploy

**Frontend klasöründe. Node/npm kullanılır, venv yok.**

```powershell
cd C:\swing_trade\frontend

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
cd C:\swing_trade

fly secrets set CORS_ORIGINS="https://swing-trade-xxx.vercel.app"
```

Birden fazla domain varsa virgülle ayır:
```
fly secrets set CORS_ORIGINS="https://app.vercel.app,https://swing-trade.com"
```

---

### ADIM 5 (Opsiyonel): Mevcut SQLite Verisini Taşı

**Sadece** daha önce lokal SQLite'ta trade'ler varsa ve bunları Supabase'e taşımak istiyorsan:

```powershell
cd C:\swing_trade

# Venv aktif et (bu komut için gerekli)
venv\Scripts\activate

# DATABASE_URL ayarla (PowerShell)
$env:DATABASE_URL = "postgresql://postgres:SIFRE@db.PROJE_ID.supabase.co:5432/postgres"

# Migrate çalıştır
python migrate_to_postgres.py

# Venv'den çık
deactivate
```

**Temiz kurulumda** bu adımı atla — uygulama ilk çalıştığında tabloları kendisi oluşturur.

---

## 📋 Hızlı Referans

| Komut | Nerede | Venv |
|-------|--------|------|
| `fly auth login` | `C:\swing_trade` | ❌ |
| `fly launch --no-deploy` | `C:\swing_trade` | ❌ |
| `fly secrets set DATABASE_URL="..."` | `C:\swing_trade` | ❌ |
| `fly deploy` | `C:\swing_trade` | ❌ |
| `fly logs` | `C:\swing_trade` | ❌ |
| `python migrate_to_postgres.py` | `C:\swing_trade` | ✅ |
| `npm install` | `C:\swing_trade\frontend` | ❌ |
| `npx vercel --prod` | `C:\swing_trade\frontend` | ❌ |

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
