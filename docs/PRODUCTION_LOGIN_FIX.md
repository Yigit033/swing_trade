# Production'da Login Sonrası Yönlendirme Sorunu

Local'de çalışıyor, production'da (https://swingtrade.vercel.app) giriş sonrası tekrar login'e atıyorsa, aşağıdaki adımları kontrol edin.

## 401 Unauthorized — Ana Kaynak

**Belirti:** Login butonuna bastıktan sonra console'da "401 Unauthorized" görünüyor ve hemen login ekranına dönülüyor.

**Gerçek sebep:** 401 **backend'den** (Fly.io) geliyor. Akış:
1. Giriş başarılı → `window.location.href = "/"` ile dashboard'a yönleniyorsun
2. Dashboard `/api/performance`, `/api/pending` vb. çağırıyor
3. Backend token'ı doğrulayamıyor → 401 dönüyor
4. Frontend 401'de sign out + `/login` redirect yapıyor

**Debug adımları:**
1. Console'da `[Auth 401] Backend token reddetti. Detail: ...` mesajına bak — `Missing authorization header` mı, `Invalid or expired token` mı?
2. `https://swing-trade.fly.dev/api/auth/status` aç — `auth_configured: true`, `cors_origins_count > 0` olmalı
3. Fly.io logs: `fly logs` — "Auth 401: Missing..." veya "Auth 401: Token rejected..." satırlarını ara

**Çözüm:** Bölüm 1 (CORS) ve Bölüm 4 (Fly.io secrets) mutlaka doğrulanmalı.

---

## Custom Domain vs Deployment URL

**Belirti:** `https://swing-trade-p7lczngci-yigit033s-projects.vercel.app` üzerinden giriş çalışıyor, ama `https://swingtrade.vercel.app` üzerinden giriş sonrası tekrar login'e atıyor.

**Neden:** İki farklı origin. Cookie'ler domain'e göre izole; deployment URL'de set edilen cookie'ler custom domain'e gönderilmez. Ayrıca Vercel CDN, auth-dependent yanıtları (ör. `/` → `/login` redirect) cache'leyebilir; cookie'siz ilk istek cache'lenirse, cookie'li sonraki istekler aynı redirect'i alır.

**Yapılan düzeltmeler:**
- Middleware: Tüm auth-dependent yanıtlara `Cache-Control: private, no-store` ve `Vercel-CDN-Cache-Control: max-age=0` eklendi.
- `next.config.ts`: Root ve korumalı path'ler için no-cache headers tanımlandı.

**Önemli:** Her zaman **custom domain** (`swingtrade.vercel.app`) üzerinden test edin. Deployment URL sadece preview için kullanılır.

---

## Kod Düzeltmesi (Yapıldı)

Login sonrası `router.push("/")` yerine `window.location.href = "/"` kullanılıyor. Client-side navigation production'da cookie'leri gecikmeli gönderebiliyor; full page redirect ile session kesin sunucuya ulaşır.

---

## Vercel Framework Settings Uyarısı

"Configuration Settings in the current Production deployment differ from your current Project Settings" uyarısı görüyorsan:

1. Vercel Dashboard → Project → **Settings** → **General**
2. **Framework Preset:** Next.js olmalı
3. **Build Command:** Override kapalıysa `npm run build` kullanılır
4. Production Overrides ile proje ayarları uyuşmuyorsa, Overrides'ı kaldır veya proje ayarlarıyla eşleştir
5. Değişiklikten sonra **Redeploy** yap

---

## 1. CORS_ORIGINS (En Sık Sebep)

**Sorun:** Backend varsayılan olarak sadece `localhost:3000` kabul eder. Production domain'i (`swingtrade.vercel.app`) izin listesinde değilse API istekleri CORS tarafından engellenir veya 401 döner.

**Çözüm:**
```powershell
cd C:\swing_trade
fly secrets set CORS_ORIGINS="https://swingtrade.vercel.app"
fly deploy
```

Birden fazla domain (preview, custom domain) kullanıyorsan:
```powershell
fly secrets set CORS_ORIGINS="https://swingtrade.vercel.app,https://www.swingtrade.vercel.app"
```

---

## 2. Supabase Redirect URLs

**Sorun:** Supabase, production URL'ini tanımıyorsa session/redirect düzgün çalışmayabilir.

**Çözüm:** [Supabase Dashboard](https://supabase.com/dashboard) → **Authentication** → **URL Configuration**

- **Site URL:** `https://swingtrade.vercel.app`
- **Redirect URLs:** Şunları ekle:
  - `https://swingtrade.vercel.app/**`
  - `https://swingtrade.vercel.app`

---

## 3. Vercel Environment Variables

**Kontrol:** Vercel Dashboard → Project → **Settings** → **Environment Variables**

| Değişken | Production Değeri |
|----------|-------------------|
| `NEXT_PUBLIC_API_URL` | `https://swing-trade.fly.dev` (veya backend URL'in) |
| `NEXT_PUBLIC_SUPABASE_URL` | `https://xxx.supabase.co` |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | `eyJhbG...` |

**Önemli:** `NEXT_PUBLIC_*` değişkenleri build-time'da gömülür. Değiştirdikten sonra **Redeploy** gerekir.

---

## 4. Fly.io Auth Secrets (401'in En Sık Sebebi)

**Kontrol:** Backend token doğrulaması için bu secret'lar **zorunlu**:

```powershell
fly secrets list
```

Şunlar olmalı:
- `SUPABASE_URL` — Supabase proje URL (örn. `https://xxx.supabase.co`)
- `SUPABASE_ANON_KEY` — Frontend ile **aynı** anon key (Supabase Dashboard → Settings → API)
- `SUPABASE_JWT_SECRET` — Supabase Dashboard → Settings → API → **JWT Secret** (JWT Settings bölümü)
- `CORS_ORIGINS` — `https://swingtrade.vercel.app` (virgülle birden fazla eklenebilir)
- `DATABASE_URL` — PostgreSQL connection string

**JWT Secret nerede?** Supabase Dashboard → Project Settings → API → "JWT Settings" → "JWT Secret" (uzun string). Bu değer token imzasını doğrulamak için kullanılır; yanlışsa backend 401 döner.

Eksikse:
```powershell
fly secrets set SUPABASE_URL="https://xxx.supabase.co"
fly secrets set SUPABASE_ANON_KEY="eyJhbG..."
fly secrets set SUPABASE_JWT_SECRET="your-jwt-secret-from-supabase-dashboard"
fly secrets set CORS_ORIGINS="https://swingtrade.vercel.app"
fly deploy
```

---

## 5. Hızlı Test

1. **CORS testi:** Tarayıcıda https://swingtrade.vercel.app aç → F12 → Network
2. Login yap
3. `/api/performance` veya `/api/pending` isteğine tıkla
4. **Status:** 200 ise CORS ve auth OK. 401 veya CORS hatası varsa yukarıdaki adımları kontrol et.

---

## Özet Checklist

- [ ] `fly secrets set CORS_ORIGINS="https://swingtrade.vercel.app"`
- [ ] Supabase → URL Configuration → Site URL + Redirect URLs
- [ ] Vercel → NEXT_PUBLIC_API_URL, SUPABASE_* (sonra Redeploy)
- [ ] Fly.io → SUPABASE_*, CORS_ORIGINS
