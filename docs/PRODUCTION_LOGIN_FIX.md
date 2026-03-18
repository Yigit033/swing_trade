# Production'da Login Sonrası Yönlendirme Sorunu

Local'de çalışıyor, production'da (https://swingtrade.vercel.app) giriş sonrası tekrar login'e atıyorsa, aşağıdaki adımları kontrol edin.

## Kod Düzeltmesi (Yapıldı)

Login sonrası `router.push("/")` yerine `window.location.href = "/"` kullanılıyor. Client-side navigation production'da cookie'leri gecikmeli gönderebiliyor; full page redirect ile session kesin sunucuya ulaşır.

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

## 4. Fly.io Auth Secrets

**Kontrol:** Backend'de auth env'leri set mi?

```powershell
fly secrets list
```

Şunlar olmalı:
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_JWT_SECRET`
- `CORS_ORIGINS`
- `DATABASE_URL`

Eksikse:
```powershell
fly secrets set SUPABASE_URL="https://xxx.supabase.co"
fly secrets set SUPABASE_ANON_KEY="eyJhbG..."
fly secrets set SUPABASE_JWT_SECRET="..."
fly secrets set CORS_ORIGINS="https://swingtrade.vercel.app"
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
