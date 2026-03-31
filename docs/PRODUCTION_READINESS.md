# Production Readiness — Local vs Deployment

Bu dokümanda sistemin production hazırlığı ve local/deployment farkları **kanıtlarla** açıklanır.

---

## 1. Production Ready mi?

### Evet — Aşağıdaki Koşullar Sağlanırsa

| Kriter | Durum | Kanıt |
|--------|-------|-------|
| **Env tabanlı config** | ✅ | `api/main.py:51` CORS_ORIGINS, `api/auth.py` SUPABASE_* env'den okunuyor |
| **Hardcoded secret yok** | ✅ | Tüm hassas veriler `.env` veya `fly secrets` |
| **CORS yapılandırılabilir** | ✅ | `CORS_ORIGINS` ile production URL eklenebilir |
| **Health check** | ✅ | `fly.toml` `/api/health` ile health check tanımlı |
| **HTTPS** | ✅ | `fly.toml:13` `force_https = true` |
| **Exception handling** | ✅ | Global handler 500'de CORS header'ları koruyor |
| **Auth (JWT/JWKS/API)** | ✅ | 3 katmanlı doğrulama, cache'li JWKS |

### Dikkat Edilmesi Gerekenler

| Konu | Açıklama |
|------|----------|
| **CORS_ORIGINS** | Production'da **mutlaka** Vercel URL'i set edilmeli. Yoksa sadece localhost kabul edilir. |
| **Tüm auth env'leri** | Fly.io'da `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_JWT_SECRET` set edilmeli. |
| **Cold start** | `min_machines_running = 0` ile ilk istek ~8–15 sn gecikmeli olabilir. |

---

## 2. Local vs Deployment — Aynı mı?

### Evet, Aynı Davranış — Kanıtlar

| Bileşen | Local | Deployment | Aynı mı? | Kanıt |
|---------|-------|------------|----------|-------|
| **Kod** | Aynı repo | Aynı repo | ✅ | Tek codebase |
| **API mantığı** | FastAPI | FastAPI | ✅ | `api/main.py` |
| **Auth** | Supabase Auth | Supabase Auth | ✅ | Aynı Supabase projesi |
| **Veritabanı** | Supabase PG | Supabase PG | ✅ | `DATABASE_URL` aynı DB'ye |
| **JWT doğrulama** | JWT → JWKS → API | JWT → JWKS → API | ✅ | `api/auth.py` aynı sıra |

### Farklı Olan (Sadece Config)

| Değişken | Local | Production |
|----------|-------|------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | `https://swing-trade.fly.dev` |
| `CORS_ORIGINS` | `*` → localhost:5000 | `https://xxx.vercel.app` |
| `.env` kaynağı | `.env` dosyası | `fly secrets` / Vercel env |

**Sonuç:** Kod ve iş mantığı birebir aynı. Sadece URL'ler ve secret'lar ortama göre değişir.

---

## 3. Deployment Checklist (Eksiksiz)

### Backend (Fly.io)

```bash
fly secrets set DATABASE_URL="postgresql://..."
fly secrets set SUPABASE_URL="https://xxx.supabase.co"
fly secrets set SUPABASE_ANON_KEY="eyJhbG..."
fly secrets set SUPABASE_JWT_SECRET="..."
fly secrets set CORS_ORIGINS="https://swing-trade-xxx.vercel.app"  # Vercel URL
# Opsiyonel: OPENAI_API_KEY, GEMINI_API_KEY, LLM_PROVIDER
fly deploy
```

### Frontend (Vercel)

| Env | Değer |
|-----|-------|
| `NEXT_PUBLIC_API_URL` | `https://swing-trade.fly.dev` |
| `NEXT_PUBLIC_SUPABASE_URL` | `https://xxx.supabase.co` |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | `eyJhbG...` |

---

## 4. Olası Sorunlar ve Çözümleri

| Sorun | Sebep | Çözüm |
|-------|-------|-------|
| CORS hatası (prod) | `CORS_ORIGINS` set değil | `fly secrets set CORS_ORIGINS="https://..."` |
| 401 Unauthorized (prod) | Auth env eksik | SUPABASE_* tümü set edilmeli |
| Cold start gecikmesi | `min_machines_running = 0` | `min_machines_running = 1` (ek maliyet) |
| Build'de API URL yanlış | NEXT_PUBLIC_* build-time | Vercel'de env'leri doğru gir, rebuild |

---

## 5. Özet

- **Production ready:** Evet, env'ler doğru set edilirse.
- **Local = Deployment:** Evet, aynı kod ve mantık; sadece config farklı.
- **Kritik adım:** `CORS_ORIGINS` ve tüm `SUPABASE_*` değişkenlerinin production'da set edilmesi.
