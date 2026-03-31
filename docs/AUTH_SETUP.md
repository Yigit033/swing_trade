# Supabase Auth Setup

Auth is **optional**. When not configured, the app works without login (backward compatible).

## Quick Setup (15 min)

### 1. Supabase Project

You likely already have one (for DATABASE_URL). Go to [Supabase Dashboard](https://supabase.com/dashboard) → your project.

### 2. Get Credentials

**Settings → API** (or Connect):

- **Project URL** → `NEXT_PUBLIC_SUPABASE_URL` / `SUPABASE_URL`
- **anon (public) key** → `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- **JWT Secret** → `SUPABASE_JWT_SECRET` (Settings → API → JWT Settings)

### 3. Environment Variables

**Backend (.env):**
```bash
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=eyJhbG...   # anon key (frontend ile aynı) — önerilen
SUPABASE_JWT_SECRET=your-jwt-secret  # fallback; SUPABASE_ANON_KEY öncelikli

# DATABASE_URL already set for Supabase
```

**Frontend (.env.local):**
```bash
NEXT_PUBLIC_SUPABASE_URL=https://xxxxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbG...
```

### 4. Create First User

Supabase Dashboard → **Authentication → Users → Add user**

- Email: your@email.com
- Password: (choose one)

### 5. Run RLS Migration (if using Supabase)

In Supabase SQL Editor, run:

```sql
-- From scripts/supabase_auth_migration.sql
ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id);
ALTER TABLE paper_trades ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own trades" ON paper_trades FOR SELECT
  USING (auth.uid() = user_id OR user_id IS NULL);
CREATE POLICY "Users can create own trades" ON paper_trades FOR INSERT
  WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own trades" ON paper_trades FOR UPDATE
  USING (auth.uid() = user_id OR user_id IS NULL);
CREATE POLICY "Users can delete own trades" ON paper_trades FOR DELETE
  USING (auth.uid() = user_id OR user_id IS NULL);
```

### 6. Deploy

**Fly.io (backend):**
```bash
fly secrets set SUPABASE_URL=https://xxx.supabase.co
fly secrets set SUPABASE_ANON_KEY=eyJhbG...
fly secrets set SUPABASE_JWT_SECRET=your-jwt-secret
fly secrets set CORS_ORIGINS=https://your-app.vercel.app  # Vercel URL
fly deploy
```

**Vercel (frontend):**
Add `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_API_URL` in project settings.

## Local dev URL (Supabase)

If you use login locally, add **`http://localhost:5000`** (and `http://127.0.0.1:5000` if needed) under **Authentication → URL Configuration** → Redirect URLs / Site URL in the Supabase dashboard. The Next.js dev server runs on port **5000** by default in this repo.

## Test

1. `npm run dev` (frontend)
2. Visit http://localhost:5000 → redirects to /login
3. Sign in with the user you created
4. Dashboard loads with your trades

## Disabling Auth

Remove or leave unset:

- Backend: `SUPABASE_JWT_SECRET`
- Frontend: `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`

The app will work without login (all routes public).
