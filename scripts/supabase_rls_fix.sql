-- ===================================================================
-- SWING_TRADE — Supabase RLS Security Fix (tüm tablolar)
-- Run in: Supabase Dashboard → SQL Editor → New query → Run
-- ===================================================================
-- Mimari not:
--   Backend → FastAPI → DATABASE_URL (postgres superuser) → RLS bypass
--   Yani backend HİÇBİR ŞEKILDE etkilenmez.
--   Sadece Supabase REST API (anon/authenticated rol) kısıtlanır.
-- ===================================================================


-- ── 1. paper_trades ────────────────────────────────────────────────
-- Per-user ticaret tablosu. Herkes sadece kendi satırlarını görür.

ALTER TABLE paper_trades
  ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id);

ALTER TABLE paper_trades ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own trades"   ON paper_trades;
DROP POLICY IF EXISTS "Users can create own trades" ON paper_trades;
DROP POLICY IF EXISTS "Users can update own trades" ON paper_trades;
DROP POLICY IF EXISTS "Users can delete own trades" ON paper_trades;

-- SELECT: kendi satırları + user_id=NULL olan eski (legacy) satırlar
CREATE POLICY "Users can view own trades"
ON paper_trades FOR SELECT
USING (auth.uid() = user_id OR user_id IS NULL);

-- INSERT: sadece kendi user_id'siyle ekleyebilir
CREATE POLICY "Users can create own trades"
ON paper_trades FOR INSERT
WITH CHECK (auth.uid() = user_id);

-- UPDATE: kendi satırları + legacy satırlar
CREATE POLICY "Users can update own trades"
ON paper_trades FOR UPDATE
USING  (auth.uid() = user_id OR user_id IS NULL)
WITH CHECK (auth.uid() = user_id);

-- DELETE: kendi satırları + legacy satırlar
CREATE POLICY "Users can delete own trades"
ON paper_trades FOR DELETE
USING (auth.uid() = user_id OR user_id IS NULL);


-- ── 2. paper_meta ──────────────────────────────────────────────────
-- Backend-only key/value store (last_price_update vb.)
-- Kullanıcı politikası yok → REST API tamamen kapalı.
-- Backend postgres superuser ile direkt erişir, etkilenmez.

ALTER TABLE paper_meta ENABLE ROW LEVEL SECURITY;


-- ── 3. smallcap_signal_runs ────────────────────────────────────────
-- Scan geçmişi. Tüm okuma/yazma FastAPI backend üzerinden gidiyor.
-- REST API erişimi kapalı (backend postgres ile bypass ediyor).

ALTER TABLE smallcap_signal_runs ENABLE ROW LEVEL SECURITY;


-- ── 4. regime_history ──────────────────────────────────────────────
-- SPY/market rejim geçmişi. Backend-only, kamuya açık olması gerekmiyor.
-- Okuma da FastAPI /api/regime/history endpoint'i üzerinden gidiyor.

ALTER TABLE regime_history ENABLE ROW LEVEL SECURITY;
