-- Supabase Auth Migration — Run in Supabase SQL Editor
-- Adds user_id and RLS to paper_trades when using Supabase Auth

-- 1. Add user_id column (if not exists)
ALTER TABLE paper_trades
ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id);

-- 2. Enable RLS
ALTER TABLE paper_trades ENABLE ROW LEVEL SECURITY;

-- 3. Policy: Users can only see their own trades (or legacy rows with NULL user_id)
CREATE POLICY "Users can view own trades"
ON paper_trades
FOR SELECT
USING (auth.uid() = user_id OR user_id IS NULL);

-- 4. Policy: Users can only insert their own trades
CREATE POLICY "Users can create own trades"
ON paper_trades
FOR INSERT
WITH CHECK (auth.uid() = user_id);

-- 5. Policy: Users can only update their own trades
CREATE POLICY "Users can update own trades"
ON paper_trades
FOR UPDATE
USING (auth.uid() = user_id OR user_id IS NULL)
WITH CHECK (auth.uid() = user_id);

-- 6. Policy: Users can only delete their own trades
CREATE POLICY "Users can delete own trades"
ON paper_trades
FOR DELETE
USING (auth.uid() = user_id OR user_id IS NULL);
