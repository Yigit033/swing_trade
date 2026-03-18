"use client";

import { useState, useEffect } from "react";
import { createSupabaseClient } from "@/lib/supabase/client";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [hasSession, setHasSession] = useState<boolean | null>(null);
  const router = useRouter();
  const supabase = createSupabaseClient();

  useEffect(() => {
    const client = createSupabaseClient();
    if (!client) {
      setHasSession(false);
      return;
    }
    client.auth.getSession().then(({ data: { session } }) => {
      setHasSession(!!session);
    });
  }, []);

  if (!supabase) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[var(--bg-primary)]">
        <div className="glass-card p-8 max-w-md text-center">
          <h1 className="text-xl font-bold mb-4">Auth Not Configured</h1>
          <p className="text-[var(--text-muted)] text-sm mb-4">
            Supabase Auth is not configured. Add NEXT_PUBLIC_SUPABASE_URL and
            NEXT_PUBLIC_SUPABASE_ANON_KEY to .env.local to enable login.
          </p>
          <a href="/" className="text-[var(--accent)] hover:underline">
            ← Back to Dashboard
          </a>
        </div>
      </div>
    );
  }

  if (hasSession === null) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[var(--bg-primary)]">
        <div className="spinner" style={{ width: 32, height: 32 }} />
      </div>
    );
  }

  if (hasSession === true) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[var(--bg-primary)] p-4">
        <div className="glass-card w-full max-w-md p-8 text-center">
          <h1 className="page-title gradient-text text-2xl mb-2">Swing Trade AI</h1>
          <p className="text-[var(--text-muted)] text-sm mb-6">
            Zaten giriş yaptınız.
          </p>
          <div className="flex flex-col gap-3">
            <Link href="/" className="btn-primary w-full py-3 text-center">
              Dashboard'a git
            </Link>
            <button
              type="button"
              onClick={async () => {
                await supabase.auth.signOut();
                setHasSession(false);
                router.refresh();
              }}
              className="text-[var(--text-muted)] hover:text-[var(--accent)] text-sm py-2"
            >
              Çıkış yap
            </button>
          </div>
        </div>
      </div>
    );
  }

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const { data, error: err } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
      if (err) {
        setError(err.message);
        return;
      }
      if (data.user) {
        // Cookie'lerin browser'a yazılması için kısa bekleme (401 redirect loop önlemi).
        // Full page redirect ile session kesin sunucuya ulaşır.
        await new Promise((r) => setTimeout(r, 250));
        window.location.href = "/";
      }
    } catch (err) {
      setError("An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--bg-primary)] p-4">
      <div className="glass-card w-full max-w-md p-8">
        <h1 className="page-title gradient-text text-2xl mb-2">Swing Trade AI</h1>
        <p className="text-[var(--text-muted)] text-sm mb-6">
          Sign in to access your paper trades
        </p>

        <form onSubmit={handleLogin} className="space-y-4">
          <div>
            <label className="block text-sm font-600 text-[var(--text-secondary)] mb-2">
              Email
            </label>
            <input
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="input w-full"
              required
              autoComplete="email"
            />
          </div>
          <div>
            <label className="block text-sm font-600 text-[var(--text-secondary)] mb-2">
              Password
            </label>
            <input
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="input w-full"
              required
              autoComplete="current-password"
            />
          </div>
          {error && (
            <div className="text-sm text-[var(--red)] bg-[rgba(239,68,68,0.1)] p-3 rounded-lg">
              {error}
            </div>
          )}
          <button
            type="submit"
            disabled={loading}
            className="btn-primary w-full py-3 flex items-center justify-center gap-2"
          >
            {loading ? (
              <span className="spinner" style={{ width: 20, height: 20 }} />
            ) : (
              "Sign In"
            )}
          </button>
        </form>

        <p className="mt-6 text-center text-xs text-[var(--text-muted)]">
          Create an account in Supabase Dashboard → Authentication → Users
        </p>
      </div>
    </div>
  );
}
