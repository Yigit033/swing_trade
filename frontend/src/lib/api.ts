import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = axios.create({
    baseURL: API_BASE,
    timeout: 120000, // 2 min for scan operations
});

// Add Supabase JWT to requests when auth is configured
api.interceptors.request.use(async (config) => {
    if (typeof window === "undefined") return config;
    try {
        const { createSupabaseClient } = await import("@/lib/supabase/client");
        const supabase = createSupabaseClient();
        if (supabase) {
            const { data: { session } } = await supabase.auth.getSession();
            if (session?.access_token) {
                config.headers.Authorization = `Bearer ${session.access_token}`;
            }
        }
    } catch {
        // Ignore — auth not configured or not in browser
    }
    return config;
});

// 401 + token gönderildiyse → token reddedildi, sign out ve login'e yönlendir
// Token GÖNDERİLMEDİYSE → timing/race (session henüz hazır değil), sign out YAPMA
api.interceptors.response.use(
    (res) => res,
    async (err) => {
        if (typeof window !== "undefined" && err?.response?.status === 401) {
            const hadToken = !!err?.config?.headers?.Authorization;
            // Debug: 401 sebebini logla (Fly.io logs + backend auth_configured kontrolü)
            if (hadToken) {
                const detail = err?.response?.data?.detail ?? "unknown";
                console.error(
                    "[Auth 401] Backend token reddetti.",
                    "URL:", err?.config?.url,
                    "Detail:", detail,
                    "→ Fly.io logs kontrol et, CORS_ORIGINS + SUPABASE_* secrets doğrula"
                );
            }
            if (hadToken) {
                try {
                    const { createSupabaseClient } = await import("@/lib/supabase/client");
                    const supabase = createSupabaseClient();
                    if (supabase) {
                        await supabase.auth.signOut();
                    }
                } catch {
                    // ignore
                }
                window.location.href = "/login";
            }
        }
        return Promise.reject(err);
    }
);

// ---- Types ----
export interface Trade {
    id: number;
    ticker: string;
    entry_date: string;
    entry_price: number;
    stop_loss: number;
    target: number;
    swing_type: string;
    quality_score: number;
    position_size: number;
    max_hold_days: number;
    status: string;
    exit_date?: string;
    exit_price?: number;
    realized_pnl?: number;
    realized_pnl_pct?: number;
    notes?: string;
    trailing_stop?: number;
    initial_stop?: number;
    atr?: number;
    signal_price?: number;
    current_price?: number;
    unrealized_pnl?: number;
    unrealized_pnl_pct?: number;
    // v3.1: Dual target & partial exit
    target_2?: number;
    partial_exit_price?: number;
    partial_exit_pct?: number;
    created_at?: string;
    updated_at?: string;
}

export interface Signal {
    ticker: string;
    date?: string;
    signal_type?: string;
    entry_price: number;
    stop_loss: number;
    target_1: number;
    target_2?: number;
    target: number;
    quality_score: number;
    original_quality_score?: number;
    swing_type?: string;
    swing_type_label?: string;
    hold_days_min?: number;
    hold_days_max?: number;
    type_reason?: string;
    close_position?: number;
    // Momentum
    volume_surge?: number;
    atr_percent?: number;
    atr?: number;
    float_millions?: number;
    market_cap_millions?: number;
    // Swing metrics
    five_day_return?: number;
    ma20_distance?: number;
    rsi?: number;
    swing_ready?: boolean;
    higher_lows?: boolean;
    // Boosters
    high_rvol?: boolean;
    gap_continuation?: boolean;
    higher_highs?: boolean;
    // Sector & Catalyst
    sector_rs_score?: number;
    sector_rs_bonus?: number;
    is_sector_leader?: boolean;
    short_percent?: number;
    days_to_cover?: number;
    is_squeeze_candidate?: boolean;
    has_insider_buying?: boolean;
    has_recent_news?: boolean;
    total_catalyst_bonus?: number;
    rsi_divergence?: boolean;
    macd_bullish?: boolean;
    // OBV Trend (v3.0)
    obv_accumulation?: boolean;
    obv_distribution?: boolean;
    obv_bonus?: number;
    // Market Regime (v4.0)
    market_regime?: string;
    regime_multiplier?: number;
    regime_confidence?: string;
    // Risk
    target_1_pct?: number;
    target_2_pct?: number;
    stop_loss_pct?: number;
    risk_reward?: number;
    risk_reward_t2?: number;
    position_size?: number;
    risk_amount?: number;
    expected_hold_min?: number;
    expected_hold_max?: number;
    max_hold_date?: string;
    expiration_date?: string;
    volatility_warning?: boolean;
    // Narrative
    narrative?: { full_text?: string; headline?: string;[key: string]: any };
    narrative_text?: string;
    narrative_headline?: string;
    // Info
    company_name?: string;
    sector?: string;
    win_probability?: number;
    notes?: string;
}

export interface PerformanceSummary {
    total_trades: number;
    open_trades: number;
    pending_trades: number;
    closed_trades: number;
    wins: number;
    losses: number;
    breakeven: number;
    win_rate: number;
    total_pnl: number;
    total_pnl_pct: number;
    avg_pnl_pct: number;
    avg_win: number;
    avg_loss: number;
}

// ---- API calls ----

// Trades
export const getTrades = (status?: string) =>
    api.get("/api/trades", { params: status ? { status } : {} }).then((r) => r.data);

export const addTrade = (trade: Partial<Trade>) =>
    api.post("/api/trades", trade).then((r) => r.data);

export const deleteTrade = (id: number) =>
    api.delete(`/api/trades/${id}`).then((r) => r.data);

export const updateTrade = (id: number, updates: Partial<Trade>) =>
    api.patch(`/api/trades/${id}`, updates).then((r) => r.data);

export const closeTrade = (id: number, exit_price: number, notes = "") =>
    api.post(`/api/trades/${id}/close`, { exit_price, notes }).then((r) => r.data);

export const updatePrices = () =>
    api.post("/api/trades/update-prices").then((r) => r.data);

export const getTradesLastUpdate = () =>
    api.get("/api/trades/last-update").then((r) => r.data);

// Pending
export const getPending = () =>
    api.get("/api/pending").then((r) => r.data);

export const checkPending = () =>
    api.post("/api/pending/check").then((r) => r.data);

export const confirmTrade = (id: number) =>
    api.post(`/api/pending/${id}/confirm`).then((r) => r.data);

// Performance
export const getPerformance = () =>
    api.get("/api/performance").then((r) => r.data);

export const getWeeklyReport = () =>
    api.get("/api/performance/weekly-report").then((r) => r.data);

// Scanner
export const runSmallcapScan = (params: {
    min_quality?: number;
    top_n?: number;
    portfolio_value?: number;
}) => api.post("/api/scanner/smallcap", params, { timeout: 600000 }).then((r) => r.data);

export const trackSignal = (signal: Signal & { hold_days_max?: number }) =>
    api.post("/api/scanner/track", {
        ticker: signal.ticker,
        entry_price: signal.entry_price,
        stop_loss: signal.stop_loss,
        target_1: signal.target_1 || signal.target,
        target_2: signal.target_2 || signal.target_1 || signal.target,
        swing_type: signal.swing_type || "A",
        quality_score: signal.quality_score,
        position_size: signal.position_size || 100,
        hold_days_max: signal.hold_days_max ?? signal.expected_hold_max ?? 7,
        atr: signal.atr || 0,
    }).then((r) => r.data);

// Market Regime
export interface RegimeData {
    regime: string;
    confidence: string;
    score_multiplier: number;
    spy_price?: number;
    ma50?: number;
    ma200?: number;
    vix?: number;
    spy_5d_return?: number;
    detected_at?: string;
}

export const getCurrentRegime = (): Promise<RegimeData> =>
    api.get("/api/regime/current").then((r) => r.data);

export const getRegimeHistory = (limit = 30) =>
    api.get(`/api/regime/history?limit=${limit}`).then((r) => r.data);

// Lookup
export const lookupTickers = (tickers: string[], portfolio_value = 10000) =>
    api.post("/api/lookup", { tickers, portfolio_value }).then((r) => r.data);

// GenAI
export const chatWithAI = (message: string, history: unknown[] = []) =>
    api.post("/api/genai/chat", { message, history }).then((r) => r.data);

export const getSignalBrief = (ticker: string, signal: Signal) =>
    api.post("/api/genai/signal-brief", { ticker, signal }).then((r) => r.data);

export const getWeeklyReportAI = () =>
    api.get("/api/genai/weekly-report-ai", { timeout: 120000 }).then((r) => r.data);

export const getModelStatus = () =>
    api.get("/api/genai/model-status").then((r) => r.data);

export const trainModel = () =>
    api.post("/api/genai/train", {}, { timeout: 120000 }).then((r) => r.data);

export const predictSignal = (params: {
    entry_price: number;
    stop_loss: number;
    target: number;
    atr?: number;
    quality_score?: number;
    swing_type?: string;
    max_hold_days?: number;
}) =>
    api.post("/api/genai/predict", params).then((r) => r.data);

// ---- Backtest types ----
export interface BacktestMetrics {
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;          // 0-1 float
    profit_factor: number;
    total_pnl_dollar: number;
    total_return: number;      // decimal, e.g. 0.12 = 12%
    max_drawdown: number;      // percent, e.g. -8.5
    avg_win_pct: number;
    avg_loss_pct: number;
    avg_win_dollar: number;
    avg_loss_dollar: number;
    avg_hold_days: number;
    initial_capital: number;
    final_capital: number;
    type_stats?: Record<string, { wins: number; losses: number; total_pnl: number }>;
    exit_stats?: Record<string, { count: number; avg_pnl: number }>;
}

export interface BacktestTrade {
    ticker: string;
    swing_type?: string;
    entry_price: number;
    exit_price: number;
    pnl_pct: number;
    pnl_dollar: number;
    exit_reason?: string;
    entry_date?: string;
    exit_date?: string;
    hold_days?: number;
}

export interface BacktestResult {
    period_days: number;
    start_date: string;
    end_date: string;
    tickers_used: string[];
    initial_capital: number;
    metrics: BacktestMetrics;
    equity_curve: { date: string; portfolio_value: number }[];
    trades: BacktestTrade[];
    error?: string;
}

// Backtest — 5-min timeout for long operations
export const runBacktest = (params: {
    period_days: number;
    initial_capital: number;
    max_concurrent: number;
    tickers?: string[];
}) =>
    api.post<BacktestResult>("/api/backtest/smallcap", params, { timeout: 300000 })
        .then((r) => r.data);

