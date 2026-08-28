import axios, { AxiosError, InternalAxiosRequestConfig } from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = axios.create({
    baseURL: API_BASE,
    timeout: 120000, // 2 min for scan operations
});

const FLY_WAKE_RETRIES = 3;
const FLY_WAKE_STATUSES = new Set([502, 503, 504]);

type RetryConfig = InternalAxiosRequestConfig & { _retryCount?: number };

function isFlyWakeFailure(err: AxiosError): boolean {
    const status = err.response?.status;
    if (status && FLY_WAKE_STATUSES.has(status)) return true;
    // Fly 502 often surfaces as CORS / ERR_NETWORK because the proxy page has no ACAO header.
    if (!err.response && (err.code === "ERR_NETWORK" || err.message === "Network Error")) {
        return true;
    }
    return false;
}

function flyWakeDelayMs(attempt: number): number {
    return 2000 * 2 ** (attempt - 1); // 2s, 4s, 8s
}

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
// 502/network: Fly scale-to-zero wake — 2s/4s/8s retry so a cold 502 is hidden
// from the UI. Backend thin-boot should bind :8000 before this budget is spent.
api.interceptors.response.use(
    (res) => res,
    async (err: AxiosError) => {
        if (typeof window !== "undefined" && err?.response?.status === 401) {
            const hadToken = !!err?.config?.headers?.Authorization;
            // Debug: 401 sebebini logla (Fly.io logs + backend auth_configured kontrolü)
            if (hadToken) {
                const detail = (err.response?.data as { detail?: string } | undefined)?.detail ?? "unknown";
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
            return Promise.reject(err);
        }

        const config = err.config as RetryConfig | undefined;
        if (config && isFlyWakeFailure(err)) {
            const n = config._retryCount ?? 0;
            if (n < FLY_WAKE_RETRIES && !config.signal?.aborted) {
                const attempt = n + 1;
                config._retryCount = attempt;
                await new Promise((r) => setTimeout(r, flyWakeDelayMs(attempt)));
                return api.request(config);
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
    // v13.3: post-exit drift — current vs exit price for closed trades
    since_exit_pct?: number;
    // v13.6: holiday-aware pending entry expectation (from GET /api/pending)
    expected_entry_date?: string;
    expected_entry_label?: string;
    pending_reason?: string;
    pending_note?: string;
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
    // v14: hangi tetikleyici kapıdan geldi — 'vce_breakout' | 'rvol_thrust'
    trigger_pathway?: string;
    trigger_reason?: string;
    entry_price: number;
    stop_loss: number;
    target_1: number;
    target_2?: number;
    target: number;
    quality_score: number;              // HAM skor — baraj bu skora uygulanır
    rank_score?: number;                // ham + VCE işaretleri — sıralama ölçütü
    rank_bonus?: number;                // 0 | 5 | 8 | 13
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
    // Göreli güç (SPY'a karşı)
    sector_rs_score?: number;
    is_sector_leader?: boolean;
    // VCE kalite işaretleri (skoru gerçekten kaydıran iki ekleme)
    vce_premium?: boolean;
    vce_tight_coil?: boolean;
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
    narrative?: { full_text?: string; headline?: string; [key: string]: unknown };
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

// Scanner — background job (sayfa değiştirsen de scan sürer)
export const startSmallcapScanJob = (params: {
    min_quality?: number;
    top_n?: number;
    portfolio_value?: number;
}) => api.post("/api/scanner/smallcap/start", params, { timeout: 60000 }).then((r) => r.data);

export const getSmallcapScanJob = (jobId: string) =>
    api.get(`/api/scanner/smallcap/job/${encodeURIComponent(jobId)}`, { timeout: 20000 }).then((r) => r.data);

// Scanner — saved history (server-side)
export type SmallcapScanRunMeta = {
    id: number;
    created_at: string;
    job_id?: string | null;
    user_id?: string | null;
    portfolio_value?: number | null;
    request_min_quality?: number | null;
    request_top_n?: number | null;
    effective_min_quality?: number | null;
    effective_top_n?: number | null;
    market_regime?: string | null;
    regime_confidence?: string | null;
};

export type SmallcapScanRunDetail = SmallcapScanRunMeta & {
    stats?: Record<string, unknown> & {
        scanned_members?: ScannedMember[];
        universe_no_signal?: number;
        stocks_scanned?: number;
        stocks_with_data?: number;
        raw_signals?: number;
        filtered_signals?: number;
    };
    signals?: Signal[];
    stale_fallback?: boolean;
};

export const getSmallcapScanHistory = (limit = 20) =>
    api.get("/api/scanner/smallcap/history", { params: { limit }, timeout: 20000 }).then((r) => r.data as { runs: SmallcapScanRunMeta[]; count: number });

export const getSmallcapScanHistoryRun = (runId: number) =>
    api.get(`/api/scanner/smallcap/history/${runId}`, { timeout: 30000 }).then((r) => r.data as SmallcapScanRunDetail);

/** Senkron scan (script / legacy); UI için startSmallcapScanJob kullan */
export const runSmallcapScan = (params: {
    min_quality?: number;
    top_n?: number;
    portfolio_value?: number;
}) => api.post("/api/scanner/smallcap", params, { timeout: 600000 }).then((r) => r.data);

export const trackSignal = (
    signal: Signal & { hold_days_max?: number },
    portfolioValue?: number,
) =>
    api.post("/api/scanner/track", {
        ticker: signal.ticker,
        // Sinyal barının tarihi (tarama günü DEĞİL) — ölçülen giriş t+1 open
        // bu tarihe göre hesaplanır; gönderilmezse backend bugünü basar ve
        // seans içi/hafta sonu taramalarda giriş bir seans geç kayar (t+2).
        date: signal.date,
        entry_price: signal.entry_price,
        stop_loss: signal.stop_loss,
        target_1: signal.target_1 || signal.target,
        target_2: signal.target_2 || signal.target_1 || signal.target,
        swing_type: signal.swing_type || "A",
        quality_score: signal.quality_score,
        position_size: signal.position_size || 100,
        hold_days_max: signal.hold_days_max ?? signal.expected_hold_max ?? 7,
        atr: signal.atr || 0,
        // Boyut sunucuda bu tabana göre YENİDEN hesaplanır (istemci boyutu yalnız
        // üst sınır). Gönderilmezse sunucu 10k varsayar.
        portfolio_value: portfolioValue ?? 10000,
    }).then((r) => r.data);

// Market Regime
export interface RegimeData {
    regime: string;
    confidence: string;
    spy_price?: number;
    ma50?: number;
    ma200?: number;
    vix?: number;
    spy_5d_return?: number;
    detected_at?: string;
    /** Present when regime is UNKNOWN (e.g. rate limit, insufficient data). */
    detect_error?: string;
    /** True when live yfinance failed and values come from last DB row. */
    stale_fallback?: boolean;
    fallback_reason?: string;
    /** True when sampled fresh from SPY/^VIX (same path as scanner). */
    live?: boolean;
}

export const getCurrentRegime = (): Promise<RegimeData> =>
    api.get("/api/regime/current").then((r) => r.data);

export const getRegimeHistory = (limit = 30) =>
    api.get(`/api/regime/history?limit=${limit}`).then((r) => r.data);

// Lookup
export const lookupTickers = (tickers: string[], portfolio_value = 10000) =>
    api.post("/api/lookup", { tickers, portfolio_value }).then((r) => r.data);

// GenAI
export const chatWithAI = (message: string, history: unknown[] = [], verbosity: "brief" | "detailed" = "detailed") =>
    api.post("/api/genai/chat", { message, history, verbosity }).then((r) => r.data);

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
    /** Engine exit bucket (STOPPED, TARGET, …) — preferred for UI labels */
    status?: string;
    entry_date?: string;
    exit_date?: string;
    /** Actual bars/days held at exit (SmallCap backtest) */
    days_held?: number;
    /** Legacy alias — prefer days_held */
    hold_days?: number;
    shares?: number;
    max_hold_days?: number;
    quality_score?: number;
    /** Lot at open (same as shares on closed trades) */
    initial_shares?: number;
    partial_shares?: number;
}

export interface BacktestEquityPoint {
    date: string;
    portfolio_value: number;
    open_trades?: number;
    market_regime?: string;
    regime_confidence?: string;
    regime_multiplier?: number;
    effective_min_quality?: number;
    effective_top_n?: number;
    request_min_quality?: number;
    request_top_n?: number;
}

export interface BacktestResult {
    period_days: number;
    start_date: string;
    end_date: string;
    tickers_used: string[];
    initial_capital: number;
    /** Finviz evreni geçmişe uygulandıysa survivorship-bias uyarısı (v33) */
    survivorship_warning?: string | null;
    min_quality?: number;
    top_n?: number;
    /** Tickers with enough bars to participate in simulation */
    data_stocks?: number;
    params?: {
        min_quality: number;
        top_n: number;
        max_concurrent: number;
        slippage_bps_per_side?: number;
        partial_at_t1_fraction?: number;
    };
    /** Walk-forward funnel: signals, pending, entry skips */
    diagnostics?: Record<string, number>;
    metrics: BacktestMetrics;
    equity_curve: BacktestEquityPoint[];
    trades: BacktestTrade[];
    error?: string;
}

// Backtest — 5-min timeout for long operations
export const runBacktest = (params: {
    period_days: number;
    initial_capital: number;
    max_concurrent: number;
    min_quality?: number;
    top_n?: number;
    tickers?: string[];
}) =>
    api.post<BacktestResult>("/api/backtest/smallcap", params, { timeout: 300000 })
        .then((r) => r.data);

/** Small-cap JSON from GET /api/settings (matches backend SmallCapSettings). */
export type SmallCapSettingsJSON = Record<string, unknown>;

export type SmallCapSettingsPutResponse = {
    ok: boolean;
    settings: SmallCapSettingsJSON;
};

/** Backend hem `/api/settings` hem `/api/settings/` kabul eder (307 yok). */
export const fetchSmallCapSettings = () =>
    api.get<SmallCapSettingsJSON>("/api/settings").then((r) => r.data);

export const updateSmallCapSettings = (body: SmallCapSettingsJSON) =>
    api.put<SmallCapSettingsPutResponse>("/api/settings", body).then((r) => r.data);

export const resetSmallCapSettings = () =>
    api.post<SmallCapSettingsPutResponse>("/api/settings/reset").then((r) => r.data);


// ── Sinyal Karnesi (forward-return tracking) ─────────────────────────────
// Motorun ürettiği TÜM ham sinyaller (eşik-altı dahil) burada izlenir. Eşiğin
// doğru yerde olup olmadığını ancak "almadıklarımızın" sonucuyla ölçebiliriz.
export type EdgeAgg = {
    n: number;
    mean: number;
    median: number;
    win_rate: number;
} | null;

export type EdgeBucket = {
    label: string;
    n: number;
    pending: number;
    mean: number | null;
    win_rate: number | null;
};

export type EdgeSide = {
    n: number;
    mean: number | null;
    win_rate: number | null;
    best?: number | null;
    worst?: number | null;
};

export type EdgeSignal = {
    ticker: string;
    signal_date: string;
    quality: number | null;
    swing_type: string | null;
    pathway: string | null;
    regime: string | null;
    entry_open: number | null;
    r3: number | null;
    r5: number | null;
    r10: number | null;
    mfe10: number | null;
    mae10: number | null;
    status: string;
    kind?: "signal" | "universe" | string | null;
    reject_reason?: string | null;
};

export type EdgeRejectReason = {
    reason: string;
    n: number;
    pending: number;
    n_mature: number;
    mean: number | null;
    win_rate: number | null;
};

export type EdgeUniverse = {
    n_tracked: number;
    n_complete: number;
    aggregates: Record<string, EdgeAgg>;
    r10: EdgeSide;
    reject_reasons: EdgeRejectReason[];
};

export type EdgeTracking = {
    n_tracked: number;
    n_complete: number;
    aggregates: Record<string, EdgeAgg>;
    mfe10: EdgeAgg;
    mae10: EdgeAgg;
    quality_buckets: EdgeBucket[];
    threshold_split: {
        reference: number;
        above: EdgeSide;
        below: EdgeSide;
    };
    harness_expectation: Record<string, string>;
    signals: EdgeSignal[];
    universe?: EdgeUniverse;
    cohort_split?: {
        signal: EdgeSide;
        universe: EdgeSide;
    };
    universe_rows?: EdgeSignal[];
};

export const REJECT_REASON_LABELS: Record<string, string> = {
    no_trigger: "Tetik yok",
    filter_failed: "Filtre",
    rsi_gate: "RSI kapısı",
    swing_not_ready: "Swing hazır değil",
    stage_rejected: "Weinstein evresi",
    insufficient_data: "Veri yetersiz",
    no_data: "Fiyat alınamadı",
    scan_error: "Tarama hatası",
    quality_type_a: "Tip A tabanı",
    quality_type_b: "Tip B tabanı",
    quality_type_c: "Tip C tabanı",
    quality_type_s: "Tip S tabanı",
    unknown: "Bilinmiyor",
};

export type ScannedMember = {
    ticker: string;
    kind: "signal" | "universe" | string;
    date?: string | null;
    quality?: number | null;
    reject_reason?: string | null;
    pathway?: string | null;
};

export const getEdgeTracking = (refresh = false) =>
    api.get<EdgeTracking>(`/api/scanner/smallcap/edge-tracking?refresh=${refresh}`,
        { timeout: 180000 }).then((r) => r.data);
