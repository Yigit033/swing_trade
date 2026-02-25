import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = axios.create({
    baseURL: API_BASE,
    timeout: 120000, // 2 min for scan operations
});

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
    atr?: number;
    signal_price?: number;
    current_price?: number;
    unrealized_pnl?: number;
    unrealized_pnl_pct?: number;
    created_at?: string;
    updated_at?: string;
}

export interface Signal {
    ticker: string;
    entry_price: number;
    stop_loss: number;
    target_1: number;
    target: number;
    quality_score: number;
    swing_type?: string;
    rsi?: number;
    volume_surge?: number;
    atr?: number;
    expected_hold_min?: number;
    expected_hold_max?: number;
    expiration_date?: string;
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
}) => api.post("/api/scanner/smallcap", params).then((r) => r.data);

// Lookup
export const lookupTickers = (tickers: string[], portfolio_value = 10000) =>
    api.post("/api/lookup", { tickers, portfolio_value }).then((r) => r.data);

// GenAI
export const chatWithAI = (message: string, history: unknown[] = []) =>
    api.post("/api/genai/chat", { message, history }).then((r) => r.data);

export const getSignalBrief = (ticker: string, signal: Signal) =>
    api.post("/api/genai/signal-brief", { ticker, signal }).then((r) => r.data);

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

