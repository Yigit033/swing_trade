"use client";
import { useState, useCallback, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import {
    ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Cell, ReferenceLine, Area, AreaChart,
} from "recharts";
import { LineChart as LineIcon, Search, TrendingUp, TrendingDown } from "lucide-react";

interface ChartRow {
    date: string;
    open: number | null; high: number | null; low: number | null; close: number | null;
    volume: number | null; volume_ma: number | null;
    rsi: number | null; macd: number | null; macd_signal: number | null; macd_hist: number | null;
    ema20: number | null; ema50: number | null;
}

const PERIODS = ["1mo", "3mo", "6mo", "1y", "2y"];

const CTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null;
    return (
        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, padding: "8px 14px", fontSize: "0.78rem" }}>
            <div style={{ color: "var(--text-muted)", marginBottom: 4, fontWeight: 600 }}>{label}</div>
            {payload.map((p: any, i: number) => (
                <div key={i} style={{ color: p.color, display: "flex", gap: 8 }}>
                    <span>{p.name}:</span>
                    <strong>{typeof p.value === "number" ? p.value.toFixed(2) : p.value}</strong>
                </div>
            ))}
        </div>
    );
};

function ChartsContent() {
    const searchParams = useSearchParams();
    const [ticker, setTicker] = useState(searchParams.get("ticker")?.toUpperCase() || "AAPL");
    const [period, setPeriod] = useState("3mo");
    const [data, setData] = useState<ChartRow[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [loaded, setLoaded] = useState(false);

    const fetchChart = useCallback(async (t = ticker, p = period) => {
        if (!t.trim()) return;
        setLoading(true); setError(""); setLoaded(false);
        try {
            const res = await api.get(`/api/scanner/chart?ticker=${t.toUpperCase()}&period=${p}`);
            if (res.data.error) { setError(res.data.error); setData([]); }
            else { setData(res.data.data || []); setLoaded(true); }
        } catch { setError("Could not load chart. Make sure the API is running."); }
        finally { setLoading(false); }
    }, [ticker, period]);

    // Auto-load if ticker param exists
    useEffect(() => {
        const t = searchParams.get("ticker");
        if (t) {
            setTicker(t.toUpperCase());
            fetchChart(t.toUpperCase(), period);
        }
    }, [searchParams, fetchChart, period]);

    // Stats from last/first data points
    const last = data[data.length - 1];
    const first = data[0];
    const pctChange = last && first && first.close && last.close
        ? ((last.close - first.close) / first.close * 100).toFixed(2)
        : null;
    const isUp = pctChange ? parseFloat(pctChange) >= 0 : null;

    // Price chart with candlestick approximation via high/low area + open/close
    // Using OHLC as Open/Close bars via ComposedChart
    const priceData = data.map(d => ({
        date: d.date?.slice(5),  // MM-DD
        close: d.close,
        ema20: d.ema20,
        ema50: d.ema50,
        high: d.high,
        low: d.low,
        open: d.open,
    }));

    return (
        <div>
            <h1 className="page-title gradient-text">Chart Analysis</h1>
            <p className="page-subtitle">Technical analysis · OHLC + RSI + MACD + Volume</p>

            {/* Ticker input */}
            <div className="glass-card" style={{ padding: 18, marginBottom: 24, display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
                <div style={{ flex: 1, minWidth: 160 }}>
                    <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 6 }}>TICKER</label>
                    <input className="input" value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())}
                        onKeyDown={e => e.key === "Enter" && fetchChart()}
                        placeholder="AAPL, TSLA..." style={{ textTransform: "uppercase" }} />
                </div>
                <div>
                    <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 6 }}>PERIOD</label>
                    <div style={{ display: "flex", gap: 6 }}>
                        {PERIODS.map(p => (
                            <button key={p} onClick={() => { setPeriod(p); if (loaded) fetchChart(ticker, p); }}
                                style={{
                                    padding: "7px 12px", borderRadius: 7, border: "1px solid",
                                    borderColor: period === p ? "var(--accent)" : "var(--border)",
                                    background: period === p ? "rgba(59,130,246,0.15)" : "transparent",
                                    color: period === p ? "var(--accent)" : "var(--text-secondary)",
                                    cursor: "pointer", fontSize: "0.78rem", fontWeight: 600, transition: "all 0.15s",
                                }}>
                                {p}
                            </button>
                        ))}
                    </div>
                </div>
                <button className="btn-primary" onClick={() => fetchChart()} disabled={loading}>
                    {loading ? <span className="spinner" style={{ width: 14, height: 14 }} /> : <Search size={14} />}
                    Load Chart
                </button>
            </div>

            {error && (
                <div style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 10, padding: "12px 18px", color: "var(--red)", marginBottom: 16 }}>
                    {error}
                </div>
            )}

            {!loaded && !loading && (
                <div className="glass-card" style={{ padding: 60, textAlign: "center" }}>
                    <LineIcon size={52} style={{ color: "var(--accent)", opacity: 0.4, marginBottom: 16 }} />
                    <div style={{ color: "var(--text-secondary)", fontWeight: 600, marginBottom: 8 }}>Enter a ticker and click Load Chart</div>
                    <div style={{ color: "var(--text-muted)", fontSize: "0.875rem" }}>Shows OHLC price, EMA 20/50, RSI, MACD, Volume</div>
                </div>
            )}

            {loaded && data.length > 0 && (
                <>
                    {/* Stats bar */}
                    <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
                        <div className="metric-card" style={{ padding: "14px 20px", flex: 1, minWidth: 120 }}>
                            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>{ticker} Close</div>
                            <div style={{ fontSize: "1.5rem", fontWeight: 800 }}>${last?.close?.toFixed(2) || "—"}</div>
                        </div>
                        <div className="metric-card" style={{ padding: "14px 20px", flex: 1, minWidth: 120 }}>
                            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>Period Return</div>
                            <div style={{ fontSize: "1.5rem", fontWeight: 800, color: isUp ? "var(--green)" : "var(--red)", display: "flex", alignItems: "center", gap: 6 }}>
                                {isUp ? <TrendingUp size={20} /> : <TrendingDown size={20} />}
                                {isUp ? "+" : ""}{pctChange}%
                            </div>
                        </div>
                        <div className="metric-card" style={{ padding: "14px 20px", flex: 1, minWidth: 120 }}>
                            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>RSI (14)</div>
                            <div style={{ fontSize: "1.5rem", fontWeight: 800, color: (last?.rsi || 50) > 70 ? "var(--red)" : (last?.rsi || 50) < 30 ? "var(--green)" : "var(--text-primary)" }}>
                                {last?.rsi?.toFixed(1) || "—"}
                            </div>
                        </div>
                        <div className="metric-card" style={{ padding: "14px 20px", flex: 1, minWidth: 120 }}>
                            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>MACD</div>
                            <div style={{ fontSize: "1.5rem", fontWeight: 800, color: (last?.macd || 0) > 0 ? "var(--green)" : "var(--red)" }}>
                                {last?.macd?.toFixed(3) || "—"}
                            </div>
                        </div>
                    </div>

                    {/* Price + EMA Chart */}
                    <div className="glass-card chart-card">
                        <div style={{ fontSize: "0.85rem", fontWeight: 700, marginBottom: 14, color: "var(--text-secondary)" }}>
                            📈 {ticker} Price · EMA 20 (blue) / EMA 50 (orange)
                        </div>
                        <div className="chart-container">
                        <ResponsiveContainer width="100%" height={260} minHeight={200}>
                            <ComposedChart data={priceData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                                <XAxis dataKey="date" tick={{ fill: "var(--text-muted)", fontSize: 10 }} interval="preserveStartEnd" />
                                <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} tickFormatter={v => `$${v.toFixed(0)}`} domain={["auto", "auto"]} />
                                <Tooltip content={<CTooltip />} />
                                {/* High-Low range area */}
                                <Area type="monotone" dataKey="high" stroke="none" fill="rgba(59,130,246,0.05)" legendType="none" />
                                <Area type="monotone" dataKey="low" stroke="none" fill="var(--bg-base)" legendType="none" />
                                {/* Close line */}
                                <Line type="monotone" dataKey="close" stroke="#60a5fa" strokeWidth={2} dot={false} name="Close" />
                                <Line type="monotone" dataKey="ema20" stroke="#3b82f6" strokeWidth={1.5} dot={false} strokeDasharray="4 2" name="EMA20" />
                                <Line type="monotone" dataKey="ema50" stroke="#f59e0b" strokeWidth={1.5} dot={false} strokeDasharray="4 2" name="EMA50" />
                            </ComposedChart>
                        </ResponsiveContainer>
                        </div>
                    </div>

                    {/* RSI Chart */}
                    <div className="glass-card chart-card">
                        <div style={{ fontSize: "0.85rem", fontWeight: 700, marginBottom: 14, color: "var(--text-secondary)" }}>
                            📊 RSI (14) · Overbought: 70 · Oversold: 30
                        </div>
                        <div className="chart-container">
                        <ResponsiveContainer width="100%" height={150} minHeight={140}>
                            <ComposedChart data={data.map(d => ({ date: d.date?.slice(5), rsi: d.rsi }))}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                                <XAxis dataKey="date" tick={{ fill: "var(--text-muted)", fontSize: 10 }} interval="preserveStartEnd" />
                                <YAxis domain={[0, 100]} tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                                <Tooltip content={<CTooltip />} />
                                <ReferenceLine y={70} stroke="rgba(239,68,68,0.5)" strokeDasharray="4 2" label={{ value: "70", fill: "var(--red)", fontSize: 10 }} />
                                <ReferenceLine y={30} stroke="rgba(34,197,94,0.5)" strokeDasharray="4 2" label={{ value: "30", fill: "var(--green)", fontSize: 10 }} />
                                <Line type="monotone" dataKey="rsi" stroke="#a855f7" strokeWidth={2} dot={false} name="RSI" />
                            </ComposedChart>
                        </ResponsiveContainer>
                        </div>
                    </div>

                    {/* MACD Chart */}
                    <div className="glass-card chart-card">
                        <div style={{ fontSize: "0.85rem", fontWeight: 700, marginBottom: 14, color: "var(--text-secondary)" }}>
                            📉 MACD (12/26/9) · Signal (orange) · Histogram (bars)
                        </div>
                        <div className="chart-container">
                        <ResponsiveContainer width="100%" height={150} minHeight={140}>
                            <ComposedChart data={data.map(d => ({ date: d.date?.slice(5), macd: d.macd, signal: d.macd_signal, hist: d.macd_hist }))}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                                <XAxis dataKey="date" tick={{ fill: "var(--text-muted)", fontSize: 10 }} interval="preserveStartEnd" />
                                <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                                <Tooltip content={<CTooltip />} />
                                <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
                                <Bar dataKey="hist" name="Hist" radius={[2, 2, 0, 0]}>
                                    {data.map((d, i) => (
                                        <Cell key={i} fill={(d.macd_hist || 0) >= 0 ? "rgba(34,197,94,0.6)" : "rgba(239,68,68,0.6)"} />
                                    ))}
                                </Bar>
                                <Line type="monotone" dataKey="macd" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="MACD" />
                                <Line type="monotone" dataKey="signal" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="Signal" />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Volume Chart */}
                    <div className="glass-card" style={{ padding: 20 }}>
                        <div style={{ fontSize: "0.85rem", fontWeight: 700, marginBottom: 14, color: "var(--text-secondary)" }}>
                            📊 Volume · MA 20 (orange)
                        </div>
                        <ResponsiveContainer width="100%" height={130}>
                            <ComposedChart data={data.map(d => ({ date: d.date?.slice(5), volume: d.volume, vol_ma: d.volume_ma, close: d.close, open: d.open }))}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                                <XAxis dataKey="date" tick={{ fill: "var(--text-muted)", fontSize: 10 }} interval="preserveStartEnd" />
                                <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} tickFormatter={v => `${(v / 1e6).toFixed(1)}M`} />
                                <Tooltip content={<CTooltip />} />
                                <Bar dataKey="volume" name="Volume">
                                    {data.map((d, i) => (
                                        <Cell key={i} fill={(d.close || 0) >= (d.open || 0) ? "rgba(34,197,94,0.5)" : "rgba(239,68,68,0.5)"} />
                                    ))}
                                </Bar>
                                <Line type="monotone" dataKey="vol_ma" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="Vol MA" />
                            </ComposedChart>
                        </ResponsiveContainer>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}

export default function ChartsPage() {
    return (
        <Suspense fallback={<div className="glass-card" style={{ padding: 60, textAlign: "center" }}><span className="spinner" /></div>}>
            <ChartsContent />
        </Suspense>
    );
}
