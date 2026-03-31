"use client";
import { useState, useCallback, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import {
    ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Cell, ReferenceLine, Area, AreaChart,
    ReferenceArea,
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

const CHART_THEME = {
    grid: "rgba(255,255,255,0.06)",
    axisMuted: "rgba(255,255,255,0.38)",
    surface: "rgba(12, 18, 32, 0.92)",
    border: "rgba(255,255,255,0.10)",
    up: "rgba(34,197,94,0.75)",
    down: "rgba(239,68,68,0.75)",
    close: "#8bd3ff",
    ema20: "#a78bfa",
    ema50: "#fbbf24",
    rsi: "#c4b5fd",
    macd: "#60a5fa",
    signal: "#f59e0b",
};

function formatCompact(n: number) {
    const abs = Math.abs(n);
    if (abs >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
    if (abs >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
    if (abs >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
    if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
    return `${n.toFixed(2)}`;
}

function ChartLegend({ payload, alignRight }: { payload: Array<{ value: string; color: string }>; alignRight?: boolean }) {
    const items = (payload || []).filter((p) => p?.value && p?.color);
    if (!items.length) return null;
    return (
        <div
            style={{
                display: "flex",
                gap: 10,
                justifyContent: alignRight ? "flex-end" : "flex-start",
                flexWrap: "wrap",
                paddingBottom: 2,
            }}
        >
            {items.map((it) => (
                <span
                    key={it.value}
                    style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 8,
                        padding: "4px 8px",
                        borderRadius: 999,
                        border: "1px solid rgba(255,255,255,0.14)",
                        background: "rgba(255,255,255,0.04)",
                        color: "rgba(255,255,255,0.78)",
                        fontSize: 12,
                        fontWeight: 700,
                        letterSpacing: "0.01em",
                        lineHeight: 1.1,
                    }}
                >
                    <span style={{ width: 8, height: 8, borderRadius: 999, background: it.color, boxShadow: "0 0 0 2px rgba(0,0,0,0.18)" }} />
                    {it.value}
                </span>
            ))}
        </div>
    );
}

const CTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null;
    const rows = payload
        .filter((p: any) => p && p.value != null && p.name && p.name !== "Close (area)")
        .map((p: any) => {
            const name = String(p.name);
            const v = p.value;
            const isVol = /vol/i.test(name);
            const val = typeof v === "number"
                ? (isVol ? formatCompact(v) : v.toFixed(2))
                : String(v);
            return { name, val, color: p.color };
        });
    return (
        <div
            style={{
                background: CHART_THEME.surface,
                border: `1px solid ${CHART_THEME.border}`,
                borderRadius: 10,
                padding: "10px 12px",
                fontSize: "0.78rem",
                minWidth: 180,
                boxShadow: "0 10px 26px rgba(0,0,0,0.40)",
                backdropFilter: "blur(8px)",
            }}
        >
            <div style={{ color: "rgba(255,255,255,0.75)", marginBottom: 8, fontWeight: 700, letterSpacing: "0.01em" }}>{label}</div>
            <div style={{ display: "grid", gap: 6 }}>
                {rows.map((r: any, i: number) => (
                    <div key={i} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
                            <span style={{ width: 8, height: 8, borderRadius: 999, background: r.color, boxShadow: `0 0 0 2px rgba(255,255,255,0.06)`, flexShrink: 0 }} />
                            <span style={{ color: "rgba(255,255,255,0.78)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.name}</span>
                        </div>
                        <strong style={{ color: "rgba(255,255,255,0.92)", fontVariantNumeric: "tabular-nums" }}>{r.val}</strong>
                    </div>
                ))}
            </div>
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
                        <div className="chart-title-row">
                            <div className="chart-title">Price</div>
                            <div className="chart-subtitle">{ticker} · OHLC band + Close + EMA20/EMA50</div>
                            <div style={{ marginLeft: "auto" }}>
                                <ChartLegend
                                    payload={[
                                        { value: "Close", color: CHART_THEME.close },
                                        { value: "EMA20", color: CHART_THEME.ema20 },
                                        { value: "EMA50", color: CHART_THEME.ema50 },
                                        { value: "High", color: "rgba(139,211,255,0.55)" },
                                        { value: "Low", color: "rgba(255,255,255,0.22)" },
                                    ]}
                                    alignRight
                                />
                            </div>
                        </div>
                        <div className="chart-container">
                        <ResponsiveContainer width="100%" height={260} minHeight={200}>
                            <ComposedChart data={priceData}>
                                <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} />
                                <XAxis dataKey="date" tick={{ fill: CHART_THEME.axisMuted, fontSize: 11 }} tickMargin={8} interval="preserveStartEnd" />
                                <YAxis tick={{ fill: CHART_THEME.axisMuted, fontSize: 11 }} tickMargin={10} width={52} tickFormatter={v => `$${Number(v).toFixed(0)}`} domain={["auto", "auto"]} />
                                <Tooltip content={<CTooltip />} cursor={{ stroke: "rgba(255,255,255,0.10)", strokeDasharray: "4 3" }} />
                                {/* Legend moved to header (right side) */}
                                {/* High-Low range area (keep feature) */}
                                <Area type="monotone" dataKey="high" stroke="none" fill="rgba(139,211,255,0.08)" name="High" legendType="none" />
                                <Area type="monotone" dataKey="low" stroke="none" fill="var(--bg-base)" name="Low" legendType="none" />
                                {/* Subtle area under close for readability */}
                                <Area type="monotone" dataKey="close" stroke="none" fill="rgba(139,211,255,0.10)" name="Close (area)" legendType="none" />
                                {/* Close + EMAs */}
                                <Line type="monotone" dataKey="close" stroke={CHART_THEME.close} strokeWidth={2.2} dot={false} name="Close" />
                                <Line type="monotone" dataKey="ema20" stroke={CHART_THEME.ema20} strokeWidth={1.7} dot={false} strokeDasharray="4 3" name="EMA20" />
                                <Line type="monotone" dataKey="ema50" stroke={CHART_THEME.ema50} strokeWidth={1.7} dot={false} strokeDasharray="4 3" name="EMA50" />
                            </ComposedChart>
                        </ResponsiveContainer>
                        </div>
                    </div>

                    {/* RSI Chart */}
                    <div className="glass-card chart-card">
                        <div className="chart-title-row">
                            <div className="chart-title">RSI</div>
                            <div className="chart-subtitle">14-period · Zones: 30 / 70</div>
                            <div style={{ marginLeft: "auto" }}>
                                <ChartLegend
                                    payload={[
                                        { value: "RSI", color: CHART_THEME.rsi },
                                        { value: "Overbought 70", color: "rgba(239,68,68,0.75)" },
                                        { value: "Oversold 30", color: "rgba(34,197,94,0.75)" },
                                    ]}
                                    alignRight
                                />
                            </div>
                        </div>
                        <div className="chart-container">
                        <ResponsiveContainer width="100%" height={150} minHeight={140}>
                            <ComposedChart data={data.map(d => ({ date: d.date?.slice(5), rsi: d.rsi }))}>
                                <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} />
                                <XAxis dataKey="date" tick={{ fill: CHART_THEME.axisMuted, fontSize: 11 }} tickMargin={8} interval="preserveStartEnd" />
                                <YAxis domain={[0, 100]} tick={{ fill: CHART_THEME.axisMuted, fontSize: 11 }} tickMargin={10} width={44} />
                                <Tooltip content={<CTooltip />} cursor={{ stroke: "rgba(255,255,255,0.10)", strokeDasharray: "4 3" }} />
                                <ReferenceArea y1={30} y2={70} fill="rgba(255,255,255,0.03)" stroke="none" />
                                <ReferenceLine y={70} stroke="rgba(239,68,68,0.55)" strokeDasharray="4 3" />
                                <ReferenceLine y={30} stroke="rgba(34,197,94,0.55)" strokeDasharray="4 3" />
                                <Line type="monotone" dataKey="rsi" stroke={CHART_THEME.rsi} strokeWidth={2.2} dot={false} name="RSI" />
                            </ComposedChart>
                        </ResponsiveContainer>
                        </div>
                    </div>

                    {/* MACD Chart */}
                    <div className="glass-card chart-card">
                        <div className="chart-title-row">
                            <div className="chart-title">MACD</div>
                            <div className="chart-subtitle">12/26/9 · Signal + Histogram</div>
                            <div style={{ marginLeft: "auto" }}>
                                <ChartLegend
                                    payload={[
                                        { value: "MACD", color: CHART_THEME.macd },
                                        { value: "Signal", color: CHART_THEME.signal },
                                        { value: "Hist Up", color: CHART_THEME.up },
                                        { value: "Hist Down", color: CHART_THEME.down },
                                    ]}
                                    alignRight
                                />
                            </div>
                        </div>
                        <div className="chart-container">
                        <ResponsiveContainer width="100%" height={150} minHeight={140}>
                            <ComposedChart data={data.map(d => ({ date: d.date?.slice(5), macd: d.macd, signal: d.macd_signal, hist: d.macd_hist }))}>
                                <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} />
                                <XAxis dataKey="date" tick={{ fill: CHART_THEME.axisMuted, fontSize: 11 }} tickMargin={8} interval="preserveStartEnd" />
                                <YAxis tick={{ fill: CHART_THEME.axisMuted, fontSize: 11 }} tickMargin={10} width={44} />
                                <Tooltip content={<CTooltip />} cursor={{ stroke: "rgba(255,255,255,0.10)", strokeDasharray: "4 3" }} />
                                <ReferenceLine y={0} stroke="rgba(255,255,255,0.18)" />
                                <Bar dataKey="hist" name="Hist" radius={[2, 2, 0, 0]}>
                                    {data.map((d, i) => (
                                        <Cell key={i} fill={(d.macd_hist || 0) >= 0 ? CHART_THEME.up : CHART_THEME.down} />
                                    ))}
                                </Bar>
                                <Line type="monotone" dataKey="macd" stroke={CHART_THEME.macd} strokeWidth={2} dot={false} name="MACD" />
                                <Line type="monotone" dataKey="signal" stroke={CHART_THEME.signal} strokeWidth={2} dot={false} name="Signal" />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Volume Chart */}
                    <div className="glass-card chart-card">
                        <div className="chart-title-row">
                            <div className="chart-title">Volume</div>
                            <div className="chart-subtitle">Bars + MA20</div>
                            <div style={{ marginLeft: "auto" }}>
                                <ChartLegend
                                    payload={[
                                        { value: "Volume", color: "rgba(255,255,255,0.30)" },
                                        { value: "Vol MA", color: CHART_THEME.signal },
                                        { value: "Up", color: "rgba(34,197,94,0.55)" },
                                        { value: "Down", color: "rgba(239,68,68,0.55)" },
                                    ]}
                                    alignRight
                                />
                            </div>
                        </div>
                        <ResponsiveContainer width="100%" height={130}>
                            <ComposedChart data={data.map(d => ({ date: d.date?.slice(5), volume: d.volume, vol_ma: d.volume_ma, close: d.close, open: d.open }))}>
                                <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} />
                                <XAxis dataKey="date" tick={{ fill: CHART_THEME.axisMuted, fontSize: 11 }} tickMargin={8} interval="preserveStartEnd" />
                                <YAxis tick={{ fill: CHART_THEME.axisMuted, fontSize: 11 }} tickMargin={10} width={54} tickFormatter={v => formatCompact(Number(v))} />
                                <Tooltip content={<CTooltip />} cursor={{ stroke: "rgba(255,255,255,0.10)", strokeDasharray: "4 3" }} />
                                <Bar dataKey="volume" name="Volume">
                                    {data.map((d, i) => (
                                        <Cell key={i} fill={(d.close || 0) >= (d.open || 0) ? "rgba(34,197,94,0.5)" : "rgba(239,68,68,0.5)"} />
                                    ))}
                                </Bar>
                                <Line type="monotone" dataKey="vol_ma" stroke={CHART_THEME.signal} strokeWidth={2} dot={false} name="Vol MA" />
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
