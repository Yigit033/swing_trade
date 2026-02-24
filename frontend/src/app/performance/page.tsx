"use client";
import { useState, useEffect } from "react";
import { getPerformance } from "@/lib/api";
import type { PerformanceSummary, Trade } from "@/lib/api";
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
    LineChart, Line, CartesianGrid, Cell,
} from "recharts";
import { Award, TrendingUp, TrendingDown, Target } from "lucide-react";

function MetricCard({ label, value, color = "#3b82f6", icon: Icon }: {
    label: string; value: string; color?: string; icon?: React.ElementType;
}) {
    return (
        <div className="metric-card" style={{ padding: "18px 22px" }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>{label}</div>
                {Icon && <Icon size={16} color={color} />}
            </div>
            <div style={{ fontSize: "1.6rem", fontWeight: 800, marginTop: 10, color }}>{value}</div>
        </div>
    );
}

const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload?.length) {
        return (
            <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, padding: "8px 14px", fontSize: "0.8rem" }}>
                <div style={{ color: "var(--text-muted)", marginBottom: 4 }}>{label}</div>
                <div style={{ color: payload[0].value >= 0 ? "var(--green)" : "var(--red)", fontWeight: 700 }}>
                    ${payload[0].value?.toFixed(2)}
                </div>
            </div>
        );
    }
    return null;
};

export default function PerformancePage() {
    const [data, setData] = useState<{ summary: PerformanceSummary; open_trades: Trade[]; recent_closed: Trade[] } | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        getPerformance().then(setData).finally(() => setLoading(false));
    }, []);

    if (loading) return (
        <div style={{ textAlign: "center", padding: 80 }}><span className="spinner" style={{ width: 40, height: 40 }} /></div>
    );

    const s = data?.summary;
    const closed = data?.recent_closed || [];

    // Build cumulative PnL curve
    const sortedClosed = [...closed].sort((a, b) => (a.exit_date || "").localeCompare(b.exit_date || ""));
    let cumPnl = 0;
    const pnlCurve = sortedClosed.map(t => {
        cumPnl += t.realized_pnl || 0;
        return { date: t.exit_date?.slice(5) || "", pnl: cumPnl, trade: t.ticker };
    });

    // Per-trade PnL bars
    const tradeBars = closed.slice(0, 15).map(t => ({
        ticker: t.ticker,
        pnl: t.realized_pnl || 0,
    }));

    const profitFactor = s && s.avg_loss < 0
        ? Math.abs((s.wins * s.avg_win) / (s.losses * s.avg_loss)).toFixed(2)
        : "—";

    return (
        <div>
            <h1 className="page-title gradient-text">Performance Analytics</h1>
            <p className="page-subtitle">Historical trade analysis · closed positions</p>

            {/* KPI grid */}
            <div className="metrics-grid" style={{ marginBottom: 28 }}>
                <MetricCard label="Win Rate" value={s ? `${s.win_rate}%` : "—"} color="#3b82f6" icon={Target} />
                <MetricCard label="Total P&L" value={s ? `$${s.total_pnl.toFixed(2)}` : "—"}
                    color={s && s.total_pnl >= 0 ? "#22c55e" : "#ef4444"} icon={s && s.total_pnl >= 0 ? TrendingUp : TrendingDown} />
                <MetricCard label="Avg Win" value={s ? `$${s.avg_win.toFixed(2)}` : "—"} color="#22c55e" icon={TrendingUp} />
                <MetricCard label="Avg Loss" value={s ? `$${s.avg_loss.toFixed(2)}` : "—"} color="#ef4444" icon={TrendingDown} />
                <MetricCard label="Closed Trades" value={s ? String(s.closed_trades) : "—"} color="#a855f7" icon={Award} />
                <MetricCard label="Profit Factor" value={String(profitFactor)} color="#f59e0b" />
            </div>

            {closed.length === 0 ? (
                <div className="glass-card" style={{ padding: 60, textAlign: "center", color: "var(--text-muted)" }}>
                    No closed trades yet — start tracking signals from the Scanner.
                </div>
            ) : (
                <>
                    {/* Cumulative PnL curve */}
                    <div className="glass-card" style={{ padding: 22, marginBottom: 20 }}>
                        <h3 style={{ margin: "0 0 20px", fontSize: "0.95rem", fontWeight: 700 }}>📈 Cumulative P&L</h3>
                        <ResponsiveContainer width="100%" height={220}>
                            <LineChart data={pnlCurve}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="date" tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                                <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickFormatter={v => `$${v}`} />
                                <Tooltip content={<CustomTooltip />} />
                                <Line type="monotone" dataKey="pnl" stroke="#3b82f6" strokeWidth={2} dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Per-trade bars */}
                    <div className="glass-card" style={{ padding: 22, marginBottom: 20 }}>
                        <h3 style={{ margin: "0 0 20px", fontSize: "0.95rem", fontWeight: 700 }}>📊 Trade P&L (Last 15)</h3>
                        <ResponsiveContainer width="100%" height={220}>
                            <BarChart data={tradeBars}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="ticker" tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                                <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickFormatter={v => `$${v}`} />
                                <Tooltip content={<CustomTooltip />} />
                                <Bar dataKey="pnl" radius={[4, 4, 0, 0]}>
                                    {tradeBars.map((entry, i) => (
                                        <Cell key={i} fill={entry.pnl >= 0 ? "#22c55e" : "#ef4444"} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Detailed table */}
                    <div className="glass-card" style={{ overflow: "hidden" }}>
                        <div style={{ padding: "16px 22px", borderBottom: "1px solid var(--border)" }}>
                            <h3 style={{ margin: 0, fontSize: "0.95rem", fontWeight: 700 }}>📋 Closed Trade History</h3>
                        </div>
                        <div style={{ overflowX: "auto" }}>
                            <table className="data-table">
                                <thead>
                                    <tr><th>Ticker</th><th>Entry</th><th>Exit</th><th>P&L</th><th>P&L %</th><th>Result</th><th>Close Date</th></tr>
                                </thead>
                                <tbody>
                                    {closed.map(t => {
                                        const win = (t.realized_pnl || 0) >= 0;
                                        return (
                                            <tr key={t.id}>
                                                <td><strong>{t.ticker}</strong></td>
                                                <td>${t.entry_price?.toFixed(2)}</td>
                                                <td>${t.exit_price?.toFixed(2) || "—"}</td>
                                                <td style={{ color: win ? "var(--green)" : "var(--red)", fontWeight: 600 }}>
                                                    {win ? "+" : ""}{t.realized_pnl?.toFixed(2)}$
                                                </td>
                                                <td style={{ color: win ? "var(--green)" : "var(--red)" }}>
                                                    {win ? "+" : ""}{t.realized_pnl_pct?.toFixed(2)}%
                                                </td>
                                                <td><span className={`badge ${win ? "badge-green" : "badge-red"}`}>{win ? "WIN" : "LOSS"}</span></td>
                                                <td style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>{t.exit_date}</td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
