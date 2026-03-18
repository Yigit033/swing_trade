"use client";
import { useState } from "react";
import { usePerformance, useInvalidateQueries } from "@/hooks/useApi";
import { deleteTrade } from "@/lib/api";
import type { PerformanceSummary, Trade } from "@/lib/api";
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
    LineChart, Line, CartesianGrid, Cell, ReferenceLine,
} from "recharts";
import { Award, TrendingUp, TrendingDown, Target, Trash2, Percent } from "lucide-react";

function MetricCard({ label, value, color = "#3b82f6", icon: Icon, sub }: {
    label: string; value: string; color?: string; icon?: React.ElementType; sub?: string;
}) {
    return (
        <div className="metric-card" style={{ padding: "18px 22px" }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>{label}</div>
                {Icon && <Icon size={16} color={color} />}
            </div>
            <div style={{ fontSize: "1.6rem", fontWeight: 800, marginTop: 10, color }}>{value}</div>
            {sub && <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 4 }}>{sub}</div>}
        </div>
    );
}

function TypeBadge({ type }: { type?: string }) {
    if (!type) return <span className="badge badge-blue">—</span>;
    const colors: Record<string, string> = { A: "badge-green", B: "badge-blue", C: "badge-yellow" };
    return <span className={`badge ${colors[type] || "badge-purple"}`}>{type}</span>;
}

function fmtDate(d?: string | null) {
    if (!d) return "—";
    return d.slice(0, 10);
}

const PnlTooltip = ({ active, payload, label }: any) => {
    if (active && payload?.length) {
        const v = payload[0].value;
        return (
            <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, padding: "8px 14px", fontSize: "0.8rem" }}>
                <div style={{ color: "var(--text-muted)", marginBottom: 4 }}>{label}</div>
                <div style={{ color: v >= 0 ? "var(--green)" : "var(--red)", fontWeight: 700 }}>
                    {v >= 0 ? "+" : ""}{v?.toFixed(2)}%
                </div>
            </div>
        );
    }
    return null;
};

const CumTooltip = ({ active, payload, label }: any) => {
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
    const { data, isLoading } = usePerformance();
    const { invalidatePerformance } = useInvalidateQueries();
    const [deletingId, setDeletingId] = useState<number | null>(null);
    const [msg, setMsg] = useState("");

    const handleDelete = async (t: Trade) => {
        if (!confirm(`"${t.ticker}" trade geçmişinden silinsin mi? Bu işlem geri alınamaz.`)) return;
        setDeletingId(t.id);
        try {
            await deleteTrade(t.id);
            setMsg(`✅ ${t.ticker} silindi`);
            invalidatePerformance();
        } catch {
            setMsg("❌ Silinemedi");
        } finally {
            setDeletingId(null);
        }
    };

    if (isLoading && !data) return (
        <div style={{ textAlign: "center", padding: 80 }}><span className="spinner" style={{ width: 40, height: 40 }} /></div>
    );

    const s = data?.summary;
    const closed = data?.recent_closed || [];

    const sortedClosed = [...closed].sort((a, b) => (a.exit_date || "").localeCompare(b.exit_date || ""));
    let cumPnl = 0;
    const pnlCurve = sortedClosed.map(t => {
        cumPnl += t.realized_pnl || 0;
        return { date: t.exit_date?.slice(5) || "", pnl: cumPnl, trade: t.ticker };
    });

    const tradeBars: { ticker: string; pnlPct: number }[] = closed.slice(0, 20).map((t: Trade) => ({
        ticker: t.ticker,
        pnlPct: t.realized_pnl_pct || 0,
    }));

    const profitFactor = s && s.avg_loss < 0
        ? Math.abs((s.wins * s.avg_win) / (s.losses * s.avg_loss)).toFixed(2)
        : s && s.avg_loss === 0 && s.wins > 0 ? "∞" : "—";

    // Best / worst trade
    const bestTrade = closed.reduce((best: Trade | undefined, t: Trade) =>
        (t.realized_pnl_pct || 0) > (best?.realized_pnl_pct || -Infinity) ? t : best, closed[0]);
    const worstTrade = closed.reduce((worst: Trade | undefined, t: Trade) =>
        (t.realized_pnl_pct || 0) < (worst?.realized_pnl_pct || Infinity) ? t : worst, closed[0]);

    return (
        <div>
            <h1 className="page-title gradient-text">Performance Analytics</h1>
            <p className="page-subtitle">Closed trade deep-dive · historical P&amp;L analysis</p>

            {msg && (
                <div style={{ background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.3)", borderRadius: 10, padding: "10px 16px", marginBottom: 16, fontSize: "0.875rem", color: "var(--accent)" }}>
                    {msg}
                </div>
            )}

            {/* KPI grid */}
            <div className="metrics-grid" style={{ marginBottom: 28 }}>
                <MetricCard
                    label="Total P&L %"
                    value={s ? `${(s.total_pnl_pct >= 0 ? "+" : "")}${s.total_pnl_pct.toFixed(2)}%` : "—"}
                    sub={s ? `Avg: ${s.avg_pnl_pct.toFixed(2)}% / trade` : undefined}
                    color={s && s.total_pnl_pct >= 0 ? "#22c55e" : "#ef4444"}
                    icon={Percent}
                />
                <MetricCard label="Win Rate" value={s ? `${s.win_rate}%` : "—"} sub={s ? `${s.wins}W / ${s.losses}L / ${s.breakeven}BE` : undefined} color="#3b82f6" icon={Target} />
                <MetricCard label="Total P&L $" value={s ? `$${s.total_pnl.toFixed(2)}` : "—"}
                    color={s && s.total_pnl >= 0 ? "#22c55e" : "#ef4444"} icon={s && s.total_pnl >= 0 ? TrendingUp : TrendingDown} />
                <MetricCard label="Avg Win" value={s ? `$${s.avg_win.toFixed(2)}` : "—"} color="#22c55e" icon={TrendingUp} />
                <MetricCard label="Closed Trades" value={s ? String(s.closed_trades) : "—"} color="#a855f7" icon={Award} />
                <MetricCard label="Profit Factor" value={String(profitFactor)} color="#f59e0b" />
            </div>

            {closed.length === 0 ? (
                <div className="glass-card" style={{ padding: 60, textAlign: "center", color: "var(--text-muted)" }}>
                    No closed trades yet — start tracking signals from the Scanner.
                </div>
            ) : (
                <>
                    {/* Best / Worst */}
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
                        {bestTrade && (
                            <div className="glass-card" style={{ padding: "14px 18px" }}>
                                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase", marginBottom: 8 }}>🏆 Best Trade</div>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                    <span style={{ fontWeight: 700, color: "var(--accent)" }}>{bestTrade.ticker}</span>
                                    <span style={{ color: "var(--green)", fontWeight: 700 }}>+{bestTrade.realized_pnl_pct?.toFixed(2)}%</span>
                                </div>
                                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 4 }}>{fmtDate(bestTrade.exit_date)}</div>
                            </div>
                        )}
                        {worstTrade && (
                            <div className="glass-card" style={{ padding: "14px 18px" }}>
                                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase", marginBottom: 8 }}>💀 Worst Trade</div>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                    <span style={{ fontWeight: 700, color: "var(--accent)" }}>{worstTrade.ticker}</span>
                                    <span style={{ color: "var(--red)", fontWeight: 700 }}>{worstTrade.realized_pnl_pct?.toFixed(2)}%</span>
                                </div>
                                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 4 }}>{fmtDate(worstTrade.exit_date)}</div>
                            </div>
                        )}
                    </div>

                    {/* Cumulative PnL curve */}
                    <div className="glass-card chart-card">
                        <h3 style={{ margin: "0 0 20px", fontSize: "0.95rem", fontWeight: 700 }}>📈 Cumulative P&L ($)</h3>
                        <div className="chart-container">
                        <ResponsiveContainer width="100%" height={220} minHeight={180}>
                            <LineChart data={pnlCurve}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="date" tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                                <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickFormatter={v => `$${v}`} />
                                <Tooltip content={<CumTooltip />} />
                                <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
                                <Line type="monotone" dataKey="pnl" stroke="#3b82f6" strokeWidth={2} dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Per-trade P&L % bars */}
                    <div className="glass-card chart-card">
                        <h3 style={{ margin: "0 0 20px", fontSize: "0.95rem", fontWeight: 700 }}>📊 Per-Trade P&L % (Last 20)</h3>
                        <div className="chart-container">
                        <ResponsiveContainer width="100%" height={200} minHeight={160}>
                            <BarChart data={tradeBars}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="ticker" tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                                <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickFormatter={v => `${v}%`} />
                                <Tooltip content={<PnlTooltip />} />
                                <ReferenceLine y={0} stroke="rgba(255,255,255,0.2)" />
                                <Bar dataKey="pnlPct" radius={[4, 4, 0, 0]} name="P&L %">
                                    {tradeBars.map((entry, i) => (
                                        <Cell key={i} fill={entry.pnlPct >= 0 ? "#22c55e" : "#ef4444"} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Closed Trade History */}
                    <div className="glass-card" style={{ overflow: "hidden" }}>
                        <div style={{ padding: "16px 22px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <h3 style={{ margin: 0, fontSize: "0.95rem", fontWeight: 700 }}>📋 Closed Trade History</h3>
                            <span className="badge badge-blue">{closed.length} trades</span>
                        </div>
                        <div style={{ overflowX: "auto" }}>
                            <table className="data-table">
                                <thead>
                                    <tr>
                                        <th>Ticker</th><th>Type</th><th>Entry Date</th><th>Close Date</th>
                                        <th>Entry $</th><th>Exit $</th><th>P&amp;L $</th><th>P&amp;L %</th>
                                        <th>Result</th><th>Delete</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {closed.map(t => {
                                        const pnl = t.realized_pnl || 0;
                                        const win = pnl > 0;
                                        const statusColor =
                                            t.status === "TARGET" ? "badge-green" :
                                                t.status === "STOPPED" ? "badge-red" :
                                                    t.status === "TRAILED" ? "badge-yellow" :
                                                        t.status === "MANUAL" ? "badge-blue" :
                                                            t.status === "REJECTED" ? "badge-red" :
                                                                win ? "badge-green" : "badge-red";
                                        return (
                                            <tr key={t.id}>
                                                <td><strong>{t.ticker}</strong></td>
                                                <td><TypeBadge type={t.swing_type} /></td>
                                                <td style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>{fmtDate(t.entry_date)}</td>
                                                <td style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>{fmtDate(t.exit_date)}</td>
                                                <td>${t.entry_price?.toFixed(2)}</td>
                                                <td>${t.exit_price?.toFixed(2) || "—"}</td>
                                                <td style={{ color: pnl >= 0 ? "var(--green)" : "var(--red)", fontWeight: 600 }}>
                                                    {pnl >= 0 ? "+$" : "-$"}{Math.abs(pnl).toFixed(2)}
                                                </td>
                                                <td style={{ color: pnl >= 0 ? "var(--green)" : "var(--red)" }}>
                                                    {pnl >= 0 ? "+" : ""}{t.realized_pnl_pct?.toFixed(2) ?? "—"}%
                                                </td>
                                                <td><span className={`badge ${statusColor}`}>{t.status}</span></td>
                                                <td>
                                                    <button
                                                        className="btn-danger"
                                                        onClick={() => handleDelete(t)}
                                                        disabled={deletingId === t.id}
                                                        title="Delete trade"
                                                    >
                                                        {deletingId === t.id
                                                            ? <span className="spinner" style={{ width: 11, height: 11 }} />
                                                            : <Trash2 size={11} />}
                                                    </button>
                                                </td>
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
