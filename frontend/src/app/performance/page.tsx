"use client";
import { useState } from "react";
import { usePerformance, useInvalidateQueries } from "@/hooks/useApi";
import { deleteTrade } from "@/lib/api";
import type { PerformanceSummary, StrategyView, Trade } from "@/lib/api";
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

const fmtSigned = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;

export default function PerformancePage() {
    const [view, setView] = useState<StrategyView>("H6S");
    const { data, isLoading, isError, refetch } = usePerformance(view);
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
    if (isError) return <div role="alert" className="card">Performans verisi yüklenemedi; boş işlem geçmişi sonucu değildir. <button onClick={() => refetch()}>Tekrar dene</button></div>;

    const s = data?.summary;
    const closed = (data?.recent_closed || []).filter(t =>
        t.status !== "REJECTED" && t.realized_pnl != null && t.realized_pnl_pct != null);

    // Gerçek para = maliyet sonrası. Eski backend net alan göndermezse brüte düşülür.
    const netPct = (t: Trade) => t.net_pnl_pct ?? t.realized_pnl_pct ?? 0;
    const netUsd = (t: Trade) => t.net_pnl ?? t.realized_pnl ?? 0;
    const costPct = s?.cost_round_trip_pct;
    const feeUsd = s?.commission_usd_per_side;

    const sortedClosed = [...closed].sort((a, b) => (a.exit_date || "").localeCompare(b.exit_date || ""));
    let cumPnl = 0;
    const pnlCurve = sortedClosed.map(t => {
        cumPnl += netUsd(t);
        return { date: t.exit_date?.slice(5) || "", pnl: cumPnl, trade: t.ticker };
    });

    const tradeBars: { ticker: string; pnlPct: number }[] = closed.slice(0, 20).map((t: Trade) => ({
        ticker: t.ticker,
        pnlPct: netPct(t),
    }));

    const profitFactor = s && s.avg_loss < 0
        ? Math.abs((s.wins * s.avg_win) / (s.losses * s.avg_loss)).toFixed(2)
        : s && s.avg_loss === 0 && s.wins > 0 ? "∞" : "—";

    // Best / worst trade
    const bestTrade = closed.reduce((best: Trade | undefined, t: Trade) =>
        netPct(t) > (best ? netPct(best) : -Infinity) ? t : best, closed[0]);
    const worstTrade = closed.reduce((worst: Trade | undefined, t: Trade) =>
        netPct(t) < (worst ? netPct(worst) : Infinity) ? t : worst, closed[0]);

    return (
        <div>
            <h1 className="page-title gradient-text">Performance Analytics</h1>
            <p className="page-subtitle">Grafikler ve en iyi/kötü işlem, API’nin son 30 kapanış kaydı içindeki gerçekleşmiş işlemlerle sınırlıdır.</p>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 16 }}>
                {([["H6S", "H6S · Profesyonel VCE (ana strateji)"], ["LEGACY", "Eski kurallar (kontrol grubu)"], ["ALL", "Tümü"]] as const).map(([k, label]) => (
                    <button key={k} className={view === k ? "btn-primary" : "btn-secondary"}
                        style={{ padding: "4px 12px", fontSize: 12 }} onClick={() => setView(k)}>{label}</button>
                ))}
                <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Her strateji ayrı sanal hesapta işler; kıyas aynı dönem, aynı maliyetle.</span>
            </div>

            {msg && (
                <div style={{ background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.3)", borderRadius: 10, padding: "10px 16px", marginBottom: 16, fontSize: "0.875rem", color: "var(--accent)" }}>
                    {msg}
                </div>
            )}

            {/* KPI grid */}
            <div className="metrics-grid" style={{ marginBottom: 28 }}>
                <MetricCard
                    label="NET işlem getirileri toplamı % (maliyet sonrası; portföy getirisi değil)"
                    value={s ? fmtSigned(s.net_total_pnl_pct ?? s.total_pnl_pct) + "%" : "—"}
                    sub={s ? `Ort. net ${(s.net_avg_pnl_pct ?? s.avg_pnl_pct).toFixed(2)}% / işlem · brüt toplam ${fmtSigned(s.total_pnl_pct)}%${costPct != null ? ` · maliyet ${costPct.toFixed(2)}% / işlem` : ""}${feeUsd ? ` + ${feeUsd}$ × emir` : ""}` : undefined}
                    color={s && (s.net_total_pnl_pct ?? s.total_pnl_pct) >= 0 ? "#22c55e" : "#ef4444"}
                    icon={Percent}
                />
                <MetricCard label="Net Win Rate" value={s ? `${s.net_win_rate ?? s.win_rate}%` : "—"}
                    sub={s ? `${s.net_wins ?? s.wins}W / ${s.net_losses ?? s.losses}L net · brüt %${s.win_rate}` : undefined} color="#3b82f6" icon={Target} />
                <MetricCard label="NET P&L $ (maliyet sonrası)" value={s ? `$${(s.net_total_pnl ?? s.total_pnl).toFixed(2)}` : "—"}
                    sub={s ? `Brüt $${s.total_pnl.toFixed(2)}` : undefined}
                    color={s && (s.net_total_pnl ?? s.total_pnl) >= 0 ? "#22c55e" : "#ef4444"} icon={s && (s.net_total_pnl ?? s.total_pnl) >= 0 ? TrendingUp : TrendingDown} />
                <MetricCard label="Avg Win (brüt)" value={s ? `$${s.avg_win.toFixed(2)}` : "—"} color="#22c55e" icon={TrendingUp} />
                <MetricCard label="Closed Trades" value={s ? String(s.closed_trades) : "—"} color="#a855f7" icon={Award} />
                <MetricCard label="Profit Factor (brüt)" value={String(profitFactor)} color="#f59e0b" />
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
                                    <span style={{ color: netPct(bestTrade) >= 0 ? "var(--green)" : "var(--red)", fontWeight: 700 }}>{fmtSigned(netPct(bestTrade))}% net</span>
                                </div>
                                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 4 }}>{fmtDate(bestTrade.exit_date)}</div>
                            </div>
                        )}
                        {worstTrade && (
                            <div className="glass-card" style={{ padding: "14px 18px" }}>
                                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase", marginBottom: 8 }}>💀 Worst Trade</div>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                    <span style={{ fontWeight: 700, color: "var(--accent)" }}>{worstTrade.ticker}</span>
                                    <span style={{ color: netPct(worstTrade) >= 0 ? "var(--green)" : "var(--red)", fontWeight: 700 }}>{fmtSigned(netPct(worstTrade))}% net</span>
                                </div>
                                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 4 }}>{fmtDate(worstTrade.exit_date)}</div>
                            </div>
                        )}
                    </div>

                    {/* Cumulative PnL curve */}
                    <div className="glass-card chart-card">
                        <h3 style={{ margin: "0 0 20px", fontSize: "0.95rem", fontWeight: 700 }}>📈 Gösterilen işlemlerin kümülatif NET P&amp;L’i ($, maliyet sonrası)</h3>
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
                        <h3 style={{ margin: "0 0 20px", fontSize: "0.95rem", fontWeight: 700 }}>📊 İşlem başı NET P&L % (son 20)</h3>
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
                                        <th>Entry $</th><th>Exit $</th><th>Net P&amp;L $</th><th>Net P&amp;L %</th>
                                        <th>Result</th><th>Delete</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {closed.map(t => {
                                        const pnl = netUsd(t);
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
                                                <td><strong>{t.ticker}</strong>{String(t.strategy || "").toUpperCase() === "H6S" ? <span className="badge badge-green" style={{ marginLeft: 6, fontSize: "0.65rem" }} title="H6S Profesyonel VCE (ana strateji)">H6S</span> : <span className="badge badge-yellow" style={{ marginLeft: 6, fontSize: "0.65rem" }} title="Eski kurallar (kontrol grubu / manuel)">Eski</span>}</td>
                                                <td><TypeBadge type={t.swing_type} /></td>
                                                <td style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>{fmtDate(t.entry_date)}</td>
                                                <td style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>{fmtDate(t.exit_date)}</td>
                                                <td>${t.entry_price?.toFixed(2)}</td>
                                                <td>${t.exit_price?.toFixed(2) || "—"}</td>
                                                <td style={{ color: pnl >= 0 ? "var(--green)" : "var(--red)", fontWeight: 600 }}>
                                                    {pnl >= 0 ? "+$" : "-$"}{Math.abs(pnl).toFixed(2)}
                                                </td>
                                                <td style={{ color: pnl >= 0 ? "var(--green)" : "var(--red)" }}
                                                    title={`Brüt ${fmtSigned(t.realized_pnl_pct ?? 0)}% · $${(t.realized_pnl ?? 0).toFixed(2)}`}>
                                                    {fmtSigned(netPct(t))}%
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
