"use client";
import { useState, useEffect, useMemo } from "react";
import { getTrades, closeTrade, deleteTrade, updateTrade, updatePrices, getTradesLastUpdate } from "@/lib/api";
import type { Trade } from "@/lib/api";
import { RefreshCw, X, CheckSquare, TrendingUp, TrendingDown, Edit2, Search, ChevronLeft, ChevronRight } from "lucide-react";

type FilterStatus = "ALL" | "OPEN" | "CLOSED" | "PENDING";

const CLOSED_STATUSES = new Set(["STOPPED", "TRAILED", "TARGET", "MANUAL", "WIN", "LOSS", "CLOSED", "REJECTED", "TIMEOUT"]);

/* ─── Badge Components ────────────────────────────── */
function TypeBadge({ type }: { type?: string }) {
    const map: Record<string, { bg: string; color: string }> = {
        A: { bg: "rgba(34,197,94,0.15)", color: "#22c55e" },
        B: { bg: "rgba(59,130,246,0.15)", color: "#3b82f6" },
        C: { bg: "rgba(245,158,11,0.15)", color: "#f59e0b" },
        S: { bg: "rgba(168,85,247,0.15)", color: "#a855f7" },
    };
    const s = map[type || ""] || { bg: "rgba(148,163,184,0.15)", color: "#94a3b8" };
    return <span style={{ padding: "2px 10px", borderRadius: 6, fontSize: "0.72rem", fontWeight: 700, background: s.bg, color: s.color }}>{type || "—"}</span>;
}

function StatusBadge({ status }: { status: string }) {
    const map: Record<string, { bg: string; color: string }> = {
        OPEN: { bg: "rgba(59,130,246,0.15)", color: "#3b82f6" },
        PENDING: { bg: "rgba(245,158,11,0.15)", color: "#f59e0b" },
        TARGET: { bg: "rgba(34,197,94,0.2)", color: "#22c55e" },
        TRAILED: { bg: "rgba(245,158,11,0.15)", color: "#f59e0b" },
        STOPPED: { bg: "rgba(239,68,68,0.15)", color: "#ef4444" },
        T1_PARTIAL: { bg: "rgba(34,197,94,0.15)", color: "#22c55e" },
        MANUAL: { bg: "rgba(59,130,246,0.15)", color: "#3b82f6" },
        "I-MANUAL": { bg: "rgba(59,130,246,0.15)", color: "#3b82f6" },
        REJECTED: { bg: "rgba(239,68,68,0.15)", color: "#ef4444" },
        WIN: { bg: "rgba(34,197,94,0.2)", color: "#22c55e" },
        LOSS: { bg: "rgba(239,68,68,0.15)", color: "#ef4444" },
        TIMEOUT: { bg: "rgba(168,85,247,0.15)", color: "#a855f7" },
        CLOSED: { bg: "rgba(148,163,184,0.15)", color: "#94a3b8" },
    };
    const s = map[status] || { bg: "rgba(148,163,184,0.15)", color: "#94a3b8" };
    return <span style={{ padding: "2px 10px", borderRadius: 6, fontSize: "0.72rem", fontWeight: 700, background: s.bg, color: s.color }}>{status}</span>;
}

/* ─── Summary Stat Card ───────────────────────────── */
function StatCard({ icon, label, value, sub, accent }: { icon: string; label: string; value: string; sub?: string; accent?: string }) {
    return (
        <div className="glass-card" style={{ padding: "14px 18px", flex: "1 1 0", minWidth: 160 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <span style={{ fontSize: "1.1rem" }}>{icon}</span>
                <span style={{ fontSize: "0.72rem", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</span>
            </div>
            <div style={{ fontSize: "1.3rem", fontWeight: 800, color: accent || "var(--text-primary)" }}>{value}</div>
            {sub && <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 2 }}>{sub}</div>}
        </div>
    );
}

/* ─── CSS Donut Chart ─────────────────────────────── */
function DonutChart({ data, total }: { data: { label: string; value: number; color: string }[]; total: number }) {
    let cumPct = 0;
    const segments = data.map(d => {
        const pct = total > 0 ? (d.value / total) * 100 : 0;
        const start = cumPct;
        cumPct += pct;
        return { ...d, pct, start };
    });

    // Build conic-gradient
    const gradParts = segments.map(s => `${s.color} ${s.start}% ${s.start + s.pct}%`).join(", ");
    const gradient = total > 0 ? `conic-gradient(${gradParts})` : "conic-gradient(#334155 0% 100%)";

    return (
        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
            <div style={{ position: "relative", width: 110, height: 110, flexShrink: 0 }}>
                <div style={{
                    width: 110, height: 110, borderRadius: "50%", background: gradient,
                }} />
                <div style={{
                    position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)",
                    width: 70, height: 70, borderRadius: "50%", background: "var(--bg-primary)",
                    display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                }}>
                    <div style={{ fontSize: "1.2rem", fontWeight: 800 }}>{total}</div>
                    <div style={{ fontSize: "0.6rem", color: "var(--text-muted)" }}>Trades</div>
                </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {segments.filter(s => s.value > 0).map(s => (
                    <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.72rem" }}>
                        <span style={{ width: 8, height: 8, borderRadius: "50%", background: s.color, flexShrink: 0 }} />
                        <span style={{ color: "var(--text-muted)" }}>{s.label}</span>
                        <span style={{ fontWeight: 700, marginLeft: "auto" }}>{s.value}</span>
                        <span style={{ color: "var(--text-muted)", fontSize: "0.65rem" }}>{s.pct.toFixed(0)}%</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

/* ─── CSS Bar Chart ───────────────────────────────── */
function BarChart({ data }: { data: { label: string; wins: number; losses: number; color: string }[] }) {
    const maxVal = Math.max(...data.map(d => d.wins + d.losses), 1);
    return (
        <div style={{ display: "flex", gap: 12, alignItems: "flex-end", height: 120 }}>
            {data.map(d => {
                const total = d.wins + d.losses;
                const barH = (total / maxVal) * 100;
                const winH = total > 0 ? (d.wins / total) * barH : 0;
                const lossH = barH - winH;
                return (
                    <div key={d.label} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                        <div style={{ fontSize: "0.7rem", fontWeight: 700 }}>{total}</div>
                        <div style={{ width: "100%", maxWidth: 40, display: "flex", flexDirection: "column", borderRadius: 4, overflow: "hidden" }}>
                            <div style={{ height: lossH, background: "rgba(239,68,68,0.6)", transition: "height 0.3s" }} />
                            <div style={{ height: winH, background: d.color, transition: "height 0.3s" }} />
                        </div>
                        <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontWeight: 600 }}>{d.label}</div>
                        <div style={{ fontSize: "0.6rem", color: "var(--text-muted)" }}>
                            {d.wins}-{d.losses}
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

/* ─── P&L Sparkline (SVG) ─────────────────────────── */
function PnLChart({ trades }: { trades: Trade[] }) {
    const sorted = [...trades]
        .filter(t => CLOSED_STATUSES.has(t.status) && t.exit_date)
        .sort((a, b) => (a.exit_date || "").localeCompare(b.exit_date || ""));

    if (sorted.length < 2) return <div style={{ padding: 20, textAlign: "center", fontSize: "0.75rem", color: "var(--text-muted)" }}>Yeterli veri yok</div>;

    let cum = 0;
    const points = sorted.map((t, i) => {
        cum += t.realized_pnl || 0;
        return { x: i, y: cum, date: t.exit_date || "" };
    });

    const W = 360, H = 130, PAD = 20;
    const minY = Math.min(0, ...points.map(p => p.y));
    const maxY = Math.max(0, ...points.map(p => p.y));
    const rangeY = maxY - minY || 1;
    const scaleX = (i: number) => PAD + (i / (points.length - 1)) * (W - PAD * 2);
    const scaleY = (v: number) => H - PAD - ((v - minY) / rangeY) * (H - PAD * 2);

    const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"} ${scaleX(p.x)} ${scaleY(p.y)}`).join(" ");
    const zeroY = scaleY(0);
    const lastPoint = points[points.length - 1];
    const isPos = lastPoint.y >= 0;

    // Gradient area
    const areaD = pathD + ` L ${scaleX(points.length - 1)} ${H - PAD} L ${PAD} ${H - PAD} Z`;

    return (
        <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} style={{ overflow: "visible" }}>
            <defs>
                <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={isPos ? "#22c55e" : "#ef4444"} stopOpacity={0.3} />
                    <stop offset="100%" stopColor={isPos ? "#22c55e" : "#ef4444"} stopOpacity={0} />
                </linearGradient>
            </defs>
            {/* Zero line */}
            <line x1={PAD} y1={zeroY} x2={W - PAD} y2={zeroY} stroke="#475569" strokeWidth={0.5} strokeDasharray="3 3" />
            {/* Area fill */}
            <path d={areaD} fill="url(#pnlGrad)" />
            {/* Line */}
            <path d={pathD} fill="none" stroke={isPos ? "#22c55e" : "#ef4444"} strokeWidth={2} strokeLinecap="round" />
            {/* Last point */}
            <circle cx={scaleX(lastPoint.x)} cy={scaleY(lastPoint.y)} r={4} fill={isPos ? "#22c55e" : "#ef4444"} />
            {/* Last value label */}
            <text x={scaleX(lastPoint.x) + 8} y={scaleY(lastPoint.y) + 4} fill={isPos ? "#22c55e" : "#ef4444"} fontSize="11" fontWeight="700">
                {isPos ? "+" : ""}${lastPoint.y.toFixed(2)}
            </text>
            {/* X-axis date labels */}
            {[0, Math.floor(points.length / 2), points.length - 1].map(idx => (
                <text key={idx} x={scaleX(idx)} y={H - 2} fill="#64748b" fontSize="8" textAnchor="middle">
                    {points[idx].date.slice(5, 10)}
                </text>
            ))}
        </svg>
    );
}

/* ─── PAGINATION ──────────────────────────────────── */
const PAGE_SIZE = 10;

/* ─── MAIN PAGE ───────────────────────────────────── */
export default function TradesPage() {
    const [trades, setTrades] = useState<Trade[]>([]);
    const [filter, setFilter] = useState<FilterStatus>("ALL");
    const [loading, setLoading] = useState(true);
    const [updatingPrices, setUpdatingPrices] = useState(false);
    const [search, setSearch] = useState("");
    const [page, setPage] = useState(1);
    const [typeFilter, setTypeFilter] = useState<string>("All Types");

    // Close modal
    const [closingId, setClosingId] = useState<number | null>(null);
    const [closeModal, setCloseModal] = useState<Trade | null>(null);
    const [exitPrice, setExitPrice] = useState("");
    const [exitNotes, setExitNotes] = useState("");

    // Edit modal
    const [editModal, setEditModal] = useState<Trade | null>(null);
    const [editStop, setEditStop] = useState("");
    const [editTarget, setEditTarget] = useState("");
    const [editHoldDays, setEditHoldDays] = useState("");
    const [editNotes, setEditNotes] = useState("");
    const [editSaving, setEditSaving] = useState(false);

    const [msg, setMsg] = useState("");

    const [lastUpdate, setLastUpdate] = useState<string | null>(null);

    const fetchLastUpdate = async (): Promise<string | null> => {
        try {
            const d = await getTradesLastUpdate();
            const ts = d.last_update || null;
            setLastUpdate(ts);
            return ts;
        } catch {
            return null;
        }
    };

    const load = () => {
        setLoading(true);
        getTrades()
            .then(async d => {
                setTrades(d.trades || []);
                await fetchLastUpdate();
            })
            .finally(() => setLoading(false));
    };

    useEffect(() => {
        const init = async () => {
            setLoading(true);
            try {
                const d = await getTrades();
                const tradesData: Trade[] = d.trades || [];
                setTrades(tradesData);

                const last = await fetchLastUpdate();
                const hasOpen = tradesData.some(t => t.status === "OPEN" || t.status === "PENDING");

                // Auto-update if:
                // 1. There are open/pending trades with missing current_price, OR
                // 2. Last update is missing or older than 15 minutes
                // Detect stale prices: null, 0, or same as entry (fallback value)
                const hasMissingPrices = tradesData.some(
                    t => (t.status === "OPEN") && (
                        t.current_price == null
                        || t.current_price === 0
                        || t.current_price === t.entry_price
                    )
                );
                const shouldAutoUpdate = () => {
                    if (!hasOpen) return false;
                    if (hasMissingPrices) return true; // Always fetch if prices are missing
                    if (!last) return true;
                    const dt = new Date(last);
                    if (isNaN(dt.getTime())) return false;
                    const now = new Date();
                    const diffMinutes = (now.getTime() - dt.getTime()) / (1000 * 60);
                    return diffMinutes > 15; // Reduced from 60 to 15 min
                };

                if (shouldAutoUpdate()) {
                    setUpdatingPrices(true);
                    try {
                        const upRes = await updatePrices();
                        // Merge live prices from update response into trades
                        const updatedMap = new Map<number, Trade>();
                        if (upRes?.trades) {
                            for (const ut of upRes.trades) {
                                if (ut.id) updatedMap.set(ut.id, ut);
                            }
                        }
                        // Re-fetch from DB (which now has persisted values)
                        const d2 = await getTrades();
                        const freshTrades: Trade[] = d2.trades || [];
                        // Overlay any live data from update response onto DB data
                        const merged = freshTrades.map(t => {
                            const live = updatedMap.get(t.id);
                            if (live && t.status === "OPEN") {
                                return {
                                    ...t,
                                    current_price: live.current_price ?? t.current_price,
                                    unrealized_pnl: live.unrealized_pnl ?? t.unrealized_pnl,
                                    unrealized_pnl_pct: live.unrealized_pnl_pct ?? t.unrealized_pnl_pct,
                                };
                            }
                            return t;
                        });
                        setTrades(merged);
                        await fetchLastUpdate();
                    } catch (err) {
                        console.error("Auto price update failed:", err);
                    } finally {
                        setUpdatingPrices(false);
                    }
                }
            } finally {
                setLoading(false);
            }
        };

        void init();
    }, []);

    const handleUpdatePrices = async () => {
        setUpdatingPrices(true);
        try {
            const upRes = await updatePrices();
            setMsg(`✅ Fiyatlar güncellendi! (${upRes?.trades?.length || 0} trade)`);
            // Re-fetch enriched trades
            const d = await getTrades();
            const freshTrades: Trade[] = d.trades || [];
            // Merge update response data as overlay
            const updatedMap = new Map<number, Trade>();
            if (upRes?.trades) {
                for (const ut of upRes.trades) {
                    if (ut.id) updatedMap.set(ut.id, ut);
                }
            }
            const merged = freshTrades.map(t => {
                const live = updatedMap.get(t.id);
                if (live && t.status === "OPEN") {
                    return {
                        ...t,
                        current_price: live.current_price ?? t.current_price,
                        unrealized_pnl: live.unrealized_pnl ?? t.unrealized_pnl,
                        unrealized_pnl_pct: live.unrealized_pnl_pct ?? t.unrealized_pnl_pct,
                    };
                }
                return t;
            });
            setTrades(merged);
            await fetchLastUpdate();
        } catch { setMsg("❌ Güncelleme başarısız"); }
        finally { setUpdatingPrices(false); }
    };

    const handleClose = async () => {
        if (!closeModal || !exitPrice) return;
        setClosingId(closeModal.id);
        try {
            await closeTrade(closeModal.id, parseFloat(exitPrice), exitNotes);
            setMsg(`✅ ${closeModal.ticker} kapatıldı!`);
            setCloseModal(null);
            setExitPrice(""); setExitNotes("");
            load();
        } catch { setMsg("❌ Kapatma hatası"); }
        finally { setClosingId(null); }
    };

    const handleDelete = async (id: number, ticker: string) => {
        if (!confirm(`${ticker} silinsin mi?`)) return;
        try {
            await deleteTrade(id);
            setMsg(`🗑️ ${ticker} silindi.`);
            load();
        } catch { setMsg("❌ Silme hatası"); }
    };

    const openEditModal = (t: Trade) => {
        setEditModal(t);
        setEditStop(String(t.stop_loss || ""));
        setEditTarget(String(t.target || ""));
        setEditHoldDays(String(t.max_hold_days || 7));
        setEditNotes("");
    };

    const handleEdit = async () => {
        if (!editModal) return;
        setEditSaving(true);
        const updates: Record<string, unknown> = {};
        if (editStop) updates.stop_loss = parseFloat(editStop);
        if (editTarget) updates.target = parseFloat(editTarget);
        if (editHoldDays) updates.max_hold_days = parseInt(editHoldDays);
        if (editNotes) updates.notes = editNotes;
        try {
            await updateTrade(editModal.id, updates);
            setMsg(`✅ ${editModal.ticker} güncellendi!`);
            setEditModal(null);
            load();
        } catch { setMsg("❌ Güncelleme hatası"); }
        finally { setEditSaving(false); }
    };

    /* ─── Computed stats ─────────────────────────────── */
    const stats = useMemo(() => {
        const closed = trades.filter(t => CLOSED_STATUSES.has(t.status));
        const open = trades.filter(t => t.status === "OPEN");
        const pending = trades.filter(t => t.status === "PENDING");
        const wins = closed.filter(t => (t.realized_pnl || 0) > 0);
        const losses = closed.filter(t => (t.realized_pnl || 0) < 0);
        const totalPnl = closed.reduce((s, t) => s + (t.realized_pnl || 0), 0);
        const winRate = closed.length > 0 ? (wins.length / closed.length) * 100 : 0;
        const avgPnl = closed.length > 0 ? totalPnl / closed.length : 0;

        // By status for donut
        const statusCounts: Record<string, number> = {};
        closed.forEach(t => { statusCounts[t.status] = (statusCounts[t.status] || 0) + 1; });

        // By type for bar chart
        const typeCounts: Record<string, { wins: number; losses: number }> = {};
        closed.forEach(t => {
            const tp = t.swing_type || "?";
            if (!typeCounts[tp]) typeCounts[tp] = { wins: 0, losses: 0 };
            if ((t.realized_pnl || 0) > 0) typeCounts[tp].wins++;
            else typeCounts[tp].losses++;
        });

        return { closed, open, pending, wins, losses, totalPnl, winRate, avgPnl, statusCounts, typeCounts };
    }, [trades]);

    /* ─── Filtered + paginated ───────────────────────── */
    const filtered = useMemo(() => {
        let result = trades;
        if (filter === "OPEN") result = result.filter(t => t.status === "OPEN");
        else if (filter === "PENDING") result = result.filter(t => t.status === "PENDING");
        else if (filter === "CLOSED") result = result.filter(t => CLOSED_STATUSES.has(t.status));

        if (typeFilter !== "All Types") result = result.filter(t => t.swing_type === typeFilter);
        if (search) result = result.filter(t => t.ticker?.toLowerCase().includes(search.toLowerCase()));
        return result;
    }, [trades, filter, typeFilter, search]);

    const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
    const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

    // Reset page when filter changes
    useEffect(() => { setPage(1); }, [filter, typeFilter, search]);

    // Donut data
    const donutData = [
        { label: "Stopped", value: stats.statusCounts["STOPPED"] || 0, color: "#ef4444" },
        { label: "Manual", value: (stats.statusCounts["MANUAL"] || 0) + (stats.statusCounts["I-MANUAL"] || 0), color: "#3b82f6" },
        { label: "Timeout", value: stats.statusCounts["TIMEOUT"] || 0, color: "#a855f7" },
        { label: "Target", value: stats.statusCounts["TARGET"] || 0, color: "#22c55e" },
        { label: "Trailed", value: stats.statusCounts["TRAILED"] || 0, color: "#f59e0b" },
        { label: "Rejected", value: stats.statusCounts["REJECTED"] || 0, color: "#64748b" },
    ].filter(d => d.value > 0);

    // Bar data
    const typeColors: Record<string, string> = { A: "#22c55e", B: "#3b82f6", C: "#f59e0b", S: "#a855f7" };
    const barData = Object.entries(stats.typeCounts)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([tp, v]) => ({ label: tp, wins: v.wins, losses: v.losses, color: typeColors[tp] || "#94a3b8" }));

    const formattedLastUpdate = useMemo(() => {
        if (!lastUpdate) return null;
        const d = new Date(lastUpdate);
        if (isNaN(d.getTime())) return { date: lastUpdate, time: "" };
        const date = d.toLocaleDateString();
        const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        return { date, time };
    }, [lastUpdate]);

    if (loading) return <div style={{ padding: 80, textAlign: "center" }}><span className="spinner" /></div>;

    return (
        <div>
            {/* ─── Header ──────────────────────────────────── */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20 }}>
                <div>
                    <h1 className="page-title gradient-text">Paper Trades</h1>
                    <p className="page-subtitle">{trades.length} total · {stats.open.length} open · {stats.closed.length} closed · {stats.pending.length} pending</p>
                </div>
                <button className="btn-secondary" onClick={handleUpdatePrices} disabled={updatingPrices}
                    style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    {updatingPrices ? <span className="spinner" style={{ width: 14, height: 14 }} /> : <RefreshCw size={14} />}
                    Update Prices
                </button>
            </div>

            {msg && (
                <div style={{ background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.3)", borderRadius: 10, padding: "10px 16px", marginBottom: 16, fontSize: "0.85rem", color: "var(--accent)" }}>
                    {msg}
                    <button onClick={() => setMsg("")} style={{ float: "right", background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)" }}>✕</button>
                </div>
            )}

            {/* ─── Summary Cards ───────────────────────────── */}
            <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
                <StatCard icon="📊" label="Total Trades" value={String(stats.closed.length)}
                    sub={`${stats.open.length} open · ${stats.pending.length} pending`} />
                <StatCard icon="💰" label="Total P&L" value={`${stats.totalPnl >= 0 ? "+" : ""}$${stats.totalPnl.toFixed(2)}`}
                    accent={stats.totalPnl >= 0 ? "#22c55e" : "#ef4444"} />
                <StatCard icon="🎯" label="Win Rate" value={`${stats.winRate.toFixed(1)}%`}
                    sub={`${stats.wins.length}W / ${stats.losses.length}L`}
                    accent={stats.winRate >= 50 ? "#22c55e" : stats.winRate >= 40 ? "#f59e0b" : "#ef4444"} />
                <StatCard icon="📈" label="Avg P&L" value={`${stats.avgPnl >= 0 ? "+" : ""}$${stats.avgPnl.toFixed(2)}`}
                    accent={stats.avgPnl >= 0 ? "#22c55e" : "#ef4444"} />
                {formattedLastUpdate && (
                    <StatCard
                        icon="⏱️"
                        label="Last Price Update"
                        value={formattedLastUpdate.date}
                        sub={formattedLastUpdate.time ? `at ${formattedLastUpdate.time}` : undefined}
                    />
                )}
            </div>

            {/* ─── Charts Row ──────────────────────────────── */}
            {stats.closed.length > 0 && (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 14, marginBottom: 20 }}>
                    {/* Donut */}
                    <div className="glass-card" style={{ padding: 18 }}>
                        <div style={{ fontSize: "0.82rem", fontWeight: 700, marginBottom: 14, color: "var(--text-secondary)" }}>Trade Outcomes</div>
                        <DonutChart data={donutData} total={stats.closed.length} />
                    </div>

                    {/* Bar Chart */}
                    {barData.length > 0 && (
                        <div className="glass-card" style={{ padding: 18 }}>
                            <div style={{ fontSize: "0.82rem", fontWeight: 700, marginBottom: 14, color: "var(--text-secondary)" }}>Trades by Type</div>
                            <BarChart data={barData} />
                        </div>
                    )}

                    {/* P&L Over Time */}
                    <div className="glass-card" style={{ padding: 18 }}>
                        <div style={{ fontSize: "0.82rem", fontWeight: 700, marginBottom: 14, color: "var(--text-secondary)" }}>Profit & Loss Over Time</div>
                        <PnLChart trades={trades} />
                    </div>
                </div>
            )}

            {/* ─── Filter + Search ─────────────────────────── */}
            <div style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap", alignItems: "center" }}>
                {(["ALL", "OPEN", "PENDING", "CLOSED"] as FilterStatus[]).map(s => {
                    const count = s === "ALL" ? trades.length
                        : s === "CLOSED" ? stats.closed.length
                            : trades.filter(t => t.status === s).length;
                    return (
                        <button key={s} onClick={() => setFilter(s)}
                            style={{
                                padding: "6px 14px", borderRadius: 8, border: "1px solid",
                                borderColor: filter === s ? "var(--accent)" : "var(--border)",
                                background: filter === s ? "rgba(59,130,246,0.15)" : "transparent",
                                color: filter === s ? "var(--accent)" : "var(--text-secondary)",
                                cursor: "pointer", fontSize: "0.78rem", fontWeight: 600,
                            }}>
                            {s} <span style={{ opacity: 0.6 }}>({count})</span>
                        </button>
                    );
                })}

                <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
                    <select className="input" value={typeFilter} onChange={e => setTypeFilter(e.target.value)}
                        style={{ padding: "5px 10px", fontSize: "0.78rem", width: 110 }}>
                        <option>All Types</option>
                        {["A", "B", "C", "S"].map(t => <option key={t} value={t}>{t}</option>)}
                    </select>
                    <div style={{ position: "relative" }}>
                        <Search size={13} style={{ position: "absolute", left: 8, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)" }} />
                        <input className="input" placeholder="Search..."
                            value={search} onChange={e => setSearch(e.target.value)}
                            style={{ paddingLeft: 28, width: 140, fontSize: "0.78rem", padding: "5px 10px 5px 28px" }} />
                    </div>
                </div>
            </div>

            {/* ─── Trade Table ─────────────────────────────── */}
            <div className="glass-card" style={{ overflow: "hidden" }}>
                {filtered.length === 0 ? (
                    <div style={{ padding: 48, textAlign: "center", color: "var(--text-muted)" }}>Bu filtre için trade bulunamadı.</div>
                ) : (
                    <>
                        <div style={{ overflowX: "auto" }}>
                            <table className="data-table">
                                <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>Ticker</th>
                                        <th>Type</th>
                                        <th>Entry Date</th>
                                        <th>Close Date</th>
                                        <th>Entry $</th>
                                        <th>Current $</th>
                                        <th>Exit $</th>
                                        <th>P&L $</th>
                                        <th>P&L %</th>
                                        <th>Result</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {paginated.map((t, idx) => {
                                        const isOpen = t.status === "OPEN";
                                        const isPending = t.status === "PENDING";
                                        const isClosed = CLOSED_STATUSES.has(t.status);
                                        const currentPrice = t.current_price ?? null;
                                        const exitPrice = isClosed ? t.exit_price : null;
                                        // Compute P&L: use realized for closed, unrealized for open
                                        // If unrealized is null but we have current_price, compute client-side
                                        let pnl = t.realized_pnl ?? t.unrealized_pnl ?? null;
                                        let pnlPct = t.realized_pnl_pct ?? t.unrealized_pnl_pct ?? null;
                                        if (pnl == null && isOpen && currentPrice && t.entry_price) {
                                            const size = t.position_size || 100;
                                            pnl = (currentPrice - t.entry_price) * size;
                                            pnlPct = ((currentPrice / t.entry_price) - 1) * 100;
                                        }
                                        // PENDING trades: no position yet, show 0
                                        if (isPending) { pnl = 0; pnlPct = 0; }
                                        const isPos = (pnl || 0) >= 0;
                                        const rowNum = (page - 1) * PAGE_SIZE + idx + 1;

                                        return (
                                            <tr
                                                key={t.id}
                                                style={{
                                                    borderLeft: `3px solid ${isPos
                                                        ? "rgba(34,197,94,0.5)"
                                                        : (pnl || 0) < 0
                                                            ? "rgba(239,68,68,0.5)"
                                                            : "transparent"
                                                        }`,
                                                }}
                                            >
                                                <td style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>{rowNum}</td>
                                                <td>
                                                    <strong style={{ color: "var(--accent)" }}>{t.ticker}</strong>
                                                    {t.partial_exit_price != null && t.partial_exit_price > 0 && (
                                                        <div style={{ fontSize: "0.62rem", color: "var(--green)", fontWeight: 600, marginTop: 1 }}>
                                                            T1 @${t.partial_exit_price.toFixed(2)}
                                                        </div>
                                                    )}
                                                </td>
                                                <td><TypeBadge type={t.swing_type} /></td>
                                                <td style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>{t.entry_date || "—"}</td>
                                                <td style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>{t.exit_date || "—"}</td>
                                                <td style={{ fontWeight: 600 }}>${t.entry_price?.toFixed(2)}</td>
                                                <td>
                                                    {isOpen && currentPrice != null && currentPrice > 0 ? (
                                                        <span
                                                            style={{
                                                                fontWeight: 700,
                                                                color: currentPrice >= (t.entry_price || 0) ? "#22c55e" : "#ef4444",
                                                            }}
                                                        >
                                                            ${currentPrice.toFixed(2)}
                                                        </span>
                                                    ) : isOpen && updatingPrices ? (
                                                        <span style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>...</span>
                                                    ) : isOpen ? (
                                                        <span style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>
                                                            ${(t.entry_price || 0).toFixed(2)}
                                                        </span>
                                                    ) : isPending ? (
                                                        <span style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>
                                                            ${(t.signal_price || t.entry_price || 0).toFixed(2)}
                                                        </span>
                                                    ) : isClosed && exitPrice != null && exitPrice > 0 ? (
                                                        <span style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>
                                                            ${exitPrice.toFixed(2)}
                                                        </span>
                                                    ) : (
                                                        <span style={{ color: "var(--text-muted)" }}>—</span>
                                                    )}
                                                </td>
                                                <td>
                                                    {exitPrice != null ? (
                                                        <span
                                                            style={{
                                                                fontWeight: 700,
                                                                color: exitPrice >= (t.entry_price || 0) ? "#22c55e" : "#ef4444",
                                                            }}
                                                        >
                                                            ${exitPrice.toFixed(2)}
                                                        </span>
                                                    ) : (
                                                        <span style={{ color: "var(--text-muted)" }}>—</span>
                                                    )}
                                                </td>
                                                <td
                                                    style={{
                                                        color: isPos ? "#22c55e" : "#ef4444",
                                                        fontWeight: 700,
                                                        fontSize: "0.85rem",
                                                    }}
                                                >
                                                    {pnl != null ? `${isPos ? "+$" : "-$"}${Math.abs(pnl).toFixed(2)}` : "—"}
                                                </td>
                                                <td style={{ color: isPos ? "#22c55e" : "#ef4444", fontWeight: 600 }}>
                                                    {pnlPct != null ? `${isPos ? "+" : ""}${pnlPct.toFixed(2)}%` : "—"}
                                                </td>
                                                <td><StatusBadge status={t.status} /></td>
                                                <td>
                                                    <div style={{ display: "flex", gap: 4 }}>
                                                        {isOpen && (
                                                            <>
                                                                <button
                                                                    className="btn-secondary"
                                                                    style={{ padding: "3px 7px", fontSize: "0.7rem" }}
                                                                    title="Stop/Target düzenle"
                                                                    onClick={() => openEditModal(t)}
                                                                >
                                                                    <Edit2 size={10} />
                                                                </button>
                                                                <button
                                                                    className="btn-secondary"
                                                                    style={{ padding: "3px 8px", fontSize: "0.7rem" }}
                                                                    onClick={() => {
                                                                        setCloseModal(t);
                                                                        setExitPrice(String(t.current_price || t.entry_price));
                                                                    }}
                                                                >
                                                                    <CheckSquare size={10} />
                                                                </button>
                                                            </>
                                                        )}
                                                        <button
                                                            className="btn-danger"
                                                            style={{ padding: "3px 7px" }}
                                                            onClick={() => handleDelete(t.id, t.ticker)}
                                                        >
                                                            <X size={10} />
                                                        </button>
                                                    </div>
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>

                        {/* Pagination */}
                        {totalPages > 1 && (
                            <div style={{ display: "flex", justifyContent: "flex-end", alignItems: "center", gap: 6, padding: "12px 18px", borderTop: "1px solid var(--border)" }}>
                                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginRight: 8 }}>
                                    Showing {filtered.length} trades
                                </span>
                                <button className="btn-secondary" style={{ padding: "4px 8px" }} onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}>
                                    <ChevronLeft size={14} />
                                </button>
                                {Array.from({ length: totalPages }, (_, i) => i + 1).map(p => (
                                    <button key={p} onClick={() => setPage(p)}
                                        style={{
                                            padding: "4px 10px", borderRadius: 6, fontSize: "0.75rem", fontWeight: 600,
                                            border: "1px solid", cursor: "pointer",
                                            borderColor: page === p ? "var(--accent)" : "var(--border)",
                                            background: page === p ? "rgba(59,130,246,0.15)" : "transparent",
                                            color: page === p ? "var(--accent)" : "var(--text-secondary)",
                                        }}>{p}</button>
                                ))}
                                <button className="btn-secondary" style={{ padding: "4px 8px" }} onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}>
                                    <ChevronRight size={14} />
                                </button>
                            </div>
                        )}
                    </>
                )}
            </div>

            {/* ── Close Trade Modal ── */}
            {closeModal && (
                <div style={{
                    position: "fixed", inset: 0, background: "rgba(0,0,0,0.7)", display: "flex",
                    alignItems: "center", justifyContent: "center", zIndex: 100, backdropFilter: "blur(4px)",
                }}>
                    <div className="glass-card" style={{ padding: 28, width: 360, maxWidth: "90vw" }}>
                        <h3 style={{ margin: "0 0 16px", fontSize: "1.05rem" }}>Close {closeModal.ticker}</h3>
                        <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
                            <div style={{ flex: 1, padding: "10px 12px", background: "rgba(239,68,68,0.1)", borderRadius: 8, fontSize: "0.8rem" }}>
                                <div style={{ color: "var(--text-muted)", fontSize: "0.7rem" }}>STOP LOSS</div>
                                <div style={{ color: "var(--red)", fontWeight: 700 }}>${closeModal.stop_loss?.toFixed(2)}</div>
                            </div>
                            <div style={{ flex: 1, padding: "10px 12px", background: "rgba(34,197,94,0.1)", borderRadius: 8, fontSize: "0.8rem" }}>
                                <div style={{ color: "var(--text-muted)", fontSize: "0.7rem" }}>TARGET</div>
                                <div style={{ color: "var(--green)", fontWeight: 700 }}>${closeModal.target?.toFixed(2)}</div>
                            </div>
                        </div>
                        <label style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontWeight: 600 }}>EXIT PRICE</label>
                        <input className="input" type="number" step="0.01" value={exitPrice}
                            onChange={e => setExitPrice(e.target.value)} style={{ margin: "6px 0 12px" }} />
                        <label style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontWeight: 600 }}>NOTES (optional)</label>
                        <input className="input" value={exitNotes} onChange={e => setExitNotes(e.target.value)} style={{ margin: "6px 0 20px" }} placeholder="e.g. hit target" />
                        <div style={{ display: "flex", gap: 10 }}>
                            <button className="btn-primary" onClick={handleClose} disabled={closingId === closeModal.id} style={{ flex: 1 }}>
                                {closingId === closeModal.id ? <span className="spinner" style={{ width: 14, height: 14 }} /> : null}
                                Confirm Close
                            </button>
                            <button className="btn-secondary" onClick={() => setCloseModal(null)}>Cancel</button>
                        </div>
                    </div>
                </div>
            )}

            {/* ── Edit Stop / Target Modal ── */}
            {editModal && (
                <div style={{
                    position: "fixed", inset: 0, background: "rgba(0,0,0,0.75)", display: "flex",
                    alignItems: "center", justifyContent: "center", zIndex: 100, backdropFilter: "blur(4px)",
                }}>
                    <div className="glass-card" style={{ padding: 28, width: 380, maxWidth: "90vw" }}>
                        <h3 style={{ margin: "0 0 6px", fontSize: "1.05rem" }}>
                            ✏️ {editModal.ticker} — Fiyat Güncelle
                        </h3>
                        <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: 18 }}>
                            Giriş: ${editModal.entry_price?.toFixed(2)} &nbsp;·&nbsp; Kalite: {editModal.quality_score?.toFixed(0)}
                        </div>

                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 16 }}>
                            <div>
                                <label style={{ fontSize: "0.75rem", color: "var(--red)", fontWeight: 700, display: "block", marginBottom: 5 }}>
                                    🛑 YENİ STOP LOSS
                                </label>
                                <input className="input" type="number" step="0.01" value={editStop}
                                    onChange={e => setEditStop(e.target.value)} style={{ borderColor: "rgba(239,68,68,0.4)" }} />
                                {editModal.stop_loss && (
                                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: 3 }}>Mevcut: ${editModal.stop_loss.toFixed(2)}</div>
                                )}
                            </div>
                            <div>
                                <label style={{ fontSize: "0.75rem", color: "var(--green)", fontWeight: 700, display: "block", marginBottom: 5 }}>
                                    🎯 YENİ HEDEF
                                </label>
                                <input className="input" type="number" step="0.01" value={editTarget}
                                    onChange={e => setEditTarget(e.target.value)} style={{ borderColor: "rgba(34,197,94,0.4)" }} />
                                {editModal.target && (
                                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: 3 }}>Mevcut: ${editModal.target.toFixed(2)}</div>
                                )}
                            </div>
                        </div>

                        <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 5 }}>
                            ⏱️ MAKS HOLD SÜRESİ (gün)
                        </label>
                        <input className="input" type="number" min="1" max="60" value={editHoldDays}
                            onChange={e => setEditHoldDays(e.target.value)} style={{ marginBottom: 16 }} />
                        {editModal.max_hold_days && (
                            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: -12, marginBottom: 12 }}>
                                Mevcut: {editModal.max_hold_days} gün
                            </div>
                        )}

                        <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 5 }}>
                            NOT (opsiyonel)
                        </label>
                        <input className="input" value={editNotes} onChange={e => setEditNotes(e.target.value)}
                            style={{ marginBottom: 20 }} placeholder="Neden güncelliyorsunuz?" />

                        <div style={{ display: "flex", gap: 10 }}>
                            <button className="btn-primary" onClick={handleEdit} disabled={editSaving} style={{ flex: 1 }}>
                                {editSaving ? <span className="spinner" style={{ width: 14, height: 14 }} /> : null}
                                Kaydet
                            </button>
                            <button className="btn-secondary" onClick={() => setEditModal(null)}>İptal</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
