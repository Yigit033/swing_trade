"use client";
import { useState, useEffect } from "react";
import { getTrades, closeTrade, deleteTrade, updateTrade, updatePrices } from "@/lib/api";
import type { Trade } from "@/lib/api";
import { RefreshCw, X, CheckSquare, TrendingUp, TrendingDown, Edit2 } from "lucide-react";

type FilterStatus = "ALL" | "OPEN" | "CLOSED" | "PENDING";

// Matches storage.get_closed_trades(): status NOT IN ('OPEN','PENDING')
const CLOSED_STATUSES = new Set(["STOPPED", "TRAILED", "TARGET", "MANUAL", "WIN", "LOSS", "CLOSED", "REJECTED"]);

function StatusBadge({ status }: { status: string }) {
    const map: Record<string, string> = {
        OPEN: "badge-blue",
        PENDING: "badge-yellow",
        TARGET: "badge-green",
        TRAILED: "badge-yellow",
        STOPPED: "badge-red",
        MANUAL: "badge-blue",
        REJECTED: "badge-red",
        WIN: "badge-green",
        LOSS: "badge-red",
        CLOSED: "badge-purple",
    };
    return <span className={`badge ${map[status] || "badge-purple"}`}>{status}</span>;
}

export default function TradesPage() {
    const [trades, setTrades] = useState<Trade[]>([]);
    const [filter, setFilter] = useState<FilterStatus>("ALL");
    const [loading, setLoading] = useState(true);
    const [updatingPrices, setUpdatingPrices] = useState(false);

    // Close modal
    const [closingId, setClosingId] = useState<number | null>(null);
    const [closeModal, setCloseModal] = useState<Trade | null>(null);
    const [exitPrice, setExitPrice] = useState("");
    const [exitNotes, setExitNotes] = useState("");

    // Edit stop/target modal
    const [editModal, setEditModal] = useState<Trade | null>(null);
    const [editStop, setEditStop] = useState("");
    const [editTarget, setEditTarget] = useState("");
    const [editHoldDays, setEditHoldDays] = useState("");
    const [editNotes, setEditNotes] = useState("");
    const [editSaving, setEditSaving] = useState(false);

    const [msg, setMsg] = useState("");

    const load = () => {
        setLoading(true);
        getTrades().then(d => setTrades(d.trades || [])).finally(() => setLoading(false));
    };

    useEffect(() => { load(); }, []);

    const handleUpdatePrices = async () => {
        setUpdatingPrices(true);
        try {
            await updatePrices();
            setMsg("Prices updated!");
            load();
        } catch { setMsg("Update failed"); }
        finally { setUpdatingPrices(false); }
    };

    const handleClose = async () => {
        if (!closeModal || !exitPrice) return;
        setClosingId(closeModal.id);
        try {
            await closeTrade(closeModal.id, parseFloat(exitPrice), exitNotes);
            setMsg(`✅ ${closeModal.ticker} closed!`);
            setCloseModal(null);
            setExitPrice(""); setExitNotes("");
            load();
        } catch { setMsg("Close failed"); }
        finally { setClosingId(null); }
    };

    const handleDelete = async (id: number, ticker: string) => {
        if (!confirm(`Delete ${ticker}?`)) return;
        try {
            await deleteTrade(id);
            setMsg(`Deleted ${ticker}`);
            load();
        } catch { setMsg("Delete failed"); }
    };

    const openEditModal = (t: Trade) => {
        setEditModal(t);
        setEditStop(String(t.stop_loss ?? ""));
        setEditTarget(String(t.target ?? ""));
        setEditHoldDays(String(t.max_hold_days ?? 7));
        setEditNotes(t.notes ?? "");
    };

    const handleEdit = async () => {
        if (!editModal) return;
        const stop = parseFloat(editStop);
        const target = parseFloat(editTarget);
        const holdDays = parseInt(editHoldDays);
        if (isNaN(stop) || isNaN(target)) { setMsg("Geçersiz stop veya hedef fiyatı"); return; }
        if (isNaN(holdDays) || holdDays < 1) { setMsg("Hold süresi en az 1 gün olmalı"); return; }
        setEditSaving(true);
        try {
            await updateTrade(editModal.id, {
                stop_loss: stop,
                trailing_stop: stop,   // sync trailing_stop too
                target,
                max_hold_days: holdDays,
                notes: editNotes || editModal.notes,
            });
            setMsg(`✅ ${editModal.ticker} güncellendi — Stop: $${stop.toFixed(2)}, Target: $${target.toFixed(2)}, Hold: ${holdDays}g`);
            setEditModal(null);
            load();
        } catch { setMsg("❌ Güncelleme başarısız"); }
        finally { setEditSaving(false); }
    };

    const filtered = filter === "ALL"
        ? trades
        : filter === "CLOSED"
            ? trades.filter(t => CLOSED_STATUSES.has(t.status))
            : trades.filter(t => t.status === filter);
    const openCount = trades.filter(t => t.status === "OPEN").length;
    const closedCount = trades.filter(t => CLOSED_STATUSES.has(t.status)).length;
    const winCount = trades.filter(t => (t.realized_pnl ?? 0) > 0).length;

    return (
        <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 24 }}>
                <div>
                    <h1 className="page-title gradient-text">Paper Trades</h1>
                    <p className="page-subtitle">{trades.length} total · {openCount} open · {closedCount} closed · {winCount} wins</p>
                </div>
                <button className="btn-secondary" onClick={handleUpdatePrices} disabled={updatingPrices}>
                    {updatingPrices ? <span className="spinner" style={{ width: 14, height: 14 }} /> : <RefreshCw size={14} />}
                    Update Prices
                </button>
            </div>

            {msg && (
                <div style={{ background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.3)", borderRadius: 10, padding: "10px 16px", marginBottom: 16, fontSize: "0.875rem", color: "var(--accent)" }}>
                    {msg}
                </div>
            )}

            {/* Filter tabs */}
            <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
                {(["ALL", "OPEN", "PENDING", "CLOSED"] as FilterStatus[]).map(s => {
                    const count = s === "ALL" ? trades.length
                        : s === "CLOSED" ? closedCount
                            : trades.filter(t => t.status === s).length;
                    return (
                        <button key={s} onClick={() => setFilter(s)}
                            style={{
                                padding: "6px 16px", borderRadius: 8, border: "1px solid",
                                borderColor: filter === s ? "var(--accent)" : "var(--border)",
                                background: filter === s ? "rgba(59,130,246,0.15)" : "transparent",
                                color: filter === s ? "var(--accent)" : "var(--text-secondary)",
                                cursor: "pointer", fontSize: "0.8rem", fontWeight: 600,
                                transition: "all 0.15s",
                            }}>
                            {s} {s !== "ALL" && <span style={{ opacity: 0.6 }}>({count})</span>}
                        </button>
                    );
                })}
            </div>

            <div className="glass-card" style={{ overflow: "hidden" }}>
                {loading ? (
                    <div style={{ padding: 48, textAlign: "center" }}><span className="spinner" /></div>
                ) : filtered.length === 0 ? (
                    <div style={{ padding: 48, textAlign: "center", color: "var(--text-muted)" }}>No {filter !== "ALL" ? filter : ""} trades found.</div>
                ) : (
                    <div style={{ overflowX: "auto" }}>
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Ticker</th><th>Status</th><th>Entry</th><th>Stop</th><th>Target</th>
                                    <th>Quality</th><th>P&L</th><th>P&L %</th><th>Date</th><th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {filtered.map(t => {
                                    const pnl = t.realized_pnl ?? t.unrealized_pnl;
                                    const pnlPct = t.realized_pnl_pct ?? t.unrealized_pnl_pct;
                                    const isPos = (pnl || 0) >= 0;
                                    const isOpen = t.status === "OPEN";
                                    return (
                                        <tr key={t.id}>
                                            <td><strong style={{ color: "var(--accent)" }}>{t.ticker}</strong></td>
                                            <td><StatusBadge status={t.status} /></td>
                                            <td>${t.entry_price?.toFixed(2)}</td>
                                            <td style={{ color: "var(--red)" }}>${t.stop_loss?.toFixed(2)}</td>
                                            <td style={{ color: "var(--green)" }}>${t.target?.toFixed(2)}</td>
                                            <td>
                                                <span className={`badge ${t.quality_score >= 80 ? "badge-green" : t.quality_score >= 65 ? "badge-blue" : "badge-yellow"}`}>
                                                    {t.quality_score?.toFixed(0)}
                                                </span>
                                            </td>
                                            <td style={{ color: isPos ? "var(--green)" : "var(--red)", fontWeight: 600 }}>
                                                {pnl != null ? `${isPos ? "+" : ""}$${pnl.toFixed(2)}` : "—"}
                                            </td>
                                            <td style={{ color: isPos ? "var(--green)" : "var(--red)" }}>
                                                {pnlPct != null ? `${isPos ? "+" : ""}${pnlPct.toFixed(2)}%` : "—"}
                                            </td>
                                            <td style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>
                                                {t.exit_date || t.entry_date}
                                            </td>
                                            <td>
                                                <div style={{ display: "flex", gap: 5 }}>
                                                    {isOpen && (
                                                        <>
                                                            {/* Edit stop/target */}
                                                            <button
                                                                className="btn-secondary"
                                                                style={{ padding: "4px 9px", fontSize: "0.75rem" }}
                                                                title="Stop/Target düzenle"
                                                                onClick={() => openEditModal(t)}
                                                            >
                                                                <Edit2 size={11} />
                                                            </button>
                                                            {/* Close trade */}
                                                            <button
                                                                className="btn-secondary"
                                                                style={{ padding: "4px 10px", fontSize: "0.75rem" }}
                                                                onClick={() => { setCloseModal(t); setExitPrice(String(t.entry_price)); }}
                                                            >
                                                                <CheckSquare size={11} /> Close
                                                            </button>
                                                        </>
                                                    )}
                                                    <button className="btn-danger" onClick={() => handleDelete(t.id, t.ticker)}>
                                                        <X size={11} />
                                                    </button>
                                                </div>
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
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
                                <input
                                    className="input"
                                    type="number"
                                    step="0.01"
                                    value={editStop}
                                    onChange={e => setEditStop(e.target.value)}
                                    style={{ borderColor: "rgba(239,68,68,0.4)" }}
                                />
                                {editModal.stop_loss && (
                                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: 3 }}>
                                        Mevcut: ${editModal.stop_loss.toFixed(2)}
                                    </div>
                                )}
                            </div>
                            <div>
                                <label style={{ fontSize: "0.75rem", color: "var(--green)", fontWeight: 700, display: "block", marginBottom: 5 }}>
                                    🎯 YENİ HEDEF
                                </label>
                                <input
                                    className="input"
                                    type="number"
                                    step="0.01"
                                    value={editTarget}
                                    onChange={e => setEditTarget(e.target.value)}
                                    style={{ borderColor: "rgba(34,197,94,0.4)" }}
                                />
                                {editModal.target && (
                                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: 3 }}>
                                        Mevcut: ${editModal.target.toFixed(2)}
                                    </div>
                                )}
                            </div>
                        </div>

                        <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 5 }}>
                            ⏱️ MAKS HOLD SÜRESİ (gün)
                        </label>
                        <input
                            className="input"
                            type="number"
                            min="1"
                            max="60"
                            value={editHoldDays}
                            onChange={e => setEditHoldDays(e.target.value)}
                            style={{ marginBottom: 16 }}
                        />
                        {editModal.max_hold_days && (
                            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: -12, marginBottom: 12 }}>
                                Mevcut: {editModal.max_hold_days} gün
                            </div>
                        )}

                        <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 5 }}>
                            NOT (opsiyonel)
                        </label>
                        <input
                            className="input"
                            value={editNotes}
                            onChange={e => setEditNotes(e.target.value)}
                            style={{ marginBottom: 20 }}
                            placeholder="Neden güncelliyorsunuz?"
                        />

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
