"use client";
import { useState, useEffect } from "react";
import { getPending, checkPending, confirmTrade, getCurrentRegime } from "@/lib/api";
import type { Trade, RegimeData } from "@/lib/api";
import { RefreshCw, CheckCircle, Clock } from "lucide-react";

export default function PendingPage() {
    const [pending, setPending] = useState<Trade[]>([]);
    const [loading, setLoading] = useState(true);
    const [checking, setChecking] = useState(false);
    const [confirming, setConfirming] = useState<number | null>(null);
    const [msg, setMsg] = useState("");
    const [regime, setRegime] = useState<RegimeData | null>(null);

    const load = () => {
        setLoading(true);
        getPending().then(d => setPending(d.pending || [])).finally(() => setLoading(false));
    };

    // Auto-check pending trades on page load (gap filter + confirm/reject)
    useEffect(() => {
        let cancelled = false;
        (async () => {
            setChecking(true);
            try {
                const res = await checkPending();
                if (!cancelled) {
                    const count = res.confirmed?.length || 0;
                    if (count > 0) setMsg(`✅ Auto-checked on load: ${count} trades processed`);
                }
            } catch { /* silent — will still load list below */ }
            finally { if (!cancelled) setChecking(false); }
            // Fetch regime in parallel
            getCurrentRegime().then(r => { if (!cancelled) setRegime(r); }).catch(() => {});
            // Always reload the (now-updated) list
            if (!cancelled) load();
        })();
        return () => { cancelled = true; };
    }, []);

    const handleCheck = async () => {
        setChecking(true);
        try {
            const res = await checkPending();
            setMsg(`✅ Auto-checked: ${res.confirmed?.length || 0} trades confirmed`);
            load();
        } catch { setMsg("Check failed"); }
        finally { setChecking(false); }
    };

    const handleConfirm = async (id: number, ticker: string) => {
        setConfirming(id);
        try {
            await confirmTrade(id);
            setMsg(`✅ ${ticker} manually confirmed and moved to OPEN!`);
            load();
        } catch { setMsg(`Failed to confirm ${ticker}`); }
        finally { setConfirming(null); }
    };

    return (
        <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 24 }}>
                <div>
                    <h1 className="page-title gradient-text">Pending Trades</h1>
                    <p className="page-subtitle">Signals waiting for price confirmation · auto-checked on load</p>
                </div>
                <button className="btn-secondary" onClick={handleCheck} disabled={checking}>
                    {checking ? <span className="spinner" style={{ width: 14, height: 14 }} /> : <RefreshCw size={14} />}
                    Auto-Check All
                </button>
            </div>

            {msg && (
                <div style={{ background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.25)", borderRadius: 10, padding: "10px 16px", marginBottom: 16, fontSize: "0.875rem", color: "var(--accent)" }}>
                    {msg}
                </div>
            )}

            {/* Market Regime Context */}
            {regime && regime.regime !== "BULL" && (
                <div style={{
                    background: regime.regime === "BEAR" ? "rgba(239,68,68,0.06)" : "rgba(245,158,11,0.06)",
                    border: `1px solid ${regime.regime === "BEAR" ? "rgba(239,68,68,0.2)" : "rgba(245,158,11,0.2)"}`,
                    borderRadius: 10, padding: "10px 16px", marginBottom: 16,
                    fontSize: "0.8rem", display: "flex", alignItems: "center", gap: 10,
                    color: regime.regime === "BEAR" ? "var(--red)" : "var(--yellow)",
                }}>
                    <span style={{ fontWeight: 700 }}>
                        {regime.regime === "BEAR" ? "BEAR" : "CAUTION"}
                    </span>
                    {regime.confidence === "TENTATIVE" && (
                        <span style={{ fontSize: "0.68rem", opacity: 0.7 }}>(Unconfirmed)</span>
                    )}
                    <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                        x{regime.score_multiplier} multiplier
                        {regime.vix != null && regime.vix > 0 && ` · VIX ${regime.vix.toFixed(1)}`}
                    </span>
                    <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginLeft: "auto" }}>
                        {regime.regime === "BEAR"
                            ? "Pending trades risk-adjusted. Confirm with caution."
                            : "Mixed signals. Verify before confirming."}
                    </span>
                </div>
            )}

            {/* Info note */}
            <div style={{ background: "rgba(245,158,11,0.05)", border: "1px solid rgba(245,158,11,0.2)", borderRadius: 10, padding: "12px 16px", marginBottom: 20, fontSize: "0.8rem", color: "var(--yellow)", display: "flex", gap: 8, alignItems: "flex-start" }}>
                <Clock size={14} style={{ marginTop: 1, flexShrink: 0 }} />
                <span>Pending trades are signals from the scanner waiting for entry confirmation.
                    Click <strong>Auto-Check All</strong> to verify current prices, or <strong>Confirm</strong> to manually move a trade to OPEN status.</span>
            </div>

            <div className="glass-card" style={{ overflow: "hidden" }}>
                {loading ? (
                    <div style={{ padding: 48, textAlign: "center" }}><span className="spinner" /></div>
                ) : pending.length === 0 ? (
                    <div style={{ padding: 60, textAlign: "center" }}>
                        <CheckCircle size={48} style={{ color: "var(--green)", marginBottom: 16, opacity: 0.6 }} />
                        <div style={{ color: "var(--text-secondary)", fontWeight: 600, marginBottom: 8 }}>No pending trades</div>
                        <div style={{ color: "var(--text-muted)", fontSize: "0.875rem" }}>
                            Run the Scanner and click "Track" to add signals here.
                        </div>
                    </div>
                ) : (
                    <div style={{ overflowX: "auto" }}>
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Ticker</th><th>Entry</th><th>Stop Loss</th><th>Target</th>
                                    <th>Quality</th><th>Type</th><th>Added</th><th>Action</th>
                                </tr>
                            </thead>
                            <tbody>
                                {pending.map(t => (
                                    <tr key={t.id}>
                                        <td>
                                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--yellow)", animation: "pulse 2s infinite" }} />
                                                <strong style={{ color: "var(--accent)" }}>{t.ticker}</strong>
                                            </div>
                                        </td>
                                        <td style={{ fontWeight: 600 }}>${t.entry_price?.toFixed(2)}</td>
                                        <td style={{ color: "var(--red)" }}>${t.stop_loss?.toFixed(2)}</td>
                                        <td style={{ color: "var(--green)" }}>${t.target?.toFixed(2)}</td>
                                        <td>
                                            <span className={`badge ${t.quality_score >= 80 ? "badge-green" : t.quality_score >= 65 ? "badge-blue" : "badge-yellow"}`}>
                                                {t.quality_score >= 80 ? "🔥" : t.quality_score >= 65 ? "✅" : "⚠️"} {t.quality_score?.toFixed(0)}
                                            </span>
                                        </td>
                                        <td>
                                            <span className={`badge ${t.swing_type === "B" ? "badge-purple" : "badge-blue"}`}>
                                                Type {t.swing_type || "A"}
                                            </span>
                                        </td>
                                        <td style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>{t.entry_date}</td>
                                        <td>
                                            <button className="btn-secondary" style={{ fontSize: "0.78rem", padding: "5px 12px" }}
                                                onClick={() => handleConfirm(t.id, t.ticker)}
                                                disabled={confirming === t.id}>
                                                {confirming === t.id ? <span className="spinner" style={{ width: 12, height: 12 }} /> : <CheckCircle size={12} />}
                                                Confirm
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
}
