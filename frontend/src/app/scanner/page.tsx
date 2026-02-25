"use client";
import { useState, useEffect } from "react";
import { runSmallcapScan, trackSignal, addTrade } from "@/lib/api";
import type { Signal } from "@/lib/api";
import { Search, Plus, TrendingUp, AlertTriangle, Settings, ChevronDown, ChevronUp } from "lucide-react";

function QualityBadge({ score }: { score: number }) {
    const cls = score >= 80 ? "badge-green" : score >= 65 ? "badge-blue" : "badge-yellow";
    const emoji = score >= 80 ? "🔥" : score >= 65 ? "✅" : "⚠️";
    return <span className={`badge ${cls}`}>{emoji} {score.toFixed(0)}</span>;
}

export default function ScannerPage() {
    // Scan params
    const [minQuality, setMinQuality] = useState(65);
    const [topN, setTopN] = useState(10);
    const [portfolioValue, setPortfolioValue] = useState(10000);

    // Auto-track settings (persisted in localStorage)
    const [autoTrackEnabled, setAutoTrackEnabled] = useState(true);
    const [autoTrackMinQuality, setAutoTrackMinQuality] = useState(65);
    const [autoTrackOpen, setAutoTrackOpen] = useState(false);

    // Results
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<{ signals: Signal[]; stats: Record<string, unknown>; market_regime: string } | null>(null);
    const [error, setError] = useState("");
    const [adding, setAdding] = useState<string | null>(null);
    const [msg, setMsg] = useState("");

    // Auto-track result state
    const [autoTrackResult, setAutoTrackResult] = useState<{ tracked: string[]; skipped: string[] } | null>(null);

    // Load persisted auto-track settings on mount
    useEffect(() => {
        try {
            const saved = localStorage.getItem("autoTrack");
            if (saved) {
                const { enabled, minQuality: mq } = JSON.parse(saved);
                if (enabled !== undefined) setAutoTrackEnabled(enabled);
                if (mq !== undefined) setAutoTrackMinQuality(mq);
            }
        } catch { /* ignore */ }
    }, []);

    // Persist auto-track settings on change
    useEffect(() => {
        try {
            localStorage.setItem("autoTrack", JSON.stringify({ enabled: autoTrackEnabled, minQuality: autoTrackMinQuality }));
        } catch { /* ignore */ }
    }, [autoTrackEnabled, autoTrackMinQuality]);

    const runAutoTrack = async (signals: Signal[]) => {
        setAutoTrackResult(null);
        const qualifying = signals.filter(s => s.quality_score >= autoTrackMinQuality);
        if (!qualifying.length) return;

        const tracked: string[] = [];
        const skipped: string[] = [];

        await Promise.all(qualifying.map(async (s) => {
            try {
                const res = await trackSignal(s);
                if (res?.status === "added") tracked.push(s.ticker);
                else skipped.push(s.ticker); // duplicate
            } catch {
                skipped.push(s.ticker);
            }
        }));

        setAutoTrackResult({ tracked, skipped });
    };

    const scan = async () => {
        setLoading(true); setError(""); setMsg(""); setAutoTrackResult(null);
        try {
            const data = await runSmallcapScan({ min_quality: minQuality, top_n: topN, portfolio_value: portfolioValue });
            setResult(data);

            // Auto-track if enabled
            if (autoTrackEnabled && data?.signals?.length > 0) {
                await runAutoTrack(data.signals);
            }
        } catch {
            setError("Scan failed. Make sure the API is running.");
        } finally {
            setLoading(false);
        }
    };

    const addToTrades = async (s: Signal) => {
        setAdding(s.ticker);
        try {
            await addTrade({
                ticker: s.ticker,
                entry_date: new Date().toISOString().slice(0, 10),
                entry_price: s.entry_price,
                stop_loss: s.stop_loss,
                target: s.target_1 || s.target,
                quality_score: s.quality_score,
                swing_type: s.swing_type || "A",
                atr: s.atr,
                signal_price: s.entry_price,
                status: "PENDING",
            });
            setMsg(`✅ ${s.ticker} added as PENDING trade!`);
        } catch {
            setMsg(`❌ Failed to add ${s.ticker}`);
        } finally {
            setAdding(null);
        }
    };

    const regime = result?.market_regime || "";
    const regimeColor = regime === "RISK_ON" ? "var(--green)" : regime === "RISK_OFF" ? "var(--red)" : "var(--yellow)";

    return (
        <div>
            <h1 className="page-title gradient-text">SmallCap Scanner</h1>
            <p className="page-subtitle">AI-powered momentum signals · SmallCap universe</p>

            {/* Controls */}
            <div className="glass-card" style={{ padding: 20, marginBottom: 16, display: "flex", gap: 20, flexWrap: "wrap", alignItems: "flex-end" }}>
                <div style={{ flex: 1, minWidth: 160 }}>
                    <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 6 }}>
                        MIN QUALITY
                    </label>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <input type="range" min={30} max={95} value={minQuality} onChange={e => setMinQuality(+e.target.value)}
                            style={{ flex: 1, accentColor: "var(--accent)" }} />
                        <span style={{ color: "var(--accent)", fontWeight: 700, minWidth: 28 }}>{minQuality}</span>
                    </div>
                </div>
                <div style={{ flex: 1, minWidth: 160 }}>
                    <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 6 }}>
                        TOP N RESULTS
                    </label>
                    <input className="input" type="number" min={1} max={30} value={topN} onChange={e => setTopN(+e.target.value)} style={{ maxWidth: 100 }} />
                </div>
                <div style={{ flex: 1, minWidth: 160 }}>
                    <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 6 }}>
                        PORTFOLIO ($)
                    </label>
                    <input className="input" type="number" value={portfolioValue} onChange={e => setPortfolioValue(+e.target.value)} style={{ maxWidth: 140 }} />
                </div>
                <button className="btn-primary" onClick={scan} disabled={loading}>
                    {loading ? <span className="spinner" /> : <Search size={15} />}
                    {loading ? "Scanning..." : "Run Scan"}
                </button>
            </div>

            {/* ⚙️ Auto-Track Ayarları */}
            <div className="glass-card" style={{ marginBottom: 24, overflow: "hidden" }}>
                <button
                    onClick={() => setAutoTrackOpen(o => !o)}
                    style={{
                        width: "100%", padding: "14px 20px", background: "transparent", border: "none",
                        display: "flex", alignItems: "center", justifyContent: "space-between",
                        cursor: "pointer", color: "var(--text-primary)", fontWeight: 600, fontSize: "0.9rem",
                    }}
                >
                    <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <Settings size={15} style={{ color: "var(--accent)" }} />
                        ⚙️ Auto-Track Ayarları
                        {autoTrackEnabled && (
                            <span className="badge badge-green" style={{ fontSize: "0.65rem", padding: "2px 7px" }}>
                                AKTİF ✓
                            </span>
                        )}
                    </span>
                    {autoTrackOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </button>

                {autoTrackOpen && (
                    <div style={{ borderTop: "1px solid var(--border)", padding: "16px 20px" }}>
                        <div style={{ display: "flex", gap: 32, flexWrap: "wrap", alignItems: "flex-start" }}>
                            {/* Enable toggle */}
                            <div>
                                <label style={{
                                    display: "flex", alignItems: "center", gap: 10, cursor: "pointer",
                                    fontSize: "0.875rem", fontWeight: 600, userSelect: "none"
                                }}>
                                    <div
                                        onClick={() => setAutoTrackEnabled(e => !e)}
                                        style={{
                                            width: 42, height: 24, borderRadius: 12, position: "relative",
                                            background: autoTrackEnabled ? "var(--green)" : "var(--border)",
                                            transition: "background 0.2s", cursor: "pointer", flexShrink: 0,
                                        }}
                                    >
                                        <div style={{
                                            position: "absolute", top: 3, left: autoTrackEnabled ? 21 : 3,
                                            width: 18, height: 18, borderRadius: "50%", background: "#fff",
                                            transition: "left 0.2s", boxShadow: "0 1px 3px rgba(0,0,0,0.3)",
                                        }} />
                                    </div>
                                    📌 Otomatik Paper Trade Takibi
                                </label>
                                <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 6, marginLeft: 52 }}>
                                    Scan sonucu kaliteli sinyalleri otomatik paper trade&apos;e ekler
                                </p>
                            </div>

                            {/* Min quality slider */}
                            <div style={{ flex: 1, minWidth: 220 }}>
                                <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 6 }}>
                                    MİN KALİTE (Auto-Track)
                                </label>
                                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                    <input
                                        type="range" min={50} max={100} value={autoTrackMinQuality}
                                        onChange={e => setAutoTrackMinQuality(+e.target.value)}
                                        disabled={!autoTrackEnabled}
                                        style={{ flex: 1, accentColor: autoTrackEnabled ? "var(--green)" : "var(--border)", opacity: autoTrackEnabled ? 1 : 0.4 }}
                                    />
                                    <span style={{ color: autoTrackEnabled ? "var(--green)" : "var(--text-muted)", fontWeight: 700, minWidth: 32 }}>
                                        {autoTrackMinQuality}
                                    </span>
                                </div>
                                <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 4 }}>
                                    Bu puanın üzerindeki sinyaller otomatik takibe alınır
                                </p>
                            </div>
                        </div>

                        {/* Status info */}
                        <div style={{
                            marginTop: 14, padding: "10px 14px", borderRadius: 8, fontSize: "0.8rem",
                            background: autoTrackEnabled ? "rgba(34,197,94,0.08)" : "rgba(255,255,255,0.03)",
                            border: `1px solid ${autoTrackEnabled ? "rgba(34,197,94,0.3)" : "var(--border)"}`,
                            color: autoTrackEnabled ? "var(--green)" : "var(--text-muted)",
                        }}>
                            {autoTrackEnabled
                                ? `✅ Kalite ≥ ${autoTrackMinQuality} olan sinyaller otomatik paper trade'e eklenecek`
                                : "⏸️ Otomatik takip kapalı — sinyalleri manuel olarak eklemeniz gerekir"
                            }
                        </div>
                    </div>
                )}
            </div>

            {error && (
                <div style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 10, padding: "12px 18px", color: "var(--red)", marginBottom: 16, display: "flex", gap: 10, alignItems: "center" }}>
                    <AlertTriangle size={16} /> {error}
                </div>
            )}

            {msg && (
                <div style={{ background: "rgba(34,197,94,0.1)", border: "1px solid rgba(34,197,94,0.3)", borderRadius: 10, padding: "12px 18px", color: "var(--green)", marginBottom: 16 }}>
                    {msg}
                </div>
            )}

            {/* Auto-track result banner */}
            {autoTrackResult && (
                <div style={{ marginBottom: 16 }}>
                    {autoTrackResult.tracked.length > 0 && (
                        <div style={{ background: "rgba(34,197,94,0.1)", border: "1px solid rgba(34,197,94,0.3)", borderRadius: 10, padding: "12px 18px", color: "var(--green)", marginBottom: 8 }}>
                            📌 <strong>Auto-Track:</strong> {autoTrackResult.tracked.length} sinyal paper trade&apos;e eklendi → {autoTrackResult.tracked.join(", ")}
                        </div>
                    )}
                    {autoTrackResult.skipped.length > 0 && (
                        <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid var(--border)", borderRadius: 10, padding: "10px 18px", color: "var(--text-muted)", fontSize: "0.85rem" }}>
                            ⏭️ Zaten takipte: {autoTrackResult.skipped.join(", ")}
                        </div>
                    )}
                    {autoTrackResult.tracked.length === 0 && autoTrackResult.skipped.length === 0 && (
                        <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid var(--border)", borderRadius: 10, padding: "10px 18px", color: "var(--text-muted)", fontSize: "0.85rem" }}>
                            ℹ️ Kalite ≥ {autoTrackMinQuality} olan sinyal bulunamadı
                        </div>
                    )}
                </div>
            )}

            {result && (
                <>
                    {/* Stats bar */}
                    <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
                        <div className="metric-card" style={{ padding: "14px 20px", flex: 1, minWidth: 120 }}>
                            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>Signals</div>
                            <div style={{ fontSize: "1.4rem", fontWeight: 800 }}>{result.signals.length}</div>
                        </div>
                        <div className="metric-card" style={{ padding: "14px 20px", flex: 1, minWidth: 120 }}>
                            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>Market</div>
                            <div style={{ fontSize: "1rem", fontWeight: 800, color: regimeColor }}>{regime || "—"}</div>
                        </div>
                        <div className="metric-card" style={{ padding: "14px 20px", flex: 1, minWidth: 120 }}>
                            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>Scanned</div>
                            <div style={{ fontSize: "1.4rem", fontWeight: 800 }}>{(result.stats as Record<string, number>).stocks_scanned || "—"}</div>
                        </div>
                    </div>

                    {/* Signals table */}
                    {result.signals.length === 0 ? (
                        <div className="glass-card" style={{ padding: 48, textAlign: "center", color: "var(--text-muted)" }}>
                            <TrendingUp size={48} style={{ marginBottom: 12, opacity: 0.3 }} />
                            <div>No signals found with current filters. Try lowering min quality.</div>
                        </div>
                    ) : (
                        <div className="glass-card" style={{ overflow: "hidden" }}>
                            <div style={{ padding: "18px 22px", borderBottom: "1px solid var(--border)" }}>
                                <h2 style={{ margin: 0, fontSize: "1rem", fontWeight: 700 }}>🎯 Top Signals</h2>
                            </div>
                            <div style={{ overflowX: "auto" }}>
                                <table className="data-table">
                                    <thead>
                                        <tr>
                                            <th>Ticker</th><th>Quality</th><th>Entry</th><th>Stop</th>
                                            <th>Target</th><th>RSI</th><th>Vol Surge</th><th>Win Prob</th><th>Action</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {result.signals.map((s, i) => (
                                            <tr key={i}>
                                                <td><strong style={{ color: "var(--accent)", fontSize: "0.95rem" }}>{s.ticker}</strong></td>
                                                <td><QualityBadge score={s.quality_score} /></td>
                                                <td style={{ fontWeight: 600 }}>${s.entry_price?.toFixed(2)}</td>
                                                <td style={{ color: "var(--red)" }}>${s.stop_loss?.toFixed(2)}</td>
                                                <td style={{ color: "var(--green)" }}>${(s.target_1 || s.target)?.toFixed(2)}</td>
                                                <td>{s.rsi?.toFixed(1)}</td>
                                                <td>{s.volume_surge?.toFixed(2)}x</td>
                                                <td>
                                                    {s.win_probability != null
                                                        ? <span className="badge badge-purple">{(s.win_probability * 100).toFixed(0)}%</span>
                                                        : <span style={{ color: "var(--text-muted)" }}>—</span>
                                                    }
                                                </td>
                                                <td>
                                                    <button className="btn-secondary" onClick={() => addToTrades(s)}
                                                        disabled={adding === s.ticker} style={{ fontSize: "0.78rem", padding: "5px 12px" }}>
                                                        {adding === s.ticker ? <span className="spinner" style={{ width: 12, height: 12 }} /> : <Plus size={12} />}
                                                        Track
                                                    </button>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </>
            )}

            {!result && !loading && (
                <div className="glass-card" style={{ padding: 60, textAlign: "center" }}>
                    <Search size={56} style={{ color: "var(--text-muted)", marginBottom: 16 }} />
                    <h3 style={{ color: "var(--text-secondary)", marginBottom: 8 }}>Ready to scan</h3>
                    <p style={{ color: "var(--text-muted)", fontSize: "0.875rem" }}>
                        Configure parameters above and click Run Scan to find opportunities.
                    </p>
                </div>
            )}
        </div>
    );
}
