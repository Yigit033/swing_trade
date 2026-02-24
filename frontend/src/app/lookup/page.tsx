"use client";
import { useState } from "react";
import { lookupTickers, addTrade } from "@/lib/api";
import type { Signal } from "@/lib/api";
import { Zap, Plus, AlertTriangle, Search } from "lucide-react";

export default function LookupPage() {
    const [input, setInput] = useState("");
    const [portfolioValue, setPortfolioValue] = useState(10000);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<Signal[]>([]);
    const [errors, setErrors] = useState<{ ticker: string; error: string }[]>([]);
    const [msg, setMsg] = useState("");
    const [adding, setAdding] = useState<string | null>(null);
    const [ran, setRan] = useState(false);

    const handleLookup = async () => {
        const tickers = input.split(/[\s,]+/).map(t => t.trim().toUpperCase()).filter(Boolean);
        if (!tickers.length) return;
        setLoading(true); setMsg(""); setRan(true);
        try {
            const data = await lookupTickers(tickers, portfolioValue);
            const ok = (data.results || []).filter((r: any) => !r.error);
            const bad = (data.results || []).filter((r: any) => r.error);
            setResults(ok);
            setErrors(bad);
        } catch {
            setMsg("Lookup failed. Is the API server running?");
        } finally {
            setLoading(false);
        }
    };

    const handleAdd = async (s: Signal) => {
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

    return (
        <div>
            <h1 className="page-title gradient-text">Manual Lookup</h1>
            <p className="page-subtitle">Analyze specific tickers on demand</p>

            {/* Input area */}
            <div className="glass-card" style={{ padding: 22, marginBottom: 24 }}>
                <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "flex-end" }}>
                    <div style={{ flex: 2, minWidth: 220 }}>
                        <label style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 6 }}>
                            TICKERS (comma or space separated)
                        </label>
                        <input className="input" value={input} onChange={e => setInput(e.target.value)}
                            onKeyDown={e => e.key === "Enter" && handleLookup()}
                            placeholder="AAPL, TSLA, NVDA..." />
                    </div>
                    <div style={{ flex: 1, minWidth: 160 }}>
                        <label style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 6 }}>
                            PORTFOLIO VALUE
                        </label>
                        <input className="input" type="number" value={portfolioValue}
                            onChange={e => setPortfolioValue(+e.target.value)} />
                    </div>
                    <button className="btn-primary" onClick={handleLookup} disabled={loading || !input.trim()}>
                        {loading ? <span className="spinner" style={{ width: 14, height: 14 }} /> : <Zap size={14} />}
                        Analyze
                    </button>
                </div>
            </div>

            {msg && (
                <div style={{ background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.25)", borderRadius: 10, padding: "10px 16px", marginBottom: 16, fontSize: "0.875rem", color: "var(--accent)" }}>
                    {msg}
                </div>
            )}

            {/* Errors */}
            {errors.map((e, i) => (
                <div key={i} style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.25)", borderRadius: 10, padding: "10px 16px", marginBottom: 8, fontSize: "0.8rem", color: "var(--red)", display: "flex", gap: 8, alignItems: "center" }}>
                    <AlertTriangle size={14} /> <strong>{e.ticker}</strong>: {e.error}
                </div>
            ))}

            {/* Results */}
            {results.length > 0 && (
                <div className="glass-card" style={{ overflow: "hidden" }}>
                    <div style={{ padding: "16px 22px", borderBottom: "1px solid var(--border)" }}>
                        <h3 style={{ margin: 0, fontSize: "0.95rem", fontWeight: 700 }}>
                            🎯 Analysis Results · {results.length} ticker{results.length !== 1 ? "s" : ""}
                        </h3>
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
                                {results.map((s, i) => (
                                    <tr key={i}>
                                        <td><strong style={{ color: "var(--accent)", fontSize: "0.95rem" }}>{s.ticker}</strong></td>
                                        <td>
                                            <span className={`badge ${s.quality_score >= 80 ? "badge-green" : s.quality_score >= 65 ? "badge-blue" : "badge-yellow"}`}>
                                                {s.quality_score >= 80 ? "🔥" : s.quality_score >= 65 ? "✅" : "⚠️"} {s.quality_score?.toFixed(0)}
                                            </span>
                                        </td>
                                        <td style={{ fontWeight: 600 }}>${s.entry_price?.toFixed(2)}</td>
                                        <td style={{ color: "var(--red)" }}>${s.stop_loss?.toFixed(2)}</td>
                                        <td style={{ color: "var(--green)" }}>${(s.target_1 || s.target)?.toFixed(2)}</td>
                                        <td>{s.rsi?.toFixed(1) || "—"}</td>
                                        <td>{s.volume_surge != null ? `${s.volume_surge.toFixed(2)}x` : "—"}</td>
                                        <td>
                                            {s.win_probability != null
                                                ? <span className="badge badge-purple">{(s.win_probability * 100).toFixed(0)}%</span>
                                                : <span style={{ color: "var(--text-muted)" }}>—</span>
                                            }
                                        </td>
                                        <td>
                                            <button className="btn-secondary" style={{ fontSize: "0.78rem", padding: "5px 12px" }}
                                                onClick={() => handleAdd(s)} disabled={adding === s.ticker}>
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

            {!loading && ran && results.length === 0 && errors.length === 0 && (
                <div className="glass-card" style={{ padding: 48, textAlign: "center", color: "var(--text-muted)" }}>
                    <Search size={40} style={{ marginBottom: 12, opacity: 0.3 }} />
                    <div>No signals generated for these tickers. They may not meet entry criteria.</div>
                </div>
            )}

            {!ran && (
                <div className="glass-card" style={{ padding: 60, textAlign: "center" }}>
                    <Zap size={52} style={{ color: "var(--accent)", opacity: 0.4, marginBottom: 16 }} />
                    <div style={{ color: "var(--text-secondary)", marginBottom: 8, fontWeight: 600 }}>On-demand stock analysis</div>
                    <div style={{ color: "var(--text-muted)", fontSize: "0.875rem" }}>
                        Enter any US stock tickers to get immediate signal analysis, stop-loss, target, and win probability.
                    </div>
                </div>
            )}
        </div>
    );
}
