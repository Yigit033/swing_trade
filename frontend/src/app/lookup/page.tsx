"use client";
import { useState } from "react";
import { lookupTickers, addTrade } from "@/lib/api";
import { Zap, Plus, CheckCircle, XCircle, Circle, ChevronDown, ChevronUp } from "lucide-react";

// Filter label mapping (snake_case → Turkish label)
const FILTER_LABELS: Record<string, string> = {
    market_cap: "Piyasa Değeri",
    avg_volume: "Ortalama Hacim",
    atr_percent: "ATR Volatilite",
    float: "Float Payları",
    earnings: "Earnings Riski",
};
const TRIGGER_LABELS: Record<string, string> = {
    volume_surge: "Hacim Patlaması",
    atr_percent: "Volatilite (ATR%)",
    breakout: "Fiyat Breakout",
};
const STAGE_NAMES = [
    "Aşama 1: Filtreler (Market Cap, Hacim, ATR%, Float, Earnings)",
    "Aşama 2: Sinyal Tetikleyiciler (Hacim Patlaması, Volatilite, Breakout)",
    "Aşama 3: Swing Onayı (5-Gün Momentum, MA20 Üzerinde, Higher Lows)",
];

type LookupResult = {
    ticker: string;
    company?: string;
    sector?: string;
    market_cap_b?: number;
    float_m?: number;
    current_price?: number;
    rsi?: number;
    five_day_pct?: number;
    result: "SIGNAL" | "REJECTED" | "ERROR";
    stage_failed?: number | null;
    failed_filter?: string | null;
    failed_reason?: string | null;
    filters?: Record<string, { passed: boolean; reason: string; value?: number }>;
    triggers?: Record<string, any> | null;
    swing?: { five_day?: any; above_ma20?: any } | null;
    signal?: any | null;
};

function StageIndicator({ idx, stageFailed }: { idx: number; stageFailed?: number | null }) {
    if (stageFailed == null) return <span style={{ color: "var(--green)" }}><CheckCircle size={14} /></span>;
    if (idx + 1 < stageFailed) return <span style={{ color: "var(--green)" }}><CheckCircle size={14} /></span>;
    if (idx + 1 === stageFailed) return <span style={{ color: "var(--red)" }}><XCircle size={14} /></span>;
    return <span style={{ color: "var(--text-muted)", opacity: 0.4 }}><Circle size={14} /></span>;
}

function FilterRow({ name, data }: { name: string; data: { passed: boolean; reason: string } }) {
    return (
        <div style={{ display: "flex", gap: 10, alignItems: "flex-start", padding: "5px 0", borderBottom: "1px solid var(--border-muted)" }}>
            <span style={{ color: data.passed ? "var(--green)" : "var(--red)", flexShrink: 0, marginTop: 2 }}>
                {data.passed ? <CheckCircle size={13} /> : <XCircle size={13} />}
            </span>
            <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)", minWidth: 140 }}>
                {FILTER_LABELS[name] || name}
            </span>
            <span style={{ fontSize: "0.8rem", color: data.passed ? "var(--text-muted)" : "var(--red)", fontWeight: data.passed ? 400 : 600 }}>
                {data.reason}
            </span>
        </div>
    );
}

function ResultCard({ r, onAdd, adding }: { r: LookupResult; onAdd: (r: LookupResult) => void; adding: boolean }) {
    const [expanded, setExpanded] = useState(true);
    const isSignal = r.result === "SIGNAL";
    const isError = r.result === "ERROR";

    return (
        <div className="glass-card" style={{ marginBottom: 16, overflow: "hidden" }}>
            {/* Header */}
            <div
                style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    padding: "14px 20px", cursor: "pointer",
                    background: isSignal
                        ? "rgba(34,197,94,0.06)"
                        : isError ? "rgba(245,158,11,0.06)" : "rgba(239,68,68,0.06)",
                    borderBottom: expanded ? "1px solid var(--border-muted)" : "none",
                }}
                onClick={() => setExpanded(e => !e)}
            >
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <span style={{ color: isSignal ? "var(--green)" : isError ? "var(--yellow)" : "var(--red)", display: "flex" }}>
                        {isSignal ? <CheckCircle size={20} /> : <XCircle size={20} />}
                    </span>
                    <div>
                        <div style={{ fontWeight: 800, fontSize: "1.05rem", letterSpacing: "-0.01em" }}>
                            {r.ticker}
                            <span style={{ fontWeight: 400, fontSize: "0.8rem", color: "var(--text-muted)", marginLeft: 10 }}>
                                {r.company}
                            </span>
                        </div>
                        <div style={{ fontSize: "0.72rem", color: isSignal ? "var(--green)" : isError ? "var(--yellow)" : "var(--red)", fontWeight: 700, letterSpacing: "0.05em", marginTop: 2 }}>
                            {isSignal ? "✅ SWING HAZIR — Sinyal Üretildi" : isError ? `⚠️ HATA — ${r.failed_reason}` : "❌ SWING HAZIR DEĞİL — Entry Kriterleri Karşılanmadı"}
                        </div>
                    </div>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                    {/* Quick meta */}
                    {r.market_cap_b != null && (
                        <div style={{ textAlign: "right", fontSize: "0.72rem" }}>
                            <div style={{ color: "var(--text-muted)" }}>MARKET CAP</div>
                            <div style={{ color: "var(--text-secondary)", fontWeight: 600 }}>${r.market_cap_b?.toFixed(2)}B</div>
                        </div>
                    )}
                    {r.rsi != null && (
                        <div style={{ textAlign: "right", fontSize: "0.72rem" }}>
                            <div style={{ color: "var(--text-muted)" }}>RSI</div>
                            <div style={{ color: r.rsi < 35 ? "var(--green)" : r.rsi > 70 ? "var(--red)" : "var(--text-secondary)", fontWeight: 600 }}>
                                {r.rsi?.toFixed(1)} {r.rsi < 35 ? "🟢" : r.rsi > 70 ? "🔴" : ""}
                            </div>
                        </div>
                    )}
                    {r.five_day_pct != null && (
                        <div style={{ textAlign: "right", fontSize: "0.72rem" }}>
                            <div style={{ color: "var(--text-muted)" }}>5-GÜN</div>
                            <div style={{ color: (r.five_day_pct ?? 0) >= 0 ? "var(--green)" : "var(--red)", fontWeight: 600 }}>
                                {(r.five_day_pct ?? 0) >= 0 ? "+" : ""}{r.five_day_pct?.toFixed(1)}%
                            </div>
                        </div>
                    )}
                    {r.float_m != null && (
                        <div style={{ textAlign: "right", fontSize: "0.72rem" }}>
                            <div style={{ color: "var(--text-muted)" }}>FLOAT</div>
                            <div style={{ color: "var(--text-secondary)", fontWeight: 600 }}>{r.float_m?.toFixed(0)}M</div>
                        </div>
                    )}
                    <span style={{ color: "var(--text-muted)", display: "flex" }}>
                        {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </span>
                </div>
            </div>

            {/* Expanded details */}
            {expanded && !isError && (
                <div style={{ padding: "18px 20px" }}>
                    {/* Stage tracker */}
                    <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 18 }}>
                        <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontWeight: 700, marginBottom: 6, letterSpacing: "0.05em" }}>
                            ANALİZ AŞAMALARI
                        </div>
                        {STAGE_NAMES.map((name, idx) => (
                            <div key={idx} style={{ display: "flex", alignItems: "center", gap: 10, fontSize: "0.82rem", color: r.stage_failed == null ? "var(--green)" : idx + 1 < (r.stage_failed ?? 99) ? "var(--green)" : idx + 1 === r.stage_failed ? "var(--red)" : "var(--text-muted)", opacity: r.stage_failed != null && idx + 1 > r.stage_failed ? 0.4 : 1 }}>
                                <StageIndicator idx={idx} stageFailed={r.stage_failed} />
                                {name}
                            </div>
                        ))}
                    </div>

                    {/* Rejection detail */}
                    {r.result === "REJECTED" && r.stage_failed === 1 && r.filters && (
                        <div>
                            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontWeight: 700, letterSpacing: "0.05em", marginBottom: 8 }}>
                                📊 AŞAMA 1 — FİLTRE DETAYI
                            </div>
                            <div style={{ background: "rgba(0,0,0,0.2)", borderRadius: 8, padding: "10px 14px" }}>
                                {Object.entries(r.filters).map(([k, v]) => (
                                    <FilterRow key={k} name={k} data={v} />
                                ))}
                            </div>
                            <div style={{ marginTop: 12, padding: "10px 14px", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: 8, fontSize: "0.82rem", color: "var(--red)" }}>
                                ⛔ <strong>Reddedilen filtre:</strong> {FILTER_LABELS[r.failed_filter ?? ""] || r.failed_filter} — {r.failed_reason}
                            </div>
                        </div>
                    )}

                    {r.result === "REJECTED" && r.stage_failed === 2 && (
                        <div>
                            <div style={{ padding: "10px 14px", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: 8, fontSize: "0.82rem", color: "var(--red)" }}>
                                ⛔ <strong>Tetikleyici başarısız:</strong> {r.failed_reason}
                                <div style={{ marginTop: 6, color: "var(--text-muted)", fontSize: "0.75rem" }}>Aşama 1 filtrelerinden geçti, ancak Volume Surge / ATR / Breakout sinyali aktif değil.</div>
                            </div>
                            {r.filters && (
                                <div style={{ marginTop: 12 }}>
                                    <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontWeight: 700, letterSpacing: "0.05em", marginBottom: 6 }}>GEÇEN FİLTRELER (1. Aşama ✅)</div>
                                    <div style={{ background: "rgba(0,0,0,0.2)", borderRadius: 8, padding: "8px 14px" }}>
                                        {Object.entries(r.filters).map(([k, v]) => <FilterRow key={k} name={k} data={v} />)}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {r.result === "REJECTED" && r.stage_failed === 3 && r.swing && (
                        <div>
                            <div style={{ padding: "10px 14px", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: 8, fontSize: "0.82rem", color: "var(--red)" }}>
                                ⛔ <strong>Swing onayı başarısız:</strong> {r.failed_reason}
                            </div>
                            <div style={{ marginTop: 12, display: "flex", gap: 12, flexWrap: "wrap" }}>
                                {r.swing.five_day && (
                                    <div style={{ flex: 1, minWidth: 180, padding: "12px 14px", background: "rgba(0,0,0,0.2)", borderRadius: 8 }}>
                                        <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 700, marginBottom: 6 }}>5-GÜN MOMENTUM</div>
                                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                            <span style={{ color: r.swing.five_day.passed ? "var(--green)" : "var(--red)" }}>
                                                {r.swing.five_day.passed ? <CheckCircle size={14} /> : <XCircle size={14} />}
                                            </span>
                                            <span style={{ fontSize: "0.85rem", fontWeight: 600 }}>
                                                {((r.swing.five_day.return ?? 0) * 100).toFixed(1)}%
                                            </span>
                                            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>(gereken: &gt; 0%)</span>
                                        </div>
                                    </div>
                                )}
                                {r.swing.above_ma20 && (
                                    <div style={{ flex: 1, minWidth: 180, padding: "12px 14px", background: "rgba(0,0,0,0.2)", borderRadius: 8 }}>
                                        <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 700, marginBottom: 6 }}>MA20 ÜSTÜNDE</div>
                                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                            <span style={{ color: r.swing.above_ma20.passed ? "var(--green)" : "var(--red)" }}>
                                                {r.swing.above_ma20.passed ? <CheckCircle size={14} /> : <XCircle size={14} />}
                                            </span>
                                            <span style={{ fontSize: "0.85rem", fontWeight: 600 }}>
                                                {((r.swing.above_ma20.distance ?? 0) * 100).toFixed(1)}%
                                            </span>
                                            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>MA20&apos;ye uzaklık</span>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Signal card */}
                    {isSignal && r.signal && (
                        <div style={{ background: "rgba(34,197,94,0.06)", border: "1px solid rgba(34,197,94,0.25)", borderRadius: 10, padding: "16px 18px" }}>
                            <div style={{ fontSize: "0.72rem", color: "var(--green)", fontWeight: 700, letterSpacing: "0.05em", marginBottom: 14 }}>
                                🎯 SİNYAL — {r.signal?.swing_type || "A"} Tipi Swing
                                {r.signal?.quality_score != null && (
                                    <span style={{ marginLeft: 10, background: r.signal.quality_score >= 80 ? "rgba(34,197,94,0.2)" : "rgba(59,130,246,0.2)", padding: "2px 8px", borderRadius: 6, color: r.signal.quality_score >= 80 ? "var(--green)" : "var(--accent)" }}>
                                        Kalite: {r.signal.quality_score?.toFixed(0)}
                                    </span>
                                )}
                            </div>
                            <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 16 }}>
                                {[
                                    { label: "GİRİŞ", value: `$${r.signal.entry_price?.toFixed(2)}`, color: "var(--text-primary)" },
                                    { label: "STOP LOSS", value: `$${r.signal.stop_loss?.toFixed(2)}`, color: "var(--red)" },
                                    { label: "HEDEF", value: `$${(r.signal.target_1 || r.signal.target)?.toFixed(2)}`, color: "var(--green)" },
                                    { label: "RSI", value: r.signal.rsi?.toFixed(1) ?? "—", color: "var(--text-secondary)" },
                                    { label: "VOL SURGE", value: r.signal.volume_surge != null ? `${r.signal.volume_surge.toFixed(2)}x` : "—", color: "var(--purple)" },
                                ].map(({ label, value, color }) => (
                                    <div key={label}>
                                        <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontWeight: 700, marginBottom: 2 }}>{label}</div>
                                        <div style={{ fontSize: "0.95rem", fontWeight: 700, color }}>{value}</div>
                                    </div>
                                ))}
                            </div>
                            <button className="btn-primary" style={{ fontSize: "0.82rem", padding: "7px 18px" }}
                                onClick={() => onAdd(r)} disabled={adding}>
                                {adding ? <span className="spinner" style={{ width: 13, height: 13 }} /> : <Plus size={13} />}
                                Takibe Al (PENDING)
                            </button>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default function LookupPage() {
    const [input, setInput] = useState("");
    const [portfolioValue, setPortfolioValue] = useState(10000);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<LookupResult[]>([]);
    const [msg, setMsg] = useState("");
    const [adding, setAdding] = useState<string | null>(null);
    const [ran, setRan] = useState(false);

    const handleLookup = async () => {
        const tickers = input.split(/[\s,]+/).map(t => t.trim().toUpperCase()).filter(Boolean);
        if (!tickers.length) return;
        setLoading(true); setMsg(""); setRan(true); setResults([]);
        try {
            const data = await lookupTickers(tickers, portfolioValue);
            setResults(data.results || []);
        } catch {
            setMsg("Lookup başarısız. API sunucusu çalışıyor mu?");
        } finally {
            setLoading(false);
        }
    };

    const handleAdd = async (r: LookupResult) => {
        if (!r.signal) return;
        setAdding(r.ticker);
        try {
            await addTrade({
                ticker: r.ticker,
                entry_date: new Date().toISOString().slice(0, 10),
                entry_price: r.signal.entry_price,
                stop_loss: r.signal.stop_loss,
                target: r.signal.target_1 || r.signal.target,
                quality_score: r.signal.quality_score,
                swing_type: r.signal.swing_type || "A",
                atr: r.signal.atr,
                signal_price: r.signal.entry_price,
                status: "PENDING",
            });
            setMsg(`✅ ${r.ticker} PENDING olarak eklendi!`);
        } catch {
            setMsg(`❌ ${r.ticker} eklenemedi`);
        } finally {
            setAdding(null);
        }
    };

    const signals = results.filter(r => r.result === "SIGNAL");
    const rejected = results.filter(r => r.result === "REJECTED");
    const errors = results.filter(r => r.result === "ERROR");

    return (
        <div>
            <h1 className="page-title gradient-text">Manual Lookup</h1>
            <p className="page-subtitle">Hisseleri adım adım analiz et — neden reddedildi, hangi kriterde takıldı göster</p>

            {/* Input area */}
            <div className="glass-card" style={{ padding: 22, marginBottom: 24 }}>
                <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "flex-end" }}>
                    <div style={{ flex: 2, minWidth: 220 }}>
                        <label style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 6 }}>
                            TICKER(LAR) — virgül veya boşlukla ayır
                        </label>
                        <input className="input" value={input} onChange={e => setInput(e.target.value)}
                            onKeyDown={e => e.key === "Enter" && handleLookup()}
                            placeholder="AAPL, VELO, TSLA..." />
                    </div>
                    <div style={{ flex: 1, minWidth: 160 }}>
                        <label style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 6 }}>
                            PORTFÖY DEĞERİ
                        </label>
                        <input className="input" type="number" value={portfolioValue}
                            onChange={e => setPortfolioValue(+e.target.value)} />
                    </div>
                    <button className="btn-primary" onClick={handleLookup} disabled={loading || !input.trim()}>
                        {loading ? <span className="spinner" style={{ width: 14, height: 14 }} /> : <Zap size={14} />}
                        Analiz Et
                    </button>
                </div>
            </div>

            {msg && (
                <div style={{ background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.25)", borderRadius: 10, padding: "10px 16px", marginBottom: 16, fontSize: "0.875rem", color: "var(--accent)" }}>
                    {msg}
                </div>
            )}

            {/* Summary bar */}
            {ran && !loading && results.length > 0 && (
                <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
                    {[
                        { label: "Toplam", count: results.length, color: "var(--text-secondary)" },
                        { label: "✅ Sinyal", count: signals.length, color: "var(--green)" },
                        { label: "❌ Reddedildi", count: rejected.length, color: "var(--red)" },
                        { label: "⚠️ Hata", count: errors.length, color: "var(--yellow)" },
                    ].map(({ label, count, color }) => (
                        <div key={label} style={{ padding: "6px 16px", borderRadius: 8, background: "rgba(255,255,255,0.04)", border: "1px solid var(--border)", fontSize: "0.8rem", color }}>
                            {label}: <strong>{count}</strong>
                        </div>
                    ))}
                </div>
            )}

            {/* Signals first */}
            {signals.length > 0 && (
                <div style={{ marginBottom: 8 }}>
                    {signals.map(r => <ResultCard key={r.ticker} r={r} onAdd={handleAdd} adding={adding === r.ticker} />)}
                </div>
            )}

            {/* Rejected */}
            {rejected.length > 0 && (
                <div>
                    {rejected.map(r => <ResultCard key={r.ticker} r={r} onAdd={handleAdd} adding={adding === r.ticker} />)}
                </div>
            )}

            {/* Errors */}
            {errors.map(r => (
                <div key={r.ticker} style={{ background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.25)", borderRadius: 10, padding: "12px 16px", marginBottom: 8, fontSize: "0.82rem", color: "var(--yellow)" }}>
                    ⚠️ <strong>{r.ticker}</strong>: {r.failed_reason}
                </div>
            ))}

            {/* Empty / intro state */}
            {!ran && (
                <div className="glass-card" style={{ padding: 60, textAlign: "center" }}>
                    <Zap size={52} style={{ color: "var(--accent)", opacity: 0.4, marginBottom: 16 }} />
                    <div style={{ color: "var(--text-secondary)", marginBottom: 8, fontWeight: 600 }}>Anlık hisse analizi</div>
                    <div style={{ color: "var(--text-muted)", fontSize: "0.875rem" }}>
                        Herhangi bir US hissesi gir — hangi aşamada takıldığını, neden reddedildiğini tam olarak görürsün.
                    </div>
                </div>
            )}

            {!loading && ran && results.length === 0 && (
                <div className="glass-card" style={{ padding: 48, textAlign: "center", color: "var(--text-muted)" }}>
                    Hiç sonuç bulunamadı.
                </div>
            )}
        </div>
    );
}
