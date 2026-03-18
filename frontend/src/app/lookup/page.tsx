"use client";
import { useState } from "react";
import { lookupTickers, addTrade } from "@/lib/api";
import { Zap, Plus, CheckCircle, XCircle, Circle, ChevronDown, ChevronUp } from "lucide-react";

// ── Label maps (match Streamlit exactly) ──────────────────────────────────
const FILTER_LABELS: Record<string, string> = {
    market_cap: "Piyasa Değeri",
    avg_volume: "Ortalama Hacim",
    atr_percent: "Volatilite (ATR%)",
    float: "Float (Halka Açık Pay)",
    earnings: "Kazanç Raporu",
    price: "Fiyat",
};
const TRIGGER_LABELS: Record<string, string> = {
    volume_surge: "Hacim Patlaması (Volume Surge)",
    atr_percent: "Volatilite Genişlemesi (ATR%)",
    breakout: "Breakout (Direnç Kırılımı)",
};
const SWING_LABELS: Record<string, string> = {
    five_day_momentum: "5-Günlük Momentum",
    above_ma20: "20-Gün Ort. Üzerinde",
    higher_lows: "Yükselen Dipler",
    multi_day_volume: "Çok-Gün Hacim Trendi",
    overextension: "Aşırı Uzama Kontrolü",
};

type AnalysisResult = {
    ticker: string;
    status?: string;
    message?: string;        // error message
    company_name?: string;
    sector?: string;
    market_cap?: number;
    float_shares?: number;
    strategy?: string;
    swing_ready?: boolean;
    filter_passed?: boolean;
    trigger_passed?: boolean;
    rejection_reason?: string;
    rsi?: number;
    five_day_return?: number;
    quality_score?: number;
    swing_type?: string;
    hold_days?: [number, number];
    type_reason?: string;
    entry_price?: number;
    stop_loss?: number;
    target_1?: number;
    target_2?: number;
    position_size?: number;
    volume_surge?: number;
    atr_percent?: number;
    // OBV Trend (v3.0)
    obv_accumulation?: boolean;
    obv_distribution?: boolean;
    obv_bonus?: number;
    // Market Regime (v3.0)
    market_regime?: string;
    regime_multiplier?: number;
    // nested diagnostic
    filter_details?: { filters?: Record<string, { passed: boolean; reason: string }> };
    trigger_details?: { triggers?: Record<string, { passed: boolean; reason: string; optional?: boolean }>; volume_surge?: number; atr_percent?: number };
    swing_details?: Record<string, { passed?: boolean; return?: number; value?: number; distance?: number }>;
};

// ── Helper components ──────────────────────────────────────────────────────
function Icon({ passed, muted }: { passed?: boolean; muted?: boolean }) {
    if (muted) return <span style={{ color: "var(--text-muted)", opacity: 0.35, display: "flex" }}><Circle size={14} /></span>;
    if (passed) return <span style={{ color: "var(--green)", display: "flex" }}><CheckCircle size={14} /></span>;
    return <span style={{ color: "var(--red)", display: "flex" }}><XCircle size={14} /></span>;
}

function DetailRow({ icon, label, reason, warn }: { icon: boolean; label: string; reason: string; warn?: boolean }) {
    return (
        <div className="lookup-detail-row">
            <Icon passed={icon} />
            <span className="lookup-detail-label">{label}</span>
            <span className="lookup-detail-reason" style={{ color: icon ? "var(--text-muted)" : warn !== false ? "var(--red)" : "var(--text-secondary)", fontWeight: icon ? 400 : 600 }}>{reason}</span>
        </div>
    );
}

function StageRow({ label, passed, muted }: { label: string; passed?: boolean; muted?: boolean }) {
    const color = muted ? "var(--text-muted)" : passed ? "var(--green)" : "var(--red)";
    return (
        <div className="lookup-stage-row" style={{ color, opacity: muted ? 0.45 : 1 }}>
            <Icon passed={passed} muted={muted} />
            <span className="lookup-stage-label">{label}</span>
        </div>
    );
}

function ResultCard({ r, onAdd, adding }: { r: AnalysisResult; onAdd: (r: AnalysisResult) => void; adding: boolean }) {
    const [open, setOpen] = useState(true);

    const isSignal = r.swing_ready === true;
    const isError = r.status === "error";
    const filterOk = r.filter_passed === true;
    const triggerOk = r.trigger_passed === true;
    const swingOk = r.swing_ready === true;

    const mcapB = ((r.market_cap ?? 0) / 1e9).toFixed(2);
    const floatM = ((r.float_shares ?? 0) / 1e6).toFixed(0);
    const rsi = r.rsi ?? 0;
    const five_d = r.five_day_return ?? 0;

    return (
        <div className="glass-card" style={{ marginBottom: 16, overflow: "hidden" }}>
            {/* Header row */}
            <div
                className="lookup-card-header"
                style={{
                    background: isSignal ? "rgba(34,197,94,0.06)" : isError ? "rgba(245,158,11,0.06)" : "rgba(239,68,68,0.06)",
                    borderBottom: open ? "1px solid var(--border-muted)" : "none",
                }}
                onClick={() => setOpen(o => !o)}
            >
                <div className="lookup-card-header-main">
                    <span style={{ color: isSignal ? "var(--green)" : isError ? "var(--yellow)" : "var(--red)", display: "flex", flexShrink: 0 }}>
                        {isSignal ? <CheckCircle size={20} /> : <XCircle size={20} />}
                    </span>
                    <div className="lookup-card-header-info">
                        <div className="lookup-card-title">
                            <span>{r.ticker}</span>
                            <span className="lookup-card-subtitle">{r.company_name} | {r.sector}</span>
                        </div>
                        <div style={{ fontSize: "0.7rem", fontWeight: 700, letterSpacing: "0.05em", marginTop: 2, color: isSignal ? "var(--green)" : isError ? "var(--yellow)" : "var(--red)" }}>
                            {isSignal ? "✅ SWING HAZIR" : isError ? `⚠️ HATA — ${r.message}` : "❌ SWING HAZIR DEĞİL"}
                        </div>
                    </div>
                </div>
                {/* Quick metrics */}
                <div className="lookup-card-metrics">
                    {[
                        { label: "STRATEJİ", value: r.strategy || "SmallCap" },
                        { label: "RSI", value: `${rsi.toFixed(0)} ${rsi < 30 ? "🟢" : rsi > 70 ? "🔴" : "⚪"}` },
                        { label: "5-GÜN", value: `${five_d >= 0 ? "+" : ""}${five_d.toFixed(0)}%` },
                        { label: "FLOAT", value: `${floatM}M` },
                        { label: "M.CAP", value: `$${mcapB}B` },
                    ].map(({ label, value }) => (
                        <div key={label} className="lookup-metric-item">
                            <div style={{ color: "var(--text-muted)" }}>{label}</div>
                            <div style={{ color: "var(--text-secondary)", fontWeight: 700 }}>{value}</div>
                        </div>
                    ))}
                    <span style={{ color: "var(--text-muted)", display: "flex", flexShrink: 0 }}>
                        {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </span>
                </div>
            </div>

            {open && !isError && (
                <div style={{ padding: "18px 20px" }}>
                    {/* ── Stage tracker (matches Streamlit) ──────────────────────── */}
                    {!isSignal && (
                        <div style={{ marginBottom: 18 }}>
                            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 700, letterSpacing: "0.05em", marginBottom: 8 }}>ANALİZ AŞAMALARI</div>
                            <StageRow label="Aşama 1: Filtreler (Market Cap, Volume, ATR%, Float, Earnings)" passed={filterOk} />
                            <StageRow label="Aşama 2: Sinyal Tetikleyiciler (Volume Patlama, Volatilite, Breakout)" passed={filterOk ? triggerOk : undefined} muted={!filterOk} />
                            <StageRow label="Aşama 3: Swing Onayı (5-Gün Momentum, MA20 Üzerinde, Higher Lows)" passed={triggerOk ? swingOk : undefined} muted={!triggerOk} />
                        </div>
                    )}

                    {/* ── STAGE 1: Filter detail ───────────────────────────────── */}
                    {r.filter_details && (
                        <div style={{ marginBottom: 16 }}>
                            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontWeight: 700, letterSpacing: "0.05em", marginBottom: 8 }}>📊 AŞAMA 1 — FİLTRELER</div>
                            <div style={{ background: "rgba(0,0,0,0.2)", borderRadius: 8, padding: "8px 14px" }}>
                                {Object.entries(r.filter_details.filters ?? {}).map(([k, v]) => (
                                    <DetailRow key={k} icon={v.passed} label={FILTER_LABELS[k] || k} reason={v.reason} />
                                ))}
                            </div>
                            {!filterOk && (
                                <>
                                    <div style={{ marginTop: 10, padding: "8px 14px", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: 8, fontSize: "0.82rem", color: "var(--red)" }}>
                                        ⛔ Filtrelerden geçemedi — sonraki aşamalar test edilmedi.
                                    </div>
                                    {/* Context tip */}
                                    {(() => {
                                        const failedEntry = Object.values(r.filter_details?.filters ?? {}).find(v => !v.passed);
                                        const reason = failedEntry?.reason ?? "";
                                        const tip =
                                            reason.includes("Market cap") ? "💡 Piyasa değeri small-cap aralığının ($250M–$2.5B) dışında." :
                                                reason.includes("Float") ? "💡 Float çok yüksek. Düşük float'lu hisseler daha patlayıcı hareket eder." :
                                                    reason.includes("Volume") ? "💡 Ortalama işlem hacmi çok düşük — likidite riski." :
                                                        reason.includes("ATR") ? "💡 Volatilite çok düşük — swing trade için yeterli hareket potansiyeli yok." :
                                                            reason.includes("Earnings") ? "💡 Kazanç raporu yakında — rapor sonrası büyük düşüş riski." : "";
                                        return tip ? <div style={{ marginTop: 6, fontSize: "0.78rem", color: "var(--text-muted)" }}>{tip}</div> : null;
                                    })()}
                                </>
                            )}
                        </div>
                    )}

                    {/* ── STAGE 2: Trigger detail (only if filters passed) ──────── */}
                    {filterOk && r.trigger_details && (
                        <div style={{ marginBottom: 16 }}>
                            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontWeight: 700, letterSpacing: "0.05em", marginBottom: 8 }}>🎯 AŞAMA 2 — SİNYAL TETİKLEYİCİLER</div>
                            <div style={{ background: "rgba(0,0,0,0.2)", borderRadius: 8, padding: "8px 14px" }}>
                                {Object.entries(r.trigger_details.triggers ?? {}).map(([k, v]) => (
                                    <DetailRow key={k} icon={v.passed}
                                        label={(TRIGGER_LABELS[k] || k) + (v.optional ? " (opsiyonel)" : "")}
                                        reason={v.reason}
                                        warn={!v.optional}
                                    />
                                ))}
                            </div>
                            {!triggerOk && (
                                <>
                                    <div style={{ marginTop: 10, padding: "8px 14px", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: 8, fontSize: "0.82rem", color: "var(--red)" }}>
                                        {(() => {
                                            const vol = r.trigger_details?.volume_surge ?? 0;
                                            const atr = (r.trigger_details?.atr_percent ?? 0) * 100;
                                            const volFail = vol < 1.3;
                                            const atrFail = atr < 2;
                                            if (volFail && atrFail)
                                                return `⛔ Sinyal tetiklenmedi — Hacim çok düşük (${vol.toFixed(1)}x, min 1.3x) ve volatilite yetersiz (ATR ${atr.toFixed(1)}%, min 2%).`;
                                            if (volFail)
                                                return `⛔ Sinyal tetiklenmedi — Hacim ortalamanın altında (${vol.toFixed(1)}x). En az 1.3x olmalı.`;
                                            if (atrFail)
                                                return `⛔ Sinyal tetiklenmedi — Volatilite çok düşük (ATR ${atr.toFixed(1)}%). En az %2 olmalı.`;
                                            return `⛔ Sinyal tetiklenmedi — Minimum eşikler karşılanmadı (Vol: ${vol.toFixed(1)}x, ATR: ${atr.toFixed(1)}%).`;
                                        })()}
                                    </div>
                                    <div style={{ marginTop: 6, fontSize: "0.78rem", color: "var(--text-muted)" }}>
                                        {(() => {
                                            const vol = r.trigger_details?.volume_surge ?? 0;
                                            const atr = (r.trigger_details?.atr_percent ?? 0) * 100;
                                            const volFail = vol < 1.3;
                                            const atrFail = atr < 2;
                                            if (volFail && atrFail)
                                                return "💡 Bu hisse ne yeterli hacim ne de volatilite gösteriyor. Swing trade için hem güçlü alım ilgisi hem de hareket potansiyeli gerekli.";
                                            if (volFail)
                                                return "💡 Bu hisse şu an yeterli alım ilgisi görmüyor. Hacim patlaması olmadan girmek riskli — momentum olmadan fiyat hareket etmez.";
                                            if (atrFail)
                                                return "💡 Bu hisse çok dar bir aralıkta işlem görüyor. Düşük volatilite = düşük kâr potansiyeli. ATR yükselmesini bekle.";
                                            return "💡 Tetikleyici eşikler tam karşılanmıyor. Hisse izleme listesinde tutulabilir.";
                                        })()}
                                    </div>
                                </>
                            )}
                        </div>
                    )}

                    {/* ── STAGE 3: Swing confirmation detail (only if triggers passed) */}
                    {triggerOk && r.swing_details && (
                        <div style={{ marginBottom: 16 }}>
                            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontWeight: 700, letterSpacing: "0.05em", marginBottom: 8 }}>✨ AŞAMA 3 — SWING ONAY</div>
                            <div style={{ background: "rgba(0,0,0,0.2)", borderRadius: 8, padding: "8px 14px" }}>
                                {Object.entries(r.swing_details).map(([k, v]) => {
                                    if (typeof v !== "object" || v === null) return null;
                                    const label = SWING_LABELS[k] || k;
                                    const passed = v.passed;
                                    const detail = v.return !== undefined ? `${(v.return ?? 0) >= 0 ? "+" : ""}${(v.return ?? 0).toFixed(1)}%` :
                                        v.distance !== undefined ? `${(v.distance ?? 0) >= 0 ? "+" : ""}${(v.distance ?? 0).toFixed(1)}% uzaklık` :
                                            v.value !== undefined ? String(v.value) : "";
                                    return (
                                        <div key={k} className="lookup-detail-row">
                                            <Icon passed={passed === undefined ? undefined : passed} muted={passed === undefined} />
                                            <span className="lookup-detail-label">{label}</span>
                                            <span className="lookup-detail-reason" style={{ color: passed === false ? "var(--red)" : "var(--text-muted)" }}>{detail}</span>
                                        </div>
                                    );
                                })}
                            </div>
                            {!swingOk && (
                                <>
                                    <div style={{ marginTop: 10, padding: "8px 14px", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: 8, fontSize: "0.82rem", color: "var(--red)" }}>
                                        {(() => {
                                            const sd = r.swing_details || {};
                                            const momFail = sd.five_day_momentum && sd.five_day_momentum.passed === false;
                                            const maFail = sd.above_ma20 && sd.above_ma20.passed === false;
                                            if (momFail && maFail)
                                                return "⛔ Swing onayı başarısız — 5 günlük momentum negatif ve fiyat 20 günlük ortalamanın altında.";
                                            if (momFail)
                                                return "⛔ Swing onayı başarısız — 5 günlük momentum negatif (son 5 günde fiyat düşmüş).";
                                            if (maFail)
                                                return "⛔ Swing onayı başarısız — Fiyat 20 günlük hareketli ortalamanın altında.";
                                            return "⛔ Swing onayı başarısız — Gerekli teknik kriterler karşılanmadı.";
                                        })()}
                                    </div>
                                    <div style={{ marginTop: 6, fontSize: "0.78rem", color: "var(--text-muted)" }}>
                                        {(() => {
                                            const sd = r.swing_details || {};
                                            const momFail = sd.five_day_momentum && sd.five_day_momentum.passed === false;
                                            const maFail = sd.above_ma20 && sd.above_ma20.passed === false;
                                            if (momFail)
                                                return "💡 Fiyat kısa vadede zayıf. Yukarı momentum oluşmadan swing trade riskli — dip avcılığı yerine trend takibi yap.";
                                            if (maFail)
                                                return "💡 Fiyat orta vadeli trendinin altında. MA20 üzerine çıkması swing trade için onay sinyali olacaktır.";
                                            return "💡 Teknik yapı henüz swing trade için uygun değil. İzleme listesinde tut ve tekrar kontrol et.";
                                        })()}
                                    </div>
                                </>
                            )}
                        </div>
                    )}

                    {/* ── SIGNAL CARD (swing_ready) ────────────────────────────── */}
                    {isSignal && (
                        <div style={{ background: "rgba(34,197,94,0.06)", border: "1px solid rgba(34,197,94,0.25)", borderRadius: 10, padding: "16px 18px", marginTop: 8 }}>
                            <div style={{ fontSize: "0.72rem", color: "var(--green)", fontWeight: 700, letterSpacing: "0.05em", marginBottom: 14 }}>
                                🎯 SİNYAL — Tip {r.swing_type}
                                {r.quality_score != null && (
                                    <span style={{ marginLeft: 10, background: (r.quality_score >= 80) ? "rgba(34,197,94,0.2)" : "rgba(59,130,246,0.2)", padding: "2px 8px", borderRadius: 6, color: (r.quality_score >= 80) ? "var(--green)" : "var(--accent)" }}>
                                        Kalite: {r.quality_score?.toFixed(0)}
                                    </span>
                                )}
                                {r.type_reason && <span style={{ marginLeft: 10, color: "var(--text-muted)", fontWeight: 400 }}>{r.type_reason}</span>}
                            </div>
                            <div style={{ display: "flex", gap: 20, flexWrap: "wrap", marginBottom: 16 }}>
                                {[
                                    { label: "GİRİŞ", value: `$${r.entry_price?.toFixed(2)}`, color: "var(--text-primary)" },
                                    { label: "STOP LOSS", value: `$${r.stop_loss?.toFixed(2) ?? "—"}`, color: "var(--red)" },
                                    { label: "HEDEF 1", value: `$${r.target_1?.toFixed(2) ?? "—"}`, color: "var(--green)" },
                                    { label: "HEDEF 2", value: `$${r.target_2?.toFixed(2) ?? "—"}`, color: "var(--green)" },
                                    { label: "RSI", value: `${r.rsi?.toFixed(1) ?? "—"}`, color: "var(--text-secondary)" },
                                    { label: "VOL SURGE", value: r.volume_surge != null ? `${r.volume_surge.toFixed(2)}x` : "—", color: "var(--purple)" },
                                    { label: "POZİSYON", value: r.position_size != null ? `${r.position_size} hisse` : "—", color: "var(--text-secondary)" },
                                    { label: "HOLDİNG", value: r.hold_days ? `${r.hold_days[0]}–${r.hold_days[1]} gün` : "—", color: "var(--text-secondary)" },
                                ].map(({ label, value, color }) => (
                                    <div key={label}>
                                        <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontWeight: 700, marginBottom: 2 }}>{label}</div>
                                        <div style={{ fontSize: "0.9rem", fontWeight: 700, color }}>{value}</div>
                                    </div>
                                ))}
                            </div>
                            {/* OBV + Regime indicators (v3.0) */}
                            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 12 }}>
                                {r.obv_accumulation && (
                                    <span style={{ fontSize: "0.75rem", background: "rgba(34,197,94,0.12)", color: "var(--green)", padding: "3px 10px", borderRadius: 6, fontWeight: 600 }}>
                                        📊 OBV Accumulation
                                    </span>
                                )}
                                {r.obv_distribution && (
                                    <span style={{ fontSize: "0.75rem", background: "rgba(239,68,68,0.12)", color: "var(--red)", padding: "3px 10px", borderRadius: 6, fontWeight: 600 }}>
                                        📊 OBV Distribution
                                    </span>
                                )}
                                {r.market_regime && r.market_regime !== "BULL" && (
                                    <span style={{ fontSize: "0.75rem", background: r.market_regime === "BEAR" ? "rgba(239,68,68,0.12)" : "rgba(234,179,8,0.12)", color: r.market_regime === "BEAR" ? "var(--red)" : "var(--yellow)", padding: "3px 10px", borderRadius: 6, fontWeight: 600 }}>
                                        {r.market_regime === "BEAR" ? "🐻" : "⚠️"} {r.market_regime}
                                    </span>
                                )}
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

// ── Page ──────────────────────────────────────────────────────────────────
export default function LookupPage() {
    const [input, setInput] = useState("");
    const [portfolioValue, setPortfolio] = useState(10000);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<AnalysisResult[]>([]);
    const [msg, setMsg] = useState("");
    const [adding, setAdding] = useState<string | null>(null);
    const [ran, setRan] = useState(false);
    const [scanTime, setScanTime] = useState("");

    const handleLookup = async () => {
        const tickers = input.split(/[\s,]+/).map(t => t.trim().toUpperCase()).filter(Boolean);
        if (!tickers.length) return;
        setLoading(true); setMsg(""); setRan(true); setResults([]);
        try {
            const data = await lookupTickers(tickers, portfolioValue);
            setResults(data.results || []);
            setScanTime(new Date().toLocaleTimeString("tr-TR"));
        } catch {
            setMsg("Lookup başarısız. API sunucusu çalışıyor mu?");
        } finally {
            setLoading(false);
        }
    };

    const handleAdd = async (r: AnalysisResult) => {
        setAdding(r.ticker);
        try {
            await addTrade({
                ticker: r.ticker,
                entry_date: new Date().toISOString().slice(0, 10),
                entry_price: r.entry_price,
                stop_loss: r.stop_loss,
                target: r.target_1,
                quality_score: r.quality_score,
                swing_type: r.swing_type || "A",
                signal_price: r.entry_price,
                status: "PENDING",
            });
            setMsg(`✅ ${r.ticker} PENDING olarak eklendi!`);
        } catch {
            setMsg(`❌ ${r.ticker} eklenemedi`);
        } finally {
            setAdding(null);
        }
    };

    const signals = results.filter(r => r.swing_ready);
    const rejected = results.filter(r => !r.swing_ready && r.status !== "error");
    const errors = results.filter(r => r.status === "error");

    return (
        <div>
            <h1 className="page-title gradient-text">Manual Lookup</h1>
            <p className="page-subtitle">
                Hisseleri adım adım analiz et — Streamlit ile aynı detayda filtre ve sinyal analizi
            </p>

            {/* Input */}
            <div className="glass-card" style={{ padding: 22, marginBottom: 24 }}>
                <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "flex-end" }}>
                    <div style={{ flex: 2, minWidth: 220 }}>
                        <label style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 6 }}>
                            TICKER(LAR) — virgül veya boşlukla ayır
                        </label>
                        <input className="input" value={input} onChange={e => setInput(e.target.value)}
                            onKeyDown={e => e.key === "Enter" && handleLookup()}
                            placeholder="AAPL, VELO, WOLF, FIVN..." />
                    </div>
                    <div style={{ flex: 1, minWidth: 160 }}>
                        <label style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 6 }}>
                            PORTFÖY DEĞERİ
                        </label>
                        <input className="input" type="number" value={portfolioValue}
                            onChange={e => setPortfolio(+e.target.value)} />
                    </div>
                    <button className="btn-primary" onClick={handleLookup} disabled={loading || !input.trim()}>
                        {loading ? <span className="spinner" style={{ width: 14, height: 14 }} /> : <Zap size={14} />}
                        Analiz Et
                    </button>
                </div>
            </div>

            {msg && <div style={{ background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.25)", borderRadius: 10, padding: "10px 16px", marginBottom: 16, fontSize: "0.875rem", color: "var(--accent)" }}>{msg}</div>}

            {/* Scan info + summary */}
            {ran && !loading && results.length > 0 && (
                <div style={{ display: "flex", gap: 10, marginBottom: 18, alignItems: "center", flexWrap: "wrap" }}>
                    <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>📊 Son tarama: {scanTime} | {results.length} ticker</span>
                    {[
                        { label: "✅ Sinyal", count: signals.length, color: "var(--green)" },
                        { label: "❌ Reddedildi", count: rejected.length, color: "var(--red)" },
                        ...(errors.length > 0 ? [{ label: "⚠️ Hata", count: errors.length, color: "var(--yellow)" }] : []),
                    ].map(({ label, count, color }) => (
                        <span key={label} style={{ padding: "3px 12px", borderRadius: 8, background: "rgba(255,255,255,0.04)", border: "1px solid var(--border)", fontSize: "0.78rem", color }}>
                            {label}: <strong>{count}</strong>
                        </span>
                    ))}
                </div>
            )}

            {/* Results: signals first, then rejections */}
            {signals.map(r => <ResultCard key={r.ticker} r={r} onAdd={handleAdd} adding={adding === r.ticker} />)}
            {rejected.map(r => <ResultCard key={r.ticker} r={r} onAdd={handleAdd} adding={adding === r.ticker} />)}
            {errors.map(r => (
                <div key={r.ticker} style={{ background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.3)", borderRadius: 10, padding: "12px 16px", marginBottom: 8, fontSize: "0.82rem", color: "var(--yellow)" }}>
                    ⚠️ <strong>{r.ticker}</strong>: {r.message}
                </div>
            ))}

            {!ran && (
                <div className="glass-card" style={{ padding: 60, textAlign: "center" }}>
                    <Zap size={52} style={{ color: "var(--accent)", opacity: 0.4, marginBottom: 16 }} />
                    <div style={{ color: "var(--text-secondary)", fontWeight: 600, marginBottom: 8 }}>Anlık hisse analizi</div>
                    <div style={{ color: "var(--text-muted)", fontSize: "0.875rem" }}>
                        Herhangi bir US hissesi gir — hangi aşamada takıldığını, neden reddedildiğini tam olarak görürsün.
                    </div>
                </div>
            )}
        </div>
    );
}
