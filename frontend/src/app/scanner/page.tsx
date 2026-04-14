"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import { trackSignal, addTrade } from "@/lib/api";
import { useScannerJob, SCAN_COMPLETE_EVENT } from "@/providers/ScannerJobProvider";
import type { Signal } from "@/lib/api";
import { Search, Plus, TrendingUp, AlertTriangle, Settings, ChevronDown, ChevronUp, Star, Zap, Shield, Target, BarChart2, SlidersHorizontal, Sparkles } from "lucide-react";

/* ───── helpers ───── */
/** Engine scan_stock early-exit reasons (API stats.reject_counts). */
const SCAN_REJECT_LABELS: Record<string, string> = {
    insufficient_data: "Veri yetersiz",
    filter_failed: "Evren filtresi",
    no_trigger: "Tetik yok",
    swing_not_ready: "Swing onayı",
    rsi_gate: "RSI eşiği",
    late_entry: "Geç giriş",
    obv_distribution: "OBV dağıtım",
    trend_phase_weak: "Zayıf trend",
    scan_error: "Tarama hatası",
};

function formatRejectSummary(stats: Record<string, unknown>): string | null {
    const rc = stats.reject_counts;
    if (!rc || typeof rc !== "object" || Array.isArray(rc)) return null;
    const entries = Object.entries(rc as Record<string, unknown>)
        .filter(([, v]) => typeof v === "number" && (v as number) > 0)
        .sort((a, b) => (b[1] as number) - (a[1] as number))
        .slice(0, 8);
    if (entries.length === 0) return null;
    return entries.map(([k, v]) => `${SCAN_REJECT_LABELS[k] ?? k}: ${v}`).join(" · ");
}

function QualityBadge({ score }: { score: number }) {
    const cls = score >= 80 ? "badge-green" : score >= 65 ? "badge-blue" : "badge-yellow";
    const emoji = score >= 80 ? "🔥" : score >= 65 ? "✅" : "⚠️";
    return <span className={`badge ${cls}`}>{emoji} {score.toFixed(0)}</span>;
}

function TypeLabel({ type, label }: { type?: string; label?: string }) {
    const colors: Record<string, string> = {
        S: "badge-red", B: "badge-yellow", C: "badge-green", A: "badge-blue",
    };
    const labels: Record<string, string> = {
        S: "Short Squeeze", B: "Momentum", C: "Erken Aşama", A: "Continuation",
    };
    const t = type || "A";
    const holdLabels: Record<string, string> = {
        S: "1-4 gün", B: "1-2 gün", C: "3-8 gün", A: "8-14 gün",
    };
    return (
        <span className={`badge ${colors[t] || "badge-blue"}`}>
            {labels[t] || label || t} ({holdLabels[t] || ""})
        </span>
    );
}

/* ───── Sinyal Analiz Kartı (Accordion — Streamlit tarzında) ───── */
function SignalCard({ s, onTrack, tracking }: { s: Signal; onTrack: (s: Signal) => void; tracking: boolean }) {
    const [open, setOpen] = useState(false);

    const qOriginal = s.original_quality_score ?? s.quality_score;
    const qAdjusted = s.quality_score;
    const hasRegimePenalty = s.regime_multiplier != null && s.regime_multiplier < 1;
    // Use ORIGINAL score for quality assessment (before regime penalty)
    const qualityEmoji = qOriginal >= 80 ? "🔥" : qOriginal >= 65 ? "🟢" : "🟡";
    const qualityLabel = qOriginal >= 80 ? "Güçlü sinyal" : qOriginal >= 65 ? "İyi sinyal" : "Zayıf sinyal";

    // Build boosters list
    const boosters: string[] = [];
    if (s.higher_lows) boosters.push("Higher lows");
    if (s.macd_bullish) boosters.push("MACD bullish cross");
    if (s.high_rvol) boosters.push("High relative volume");
    if (s.gap_continuation) boosters.push("Gap continuation");
    if (s.rsi_divergence) boosters.push("RSI divergence");
    if (s.has_recent_news) boosters.push("Recent news catalyst");
    if (s.has_insider_buying) boosters.push("Insider buying detected");
    if (s.is_squeeze_candidate) boosters.push(`Squeeze candidate (SI: ${s.short_percent?.toFixed(1)}%)`);
    if (s.obv_accumulation) boosters.push("OBV accumulation (smart money)");
    if (s.obv_distribution) boosters.push("OBV distribution warning");

    return (
        <div className="glass-card" style={{ marginBottom: 12, overflow: "hidden" }}>
            {/* Accordion Header */}
            <button
                onClick={() => setOpen(o => !o)}
                style={{
                    width: "100%", padding: "14px 20px", background: "transparent", border: "none",
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    cursor: "pointer", color: "var(--text-primary)", gap: 12,
                }}
            >
                <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                    <span style={{ fontSize: "1.05rem", fontWeight: 800, color: "var(--accent)" }}>{s.ticker}</span>
                    <TypeLabel type={s.swing_type} label={s.swing_type_label} />
                    <span>{qualityEmoji} {qualityLabel}</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    {hasRegimePenalty ? (
                        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                            <QualityBadge score={qOriginal} />
                            <span style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>
                                (x{s.regime_multiplier?.toFixed(2)})
                            </span>
                        </span>
                    ) : (
                        <QualityBadge score={qOriginal} />
                    )}
                    {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </div>
            </button>

            {/* Accordion Content */}
            {open && (
                <div style={{ borderTop: "1px solid var(--border)", padding: "18px 20px" }}>
                    {/* Title row */}
                    <div style={{ fontWeight: 700, fontSize: "0.95rem", marginBottom: 14, color: "var(--text-primary)" }}>
                        {s.ticker} - {s.swing_type_label || s.swing_type} ({s.expected_hold_min || s.hold_days_min}-{s.expected_hold_max || s.hold_days_max} gün) | {qualityEmoji} {qualityLabel}
                    </div>

                    {/* Narrative / Setup */}
                    {s.narrative_text && (
                        <div style={{
                            background: "rgba(59,130,246,0.06)", border: "1px solid rgba(59,130,246,0.15)",
                            borderRadius: 10, padding: "14px 18px", marginBottom: 16,
                            fontSize: "0.85rem", lineHeight: 1.7, color: "var(--text-secondary)",
                        }}>
                            <div style={{ fontWeight: 700, marginBottom: 6, color: "var(--accent)", fontSize: "0.8rem" }}>
                                ✨ Setup:
                            </div>
                            {s.narrative_text}
                        </div>
                    )}

                    {/* Fiyat Seviyeleri */}
                    <div style={{
                        display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
                        gap: 10, marginBottom: 16,
                    }}>
                        <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: 8, padding: "10px 14px" }}>
                            <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>🎯 Entry</div>
                            <div style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--text-primary)", marginTop: 4 }}>${s.entry_price?.toFixed(2)}</div>
                        </div>
                        <div style={{ background: "rgba(239,68,68,0.06)", borderRadius: 8, padding: "10px 14px" }}>
                            <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>🛑 Stop</div>
                            <div style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--red)", marginTop: 4 }}>
                                ${s.stop_loss?.toFixed(2)} <span style={{ fontSize: "0.75rem", opacity: 0.7 }}>({s.stop_loss_pct?.toFixed(1)}%)</span>
                            </div>
                        </div>
                        <div style={{ background: "rgba(34,197,94,0.06)", borderRadius: 8, padding: "10px 14px" }}>
                            <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>🎯 T1</div>
                            <div style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--green)", marginTop: 4 }}>
                                ${s.target_1?.toFixed(2)} <span style={{ fontSize: "0.75rem", opacity: 0.7 }}>(+{s.target_1_pct?.toFixed(1)}%)</span>
                            </div>
                        </div>
                        <div style={{ background: "rgba(34,197,94,0.08)", borderRadius: 8, padding: "10px 14px" }}>
                            <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>🚀 T2</div>
                            <div style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--green)", marginTop: 4 }}>
                                ${s.target_2?.toFixed(2)} <span style={{ fontSize: "0.75rem", opacity: 0.7 }}>(+{s.target_2_pct?.toFixed(1)}%)</span>
                            </div>
                        </div>
                    </div>

                    {/* Risk/Reward */}
                    <div style={{
                        display: "flex", gap: 14, flexWrap: "wrap", marginBottom: 16,
                        fontSize: "0.82rem", color: "var(--text-secondary)",
                    }}>
                        <span>🔥 <strong>Risk/Ödül:</strong> T1 → 1:{s.risk_reward?.toFixed(1)} | T2 → 1:{s.risk_reward_t2?.toFixed(1)}</span>
                        {s.position_size != null && s.position_size > 0 && (
                            <span>📊 <strong>Pozisyon:</strong> {s.position_size} adet (${((s.position_size || 0) * (s.entry_price || 0)).toFixed(0)})</span>
                        )}
                        {s.risk_amount != null && s.risk_amount > 0 && (
                            <span>⚠️ <strong>Risk:</strong> ${(s.risk_amount || 0).toFixed(0)}</span>
                        )}
                    </div>

                    {/* Teknik Bilgiler */}
                    <div style={{
                        display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 16,
                        fontSize: "0.82rem", color: "var(--text-secondary)",
                    }}>
                        <span>📉 <strong>Teknik:</strong> RSI {s.rsi?.toFixed(0)} — {
                            (s.rsi || 50) <= 40 ? "aşırı satım" :
                                (s.rsi || 50) <= 55 ? "sağlıklı seviye" :
                                    (s.rsi || 50) <= 70 ? "yükseliyor" : "aşırı alım ⚠️"
                        }</span>
                        <span>📊 <strong>Vol Surge:</strong> {s.volume_surge?.toFixed(1)}x</span>
                        <span>📈 <strong>5 Gün:</strong> {s.five_day_return != null ? `${s.five_day_return >= 0 ? "+" : ""}${s.five_day_return.toFixed(1)}%` : "—"}</span>
                        {s.atr_percent != null && <span>📐 <strong>ATR%:</strong> {s.atr_percent.toFixed(1)}%</span>}
                    </div>

                    {/* Extras: Sector RS, Catalyst */}
                    {(s.sector_rs_score != null && s.sector_rs_score > 0 || s.total_catalyst_bonus != null && s.total_catalyst_bonus > 0) && (
                        <div style={{
                            display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 16,
                            fontSize: "0.82rem", color: "var(--text-secondary)",
                        }}>
                            {s.sector_rs_score != null && s.sector_rs_score > 0 && (
                                <span>⚡ <strong>Ekstra:</strong> Sektör performansı: +{s.sector_rs_score.toFixed(0)}{s.is_sector_leader ? " (Lider!)" : ""}</span>
                            )}
                            {s.total_catalyst_bonus != null && s.total_catalyst_bonus > 0 && (
                                <span>📰 <strong>Katalist bonus:</strong> +{s.total_catalyst_bonus.toFixed(0)}</span>
                            )}
                        </div>
                    )}

                    {/* Boosters */}
                    {boosters.length > 0 && (
                        <div style={{ marginBottom: 16, fontSize: "0.82rem" }}>
                            <span style={{ color: "var(--text-muted)", fontWeight: 600 }}>✅ Teknik onaylar: </span>
                            <span style={{ color: "var(--text-secondary)" }}>{boosters.join(", ")}</span>
                        </div>
                    )}

                    {/* OBV Smart Money indicator */}
                    {(s.obv_accumulation || s.obv_distribution) && (
                        <div style={{
                            background: s.obv_accumulation ? "rgba(34,197,94,0.08)" : "rgba(239,68,68,0.08)",
                            border: `1px solid ${s.obv_accumulation ? "rgba(34,197,94,0.2)" : "rgba(239,68,68,0.2)"}`,
                            borderRadius: 8, padding: "8px 14px", marginBottom: 12,
                            fontSize: "0.8rem", color: s.obv_accumulation ? "var(--green)" : "var(--red)",
                        }}>
                            {s.obv_accumulation
                                ? "📊 OBV Accumulation — Smart money birikim yapıyor (hacim artarken fiyat konsolide)"
                                : "📊 OBV Distribution — Satış baskısı var (hacim düşerken fiyat yükseliyor)"}
                        </div>
                    )}

                    {/* Regime warning */}
                    {s.regime_multiplier != null && s.regime_multiplier < 1 && (
                        <div style={{
                            background: s.market_regime === "BEAR" ? "rgba(239,68,68,0.08)" : "rgba(234,179,8,0.08)",
                            border: `1px solid ${s.market_regime === "BEAR" ? "rgba(239,68,68,0.2)" : "rgba(234,179,8,0.2)"}`,
                            borderRadius: 8, padding: "8px 14px", marginBottom: 12,
                            fontSize: "0.8rem", color: s.market_regime === "BEAR" ? "var(--red)" : "var(--yellow)",
                        }}>
                            {s.market_regime === "BEAR"
                                ? `🐻 Bear Market — Score %${((1 - s.regime_multiplier) * 100).toFixed(0)} düşürüldü. Pozisyon küçült!`
                                : `⚠️ Piyasa temkinli — Score %${((1 - s.regime_multiplier) * 100).toFixed(0)} düşürüldü.`}
                            {s.regime_confidence === "TENTATIVE" && " (henüz teyit edilmedi)"}
                        </div>
                    )}

                    {/* Öneri — dynamic, regime-aware */}
                    <div style={{
                        background: "rgba(168,85,247,0.06)", border: "1px solid rgba(168,85,247,0.15)",
                        borderRadius: 10, padding: "12px 16px", marginBottom: 16,
                        fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.8,
                    }}>
                        <div style={{ fontWeight: 700, color: "#a855f7", marginBottom: 6 }}>💡 Öneri:</div>
                        <div>⏳ <strong>{s.expected_hold_min || s.hold_days_min}-{s.expected_hold_max || s.hold_days_max} gün</strong> hold önerisi
                            {s.type_reason && <span> — {s.type_reason}</span>}
                        </div>
                        <div>
                            {s.swing_type === "S" && "⚡ Squeeze setup — Hızlı hareket bekleniyor, kademeli kâr al. Ani düşüşlere hazır ol!"}
                            {s.swing_type === "B" && "🏃 Momentum play — Trend tarafında kal, trailing stop ile kâr koru."}
                            {s.swing_type === "C" && "🎯 Erken giriş — Sabırlı ol, setup gelişiyor. Erken girişin avantajını kullan."}
                            {s.swing_type === "A" && "📊 Trend devamı — Mevcut trende uygun giriş. Planlı kâr al."}
                        </div>
                        {s.target_2 != null && s.target_1 != null && s.target_2 > s.target_1 && (
                            <div>🎯 T1'de %50 sat (${s.target_1.toFixed(2)}), kalan %50'yi T2'ye taşı (${s.target_2.toFixed(2)}). Stop → breakeven.</div>
                        )}
                        {(s.rsi || 50) > 70 && (
                            <div style={{ color: "var(--yellow)" }}>⚠️ RSI yüksek ({s.rsi?.toFixed(0)}) — kademeli kâr al, tam pozisyon girme.</div>
                        )}
                        {s.atr_percent != null && s.atr_percent > 10 && (
                            <div style={{ color: "var(--yellow)" }}>⚠️ Volatilite yüksek (ATR %{s.atr_percent.toFixed(1)}) — spread geniş olabilir, limit order kullan.</div>
                        )}
                        {s.market_regime === "BEAR" && (
                            <div style={{ color: "var(--red)" }}>🐻 Bear market — Pozisyon boyutunu %50 küçült, daha sıkı stop kullan.</div>
                        )}
                        {s.market_regime === "CAUTION" && (
                            <div style={{ color: "var(--yellow)" }}>⚠️ Piyasa temkinli — Normal pozisyonun %75'i ile gir.</div>
                        )}
                        {s.obv_accumulation && (
                            <div style={{ color: "var(--green)" }}>📊 Smart money birikim yapıyor — güçlü destek sinyali.</div>
                        )}
                        {s.obv_distribution && (
                            <div style={{ color: "var(--red)" }}>📊 OBV dağılım — satış baskısı var, dikkatli ol.</div>
                        )}
                        {s.volatility_warning && <div style={{ color: "var(--red)" }}>⚠️ Yüksek volatilite uyarısı — risk yönetimi kritik.</div>}
                    </div>

                    {/* Track Button */}
                    <button
                        className="btn-primary"
                        onClick={() => onTrack(s)}
                        disabled={tracking}
                        style={{ fontSize: "0.85rem", padding: "8px 20px" }}
                    >
                        {tracking ? <span className="spinner" style={{ width: 14, height: 14 }} /> : <Star size={14} />}
                        Track {s.ticker}
                    </button>
                </div>
            )}
        </div>
    );
}

/* ───── STORAGE KEYS ───── */
const STORAGE_KEY = "scannerResults";
const STORAGE_STATS_KEY = "scannerStats";

/** Bu taramada sunucunun uyguladığı ham skor eşiği + top_n + rejim özeti */
function AppliedThresholdPanel({
    stats,
    regimeLabel,
    regimeColor,
    regimeConfidence,
    regimeMultiplier,
    regimeDetectError,
    requestedMin,
    requestedTop,
}: {
    stats: Record<string, unknown>;
    regimeLabel: string;
    regimeColor: string;
    regimeConfidence: string;
    regimeMultiplier: number | undefined;
    regimeDetectError?: string;
    requestedMin: number;
    requestedTop: number;
}) {
    const reqMin =
        typeof stats.request_min_quality === "number" && Number.isFinite(stats.request_min_quality as number)
            ? (stats.request_min_quality as number)
            : requestedMin;
    const reqTop =
        typeof stats.request_top_n === "number" && Number.isFinite(stats.request_top_n as number)
            ? (stats.request_top_n as number)
            : requestedTop;

    const effMin = stats.effective_min_quality;
    const effTop = stats.effective_top_n;
    const hasNumeric =
        typeof effMin === "number" && Number.isFinite(effMin) && typeof effTop === "number" && Number.isFinite(effTop);

    if (!hasNumeric) {
        return (
            <div
                className="glass-card"
                style={{
                    padding: "14px 18px",
                    marginBottom: 20,
                    borderLeft: `3px solid ${regimeColor}`,
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    opacity: 0.85,
                }}
            >
                <SlidersHorizontal size={18} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
                <p style={{ margin: 0, fontSize: "0.82rem", color: "var(--text-muted)" }}>
                    Bu kayıt için uygulanan eşik ayrıntıları yok (eski tarama). Yeni bir tarama çalıştırın.
                </p>
            </div>
        );
    }

    const minN = effMin as number;
    const topN = effTop as number;

    const confTr =
        regimeConfidence === "TENTATIVE"
            ? "Teyitsiz"
            : regimeConfidence === "CONFIRMED"
              ? "Teyitli"
              : regimeConfidence || "—";

    return (
        <div
            className="glass-card"
            style={{
                marginBottom: 20,
                padding: 0,
                overflow: "hidden",
                borderLeft: `4px solid ${regimeColor}`,
                background:
                    "linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%)",
            }}
        >
            <div
                style={{
                    padding: "16px 20px",
                    display: "flex",
                    flexWrap: "wrap",
                    alignItems: "flex-start",
                    justifyContent: "space-between",
                    gap: 16,
                }}
            >
                <div style={{ display: "flex", gap: 14, minWidth: 0, flex: "1 1 280px" }}>
                    <div
                        style={{
                            width: 40,
                            height: 40,
                            borderRadius: 12,
                            background: `color-mix(in srgb, ${regimeColor} 18%, transparent)`,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            flexShrink: 0,
                        }}
                    >
                        <Sparkles size={20} style={{ color: regimeColor }} />
                    </div>
                    <div style={{ minWidth: 0 }}>
                        <div
                            style={{
                                fontSize: "0.68rem",
                                fontWeight: 700,
                                letterSpacing: "0.06em",
                                color: "var(--text-muted)",
                                textTransform: "uppercase",
                                marginBottom: 6,
                            }}
                        >
                            Uygulanan filtre
                        </div>
                        <p
                            style={{
                                margin: 0,
                                fontSize: "0.95rem",
                                fontWeight: 700,
                                color: "var(--text-primary)",
                                lineHeight: 1.45,
                            }}
                        >
                            Ham kalite skoru{" "}
                            <span style={{ color: "var(--accent)", fontVariantNumeric: "tabular-nums" }}>≥ {reqMin}</span>
                            {" · "}
                            En fazla{" "}
                            <span style={{ color: "var(--accent)", fontVariantNumeric: "tabular-nums" }}>{topN}</span>{" "}
                            sonuç gösteriliyor
                        </p>
                        <p style={{ margin: "8px 0 0", fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                            Rejim:{" "}
                            <strong style={{ color: regimeColor }}>{regimeLabel}</strong>
                            {regimeConfidence ? (
                                <span style={{ color: "var(--text-muted)", fontWeight: 500 }}>
                                    {" "}
                                    · {confTr}
                                </span>
                            ) : null}
                        </p>
                        {regimeDetectError ? (
                            <p
                                style={{
                                    margin: "8px 0 0",
                                    fontSize: "0.72rem",
                                    color: "var(--text-muted)",
                                    padding: "8px 12px",
                                    borderRadius: 8,
                                    background: "rgba(148,163,184,0.1)",
                                    border: "1px solid rgba(148,163,184,0.22)",
                                    wordBreak: "break-word",
                                }}
                            >
                                Rejim okunamadı:{" "}
                                {regimeDetectError.length > 220 ? `${regimeDetectError.slice(0, 220)}…` : regimeDetectError}
                            </p>
                        ) : null}
                        {topN < reqTop ? (
                            <p style={{ margin: "10px 0 0", fontSize: "0.75rem", color: "var(--text-muted)" }}>
                                Rejim: <strong style={{ color: regimeColor }}>{regimeLabel}</strong> · En fazla <strong>{topN}</strong> sonuç gösteriliyor.
                            </p>
                        ) : (
                            <p style={{ margin: "10px 0 0", fontSize: "0.75rem", color: "var(--text-muted)" }}>
                                Rejim: <strong style={{ color: regimeColor }}>{regimeLabel}</strong>.
                            </p>
                        )}
                    </div>
                </div>
                <div
                    style={{
                        display: "flex",
                        flexWrap: "wrap",
                        gap: 8,
                        alignItems: "center",
                        justifyContent: "flex-end",
                    }}
                >
                    <span
                        style={{
                            fontSize: "0.72rem",
                            fontWeight: 700,
                            padding: "6px 12px",
                            borderRadius: 999,
                            background: `color-mix(in srgb, ${regimeColor} 12%, transparent)`,
                            color: regimeColor,
                            border: `1px solid color-mix(in srgb, ${regimeColor} 35%, transparent)`,
                            fontVariantNumeric: "tabular-nums",
                        }}
                    >
                        min ≥ {reqMin}
                    </span>
                    <span
                        style={{
                            fontSize: "0.72rem",
                            fontWeight: 700,
                            padding: "6px 12px",
                            borderRadius: 999,
                            background: "rgba(255,255,255,0.06)",
                            color: "var(--text-secondary)",
                            border: "1px solid var(--border)",
                            fontVariantNumeric: "tabular-nums",
                        }}
                    >
                        top {topN}
                    </span>
                </div>
            </div>
        </div>
    );
}

function saveScanResults(data: { signals: Signal[]; stats: Record<string, unknown>; market_regime: string }) {
    try {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch { /* quota exceeded — ignore */ }
}

function loadScanResults(): { signals: Signal[]; stats: Record<string, unknown>; market_regime: string } | null {
    try {
        const raw = sessionStorage.getItem(STORAGE_KEY);
        return raw ? JSON.parse(raw) : null;
    } catch { return null; }
}

/* ───── PAGE ───── */
export default function ScannerPage() {
    const { isScanning, poll, scanError: jobScanError, startBackgroundScan, dismissScanFeedback } = useScannerJob();

    // Scan params
    const [minQuality, setMinQuality] = useState(65);
    const [topN, setTopN] = useState(10);
    const [portfolioValue, setPortfolioValue] = useState(10000);

    // Auto-track settings
    const [autoTrackEnabled, setAutoTrackEnabled] = useState(true);
    const [autoTrackMinQuality, setAutoTrackMinQuality] = useState(65);
    const [autoTrackOpen, setAutoTrackOpen] = useState(false);

    // Results
    const [result, setResult] = useState<{ signals: Signal[]; stats: Record<string, unknown>; market_regime: string } | null>(null);
    const [error, setError] = useState("");
    const [adding, setAdding] = useState<string | null>(null);
    const [msg, setMsg] = useState("");
    const [autoTrackResult, setAutoTrackResult] = useState<{ tracked: string[]; skipped: string[] } | null>(null);

    // Load persisted auto-track settings + scan results on mount
    useEffect(() => {
        try {
            const saved = localStorage.getItem("autoTrack");
            if (saved) {
                const { enabled, minQuality: mq } = JSON.parse(saved);
                if (enabled !== undefined) setAutoTrackEnabled(enabled);
                if (mq !== undefined) setAutoTrackMinQuality(mq);
            }
        } catch { /* ignore */ }

        // Restore last scan results from sessionStorage
        const cached = loadScanResults();
        if (cached) setResult(cached);
    }, []);

    useEffect(() => {
        try {
            localStorage.setItem("autoTrack", JSON.stringify({ enabled: autoTrackEnabled, minQuality: autoTrackMinQuality }));
        } catch { /* ignore */ }
    }, [autoTrackEnabled, autoTrackMinQuality]);

    // Arka planda biten scan — sonuç + isteğe bağlı auto-track
    useEffect(() => {
        const onComplete = (e: Event) => {
            const detail = (e as CustomEvent<{ signals: Signal[]; stats: Record<string, unknown>; market_regime: string }>).detail;
            if (!detail) return;
            setResult(detail);
            saveScanResults(detail);
            setMsg("");
            setError("");

            let enabled = true;
            let mq = 65;
            try {
                const raw = localStorage.getItem("autoTrack");
                if (raw) {
                    const j = JSON.parse(raw) as { enabled?: boolean; minQuality?: number };
                    if (j.enabled === false) enabled = false;
                    if (j.minQuality != null) mq = j.minQuality;
                }
            } catch { /* ignore */ }

            if (!enabled || !detail.signals?.length) return;
            const qualifying = detail.signals.filter(
                (s) => (s.original_quality_score ?? s.quality_score) >= mq
            );
            if (!qualifying.length) return;

            void (async () => {
                const tracked: string[] = [];
                const skipped: string[] = [];
                await Promise.all(
                    qualifying.map(async (s) => {
                        try {
                            const res = await trackSignal(s);
                            if (res?.status === "added") tracked.push(s.ticker);
                            else skipped.push(s.ticker);
                        } catch {
                            skipped.push(s.ticker);
                        }
                    })
                );
                setAutoTrackResult({ tracked, skipped });
            })();
        };
        window.addEventListener(SCAN_COMPLETE_EVENT, onComplete);
        return () => window.removeEventListener(SCAN_COMPLETE_EVENT, onComplete);
    }, []);

    const scan = async () => {
        setError(""); setMsg(""); setAutoTrackResult(null);
        dismissScanFeedback();
        await startBackgroundScan({
            min_quality: minQuality,
            top_n: topN,
            portfolio_value: portfolioValue,
        });
    };

    const handleTrack = async (s: Signal) => {
        setAdding(s.ticker);
        try {
            const res = await trackSignal(s);
            if (res?.status === "added") {
                setMsg(`✅ ${s.ticker} paper trade'e eklendi! (ID: ${res.trade_id})`);
            } else {
                setMsg(`⏭️ ${s.ticker} zaten takipte (duplicate)`);
            }
        } catch {
            setMsg(`❌ ${s.ticker} eklenemedi`);
        } finally {
            setAdding(null);
        }
    };

    const regime = result?.market_regime || "";
    const regimeColor =
        regime === "BULL" ? "var(--green)"
        : regime === "BEAR" ? "var(--red)"
        : regime === "CAUTION" ? "var(--yellow)"
        : regime === "UNKNOWN" ? "var(--text-muted)"
        : "var(--text-muted)";
    const regimeLabel =
        regime === "BULL" ? "BULL"
        : regime === "BEAR" ? "BEAR"
        : regime === "CAUTION" ? "CAUTION"
        : regime === "UNKNOWN" ? "BİLİNMİYOR"
        : regime || "—";
    // Rejim çarpanı artık skora uygulanmıyor; UI’da kullanılmıyor.
    const regimeMultiplier = (result?.stats as Record<string, number>)?.regime_multiplier;
    const regimeConfidence = (result?.stats as Record<string, string>)?.regime_confidence || "";
    const regimeDetectError = (result?.stats as Record<string, string>)?.regime_detect_error;

    return (
        <div>
            <h1 className="page-title gradient-text">SmallCap Scanner</h1>
            <p className="page-subtitle">
                AI-powered momentum signals · SmallCap universe ·{" "}
                <Link href="/how-it-works" style={{ color: "var(--accent)", fontWeight: 600 }}>
                    Akış ve ayar rehberi
                </Link>
            </p>

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
                <button className="btn-primary" onClick={scan} disabled={isScanning}>
                    {isScanning ? <span className="spinner" /> : <Search size={15} />}
                    {isScanning ? "Scanning…" : "Run Scan"}
                </button>
            </div>

            {isScanning && poll && (
                <div className="glass-card" style={{ padding: 16, marginBottom: 16 }}>
                    <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: 8 }}>
                        Tarama arka planda sürüyor — başka sayfaya gidebilirsiniz.
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
                        <div style={{ flex: 1, height: 8, borderRadius: 4, background: "rgba(255,255,255,0.08)", overflow: "hidden" }}>
                            <div style={{ height: "100%", width: `${Math.round(poll.progress)}%`, background: "var(--accent)", transition: "width 0.3s ease" }} />
                        </div>
                        <span style={{ fontWeight: 700, color: "var(--accent)", minWidth: 40 }}>{Math.round(poll.progress)}%</span>
                    </div>
                    {poll.message ? (
                        <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>{poll.message}</div>
                    ) : null}
                </div>
            )}

            {/* Auto-Track Ayarları */}
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
                            <div>
                                <label style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer", fontSize: "0.875rem", fontWeight: 600, userSelect: "none" }}>
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
                            </div>
                        </div>
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

            {/* Error */}
            {(error || jobScanError) && (
                <div style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 10, padding: "12px 18px", color: "var(--red)", marginBottom: 16, display: "flex", gap: 10, alignItems: "center" }}>
                    <AlertTriangle size={16} /> {error || jobScanError}
                </div>
            )}

            {/* Success / info messages */}
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
                </div>
            )}

            {/* Results (önceki tarama görünür kalır; yeni tarama sürerken üstte progress var) */}
            {result && (
                <>
                    {/* Stats bar */}
                    <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
                        <div className="metric-card" style={{ padding: "14px 20px", flex: 1, minWidth: 120 }}>
                            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>Signals</div>
                            <div style={{ fontSize: "1.4rem", fontWeight: 800 }}>{result.signals.length}</div>
                        </div>
                        <div className="metric-card" style={{ padding: "14px 20px", flex: 1, minWidth: 160 }}>
                            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>Market Regime</div>
                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                <span style={{ fontSize: "1rem", fontWeight: 800, color: regimeColor }}>
                                    {regimeLabel}
                                </span>
                                {regime === "UNKNOWN" ? (
                                    <span style={{ fontSize: "0.6rem", color: "var(--text-muted)", fontWeight: 500 }}>
                                        Veri yok
                                    </span>
                                ) : regimeConfidence === "TENTATIVE" ? (
                                    <span style={{ fontSize: "0.6rem", color: "var(--text-muted)", background: "rgba(255,255,255,0.06)", padding: "2px 6px", borderRadius: 4, fontWeight: 500 }}>
                                        Unconfirmed
                                    </span>
                                ) : null}
                                {/* Rejim çarpanı artık skora uygulanmıyor; sadece bilgi gösteriyoruz. */}
                            </div>
                        </div>
                        <div className="metric-card" style={{ padding: "14px 20px", flex: 1, minWidth: 120 }}>
                            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>Scanned</div>
                            <div style={{ fontSize: "1.4rem", fontWeight: 800 }}>{(result.stats as Record<string, number>).stocks_scanned || "—"}</div>
                        </div>
                        <div className="metric-card" style={{ padding: "14px 20px", flex: 1, minWidth: 120 }}>
                            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>Raw Signals</div>
                            <div style={{ fontSize: "1.4rem", fontWeight: 800 }}>{(result.stats as Record<string, number>).raw_signals || "—"}</div>
                        </div>
                    </div>

                    {(() => {
                        const rejectLine = formatRejectSummary(result.stats as Record<string, unknown>);
                        if (!rejectLine) return null;
                        return (
                            <div
                                style={{
                                    fontSize: "0.8rem",
                                    color: "var(--text-muted)",
                                    marginBottom: 16,
                                    lineHeight: 1.5,
                                    padding: "10px 14px",
                                    background: "rgba(255,255,255,0.03)",
                                    border: "1px solid var(--border)",
                                    borderRadius: 10,
                                }}
                            >
                                <span style={{ color: "var(--text-secondary)", fontWeight: 600 }}>Ham sinyal öncesi elenmeler: </span>
                                {rejectLine}
                            </div>
                        );
                    })()}

                    <AppliedThresholdPanel
                        stats={result.stats}
                        regimeLabel={regimeLabel}
                        regimeColor={regimeColor}
                        regimeConfidence={regimeConfidence}
                        regimeMultiplier={regimeMultiplier}
                        regimeDetectError={regimeDetectError}
                        requestedMin={minQuality}
                        requestedTop={topN}
                    />

                    {/* Signal cards */}
                    {result.signals.length === 0 ? (
                        <div className="glass-card" style={{ padding: 48, textAlign: "center", color: "var(--text-muted)" }}>
                            <TrendingUp size={48} style={{ marginBottom: 12, opacity: 0.3 }} />
                            <div>Mevcut filtrelerle sinyal bulunamadı. Min Quality&apos;yi düşürmeyi deneyin.</div>
                        </div>
                    ) : (
                        <div>
                            <h2 style={{ fontSize: "1.05rem", fontWeight: 700, marginBottom: 14, display: "flex", alignItems: "center", gap: 8 }}>
                                📊 Sinyal Analizleri
                                <span style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontWeight: 400 }}>
                                    Her sinyal için detaylı yorum ve öneri
                                </span>
                            </h2>
                            {result.signals.map((s, i) => (
                                <SignalCard
                                    key={`${s.ticker}-${i}`}
                                    s={s}
                                    onTrack={handleTrack}
                                    tracking={adding === s.ticker}
                                />
                            ))}
                        </div>
                    )}
                </>
            )}

            {/* Empty state */}
            {!result && !isScanning && (
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
