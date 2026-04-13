"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getSmallcapScanHistory, getSmallcapScanHistoryRun } from "@/lib/api";
import type { SmallcapScanRunMeta, SmallcapScanRunDetail, Signal } from "@/lib/api";
import { Calendar, RefreshCw, Search, Shield, Target, TrendingUp } from "lucide-react";

function fmtTs(ts?: string | null): string {
    if (!ts) return "—";
    // Prefer YYYY-MM-DD HH:MM from ISO-ish strings
    const s = String(ts);
    if (s.includes("T")) return s.replace("T", " ").slice(0, 16);
    return s.slice(0, 16);
}

function regimeColor(regime?: string | null): string {
    if (regime === "BULL") return "var(--green)";
    if (regime === "BEAR") return "var(--red)";
    if (regime === "CAUTION") return "var(--yellow)";
    return "var(--text-muted)";
}

function QualityPill({ score }: { score: number }) {
    const cls = score >= 80 ? "badge-green" : score >= 65 ? "badge-blue" : "badge-yellow";
    return <span className={`badge ${cls}`}>{score.toFixed(0)}</span>;
}

function SignalRow({ s }: { s: Signal }) {
    const catalysts: string[] = [];
    if (s.total_catalyst_bonus != null && s.total_catalyst_bonus > 0) catalysts.push(`Catalyst +${s.total_catalyst_bonus.toFixed(0)}`);
    if (s.is_sector_leader) catalysts.push("Sector leader");
    if (s.is_squeeze_candidate) catalysts.push(`Squeeze (SI ${s.short_percent?.toFixed(1)}%)`);
    if (s.has_insider_buying) catalysts.push("Insider");
    if (s.has_recent_news) catalysts.push("News");
    const extras = catalysts.length ? catalysts.join(" · ") : "—";

    return (
        <tr>
            <td style={{ fontWeight: 800, color: "var(--accent)" }}>{s.ticker}</td>
            <td><span className={`badge ${s.swing_type === "S" ? "badge-red" : s.swing_type === "B" ? "badge-yellow" : s.swing_type === "C" ? "badge-green" : "badge-blue"}`}>Type {s.swing_type || "A"}</span></td>
            <td><QualityPill score={Number(s.quality_score || 0)} /></td>
            <td style={{ color: "var(--text-muted)" }}>${s.entry_price?.toFixed(2)}</td>
            <td style={{ color: "var(--red)" }}>${s.stop_loss?.toFixed(2)}</td>
            <td style={{ color: "var(--green)" }}>${(s.target_2 ?? s.target_1 ?? s.target)?.toFixed(2)}</td>
            <td style={{ color: "var(--text-secondary)" }}>{extras}</td>
        </tr>
    );
}

function RunDetail({ run }: { run: SmallcapScanRunDetail }) {
    const signals = (run.signals || []) as Signal[];
    const stats = run.stats || {};
    const stockScanned = (stats as Record<string, unknown>)["stocks_scanned"];
    const stockWithData = (stats as Record<string, unknown>)["stocks_with_data"];
    const rawSignals = (stats as Record<string, unknown>)["raw_signals"];
    const filteredSignals = (stats as Record<string, unknown>)["filtered_signals"];
    const regime = run.market_regime || "UNKNOWN";

    return (
        <div className="glass-card" style={{ padding: 18 }}>
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12, flexWrap: "wrap", marginBottom: 14 }}>
                <div style={{ minWidth: 0 }}>
                    <div style={{ fontWeight: 900, letterSpacing: "-0.02em", fontSize: "1.05rem" }}>
                        Run #{run.id}
                        <span style={{ marginLeft: 10, fontSize: "0.78rem", fontWeight: 700, color: regimeColor(regime) }}>
                            {regime}{run.regime_confidence ? ` (${run.regime_confidence})` : ""}
                        </span>
                    </div>
                    <div style={{ display: "flex", gap: 10, flexWrap: "wrap", color: "var(--text-muted)", fontSize: "0.78rem", marginTop: 6 }}>
                        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><Calendar size={14} /> {fmtTs(run.created_at)}</span>
                        {run.job_id ? <span>job {String(run.job_id).slice(0, 8)}…</span> : null}
                    </div>
                </div>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap", fontSize: "0.78rem" }}>
                    <span className="badge badge-blue"><Shield size={12} style={{ marginRight: 6 }} /> min {run.effective_min_quality ?? run.request_min_quality ?? "—"}</span>
                    <span className="badge badge-purple"><Target size={12} style={{ marginRight: 6 }} /> top {run.effective_top_n ?? run.request_top_n ?? "—"}</span>
                    {run.portfolio_value ? <span className="badge badge-yellow"><TrendingUp size={12} style={{ marginRight: 6 }} /> ${Number(run.portfolio_value).toFixed(0)}</span> : null}
                </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: 10, marginBottom: 14 }}>
                <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid var(--border-muted)", borderRadius: 10, padding: "10px 12px" }}>
                    <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase" }}>Universe</div>
                    <div style={{ fontSize: "1.05rem", fontWeight: 900, marginTop: 4 }}>{String(stockScanned ?? "—")}</div>
                    <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>with data {String(stockWithData ?? "—")}</div>
                </div>
                <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid var(--border-muted)", borderRadius: 10, padding: "10px 12px" }}>
                    <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase" }}>Signals</div>
                    <div style={{ fontSize: "1.05rem", fontWeight: 900, marginTop: 4 }}>{String(filteredSignals ?? signals.length)}</div>
                    <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>raw {String(rawSignals ?? "—")}</div>
                </div>
            </div>

            {signals.length === 0 ? (
                <div style={{ padding: 30, textAlign: "center", color: "var(--text-muted)" }}>
                    Bu run’da kaydedilmiş sinyal yok.
                </div>
            ) : (
                <div style={{ overflowX: "auto" }}>
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Ticker</th>
                                <th>Type</th>
                                <th>Quality</th>
                                <th>Entry</th>
                                <th>Stop</th>
                                <th>T2</th>
                                <th>Context</th>
                            </tr>
                        </thead>
                        <tbody>
                            {signals.map((s) => <SignalRow key={`${s.ticker}-${s.entry_price}`} s={s} />)}
                        </tbody>
                    </table>
                </div>
            )}

            {signals.some((s) => !!s.narrative_text) ? (
                <div style={{ marginTop: 14 }}>
                    <div style={{ fontWeight: 900, marginBottom: 10 }}>Sinyal notları (narrative)</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                        {signals.filter((s) => !!s.narrative_text).map((s) => (
                            <div key={`n-${s.ticker}`} style={{ background: "rgba(59,130,246,0.06)", border: "1px solid rgba(59,130,246,0.15)", borderRadius: 10, padding: "12px 14px" }}>
                                <div style={{ display: "flex", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
                                    <div style={{ fontWeight: 900, color: "var(--accent)" }}>{s.ticker}</div>
                                    <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{s.narrative_headline || ""}</div>
                                </div>
                                <div style={{ marginTop: 8, fontSize: "0.85rem", lineHeight: 1.7, color: "var(--text-secondary)", whiteSpace: "pre-wrap" }}>
                                    {s.narrative_text}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            ) : null}
        </div>
    );
}

export default function ScannerHistoryPage() {
    const [selectedId, setSelectedId] = useState<number | null>(null);
    const [q, setQ] = useState("");

    const runsQuery = useQuery({
        queryKey: ["scannerHistory", "runs"],
        queryFn: () => getSmallcapScanHistory(60),
        staleTime: 30 * 1000,
    });

    const runs: SmallcapScanRunMeta[] = runsQuery.data?.runs || [];

    const filteredRuns = useMemo(() => {
        const qq = q.trim().toUpperCase();
        if (!qq) return runs;
        return runs.filter((r) => {
            const txt = `${r.market_regime || ""} ${r.regime_confidence || ""} ${r.created_at || ""} ${r.job_id || ""}`.toUpperCase();
            return txt.includes(qq) || String(r.id).includes(qq);
        });
    }, [runs, q]);

    const chosenId = selectedId ?? (filteredRuns[0]?.id ?? null);

    const runQuery = useQuery({
        queryKey: ["scannerHistory", "run", chosenId],
        queryFn: () => (chosenId ? getSmallcapScanHistoryRun(chosenId) : Promise.resolve(null)),
        enabled: !!chosenId,
        staleTime: 30 * 1000,
    });

    return (
        <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, flexWrap: "wrap", marginBottom: 18 }}>
                <div>
                    <h1 className="page-title gradient-text">Geçmiş Swing Trade Sinyal Analizleri</h1>
                    <p className="page-subtitle">Scanner’ın ürettiği run’lar server-side saklanır · “Neyi, ne zaman, neden seçtik?”</p>
                </div>
                <button
                    className="btn-secondary"
                    onClick={() => runsQuery.refetch()}
                    disabled={runsQuery.isFetching}
                >
                    {runsQuery.isFetching ? <span className="spinner" style={{ width: 14, height: 14 }} /> : <RefreshCw size={14} />}
                    Yenile
                </button>
            </div>

            <div className="glass-card" style={{ padding: 14, marginBottom: 14, display: "flex", alignItems: "center", gap: 10 }}>
                <Search size={16} style={{ color: "var(--text-muted)" }} />
                <input
                    value={q}
                    onChange={(e) => setQ(e.target.value)}
                    placeholder="Run ID / regime / tarih…"
                    style={{ width: "100%", background: "transparent", border: "none", outline: "none", color: "var(--text-primary)" }}
                />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "minmax(260px, 380px) 1fr", gap: 14, alignItems: "start" }}>
                <div className="glass-card" style={{ overflow: "hidden" }}>
                    <div style={{ padding: "12px 14px", borderBottom: "1px solid var(--border)" }}>
                        <div style={{ fontWeight: 900 }}>Run listesi</div>
                        <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
                            {runsQuery.isLoading ? "Yükleniyor…" : `${filteredRuns.length} kayıt`}
                        </div>
                    </div>
                    <div style={{ maxHeight: "68vh", overflowY: "auto" }}>
                        {runsQuery.isError ? (
                            <div style={{ padding: 16, color: "var(--red)" }}>History alınamadı.</div>
                        ) : filteredRuns.length === 0 ? (
                            <div style={{ padding: 16, color: "var(--text-muted)" }}>Kayıt bulunamadı.</div>
                        ) : (
                            filteredRuns.map((r) => {
                                const active = (chosenId === r.id);
                                return (
                                    <button
                                        key={r.id}
                                        type="button"
                                        onClick={() => setSelectedId(r.id)}
                                        style={{
                                            width: "100%",
                                            textAlign: "left",
                                            padding: "12px 14px",
                                            background: active ? "rgba(59,130,246,0.10)" : "transparent",
                                            border: "none",
                                            borderBottom: "1px solid var(--border-muted)",
                                            cursor: "pointer",
                                            color: "var(--text-primary)",
                                        }}
                                    >
                                        <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
                                            <div style={{ fontWeight: 900 }}>#{r.id}</div>
                                            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{fmtTs(r.created_at)}</div>
                                        </div>
                                        <div style={{ marginTop: 6, display: "flex", gap: 8, flexWrap: "wrap", fontSize: "0.72rem" }}>
                                            <span style={{ color: regimeColor(r.market_regime) }}>{r.market_regime || "UNKNOWN"}</span>
                                            {r.regime_confidence ? <span style={{ color: "var(--text-muted)" }}>{r.regime_confidence}</span> : null}
                                            <span style={{ color: "var(--text-muted)" }}>min {r.effective_min_quality ?? r.request_min_quality ?? "—"}</span>
                                            <span style={{ color: "var(--text-muted)" }}>top {r.effective_top_n ?? r.request_top_n ?? "—"}</span>
                                        </div>
                                    </button>
                                );
                            })
                        )}
                    </div>
                </div>

                <div>
                    {runQuery.isLoading ? (
                        <div className="glass-card" style={{ padding: 40, textAlign: "center" }}><span className="spinner" /></div>
                    ) : runQuery.isError ? (
                        <div className="glass-card" style={{ padding: 16, color: "var(--red)" }}>Run detayı alınamadı.</div>
                    ) : runQuery.data ? (
                        <RunDetail run={runQuery.data as SmallcapScanRunDetail} />
                    ) : (
                        <div className="glass-card" style={{ padding: 18, color: "var(--text-muted)" }}>
                            Bir run seçin.
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

