"use client";
import { useState, useCallback } from "react";
import { runBacktest } from "@/lib/api";
import type { BacktestResult, BacktestMetrics } from "@/lib/api";
import {
    ComposedChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Bar, Cell, ReferenceLine,
} from "recharts";
import { FlaskConical, TrendingUp, TrendingDown, AlertCircle, Play } from "lucide-react";

const PERIOD_OPTIONS = [
    { label: "1 Ay", days: 30 },
    { label: "2 Ay", days: 60 },
    { label: "3 Ay", days: 90 },
    { label: "6 Ay", days: 180 },
];

const EXIT_LABELS: Record<string, string> = {
    STOPPED: "🛑 Stop Loss",
    TARGET: "🎯 Target Hit",
    TIMEOUT: "⏰ Timeout",
    FORCED: "🔚 Backtest Sonu",
};

function MetricBox({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
    return (
        <div className="metric-card" style={{ padding: "14px 18px" }}>
            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase", marginBottom: 6 }}>{label}</div>
            <div style={{ fontSize: "1.6rem", fontWeight: 800, color: color || "var(--text-primary)" }}>{value}</div>
            {sub && <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: 3 }}>{sub}</div>}
        </div>
    );
}

const CTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null;
    return (
        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, padding: "8px 14px", fontSize: "0.78rem" }}>
            <div style={{ color: "var(--text-muted)", marginBottom: 4, fontWeight: 600 }}>{label}</div>
            {payload.map((p: any, i: number) => (
                <div key={i} style={{ color: p.color, display: "flex", gap: 8 }}>
                    <span>{p.name}:</span>
                    <strong>{typeof p.value === "number" ? p.value.toFixed(2) : p.value}</strong>
                </div>
            ))}
        </div>
    );
};

export default function BacktestPage() {
    const [periodDays, setPeriodDays] = useState(90);
    const [capital, setCapital] = useState(10000);
    const [maxConcurrent, setMaxConcurrent] = useState(3);
    const [tickerInput, setTickerInput] = useState("AEHR,AXTI,VELO,FSLY,NMRA,NVCR,SGRY,MLTX,FJET,WIN");
    const [useFinviz, setUseFinviz] = useState(false);
    const [result, setResult] = useState<BacktestResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const run = useCallback(async () => {
        setLoading(true);
        setError("");
        setResult(null);
        try {
            const tickers = useFinviz
                ? undefined
                : tickerInput.split(",").map(t => t.trim().toUpperCase()).filter(Boolean);

            const data = await runBacktest({
                period_days: periodDays,
                initial_capital: capital,
                max_concurrent: maxConcurrent,
                tickers,
            });

            if (data.error) throw new Error(data.error);
            setResult(data);
        } catch (e: any) {
            setError(e?.response?.data?.detail || e?.message || "Backtest başlatılamadı.");
        } finally {
            setLoading(false);
        }
    }, [periodDays, capital, maxConcurrent, tickerInput, useFinviz]);

    const m: BacktestMetrics | undefined = result?.metrics;
    const winRatePct = m ? (m.win_rate * 100).toFixed(1) : null;
    const returnPct = m ? (m.total_return * 100).toFixed(1) : null;

    // Equity curve chart data
    const eqData = result?.equity_curve?.map(e => ({
        date: e.date?.slice(5),           // MM-DD
        value: e.portfolio_value,
    })) ?? [];

    // Trade P&L bar data (last 20)
    const tradeData = result?.trades?.slice(-20).map(t => ({
        ticker: t.ticker,
        pnl: t.pnl_dollar,
        pnlPct: t.pnl_pct,
    })) ?? [];

    return (
        <div>
            <h1 className="page-title gradient-text">SmallCap Backtest</h1>
            <p className="page-subtitle">Walk-forward geçmiş simülasyon · Gerçek SmallCap sinyalleri</p>

            {/* Params */}
            <div className="glass-card" style={{ padding: 20, marginBottom: 20 }}>
                <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-end" }}>

                    {/* Period */}
                    <div>
                        <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 6 }}>TEST PERİYODU</label>
                        <div style={{ display: "flex", gap: 6 }}>
                            {PERIOD_OPTIONS.map(p => (
                                <button key={p.days} onClick={() => setPeriodDays(p.days)}
                                    style={{
                                        padding: "7px 12px", borderRadius: 7, border: "1px solid",
                                        borderColor: periodDays === p.days ? "var(--accent)" : "var(--border)",
                                        background: periodDays === p.days ? "rgba(59,130,246,0.15)" : "transparent",
                                        color: periodDays === p.days ? "var(--accent)" : "var(--text-secondary)",
                                        cursor: "pointer", fontSize: "0.78rem", fontWeight: 600, transition: "all 0.15s",
                                    }}>
                                    {p.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Capital */}
                    <div style={{ minWidth: 140 }}>
                        <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 6 }}>BAŞLANGIÇ SERMAYE ($)</label>
                        <input className="input" type="number" step={1000} value={capital}
                            onChange={e => setCapital(Number(e.target.value))} />
                    </div>

                    {/* Max concurrent */}
                    <div style={{ minWidth: 120 }}>
                        <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 6 }}>MAKS EŞ ZAMANLI</label>
                        <input className="input" type="number" min={1} max={10} value={maxConcurrent}
                            onChange={e => setMaxConcurrent(Number(e.target.value))} />
                    </div>

                    {/* Run button */}
                    <button className="btn-primary" onClick={run} disabled={loading}
                        style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 20px" }}>
                        {loading ? <span className="spinner" style={{ width: 14, height: 14 }} /> : <Play size={14} />}
                        {loading ? "Çalışıyor…" : "Backtest Başlat"}
                    </button>
                </div>

                {/* Ticker source */}
                <div style={{ marginTop: 16, display: "flex", gap: 18, alignItems: "center", flexWrap: "wrap" }}>
                    <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: "0.8rem" }}>
                        <input type="radio" checked={!useFinviz} onChange={() => setUseFinviz(false)} />
                        Manuel Hisse Listesi
                    </label>
                    <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: "0.8rem" }}>
                        <input type="radio" checked={useFinviz} onChange={() => setUseFinviz(true)} />
                        Finviz Universe (daha yavaş ~1-2 dk)
                    </label>
                </div>

                {!useFinviz && (
                    <div style={{ marginTop: 12 }}>
                        <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, display: "block", marginBottom: 6 }}>HISSE LİSTESİ (virgülle)</label>
                        <input className="input" value={tickerInput}
                            onChange={e => setTickerInput(e.target.value)}
                            placeholder="AEHR, AXTI, VELO, ..."
                            style={{ width: "100%", maxWidth: 500 }} />
                    </div>
                )}

                {loading && (
                    <div style={{ marginTop: 14, display: "flex", alignItems: "center", gap: 10, color: "var(--text-secondary)", fontSize: "0.82rem" }}>
                        <span className="spinner" style={{ width: 14, height: 14 }} />
                        Walk-forward simülasyon çalışıyor... Bu işlem 1-3 dakika sürebilir.
                    </div>
                )}
            </div>

            {/* Error */}
            {error && (
                <div style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 10, padding: "12px 18px", color: "var(--red)", marginBottom: 16, display: "flex", gap: 10, alignItems: "center" }}>
                    <AlertCircle size={16} /> {error}
                </div>
            )}

            {/* Results */}
            {result && m && (
                <>
                    {/* Period info */}
                    <div style={{ fontSize: "0.82rem", color: "var(--text-muted)", marginBottom: 16 }}>
                        📅 {result.start_date} → {result.end_date} · {result.tickers_used.length} hisse · ${result.initial_capital.toLocaleString()} başlangıç
                    </div>

                    {m.total_trades === 0 ? (
                        <div className="glass-card" style={{ padding: 40, textAlign: "center", color: "var(--text-secondary)" }}>
                            ⚠️ Bu dönemde hiç sinyal üretilmedi. Daha uzun periyot veya farklı hisseler deneyin.
                        </div>
                    ) : (
                        <>
                            {/* Key metrics */}
                            <div className="metrics-grid" style={{ marginBottom: 20 }}>
                                <MetricBox label="Toplam Trade" value={`${m.total_trades}`} sub={`${m.winning_trades}W / ${m.losing_trades}L`} />
                                <MetricBox label="Win Rate" value={`${winRatePct}%`}
                                    color={m.win_rate >= 0.5 ? "var(--green)" : "var(--red)"} />
                                <MetricBox label="Profit Factor" value={`${m.profit_factor?.toFixed(2) ?? "—"}`}
                                    color={m.profit_factor > 1.5 ? "var(--green)" : "var(--text-primary)"} />
                                <MetricBox label="Toplam P/L"
                                    value={`${m.total_pnl_dollar >= 0 ? "+" : ""}$${m.total_pnl_dollar?.toFixed(0)}`}
                                    sub={`${returnPct}% getiri`}
                                    color={m.total_pnl_dollar >= 0 ? "var(--green)" : "var(--red)"} />
                                <MetricBox label="Max Drawdown" value={`${m.max_drawdown?.toFixed(1)}%`}
                                    color="var(--red)" />
                                <MetricBox label="Ort. Tutma Süresi" value={`${m.avg_hold_days?.toFixed(1)} gün`} />
                            </div>

                            {/* Win/Loss analytics */}
                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
                                <div className="glass-card" style={{ padding: 18 }}>
                                    <div style={{ fontWeight: 700, marginBottom: 12, fontSize: "0.85rem" }}>💰 Kazanç / Kayıp Analizi</div>
                                    <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: "0.82rem" }}>
                                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                                            <span style={{ color: "var(--text-muted)" }}>Ort. Kazanç</span>
                                            <span style={{ color: "var(--green)", fontWeight: 700 }}>+{m.avg_win_pct?.toFixed(1)}% (${m.avg_win_dollar?.toFixed(0)})</span>
                                        </div>
                                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                                            <span style={{ color: "var(--text-muted)" }}>Ort. Kayıp</span>
                                            <span style={{ color: "var(--red)", fontWeight: 700 }}>{m.avg_loss_pct?.toFixed(1)}% (${m.avg_loss_dollar?.toFixed(0)})</span>
                                        </div>
                                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                                            <span style={{ color: "var(--text-muted)" }}>Risk/Reward</span>
                                            <span style={{ fontWeight: 700 }}>
                                                {m.avg_loss_pct ? `${Math.abs(m.avg_win_pct / m.avg_loss_pct).toFixed(1)}:1` : "—"}
                                            </span>
                                        </div>
                                        <hr style={{ borderColor: "var(--border)" }} />
                                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                                            <span style={{ color: "var(--text-muted)" }}>Başlangıç</span>
                                            <span>${m.initial_capital?.toLocaleString()}</span>
                                        </div>
                                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                                            <span style={{ color: "var(--text-muted)" }}>Bitiş</span>
                                            <span style={{ color: m.final_capital > m.initial_capital ? "var(--green)" : "var(--red)", fontWeight: 700 }}>
                                                ${m.final_capital?.toLocaleString()}
                                            </span>
                                        </div>
                                    </div>
                                </div>

                                {/* Exit analysis */}
                                <div className="glass-card" style={{ padding: 18 }}>
                                    <div style={{ fontWeight: 700, marginBottom: 12, fontSize: "0.85rem" }}>🚪 Çıkış Analizi</div>
                                    {m.exit_stats && Object.entries(m.exit_stats).map(([reason, stats]) => (
                                        <div key={reason} style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem", marginBottom: 8 }}>
                                            <span style={{ color: "var(--text-secondary)" }}>{EXIT_LABELS[reason] || reason}</span>
                                            <span>
                                                <strong>{stats.count}</strong>
                                                <span style={{ color: stats.avg_pnl >= 0 ? "var(--green)" : "var(--red)", marginLeft: 8, fontSize: "0.78rem" }}>
                                                    Ort: {stats.avg_pnl >= 0 ? "+" : ""}{stats.avg_pnl?.toFixed(1)}%
                                                </span>
                                            </span>
                                        </div>
                                    ))}
                                    {/* Swing type stats */}
                                    {m.type_stats && Object.keys(m.type_stats).length > 0 && (
                                        <>
                                            <hr style={{ borderColor: "var(--border)", margin: "10px 0" }} />
                                            <div style={{ fontWeight: 600, fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: 8 }}>SWING TİPİ</div>
                                            {Object.entries(m.type_stats).map(([type, stats]) => {
                                                const total = stats.wins + stats.losses;
                                                const wr = total > 0 ? (stats.wins / total * 100).toFixed(0) : 0;
                                                return (
                                                    <div key={type} style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", marginBottom: 6 }}>
                                                        <span>Tip {type}</span>
                                                        <span>{total} trade · WR: {wr}% · P/L: {stats.total_pnl >= 0 ? "+" : ""}{stats.total_pnl?.toFixed(1)}%</span>
                                                    </div>
                                                );
                                            })}
                                        </>
                                    )}
                                </div>
                            </div>

                            {/* Equity curve */}
                            {eqData.length > 0 && (
                                <div className="glass-card" style={{ padding: 20, marginBottom: 16 }}>
                                    <div style={{ fontWeight: 700, marginBottom: 14, fontSize: "0.85rem" }}>
                                        📈 Equity Curve (Portföy Değeri)
                                    </div>
                                    <ResponsiveContainer width="100%" height={260}>
                                        <ComposedChart data={eqData}>
                                            <defs>
                                                <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
                                                </linearGradient>
                                            </defs>
                                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                                            <XAxis dataKey="date" tick={{ fill: "var(--text-muted)", fontSize: 10 }} interval="preserveStartEnd" />
                                            <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} tickFormatter={v => `$${v.toLocaleString()}`} domain={["auto", "auto"]} />
                                            <Tooltip content={<CTooltip />} />
                                            <ReferenceLine y={capital} stroke="rgba(255,255,255,0.2)" strokeDasharray="4 2" label={{ value: "Başlangıç", fill: "var(--text-muted)", fontSize: 10 }} />
                                            <Area type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} fill="url(#eqGrad)" name="Portföy" />
                                        </ComposedChart>
                                    </ResponsiveContainer>
                                </div>
                            )}

                            {/* Trade P&L bars */}
                            {tradeData.length > 0 && (
                                <div className="glass-card" style={{ padding: 20, marginBottom: 16 }}>
                                    <div style={{ fontWeight: 700, marginBottom: 14, fontSize: "0.85rem" }}>
                                        📊 Trade P/L (Son {tradeData.length})
                                    </div>
                                    <ResponsiveContainer width="100%" height={160}>
                                        <ComposedChart data={tradeData}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                                            <XAxis dataKey="ticker" tick={{ fill: "var(--text-muted)", fontSize: 9 }} />
                                            <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} tickFormatter={v => `$${v.toFixed(0)}`} />
                                            <Tooltip content={<CTooltip />} />
                                            <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
                                            <Bar dataKey="pnl" name="P/L $" radius={[3, 3, 0, 0]}>
                                                {tradeData.map((d, i) => (
                                                    <Cell key={i} fill={d.pnl >= 0 ? "rgba(34,197,94,0.7)" : "rgba(239,68,68,0.7)"} />
                                                ))}
                                            </Bar>
                                        </ComposedChart>
                                    </ResponsiveContainer>
                                </div>
                            )}

                            {/* Trade log */}
                            <div className="glass-card" style={{ overflow: "hidden" }}>
                                <div style={{ padding: "16px 20px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                    <div style={{ fontWeight: 700, fontSize: "0.9rem" }}>📝 Trade Geçmişi</div>
                                    <span className="badge badge-blue">{result.trades.length} trade</span>
                                </div>
                                <div style={{ overflowX: "auto" }}>
                                    <table className="data-table">
                                        <thead>
                                            <tr>
                                                <th>Hisse</th><th>Tip</th><th>Giriş</th><th>Çıkış</th>
                                                <th>P/L %</th><th>P/L $</th><th>Çıkış Nedeni</th><th>Süre</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {result.trades.map((t, i) => {
                                                const win = t.pnl_dollar >= 0;
                                                return (
                                                    <tr key={i}>
                                                        <td><strong style={{ color: "var(--accent)" }}>{t.ticker}</strong></td>
                                                        <td><span className="badge badge-blue">{t.swing_type || "—"}</span></td>
                                                        <td>${t.entry_price?.toFixed(2)}</td>
                                                        <td>${t.exit_price?.toFixed(2)}</td>
                                                        <td style={{ color: win ? "var(--green)" : "var(--red)", fontWeight: 700 }}>
                                                            {win ? "+" : ""}{t.pnl_pct?.toFixed(1)}%
                                                        </td>
                                                        <td style={{ color: win ? "var(--green)" : "var(--red)", fontWeight: 700 }}>
                                                            {win ? "+" : ""}${t.pnl_dollar?.toFixed(0)}
                                                        </td>
                                                        <td>
                                                            <span className={`badge ${t.exit_reason === "TARGET" ? "badge-green" : t.exit_reason === "STOPPED" ? "badge-red" : "badge-yellow"}`}>
                                                                {EXIT_LABELS[t.exit_reason || ""] || t.exit_reason || "—"}
                                                            </span>
                                                        </td>
                                                        <td style={{ color: "var(--text-muted)" }}>{t.hold_days ?? "—"} gün</td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </>
                    )}
                </>
            )}

            {/* Empty state */}
            {!result && !loading && !error && (
                <div className="glass-card" style={{ padding: 60, textAlign: "center" }}>
                    <FlaskConical size={52} style={{ color: "var(--accent)", opacity: 0.4, marginBottom: 16 }} />
                    <div style={{ color: "var(--text-secondary)", fontWeight: 600, marginBottom: 8 }}>
                        Parametreleri seç ve Backtest Başlat&apos;a bas
                    </div>
                    <div style={{ color: "var(--text-muted)", fontSize: "0.875rem" }}>
                        Walk-forward simülasyon · Geçmiş veri üzerinde SmallCap sinyallerini test eder
                    </div>
                </div>
            )}
        </div>
    );
}
