"use client";
import { usePerformance, usePending, useRegime } from "@/hooks/useApi";
import type { Trade } from "@/lib/api";
import { TrendingUp, TrendingDown, Activity, Clock, Target } from "lucide-react";

function MetricCard({
  label, value, sub, positive, icon: Icon, color = "blue"
}: {
  label: string; value: string; sub?: string;
  positive?: boolean; icon?: React.ElementType; color?: string;
}) {
  const colors = { blue: "#3b82f6", green: "#22c55e", red: "#ef4444", purple: "#a855f7", yellow: "#f59e0b", teal: "#14b8a6" };
  const c = colors[color as keyof typeof colors] || colors.blue;
  return (
    <div className="metric-card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>
          {label}
        </div>
        {Icon && (
          <div style={{ width: 32, height: 32, borderRadius: 8, background: `${c}20`, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Icon size={15} color={c} />
          </div>
        )}
      </div>
      <div style={{ fontSize: "1.75rem", fontWeight: 800, marginTop: 10, letterSpacing: "-0.02em", color: positive === true ? "var(--green)" : positive === false ? "var(--red)" : "var(--text-primary)" }}>
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: 4 }}>
          {sub}
        </div>
      )}
    </div>
  );
}

function TypeBadge({ type }: { type?: string }) {
  if (!type) return <span className="badge badge-blue">—</span>;
  const colors: Record<string, string> = { A: "badge-green", B: "badge-blue", C: "badge-yellow" };
  return <span className={`badge ${colors[type] || "badge-purple"}`}>{type}</span>;
}

function PnlCell({ pnl, pct }: { pnl?: number | null; pct?: number | null }) {
  if (pnl == null) return <span style={{ color: "var(--text-muted)" }}>—</span>;
  const pos = pnl >= 0;
  return (
    <div>
      <span style={{ color: pos ? "var(--green)" : "var(--red)", fontWeight: 600 }}>
        {pos ? "+$" : "-$"}{Math.abs(pnl).toFixed(2)}
      </span>
      {pct != null && (
        <div style={{ fontSize: "0.72rem", color: pos ? "var(--green)" : "var(--red)", opacity: 0.8 }}>
          {pos ? "+" : ""}{pct.toFixed(2)}%
        </div>
      )}
    </div>
  );
}

function fmtDate(d?: string | null) {
  if (!d) return "—";
  return d.slice(0, 10);
}

export default function DashboardPage() {
  const { data: perf, isLoading: perfLoading } = usePerformance();
  const { data: pendingCount = 0, isLoading: pendingLoading } = usePending();
  const { data: regime, isLoading: regimeLoading } = useRegime();

  const loading = perfLoading || pendingLoading || regimeLoading;

  if (loading && !perf) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "60vh", flexDirection: "column", gap: 16 }}>
      <div className="spinner" style={{ width: 40, height: 40 }} />
      <div style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>Loading live portfolio...</div>
    </div>
  );

  const s = perf?.summary;
  const openTrades = perf?.open_trades || [];
  const recentClosed = perf?.recent_closed || [];

  return (
    <div>
      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h1 className="page-title gradient-text">Live Portfolio</h1>
        <p className="page-subtitle">Real-time position monitor · SmallCap Momentum v2.1</p>
      </div>

      {/* Market Regime Banner */}
      {regime && (() => {
        const rr = regime.regime;
        const isUnknown = rr === "UNKNOWN";
        const border =
          rr === "BULL" ? "var(--green)"
          : rr === "BEAR" ? "var(--red)"
          : rr === "CAUTION" ? "var(--yellow)"
          : "var(--text-muted)";
        const pillBg =
          rr === "BULL" ? "rgba(34,197,94,0.12)"
          : rr === "BEAR" ? "rgba(239,68,68,0.12)"
          : rr === "CAUTION" ? "rgba(245,158,11,0.12)"
          : "rgba(148,163,184,0.12)";
        const pillColor =
          rr === "BULL" ? "var(--green)"
          : rr === "BEAR" ? "var(--red)"
          : rr === "CAUTION" ? "var(--yellow)"
          : "var(--text-muted)";
        const pillText =
          rr === "BULL" ? "BULL"
          : rr === "BEAR" ? "BEAR"
          : rr === "CAUTION" ? "CAUTION"
          : "BİLİNMİYOR";
        return (
        <div className="glass-card" style={{
          padding: "14px 22px", marginBottom: 20,
          display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap",
          borderLeft: `3px solid ${border}`,
        }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 0 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
              <span style={{
                fontSize: "0.8rem", fontWeight: 800, letterSpacing: "0.05em",
                padding: "4px 12px", borderRadius: 6,
                background: pillBg,
                color: pillColor,
              }}>
                {pillText}
              </span>
              {isUnknown ? (
                <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
                  Piyasa rejimi okunamadı; skor çarpanı nötr (×1).
                </span>
              ) : regime.confidence === "TENTATIVE" ? (
                <span style={{ fontSize: "0.68rem", color: "var(--text-muted)", background: "rgba(255,255,255,0.06)", padding: "2px 8px", borderRadius: 4 }}>
                  Unconfirmed
                </span>
              ) : null}
              {/* score_multiplier removed */}
            </div>
            {isUnknown && regime.detect_error ? (
              <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", wordBreak: "break-word" }}>
                {regime.detect_error.length > 160 ? `${regime.detect_error.slice(0, 160)}…` : regime.detect_error}
              </span>
            ) : null}
          </div>
          {!isUnknown ? (
          <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", display: "flex", gap: 14, flexWrap: "wrap" }}>
            {regime.spy_price != null && regime.spy_price > 0 && <span>SPY ${regime.spy_price.toFixed(2)}</span>}
            {regime.vix != null && regime.vix > 0 && (
              <span style={{ color: regime.vix > 25 ? "var(--red)" : "var(--text-muted)" }}>
                VIX {regime.vix.toFixed(1)}
              </span>
            )}
            {regime.spy_5d_return != null && (
              <span style={{ color: regime.spy_5d_return >= 0 ? "var(--green)" : "var(--red)" }}>
                5d {regime.spy_5d_return >= 0 ? "+" : ""}{regime.spy_5d_return.toFixed(1)}%
              </span>
            )}
            {regime.stale_fallback ? (
              <span style={{ fontSize: "0.68rem", color: "var(--yellow)", opacity: 0.9 }}>
                (önbellek — canlı SPY/VIX alınamadı)
              </span>
            ) : null}
          </div>
          ) : null}
        </div>
        );
      })()}

      {/* Metrics */}
      <div className="metrics-grid">
        <MetricCard
          label="Total P&L %"
          value={s ? `${s.total_pnl_pct >= 0 ? "+" : ""}${s.total_pnl_pct.toFixed(2)}%` : "—"}
          sub={s ? `Avg: ${s.avg_pnl_pct.toFixed(2)}% per trade` : undefined}
          positive={s ? s.total_pnl_pct >= 0 : undefined}
          icon={s && s.total_pnl_pct >= 0 ? TrendingUp : TrendingDown}
          color={s && s.total_pnl_pct >= 0 ? "green" : "red"}
        />
        <MetricCard label="Win Rate" value={s ? `${s.win_rate}%` : "—"} sub={s ? `${s.wins}W / ${s.losses}L` : "—"} icon={Target} color="blue" />
        <MetricCard label="Open Trades" value={s ? `${s.open_trades}` : "—"} icon={Activity} color="purple" />
        <MetricCard label="Pending" value={`${pendingCount}`} sub="awaiting confirmation" icon={Clock} color="yellow" />
        <MetricCard label="Avg Win" value={s ? `$${s.avg_win.toFixed(2)}` : "—"} positive={true} icon={TrendingUp} color="green" />
        <MetricCard label="Avg Loss" value={s ? `$${s.avg_loss.toFixed(2)}` : "—"} positive={false} icon={TrendingDown} color="red" />
      </div>

      {/* Active Positions */}
      <div className="glass-card" style={{ marginBottom: 24, overflow: "hidden" }}>
        <div style={{ padding: "18px 22px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ margin: 0, fontSize: "1rem", fontWeight: 700 }}>🟢 Active Positions</h2>
          <span className="badge badge-blue">{openTrades.length} OPEN</span>
        </div>
        <div style={{ overflowX: "auto" }}>
          {openTrades.length === 0 ? (
            <div style={{ padding: 32, textAlign: "center", color: "var(--text-muted)" }}>No open positions</div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Ticker</th><th>Type</th><th>Entry Date</th>
                  <th>Entry $</th><th>Current $</th><th>Stop</th><th>Target</th>
                  <th>Quality</th><th>P&amp;L</th>
                </tr>
              </thead>
              <tbody>
                {openTrades.map((t: Trade) => {
                  const cp = t.current_price;
                  const hasLive = cp != null;
                  return (
                    <tr key={t.id}>
                      <td><strong style={{ color: "var(--accent)" }}>{t.ticker}</strong></td>
                      <td><TypeBadge type={t.swing_type} /></td>
                      <td style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>{fmtDate(t.entry_date)}</td>
                      <td>${t.entry_price?.toFixed(2)}</td>
                      <td>
                        {hasLive ? (
                          <span style={{ fontWeight: 700, color: (cp! >= (t.entry_price || 0)) ? "var(--green)" : "var(--red)" }}>
                            ${cp!.toFixed(2)}
                          </span>
                        ) : <span style={{ color: "var(--text-muted)" }}>—</span>}
                      </td>
                      <td style={{ color: "var(--red)" }}>${t.stop_loss?.toFixed(2)}</td>
                      <td style={{ color: "var(--green)" }}>${t.target?.toFixed(2)}</td>
                      <td>
                        <span className={`badge ${(t.quality_score || 0) >= 80 ? "badge-green" : (t.quality_score || 0) >= 65 ? "badge-blue" : "badge-yellow"}`}>
                          {t.quality_score?.toFixed(0)}
                        </span>
                      </td>
                      <td><PnlCell pnl={t.unrealized_pnl} pct={t.unrealized_pnl_pct} /></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* Recent Closed — compact, link to Performance for full history */}
      <div className="glass-card" style={{ overflow: "hidden" }}>
        <div style={{ padding: "18px 22px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ margin: 0, fontSize: "1rem", fontWeight: 700 }}>📋 Recent Closed Trades</h2>
          <a href="/performance" style={{ fontSize: "0.78rem", color: "var(--accent)", textDecoration: "none", opacity: 0.8 }}>
            View full history →
          </a>
        </div>
        <div style={{ overflowX: "auto" }}>
          {recentClosed.length === 0 ? (
            <div style={{ padding: 32, textAlign: "center", color: "var(--text-muted)" }}>No closed trades yet</div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Ticker</th><th>Type</th><th>Entry Date</th><th>Close Date</th>
                  <th>Entry $</th><th>Exit $</th><th>P&amp;L %</th><th>Result</th>
                </tr>
              </thead>
              <tbody>
                {recentClosed.slice(0, 8).map((t: Trade) => {
                  const pct = t.realized_pnl_pct;
                  const win = (t.realized_pnl || 0) > 0;
                  const statusColor =
                    t.status === "TARGET" ? "badge-green" :
                      t.status === "STOPPED" ? "badge-red" :
                        t.status === "TRAILED" ? "badge-yellow" :
                          t.status === "MANUAL" ? "badge-blue" : "badge-red";
                  return (
                    <tr key={t.id}>
                      <td><strong>{t.ticker}</strong></td>
                      <td><TypeBadge type={t.swing_type} /></td>
                      <td style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>{fmtDate(t.entry_date)}</td>
                      <td style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>{fmtDate(t.exit_date)}</td>
                      <td>${t.entry_price?.toFixed(2)}</td>
                      <td>${t.exit_price?.toFixed(2) || "—"}</td>
                      <td style={{ color: win ? "var(--green)" : "var(--red)", fontWeight: 600 }}>
                        {pct != null ? `${win ? "+" : ""}${pct.toFixed(2)}%` : "—"}
                      </td>
                      <td><span className={`badge ${statusColor}`}>{t.status}</span></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
