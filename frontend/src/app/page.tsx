"use client";
import { useState, useEffect } from "react";
import { getPerformance, getPending, getTrades } from "@/lib/api";
import type { PerformanceSummary, Trade } from "@/lib/api";
import { TrendingUp, TrendingDown, Activity, Clock, DollarSign, Target } from "lucide-react";

function MetricCard({
  label, value, sub, positive, icon: Icon, color = "blue"
}: {
  label: string; value: string; sub?: string;
  positive?: boolean; icon?: React.ElementType; color?: string;
}) {
  const colors = { blue: "#3b82f6", green: "#22c55e", red: "#ef4444", purple: "#a855f7", yellow: "#f59e0b" };
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

function PnlBadge({ pnl }: { pnl?: number | null }) {
  if (pnl == null) return <span className="badge badge-blue">Open</span>;
  const pos = pnl >= 0;
  return (
    <span className={`badge ${pos ? "badge-green" : "badge-red"}`}>
      {pos ? "+" : ""}{pnl.toFixed(2)}$
    </span>
  );
}

export default function DashboardPage() {
  const [perf, setPerf] = useState<{ summary: PerformanceSummary; open_trades: Trade[]; recent_closed: Trade[] } | null>(null);
  const [pendingCount, setPendingCount] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getPerformance().catch(() => null),
      getPending().catch(() => ({ pending: [] })),
    ]).then(([p, pnd]) => {
      setPerf(p);
      setPendingCount(pnd?.count || 0);
    }).finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "60vh", flexDirection: "column", gap: 16 }}>
      <div className="spinner" style={{ width: 40, height: 40 }} />
      <div style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>Loading dashboard...</div>
    </div>
  );

  const s = perf?.summary;

  return (
    <div>
      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h1 className="page-title gradient-text">Trading Dashboard</h1>
        <p className="page-subtitle">AI-powered swing trading overview · SmallCap Momentum v2.1</p>
      </div>

      {/* Metrics */}
      <div className="metrics-grid">
        <MetricCard label="Total P&L" value={s ? `$${s.total_pnl.toFixed(2)}` : "—"} positive={s && s.total_pnl >= 0} icon={DollarSign} color={s && s.total_pnl >= 0 ? "green" : "red"} />
        <MetricCard label="Win Rate" value={s ? `${s.win_rate}%` : "—"} sub={s ? `${s.wins}W / ${s.losses}L` : "—"} icon={Target} color="blue" />
        <MetricCard label="Open Trades" value={s ? `${s.open_trades}` : "—"} icon={Activity} color="purple" />
        <MetricCard label="Pending" value={`${pendingCount}`} sub="awaiting confirmation" icon={Clock} color="yellow" />
        <MetricCard label="Avg Win" value={s ? `$${s.avg_win.toFixed(2)}` : "—"} positive={true} icon={TrendingUp} color="green" />
        <MetricCard label="Avg Loss" value={s ? `$${s.avg_loss.toFixed(2)}` : "—"} positive={false} icon={TrendingDown} color="red" />
      </div>

      {/* Active Trades */}
      <div className="glass-card" style={{ marginBottom: 24, overflow: "hidden" }}>
        <div style={{ padding: "18px 22px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ margin: 0, fontSize: "1rem", fontWeight: 700 }}>🟢 Active Positions</h2>
          <span className="badge badge-blue">{perf?.open_trades?.length || 0} open</span>
        </div>
        <div style={{ overflowX: "auto" }}>
          {(!perf?.open_trades?.length) ? (
            <div style={{ padding: 32, textAlign: "center", color: "var(--text-muted)" }}>No open positions</div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Ticker</th><th>Entry</th><th>Stop</th><th>Target</th>
                  <th>Quality</th><th>Current P&L</th><th>Status</th>
                </tr>
              </thead>
              <tbody>
                {perf.open_trades.map(t => (
                  <tr key={t.id}>
                    <td><strong style={{ color: "var(--accent)" }}>{t.ticker}</strong></td>
                    <td>${t.entry_price?.toFixed(2)}</td>
                    <td style={{ color: "var(--red)" }}>${t.stop_loss?.toFixed(2)}</td>
                    <td style={{ color: "var(--green)" }}>${t.target?.toFixed(2)}</td>
                    <td>
                      <span className={`badge ${t.quality_score >= 80 ? "badge-green" : t.quality_score >= 65 ? "badge-blue" : "badge-yellow"}`}>
                        {t.quality_score?.toFixed(0)}
                      </span>
                    </td>
                    <td><PnlBadge pnl={t.unrealized_pnl} /></td>
                    <td><span className="badge badge-green">OPEN</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* Recent Closed */}
      <div className="glass-card" style={{ overflow: "hidden" }}>
        <div style={{ padding: "18px 22px", borderBottom: "1px solid var(--border)" }}>
          <h2 style={{ margin: 0, fontSize: "1rem", fontWeight: 700 }}>📋 Recent Closed Trades</h2>
        </div>
        <div style={{ overflowX: "auto" }}>
          {(!perf?.recent_closed?.length) ? (
            <div style={{ padding: 32, textAlign: "center", color: "var(--text-muted)" }}>No closed trades yet</div>
          ) : (
            <table className="data-table">
              <thead>
                <tr><th>Ticker</th><th>Entry</th><th>Exit</th><th>P&L</th><th>P&L %</th><th>Result</th></tr>
              </thead>
              <tbody>
                {perf.recent_closed.slice(0, 10).map(t => {
                  const win = (t.realized_pnl || 0) > 0;
                  return (
                    <tr key={t.id}>
                      <td><strong>{t.ticker}</strong></td>
                      <td>${t.entry_price?.toFixed(2)}</td>
                      <td>${t.exit_price?.toFixed(2)}</td>
                      <td style={{ color: win ? "var(--green)" : "var(--red)" }}>
                        {win ? "+" : ""}{t.realized_pnl?.toFixed(2)}$
                      </td>
                      <td style={{ color: win ? "var(--green)" : "var(--red)" }}>
                        {win ? "+" : ""}{t.realized_pnl_pct?.toFixed(2)}%
                      </td>
                      <td>
                        <span className={`badge ${win ? "badge-green" : t.status === "REJECTED" ? "badge-red" : "badge-red"}`}>
                          {t.status || (win ? "WIN" : "LOSS")}
                        </span>
                      </td>
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
