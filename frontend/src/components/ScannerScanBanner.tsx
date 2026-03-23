"use client";

import Link from "next/link";
import { useScannerJob } from "@/providers/ScannerJobProvider";

/** Tarama arka planda sürerken tüm sayfalarda üst bant */
export default function ScannerScanBanner() {
    const { isScanning, poll, scanError, dismissScanFeedback } = useScannerJob();

    if (scanError) {
        return (
            <div
                style={{
                    position: "sticky",
                    top: 0,
                    zIndex: 50,
                    padding: "10px 16px",
                    background: "rgba(239,68,68,0.15)",
                    borderBottom: "1px solid rgba(239,68,68,0.35)",
                    fontSize: "0.85rem",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: 12,
                    flexWrap: "wrap",
                }}
            >
                <span style={{ color: "var(--red)" }}>Scanner: {scanError}</span>
                <button
                    type="button"
                    onClick={dismissScanFeedback}
                    className="text-xs underline opacity-80 hover:opacity-100"
                    style={{ background: "none", border: "none", cursor: "pointer", color: "inherit" }}
                >
                    Kapat
                </button>
            </div>
        );
    }

    if (!isScanning || !poll) return null;

    const pct = Math.round(poll.progress ?? 0);
    return (
        <div
            style={{
                position: "sticky",
                top: 0,
                zIndex: 50,
                padding: "10px 16px",
                background: "rgba(59,130,246,0.12)",
                borderBottom: "1px solid rgba(59,130,246,0.25)",
                fontSize: "0.85rem",
            }}
        >
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                <span style={{ color: "var(--text-secondary)" }}>
                    <strong style={{ color: "var(--accent)" }}>SmallCap scan</strong> arka planda çalışıyor — sayfayı değiştirebilirsiniz.
                    {" "}
                    <Link href="/scanner" style={{ color: "var(--accent)", textDecoration: "underline" }}>
                        Scanner’a git
                    </Link>
                </span>
                <span style={{ fontWeight: 700, color: "var(--accent)", minWidth: 42 }}>{pct}%</span>
            </div>
            <div
                style={{
                    marginTop: 8,
                    height: 4,
                    borderRadius: 2,
                    background: "rgba(255,255,255,0.08)",
                    overflow: "hidden",
                }}
            >
                <div
                    style={{
                        height: "100%",
                        width: `${pct}%`,
                        background: "var(--accent)",
                        transition: "width 0.3s ease",
                    }}
                />
            </div>
            {poll.message ? (
                <div style={{ marginTop: 6, fontSize: "0.78rem", color: "var(--text-muted)" }}>
                    {poll.message}
                </div>
            ) : null}
        </div>
    );
}
