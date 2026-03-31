"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
    LayoutDashboard, Search, TrendingUp, Clock, BarChart3,
    LineChart, MessageSquare, Zap, FlaskConical, X, LogOut, Settings, BookOpen,
} from "lucide-react";
import { createSupabaseClient } from "@/lib/supabase/client";

interface SidebarProps {
    isOpen?: boolean;
    onClose?: () => void;
    isMobile?: boolean;
}

const navItems = [
    { href: "/", label: "Dashboard", icon: LayoutDashboard },
    { href: "/how-it-works", label: "Nasıl çalışır?", icon: BookOpen },
    { href: "/scanner", label: "Scanner", icon: Search },
    { href: "/lookup", label: "Manual Lookup", icon: Zap },
    { href: "/trades", label: "Paper Trades", icon: TrendingUp },
    { href: "/pending", label: "Pending", icon: Clock },
    { href: "/performance", label: "Performance", icon: BarChart3 },
    { href: "/charts", label: "Charts", icon: LineChart },
    { href: "/backtest", label: "Backtest", icon: FlaskConical },
    { href: "/settings", label: "Ayarlar", icon: Settings },
    { href: "/chat", label: "AI Chat", icon: MessageSquare },
];

export default function Sidebar({ isOpen = false, onClose, isMobile = false }: SidebarProps) {
    const pathname = usePathname();
    const router = useRouter();
    const supabase = createSupabaseClient();

    const handleSignOut = async () => {
        if (supabase) {
            await supabase.auth.signOut();
            router.push("/login");
            router.refresh();
        }
    };

    return (
        <aside className={`sidebar ${isMobile && isOpen ? "sidebar-open" : ""}`}>
            {/* Logo */}
            <div style={{ padding: "20px 18px 16px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <div style={{
                        width: 36, height: 36, borderRadius: 10,
                        background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: "16px",
                    }}>📈</div>
                    <div>
                        <div style={{ fontWeight: 800, fontSize: "0.95rem", letterSpacing: "-0.02em" }}>
                            Swing Trade
                        </div>
                        <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", letterSpacing: "0.06em" }}>
                            AI DASHBOARD
                        </div>
                    </div>
                </div>
                {isMobile && onClose && (
                    <button type="button" onClick={onClose} aria-label="Close menu" className="sidebar-close-btn">
                        <X size={20} />
                    </button>
                )}
            </div>

            {/* Live indicator */}
            <div style={{ padding: "10px 18px", borderBottom: "1px solid var(--border-muted)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: "0.72rem", color: "var(--text-secondary)" }}>
                    <span className="live-dot" />
                    Live Market Mode
                </div>
            </div>

            {/* Nav */}
            <nav style={{ flex: 1, padding: "8px 0", overflowY: "auto" }}>
                {navItems.map(({ href, label, icon: Icon }) => {
                    const active = pathname === href || (href !== "/" && pathname.startsWith(href));
                    return (
                        <Link key={href} href={href} className={`sidebar-nav-item ${active ? "active" : ""}`} onClick={isMobile ? onClose : undefined}>
                            <Icon size={16} />
                            <span>{label}</span>
                            {active && (
                                <div style={{
                                    marginLeft: "auto", width: 4, height: 4,
                                    borderRadius: "50%", background: "var(--accent)",
                                }} />
                            )}
                        </Link>
                    );
                })}
            </nav>

            {/* Footer */}
            <div style={{ padding: "14px 18px", borderTop: "1px solid var(--border)" }}>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: supabase ? 8 : 0 }}>
                    SmallCap Momentum v2.1
                </div>
                {supabase && (
                    <button
                        type="button"
                        onClick={handleSignOut}
                        className="sidebar-nav-item"
                        style={{ width: "100%", justifyContent: "flex-start", color: "var(--text-muted)", fontSize: "0.75rem" }}
                    >
                        <LogOut size={14} />
                        <span>Sign Out</span>
                    </button>
                )}
            </div>
        </aside>
    );
}
