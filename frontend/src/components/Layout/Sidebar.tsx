"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
    LayoutDashboard, Search, TrendingUp, Clock, BarChart3,
    LineChart, MessageSquare, Zap, FlaskConical, X, LogOut, Settings, BookOpen,
    Target, PanelLeftClose, PanelLeftOpen,
} from "lucide-react";
import { createSupabaseClient } from "@/lib/supabase/client";

interface SidebarProps {
    isOpen?: boolean;
    onClose?: () => void;
    isMobile?: boolean;
    collapsed?: boolean;
    onToggleCollapse?: () => void;
}

const navItems = [
    { href: "/", label: "Dashboard", icon: LayoutDashboard },
    { href: "/how-it-works", label: "Nasıl çalışır?", icon: BookOpen },
    { href: "/scanner", label: "Scanner", icon: Search },
    { href: "/scanner/history", label: "Scanner Geçmişi", icon: Clock },
    { href: "/lookup", label: "Manual Lookup", icon: Zap },
    { href: "/trades", label: "Paper Trades", icon: TrendingUp },
    { href: "/pending", label: "Pending", icon: Clock },
    { href: "/performance", label: "Performance", icon: BarChart3 },
    { href: "/edge", label: "Sinyal Karnesi", icon: Target },
    { href: "/charts", label: "Charts", icon: LineChart },
    { href: "/backtest", label: "Backtest", icon: FlaskConical },
    { href: "/settings", label: "Ayarlar", icon: Settings },
    { href: "/chat", label: "AI Chat", icon: MessageSquare },
];

export default function Sidebar({ isOpen = false, onClose, isMobile = false, collapsed = false, onToggleCollapse }: SidebarProps) {
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

    const isCollapsed = collapsed && !isMobile;

    return (
        <aside className={`sidebar ${isMobile && isOpen ? "sidebar-open" : ""} ${isCollapsed ? "sidebar-collapsed" : ""}`}>
            {/* Logo */}
            <div className="sidebar-header">
                <div className="sidebar-logo">
                    <div className="sidebar-logo-icon">📈</div>
                    {!isCollapsed && (
                        <div className="sidebar-logo-text">
                            <div style={{ fontWeight: 800, fontSize: "0.95rem", letterSpacing: "-0.02em" }}>
                                Swing Trade
                            </div>
                            <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", letterSpacing: "0.06em" }}>
                                AI DASHBOARD
                            </div>
                        </div>
                    )}
                </div>
                {isMobile && onClose ? (
                    <button type="button" onClick={onClose} aria-label="Close menu" className="sidebar-close-btn">
                        <X size={20} />
                    </button>
                ) : (
                    !isMobile && onToggleCollapse && (
                        <button
                            type="button"
                            onClick={onToggleCollapse}
                            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
                            className="sidebar-toggle-btn"
                            title={collapsed ? "Genişlet" : "Daralt"}
                        >
                            {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
                        </button>
                    )
                )}
            </div>

            {/* Live indicator */}
            <div className="sidebar-live-indicator">
                <span className="live-dot" />
                {!isCollapsed && <span>Live Market Mode</span>}
            </div>

            {/* Nav */}
            <nav style={{ flex: 1, padding: "8px 0", overflowY: "auto" }}>
                {navItems.map(({ href, label, icon: Icon }) => {
                    const active = pathname === href || (href !== "/" && pathname.startsWith(href));
                    return (
                        <Link
                            key={href}
                            href={href}
                            className={`sidebar-nav-item ${active ? "active" : ""}`}
                            onClick={isMobile ? onClose : undefined}
                            title={isCollapsed ? label : undefined}
                        >
                            <Icon size={16} />
                            {!isCollapsed && <span>{label}</span>}
                            {active && !isCollapsed && (
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
            <div className="sidebar-footer">
                {!isCollapsed && (
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: supabase ? 8 : 0 }}>
                        SmallCap Momentum v2.1
                    </div>
                )}
                {supabase && (
                    <button
                        type="button"
                        onClick={handleSignOut}
                        className="sidebar-nav-item"
                        style={{ width: "100%", justifyContent: isCollapsed ? "center" : "flex-start", color: "var(--text-muted)", fontSize: "0.75rem" }}
                        title={isCollapsed ? "Sign Out" : undefined}
                    >
                        <LogOut size={14} />
                        {!isCollapsed && <span>Sign Out</span>}
                    </button>
                )}
            </div>
        </aside>
    );
}
