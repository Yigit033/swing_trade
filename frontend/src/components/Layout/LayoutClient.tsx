"use client";

import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import Sidebar from "./Sidebar";
import { Menu } from "lucide-react";
import { QueryProvider } from "@/providers/QueryProvider";
import { ScannerJobProvider } from "@/providers/ScannerJobProvider";
import ScannerScanBanner from "@/components/ScannerScanBanner";

export default function LayoutClient({ children }: { children: React.ReactNode }) {
    const pathname = usePathname();
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [isMobile, setIsMobile] = useState(false);

    useEffect(() => {
        const check = () => setIsMobile(window.innerWidth < 768);
        check();
        window.addEventListener("resize", check);
        return () => window.removeEventListener("resize", check);
    }, []);

    const closeSidebar = () => setSidebarOpen(false);
    const isLoginPage = pathname === "/login";

    if (isLoginPage) {
        return <QueryProvider>{children}</QueryProvider>;
    }

    return (
        <QueryProvider>
            <ScannerJobProvider>
            <Sidebar
                isOpen={sidebarOpen}
                onClose={closeSidebar}
                isMobile={isMobile}
            />
            <main className="main-content">
                <ScannerScanBanner />
                {/* Mobile hamburger — only visible on small screens */}
                {isMobile && (
                    <button
                        type="button"
                        aria-label="Open menu"
                        onClick={() => setSidebarOpen(true)}
                        className="mobile-menu-btn"
                    >
                        <Menu size={22} />
                    </button>
                )}
                {children}
            </main>
            {/* Overlay when sidebar open on mobile */}
            {isMobile && sidebarOpen && (
                <div
                    className="sidebar-overlay"
                    onClick={closeSidebar}
                    onKeyDown={(e) => e.key === "Escape" && closeSidebar()}
                    role="button"
                    tabIndex={0}
                    aria-label="Close menu"
                />
            )}
            </ScannerJobProvider>
        </QueryProvider>
    );
}
