"use client";

import { useState, useEffect } from "react";
import Sidebar from "./Sidebar";
import { Menu } from "lucide-react";

export default function LayoutClient({ children }: { children: React.ReactNode }) {
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [isMobile, setIsMobile] = useState(false);

    useEffect(() => {
        const check = () => setIsMobile(window.innerWidth < 768);
        check();
        window.addEventListener("resize", check);
        return () => window.removeEventListener("resize", check);
    }, []);

    const closeSidebar = () => setSidebarOpen(false);

    return (
        <>
            <Sidebar
                isOpen={sidebarOpen}
                onClose={closeSidebar}
                isMobile={isMobile}
            />
            <main className="main-content">
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
        </>
    );
}
