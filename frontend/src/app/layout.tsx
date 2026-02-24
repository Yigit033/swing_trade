import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Layout/Sidebar";

export const metadata: Metadata = {
  title: "Swing Trade AI",
  description: "AI-powered swing trading dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
      </head>
      <body>
        <Sidebar />
        <main className="main-content">{children}</main>
      </body>
    </html>
  );
}
