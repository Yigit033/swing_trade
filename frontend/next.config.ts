import type { NextConfig } from "next";

/** Auth-dependent sayfalar için CDN cache'i engeller (custom domain redirect loop önlemi). */
const NO_CACHE_HEADERS = [
  { key: "Cache-Control", value: "private, no-store, must-revalidate" },
  { key: "Vercel-CDN-Cache-Control", value: "max-age=0" },
];

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/:path*`,
      },
    ];
  },
  async headers() {
    const protectedPaths = ["/", "/trades", "/performance", "/pending", "/scanner", "/lookup", "/charts", "/chat", "/backtest", "/login"];
    return protectedPaths.map((path) => ({
      source: path === "/" ? path : `${path}/:path*`,
      headers: NO_CACHE_HEADERS,
    }));
  },
};

export default nextConfig;
