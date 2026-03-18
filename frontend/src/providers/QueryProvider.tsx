"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

const STALE_MS = 60 * 1000;   // 60s — data considered fresh
const CACHE_MS = 5 * 60 * 1000; // 5 min — keep in cache

export function QueryProvider({ children }: { children: React.ReactNode }) {
    const [queryClient] = useState(
        () =>
            new QueryClient({
                defaultOptions: {
                    queries: {
                        staleTime: STALE_MS,
                        gcTime: CACHE_MS,
                        refetchOnWindowFocus: false,
                        retry: 1,
                    },
                },
            })
    );
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
