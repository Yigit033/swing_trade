"use client";

import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import {
    getPerformance,
    getPending,
    getCurrentRegime,
    getTrades,
    getTradesLastUpdate,
    updatePrices,
} from "@/lib/api";
import type { PerformanceSummary, Trade } from "@/lib/api";

export type PerformanceData = {
    summary: PerformanceSummary;
    open_trades: Trade[];
    recent_closed: Trade[];
};

// Query keys — central place for cache keys
export const queryKeys = {
    performance: ["performance"] as const,
    pending: ["pending"] as const,
    regime: ["regime"] as const,
    trades: ["trades"] as const,
    tradesLastUpdate: ["trades", "lastUpdate"] as const,
};

// Performance: shared by Dashboard & Performance pages
export function usePerformance() {
    return useQuery<PerformanceData>({
        queryKey: queryKeys.performance,
        queryFn: () => getPerformance() as Promise<PerformanceData>,
        staleTime: 60 * 1000,   // 1 min fresh
        gcTime: 5 * 60 * 1000, // 5 min cache
    });
}

// Pending count: Dashboard
export function usePending() {
    return useQuery({
        queryKey: queryKeys.pending,
        queryFn: async () => {
            const d = await getPending();
            return d?.count ?? 0;
        },
        staleTime: 45 * 1000,
        gcTime: 5 * 60 * 1000,
    });
}

// Market regime: Dashboard
export function useRegime() {
    return useQuery({
        queryKey: queryKeys.regime,
        queryFn: () => getCurrentRegime(),
        staleTime: 45 * 1000, // API now live-samples; refresh a bit more often
        gcTime: 5 * 60 * 1000,
    });
}

// Trades: Paper Trades page
export function useTrades() {
    return useQuery<Trade[]>({
        queryKey: queryKeys.trades,
        queryFn: async () => {
            const d = await getTrades();
            return (d.trades ?? []) as Trade[];
        },
        staleTime: 45 * 1000,
        gcTime: 5 * 60 * 1000,
    });
}

// Last price update timestamp
export function useTradesLastUpdate() {
    return useQuery({
        queryKey: queryKeys.tradesLastUpdate,
        queryFn: async () => {
            const d = await getTradesLastUpdate();
            return d.last_update ?? null;
        },
        staleTime: 30 * 1000,
        gcTime: 2 * 60 * 1000,
    });
}

// Update prices mutation — invalidates trades on success
export function useUpdatePrices() {
    const qc = useQueryClient();
    return useMutation({
        mutationFn: () => updatePrices(),
        onSuccess: () => {
            qc.invalidateQueries({ queryKey: queryKeys.trades });
            qc.invalidateQueries({ queryKey: queryKeys.tradesLastUpdate });
        },
    });
}

// Invalidation helpers — call after mutations
export function useInvalidateQueries() {
    const qc = useQueryClient();
    return {
        invalidatePerformance: () => qc.invalidateQueries({ queryKey: queryKeys.performance }),
        invalidatePending: () => qc.invalidateQueries({ queryKey: queryKeys.pending }),
        invalidateTrades: () => {
            qc.invalidateQueries({ queryKey: queryKeys.trades });
            qc.invalidateQueries({ queryKey: queryKeys.tradesLastUpdate });
        },
        invalidateAll: () => {
            qc.invalidateQueries({ queryKey: queryKeys.performance });
            qc.invalidateQueries({ queryKey: queryKeys.pending });
            qc.invalidateQueries({ queryKey: queryKeys.regime });
            qc.invalidateQueries({ queryKey: queryKeys.trades });
            qc.invalidateQueries({ queryKey: queryKeys.tradesLastUpdate });
        },
    };
}
