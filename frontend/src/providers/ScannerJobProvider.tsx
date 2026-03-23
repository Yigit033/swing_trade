"use client";

import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useRef,
    useState,
    type ReactNode,
} from "react";
import { startSmallcapScanJob, getSmallcapScanJob } from "@/lib/api";

/** Job id — localStorage: yeni sekme / aynı origin’de polling devam eder */
const STORAGE_JOB_KEY = "scanner_active_job_id";
const STORAGE_RESULTS_KEY = "scannerResults";
const EVENT_COMPLETE = "scanner-scan-complete";

function readStoredJobId(): string | null {
    if (typeof window === "undefined") return null;
    try {
        const fromLocal = localStorage.getItem(STORAGE_JOB_KEY);
        if (fromLocal) return fromLocal;
        const legacy = sessionStorage.getItem(STORAGE_JOB_KEY);
        if (legacy) {
            localStorage.setItem(STORAGE_JOB_KEY, legacy);
            sessionStorage.removeItem(STORAGE_JOB_KEY);
            return legacy;
        }
    } catch {
        /* private mode / quota */
    }
    return null;
}

function writeStoredJobId(id: string) {
    if (typeof window === "undefined") return;
    try {
        localStorage.setItem(STORAGE_JOB_KEY, id);
        sessionStorage.removeItem(STORAGE_JOB_KEY);
    } catch {
        /* ignore */
    }
}

function clearStoredJobId() {
    if (typeof window === "undefined") return;
    try {
        localStorage.removeItem(STORAGE_JOB_KEY);
        sessionStorage.removeItem(STORAGE_JOB_KEY);
    } catch {
        /* ignore */
    }
}

export type ScanJobPollState = {
    status: string;
    progress: number;
    phase: string;
    message: string;
    error?: string | null;
};

type ScannerJobContextValue = {
    /** Aktif job var veya poll sürüyor */
    isScanning: boolean;
    poll: ScanJobPollState | null;
    scanError: string | null;
    /** Yeni tarama başlat (409 → mevcut job’a bağlanır) */
    startBackgroundScan: (params: {
        min_quality: number;
        top_n: number;
        portfolio_value: number;
    }) => Promise<void>;
    /** Üst bant / hata mesajını kapat */
    dismissScanFeedback: () => void;
};

const ScannerJobContext = createContext<ScannerJobContextValue | null>(null);

function persistScanResultToSession(payload: {
    signals: unknown[];
    stats: Record<string, unknown>;
    market_regime: string;
}) {
    try {
        sessionStorage.setItem(STORAGE_RESULTS_KEY, JSON.stringify(payload));
    } catch {
        /* quota */
    }
}

export function ScannerJobProvider({ children }: { children: ReactNode }) {
    const [jobId, setJobId] = useState<string | null>(null);
    const [poll, setPoll] = useState<ScanJobPollState | null>(null);
    const [scanError, setScanError] = useState<string | null>(null);
    const jobIdRef = useRef<string | null>(null);
    jobIdRef.current = jobId;

    // Sayfa / sekme açılışında devam eden job (localStorage + eski sessionStorage migrasyonu)
    useEffect(() => {
        const saved = readStoredJobId();
        if (saved) setJobId(saved);
    }, []);

    useEffect(() => {
        if (!jobId) return;

        writeStoredJobId(jobId);

        let cancelled = false;
        let consecutiveErrors = 0;

        const tick = async () => {
            try {
                const d = await getSmallcapScanJob(jobId);
                if (cancelled) return;
                consecutiveErrors = 0;
                setPoll({
                    status: d.status,
                    progress: d.progress ?? 0,
                    phase: d.phase ?? "",
                    message: d.message ?? "",
                    error: d.error,
                });

                if (d.status === "completed" && d.result) {
                    clearStoredJobId();
                    setJobId(null);
                    setScanError(null);
                    persistScanResultToSession(d.result);
                    if (typeof window !== "undefined") {
                        window.dispatchEvent(
                            new CustomEvent(EVENT_COMPLETE, { detail: d.result })
                        );
                    }
                }

                if (d.status === "failed") {
                    clearStoredJobId();
                    setJobId(null);
                    setScanError(
                        d.error === "busy"
                            ? "Başka bir tarama zaten çalışıyor."
                            : d.error || d.message || "Scan failed"
                    );
                }
            } catch (err: unknown) {
                if (cancelled) return;
                const ax = err as { response?: { status?: number } };
                if (ax?.response?.status === 404) {
                    clearStoredJobId();
                    setJobId(null);
                    setPoll(null);
                    setScanError(null);
                    return;
                }
                consecutiveErrors++;
                if (consecutiveErrors >= 5 && jobIdRef.current === jobId) {
                    clearStoredJobId();
                    setJobId(null);
                    setPoll(null);
                    setScanError("Durum alınamadı — ağ veya oturum kontrol edin.");
                }
            }
        };

        void tick();
        const interval = setInterval(() => void tick(), 5000);
        return () => {
            cancelled = true;
            clearInterval(interval);
        };
    }, [jobId]);

    const startBackgroundScan = useCallback(
        async (params: { min_quality: number; top_n: number; portfolio_value: number }) => {
            setScanError(null);
            setPoll(null);
            try {
                const res = await startSmallcapScanJob(params);
                if (res?.job_id) {
                    setJobId(res.job_id);
                    return;
                }
                setScanError("Sunucu job_id dönmedi.");
            } catch (err: unknown) {
                const ax = err as {
                    response?: { status?: number; data?: { active_job_id?: string; detail?: string } };
                };
                if (ax?.response?.status === 409) {
                    const aid = ax.response?.data?.active_job_id;
                    if (aid) {
                        setJobId(aid);
                        return;
                    }
                }
                const msg =
                    ax?.response?.data?.detail ||
                    (err instanceof Error ? err.message : "Scan başlatılamadı");
                setScanError(String(msg));
            }
        },
        []
    );

    const dismissScanFeedback = useCallback(() => {
        setScanError(null);
        if (!jobId) setPoll(null);
    }, [jobId]);

    const value = useMemo<ScannerJobContextValue>(
        () => ({
            isScanning: !!jobId,
            poll,
            scanError,
            startBackgroundScan,
            dismissScanFeedback,
        }),
        [jobId, poll, scanError, startBackgroundScan, dismissScanFeedback]
    );

    return (
        <ScannerJobContext.Provider value={value}>{children}</ScannerJobContext.Provider>
    );
}

export function useScannerJob() {
    const ctx = useContext(ScannerJobContext);
    if (!ctx) {
        throw new Error("useScannerJob must be used within ScannerJobProvider");
    }
    return ctx;
}

export const SCAN_COMPLETE_EVENT = EVENT_COMPLETE;
