"use client";

import { useCallback, useEffect, useState, type Dispatch, type SetStateAction, type ReactNode } from "react";
import Link from "next/link";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
    fetchSmallCapSettings,
    resetSmallCapSettings,
    updateSmallCapSettings,
    type SmallCapSettingsJSON,
} from "@/lib/api";
import {
    Settings,
    Save,
    RotateCcw,
    Loader2,
    CheckCircle2,
    AlertCircle,
    Plus,
    Trash2,
    ArrowDownWideNarrow,
    ChevronDown,
} from "lucide-react";

const SWING_TYPES = ["C", "A", "B", "S"] as const;

function formatApiError(e: unknown): string {
    const err = e as { response?: { data?: { detail?: unknown } }; message?: string };
    const d = err?.response?.data?.detail;
    if (Array.isArray(d))
        return d.map((x: { msg?: string; loc?: unknown }) => x.msg || JSON.stringify(x)).join("; ");
    if (typeof d === "string") return d;
    return err?.message || "İstek başarısız";
}

function clone<T>(x: T): T {
    return JSON.parse(JSON.stringify(x));
}

/** Hash hedefi veya alt eleman için tüm üst <details> zincirini aç (iç içe kartlar). */
function openDetailsAncestors(el: HTMLElement | null) {
    let cur: HTMLElement | null = el;
    while (cur) {
        if (cur instanceof HTMLDetailsElement) {
            cur.open = true;
        }
        cur = cur.parentElement;
    }
}

function getNestedNum(draft: Record<string, unknown> | null, path: string[], fallback: number): number {
    if (!draft) return fallback;
    let cur: unknown = draft;
    for (const k of path) {
        if (!cur || typeof cur !== "object" || cur === null) return fallback;
        cur = (cur as Record<string, unknown>)[k];
    }
    return typeof cur === "number" && Number.isFinite(cur) ? cur : fallback;
}

function setNestedNum(
    prev: Record<string, unknown> | null,
    path: string[],
    value: number,
): Record<string, unknown> | null {
    if (!prev) return prev;
    const next = clone(prev);
    const leaf = path[path.length - 1];
    const parents = path.slice(0, -1);
    let cur: Record<string, unknown> = next;
    for (const k of parents) {
        const ex = cur[k];
        if (!ex || typeof ex !== "object" || ex === null || Array.isArray(ex)) cur[k] = {};
        cur = cur[k] as Record<string, unknown>;
    }
    cur[leaf] = value;
    return next;
}

function getNestedBool(draft: Record<string, unknown> | null, path: string[], fallback: boolean): boolean {
    if (!draft) return fallback;
    let cur: unknown = draft;
    for (const k of path) {
        if (!cur || typeof cur !== "object" || cur === null) return fallback;
        cur = (cur as Record<string, unknown>)[k];
    }
    return typeof cur === "boolean" ? cur : fallback;
}

function setNestedBool(prev: Record<string, unknown> | null, path: string[], value: boolean): Record<string, unknown> | null {
    if (!prev) return prev;
    const next = clone(prev);
    const leaf = path[path.length - 1];
    const parents = path.slice(0, -1);
    let cur: Record<string, unknown> = next;
    for (const k of parents) {
        const ex = cur[k];
        if (!ex || typeof ex !== "object" || ex === null || Array.isArray(ex)) cur[k] = {};
        cur = cur[k] as Record<string, unknown>;
    }
    cur[leaf] = value;
    return next;
}

function getTuple1(
    draft: Record<string, unknown> | null,
    pathToTuple: string[],
    index: 0 | 1,
    fallback: number,
): number {
    if (!draft) return fallback;
    let cur: unknown = draft;
    for (const k of pathToTuple) {
        if (!cur || typeof cur !== "object" || cur === null) return fallback;
        cur = (cur as Record<string, unknown>)[k];
    }
    if (!Array.isArray(cur) || cur.length <= index) return fallback;
    const x = cur[index];
    return typeof x === "number" && Number.isFinite(x) ? x : fallback;
}

function getNestedValue(obj: unknown, path: string[]): unknown {
    let cur: unknown = obj;
    for (const k of path) {
        if (!cur || typeof cur !== "object" || cur === null) return undefined;
        cur = (cur as Record<string, unknown>)[k];
    }
    return cur;
}

function replaceNestedLeaf(
    prev: Record<string, unknown> | null,
    path: string[],
    value: unknown,
): Record<string, unknown> | null {
    if (!prev) return prev;
    const next = clone(prev);
    const leaf = path[path.length - 1];
    const parents = path.slice(0, -1);
    let cur: Record<string, unknown> = next;
    for (const k of parents) {
        const ex = cur[k];
        if (!ex || typeof ex !== "object" || ex === null || Array.isArray(ex)) cur[k] = {};
        cur = cur[k] as Record<string, unknown>;
    }
    cur[leaf] = value;
    return next;
}

function parseNumericObjectRows(raw: unknown, keys: string[]): Record<string, number>[] {
    if (!Array.isArray(raw)) return [];
    return raw.map((item) => {
        const o = typeof item === "object" && item !== null ? (item as Record<string, unknown>) : {};
        const row: Record<string, number> = {};
        for (const k of keys) {
            const v = o[k];
            row[k] = typeof v === "number" && Number.isFinite(v) ? v : 0;
        }
        return row;
    });
}

function setTuple1(
    prev: Record<string, unknown> | null,
    pathToTuple: string[],
    index: 0 | 1,
    value: number,
): Record<string, unknown> | null {
    if (!prev) return prev;
    const next = clone(prev);
    const leaf = pathToTuple[pathToTuple.length - 1];
    const parents = pathToTuple.slice(0, -1);
    let cur: Record<string, unknown> = next;
    for (const k of parents) {
        const ex = cur[k];
        if (!ex || typeof ex !== "object" || ex === null || Array.isArray(ex)) cur[k] = {};
        cur = cur[k] as Record<string, unknown>;
    }
    const existing = cur[leaf];
    const tuple =
        Array.isArray(existing) && existing.length >= 2
            ? [Number(existing[0]), Number(existing[1])]
            : [1, 2];
    tuple[index] = value;
    cur[leaf] = tuple;
    return next;
}

function FieldBool({
    label,
    hint,
    value,
    onChange,
}: {
    label: string;
    hint?: string;
    value: boolean;
    onChange: (b: boolean) => void;
}) {
    return (
        <div style={{ marginBottom: 14 }}>
            <label style={{ display: "flex", alignItems: "flex-start", gap: 10, cursor: "pointer" }}>
                <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} style={{ marginTop: 3 }} />
                <span>
                    <span style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontWeight: 600, display: "block" }}>{label}</span>
                    {hint && <span style={{ fontSize: "0.74rem", color: "var(--text-secondary)", display: "block", marginTop: 4 }}>{hint}</span>}
                </span>
            </label>
        </div>
    );
}

function FieldNum({
    label,
    hint,
    value,
    onChange,
    step = "any",
    min,
    max,
}: {
    label: string;
    hint?: string;
    value: number;
    onChange: (n: number) => void;
    step?: string | number;
    min?: number;
    max?: number;
}) {
    const stepAttr = step === "any" ? "any" : String(step);
    return (
        <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontWeight: 600, marginBottom: 5 }}>
                {label}
            </div>
            {hint && (
                <div style={{ fontSize: "0.74rem", color: "var(--text-secondary)", marginBottom: 7, lineHeight: 1.45 }}>{hint}</div>
            )}
            <input
                type="number"
                value={Number.isFinite(value) ? value : 0}
                step={stepAttr}
                min={min !== undefined ? String(min) : undefined}
                max={max !== undefined ? String(max) : undefined}
                onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    onChange(Number.isFinite(v) ? v : 0);
                }}
                style={{
                    width: "100%",
                    maxWidth: 260,
                    padding: "9px 12px",
                    fontSize: "0.875rem",
                    borderRadius: 8,
                    border: "1px solid var(--border)",
                    background: "var(--bg-surface)",
                    color: "var(--text-primary)",
                }}
            />
        </div>
    );
}

function ScoringTierTables({
    draft,
    setDraft,
}: {
    draft: Record<string, unknown>;
    setDraft: Dispatch<SetStateAction<SmallCapSettingsJSON | null>>;
}) {
    const cell: React.CSSProperties = { padding: "8px 10px", borderBottom: "1px solid var(--border)", verticalAlign: "middle" as const };
    const inp: React.CSSProperties = {
        width: "100%",
        minWidth: 72,
        maxWidth: 130,
        padding: "6px 8px",
        borderRadius: 6,
        border: "1px solid var(--border)",
        background: "var(--bg-surface)",
        color: "var(--text-primary)",
    };

    const setArray = (path: string[], rows: Record<string, number>[]) => {
        setDraft((prev) => replaceNestedLeaf(prev as Record<string, unknown> | null, path, rows));
    };

    const tableBlock = (
        title: string,
        hint: string,
        path: string[],
        keys: [string, string],
        labels: [string, string],
        defaultRow: Record<string, number>,
        sortRows: (r: Record<string, number>[]) => Record<string, number>[],
        stepA: string,
        stepB: string,
    ) => {
        const raw = getNestedValue(draft, path);
        let rows = parseNumericObjectRows(raw, [keys[0], keys[1]]);
        if (rows.length === 0) rows = [{ ...defaultRow }];

        const push = (next: Record<string, number>[]) => setArray(path, next);
        const update = (i: number, key: string, val: number) => {
            push(rows.map((r, j) => (j === i ? { ...r, [key]: val } : r)));
        };
        const del = (i: number) => {
            if (rows.length <= 1) return;
            push(rows.filter((_, j) => j !== i));
        };
        const add = () => push([...rows, { ...defaultRow }]);
        const sort = () => push(sortRows(rows.map((r) => ({ ...r }))));

        return (
            <div style={{ marginBottom: 22 }}>
                <div style={{ fontSize: "0.86rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: 6 }}>{title}</div>
                <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", margin: "0 0 12px", lineHeight: 1.5 }}>{hint}</p>
                <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem" }}>
                        <thead>
                            <tr style={{ color: "var(--text-muted)", fontWeight: 600, textAlign: "left" as const }}>
                                <th style={{ ...cell, width: 40 }}>#</th>
                                <th style={cell}>{labels[0]}</th>
                                <th style={cell}>{labels[1]}</th>
                                <th style={{ ...cell, width: 88 }} />
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map((r, i) => (
                                <tr key={`${path.join(".")}-${i}`}>
                                    <td style={cell}>{i + 1}</td>
                                    <td style={cell}>
                                        <input
                                            type="number"
                                            value={r[keys[0]]}
                                            step={stepA}
                                            onChange={(e) => {
                                                const v = parseFloat(e.target.value);
                                                update(i, keys[0], Number.isFinite(v) ? v : 0);
                                            }}
                                            style={inp}
                                        />
                                    </td>
                                    <td style={cell}>
                                        <input
                                            type="number"
                                            value={r[keys[1]]}
                                            step={stepB}
                                            onChange={(e) => {
                                                const v = parseFloat(e.target.value);
                                                update(i, keys[1], Number.isFinite(v) ? v : 0);
                                            }}
                                            style={inp}
                                        />
                                    </td>
                                    <td style={cell}>
                                        <button
                                            type="button"
                                            className="btn-secondary"
                                            style={{ padding: "6px 8px", display: "inline-flex", alignItems: "center" }}
                                            onClick={() => del(i)}
                                            disabled={rows.length <= 1}
                                            title="Satırı sil"
                                        >
                                            <Trash2 size={15} />
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 10 }}>
                    <button type="button" className="btn-secondary" style={{ display: "inline-flex", alignItems: "center", gap: 6 }} onClick={add}>
                        <Plus size={16} /> Satır ekle
                    </button>
                    <button type="button" className="btn-secondary" style={{ display: "inline-flex", alignItems: "center", gap: 6 }} onClick={sort}>
                        <ArrowDownWideNarrow size={16} /> Önerilen sıra
                    </button>
                </div>
            </div>
        );
    };

    return (
        <div>
            {tableBlock(
                "Volume surge kademeleri",
                "volume_surge ≥ min_surge olan en büyük eşik seçilir (motor azalan min_surge ile tarar).",
                ["scoring_tuning", "volume_surge_tiers"],
                ["min_surge", "score"],
                ["Min. surge (×)", "Ham puan"],
                { min_surge: 1, score: 0 },
                (r) => [...r].sort((a, b) => b.min_surge - a.min_surge),
                "0.1",
                "1",
            )}
            {tableBlock(
                "ATR % kademeleri (ATR/close)",
                "atr_percent ≥ min_atr_frac olan en büyük eşik seçilir. Örn. 0.15 = %15 volatilite.",
                ["scoring_tuning", "atr_percent_tiers"],
                ["min_atr_frac", "score"],
                ["Min. ATR oranı", "Ham puan"],
                { min_atr_frac: 0.03, score: 0 },
                (r) => [...r].sort((a, b) => b.min_atr_frac - a.min_atr_frac),
                "0.005",
                "1",
            )}
            {tableBlock(
                "Float bantları (milyon hisse, üst sınır dahil)",
                "float_millions ≤ max_millions_le olan en küçük bant seçilir; son banttan büyükse ceza puanı kullanılır.",
                ["scoring_tuning", "float_millions_bands"],
                ["max_millions_le", "score"],
                ["Max float (M$)", "Ham puan"],
                { max_millions_le: 100, score: 0 },
                (r) => [...r].sort((a, b) => a.max_millions_le - b.max_millions_le),
                "1",
                "1",
            )}
            <div
                style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
                    gap: "8px 22px",
                    marginTop: 4,
                }}
            >
                <FieldNum
                    label="Float bilinmiyor (ham puan)"
                    hint="float yok veya ≤0"
                    value={getNestedNum(draft, ["scoring_tuning", "float_score_unknown"], 5)}
                    onChange={(v) =>
                        setDraft((p) => setNestedNum(p as Record<string, unknown> | null, ["scoring_tuning", "float_score_unknown"], v))
                    }
                    step="0.5"
                    min={-50}
                    max={50}
                />
                <FieldNum
                    label="Son bant üstü float (ham puan)"
                    hint="Tüm max_millions_le değerlerinden büyük float"
                    value={getNestedNum(draft, ["scoring_tuning", "float_score_above_max_band"], -8)}
                    onChange={(v) =>
                        setDraft((p) => setNestedNum(p as Record<string, unknown> | null, ["scoring_tuning", "float_score_above_max_band"], v))
                    }
                    step="0.5"
                    min={-50}
                    max={50}
                />
            </div>
        </div>
    );
}

const sectionHelpBtn: React.CSSProperties = {
    flexShrink: 0,
    width: 22,
    height: 22,
    borderRadius: "50%",
    border: "1px solid var(--border)",
    background: "var(--bg-surface)",
    color: "var(--text-muted)",
    fontSize: "0.75rem",
    fontWeight: 700,
    cursor: "pointer",
    lineHeight: 1,
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
};

const sectionHelpPanel: React.CSSProperties = {
    marginTop: 10,
    marginBottom: 2,
    fontSize: "0.8rem",
    color: "var(--text-secondary)",
    lineHeight: 1.55,
    padding: "10px 12px",
    borderRadius: 8,
    border: "1px solid var(--border)",
    background: "rgba(59,130,246,0.07)",
};

/** Üst şerit: gruba atla (içerik sırası değişmez; yalnızca kaydırır). */
type SettingsNavGroupId = "filters" | "universe" | "risk" | "regime" | "scoring" | "backtest";

const SETTINGS_NAV_GROUPS: { id: SettingsNavGroupId; label: string; scrollToId: string }[] = [
    { id: "filters", label: "Filtreler & giriş", scrollToId: "settings-section-sinyal-filtresi" },
    { id: "universe", label: "Evren", scrollToId: "settings-section-evren-filtreleri" },
    { id: "risk", label: "Risk & hedefler", scrollToId: "settings-section-risk-yonetimi" },
    { id: "regime", label: "Rejim", scrollToId: "settings-section-rejim-min-kalite" },
    { id: "scoring", label: "Skor & tip", scrollToId: "settings-section-skorlama" },
    { id: "backtest", label: "Backtest motoru", scrollToId: "settings-section-backtest-dongu" },
];

const HASH_TO_NAV_GROUP: Partial<Record<string, SettingsNavGroupId>> = {
    "settings-ayar-rehberi-giris": "filters",
    "settings-section-sinyal-filtresi": "filters",
    "settings-section-tarama-gecitleri": "filters",
    "settings-section-swing-hazirlik": "filters",
    "settings-section-backtest-giris-yurutme": "filters",
    "settings-section-evren-filtreleri": "universe",
    "settings-section-finviz-evren": "universe",
    "settings-section-risk-yonetimi": "risk",
    "settings-section-hedefler-atr": "risk",
    "settings-section-risk-hedefleri-rejim": "risk",
    "settings-section-rejim-min-kalite": "regime",
    "settings-section-skorlama": "scoring",
    "settings-details-skorlama-kademe-tablolari": "scoring",
    "settings-details-skorlama-momentum-ham": "scoring",
    "settings-section-swing-siniflandirma-gelismis": "scoring",
    "settings-section-backtest-dongu": "backtest",
    "settings-section-backtest-tip-kalitesi": "backtest",
    "settings-section-backtest-giris-ema-gap": "backtest",
    "settings-section-backtest-cikis-trailing": "backtest",
};

/** Başlık + isteğe bağlı ? ile açılan kısa bölüm notu (tuning kılavuzu). */
function SectionTitleWithHelp({
    title,
    help,
    insideSummary,
}: {
    title: string;
    help?: string;
    /** `<summary>` içindeyken ? tıklaması üst <details>’i kapatmayı tetiklemesin. */
    insideSummary?: boolean;
}) {
    const [open, setOpen] = useState(false);
    const toggleHelp = (e: React.MouseEvent) => {
        if (insideSummary) {
            e.preventDefault();
            e.stopPropagation();
        }
        setOpen((o) => !o);
    };
    return (
        <div style={{ marginBottom: insideSummary ? 0 : 14 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <h2 style={{ fontSize: "0.94rem", fontWeight: 700, margin: 0, flex: 1, color: "var(--text-primary)", letterSpacing: "-0.02em" }}>{title}</h2>
                {help ? (
                    <button
                        type="button"
                        onClick={toggleHelp}
                        onPointerDown={insideSummary ? (e) => e.stopPropagation() : undefined}
                        aria-expanded={open}
                        aria-label={`${title}: bu bölüm ne işe yarar?`}
                        title="Kısa açıklama"
                        style={sectionHelpBtn}
                    >
                        ?
                    </button>
                ) : null}
            </div>
            {help && open ? <div style={{ ...sectionHelpPanel, marginTop: insideSummary ? 8 : 10 }}>{help}</div> : null}
        </div>
    );
}

/** Kapalı bir blok (ör. details) içinde: sadece ? + açılır metin. */
function BlockHelp({ text, stopSummaryToggle }: { text: string; stopSummaryToggle?: boolean }) {
    const [open, setOpen] = useState(false);
    const onToggle = (e: React.MouseEvent) => {
        if (stopSummaryToggle) {
            e.preventDefault();
            e.stopPropagation();
        }
        setOpen((o) => !o);
    };
    return (
        <div style={{ margin: "4px 0 12px" }}>
            <button
                type="button"
                onClick={onToggle}
                onPointerDown={stopSummaryToggle ? (e) => e.stopPropagation() : undefined}
                aria-expanded={open}
                aria-label="Bu bölüm ne işe yarar?"
                title="Kısa açıklama"
                style={sectionHelpBtn}
            >
                ?
            </button>
            {open ? (
                <div style={{ ...sectionHelpPanel, marginTop: 10 }}>
                    {text}
                </div>
            ) : null}
        </div>
    );
}

function Section({
    title,
    help,
    children,
    id,
}: {
    title: string;
    /** Tıklanınca ? ile açılan kısa metin (ne zaman dokunulur, ne etkiler). */
    help?: string;
    children: ReactNode;
    /** Sayfa içi bağlantı (/settings#...) için; Nasıl çalışır? rehberi kullanır. */
    id?: string;
}) {
    return (
        <details
            id={id}
            className="glass-card settings-section-details"
            style={{ padding: 0, marginBottom: 18, scrollMarginTop: 92 }}
        >
            <summary
                style={{
                    cursor: "pointer",
                    padding: "16px 20px",
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 12,
                    userSelect: "none",
                }}
            >
                <ChevronDown size={18} className="settings-section-chevron" aria-hidden style={{ marginTop: 2 }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                    <SectionTitleWithHelp title={title} help={help} insideSummary />
                </div>
            </summary>
            <div
                style={{
                    padding: "0 20px 20px 22px",
                    borderTop: "1px solid var(--border-muted)",
                }}
            >
                <div
                    style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
                        gap: "8px 22px",
                        paddingTop: 14,
                    }}
                >
                    {children}
                </div>
            </div>
        </details>
    );
}

export default function SettingsPage() {
    const qc = useQueryClient();
    const { data, isLoading, isError, error, refetch } = useQuery({
        queryKey: ["smallcap-settings"],
        queryFn: fetchSmallCapSettings,
    });

    const [draft, setDraft] = useState<SmallCapSettingsJSON | null>(null);
    const [banner, setBanner] = useState<{ type: "ok" | "err"; text: string } | null>(null);
    const [settingsNavGroup, setSettingsNavGroup] = useState<SettingsNavGroupId>("filters");

    useEffect(() => {
        if (data) setDraft(clone(data));
    }, [data]);

    /** Hash: ilgili <details> aç, üst grup şeridini eşle (how-it-works / dahili linkler). */
    useEffect(() => {
        const syncHash = () => {
            const hash = typeof window !== "undefined" ? window.location.hash.slice(1) : "";
            if (hash && HASH_TO_NAV_GROUP[hash]) {
                setSettingsNavGroup(HASH_TO_NAV_GROUP[hash]!);
            }
            if (!hash) return;
            const el = document.getElementById(hash);
            if (el) openDetailsAncestors(el);
        };
        syncHash();
        window.addEventListener("hashchange", syncHash);
        return () => window.removeEventListener("hashchange", syncHash);
    }, []);

    const scrollToSettingsGroup = useCallback((scrollToId: string, groupId: SettingsNavGroupId) => {
        setSettingsNavGroup(groupId);
        window.history.replaceState(null, "", `#${scrollToId}`);
        const el = document.getElementById(scrollToId);
        if (el) openDetailsAncestors(el);
        el?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, []);

    const saveMut = useMutation({
        mutationFn: updateSmallCapSettings,
        onSuccess: (res) => {
            setDraft(clone(res.settings));
            qc.setQueryData(["smallcap-settings"], res.settings);
            setBanner({ type: "ok", text: "Ayarlar kaydedildi. Sonraki tarama yeni parametreleri kullanır." });
        },
        onError: (e) => {
            setBanner({ type: "err", text: formatApiError(e) });
        },
    });

    const resetMut = useMutation({
        mutationFn: resetSmallCapSettings,
        onSuccess: (res) => {
            setDraft(clone(res.settings));
            qc.setQueryData(["smallcap-settings"], res.settings);
            setBanner({ type: "ok", text: "Varsayılan ayarlar yüklendi ve kaydedildi." });
        },
        onError: (e) => {
            setBanner({ type: "err", text: formatApiError(e) });
        },
    });

    const onSave = useCallback(() => {
        if (!draft) return;
        setBanner(null);
        saveMut.mutate(draft);
    }, [draft, saveMut]);

    const onReset = useCallback(() => {
        if (!window.confirm("Tüm small-cap ayarları fabrika varsayılanlarına dönsün mü? Bu işlem diske yazılır.")) {
            return;
        }
        setBanner(null);
        resetMut.mutate();
    }, [resetMut]);

    const num = (k: string, d = 0) => {
        const v = draft?.[k];
        return typeof v === "number" && Number.isFinite(v) ? v : d;
    };

    const setNum = (k: string, v: number) => {
        setDraft((prev) => (prev ? { ...prev, [k]: v } : prev));
    };

    const dictNum = (top: string, key: string, d = 0) => {
        const o = draft?.[top];
        if (!o || typeof o !== "object" || o === null) return d;
        const v = (o as Record<string, unknown>)[key];
        return typeof v === "number" && Number.isFinite(v) ? v : d;
    };

    const setDictNum = (top: string, key: string, v: number) => {
        setDraft((prev) => {
            if (!prev) return prev;
            const cur = prev[top];
            const base =
                cur && typeof cur === "object" && cur !== null
                    ? { ...(cur as Record<string, unknown>) }
                    : {};
            return { ...prev, [top]: { ...base, [key]: v } };
        });
    };

    const capNum = (t: string, field: "t1_max_pct" | "t2_max_pct", d = 0) => {
        const caps = draft?.type_target_caps;
        if (!caps || typeof caps !== "object") return d;
        const row = (caps as Record<string, unknown>)[t];
        if (!row || typeof row !== "object") return d;
        const v = (row as Record<string, unknown>)[field];
        return typeof v === "number" && Number.isFinite(v) ? v : d;
    };

    const setCapNum = (t: string, field: "t1_max_pct" | "t2_max_pct", v: number) => {
        setDraft((prev) => {
            if (!prev) return prev;
            const caps = { ...((prev.type_target_caps as object) || {}) } as Record<string, Record<string, number>>;
            const row = { ...(caps[t] || {}) };
            row[field] = v;
            caps[t] = row;
            return { ...prev, type_target_caps: caps };
        });
    };

    if (isLoading || !draft) {
        return (
            <div style={{ padding: 48, display: "flex", alignItems: "center", gap: 12, color: "var(--text-secondary)" }}>
                <Loader2 className="animate-spin" size={22} />
                Ayarlar yükleniyor…
            </div>
        );
    }

    if (isError) {
        return (
            <div className="glass-card" style={{ padding: 24, margin: 24 }}>
                <p style={{ color: "var(--red)", marginBottom: 12 }}>{formatApiError(error)}</p>
                <button type="button" className="btn-primary" onClick={() => refetch()}>
                    Tekrar dene
                </button>
            </div>
        );
    }

    const draftRec = draft as Record<string, unknown>;
    const nn = (path: string[], fb: number) => getNestedNum(draftRec, path, fb);
    const sn = (path: string[], v: number) => setDraft((p) => setNestedNum(p as Record<string, unknown> | null, path, v));
    const nb = (path: string[], fb: boolean) => getNestedBool(draftRec, path, fb);
    const sb = (path: string[], v: boolean) => setDraft((p) => setNestedBool(p as Record<string, unknown> | null, path, v));
    const nt = (path: string[], i: 0 | 1, fb: number) => getTuple1(draftRec, path, i, fb);
    const st = (path: string[], i: 0 | 1, v: number) =>
        setDraft((p) => setTuple1(p as Record<string, unknown> | null, path, i, v));

    return (
        <div style={{ maxWidth: 1040, padding: "6px 6px 40px" }}>
            <h1 className="page-title gradient-text" style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <Settings size={28} />
                Small-cap ayarları
            </h1>
            <p className="page-subtitle">
                Parametreler <code style={{ fontSize: "0.85em" }}>data/smallcap_settings.json</code> dosyasına yazılır; kayıttan sonra
                önbellekteki motor yenilenir. Backtest her koşuda dosyadan okur. SmallCap Scanner ve tek-ticker analiz bu dosyayı kullanır;
                kaydettiğiniz değerler bir sonraki taramada kodda sabitmiş gibi uygulanır (motor önbelleği temizlenir).
            </p>

            <details
                id="settings-ayar-rehberi-giris"
                className="glass-card settings-section-details"
                style={{
                    marginBottom: 20,
                    borderColor: "rgba(59,130,246,0.22)",
                    scrollMarginTop: 92,
                    padding: 0,
                }}
            >
                <summary
                    style={{
                        cursor: "pointer",
                        padding: "16px 20px",
                        display: "flex",
                        alignItems: "flex-start",
                        gap: 12,
                        userSelect: "none",
                    }}
                >
                    <ChevronDown size={18} className="settings-section-chevron" aria-hidden style={{ marginTop: 2 }} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontWeight: 700, color: "var(--text-primary)", fontSize: "0.93rem", lineHeight: 1.35 }}>
                            Zarar ediyorsanız — hangi ayar neyi etkiler?
                        </div>
                        <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: 6 }}>
                            Pipeline özeti, Nasıl çalışır? ve dört ana ayar grubunun etkisi
                        </div>
                    </div>
                </summary>
                <div
                    style={{
                        padding: "0 20px 18px 22px",
                        borderTop: "1px solid var(--border-muted)",
                        fontSize: "0.84rem",
                        lineHeight: 1.6,
                        color: "var(--text-secondary)",
                    }}
                >
                    <div style={{ marginBottom: 12, paddingTop: 14 }}>
                        <Link
                            href="/how-it-works"
                            style={{ color: "var(--accent)", fontWeight: 600, fontSize: "0.84rem" }}
                        >
                            Scan → Track pipeline özeti ve semptom bazlı rehber (Nasıl çalışır?) →
                        </Link>
                    </div>
                    <ul style={{ margin: 0, paddingLeft: 18 }}>
                        <li style={{ marginBottom: 8 }}>
                            <strong style={{ color: "var(--text-primary)" }}>Aday havuzu (kim hiç listelenmez)</strong> — Finviz evren
                            taraması (hangi ~N hisse taranır), evren filtreleri, sinyal filtresi (RSI / volume / ATR), tarama geçitleri,
                            swing sınıflandırma (S/C/B/A). Çok sıkıysa fırsat azalır; çok gevşekse kalite düşer.
                        </li>
                        <li style={{ marginBottom: 8 }}>
                            <strong style={{ color: "var(--text-primary)" }}>Skor ve sıralama (kim öne çıkar)</strong> — Skorlama (tablolar + ağırlık
                            / bonus / ceza), rejim tabanları. Scanner ekranındaki min kalite ve top N bu katmanın üstüne uygulanır.
                        </li>
                        <li style={{ marginBottom: 8 }}>
                            <strong style={{ color: "var(--text-primary)" }}>Stop ve hedef</strong> — Risk yönetimi, hedef ATR / tavan %, risk hedef
                            rejimi. İşlem planınız buradan şekillenir.
                        </li>
                        <li>
                            <strong style={{ color: "var(--text-primary)" }}>Backtest davranışı</strong> — Backtest bölümleri (döngü, giriş, çıkış,
                            tip kalite zeminleri) yürütmeyi ayarlar; canlı tarama ile aynı dosyayı paylaşır.
                        </li>
                    </ul>
                    <p style={{ margin: "14px 0 0", fontSize: "0.78rem", color: "var(--text-muted)" }}>
                        <code style={{ fontSize: "0.85em" }}>config.yaml</code> ve Streamlit dashboard özel parametreleri bu sayfada değildir;
                        small-cap motorunun JSON ile yönetilen kısmı burada toplanmıştır.
                    </p>
                </div>
            </details>

            {banner && (
                <div
                    className="glass-card"
                    style={{
                        padding: "10px 14px",
                        marginBottom: 14,
                        display: "flex",
                        alignItems: "center",
                        gap: 10,
                        borderColor: banner.type === "ok" ? "rgba(34,197,94,0.35)" : "rgba(239,68,68,0.35)",
                    }}
                >
                    {banner.type === "ok" ? (
                        <CheckCircle2 size={20} color="var(--green)" />
                    ) : (
                        <AlertCircle size={20} color="var(--red)" />
                    )}
                    <span style={{ fontSize: "0.88rem" }}>{banner.text}</span>
                </div>
            )}

            <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginBottom: 20, alignItems: "center" }}>
                <button
                    type="button"
                    className="btn-primary"
                    onClick={onSave}
                    disabled={saveMut.isPending}
                    style={{ display: "inline-flex", alignItems: "center", gap: 8 }}
                >
                    {saveMut.isPending ? <Loader2 className="animate-spin" size={18} /> : <Save size={18} />}
                    Kaydet
                </button>
                <button
                    type="button"
                    className="btn-secondary"
                    onClick={onReset}
                    disabled={resetMut.isPending}
                    style={{ display: "inline-flex", alignItems: "center", gap: 8 }}
                >
                    {resetMut.isPending ? <Loader2 className="animate-spin" size={18} /> : <RotateCcw size={18} />}
                    Varsayılana dön
                </button>
            </div>

            <nav
                aria-label="Ayar grupları"
                style={{
                    position: "sticky",
                    top: 0,
                    zIndex: 6,
                    marginBottom: 22,
                    padding: "14px 18px 16px",
                    borderRadius: 12,
                    border: "1px solid var(--border-muted)",
                    background: "rgba(10,15,30,0.82)",
                    backdropFilter: "blur(14px)",
                    WebkitBackdropFilter: "blur(14px)",
                    boxShadow: "0 12px 40px rgba(0,0,0,0.25)",
                }}
            >
                <div
                    style={{
                        fontSize: "0.72rem",
                        fontWeight: 700,
                        color: "var(--text-muted)",
                        textTransform: "uppercase",
                        letterSpacing: "0.08em",
                        marginBottom: 10,
                    }}
                >
                    Gruplar
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                    {SETTINGS_NAV_GROUPS.map((g) => {
                        const active = settingsNavGroup === g.id;
                        return (
                            <button
                                key={g.id}
                                type="button"
                                role="tab"
                                aria-selected={active}
                                onClick={() => scrollToSettingsGroup(g.scrollToId, g.id)}
                                style={{
                                    padding: "9px 16px",
                                    borderRadius: 999,
                                    border: `1px solid ${active ? "rgba(59,130,246,0.5)" : "var(--border)"}`,
                                    background: active ? "rgba(59,130,246,0.18)" : "rgba(26,34,53,0.9)",
                                    color: active ? "var(--text-primary)" : "var(--text-secondary)",
                                    fontSize: "0.82rem",
                                    fontWeight: 600,
                                    cursor: "pointer",
                                    lineHeight: 1.25,
                                    transition: "border-color 0.15s, background 0.15s",
                                }}
                            >
                                {g.label}
                            </button>
                        );
                    })}
                </div>
                <p style={{ margin: "10px 0 0", fontSize: "0.76rem", color: "var(--text-muted)", lineHeight: 1.5 }}>
                    İçerik aynı sırayla aşağıda; gruplar ilgili bölüme kaydırır.
                </p>
            </nav>

            <Section
                id="settings-section-sinyal-filtresi"
                title="Sinyal filtresi"
                help="Tarama öncesi sert kapı: RSI üst sınırı (Tip S muaf), min volume surge ve min ATR%. Çok sıkı → çok az aday; çok gevşek → zayıf kurulumlar listelenir. Önce burayı, sonra geçitleri oynatın."
            >
                <FieldNum
                    label="RSI üst sınırı (S hariç red)"
                    hint="RSI bu değerin üstündeyse (Tip S hariç) aday elenir. Çok düşük → az işlem; çok yüksek → aşırı alım kurulumları geçer."
                    value={num("max_entry_rsi", 70)}
                    onChange={(v) => setNum("max_entry_rsi", v)}
                    min={30}
                    max={95}
                />
                <FieldNum
                    label="Volume surge tetik (×)"
                    hint="Bugünkü hacmin 20g ortalamasına oranı bu değerin altındaysa tetikleyici geçmez (sert eşik)."
                    value={num("volume_surge_trigger", 1.5)}
                    onChange={(v) => setNum("volume_surge_trigger", v)}
                    step="0.1"
                    min={1}
                    max={5}
                />
                <FieldNum
                    label="Volume surge (soft min, ×)"
                    hint="Mesaj / yumuşak uyarı için alt referans; tetikleyici kadar sert değildir."
                    value={num("min_volume_surge_soft", 1.2)}
                    onChange={(v) => setNum("min_volume_surge_soft", v)}
                    step="0.1"
                    min={1}
                    max={3}
                />
                <FieldNum
                    label="Min ATR % (ondalık, örn. 0.03 = %3)"
                    hint="Filtre + tetikleyici"
                    value={num("min_atr_percent", 0.03)}
                    onChange={(v) => setNum("min_atr_percent", v)}
                    step="0.005"
                    min={0.01}
                    max={0.2}
                />
            </Section>

            <Section
                id="settings-section-risk-yonetimi"
                title="Risk yönetimi"
                help="İşlem başına portföy riski, stop (ATR çarpanı ve min/max %), günlük kayıp ve tip bazlı stop tavanları. Sık stop yiyorsanız sadece burayı gevşetmeyin: giriş kalitesi ve rejim de kritik. Küçük adımla değiştirin."
            >
                <FieldNum
                    label="İşlem başına risk (portföy oranı)"
                    hint="Örn. 0.015 = %1.5"
                    value={num("max_risk_per_trade", 0.015)}
                    onChange={(v) => setNum("max_risk_per_trade", v)}
                    step="0.001"
                    min={0.001}
                    max={0.1}
                />
                <FieldNum
                    label="Stop ATR çarpanı"
                    hint="Stop mesafesi ≈ ATR × bu çarpan (tip bazlı tavanlar ayrıca uygulanır)."
                    value={num("stop_atr_multiplier", 1.5)}
                    onChange={(v) => setNum("stop_atr_multiplier", v)}
                    step="0.1"
                    min={0.5}
                    max={5}
                />
                <FieldNum
                    label="Min stop %"
                    hint="Hesaplanan stop çok darsa en az bu yüzde kullanılır (gürültü stop’u önler)."
                    value={num("min_stop_percent", 0.03)}
                    onChange={(v) => setNum("min_stop_percent", v)}
                    step="0.005"
                    min={0.01}
                    max={0.25}
                />
                <FieldNum
                    label="Max stop % (fallback tip bilinmezse)"
                    hint="Swing tipi çıkmadığında kullanılan üst sınır."
                    value={num("max_stop_percent_fallback", 0.08)}
                    onChange={(v) => setNum("max_stop_percent_fallback", v)}
                    step="0.01"
                    min={0.03}
                    max={0.3}
                />
                <FieldNum
                    label="Max tutma (gün)"
                    hint="Motorun planladığı maksimum tutma süresi (çıkış kuralları bununla birlikte çalışır)."
                    value={num("max_holding_days", 14)}
                    onChange={(v) => setNum("max_holding_days", Math.round(v))}
                    step={1}
                    min={1}
                    max={60}
                />
                {SWING_TYPES.map((t) => (
                    <FieldNum
                        key={`ms-${t}`}
                        label={`Max stop % — tip ${t}`}
                        hint="Bu tip için stop mesafesi girişe göre en fazla bu yüzde olur."
                        value={dictNum("max_stop_by_type", t, 0.08)}
                        onChange={(v) => setDictNum("max_stop_by_type", t, v)}
                        step="0.01"
                        min={0.03}
                        max={0.3}
                    />
                ))}
                {SWING_TYPES.map((t) => (
                    <FieldNum
                        key={`pc-${t}`}
                        label={`Pozisyon tavanı (portföy %) — ${t}`}
                        hint="Tek işlemde bu tipe ayrılabilecek portföy oranı üst sınırı."
                        value={dictNum("type_position_caps", t, 0.2)}
                        onChange={(v) => setDictNum("type_position_caps", t, v)}
                        step="0.05"
                        min={0.05}
                        max={0.5}
                    />
                ))}
            </Section>

            <Section
                id="settings-section-hedefler-atr"
                title="Hedefler (ATR + tavan %)"
                help="T1/T2 mesafeleri (ATR çarpanı) ve her swing tipi için hedef tavan yüzdeleri. Hedefler çok uzaksa fiyat sık erişemez; çok yakınsa R azalır. Backtest ve canlı planlama aynı mantığı paylaşır."
            >
                <FieldNum
                    label="T2 ATR oranı (baz)"
                    hint="T2 mesafesi T1’e göre ölçeklenirken kullanılan çarpan (tip + tavanlar ile sınırlanır)."
                    value={num("t2_atr_ratio", 2)}
                    onChange={(v) => setNum("t2_atr_ratio", v)}
                    step="0.1"
                    min={1}
                    max={4}
                />
                {SWING_TYPES.map((t) => (
                    <FieldNum
                        key={`am-${t}`}
                        label={`T1 ATR çarpanı — ${t}`}
                        hint="T1 hedefi ≈ ATR × bu çarpan; ardından tavan % uygulanır."
                        value={dictNum("type_atr_multipliers", t, 1.8)}
                        onChange={(v) => setDictNum("type_atr_multipliers", t, v)}
                        step="0.1"
                        min={0.5}
                        max={5}
                    />
                ))}
                {SWING_TYPES.map((t) => (
                    <div key={`tc-${t}`} style={{ gridColumn: "1 / -1" }}>
                        <div style={{ fontWeight: 600, fontSize: "0.84rem", marginBottom: 8, color: "var(--text-secondary)" }}>
                            Hedef tavan % — {t}
                        </div>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 24 }}>
                            <FieldNum
                                label="T1 max %"
                                value={capNum(t, "t1_max_pct", 0.1)}
                                onChange={(v) => setCapNum(t, "t1_max_pct", v)}
                                step="0.01"
                                min={0.01}
                                max={0.5}
                            />
                            <FieldNum
                                label="T2 max %"
                                value={capNum(t, "t2_max_pct", 0.18)}
                                onChange={(v) => setCapNum(t, "t2_max_pct", v)}
                                step="0.01"
                                min={0.01}
                                max={0.8}
                            />
                        </div>
                    </div>
                ))}
            </Section>

            <Section
                id="settings-section-rejim-min-kalite"
                title="Rejim (min kalite / top N tavanı)"
                help="BULL / BEAR / CAUTION için minimum kalite puanı ve listelenen sinyal sayısı tavanı. Ayı veya temkinli rejimde daha seçici olmak için bu değerleri yükseltin; çok sinyal ama düşük kalite varsa buradan sıkılaştırın."
            >
                {(
                    [
                        ["bear_tentative_min_quality", "BEAR + TENTATIVE min kalite"],
                        ["bear_tentative_top_n_max", "BEAR + TENTATIVE top N max"],
                        ["bear_confirmed_min_quality", "BEAR + onaylı min kalite"],
                        ["bear_confirmed_top_n_max", "BEAR + onaylı top N max"],
                        ["caution_confirmed_min_quality", "CAUTION + CONFIRMED min kalite"],
                        ["caution_confirmed_top_n_max", "CAUTION + CONFIRMED top N max"],
                        ["caution_other_min_quality", "CAUTION (diğer) min kalite"],
                        ["caution_other_top_n_max", "CAUTION (diğer) top N max"],
                    ] as const
                ).map(([key, label]) => (
                    <FieldNum
                        key={key}
                        label={label}
                        hint={
                            key.includes("top")
                                ? "Scanner çıktısında bu rejimde en fazla kaç sinyal gösterilir."
                                : "Bu rejim + alt durumda sinyalin geçmesi için min kalite skoru."
                        }
                        value={dictNum("regime_thresholds", key, key.includes("top") ? 4 : 70)}
                        onChange={(v) => setDictNum("regime_thresholds", key, Math.round(v))}
                        step={1}
                        min={1}
                        max={key.includes("top") ? 50 : 100}
                    />
                ))}
            </Section>

            <Section
                id="settings-section-backtest-giris-yurutme"
                title="Backtest / giriş yürütme"
                help="Min ödül:risk, gap sınırları, kısmi satış oranı, kalite tabanları ve gap sonrası portföy risk limiti gibi hem backtest hem canlı yürütmeyle uyumlu giriş kuralları. Sadece simülasyon değil; motor bu alanı da okur."
            >
                <FieldNum
                    label="Min R:R (C değil)"
                    hint="Girişte beklenen ödül/risk oranı; tip C hariç tipler için."
                    value={num("min_rr_at_entry", 1.2)}
                    onChange={(v) => setNum("min_rr_at_entry", v)}
                    step="0.1"
                    min={0.5}
                    max={10}
                />
                <FieldNum
                    label="Min R:R — tip C"
                    hint="Erken tip C için genelde biraz daha yüksek R:R ister."
                    value={num("min_rr_type_c", 1.5)}
                    onChange={(v) => setNum("min_rr_type_c", v)}
                    step="0.1"
                    min={0.5}
                    max={10}
                />
                <FieldNum
                    label="T1 kısmi çıkış oranı"
                    hint="T1’de pozisyonun ne kadarı kapatılır (kalan T2’ye taşınır)."
                    value={num("partial_at_t1_fraction", 0.5)}
                    onChange={(v) => setNum("partial_at_t1_fraction", v)}
                    step="0.05"
                    min={0.05}
                    max={1}
                />
                <FieldNum
                    label="Min kalite — C"
                    hint="Tip C sinyali için taban kalite; altı elenir."
                    value={num("min_quality_type_c", 65)}
                    onChange={(v) => setNum("min_quality_type_c", Math.round(v))}
                    step={1}
                    min={30}
                    max={100}
                />
                <FieldNum
                    label="Min kalite — A"
                    hint="Tip A için taban kalite."
                    value={num("min_quality_type_a", 60)}
                    onChange={(v) => setNum("min_quality_type_a", Math.round(v))}
                    step={1}
                    min={30}
                    max={100}
                />
                <FieldNum
                    label="Min kalite — B"
                    hint="Tip B için taban kalite."
                    value={num("min_quality_type_b", 60)}
                    onChange={(v) => setNum("min_quality_type_b", Math.round(v))}
                    step={1}
                    min={30}
                    max={100}
                />
                <FieldNum
                    label="Max gap yukarı %"
                    hint="Açılış, sinyal fiyatına göre çok yukarı gap’te girişten vazgeçilir."
                    value={num("max_gap_up_pct", 5)}
                    onChange={(v) => setNum("max_gap_up_pct", v)}
                    step={0.5}
                    min={0}
                    max={30}
                />
                <FieldNum
                    label="Max gap aşağı %"
                    hint="Aşağı gap çok derinse giriş reddedilir (slippage / yapı riski)."
                    value={num("max_gap_down_pct", 4)}
                    onChange={(v) => setNum("max_gap_down_pct", v)}
                    step={0.5}
                    min={0}
                    max={30}
                />
                <FieldNum
                    label="Max kayıp / işlem (stop mesafesi %)"
                    hint="Stop mesafesi girişe göre bu yüzdeneyi aşmamalı (işlem boyutu ile birlikte düşünülür)."
                    value={num("max_loss_per_trade_pct", 0.07)}
                    onChange={(v) => setNum("max_loss_per_trade_pct", v)}
                    step="0.01"
                    min={0.02}
                    max={0.25}
                />
                <FieldNum
                    label="Gap risk bütçesi (portföy %)"
                    hint="Kötü senaryo gap’inde portföyün bu kadarından fazla riske izin verilmez (pozisyon küçültülür)."
                    value={num("max_gap_risk_portfolio_pct", 0.02)}
                    onChange={(v) => setNum("max_gap_risk_portfolio_pct", v)}
                    step="0.005"
                    min={0.005}
                    max={0.1}
                />
                <FieldNum
                    label="Max notional (portföy %)"
                    hint="Tek işlemde kullanılacak sermaye üst sınırı (portföy yüzdesi)."
                    value={num("max_position_cost_portfolio_pct", 0.15)}
                    onChange={(v) => setNum("max_position_cost_portfolio_pct", v)}
                    step="0.05"
                    min={0.05}
                    max={0.5}
                />
                <FieldNum
                    label="Cooldown (gün)"
                    hint="Aynı tickerde ardışık kayıptan sonra bekleme (ban ile birlikte)."
                    value={num("cooldown_days", 5)}
                    onChange={(v) => setNum("cooldown_days", Math.round(v))}
                    step={1}
                    min={0}
                    max={30}
                />
                <FieldNum
                    label="Ticker max zarar sayısı (ban)"
                    hint="Bu kadar zararlı çıkıştan sonra ticker geçici elenir."
                    value={num("ticker_max_losses", 2)}
                    onChange={(v) => setNum("ticker_max_losses", Math.round(v))}
                    step={1}
                    min={1}
                    max={10}
                />
                <FieldNum
                    label="Slippage (bps / taraf)"
                    hint="Backtest / yürütmede alış-satış için baz puan kayması (taraf başına)."
                    value={num("slippage_bps_per_side", 5)}
                    onChange={(v) => setNum("slippage_bps_per_side", Math.round(v))}
                    step={1}
                    min={0}
                    max={100}
                />
                <FieldNum
                    label="Kısmi için min lot"
                    hint="T1 kısmını satmak için gereken minimum adet (küçük lotlarda kısmi atlanabilir)."
                    value={num("min_shares_for_partial", 2)}
                    onChange={(v) => setNum("min_shares_for_partial", Math.round(v))}
                    step={1}
                    min={1}
                    max={100}
                />
            </Section>

            <Section
                id="settings-section-tarama-gecitleri"
                title="Tarama geçitleri"
                help="Tip S/C/B/A için ek güvenlik: aşırı 5 günlük getiri, ekstrem RSI, ‘geç giriş’ eşikleri. FOMO ve çok geç koşuya katılmayı kesmek için. Az aday kalıyorsa önce sinyal filtresine, çok ‘tepeden’ giriş varsa buraya bakın."
            >
                <FieldNum
                    label="Parabol 5g getiri > %"
                    hint="5 günlük getiri bu eşiği aşarsa parabolik / aşırı momentum sayılır (tip kurallarına bağlı)."
                    value={nn(["scan_gates", "parabolic_five_day_return_gt"], 70)}
                    onChange={(v) => sn(["scan_gates", "parabolic_five_day_return_gt"], v)}
                    min={10}
                    max={200}
                />
                <FieldNum
                    label="Ekstrem 5g getiri > %"
                    hint="Çok uçmuş trend; ilgili tip için ek koruma eşiği."
                    value={nn(["scan_gates", "extreme_five_day_return_gt"], 60)}
                    onChange={(v) => sn(["scan_gates", "extreme_five_day_return_gt"], v)}
                    min={10}
                    max={200}
                />
                <FieldNum
                    label="Ekstrem RSI >"
                    hint="RSI bu üstündeyse ‘çok ısınmış’ kabul edilir (tip bazlı geçit)."
                    value={nn(["scan_gates", "extreme_rsi_gt"], 85)}
                    onChange={(v) => sn(["scan_gates", "extreme_rsi_gt"], v)}
                    min={50}
                    max={100}
                />
                <FieldNum
                    label="Geç giriş: 5g toplam > %"
                    hint="5g toplam getiri bu kadar yüksekse + RSI aşağıdaki eşikteyse geç giriş sayılır."
                    value={nn(["scan_gates", "late_entry_five_day_total_gt"], 30)}
                    onChange={(v) => sn(["scan_gates", "late_entry_five_day_total_gt"], v)}
                    min={0}
                    max={100}
                />
                <FieldNum
                    label="Geç giriş: RSI >"
                    hint="Yukarıdaki 5g koşuluyla birlikte; geç FOMO girişlerini kesmek için."
                    value={nn(["scan_gates", "late_entry_rsi_gt"], 65)}
                    onChange={(v) => sn(["scan_gates", "late_entry_rsi_gt"], v)}
                    min={40}
                    max={95}
                />
            </Section>

            <Section
                id="settings-section-risk-hedefleri-rejim"
                title="Risk hedefleri (rejim / kalite)"
                help="Kalite skoru yüksek/orta bantlarında hedef genişletme ve rejime göre T2 ATR çarpanı ile T2/T1 mesafe zeminleri. İyi kurulumlara biraz daha pay, zayıf rejimde daha muhafazakâr hedef için."
            >
                <FieldNum
                    label="Kalite boost eşik (yüksek)"
                    hint="Kalite skoru bu değerin üstündeyse hedef genişletme (yüksek bant) devreye girer."
                    value={nn(["risk_targets", "quality_tier_high"], 85)}
                    onChange={(v) => sn(["risk_targets", "quality_tier_high"], Math.round(v))}
                    step={1}
                    min={50}
                    max={100}
                />
                <FieldNum
                    label="Kalite boost (yüksek) çarpan"
                    hint="Yüksek kalitede T1/T2 mesafeleri bu çarpanla ölçeklenir."
                    value={nn(["risk_targets", "quality_boost_high"], 1.15)}
                    onChange={(v) => sn(["risk_targets", "quality_boost_high"], v)}
                    step="0.01"
                    min={1}
                    max={2}
                />
                <FieldNum
                    label="Kalite boost eşik (orta)"
                    hint="Orta kalite bandı; altı ‘düşük’, üstü yüksek banda yaklaşır."
                    value={nn(["risk_targets", "quality_tier_mid"], 75)}
                    onChange={(v) => sn(["risk_targets", "quality_tier_mid"], Math.round(v))}
                    step={1}
                    min={50}
                    max={100}
                />
                <FieldNum
                    label="Kalite boost (orta) çarpan"
                    hint="Orta kalite için daha hafif hedef genişletme."
                    value={nn(["risk_targets", "quality_boost_mid"], 1.08)}
                    onChange={(v) => sn(["risk_targets", "quality_boost_mid"], v)}
                    step="0.01"
                    min={1}
                    max={2}
                />
                <FieldNum
                    label="T2 ATR çarpanı — CAUTION"
                    hint="CAUTION rejiminde T2 için kullanılan ATR çarpanı (daha temkinli hedef)."
                    value={nn(["risk_targets", "t2_atr_mult_caution"], 1.6)}
                    onChange={(v) => sn(["risk_targets", "t2_atr_mult_caution"], v)}
                    step="0.05"
                    min={1}
                    max={4}
                />
                <FieldNum
                    label="T2 ATR çarpanı — BEAR"
                    hint="Ayı rejiminde T2 mesafesi daha sıkı tutulur."
                    value={nn(["risk_targets", "t2_atr_mult_bear"], 1.05)}
                    onChange={(v) => sn(["risk_targets", "t2_atr_mult_bear"], v)}
                    step="0.05"
                    min={1}
                    max={4}
                />
                <FieldNum
                    label="Min ödül:risk (T1)"
                    hint="Hedefler hesaplanırken T1 için kabul edilen minimum R (rejim hedefleri)."
                    value={nn(["risk_targets", "min_reward_risk_multiple_t1"], 1.5)}
                    onChange={(v) => sn(["risk_targets", "min_reward_risk_multiple_t1"], v)}
                    step="0.05"
                    min={0.5}
                    max={5}
                />
                <FieldNum
                    label="T2/T1 min boşluk — BULL"
                    hint="T2, T1’in en az bu katı kadar uzak olmalı (iyi rejim)."
                    value={nn(["risk_targets", "t2_min_gap_vs_t1_bull"], 1.15)}
                    onChange={(v) => sn(["risk_targets", "t2_min_gap_vs_t1_bull"], v)}
                    step="0.01"
                    min={1}
                    max={2}
                />
                <FieldNum
                    label="T2/T1 min boşluk — BEAR"
                    hint="Ayıda T2–T1 farkı daha az zorunlu tutulabilir."
                    value={nn(["risk_targets", "t2_min_gap_vs_t1_bear"], 1.05)}
                    onChange={(v) => sn(["risk_targets", "t2_min_gap_vs_t1_bear"], v)}
                    step="0.01"
                    min={1}
                    max={2}
                />
                <FieldNum
                    label="T2/T1 min boşluk — CAUTION"
                    hint="Temkinli rejimde T2’nin T1’e göre minimum genişliği."
                    value={nn(["risk_targets", "t2_min_gap_vs_t1_caution"], 1.1)}
                    onChange={(v) => sn(["risk_targets", "t2_min_gap_vs_t1_caution"], v)}
                    step="0.01"
                    min={1}
                    max={2}
                />
                <FieldNum
                    label="T2 yakın tavan tabanı (×T1)"
                    hint="T2, tavanlara yaklaşınca T1’e göre en az bu oranda mesafe korunur."
                    value={nn(["risk_targets", "t2_vs_t1_near_cap_floor"], 1.005)}
                    onChange={(v) => sn(["risk_targets", "t2_vs_t1_near_cap_floor"], v)}
                    step="0.001"
                    min={1}
                    max={1.2}
                />
            </Section>

            <Section
                id="settings-section-evren-filtreleri"
                title="Evren filtreleri"
                help="Hisse bar verisiyle elenir: piyasa değeri, ortalama hacim, fiyat bandı, max float, kazanç penceresi, ATR periyodu. Finviz listesinden sonra ‘kim hiç işlenmez’ tarafı burasıdır; çok boş liste → burayı veya Finviz tavanını gevşetin."
            >
                <FieldNum
                    label="Min piyasa değeri ($)"
                    hint="Bar/fundamental veride MCAP bu altındaysa hisse elenir."
                    value={nn(["universe_filters", "min_market_cap"], 250000000)}
                    onChange={(v) => sn(["universe_filters", "min_market_cap"], Math.round(v))}
                    step={1_000_000}
                    min={1}
                    max={10_000_000_000}
                />
                <FieldNum
                    label="Max piyasa değeri ($)"
                    hint="Çok büyük şirketleri small-cap dışı bırakmak için üst sınır."
                    value={nn(["universe_filters", "max_market_cap"], 2500000000)}
                    onChange={(v) => sn(["universe_filters", "max_market_cap"], Math.round(v))}
                    step={1_000_000}
                    min={1}
                    max={50_000_000_000}
                />
                <FieldNum
                    label="Min ort. hacim"
                    hint="20g ortalama günlük hacim (adet); likidite tabanı."
                    value={nn(["universe_filters", "min_avg_volume"], 750000)}
                    onChange={(v) => sn(["universe_filters", "min_avg_volume"], Math.round(v))}
                    step={50_000}
                    min={0}
                    max={50_000_000}
                />
                <FieldNum
                    label="Min fiyat ($)"
                    hint="Penny / çok düşük fiyatları kesmek için."
                    value={nn(["universe_filters", "min_price"], 3)}
                    onChange={(v) => sn(["universe_filters", "min_price"], v)}
                    step="0.5"
                    min={0.5}
                    max={500}
                />
                <FieldNum
                    label="Max fiyat ($)"
                    hint="Çok pahalı hisseleri tarama dışı bırakmak için."
                    value={nn(["universe_filters", "max_price"], 200)}
                    onChange={(v) => sn(["universe_filters", "max_price"], v)}
                    step={1}
                    min={1}
                    max={2000}
                />
                <FieldNum
                    label="Max float (adet)"
                    hint="Dolaşımdaki pay üst sınırı; dar float odağı için."
                    value={nn(["universe_filters", "max_float_shares"], 150000000)}
                    onChange={(v) => sn(["universe_filters", "max_float_shares"], Math.round(v))}
                    step={1_000_000}
                    min={1}
                    max={2_000_000_000}
                />
                <FieldNum
                    label="Kazanç hariç gün"
                    hint="Kazanç tarihine bu kadar gün içindeyse hisse elenebilir."
                    value={nn(["universe_filters", "earnings_exclusion_days"], 3)}
                    onChange={(v) => sn(["universe_filters", "earnings_exclusion_days"], Math.round(v))}
                    step={1}
                    min={0}
                    max={14}
                />
                <FieldNum
                    label="ATR periyodu"
                    hint="ATR ve volatilite hesaplarında kullanılan periyot (bar sayısı)."
                    value={nn(["universe_filters", "atr_period"], 10)}
                    onChange={(v) => sn(["universe_filters", "atr_period"], Math.round(v))}
                    step={1}
                    min={5}
                    max={30}
                />
            </Section>

            <Section
                title="Finviz evren taraması (~N hisse)"
                help="Günlük aday listesi: kaç hisse, Finviz’de hangi üç sorgu açık, önbellek süresi, Finviz sonucu azsa statik liste ile birleştirme eşiği, sıralama ağırlıkları ve ‘chase’ cezası. Tarama çok kalabalık veya çok boşsa önce tavan ve sorgu anahtarlarını oynatın; ekran URL’leri kodda sabittir."
            >
                <p style={{ margin: "0 0 12px", fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.5, gridColumn: "1 / -1" }}>
                    Finviz ekran tanımları kodda sabittir; burada tavan adet, hangi üç sorgunun açık olduğu, önbellek, statik
                    liste ile birleştirme eşiği ve bileşik sıralama ağırlıkları ayarlanır. Evren filtreleri (üstte) bar verisi
                    tarafında; bu bölüm günlük aday listesini seçer.
                </p>
                <FieldBool
                    label="Finviz kullan (kapalı → yalnız statik liste)"
                    hint="Kapalıyken günlük liste kod içindeki statik/seeding listesine döner; Finviz çağrısı yapılmaz."
                    value={nb(["universe_scan", "use_finviz"], true)}
                    onChange={(v) => sb(["universe_scan", "use_finviz"], v)}
                />
                <FieldNum
                    label="Tarama tavanı (max hisse)"
                    hint="Bir taramada işlenecek maksimum ticker sayısı (API scanner / backtest birleşik liste)."
                    value={nn(["universe_scan", "max_scan_tickers"], 200)}
                    onChange={(v) => sn(["universe_scan", "max_scan_tickers"], Math.round(v))}
                    step={1}
                    min={20}
                    max={500}
                />
                <FieldNum
                    label="Önbellek süresi (dk)"
                    hint="0 = her taramada Finviz’e yeniden git. &gt;0 = aynı motor örneğinde bu süre dolana kadar önceki listeyi kullan (API’de ayar kaydı motoru yeniler)."
                    value={nn(["universe_scan", "cache_duration_minutes"], 60)}
                    onChange={(v) => sn(["universe_scan", "cache_duration_minutes"], Math.round(v))}
                    step={1}
                    min={0}
                    max={10080}
                />
                <FieldNum
                    label="Finviz sonuç sayısı eşiği (üstünde statik birleştirme yok)"
                    hint="Finviz bu kadar veya daha fazla dönerse yalnız Finviz sıralaması kullanılır; altında statik liste eklenir."
                    value={nn(["universe_scan", "min_finviz_tickers_skip_static_merge"], 30)}
                    onChange={(v) => sn(["universe_scan", "min_finviz_tickers_skip_static_merge"], Math.round(v))}
                    step={1}
                    min={0}
                    max={500}
                />
                <FieldBool
                    label="Sorgu: Momentum hunters"
                    hint="RVOL + volatilite ağırlıklı dar Finviz ekranı; kapatırsanız o kanaldan aday gelmez."
                    value={nb(["universe_scan", "enable_finviz_query_momentum"], true)}
                    onChange={(v) => sb(["universe_scan", "enable_finviz_query_momentum"], v)}
                />
                <FieldBool
                    label="Sorgu: Setup builders"
                    hint="Hacim + RSI ‘aşırı alım değil’ tabanlı ikinci ekran."
                    value={nb(["universe_scan", "enable_finviz_query_setup"], true)}
                    onChange={(v) => sb(["universe_scan", "enable_finviz_query_setup"], v)}
                />
                <FieldBool
                    label="Sorgu: Wider net"
                    hint="Daha geniş float bandı, yüksek RVOL şartı ile üçüncü kaynak."
                    value={nb(["universe_scan", "enable_finviz_query_wider"], true)}
                    onChange={(v) => sb(["universe_scan", "enable_finviz_query_wider"], v)}
                />
                <FieldNum
                    label="Son fiyat filtresi — min ($)"
                    hint="Finviz sonrası kod içi fiyat bandı (tablodaki Price sütunu)."
                    value={nn(["universe_scan", "post_filter_price_min"], 3)}
                    onChange={(v) => sn(["universe_scan", "post_filter_price_min"], v)}
                    step="0.5"
                    min={0.5}
                    max={500}
                />
                <FieldNum
                    label="Son fiyat filtresi — max ($)"
                    hint="Çok yüksek fiyatlı adayları kesmek için üst sınır."
                    value={nn(["universe_scan", "post_filter_price_max"], 200)}
                    onChange={(v) => sn(["universe_scan", "post_filter_price_max"], v)}
                    step={1}
                    min={1}
                    max={50000}
                />
                <FieldNum
                    label="Sıralama ağırlığı: rel. volume"
                    hint="Dört ağırlığın toplamı 1.0 olmalı (±0.02)."
                    value={nn(["universe_scan", "rank_weight_rvol"], 0.3)}
                    onChange={(v) => sn(["universe_scan", "rank_weight_rvol"], v)}
                    step="0.01"
                    min={0}
                    max={1}
                />
                <FieldNum
                    label="Sıralama ağırlığı: günlük değişim"
                    value={nn(["universe_scan", "rank_weight_change"], 0.25)}
                    onChange={(v) => sn(["universe_scan", "rank_weight_change"], v)}
                    step="0.01"
                    min={0}
                    max={1}
                />
                <FieldNum
                    label="Sıralama ağırlığı: hacim"
                    value={nn(["universe_scan", "rank_weight_volume"], 0.25)}
                    onChange={(v) => sn(["universe_scan", "rank_weight_volume"], v)}
                    step="0.01"
                    min={0}
                    max={1}
                />
                <FieldNum
                    label="Sıralama ağırlığı: piyasa değeri şeridi"
                    value={nn(["universe_scan", "rank_weight_mcap"], 0.2)}
                    onChange={(v) => sn(["universe_scan", "rank_weight_mcap"], v)}
                    step="0.01"
                    min={0}
                    max={1}
                />
                <FieldNum
                    label="Chase cezası: değişim > % (yüksek)"
                    value={nn(["universe_scan", "chase_penalty_change_pct_high"], 15)}
                    onChange={(v) => sn(["universe_scan", "chase_penalty_change_pct_high"], v)}
                    min={5}
                    max={50}
                />
                <FieldNum
                    label="Chase cezası: değişim > % (orta)"
                    value={nn(["universe_scan", "chase_penalty_change_pct_mid"], 10)}
                    onChange={(v) => sn(["universe_scan", "chase_penalty_change_pct_mid"], v)}
                    min={1}
                    max={40}
                />
                <FieldNum
                    label="Chase ceza puanı (yüksek)"
                    value={nn(["universe_scan", "chase_penalty_points_high"], 50)}
                    onChange={(v) => sn(["universe_scan", "chase_penalty_points_high"], Math.round(v))}
                    min={0}
                    max={100}
                />
                <FieldNum
                    label="Chase ceza puanı (orta)"
                    value={nn(["universe_scan", "chase_penalty_points_mid"], 25)}
                    onChange={(v) => sn(["universe_scan", "chase_penalty_points_mid"], Math.round(v))}
                    min={0}
                    max={100}
                />
            </Section>

            <Section
                id="settings-section-swing-hazirlik"
                title="Swing hazırlık / aşırı uzama (sinyal)"
                help="MA20’ye göre ‘hazır’ mesafe ve günlük / tek gün / 5 günlük aşırı hareket üst sınırları. Yapı bozulmadan önce aşırı uzamış çıkışları elemek için. Çok erken eleniyorsa eşikleri hafifçe gevşetin."
            >
                <FieldNum
                    label="MA20 altı kabul (%)"
                    hint="Swing ready: fiyat MA20 altında en fazla bu kadar olabilir"
                    value={nn(["signal_confirmation", "ma20_max_distance_below_pct"], 3)}
                    onChange={(v) => sn(["signal_confirmation", "ma20_max_distance_below_pct"], v)}
                    step="0.5"
                    min={0}
                    max={20}
                />
                <FieldNum
                    label="Günlük değişim üst sınırı (güvenli) %"
                    hint="Bugünkü % değişim bu kadar üstündeyse ‘aşırı uzamış’ sayılır (onay / swing ready ile ilişkili)."
                    value={nn(["signal_confirmation", "overext_today_change_max"], 15)}
                    onChange={(v) => sn(["signal_confirmation", "overext_today_change_max"], v)}
                    min={5}
                    max={50}
                />
                <FieldNum
                    label="Tek gün max (güvenli) %"
                    hint="Son barda tek gün hareketi bu üstündeyse aşırı volatilite."
                    value={nn(["signal_confirmation", "overext_single_day_max"], 25)}
                    onChange={(v) => sn(["signal_confirmation", "overext_single_day_max"], v)}
                    min={5}
                    max={80}
                />
                <FieldNum
                    label="5g toplam üst sınır (güvenli) %"
                    hint="Beş günlük toplam getiri limiti; FOMO rallilerini yumuşatmak için."
                    value={nn(["signal_confirmation", "overext_five_day_total_max"], 40)}
                    onChange={(v) => sn(["signal_confirmation", "overext_five_day_total_max"], v)}
                    min={10}
                    max={200}
                />
            </Section>

            <Section
                id="settings-section-backtest-dongu"
                title="Backtest döngü (rejim / drawdown)"
                help="Simülasyonda ayıda yeni giriş kapatma, CAUTION’da eşzamanlı pozisyon sınırı, drawdown’a göre giriş durdurma ve tek pozisyona indirme. Canlı risk yönetiminizle aynı fikirleri test etmek için; parametreleri agresif yapınca equity eğrisi yumuşar ama fırsat da azalır."
            >
                <FieldBool
                    label="BEAR’da yeni giriş yok"
                    hint="Simülasyonda ayı onaylandığında yeni pozisyon açılmaz."
                    value={nb(["backtest_loop", "bear_block_new_entries"], true)}
                    onChange={(v) => sb(["backtest_loop", "bear_block_new_entries"], v)}
                />
                <FieldNum
                    label="CAUTION max eşzamanlı"
                    hint="CAUTION rejiminde aynı anda en fazla kaç açık pozisyon."
                    value={nn(["backtest_loop", "caution_max_concurrent"], 1)}
                    onChange={(v) => sn(["backtest_loop", "caution_max_concurrent"], Math.round(v))}
                    step={1}
                    min={1}
                    max={20}
                />
                <FieldNum
                    label="Drawdown — giriş durdur (oran)"
                    hint="Özsermaye tepeye göre bu kadar gerilediyse yeni girişler durur."
                    value={nn(["backtest_loop", "drawdown_pause_entries_fraction"], 0.25)}
                    onChange={(v) => sn(["backtest_loop", "drawdown_pause_entries_fraction"], v)}
                    step="0.01"
                    min={0.05}
                    max={0.9}
                />
                <FieldNum
                    label="Drawdown — tek pozisyona indir (oran)"
                    hint="Daha hafif drawdown eşiğinde eşzamanlı pozisyon tek pozisyona indirilir."
                    value={nn(["backtest_loop", "drawdown_reduce_to_one_position_fraction"], 0.15)}
                    onChange={(v) => sn(["backtest_loop", "drawdown_reduce_to_one_position_fraction"], v)}
                    step="0.01"
                    min={0.05}
                    max={0.5}
                />
            </Section>

            <Section
                id="settings-section-backtest-tip-kalitesi"
                title="Backtest tip kalitesi (BEAR / CAUTION tabanları)"
                help="BEAR ve CAUTION rejiminde Tip C/A/B için minimum kalite tabanı. Ayıda daha seçici sinyal için yükseltin; çok az işlem kalıyorsa düşürün. BULL tabanları bu blokta yoktur (üst rejim bölümüne bakın)."
            >
                <FieldNum
                    label="Tip C — BEAR min"
                    hint="Ayı rejiminde Tip C için simülasyonda istenen min kalite (üst bölümdeki genel min’e ek taban)."
                    value={nn(["backtest_type_quality", "type_c_bear"], 82)}
                    onChange={(v) => sn(["backtest_type_quality", "type_c_bear"], Math.round(v))}
                    step={1}
                    min={30}
                    max={100}
                />
                <FieldNum
                    label="Tip C — CAUTION min"
                    hint="CAUTION rejiminde Tip C taban kalitesi."
                    value={nn(["backtest_type_quality", "type_c_caution"], 75)}
                    onChange={(v) => sn(["backtest_type_quality", "type_c_caution"], Math.round(v))}
                    step={1}
                    min={30}
                    max={100}
                />
                <FieldNum
                    label="Tip A — BEAR min"
                    hint="Ayıda Tip A için min kalite."
                    value={nn(["backtest_type_quality", "type_a_bear"], 72)}
                    onChange={(v) => sn(["backtest_type_quality", "type_a_bear"], Math.round(v))}
                    step={1}
                    min={30}
                    max={100}
                />
                <FieldNum
                    label="Tip A — CAUTION min"
                    hint="CAUTION’da Tip A min kalite."
                    value={nn(["backtest_type_quality", "type_a_caution"], 66)}
                    onChange={(v) => sn(["backtest_type_quality", "type_a_caution"], Math.round(v))}
                    step={1}
                    min={30}
                    max={100}
                />
                <FieldNum
                    label="Tip B — BEAR min"
                    hint="Ayıda Tip B min kalite."
                    value={nn(["backtest_type_quality", "type_b_bear"], 67)}
                    onChange={(v) => sn(["backtest_type_quality", "type_b_bear"], Math.round(v))}
                    step={1}
                    min={30}
                    max={100}
                />
                <FieldNum
                    label="Tip B — CAUTION min"
                    hint="CAUTION’da Tip B min kalite."
                    value={nn(["backtest_type_quality", "type_b_caution"], 60)}
                    onChange={(v) => sn(["backtest_type_quality", "type_b_caution"], Math.round(v))}
                    step={1}
                    min={30}
                    max={100}
                />
            </Section>

            <Section
                id="settings-section-backtest-giris-ema-gap"
                title="Backtest giriş (EMA / gap)"
                help="Tip C için açılış/kapanış oranı, trend EMA periyotları ve gap planlamada ATR çarpanı. Simülasyonun ‘ne zaman ve nasıl girildiği’; canlı ile hizalamak için küçük adımla değiştirin, tek seferde çok alanı oynamayın."
            >
                <FieldNum
                    label="Tip C: min açılış / sinyal kapanış oranı"
                    hint="Açılış, sinyal günü kapanışına göre çok aşağıdaysa (gap aşağı) Tip C girişi reddedilir."
                    value={nn(["backtest_entry", "type_c_min_open_vs_signal_close_ratio"], 0.98)}
                    onChange={(v) => sn(["backtest_entry", "type_c_min_open_vs_signal_close_ratio"], v)}
                    step="0.01"
                    min={0.8}
                    max={1}
                />
                <FieldNum
                    label="Trend EMA hızlı span"
                    hint="Trend onayı için hızlı EMA periyodu (EMA_fast > EMA_slow)."
                    value={nn(["backtest_entry", "trend_ema_fast_span"], 10)}
                    onChange={(v) => sn(["backtest_entry", "trend_ema_fast_span"], Math.round(v))}
                    step={1}
                    min={3}
                    max={60}
                />
                <FieldNum
                    label="Trend EMA yavaş span"
                    hint="Yavaş EMA periyodu; hızlıdan büyük olmalıdır."
                    value={nn(["backtest_entry", "trend_ema_slow_span"], 20)}
                    onChange={(v) => sn(["backtest_entry", "trend_ema_slow_span"], Math.round(v))}
                    step={1}
                    min={5}
                    max={120}
                />
                <FieldNum
                    label="Trend min bar"
                    hint="EMA hesabı için gereken minimum geçmiş bar sayısı."
                    value={nn(["backtest_entry", "trend_min_bars"], 21)}
                    onChange={(v) => sn(["backtest_entry", "trend_min_bars"], Math.round(v))}
                    step={1}
                    min={10}
                    max={120}
                />
                <FieldNum
                    label="Gap planlama ATR çarpanı"
                    hint="Gap sonrası plan fiyatı / stop mesafesi hesabında ATR ile çarpan."
                    value={nn(["backtest_entry", "gap_atr_multiplier"], 2)}
                    onChange={(v) => sn(["backtest_entry", "gap_atr_multiplier"], v)}
                    step="0.1"
                    min={0.5}
                    max={6}
                />
                <FieldNum
                    label="Kısmi sonrası T2 fallback çarpanı"
                    hint="T1 kısmı alındıktan sonra T2 hedefi sıkılaşırsa uygulanan küçük genişletme."
                    value={nn(["backtest_entry", "partial_fallback_target_bump"], 1.15)}
                    onChange={(v) => sn(["backtest_entry", "partial_fallback_target_bump"], v)}
                    step="0.01"
                    min={1}
                    max={2}
                />
            </Section>

            <Section
                id="settings-section-backtest-cikis-trailing"
                title="Backtest çıkış (zaman stop / trailing)"
                help="Zaman stop, tepeye göre trailing basamakları, breakeven ve kapanış sıkılaştırmaları. Erken stop veya geç çıkış şikâyeti varsa önce bir–iki parametre seçip backtest ile karşılaştırın."
            >
                <FieldNum
                    label="Zaman stop min gün"
                    hint="Pozisyon en az bu kadar gündür açıksa ve küçük zarar/zaman koşulu sağlanırsa çıkış."
                    value={nn(["backtest_exit_trailing", "time_stop_min_days"], 5)}
                    onChange={(v) => sn(["backtest_exit_trailing", "time_stop_min_days"], Math.round(v))}
                    step={1}
                    min={1}
                    max={30}
                />
                <FieldNum
                    label="Zaman stop min kayıp oranı"
                    hint="Zaman stop ile çıkış için gereken minimum kayıp (çok kârdayken erken kesmez)."
                    value={nn(["backtest_exit_trailing", "time_stop_min_loss_fraction"], 0.05)}
                    onChange={(v) => sn(["backtest_exit_trailing", "time_stop_min_loss_fraction"], v)}
                    step="0.01"
                    min={0.01}
                    max={0.5}
                />
                <FieldNum
                    label="Trail eşik ATR (tepe 2.5)"
                    hint="Tepe kazancı ≥ 2.5 ATR iken trailing kural seti için eşik."
                    value={nn(["backtest_exit_trailing", "trail_peak_atr_25"], 2.5)}
                    onChange={(v) => sn(["backtest_exit_trailing", "trail_peak_atr_25"], v)}
                    step="0.1"
                    min={0.5}
                    max={10}
                />
                <FieldNum
                    label="Trail high− ×ATR (2.5 basamak)"
                    hint="Yüksek tepe bandında stop, high − bu ×ATR altına çekilir."
                    value={nn(["backtest_exit_trailing", "trail_high_minus_atr_25"], 0.8)}
                    onChange={(v) => sn(["backtest_exit_trailing", "trail_high_minus_atr_25"], v)}
                    step="0.1"
                    min={0}
                    max={5}
                />
                <FieldNum
                    label="Trail eşik ATR (tepe 2.0)"
                    hint="Orta-yüksek tepe (2 ATR) basamağı."
                    value={nn(["backtest_exit_trailing", "trail_peak_atr_20"], 2)}
                    onChange={(v) => sn(["backtest_exit_trailing", "trail_peak_atr_20"], v)}
                    step="0.1"
                    min={0.5}
                    max={10}
                />
                <FieldNum
                    label="Trail peak_gain oranı (2.0)"
                    hint="2 ATR basamağında tepe kazancının ne kadarı korunmaya çalışılır (kesir)."
                    value={nn(["backtest_exit_trailing", "trail_peak_frac_20"], 0.5)}
                    onChange={(v) => sn(["backtest_exit_trailing", "trail_peak_frac_20"], v)}
                    step="0.05"
                    min={0}
                    max={1}
                />
                <FieldNum
                    label="Trail eşik ATR (tepe 1.5)"
                    hint="Daha alçak tepe bandı (1.5 ATR) için trailing."
                    value={nn(["backtest_exit_trailing", "trail_peak_atr_15"], 1.5)}
                    onChange={(v) => sn(["backtest_exit_trailing", "trail_peak_atr_15"], v)}
                    step="0.1"
                    min={0.5}
                    max={10}
                />
                <FieldNum
                    label="Trail peak_gain oranı (1.5)"
                    hint="1.5 ATR bandında korunan kazanç oranı."
                    value={nn(["backtest_exit_trailing", "trail_peak_frac_15"], 0.3)}
                    onChange={(v) => sn(["backtest_exit_trailing", "trail_peak_frac_15"], v)}
                    step="0.05"
                    min={0}
                    max={1}
                />
                <FieldNum
                    label="Breakeven tepe ATR"
                    hint="Fiyat bu kadar ATR kazanca ulaşınca stop girişe yaklaştırılır (breakeven)."
                    value={nn(["backtest_exit_trailing", "breakeven_peak_atr"], 1.5)}
                    onChange={(v) => sn(["backtest_exit_trailing", "breakeven_peak_atr"], v)}
                    step="0.1"
                    min={0.5}
                    max={10}
                />
                <FieldNum
                    label="Hafif koruma tepe ATR"
                    hint="Daha erken hafif kâr kilidi için tepe ATR eşiği."
                    value={nn(["backtest_exit_trailing", "light_protect_peak_atr"], 1)}
                    onChange={(v) => sn(["backtest_exit_trailing", "light_protect_peak_atr"], v)}
                    step="0.1"
                    min={0.5}
                    max={10}
                />
                <FieldNum
                    label="Hafif koruma giriş altı ×ATR"
                    hint="Hafif koruma stop’u girişin bu ×ATR altına inerse tetiklenir."
                    value={nn(["backtest_exit_trailing", "light_protect_below_entry_atr"], 0.2)}
                    onChange={(v) => sn(["backtest_exit_trailing", "light_protect_below_entry_atr"], v)}
                    step="0.05"
                    min={0}
                    max={2}
                />
                <FieldNum
                    label="Kapanış sıkılaştırma — kazanç ATR"
                    hint="Gün sonu kapanışında kâr bu ATR’yi geçtiyse trail sıkılaştırması devreye girer."
                    value={nn(["backtest_exit_trailing", "close_gain_atr_20"], 2)}
                    onChange={(v) => sn(["backtest_exit_trailing", "close_gain_atr_20"], v)}
                    step="0.1"
                    min={0.5}
                    max={10}
                />
                <FieldNum
                    label="Kapanış trail ×ATR (2.0 basamak)"
                    hint="Yukarıdaki kapanış kazancında high − trail ×ATR stop."
                    value={nn(["backtest_exit_trailing", "close_trail_atr_20"], 1)}
                    onChange={(v) => sn(["backtest_exit_trailing", "close_trail_atr_20"], v)}
                    step="0.1"
                    min={0}
                    max={5}
                />
                <FieldNum
                    label="Kapanış sıkılaştırma — kazanç ATR (1.5)"
                    hint="Daha düşük kazanç eşiğinde ikinci kapanış trail basamağı."
                    value={nn(["backtest_exit_trailing", "close_gain_atr_15"], 1.5)}
                    onChange={(v) => sn(["backtest_exit_trailing", "close_gain_atr_15"], v)}
                    step="0.1"
                    min={0.5}
                    max={10}
                />
                <FieldNum
                    label="Kapanış trail ×ATR (1.5 basamak)"
                    hint="1.5 ATR kapanış kazancı için trail mesafesi."
                    value={nn(["backtest_exit_trailing", "close_trail_atr_15"], 1.2)}
                    onChange={(v) => sn(["backtest_exit_trailing", "close_trail_atr_15"], v)}
                    step="0.1"
                    min={0}
                    max={5}
                />
            </Section>

            <details
                id="settings-section-skorlama"
                className="glass-card settings-section-details"
                style={{ marginBottom: 12, scrollMarginTop: 96, padding: 0 }}
            >
                <summary
                    style={{
                        cursor: "pointer",
                        padding: "16px 20px",
                        display: "flex",
                        alignItems: "flex-start",
                        gap: 12,
                        userSelect: "none",
                    }}
                >
                    <ChevronDown size={18} className="settings-section-chevron" aria-hidden style={{ marginTop: 2 }} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                        <SectionTitleWithHelp
                            title="Skorlama ayarları"
                            help="Kalite skorunu oluşturan kademeli tablolar (hacim patlaması, ATR%, float bantları), bileşen ağırlıkları ve ham puan tavanları. Kimin listede üst sıralara çıkacağını belirler; sıkılaştırırsanız daha az hisse ‘yüksek skor’ alır. Kademe tabloları boş bırakılamaz (kayıt reddeder); ağırlık toplamını ~1.0 tutmak iyi pratiktir."
                            insideSummary
                        />
                    </div>
                </summary>
                <div style={{ padding: "0 20px 20px 22px", borderTop: "1px solid var(--border-muted)" }}>
                    <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: 14, lineHeight: 1.55, paddingTop: 14 }}>
                        Ağırlıklar, bonus/ceza ve ham bileşen tavanları. Kalite skoru 0–{nn(["scoring_tuning", "final_score_max"], 140)}. Kademeleri
                        tablolardan düzenleyin; en az bir satır gerekir (kaydetme API doğrular).
                    </p>
                    <details
                        id="settings-details-skorlama-kademe-tablolari"
                        className="settings-nested-details"
                        style={{ marginBottom: 16, borderTop: "1px solid var(--border)", paddingTop: 12, scrollMarginTop: 92 }}
                    >
                        <summary
                            style={{
                                cursor: "pointer",
                                display: "flex",
                                alignItems: "center",
                                gap: 8,
                                fontWeight: 600,
                                fontSize: "0.84rem",
                                marginBottom: 8,
                                color: "var(--text-primary)",
                                listStyle: "none",
                            }}
                        >
                            <ChevronDown size={16} className="settings-section-chevron" aria-hidden />
                            Kademe tabloları (volume / ATR / float)
                        </summary>
                        <ScoringTierTables draft={draftRec} setDraft={setDraft} />
                    </details>
                    <details
                        id="settings-details-skorlama-momentum-ham"
                        className="settings-nested-details"
                        style={{ marginBottom: 18, borderTop: "1px solid var(--border)", paddingTop: 12, scrollMarginTop: 92 }}
                    >
                        <summary
                            style={{
                                cursor: "pointer",
                                display: "flex",
                                alignItems: "center",
                                gap: 8,
                                fontWeight: 600,
                                fontSize: "0.84rem",
                                marginBottom: 8,
                                color: "var(--text-primary)",
                            }}
                        >
                            <ChevronDown size={16} className="settings-section-chevron" aria-hidden />
                            Momentum ve risk — ham alt puanlar
                        </summary>
                    <div
                        style={{
                            display: "grid",
                            gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
                            gap: "8px 22px",
                        }}
                    >
                        <FieldNum
                            label="HH tam seri puan"
                            hint="Üst üste higher high serisi tam ise verilen ham momentum puanı."
                            value={nn(["scoring_tuning", "momentum_points", "higher_highs_full"], 6)}
                            onChange={(v) => sn(["scoring_tuning", "momentum_points", "higher_highs_full"], Math.round(v))}
                            step={1}
                            max={20}
                        />
                        <FieldNum
                            label="HH kısmi puan"
                            hint="HH yapısı kısmen sağlanıyorsa daha düşük puan."
                            value={nn(["scoring_tuning", "momentum_points", "higher_highs_partial"], 3)}
                            onChange={(v) => sn(["scoring_tuning", "momentum_points", "higher_highs_partial"], Math.round(v))}
                            step={1}
                            max={20}
                        />
                        <FieldNum
                            label="Kapanış tam seri puan"
                            hint="Üst üste higher close serisi tam ise puan."
                            value={nn(["scoring_tuning", "momentum_points", "higher_closes_full"], 6)}
                            onChange={(v) => sn(["scoring_tuning", "momentum_points", "higher_closes_full"], Math.round(v))}
                            step={1}
                            max={20}
                        />
                        <FieldNum
                            label="Kapanış kısmi puan"
                            hint="Kapanış yapısı kısmen uyuyorsa puan."
                            value={nn(["scoring_tuning", "momentum_points", "higher_closes_partial"], 3)}
                            onChange={(v) => sn(["scoring_tuning", "momentum_points", "higher_closes_partial"], Math.round(v))}
                            step={1}
                            max={20}
                        />
                        <FieldNum
                            label="Gün içi üst bölge min (0–1)"
                            hint="Kapanışın gün aralığının en üst % kaçında olması gerekir (örn. 0.8 = üst %20’de)."
                            value={nn(["scoring_tuning", "momentum_points", "close_in_top_of_range_min"], 0.8)}
                            onChange={(v) => sn(["scoring_tuning", "momentum_points", "close_in_top_of_range_min"], v)}
                            step="0.05"
                            min={0}
                            max={1}
                        />
                        <FieldNum
                            label="Üst bölge kapanış puanı"
                            hint="Bu eşik sağlanınca eklenen ‘güçlü kapanış’ puanı."
                            value={nn(["scoring_tuning", "momentum_points", "close_near_high_pts"], 3)}
                            onChange={(v) => sn(["scoring_tuning", "momentum_points", "close_near_high_pts"], Math.round(v))}
                            step={1}
                            max={20}
                        />
                        <FieldNum
                            label="Momentum ham tavan"
                            hint="Momentum alt skorunun toplamda aşamayacağı üst sınır (ağırlıktan önce)."
                            value={nn(["scoring_tuning", "momentum_points", "raw_cap"], 15)}
                            onChange={(v) => sn(["scoring_tuning", "momentum_points", "raw_cap"], Math.round(v))}
                            step={1}
                            min={1}
                            max={50}
                        />
                        <FieldNum
                            label="Yetersiz bar skoru"
                            hint="Geçmiş bar yetersizse momentum yerine verilen düşük/düz puan."
                            value={nn(["scoring_tuning", "momentum_points", "insufficient_bars_score"], 5)}
                            onChange={(v) => sn(["scoring_tuning", "momentum_points", "insufficient_bars_score"], Math.round(v))}
                            step={1}
                            max={20}
                        />
                        <FieldNum
                            label="Stop ≤%5 puan"
                            hint="Stop mesafesi girişe göre ≤%5 ise risk skoruna eklenen puan (sıkı stop = iyi R)."
                            value={nn(["scoring_tuning", "risk_bands", "stop_le_05_pts"], 10)}
                            onChange={(v) => sn(["scoring_tuning", "risk_bands", "stop_le_05_pts"], Math.round(v))}
                            step={1}
                            max={30}
                        />
                        <FieldNum
                            label="Stop ≤%8 puan"
                            hint="Stop ≤%8 bandı için puan (biraz daha geniş)."
                            value={nn(["scoring_tuning", "risk_bands", "stop_le_08_pts"], 7)}
                            onChange={(v) => sn(["scoring_tuning", "risk_bands", "stop_le_08_pts"], Math.round(v))}
                            step={1}
                            max={30}
                        />
                        <FieldNum
                            label="Stop ≤%10 puan"
                            hint="Stop ≤%10 bandı; daha geniş stop, daha az puan."
                            value={nn(["scoring_tuning", "risk_bands", "stop_le_10_pts"], 5)}
                            onChange={(v) => sn(["scoring_tuning", "risk_bands", "stop_le_10_pts"], Math.round(v))}
                            step={1}
                            max={30}
                        />
                        <FieldNum
                            label="Stop geniş puan"
                            hint="Yukarıdaki bantların dışında kalan geniş stop için taban puan."
                            value={nn(["scoring_tuning", "risk_bands", "stop_else_pts"], 3)}
                            onChange={(v) => sn(["scoring_tuning", "risk_bands", "stop_else_pts"], Math.round(v))}
                            step={1}
                            max={30}
                        />
                        <FieldNum
                            label="Gün aralığı ≤%5 puan"
                            hint="Günlük high−low çok dar ise (≤%5) ek puan — sıkışık yapı."
                            value={nn(["scoring_tuning", "risk_bands", "range_le_05_pts"], 5)}
                            onChange={(v) => sn(["scoring_tuning", "risk_bands", "range_le_05_pts"], Math.round(v))}
                            step={1}
                            max={20}
                        />
                        <FieldNum
                            label="Gün aralığı ≤%8 puan"
                            hint="Biraz daha geniş ama hâlä kontrollü gün aralığı için puan."
                            value={nn(["scoring_tuning", "risk_bands", "range_le_08_pts"], 3)}
                            onChange={(v) => sn(["scoring_tuning", "risk_bands", "range_le_08_pts"], Math.round(v))}
                            step={1}
                            max={20}
                        />
                        <FieldNum
                            label="Risk ham tavan"
                            hint="Risk alt skorunun (stop + aralık) ham toplam tavanı."
                            value={nn(["scoring_tuning", "risk_bands", "raw_cap"], 15)}
                            onChange={(v) => sn(["scoring_tuning", "risk_bands", "raw_cap"], Math.round(v))}
                            step={1}
                            min={1}
                            max={50}
                        />
                        <FieldNum
                            label="Risk yetersiz veri skoru"
                            hint="ATR/stop hesaplanamıyorsa risk bileşenine verilen yedek skor."
                            value={nn(["scoring_tuning", "risk_bands", "insufficient_bars_score"], 5)}
                            onChange={(v) => sn(["scoring_tuning", "risk_bands", "insufficient_bars_score"], Math.round(v))}
                            step={1}
                            max={20}
                        />
                    </div>
                </details>
                <div
                    style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
                        gap: "8px 22px",
                    }}
                >
                    <FieldNum
                        label="Ağırlık — hacim"
                        hint="Beş bileşen ağırlığının toplamı pratikte 1.0 olmalı; motor skorları bu oranlarla birleştirir."
                        value={nn(["scoring_tuning", "weight_volume"], 0.3)}
                        onChange={(v) => sn(["scoring_tuning", "weight_volume"], v)}
                        step="0.05"
                        min={0}
                        max={1}
                    />
                    <FieldNum
                        label="Ağırlık — volatilite"
                        hint="ATR% / volatilite bileşeninin payı."
                        value={nn(["scoring_tuning", "weight_volatility"], 0.2)}
                        onChange={(v) => sn(["scoring_tuning", "weight_volatility"], v)}
                        step="0.05"
                        min={0}
                        max={1}
                    />
                    <FieldNum
                        label="Ağırlık — float"
                        hint="Dar float skorunun payı."
                        value={nn(["scoring_tuning", "weight_float"], 0.2)}
                        onChange={(v) => sn(["scoring_tuning", "weight_float"], v)}
                        step="0.05"
                        min={0}
                        max={1}
                    />
                    <FieldNum
                        label="Ağırlık — momentum"
                        hint="HH/HL ve kapanış yapısı gibi momentum alt skorunun payı."
                        value={nn(["scoring_tuning", "weight_momentum"], 0.15)}
                        onChange={(v) => sn(["scoring_tuning", "weight_momentum"], v)}
                        step="0.05"
                        min={0}
                        max={1}
                    />
                    <FieldNum
                        label="Ağırlık — risk"
                        hint="Stop ve gün aralığına dayalı risk alt skorunun payı."
                        value={nn(["scoring_tuning", "weight_risk"], 0.15)}
                        onChange={(v) => sn(["scoring_tuning", "weight_risk"], v)}
                        step="0.05"
                        min={0}
                        max={1}
                    />
                    <FieldNum
                        label="Max ham — hacim"
                        hint="Hacim bileşeninden ham puana eklenebilecek üst sınır (kademe tablosu sonrası)."
                        value={nn(["scoring_tuning", "max_volume_score"], 30)}
                        onChange={(v) => sn(["scoring_tuning", "max_volume_score"], v)}
                        step={1}
                        min={1}
                        max={100}
                    />
                    <FieldNum
                        label="Max ham — volatilite"
                        hint="Volatilite ham puan tavanı."
                        value={nn(["scoring_tuning", "max_volatility_score"], 25)}
                        onChange={(v) => sn(["scoring_tuning", "max_volatility_score"], v)}
                        step={1}
                        min={1}
                        max={100}
                    />
                    <FieldNum
                        label="Max ham — float"
                        hint="Float bileşeni ham puan tavanı."
                        value={nn(["scoring_tuning", "max_float_score"], 20)}
                        onChange={(v) => sn(["scoring_tuning", "max_float_score"], v)}
                        step={1}
                        min={1}
                        max={100}
                    />
                    <FieldNum
                        label="Max ham — momentum"
                        hint="Momentum ham puan tavanı."
                        value={nn(["scoring_tuning", "max_momentum_score"], 15)}
                        onChange={(v) => sn(["scoring_tuning", "max_momentum_score"], v)}
                        step={1}
                        min={1}
                        max={100}
                    />
                    <FieldNum
                        label="Max ham — risk"
                        hint="Risk ham puan tavanı."
                        value={nn(["scoring_tuning", "max_risk_score"], 15)}
                        onChange={(v) => sn(["scoring_tuning", "max_risk_score"], v)}
                        step={1}
                        min={1}
                        max={100}
                    />
                    <FieldNum
                        label="Bonus tavanı"
                        hint="Katalizör / RVOL / swing ready vb. bonusların toplamına uygulanan üst sınır."
                        value={nn(["scoring_tuning", "bonus_cap"], 40)}
                        onChange={(v) => sn(["scoring_tuning", "bonus_cap"], Math.round(v))}
                        step={1}
                        min={0}
                        max={100}
                    />
                    <FieldNum
                        label="Final skor tavanı"
                        hint="Listede gördüğünüz kalite skorunun teorik üst sınırı (normalize üst tavan)."
                        value={nn(["scoring_tuning", "final_score_max"], 140)}
                        onChange={(v) => sn(["scoring_tuning", "final_score_max"], Math.round(v))}
                        step={1}
                        min={50}
                        max={300}
                    />
                    <FieldNum
                        label="Risk skoru ATR çarpanı"
                        hint="Risk alt skorunda stop genişliğini ATR ile ölçeklerken kullanılan çarpan."
                        value={nn(["scoring_tuning", "risk_score_atr_mult"], 1.5)}
                        onChange={(v) => sn(["scoring_tuning", "risk_score_atr_mult"], v)}
                        step="0.1"
                        min={0.5}
                        max={5}
                    />
                    <FieldNum
                        label="Bonus high RVOL"
                        hint="Günlük rel. volume çok yüksekse kalite skoruna eklenen bonus (bonus_cap altında)."
                        value={nn(["scoring_tuning", "bonus_high_rvol"], 3)}
                        onChange={(v) => sn(["scoring_tuning", "bonus_high_rvol"], Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Bonus gap devam"
                        hint="Önceki gün gap’i ve bugünkü devam yapısı uyuyorsa ek puan."
                        value={nn(["scoring_tuning", "bonus_gap_continuation"], 4)}
                        onChange={(v) => sn(["scoring_tuning", "bonus_gap_continuation"], Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Bonus higher highs"
                        hint="HH yapısı skorlamada zaten var; burada ekstra küçük bonus katsayısı."
                        value={nn(["scoring_tuning", "bonus_higher_highs"], 3)}
                        onChange={(v) => sn(["scoring_tuning", "bonus_higher_highs"], Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Bonus swing ready"
                        hint="MA20 / yapı ‘hazır’ filtresi geçildiğinde en büyük tek satır bonuslardan biri."
                        value={nn(["scoring_tuning", "bonus_swing_ready"], 10)}
                        onChange={(v) => sn(["scoring_tuning", "bonus_swing_ready"], Math.round(v))}
                        step={1}
                        max={50}
                    />
                    <FieldNum
                        label="Bonus higher lows"
                        hint="Yükselen dipler (trend sağlığı) için küçük ek bonus."
                        value={nn(["scoring_tuning", "bonus_higher_lows"], 5)}
                        onChange={(v) => sn(["scoring_tuning", "bonus_higher_lows"], Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Bonus multi-day vol"
                        hint="Ardışık günlerde hacim birikimi tespit edilirse ek puan."
                        value={nn(["scoring_tuning", "bonus_multi_day_volume"], 3)}
                        onChange={(v) => sn(["scoring_tuning", "bonus_multi_day_volume"], Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Bonus surge gün ≥3"
                        hint="Son N günde en az 3 gün volume surge koşulu — sıkı kurulum."
                        value={nn(["scoring_tuning", "bonus_surge_days_3"], 5)}
                        onChange={(v) => sn(["scoring_tuning", "bonus_surge_days_3"], Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Bonus surge gün ≥2"
                        hint="Daha az sıkı: 2 gün surge ile ek puan."
                        value={nn(["scoring_tuning", "bonus_surge_days_2"], 3)}
                        onChange={(v) => sn(["scoring_tuning", "bonus_surge_days_2"], Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Erken giriş 5g min %"
                        hint="5g getiri bu ile üst sınır arasında ‘erken trend’ bonusu için bant altı."
                        value={nn(["scoring_tuning", "bonus_early_entry_lo"], 5)}
                        onChange={(v) => sn(["scoring_tuning", "bonus_early_entry_lo"], v)}
                        max={50}
                    />
                    <FieldNum
                        label="Erken giriş 5g max %"
                        hint="Erken giriş bandının üst sınırı; çok koşmuş hisseye açık bırakmamak için."
                        value={nn(["scoring_tuning", "bonus_early_entry_hi"], 15)}
                        onChange={(v) => sn(["scoring_tuning", "bonus_early_entry_hi"], v)}
                        max={80}
                    />
                    <FieldNum
                        label="Erken giriş puan"
                        hint="Bant içindeyse kaliteye eklenen ham puan."
                        value={nn(["scoring_tuning", "bonus_early_entry_pts"], 8)}
                        onChange={(v) => sn(["scoring_tuning", "bonus_early_entry_pts"], Math.round(v))}
                        max={40}
                    />
                    <FieldNum
                        label="Çok erken 5g üst %"
                        hint="5g getiri bu kadar düşükse ‘çok erken’ sayılır; ayrı küçük bonus/etiket."
                        value={nn(["scoring_tuning", "bonus_very_early_hi"], 5)}
                        onChange={(v) => sn(["scoring_tuning", "bonus_very_early_hi"], v)}
                        max={30}
                    />
                    <FieldNum
                        label="Çok erken puan"
                        hint="Çok erken senaryoda eklenen puan (düşük momentum ile uyumlu)."
                        value={nn(["scoring_tuning", "bonus_very_early_pts"], 5)}
                        onChange={(v) => sn(["scoring_tuning", "bonus_very_early_pts"], Math.round(v))}
                        max={30}
                    />
                    <FieldNum
                        label="Bonus RSI divergence"
                        hint="Fiyat vs RSI uyumsuzluğu (momentum sönümü) tespit edilirse ek puan."
                        value={nn(["scoring_tuning", "bonus_rsi_divergence"], 8)}
                        onChange={(v) => sn(["scoring_tuning", "bonus_rsi_divergence"], Math.round(v))}
                        max={30}
                    />
                    <FieldNum
                        label="Ceza A RSI&gt;70"
                        hint="Tip A sınıflandırmasında RSI aşırı yüksekse kaliteden düşülen puan."
                        value={nn(["scoring_tuning", "pen_a_rsi_gt_70"], 10)}
                        onChange={(v) => sn(["scoring_tuning", "pen_a_rsi_gt_70"], Math.round(v))}
                        max={50}
                    />
                    <FieldNum
                        label="Ceza A RSI&gt;65"
                        hint="Daha düşük eşikte ikinci kademe A cezası."
                        value={nn(["scoring_tuning", "pen_a_rsi_gt_65"], 5)}
                        onChange={(v) => sn(["scoring_tuning", "pen_a_rsi_gt_65"], Math.round(v))}
                        max={50}
                    />
                    <FieldNum
                        label="Ceza B RSI&gt;85"
                        hint="Momentum tip B için çok ısınmış RSI cezası."
                        value={nn(["scoring_tuning", "pen_b_rsi_gt_85"], 15)}
                        onChange={(v) => sn(["scoring_tuning", "pen_b_rsi_gt_85"], Math.round(v))}
                        max={50}
                    />
                    <FieldNum
                        label="Ceza B RSI&gt;80"
                        hint="Orta-yüksek RSI bandı cezası."
                        value={nn(["scoring_tuning", "pen_b_rsi_gt_80"], 10)}
                        onChange={(v) => sn(["scoring_tuning", "pen_b_rsi_gt_80"], Math.round(v))}
                        max={50}
                    />
                    <FieldNum
                        label="Ceza B RSI&gt;75"
                        hint="Hafif B cezası (daha erken uyarı)."
                        value={nn(["scoring_tuning", "pen_b_rsi_gt_75"], 5)}
                        onChange={(v) => sn(["scoring_tuning", "pen_b_rsi_gt_75"], Math.round(v))}
                        max={50}
                    />
                    <FieldNum
                        label="Ceza C RSI&gt;65"
                        hint="Erken tip C için ‘çok geç / ısınmış’ RSI cezası."
                        value={nn(["scoring_tuning", "pen_c_rsi_gt_65"], 10)}
                        onChange={(v) => sn(["scoring_tuning", "pen_c_rsi_gt_65"], Math.round(v))}
                        max={50}
                    />
                    <FieldNum
                        label="Ceza C RSI&gt;60"
                        hint="Daha hafif C RSI cezası."
                        value={nn(["scoring_tuning", "pen_c_rsi_gt_60"], 5)}
                        onChange={(v) => sn(["scoring_tuning", "pen_c_rsi_gt_60"], Math.round(v))}
                        max={50}
                    />
                    <FieldNum
                        label="Ceza tek gün &gt;25%"
                        hint="Tek günde aşırı hareket (ör. %25+) kaliteyi düşürür."
                        value={nn(["scoring_tuning", "pen_ext_day_gt_25"], 15)}
                        onChange={(v) => sn(["scoring_tuning", "pen_ext_day_gt_25"], Math.round(v))}
                        max={50}
                    />
                    <FieldNum
                        label="Ceza tek gün &gt;20%"
                        hint="Bir alt kademe tek gün aşırı hareket cezası."
                        value={nn(["scoring_tuning", "pen_ext_day_gt_20"], 8)}
                        onChange={(v) => sn(["scoring_tuning", "pen_ext_day_gt_20"], Math.round(v))}
                        max={50}
                    />
                    <FieldNum
                        label="Ceza bugün &gt;15%"
                        hint="Bugünkü günlük % değişim bu üstündeyse ceza."
                        value={nn(["scoring_tuning", "pen_today_gt_15"], 10)}
                        onChange={(v) => sn(["scoring_tuning", "pen_today_gt_15"], Math.round(v))}
                        max={50}
                    />
                    <FieldNum
                        label="Ceza bugün &gt;10%"
                        hint="Daha düşük eşikte günlük hareket cezası."
                        value={nn(["scoring_tuning", "pen_today_gt_10"], 5)}
                        onChange={(v) => sn(["scoring_tuning", "pen_today_gt_10"], Math.round(v))}
                        max={50}
                    />
                    <FieldNum
                        label="Ceza 5g &gt;40%"
                        hint="Beş günlük toplam getiri aşırı yüksekse agresif ceza."
                        value={nn(["scoring_tuning", "pen_5d_gt_40"], 15)}
                        onChange={(v) => sn(["scoring_tuning", "pen_5d_gt_40"], Math.round(v))}
                        max={50}
                    />
                    <FieldNum
                        label="Ceza 5g &gt;30%"
                        hint="Orta-yüksek 5g ralli cezası."
                        value={nn(["scoring_tuning", "pen_5d_gt_30"], 10)}
                        onChange={(v) => sn(["scoring_tuning", "pen_5d_gt_30"], Math.round(v))}
                        max={50}
                    />
                    <FieldNum
                        label="Ceza 5g &gt;25%"
                        hint="Hafif 5g uzama uyarısı."
                        value={nn(["scoring_tuning", "pen_5d_gt_25"], 5)}
                        onChange={(v) => sn(["scoring_tuning", "pen_5d_gt_25"], Math.round(v))}
                        max={50}
                    />
                    <FieldNum
                        label="Ceza parabolik"
                        hint="Parabolik / blow-off paternine girildiğinde kaliteden düşülen toplam puan."
                        value={nn(["scoring_tuning", "pen_parabolic"], 15)}
                        onChange={(v) => sn(["scoring_tuning", "pen_parabolic"], Math.round(v))}
                        max={50}
                    />
                    <FieldNum
                        label="Parabolik gün3 min %"
                        hint="3. gün kapanışında getiri bu altındaysa parabolik sayılmaz (ek koşul)."
                        value={nn(["scoring_tuning", "parabolic_day3_min_pct"], 10)}
                        onChange={(v) => sn(["scoring_tuning", "parabolic_day3_min_pct"], v)}
                        max={50}
                    />
                    <FieldNum
                        label="Ceza swing ready değil"
                        hint="Yapı / MA20 hazır değilse kaliteye küçük ceza."
                        value={nn(["scoring_tuning", "pen_not_swing_ready"], 5)}
                        onChange={(v) => sn(["scoring_tuning", "pen_not_swing_ready"], Math.round(v))}
                        max={30}
                    />
                </div>
                </div>
            </details>

            <details
                id="settings-section-swing-siniflandirma-gelismis"
                className="glass-card settings-section-details"
                style={{ padding: 0, marginBottom: 18, scrollMarginTop: 92 }}
            >
                <summary
                    style={{
                        cursor: "pointer",
                        padding: "16px 20px",
                        display: "flex",
                        alignItems: "flex-start",
                        gap: 12,
                        userSelect: "none",
                    }}
                >
                    <ChevronDown size={18} className="settings-section-chevron" aria-hidden style={{ marginTop: 2 }} />
                    <span style={{ fontWeight: 700, fontSize: "0.93rem", color: "var(--text-primary)", lineHeight: 1.35 }}>
                        Swing sınıflandırma (S / C / B / A) — gelişmiş
                    </span>
                </summary>
                <div style={{ padding: "0 20px 20px 22px", borderTop: "1px solid var(--border-muted)" }}>
                    <BlockHelp text="Hisseyi S (squeeze), C (erken), B (momentum), A (devam) olarak etiketleyen eşikler ve tutma gün aralıkları. Yanlış tip çok görüyorsanız önce tek tipin bandını oynatın; hepsini birden değiştirmek sonucu okunmaz yapar. Parabolik satırlarda tutma (min, max) çifti kullanılır." />
                    <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", margin: "0 0 16px" }}>
                        Tip atama eşikleri ve tutma gün aralıkları. Parabolik satırlar için tutma: iki değer (min, max gün).
                    </p>
                    <div
                        style={{
                            display: "grid",
                            gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
                            gap: "8px 22px",
                            marginTop: 8,
                        }}
                    >
                    <div style={{ gridColumn: "1 / -1", fontWeight: 600, fontSize: "0.82rem", color: "var(--text-secondary)" }}>Parabolik</div>
                    <FieldNum
                        label="5g &gt; % (parabolik)"
                        hint="Bu 5g getiri üstünde parabolik sayılmaya aday (motor diğer koşullarla birleştirir)."
                        value={nn(["swing", "parabolic", "five_day_gt"], 70)}
                        onChange={(v) => sn(["swing", "parabolic", "five_day_gt"], v)}
                        min={10}
                        max={200}
                    />
                    <FieldNum
                        label="5g ekstrem &gt; %"
                        hint="Daha uç ‘ekstrem’ parabolik bandı."
                        value={nn(["swing", "parabolic", "five_day_extreme_gt"], 60)}
                        onChange={(v) => sn(["swing", "parabolic", "five_day_extreme_gt"], v)}
                        min={10}
                        max={200}
                    />
                    <FieldNum
                        label="RSI ekstrem &gt;"
                        hint="Parabolik senaryoda RSI üst eşiği."
                        value={nn(["swing", "parabolic", "rsi_extreme_gt"], 85)}
                        onChange={(v) => sn(["swing", "parabolic", "rsi_extreme_gt"], v)}
                        min={50}
                        max={100}
                    />
                    <FieldNum
                        label="Tutma min (gün)"
                        hint="Parabolik kısa tutma aralığının alt sınırı (tuple ilk eleman)."
                        value={nt(["swing", "parabolic", "hold_short"], 0, 1)}
                        onChange={(v) => st(["swing", "parabolic", "hold_short"], 0, Math.round(v))}
                        step={1}
                        min={1}
                        max={14}
                    />
                    <FieldNum
                        label="Tutma max (gün)"
                        hint="Parabolik kısa tutma aralığının üst sınırı."
                        value={nt(["swing", "parabolic", "hold_short"], 1, 2)}
                        onChange={(v) => st(["swing", "parabolic", "hold_short"], 1, Math.round(v))}
                        step={1}
                        min={1}
                        max={14}
                    />

                    <div style={{ gridColumn: "1 / -1", fontWeight: 600, fontSize: "0.84rem", color: "var(--text-secondary)", marginTop: 8 }}>Tip S — birincil</div>
                    <FieldNum
                        label="SI min %"
                        hint="Short interest (%) — squeeze için birincil yol alt sınırı."
                        value={nn(["swing", "type_s", "primary_si_min"], 20)}
                        onChange={(v) => sn(["swing", "type_s", "primary_si_min"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="DTC min"
                        hint="Days to cover — likidite sıkışması için minimum."
                        value={nn(["swing", "type_s", "primary_dtc_min"], 5)}
                        onChange={(v) => sn(["swing", "type_s", "primary_dtc_min"], v)}
                        max={30}
                    />
                    <FieldNum
                        label="Vol min ×"
                        hint="Volume surge tetikleyicisi (× ortalama) — birincil S için."
                        value={nn(["swing", "type_s", "primary_vol_min"], 4)}
                        onChange={(v) => sn(["swing", "type_s", "primary_vol_min"], v)}
                        step="0.1"
                        max={10}
                    />
                    <FieldNum
                        label="5g min %"
                        hint="Son 5 gün getirisi bandı alt sınırı (birincil S)."
                        value={nn(["swing", "type_s", "primary_5d_min"], 15)}
                        onChange={(v) => sn(["swing", "type_s", "primary_5d_min"], v)}
                        max={200}
                    />
                    <FieldNum
                        label="5g max %"
                        hint="5g getiri üst sınırı — çok uçmuş squeeze dışı bırakmak için."
                        value={nn(["swing", "type_s", "primary_5d_max"], 60)}
                        onChange={(v) => sn(["swing", "type_s", "primary_5d_max"], v)}
                        max={200}
                    />
                    <FieldNum
                        label="RSI min"
                        hint="Momentum için RSI taban."
                        value={nn(["swing", "type_s", "primary_rsi_min"], 60)}
                        onChange={(v) => sn(["swing", "type_s", "primary_rsi_min"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="RSI max"
                        hint="Çok aşırı alımı S dışı bırakmak için RSI tavanı."
                        value={nn(["swing", "type_s", "primary_rsi_max"], 80)}
                        onChange={(v) => sn(["swing", "type_s", "primary_rsi_max"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="Tutma min"
                        hint="Beklenen tutma günü aralığı (birincil yol)."
                        value={nn(["swing", "type_s", "primary_hold_min"], 1)}
                        onChange={(v) => sn(["swing", "type_s", "primary_hold_min"], Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Tutma max"
                        hint="Beklenen tutma günü üst sınırı."
                        value={nn(["swing", "type_s", "primary_hold_max"], 4)}
                        onChange={(v) => sn(["swing", "type_s", "primary_hold_max"], Math.round(v))}
                        step={1}
                        max={30}
                    />

                    <div style={{ gridColumn: "1 / -1", fontWeight: 600, fontSize: "0.84rem", color: "var(--text-secondary)", marginTop: 8 }}>Tip S — ikincil</div>
                    <FieldNum
                        label="SI min %"
                        hint="İkincil S yolu: daha düşük SI ile de squeeze kabulü."
                        value={nn(["swing", "type_s", "secondary_si_min"], 15)}
                        onChange={(v) => sn(["swing", "type_s", "secondary_si_min"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="DTC min"
                        hint="İkincil yol için DTC tabanı."
                        value={nn(["swing", "type_s", "secondary_dtc_min"], 3)}
                        onChange={(v) => sn(["swing", "type_s", "secondary_dtc_min"], v)}
                        max={30}
                    />
                    <FieldNum
                        label="Vol min ×"
                        hint="İkincil yol için volume surge eşiği (genelde birincilden düşük)."
                        value={nn(["swing", "type_s", "secondary_vol_min"], 3)}
                        onChange={(v) => sn(["swing", "type_s", "secondary_vol_min"], v)}
                        step="0.1"
                        max={10}
                    />
                    <FieldNum
                        label="5g min %"
                        hint="İkincil yol 5g bandı alt sınırı."
                        value={nn(["swing", "type_s", "secondary_5d_min"], 10)}
                        onChange={(v) => sn(["swing", "type_s", "secondary_5d_min"], v)}
                        max={200}
                    />
                    <FieldNum
                        label="5g max %"
                        hint="İkincil yol 5g üst sınırı."
                        value={nn(["swing", "type_s", "secondary_5d_max"], 40)}
                        onChange={(v) => sn(["swing", "type_s", "secondary_5d_max"], v)}
                        max={200}
                    />
                    <FieldNum
                        label="RSI min"
                        hint="İkincil yol RSI tabanı."
                        value={nn(["swing", "type_s", "secondary_rsi_min"], 55)}
                        onChange={(v) => sn(["swing", "type_s", "secondary_rsi_min"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="RSI max"
                        hint="İkincil yol RSI tavanı."
                        value={nn(["swing", "type_s", "secondary_rsi_max"], 75)}
                        onChange={(v) => sn(["swing", "type_s", "secondary_rsi_max"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="Tutma min"
                        hint="İkincil yol tutma günü alt sınırı."
                        value={nn(["swing", "type_s", "secondary_hold_min"], 2)}
                        onChange={(v) => sn(["swing", "type_s", "secondary_hold_min"], Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Tutma max"
                        hint="İkincil yol tutma günü üst sınırı."
                        value={nn(["swing", "type_s", "secondary_hold_max"], 4)}
                        onChange={(v) => sn(["swing", "type_s", "secondary_hold_max"], Math.round(v))}
                        step={1}
                        max={30}
                    />

                    <div style={{ gridColumn: "1 / -1", fontWeight: 600, fontSize: "0.84rem", color: "var(--text-secondary)", marginTop: 8 }}>Tip C (skor)</div>
                    <FieldNum
                        label="5g min %"
                        hint="Erken devşirme: 5g getiri bu ile max arasında puanlanır (negatif = hafif düzeltme kabulü)."
                        value={nn(["swing", "type_c", "return_min"], -5)}
                        onChange={(v) => sn(["swing", "type_c", "return_min"], v)}
                        min={-50}
                        max={50}
                    />
                    <FieldNum
                        label="5g max %"
                        hint="Çok koşmuş hisseyi C dışı bırakmak için 5g üst sınırı."
                        value={nn(["swing", "type_c", "return_max"], 15)}
                        onChange={(v) => sn(["swing", "type_c", "return_max"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="5g bant puan"
                        hint="5g bandına uyduğunda skorlamaya eklenen taban puan."
                        value={nn(["swing", "type_c", "return_band_pts"], 4)}
                        onChange={(v) => sn(["swing", "type_c", "return_band_pts"], Math.round(v))}
                        max={20}
                    />
                    <FieldNum
                        label="Tatlı 5g min"
                        hint="‘Tatlı nokta’ erken giriş bandının alt sınırı (%)."
                        value={nn(["swing", "type_c", "sweet_return_min"], 0)}
                        onChange={(v) => sn(["swing", "type_c", "sweet_return_min"], v)}
                        max={50}
                    />
                    <FieldNum
                        label="Tatlı 5g max"
                        hint="Tatlı nokta bandının üst sınırı."
                        value={nn(["swing", "type_c", "sweet_return_max"], 10)}
                        onChange={(v) => sn(["swing", "type_c", "sweet_return_max"], v)}
                        max={50}
                    />
                    <FieldNum
                        label="Tatlı bonus puan"
                        hint="Tatlı bant içindeyken ek skor."
                        value={nn(["swing", "type_c", "sweet_bonus_pts"], 1)}
                        onChange={(v) => sn(["swing", "type_c", "sweet_bonus_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="RSI min"
                        hint="Tip C için RSI alt sınırı (aşırı satım değil, erken bölge)."
                        value={nn(["swing", "type_c", "rsi_min"], 40)}
                        onChange={(v) => sn(["swing", "type_c", "rsi_min"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="RSI max"
                        hint="Erken giriş için RSI üst sınırı (çok geç kalınmasın)."
                        value={nn(["swing", "type_c", "rsi_max"], 60)}
                        onChange={(v) => sn(["swing", "type_c", "rsi_max"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="RSI bant puan"
                        hint="RSI ana bandına uyum puanı."
                        value={nn(["swing", "type_c", "rsi_band_pts"], 4)}
                        onChange={(v) => sn(["swing", "type_c", "rsi_band_pts"], Math.round(v))}
                        max={20}
                    />
                    <FieldNum
                        label="RSI düşük max"
                        hint="Bu RSI altı ‘düşük’ kabul edilir (aşırı zayıf momentum)."
                        value={nn(["swing", "type_c", "rsi_low_max"], 50)}
                        onChange={(v) => sn(["swing", "type_c", "rsi_low_max"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="RSI düşük bonus"
                        hint="RSI düşük bandında contrarian erken giriş bonusu."
                        value={nn(["swing", "type_c", "rsi_low_bonus_pts"], 1)}
                        onChange={(v) => sn(["swing", "type_c", "rsi_low_bonus_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="RSI orta max"
                        hint="Orta momentum bandı üst sınırı."
                        value={nn(["swing", "type_c", "rsi_mid_max"], 65)}
                        onChange={(v) => sn(["swing", "type_c", "rsi_mid_max"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="RSI orta puan"
                        hint="Orta RSI bandında küçük ek puan."
                        value={nn(["swing", "type_c", "rsi_mid_pts"], 2)}
                        onChange={(v) => sn(["swing", "type_c", "rsi_mid_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="Vol min ×"
                        hint="Tip C için minimum volume surge (×)."
                        value={nn(["swing", "type_c", "vol_min"], 1.8)}
                        onChange={(v) => sn(["swing", "type_c", "vol_min"], v)}
                        step="0.1"
                        max={10}
                    />
                    <FieldNum
                        label="Vol max ×"
                        hint="Aşırı vol spike (manipülasyon) üst sınırı."
                        value={nn(["swing", "type_c", "vol_max"], 4)}
                        onChange={(v) => sn(["swing", "type_c", "vol_max"], v)}
                        step="0.1"
                        max={20}
                    />
                    <FieldNum
                        label="Vol bant puan"
                        hint="Vol min–max bandına uyduğunda puan."
                        value={nn(["swing", "type_c", "vol_band_pts"], 2)}
                        onChange={(v) => sn(["swing", "type_c", "vol_band_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="Vol yüksek min ×"
                        hint="Yüksek hacim onayı için ek eşik (×)."
                        value={nn(["swing", "type_c", "vol_high_min"], 2.5)}
                        onChange={(v) => sn(["swing", "type_c", "vol_high_min"], v)}
                        step="0.1"
                        max={10}
                    />
                    <FieldNum
                        label="Vol yüksek bonus"
                        hint="Yüksek vol onayında ek puan."
                        value={nn(["swing", "type_c", "vol_high_bonus_pts"], 1)}
                        onChange={(v) => sn(["swing", "type_c", "vol_high_bonus_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="MA mesafe min %"
                        hint="Fiyatın MA’ya göre min mesafesi (%; negatif = MA altı toleransı)."
                        value={nn(["swing", "type_c", "ma_dist_min"], -3)}
                        onChange={(v) => sn(["swing", "type_c", "ma_dist_min"], v)}
                        min={-50}
                        max={50}
                    />
                    <FieldNum
                        label="MA mesafe max %"
                        hint="MA’dan çok uzak ‘kopuk’ yapıları elenmek için üst sınır."
                        value={nn(["swing", "type_c", "ma_dist_max"], 8)}
                        onChange={(v) => sn(["swing", "type_c", "ma_dist_max"], v)}
                        max={50}
                    />
                    <FieldNum
                        label="MA bant puan"
                        hint="MA mesafe bandına uyum puanı."
                        value={nn(["swing", "type_c", "ma_band_pts"], 2)}
                        onChange={(v) => sn(["swing", "type_c", "ma_band_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="Kapanış konumu min"
                        hint="Kapanışın gün aralığında üst kısımda olması (0–1, örn. 0.55 = üst yarı)."
                        value={nn(["swing", "type_c", "close_position_min"], 0.55)}
                        onChange={(v) => sn(["swing", "type_c", "close_position_min"], v)}
                        step="0.05"
                        max={1}
                    />
                    <FieldNum
                        label="Kapanış konumu puan"
                        hint="Güçlü kapanışa verilen skor puanı."
                        value={nn(["swing", "type_c", "close_position_pts"], 1)}
                        onChange={(v) => sn(["swing", "type_c", "close_position_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="RSI div puan"
                        hint="RSI uyumsuzluğu (momentum sönümü) lehine puan."
                        value={nn(["swing", "type_c", "rsi_div_pts"], 3)}
                        onChange={(v) => sn(["swing", "type_c", "rsi_div_pts"], Math.round(v))}
                        max={15}
                    />
                    <FieldNum
                        label="MACD puan"
                        hint="MACD lehine küçük ek puan."
                        value={nn(["swing", "type_c", "macd_pts"], 1)}
                        onChange={(v) => sn(["swing", "type_c", "macd_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="Higher lows puan"
                        hint="Yükselen dipler (yapı) için ek puan."
                        value={nn(["swing", "type_c", "higher_lows_pts"], 1)}
                        onChange={(v) => sn(["swing", "type_c", "higher_lows_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="Min skor"
                        hint="Tip C olarak etiketlemek için toplam skor eşiği."
                        value={nn(["swing", "type_c", "min_score"], 10)}
                        onChange={(v) => sn(["swing", "type_c", "min_score"], Math.round(v))}
                        max={50}
                    />
                    <FieldNum
                        label="Tutma min"
                        hint="Beklenen tutma günü (Tip C) alt sınır."
                        value={nn(["swing", "type_c", "hold_min"], 3)}
                        onChange={(v) => sn(["swing", "type_c", "hold_min"], Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Tutma max"
                        hint="Beklenen tutma günü üst sınırı."
                        value={nn(["swing", "type_c", "hold_max"], 8)}
                        onChange={(v) => sn(["swing", "type_c", "hold_max"], Math.round(v))}
                        step={1}
                        max={30}
                    />

                    <div style={{ gridColumn: "1 / -1", fontWeight: 600, fontSize: "0.84rem", color: "var(--text-secondary)", marginTop: 8 }}>Tip B</div>
                    <FieldNum
                        label="5g 30–70 puan"
                        hint="Momentum (5g) %30–70 bandında verilen skor puanı (Tip B = güçlü trend devamı)."
                        value={nn(["swing", "type_b", "r_30_70_pts"], 3)}
                        onChange={(v) => sn(["swing", "type_b", "r_30_70_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="5g 20–30 puan"
                        hint="Daha düşük momentum bandı için daha düşük puan."
                        value={nn(["swing", "type_b", "r_20_30_pts"], 2)}
                        onChange={(v) => sn(["swing", "type_b", "r_20_30_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="5g &gt;70 puan"
                        hint="Çok koşmuş hisse için küçük puan (aşırı ısınma riski)."
                        value={nn(["swing", "type_b", "r_gt_70_pts"], 1)}
                        onChange={(v) => sn(["swing", "type_b", "r_gt_70_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="RSI 68–85 puan"
                        hint="Tip B için ana RSI bandı (güçlü ama henüz aşırı değil)."
                        value={nn(["swing", "type_b", "rsi_68_85_pts"], 3)}
                        onChange={(v) => sn(["swing", "type_b", "rsi_68_85_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="RSI 60–68 puan"
                        hint="Biraz daha erken / daha az ‘sıcak’ RSI bandı."
                        value={nn(["swing", "type_b", "rsi_60_68_pts"], 2)}
                        onChange={(v) => sn(["swing", "type_b", "rsi_60_68_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="RSI &gt;85 puan"
                        hint="Aşırı alım bölgesinde minimal puan (çöküş riski)."
                        value={nn(["swing", "type_b", "rsi_gt_85_pts"], 1)}
                        onChange={(v) => sn(["swing", "type_b", "rsi_gt_85_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="Vol yüksek puan"
                        hint="Yüksek hacim patlamasına verilen puan (ör. ~3.5× üstü senaryoda)."
                        value={nn(["swing", "type_b", "vol_35_pts"], 3)}
                        onChange={(v) => sn(["swing", "type_b", "vol_35_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="Vol orta puan"
                        hint="Orta seviye hacim onayı puanı."
                        value={nn(["swing", "type_b", "vol_25_pts"], 2)}
                        onChange={(v) => sn(["swing", "type_b", "vol_25_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="Vol ikincil eşik ×"
                        hint="RSI/vol kombinasyonunda ikinci bir hacim eşiği (×)."
                        value={nn(["swing", "type_b", "vol_surge_secondary_min"], 2.5)}
                        onChange={(v) => sn(["swing", "type_b", "vol_surge_secondary_min"], v)}
                        step="0.1"
                        max={10}
                    />
                    <FieldNum
                        label="Kapanış konumu min"
                        hint="Günün üstünde kapanış (0–1); Tip B’de sıkı eşik (örn. 0.75)."
                        value={nn(["swing", "type_b", "close_pos_min"], 0.75)}
                        onChange={(v) => sn(["swing", "type_b", "close_pos_min"], v)}
                        step="0.05"
                        max={1}
                    />
                    <FieldNum
                        label="Kapanış puan"
                        hint="Güçlü kapanışa eklenen skor."
                        value={nn(["swing", "type_b", "close_pos_pts"], 2)}
                        onChange={(v) => sn(["swing", "type_b", "close_pos_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="Katalizör puan"
                        hint="Haber/hikâye (katalizör) varsa küçük ek puan."
                        value={nn(["swing", "type_b", "catalyst_pts"], 1)}
                        onChange={(v) => sn(["swing", "type_b", "catalyst_pts"], Math.round(v))}
                        max={10}
                    />
                    <FieldNum
                        label="Min skor"
                        hint="Tip B etiketi için toplam skor eşiği."
                        value={nn(["swing", "type_b", "min_score"], 6)}
                        onChange={(v) => sn(["swing", "type_b", "min_score"], Math.round(v))}
                        max={30}
                    />
                    <FieldNum
                        label="Gate vol min ×"
                        hint="Vol bu eşiğin altındaysa Tip B ‘gate’ ile elenir (yetersiz ilgi)."
                        value={nn(["swing", "type_b", "gate_vol_min"], 3.5)}
                        onChange={(v) => sn(["swing", "type_b", "gate_vol_min"], v)}
                        step="0.1"
                        max={10}
                    />
                    <FieldNum
                        label="Gate güvenli RSI max"
                        hint="Vol düşükse RSI bu üst sınırın altında olmalı (güvenli momentum)."
                        value={nn(["swing", "type_b", "gate_rsi_safe_max"], 72)}
                        onChange={(v) => sn(["swing", "type_b", "gate_rsi_safe_max"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="Aşırı alım RSI &gt;"
                        hint="RSI bu değerin üstündeyse ‘aşırı alım’ tutma aralığı kullanılır."
                        value={nn(["swing", "type_b", "rsi_overbought_hold_gt"], 73)}
                        onChange={(v) => sn(["swing", "type_b", "rsi_overbought_hold_gt"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="Tutma aşırı alım min"
                        hint="Aşırı alımda beklenen tutma günü (min)."
                        value={nt(["swing", "type_b", "hold_overbought"], 0, 2)}
                        onChange={(v) => st(["swing", "type_b", "hold_overbought"], 0, Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Tutma aşırı alım max"
                        hint="Aşırı alımda beklenen tutma günü (max)."
                        value={nt(["swing", "type_b", "hold_overbought"], 1, 4)}
                        onChange={(v) => st(["swing", "type_b", "hold_overbought"], 1, Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Yükseltilmiş RSI &gt;"
                        hint="RSI bu ile aşırı alım arası ‘yükseltilmiş’ bölge (orta tutma)."
                        value={nn(["swing", "type_b", "rsi_elevated_gt"], 68)}
                        onChange={(v) => sn(["swing", "type_b", "rsi_elevated_gt"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="Tutma yükseltilmiş min"
                        hint="Yükseltilmiş RSI’da tutma günü (min)."
                        value={nt(["swing", "type_b", "hold_elevated"], 0, 3)}
                        onChange={(v) => st(["swing", "type_b", "hold_elevated"], 0, Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Tutma yükseltilmiş max"
                        hint="Yükseltilmiş RSI’da tutma günü (max)."
                        value={nt(["swing", "type_b", "hold_elevated"], 1, 5)}
                        onChange={(v) => st(["swing", "type_b", "hold_elevated"], 1, Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Tutma varsayılan min"
                        hint="Diğer durumlarda varsayılan tutma aralığı (min)."
                        value={nt(["swing", "type_b", "hold_default"], 0, 4)}
                        onChange={(v) => st(["swing", "type_b", "hold_default"], 0, Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Tutma varsayılan max"
                        hint="Varsayılan tutma aralığı (max)."
                        value={nt(["swing", "type_b", "hold_default"], 1, 6)}
                        onChange={(v) => st(["swing", "type_b", "hold_default"], 1, Math.round(v))}
                        step={1}
                        max={30}
                    />

                    <div style={{ gridColumn: "1 / -1", fontWeight: 600, fontSize: "0.84rem", color: "var(--text-secondary)", marginTop: 8 }}>Tip A</div>
                    <FieldNum
                        label="Erken 5g max %"
                        hint="Erken (daha düşük risk) Tip A için 5g üst sınırı; üstüne çıkarsa bu kademe uygun olmayabilir."
                        value={nn(["swing", "type_a", "five_d_max_early"], 15)}
                        onChange={(v) => sn(["swing", "type_a", "five_d_max_early"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="Erken RSI max"
                        hint="Erken kademede RSI üst sınırı (henüz aşırı ısınmamış)."
                        value={nn(["swing", "type_a", "rsi_max_early"], 55)}
                        onChange={(v) => sn(["swing", "type_a", "rsi_max_early"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="Erken tutma min"
                        hint="Erken kademe için beklenen tutma (min gün)."
                        value={nt(["swing", "type_a", "hold_early"], 0, 5)}
                        onChange={(v) => st(["swing", "type_a", "hold_early"], 0, Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Erken tutma max"
                        hint="Erken kademe için beklenen tutma (max gün)."
                        value={nt(["swing", "type_a", "hold_early"], 1, 10)}
                        onChange={(v) => st(["swing", "type_a", "hold_early"], 1, Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Standart 5g max %"
                        hint="Standart Tip A için 5g üst sınırı (daha gevşek)."
                        value={nn(["swing", "type_a", "five_d_max_std"], 25)}
                        onChange={(v) => sn(["swing", "type_a", "five_d_max_std"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="Standart RSI max"
                        hint="Standart kademede RSI üst sınırı."
                        value={nn(["swing", "type_a", "rsi_max_std"], 62)}
                        onChange={(v) => sn(["swing", "type_a", "rsi_max_std"], v)}
                        max={100}
                    />
                    <FieldNum
                        label="Standart tutma min"
                        hint="Standart kademe tutma (min gün)."
                        value={nt(["swing", "type_a", "hold_std"], 0, 7)}
                        onChange={(v) => st(["swing", "type_a", "hold_std"], 0, Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Standart tutma max"
                        hint="Standart kademe tutma (max gün)."
                        value={nt(["swing", "type_a", "hold_std"], 1, 12)}
                        onChange={(v) => st(["swing", "type_a", "hold_std"], 1, Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Uzatılmış tutma min"
                        hint="Uzatılmış senaryo (daha uzun swing) tutma (min)."
                        value={nt(["swing", "type_a", "hold_extended"], 0, 8)}
                        onChange={(v) => st(["swing", "type_a", "hold_extended"], 0, Math.round(v))}
                        step={1}
                        max={30}
                    />
                    <FieldNum
                        label="Uzatılmış tutma max"
                        hint="Uzatılmış senaryo tutma (max)."
                        value={nt(["swing", "type_a", "hold_extended"], 1, 14)}
                        onChange={(v) => st(["swing", "type_a", "hold_extended"], 1, Math.round(v))}
                        step={1}
                        max={30}
                    />
                    </div>
                </div>
            </details>

            <details className="glass-card settings-section-details" style={{ padding: 0, marginTop: 8 }}>
                <summary
                    style={{
                        cursor: "pointer",
                        padding: "12px 16px",
                        display: "flex",
                        alignItems: "center",
                        gap: 10,
                        userSelect: "none",
                        color: "var(--text-muted)",
                        fontSize: "0.8rem",
                    }}
                >
                    <ChevronDown size={16} className="settings-section-chevron" aria-hidden />
                    <span>
                        Şema sürümü: <strong style={{ color: "var(--text-secondary)" }}>{String(draft.schema_version ?? "—")}</strong>
                    </span>
                </summary>
                <div style={{ padding: "10px 16px 14px", borderTop: "1px solid var(--border-muted)", color: "var(--text-muted)", fontSize: "0.78rem", lineHeight: 1.55 }}>
                    Bu değer, ayar dosyasının (JSON) yapı sürümüdür. Normal kullanımda değiştirmeniz gerekmez; geriye dönük uyumluluk ve
                    validasyon için tutulur.
                </div>
            </details>
        </div>
    );
}
