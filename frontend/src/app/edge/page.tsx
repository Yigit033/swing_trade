"use client";
/**
 * SİNYAL KARNESİ — sistemin kendi kendini denetlediği sayfa.
 *
 * İki kohort:
 *  1) Sinyal — motorun ürettiği ham sinyaller (gösterim eşiğinin altı dahil)
 *  2) Evren — tarandı ama tetik/kapı yüzünden sinyal olmadı (Q çoğu zaman yok)
 *
 * Getiri kuralı her iki grupta aynı: t+1 açılış → R3/R5/R10.
 */
import { useEffect, useMemo, useState } from "react";
import {
    getEdgeTracking,
    REJECT_REASON_LABELS,
    type EdgeTracking,
    type EdgeSignal,
    type EdgeSide,
} from "@/lib/api";
import { RefreshCw, Target, TrendingUp, TrendingDown, Info } from "lucide-react";

const pct = (v: number | null | undefined, digits = 2) =>
    v === null || v === undefined ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(digits)}%`;

const tone = (v: number | null | undefined) =>
    v === null || v === undefined ? "var(--text-muted)" : v > 0 ? "var(--success)" : v < 0 ? "var(--danger)" : "var(--text-muted)";

type KindFilter = "all" | "signal" | "universe";
type SortKey = "date" | "r10" | "ticker";

function reasonLabel(code?: string | null) {
    if (!code) return "—";
    return REJECT_REASON_LABELS[code] || code;
}

export default function EdgePage() {
    const [data, setData] = useState<EdgeTracking | null>(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState("");
    const [onlyMature, setOnlyMature] = useState(false);
    const [kind, setKind] = useState<KindFilter>("all");
    const [sortKey, setSortKey] = useState<SortKey>("date");

    const load = async (refresh = false) => {
        if (refresh) setRefreshing(true); else setLoading(true);
        setError("");
        try {
            setData(await getEdgeTracking(refresh));
        } catch (e: unknown) {
            const ax = e as { response?: { data?: { detail?: string } } };
            setError(ax?.response?.data?.detail || "Veri alınamadı");
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    };

    useEffect(() => { void load(false); }, []);

    const rows = useMemo(() => {
        if (!data) return [];
        const signalRows = (data.signals || []).map(s => ({ ...s, kind: s.kind || "signal" as const }));
        const universeRows = (data.universe_rows || []).map(s => ({ ...s, kind: "universe" as const }));
        let list: EdgeSignal[] =
            kind === "signal" ? signalRows : kind === "universe" ? universeRows : [...signalRows, ...universeRows];
        if (onlyMature) list = list.filter(s => s.r10 !== null);
        const copy = [...list];
        copy.sort((a, b) => {
            if (sortKey === "ticker") return (a.ticker || "").localeCompare(b.ticker || "");
            if (sortKey === "r10") {
                const ar = a.r10, br = b.r10;
                if (ar == null && br == null) return 0;
                if (ar == null) return 1;
                if (br == null) return -1;
                return br - ar;
            }
            return String(b.signal_date || "").localeCompare(String(a.signal_date || ""));
        });
        return copy;
    }, [data, kind, onlyMature, sortKey]);

    if (loading) return <div className="card">Sinyal karnesi yükleniyor…</div>;
    if (error) return <div className="card" style={{ color: "var(--danger)" }}>{error}</div>;
    if (!data) return null;

    const { above, below, reference } = data.threshold_split;
    const uni = data.universe;
    const cohort = data.cohort_split;
    const verdict =
        above.n < 3 || below.n < 3
            ? { text: "Henüz karar için yetersiz veri — her iki kalite grubunda da en az 3 olgun sinyal gerekir.", color: "var(--text-muted)" }
            : (above.mean ?? 0) > (below.mean ?? 0)
                ? { text: `Eşik İŞE YARIYOR: Q${reference}+ grubu ${pct(above.mean)} vs altı ${pct(below.mean)}.`, color: "var(--success)" }
                : { text: `⚠️ Eşik SORGULANMALI: Q${reference}+ grubu (${pct(above.mean)}) eşik-altından (${pct(below.mean)}) daha iyi DEĞİL.`, color: "var(--danger)" };

    const gateVerdict = gateCopy(cohort?.signal, cohort?.universe);

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
                <div>
                    <h1 style={{ display: "flex", alignItems: "center", gap: 8, margin: 0 }}>
                        <Target size={22} /> Sinyal Karnesi
                    </h1>
                    <p style={{ color: "var(--text-muted)", margin: "6px 0 0", fontSize: 13, maxWidth: 760 }}>
                        Motorun ürettiği <strong>sinyaller</strong> (eşik-altı dahil) ve tarandığı halde tetik olmayan{" "}
                        <strong>evren</strong> hisseleri aynı t+1 kuralıyla izlenir. Birkaç gün sonra “bu Q55’ti, +%17 oldu”
                        demek için Yahoo’yu elle açmana gerek yok — tabloyu R10’a göre sırala.
                    </p>
                </div>
                <button className="btn-secondary" onClick={() => void load(true)} disabled={refreshing}
                    title="Olgunlaşan getirileri doldur (yfinance, biraz sürebilir)">
                    <RefreshCw size={15} style={{ marginRight: 6, animation: refreshing ? "spin 1s linear infinite" : undefined }} />
                    {refreshing ? "Güncelleniyor…" : "Getirileri güncelle"}
                </button>
            </div>

            <div className="card" style={{ borderLeft: `3px solid ${verdict.color}` }}>
                <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                    <Info size={18} style={{ color: verdict.color, flexShrink: 0, marginTop: 2 }} />
                    <div>
                        <div style={{ fontWeight: 600, color: verdict.color }}>{verdict.text}</div>
                        <div style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 4 }}>
                            {data.n_tracked} sinyal izleniyor · {data.n_complete} olgun
                            {uni ? ` · ${uni.n_tracked} evren ismi · ${uni.n_complete} olgun` : ""}
                        </div>
                    </div>
                </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
                {[
                    { title: `Sinyal Q${reference}+`, s: above, icon: TrendingUp, hint: "Eşiği geçen — işleme alınabilir" },
                    { title: `Sinyal Q<${reference}`, s: below, icon: TrendingDown, hint: "Üretildi, gösterilmedi — ölçüm" },
                    {
                        title: "Evren (tetik yok)",
                        s: uni?.r10,
                        icon: Target,
                        hint: "Tarandı, kapı yanmadı — kontrol grubu",
                    },
                ].map(({ title, s, icon: Icon, hint }) => (
                    <div className="card" key={title}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                            <Icon size={17} style={{ color: tone(s?.mean) }} />
                            <strong>{title}</strong>
                        </div>
                        <div style={{ color: "var(--text-muted)", fontSize: 12, marginBottom: 12 }}>{hint}</div>
                        {!s || s.n === 0 ? (
                            <div style={{ color: "var(--text-muted)" }}>Henüz olgun R10 yok</div>
                        ) : (
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 12 }}>
                                <Stat label="Ortalama R10" value={pct(s.mean)} color={tone(s.mean)} big />
                                <Stat label="Kazanma" value={s.win_rate !== null ? `%${s.win_rate.toFixed(0)}` : "—"} big />
                                <Stat label="Olgun" value={String(s.n)} />
                                <Stat label="En iyi / en kötü" value={`${pct(s.best, 1)} / ${pct(s.worst, 1)}`} />
                            </div>
                        )}
                    </div>
                ))}
            </div>

            {gateVerdict && (
                <div className="card" style={{ borderLeft: `3px solid ${gateVerdict.color}` }}>
                    <strong>Kapılar işe yarıyor mu?</strong>
                    <div style={{ marginTop: 6, color: gateVerdict.color, fontWeight: 600 }}>{gateVerdict.text}</div>
                    <div style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 4 }}>
                        Sinyal kohortunun R10’u, tetik olmayan evrenden belirgin daha iyiyse kapılar kenar ayıklıyor demektir.
                    </div>
                </div>
            )}

            <div className="card">
                <strong>Kalite kovaları — yalnız sinyaller (skor gerçekten ayırt ediyor mu?)</strong>
                <div style={{ color: "var(--text-muted)", fontSize: 12, margin: "6px 0 14px" }}>
                    Evren hisselerinin çoğunda Q hesaplanmaz (tetikten önce elenir). Kovalar bu yüzden sinyal kohortuna aittir.
                </div>
                <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", minWidth: 520, borderCollapse: "collapse", fontSize: 13 }}>
                        <thead>
                            <tr style={{ textAlign: "left", color: "var(--text-muted)" }}>
                                <th style={{ padding: "6px 8px" }}>Kova</th>
                                <th style={{ padding: "6px 8px", textAlign: "right" }}>Olgun</th>
                                <th style={{ padding: "6px 8px", textAlign: "right" }}>Bekleyen</th>
                                <th style={{ padding: "6px 8px", textAlign: "right" }}>Ort. R10</th>
                                <th style={{ padding: "6px 8px", textAlign: "right" }}>Kazanma</th>
                                <th style={{ padding: "6px 8px", width: "34%" }}></th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.quality_buckets.map(b => {
                                const w = Math.min(Math.abs(b.mean ?? 0) * 6, 100);
                                return (
                                    <tr key={b.label} style={{ borderTop: "1px solid var(--border)" }}>
                                        <td style={{ padding: "8px", fontWeight: 600 }}>{b.label}</td>
                                        <td style={{ padding: "8px", textAlign: "right" }}>{b.n || "—"}</td>
                                        <td style={{ padding: "8px", textAlign: "right", color: "var(--text-muted)" }}>{b.pending || "—"}</td>
                                        <td style={{ padding: "8px", textAlign: "right", color: tone(b.mean), fontWeight: 600 }}>{pct(b.mean)}</td>
                                        <td style={{ padding: "8px", textAlign: "right" }}>{b.win_rate !== null ? `%${b.win_rate.toFixed(0)}` : "—"}</td>
                                        <td style={{ padding: "8px" }}>
                                            {b.mean !== null && (
                                                <div style={{ height: 8, background: "var(--bg-subtle)", borderRadius: 4, overflow: "hidden" }}>
                                                    <div style={{ width: `${w}%`, height: "100%", background: tone(b.mean), borderRadius: 4 }} />
                                                </div>
                                            )}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>

            {uni && uni.reject_reasons?.length > 0 && (
                <div className="card">
                    <strong>Evren — neden elendi?</strong>
                    <div style={{ color: "var(--text-muted)", fontSize: 12, margin: "6px 0 12px" }}>
                        Aynı reddetme nedeninin R10’u yüksekse o kapı fırsat kaçırıyor olabilir.
                    </div>
                    <div style={{ overflowX: "auto" }}>
                        <table style={{ width: "100%", minWidth: 480, borderCollapse: "collapse", fontSize: 13 }}>
                            <thead>
                                <tr style={{ textAlign: "left", color: "var(--text-muted)" }}>
                                    <th style={{ padding: "6px 8px" }}>Neden</th>
                                    <th style={{ padding: "6px 8px", textAlign: "right" }}>Adet</th>
                                    <th style={{ padding: "6px 8px", textAlign: "right" }}>Olgun</th>
                                    <th style={{ padding: "6px 8px", textAlign: "right" }}>Ort. R10</th>
                                    <th style={{ padding: "6px 8px", textAlign: "right" }}>Kazanma</th>
                                </tr>
                            </thead>
                            <tbody>
                                {uni.reject_reasons.map(r => (
                                    <tr key={r.reason} style={{ borderTop: "1px solid var(--border)" }}>
                                        <td style={{ padding: "8px" }}>{reasonLabel(r.reason)}</td>
                                        <td style={{ padding: "8px", textAlign: "right" }}>{r.n}</td>
                                        <td style={{ padding: "8px", textAlign: "right", color: "var(--text-muted)" }}>{r.n_mature || "—"}</td>
                                        <td style={{ padding: "8px", textAlign: "right", color: tone(r.mean), fontWeight: 600 }}>{pct(r.mean)}</td>
                                        <td style={{ padding: "8px", textAlign: "right" }}>{r.win_rate !== null ? `%${r.win_rate.toFixed(0)}` : "—"}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            <div className="card">
                <strong>Tutma süresine göre (sinyaller)</strong>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 14, marginTop: 12 }}>
                    {(["r3", "r5", "r10"] as const).map(k => {
                        const a = data.aggregates[k];
                        return (
                            <div key={k}>
                                <div style={{ color: "var(--text-muted)", fontSize: 12 }}>
                                    {k.toUpperCase()} ({k.slice(1)} iş günü)
                                </div>
                                <div style={{ fontSize: 20, fontWeight: 700, color: tone(a?.mean) }}>{pct(a?.mean)}</div>
                                <div style={{ color: "var(--text-muted)", fontSize: 12 }}>
                                    {a ? `n=${a.n} · kazanma %${a.win_rate.toFixed(0)}` : "veri yok"}
                                </div>
                            </div>
                        );
                    })}
                    <div>
                        <div style={{ color: "var(--text-muted)", fontSize: 12 }}>MFE / MAE (10g)</div>
                        <div style={{ fontSize: 20, fontWeight: 700 }}>
                            <span style={{ color: "var(--success)" }}>{pct(data.mfe10?.mean, 1)}</span>
                            {" / "}
                            <span style={{ color: "var(--danger)" }}>{pct(data.mae10?.mean, 1)}</span>
                        </div>
                        <div style={{ color: "var(--text-muted)", fontSize: 12 }}>en yüksek kâr / en derin zarar</div>
                    </div>
                </div>
            </div>

            <div className="card">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
                    <strong>Kayıtlar ({rows.length})</strong>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
                        {(["all", "signal", "universe"] as const).map(k => (
                            <button
                                key={k}
                                type="button"
                                className={kind === k ? "btn-primary" : "btn-secondary"}
                                style={{ fontSize: 12, padding: "4px 10px" }}
                                onClick={() => setKind(k)}
                            >
                                {k === "all" ? "Hepsi" : k === "signal" ? "Sinyaller" : "Evren"}
                            </button>
                        ))}
                        <select
                            value={sortKey}
                            onChange={e => setSortKey(e.target.value as SortKey)}
                            style={{ fontSize: 12, padding: "4px 8px", background: "var(--bg-subtle)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 6 }}
                        >
                            <option value="date">Sırala: tarih</option>
                            <option value="r10">Sırala: R10 (yüksek → düşük)</option>
                            <option value="ticker">Sırala: hisse</option>
                        </select>
                        <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, color: "var(--text-muted)", cursor: "pointer" }}>
                            <input type="checkbox" checked={onlyMature} onChange={e => setOnlyMature(e.target.checked)} />
                            Sadece olgun
                        </label>
                    </div>
                </div>
                <div style={{ overflowX: "auto", marginTop: 12 }}>
                    <table style={{ width: "100%", minWidth: 860, borderCollapse: "collapse", fontSize: 13 }}>
                        <thead>
                            <tr style={{ textAlign: "left", color: "var(--text-muted)" }}>
                                {["Tarih", "Hisse", "Kohort", "Q", "Neden / yol", "R3", "R5", "R10", "MFE", "MAE", "Durum"].map(h => (
                                    <th key={h} style={{ padding: "6px 8px", textAlign: ["R3", "R5", "R10", "MFE", "MAE", "Q"].includes(h) ? "right" : "left" }}>{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map((s: EdgeSignal, i) => {
                                const isSignal = (s.kind || "signal") !== "universe";
                                const taken = isSignal && (s.quality ?? 0) >= reference;
                                return (
                                    <tr key={`${s.kind}-${s.ticker}-${s.signal_date}-${i}`} style={{ borderTop: "1px solid var(--border)" }}>
                                        <td style={{ padding: "8px", color: "var(--text-muted)" }}>{s.signal_date?.slice(0, 10)}</td>
                                        <td style={{ padding: "8px", fontWeight: 600 }}>{s.ticker}</td>
                                        <td style={{ padding: "8px" }}>
                                            <span style={{
                                                padding: "1px 7px", borderRadius: 10, fontSize: 11, fontWeight: 600,
                                                background: isSignal ? "rgba(59,130,246,.15)" : "var(--bg-subtle)",
                                                color: isSignal ? "var(--accent, #60a5fa)" : "var(--text-muted)",
                                            }}>
                                                {isSignal ? "sinyal" : "evren"}
                                            </span>
                                        </td>
                                        <td style={{ padding: "8px", textAlign: "right" }}>
                                            {s.quality != null ? (
                                                <span title={taken ? "Eşiği geçti" : "Eşik altı / skorlu evren"}
                                                    style={{
                                                        padding: "1px 7px", borderRadius: 10, fontSize: 12, fontWeight: 600,
                                                        background: taken ? "var(--success-bg, rgba(34,197,94,.15))" : "var(--bg-subtle)",
                                                        color: taken ? "var(--success)" : "var(--text-muted)",
                                                    }}>
                                                    {s.quality.toFixed(0)}
                                                </span>
                                            ) : <span style={{ color: "var(--text-muted)" }}>—</span>}
                                        </td>
                                        <td style={{ padding: "8px", color: "var(--text-muted)" }}>
                                            {isSignal
                                                ? (s.pathway === "vce_breakout" ? "VCE" : s.pathway === "rvol_thrust" ? "RVOL" : (s.pathway ?? "—"))
                                                : reasonLabel(s.reject_reason)}
                                        </td>
                                        {([s.r3, s.r5, s.r10, s.mfe10, s.mae10]).map((v, j) => (
                                            <td key={j} style={{ padding: "8px", textAlign: "right", color: tone(v), fontWeight: j === 2 ? 700 : 400 }}>{pct(v, 1)}</td>
                                        ))}
                                        <td style={{ padding: "8px", color: "var(--text-muted)" }}>
                                            {s.status === "complete" ? "tamam" : s.status === "partial" ? "kısmi" : "bekliyor"}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                    {rows.length === 0 && (
                        <div style={{ padding: 20, textAlign: "center", color: "var(--text-muted)" }}>
                            {kind === "universe"
                                ? "Henüz evren kaydı yok. Bir tarama koşunca tetik olmayan isimler burada birikir."
                                : "Henüz izlenen sinyal yok. Bir tarama koşunca burada görünmeye başlar."}
                        </div>
                    )}
                </div>
            </div>

            <div className="card" style={{ fontSize: 12, color: "var(--text-muted)" }}>
                <strong style={{ color: "var(--text)" }}>Nasıl okunur:</strong>{" "}
                R3/R5/R10 = bar tarihinin ertesi seans <em>açılışından</em> 3/5/10 iş günü sonraki getiri.
                Bu işlem P&amp;L&apos;i değildir (stop/hedef yok). Q yalnızca motor skorladığında doludur.
                Evren satırında “—” normaldir. Harness beklentisi (sinyaller): {data.harness_expectation?.r10_mean}.
            </div>
        </div>
    );
}

function gateCopy(signal?: EdgeSide, universe?: EdgeSide): { text: string; color: string } | null {
    if (!signal || !universe) return null;
    if (signal.n < 3 || universe.n < 3) {
        return {
            text: "Karşılaştırma için her iki kohortta da en az 3 olgun R10 gerekir. Yeni taramalar biriktikçe dolar.",
            color: "var(--text-muted)",
        };
    }
    const s = signal.mean ?? 0;
    const u = universe.mean ?? 0;
    if (s > u + 0.5) {
        return {
            text: `Kapılar kenar ayıklıyor: sinyal R10 ${pct(signal.mean)} vs evren ${pct(universe.mean)}.`,
            color: "var(--success)",
        };
    }
    if (u > s + 0.5) {
        return {
            text: `⚠️ Evren, sinyal grubundan daha iyi (R10 ${pct(universe.mean)} vs ${pct(signal.mean)}). Kapılar fırsat kesiyor olabilir.`,
            color: "var(--danger)",
        };
    }
    return {
        text: `Sinyal ve evren R10 yakın (${pct(signal.mean)} vs ${pct(universe.mean)}). Henüz net bir kapı kenarı yok.`,
        color: "var(--text-muted)",
    };
}

function Stat({ label, value, color, big }: { label: string; value: string; color?: string; big?: boolean }) {
    return (
        <div>
            <div style={{ color: "var(--text-muted)", fontSize: 12 }}>{label}</div>
            <div style={{ fontSize: big ? 22 : 15, fontWeight: 700, color: color ?? "var(--text)" }}>{value}</div>
        </div>
    );
}
