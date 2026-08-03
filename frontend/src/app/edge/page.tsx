"use client";
/**
 * SİNYAL KARNESİ — sistemin kendi kendini denetlediği sayfa.
 *
 * Motorun ürettiği HER ham sinyal (kalite eşiğini geçmeyenler DAHİL) burada
 * izlenir ve 3/5/10 iş günü sonraki gerçek getirisi otomatik doldurulur.
 *
 * Neden eşik-altı sinyaller de kaydediliyor: eşiğin doğru yerde olup olmadığını
 * ancak "almadıklarımızın" sonucuyla ölçebiliriz. Eşik-altı grup kazanıyorsa
 * eşik yanlış; kaybediyorsa eşik bizi koruyor. Bu sayfa o soruyu veriyle
 * cevaplar — sezgiyle değil.
 */
import { useEffect, useState } from "react";
import { getEdgeTracking, type EdgeTracking, type EdgeSignal } from "@/lib/api";
import { RefreshCw, Target, TrendingUp, TrendingDown, Info } from "lucide-react";

const pct = (v: number | null | undefined, digits = 2) =>
    v === null || v === undefined ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(digits)}%`;

const tone = (v: number | null | undefined) =>
    v === null || v === undefined ? "var(--text-muted)" : v > 0 ? "var(--success)" : v < 0 ? "var(--danger)" : "var(--text-muted)";

export default function EdgePage() {
    const [data, setData] = useState<EdgeTracking | null>(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState("");
    const [onlyMature, setOnlyMature] = useState(false);

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

    if (loading) return <div className="card">Sinyal karnesi yükleniyor…</div>;
    if (error) return <div className="card" style={{ color: "var(--danger)" }}>{error}</div>;
    if (!data) return null;

    const { above, below, reference } = data.threshold_split;
    const verdict =
        above.n < 3 || below.n < 3
            ? { text: "Henüz karar için yetersiz veri — her iki grupta da en az 3 olgun sinyal gerekir.", color: "var(--text-muted)" }
            : (above.mean ?? 0) > (below.mean ?? 0)
                ? { text: `Eşik İŞE YARIYOR: Q${reference}+ grubu ${pct(above.mean)} vs altı ${pct(below.mean)}.`, color: "var(--success)" }
                : { text: `⚠️ Eşik SORGULANMALI: Q${reference}+ grubu (${pct(above.mean)}) eşik-altından (${pct(below.mean)}) daha iyi DEĞİL.`, color: "var(--danger)" };

    const signals = onlyMature ? data.signals.filter(s => s.r10 !== null) : data.signals;

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
                <div>
                    <h1 style={{ display: "flex", alignItems: "center", gap: 8, margin: 0 }}>
                        <Target size={22} /> Sinyal Karnesi
                    </h1>
                    <p style={{ color: "var(--text-muted)", margin: "6px 0 0", fontSize: 13, maxWidth: 720 }}>
                        Motorun ürettiği <strong>tüm</strong> sinyaller — kalite eşiğini geçmeyenler dahil — burada izlenir.
                        Eşik-altı sinyalleri işleme <em>almıyoruz</em> ama sonuçlarını <em>ölçüyoruz</em>: eşiğin doğru yerde
                        olduğunu ancak böyle bilebiliriz.
                    </p>
                </div>
                <button className="btn-secondary" onClick={() => void load(true)} disabled={refreshing}
                    title="Olgunlaşan getirileri doldur (yfinance'tan çeker, biraz sürebilir)">
                    <RefreshCw size={15} style={{ marginRight: 6, animation: refreshing ? "spin 1s linear infinite" : undefined }} />
                    {refreshing ? "Güncelleniyor…" : "Getirileri güncelle"}
                </button>
            </div>

            {/* ── Ana karar ── */}
            <div className="card" style={{ borderLeft: `3px solid ${verdict.color}` }}>
                <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                    <Info size={18} style={{ color: verdict.color, flexShrink: 0, marginTop: 2 }} />
                    <div>
                        <div style={{ fontWeight: 600, color: verdict.color }}>{verdict.text}</div>
                        <div style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 4 }}>
                            {data.n_tracked} sinyal izleniyor · {data.n_complete} tanesi olgunlaştı (10 iş günü doldu)
                        </div>
                    </div>
                </div>
            </div>

            {/* ── Eşik-üstü vs eşik-altı ── */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16 }}>
                {[
                    { title: `Aldıklarımız (Q${reference}+)`, s: above, icon: TrendingUp, hint: "Eşiği geçen — işleme alınanlar" },
                    { title: `Almadıklarımız (Q<${reference})`, s: below, icon: TrendingDown, hint: "Eşiğin altında — sadece kaydedildi" },
                ].map(({ title, s, icon: Icon, hint }) => (
                    <div className="card" key={title}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                            <Icon size={17} style={{ color: tone(s.mean) }} />
                            <strong>{title}</strong>
                        </div>
                        <div style={{ color: "var(--text-muted)", fontSize: 12, marginBottom: 12 }}>{hint}</div>
                        {s.n === 0 ? (
                            <div style={{ color: "var(--text-muted)" }}>Henüz olgunlaşmış sinyal yok</div>
                        ) : (
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 12 }}>
                                <Stat label="Ortalama R10" value={pct(s.mean)} color={tone(s.mean)} big />
                                <Stat label="Kazanma oranı" value={s.win_rate !== null ? `%${s.win_rate.toFixed(0)}` : "—"} big />
                                <Stat label="Sinyal" value={String(s.n)} />
                                <Stat label="En iyi / en kötü" value={`${pct(s.best, 1)} / ${pct(s.worst, 1)}`} />
                            </div>
                        )}
                    </div>
                ))}
            </div>

            {/* ── Kalite kovaları ── */}
            <div className="card">
                <strong>Kalite kovaları — skor gerçekten ayırt ediyor mu?</strong>
                <div style={{ color: "var(--text-muted)", fontSize: 12, margin: "6px 0 14px" }}>
                    Skor işe yarıyorsa yukarı doğru gidildikçe getiri artmalı. Artmıyorsa skor bilgi taşımıyor demektir.
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

            {/* ── Zaman ufukları ── */}
            <div className="card">
                <strong>Tutma süresine göre</strong>
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

            {/* ── Sinyal listesi ── */}
            <div className="card">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
                    <strong>Sinyaller ({signals.length})</strong>
                    <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, color: "var(--text-muted)", cursor: "pointer" }}>
                        <input type="checkbox" checked={onlyMature} onChange={e => setOnlyMature(e.target.checked)} />
                        Sadece olgunlaşanlar
                    </label>
                </div>
                <div style={{ overflowX: "auto", marginTop: 12 }}>
                    <table style={{ width: "100%", minWidth: 780, borderCollapse: "collapse", fontSize: 13 }}>
                        <thead>
                            <tr style={{ textAlign: "left", color: "var(--text-muted)" }}>
                                {["Tarih", "Hisse", "Q", "Yol", "Rejim", "R3", "R5", "R10", "MFE", "MAE", "Durum"].map(h => (
                                    <th key={h} style={{ padding: "6px 8px", textAlign: ["Tarih", "Hisse", "Yol", "Rejim", "Durum"].includes(h) ? "left" : "right" }}>{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {signals.map((s: EdgeSignal, i) => {
                                const taken = (s.quality ?? 0) >= reference;
                                return (
                                    <tr key={`${s.ticker}-${s.signal_date}-${i}`} style={{ borderTop: "1px solid var(--border)" }}>
                                        <td style={{ padding: "8px", color: "var(--text-muted)" }}>{s.signal_date?.slice(0, 10)}</td>
                                        <td style={{ padding: "8px", fontWeight: 600 }}>{s.ticker}</td>
                                        <td style={{ padding: "8px", textAlign: "right" }}>
                                            <span title={taken ? "Eşiği geçti — işleme alınabilir" : "Eşik altı — sadece ölçüm için kaydedildi"}
                                                style={{
                                                    padding: "1px 7px", borderRadius: 10, fontSize: 12, fontWeight: 600,
                                                    background: taken ? "var(--success-bg, rgba(34,197,94,.15))" : "var(--bg-subtle)",
                                                    color: taken ? "var(--success)" : "var(--text-muted)",
                                                }}>
                                                {s.quality?.toFixed(0) ?? "—"}
                                            </span>
                                        </td>
                                        <td style={{ padding: "8px", color: "var(--text-muted)" }}>
                                            {s.pathway === "vce_breakout" ? "🎯 VCE" : s.pathway === "rvol_thrust" ? "⚡ RVOL" : (s.pathway ?? "—")}
                                        </td>
                                        <td style={{ padding: "8px", color: "var(--text-muted)" }}>{s.regime ?? "—"}</td>
                                        {([s.r3, s.r5, s.r10, s.mfe10, s.mae10]).map((v, j) => (
                                            <td key={j} style={{ padding: "8px", textAlign: "right", color: tone(v) }}>{pct(v, 1)}</td>
                                        ))}
                                        <td style={{ padding: "8px", color: "var(--text-muted)" }}>
                                            {s.status === "complete" ? "tamam" : s.status === "partial" ? "kısmi" : "bekliyor"}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                    {signals.length === 0 && (
                        <div style={{ padding: 20, textAlign: "center", color: "var(--text-muted)" }}>
                            Henüz izlenen sinyal yok. Bir tarama koşunca burada görünmeye başlar.
                        </div>
                    )}
                </div>
            </div>

            <div className="card" style={{ fontSize: 12, color: "var(--text-muted)" }}>
                <strong style={{ color: "var(--text)" }}>Nasıl okunur:</strong>{" "}
                R3/R5/R10 = sinyalin ertesi günkü <em>açılıştan</em> girildiğinde 3/5/10 iş günü sonraki getiri.
                Bu, işlem P&amp;L&apos;i <em>değildir</em> — stop/hedef/trailing uygulanmadan ham sinyal kalitesini ölçer
                (gerçek strateji stop&apos;la kayıpları keser, dolayısıyla genelde bundan iyidir).
                Harness beklentisi: {data.harness_expectation?.r10_mean}.
            </div>
        </div>
    );
}

function Stat({ label, value, color, big }: { label: string; value: string; color?: string; big?: boolean }) {
    return (
        <div>
            <div style={{ color: "var(--text-muted)", fontSize: 12 }}>{label}</div>
            <div style={{ fontSize: big ? 22 : 15, fontWeight: 700, color: color ?? "var(--text)" }}>{value}</div>
        </div>
    );
}
