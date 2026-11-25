"use client";
/**
 * GÖLGE SİNYALLER — H4 (içeriden alım), H5 (bilanço sürprizi), H6/H6S/H6SR (VCE), MOM (lider rotasyonu) ileri testi.
 *
 * Gerçek para YOK; ana ürünün paper işlemlerinden ayrı defter. Kurallar ve terfi kapısı
 * sonuç görülmeden yazıldı (lab/PREREGISTRATION.md Ek A + Ek C, H6: Ek F, MOM: Ek U). Terfi otomatik değildir.
 */
import { useEffect, useMemo, useState } from "react";
import { getShadow, type ShadowChampion, type ShadowData, type ShadowHealth, type ShadowStrategyReport, type ShadowTrade } from "@/lib/api";
import { AlertTriangle, CheckCircle2, Clock, Ghost, Info, RefreshCw } from "lucide-react";

const pct = (v: number | null | undefined, d = 2) =>
    v === null || v === undefined ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(d)}%`;
const tone = (v: number | null | undefined) =>
    v === null || v === undefined ? "var(--text-muted)" : v > 0 ? "var(--success)" : v < 0 ? "var(--danger)" : "var(--text-muted)";

const NAMES: Record<string, { title: string; desc: string }> = {
    H5: {
        title: "H5 · Bilanço sürprizi",
        desc: "EPS sürprizi ≥ %10 ve tepki günü yükselen kapanış → ertesi açılış, 2.5×ATR stop, 20 seans. Düzeltilmiş geçmiş testlerde rastgele hisseden iyi değil; portföy 2016–2020 −%18, 2021–26 −%27 (Lab v3 Ek S).",
    },
    H4: {
        title: "H4 · İçeriden alım kümesi",
        desc: "30 günde ≥2 yönetici/direktör, ≥100K$ açık piyasa alımı (SEC Form 4) → ertesi açılış, 2.5×ATR stop, 20 seans. Düzeltilmiş geçmiş testlerde rastgele hisseden anlamlı biçimde iyi değil; portföy 2016–2020 +%16 (endeks +%102), 2021–26 −%12 (Lab v3 Ek S).",
    },
    H6S: {
        title: "H6S · Profesyonel VCE (canlı paper ürün)",
        desc: "H6 sinyalleri + küçük hisse (IWM) piyasa freni (ayı piyasasında giriş yok, temkinli piyasada en çok 1) + 2.5×ATR felaket stop'u → ertesi açılış, 8. seans kapanışı. Geçmiş testlerde para kazandırmadı: hiç kullanılmamış 2016–2020 yıllarında hesap −%56 (endeks +%98); 2021–26'daki eski +%66 sonucu bir veri kusurundan geliyordu, düzeltilince −%30 (Lab v3 Ek R). Gerçek para adayı değil; karar bu ileri testte.",
    },
    H6SR: {
        title: "H6SR · Kazananı koştur (H6S'nin çıkış varyantı)",
        desc: "H6S ile aynı sinyal, aynı piyasa freni ve aynı 2.5×ATR stop. Fark yalnız çıkışta: 8. seans kapanışında kârdaysa satmaz, iz süren stopla (en yüksek kapanış − 2.5×ATR) en çok 40 seans tutar. Düzeltilmiş geçmiş testlerde o da ekside (2016–2020 −%27, 2021–26 −%23; Lab v3 Ek R) — karar bu ileri testte.",
    },
    MOM: {
        title: "MOM · Lider rotasyonu (büyük hisseler, aylık)",
        desc: "Piyasa değeri 10 milyar $ üstü likit hisselerde ay sonunda karar: son 12 ayın (son ay hariç) en güçlü 10 hissesi tutulur; sırası 20'nin altına düşen ya da 200 günlük ortalamasının altına inen satılır; yeni alım yalnız piyasa yükseliş rejimindeyken. Stop yok. Geçmiş sınavda KALDI: 2016–2020 +%301 ama 2021–26 +%22 ve hesap −%71 eridi (Lab v3 Ek T). Kullanıcı kararıyla yalnız ileri testte; gerçek para adayı değil.",
    },
    H6: {
        title: "H6 · Sade VCE (canlı ürünün rakibi)",
        desc: "Canlı evren + sert filtreler + bilanço kapısı + VCE tetiği; RVOL, RSI, Weinstein, skor, rejim, top N, gap, stop yok → ertesi açılış, 8. seans kapanışı.",
    },
};

const GATE_COLOR: Record<string, string> = {
    "TOPLANIYOR": "var(--text-muted)",
    "BELİRSİZ": "var(--warning, #f59e0b)",
    "TERFİ ADAYI": "var(--success)",
    "KALDI": "var(--danger)",
};

const STATUS_LABEL: Record<string, string> = {
    PENDING: "⏳ Girişi bekliyor",
    OPEN: "🟢 Açık",
    CLOSED: "✅ Kapandı",
    SKIPPED: "⏭️ Atlandı",
};

const REASON_LABEL: Record<string, string> = {
    NOT_IN_POOL: "Havuz dışı (fiyat/hacim/cap)",
    NOT_IN_UNIVERSE: "NASDAQ/NYSE hisse senedi değil",
    TICKER_BUSY: "Aynı hissede açık gölge pozisyon",
    NO_ENTRY_BAR: "Giriş günü bar yok",
    ATR_UNAVAILABLE: "ATR hesaplanamadı",
    NO_PRICE_DATA: "Fiyat verisi yok",
    NO_SIGNAL_BAR: "Sinyal günü bar yok",
    STOP: "Stop",
    TIME: "Süre doldu (H4/H5 20, H6/H6S 8 seans; H6SR 8. seansta kârda değildi)",
    TRAIL: "İz süren stop (kazanan geri döndü)",
    TIME_MAX: "En uzun tutuş doldu (40 seans)",
    RANK: "Rotasyon: sırası 20'nin altına düştü",
    TREND: "Rotasyon: 200 günlük ortalamanın altına indi",
    POOL: "Rotasyon: büyük hisse havuzundan çıktı",
    NA: "Rotasyon: skor hesaplanamadı",
    BEAR_REGIME: "Küçük hisse piyasası ayıda (giriş yok)",
    CAUTION_CAPACITY: "Temkinli piyasa: açık pozisyon sınırı dolu",
};

function Check({ ok, label }: { ok: boolean; label: string }) {
    return (
        <div style={{ display: "flex", gap: 6, fontSize: 12, color: ok ? "var(--success)" : "var(--text-muted)" }}>
            <span>{ok ? "✔" : "○"}</span>
            <span>{label}</span>
        </div>
    );
}

const tsi = (iso: string) =>
    new Date(iso).toLocaleString("tr-TR", {
        weekday: "short", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "Europe/Istanbul",
    }) + " TSİ";

/** Koşu sağlığı: kaçırılan seans (kırmızı), eksik koşu (sarı), bekleniyor (gri), tamam (yeşil). */
function HealthBanner({ h }: { h: ShadowHealth }) {
    const overdue = (h.overdue_sessions ?? 0) > 0;
    const fresh = h.last_asof === h.expected_asof;
    const [color, Icon, text]: [string, typeof AlertTriangle, string] = overdue
        ? ["var(--danger)", AlertTriangle,
           `Gölge koşusu ${h.overdue_sessions} seansın kararını kaçırdı (son koşu: ${h.last_asof}). O günlerin sinyalleri ileri teste girmedi — GitHub tetikleyicilerini kontrol edin.`]
        : h.complete
            ? ["var(--success)", CheckCircle2,
               `${h.expected_asof} sinyalleri için koşu tamam${h.on_time ? " · zamanında (açılıştan önce)" : " · GEÇ (açılıştan sonra, değerlendirmeye girmez)"}.`]
            : fresh
                ? ["var(--warning, #f59e0b)", AlertTriangle,
                   `Son koşu eksik: ${h.issues.join(" · ")}. Yedek tetikleyiciler son ana (${tsi(h.next_deadline)}) kadar yeniden deneyecek.`]
                : ["var(--text-muted)", Clock,
                   `${h.expected_asof} sinyalleri için koşu bekleniyor · karar için son an ${tsi(h.next_deadline)}.`];
    return (
        <div className="card" style={{ display: "flex", gap: 10, alignItems: "flex-start", borderLeft: `3px solid ${color}`, padding: "10px 14px" }}>
            <Icon size={17} style={{ color, flexShrink: 0, marginTop: 1 }} />
            <div style={{ fontSize: 13, color: overdue ? color : undefined }}>{text}</div>
        </div>
    );
}

function ChampionCard({ c, h6 }: { c: ShadowChampion; h6?: ShadowStrategyReport }) {
    return (
        <div className="card" style={{ borderLeft: "3px solid var(--accent)" }}>
            <strong>Canlı ürün vs H6 — aynı dönem, aynı maliyet ({c.cost_pct.toFixed(2)}% / işlem)</strong>
            <div style={{ color: "var(--text-muted)", fontSize: 12, margin: "6px 0 12px" }}>
                {c.label}, girişi {c.since} ve sonrası. Hangisi gerçek parayı daha iyi büyütüyor, burada görünür.
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 10 }}>
                <div><div style={{ fontSize: 11, color: "var(--text-muted)" }}>Canlı ürün net ort.</div>
                    <div style={{ fontWeight: 700, color: tone(c.net_mean) }}>{pct(c.net_mean)} <span style={{ fontWeight: 400, color: "var(--text-muted)", fontSize: 12 }}>({c.closed} işlem)</span></div></div>
                <div><div style={{ fontSize: 11, color: "var(--text-muted)" }}>H6 net ort.</div>
                    <div style={{ fontWeight: 700, color: tone(h6?.net_mean) }}>{pct(h6?.net_mean)} <span style={{ fontWeight: 400, color: "var(--text-muted)", fontSize: 12 }}>({h6?.eligible_closed ?? 0} işlem)</span></div></div>
                <div><div style={{ fontSize: 11, color: "var(--text-muted)" }}>Canlı ürün kazanma (net)</div>
                    <div style={{ fontWeight: 600 }}>{c.win_rate === undefined ? "—" : `%${c.win_rate.toFixed(0)}`}</div></div>
                <div><div style={{ fontSize: 11, color: "var(--text-muted)" }}>H6 kazanma (net)</div>
                    <div style={{ fontWeight: 600 }}>{h6?.win_rate === undefined ? "—" : `%${h6.win_rate.toFixed(0)}`}</div></div>
            </div>
            {c.dilution_split && (
                <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 10 }}>
                    Hisse ihracı kontrolü (ileri ölçüm): girişten önceki 30 günde başvurusu olan işlemler{" "}
                    <strong style={{ color: tone(c.dilution_split.dilution.net_mean) }}>{pct(c.dilution_split.dilution.net_mean)}</strong>{" "}
                    ({c.dilution_split.dilution.closed}) · olmayanlar{" "}
                    <strong style={{ color: tone(c.dilution_split.clean.net_mean) }}>{pct(c.dilution_split.clean.net_mean)}</strong>{" "}
                    ({c.dilution_split.clean.closed}). Fark sürerse filtre kapı olur.
                </div>
            )}
        </div>
    );
}

function StrategyCard({ r }: { r: ShadowStrategyReport }) {
    const meta = NAMES[r.strategy] || { title: r.strategy, desc: "" };
    const g = r.gate;
    const color = GATE_COLOR[g.status] || "var(--text-muted)";
    const ci = r.net_ci95 || [null, null];
    const bench = r.benchmark || "IWM";
    const bps = Math.round((r.cost_pct ?? 0.5) * 100);
    return (
        <div className="card" style={{ borderLeft: `3px solid ${color}` }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
                <strong>{meta.title}</strong>
                <span style={{ color, fontWeight: 700, fontSize: 13 }}>{g.status}</span>
            </div>
            <div style={{ color: "var(--text-muted)", fontSize: 12, margin: "6px 0 12px" }}>{meta.desc}</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 10, marginBottom: 12 }}>
                <div><div style={{ fontSize: 11, color: "var(--text-muted)" }}>Net ort. ({bps} bps)</div>
                    <div style={{ fontWeight: 700, color: tone(r.net_mean) }}>{pct(r.net_mean)}</div></div>
                <div><div style={{ fontSize: 11, color: "var(--text-muted)" }}>%95 güven aralığı</div>
                    <div style={{ fontWeight: 600 }}>{ci[0] === null ? "—" : `${pct(ci[0])} … ${pct(ci[1])}`}</div></div>
                <div><div style={{ fontSize: 11, color: "var(--text-muted)" }}>Kazanma oranı</div>
                    <div style={{ fontWeight: 600 }}>{r.win_rate === undefined ? "—" : `%${r.win_rate.toFixed(0)}`}</div></div>
                <div><div style={{ fontSize: 11, color: "var(--text-muted)" }}>{bench}’e göre (işlem başı)</div>
                    <div style={{ fontWeight: 600, color: tone(r.excess_iwm_mean) }}>{pct(r.excess_iwm_mean)}</div></div>
                <div><div style={{ fontSize: 11, color: "var(--text-muted)" }}>Portföy (%10/işlem)</div>
                    <div style={{ fontWeight: 600, color: tone(r.portfolio_10pct?.total_pct) }}>{pct(r.portfolio_10pct?.total_pct, 1)}</div></div>
                <div><div style={{ fontSize: 11, color: "var(--text-muted)" }}>{bench} al-tut</div>
                    <div style={{ fontWeight: 600, color: tone(r.iwm_buy_hold_pct) }}>{pct(r.iwm_buy_hold_pct, 1)}</div></div>
            </div>
            <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>
                Kapanan {r.closed} · değerlendirmeye giren {r.eligible_closed} / {g.need_closed} · {r.months_live.toFixed(1)} / {g.need_months} ay
                {r.late_decisions > 0 && ` · geç karar ${r.late_decisions}`}
                {r.warmup > 0 && ` · ısınma ${r.warmup}`}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 4 }}>
                <Check ok={g.F1_sample} label={`F1 örneklem: ≥${g.need_months} ay ve ≥${g.need_closed} işlem`} />
                <Check ok={g.F2_net_ci_gt0} label="F2 net ort. > 0 ve güven aralığı alt sınırı > 0" />
                <Check ok={g.F3_beats_iwm} label={`F3 portföy ${bench} al-tut'u geçiyor`} />
                <Check ok={g.F4_maxdd_lt35} label="F4 maks. düşüş < %35" />
            </div>
        </div>
    );
}

export default function ShadowPage() {
    const [data, setData] = useState<ShadowData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const [strategy, setStrategy] = useState<"all" | "H5" | "H4" | "H6" | "H6S" | "H6SR" | "MOM">("all");
    const [hideSkipped, setHideSkipped] = useState(true);

    const load = async () => {
        setLoading(true);
        setError("");
        try {
            setData(await getShadow());
        } catch (e: unknown) {
            const ax = e as { response?: { data?: { detail?: string } }; message?: string };
            setError(ax?.response?.data?.detail || ax?.message || "Gölge verisi alınamadı");
        } finally {
            setLoading(false);
        }
    };
    useEffect(() => { void load(); }, []);

    const rows = useMemo(() => {
        let list: ShadowTrade[] = data?.trades || [];
        if (strategy !== "all") list = list.filter(t => t.strategy === strategy);
        if (hideSkipped) list = list.filter(t => t.status !== "SKIPPED");
        return list;
    }, [data, strategy, hideSkipped]);

    if (loading) return <div className="card">Gölge sinyaller yükleniyor…</div>;
    if (error) return <div className="card" style={{ color: "var(--danger)" }}>{error}</div>;
    if (!data) return null;

    const ORDER = ["H6S", "H6SR", "H6", "MOM", "H5", "H4"];
    const strategies = data.report
        ? Object.values(data.report.strategies).sort((a, b) => ORDER.indexOf(a.strategy) - ORDER.indexOf(b.strategy))
        : [];

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
                <div>
                    <h1 style={{ display: "flex", alignItems: "center", gap: 8, margin: 0 }}>
                        <Ghost size={22} /> Gölge Sinyaller
                    </h1>
                    <p style={{ color: "var(--text-muted)", margin: "6px 0 0", fontSize: 13, maxWidth: 820 }}>
                        İki bilgi tabanlı sinyal (H4, H5), canlı ürünün VCE varyantları (H6, H6S, H6SR) ve büyük hisselerde
                        aylık lider rotasyonu (MOM) burada <strong>gerçek para olmadan</strong> ileriye doğru izleniyor.
                        Düzeltilmiş geçmiş testlerde (2016–2020 ve 2021–2026) hiçbiri iki dönemde de endeksi geçemedi
                        (Lab v3 Ek R, S ve T); geçerli kanıt yalnız bu ileri testten gelebilir.
                        Kurallar ve geçme ölçütleri sonuç görülmeden yazıldı; değişmez. Ana ürünün işlemlerine dokunmaz.
                    </p>
                </div>
                <button className="btn-secondary" onClick={() => void load()}>
                    <RefreshCw size={15} style={{ marginRight: 6 }} /> Yenile
                </button>
            </div>

            <div className="card" style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                <Info size={18} style={{ color: "var(--accent)", flexShrink: 0, marginTop: 2 }} />
                <div style={{ fontSize: 13 }}>
                    {data.report
                        ? <>Canlı başlangıç <strong>{data.report.live_start}</strong> · son veri {data.report.asof}
                            {data.h4_observation_start && <> · H4 gözlem başlangıcı {data.h4_observation_start} (ilk 30 gün ısınma)</>}
                            {(() => {
                                const h6 = (data.last_run?.H6 ?? null) as Record<string, unknown> | null;
                                if (!h6) return null;
                                const src = String(h6.universe_source ?? "?");
                                const bad = src === "static_fallback" || "error" in h6;
                                return (
                                    <div style={{ fontSize: 12, marginTop: 4, color: bad ? "var(--danger)" : "var(--text-muted)" }}>
                                        H6 son koşu: evren {String(h6.universe ?? "—")} hisse (kaynak: {src}) · güncel bar {String(h6.current_bar ?? "—")} ·
                                        VCE {String(h6.vce ?? "—")} · eklenen {String(h6.added ?? 0)}
                                        {"error" in h6 && ` · hata: ${String(h6.error)}`}
                                    </div>
                                );
                            })()}
                            {(() => {
                                const m = (data.last_run?.MOM ?? null) as Record<string, unknown> | null;
                                if (!m) return null;
                                const bad = "error" in m || "deferred" in m;
                                return (
                                    <div style={{ fontSize: 12, marginTop: 4, color: bad ? "var(--danger)" : "var(--text-muted)" }}>
                                        {m.rebalance
                                            ? <>MOM ay sonu kararı {String(m.rebalance)}: rejim {String(m.regime ?? "—")} · evren {String(m.universe ?? "—")} hisse ·
                                                uygun {String(m.eligible ?? "—")} · alım {String(m.buys ?? 0)} · satış {String(m.sells ?? 0)}</>
                                            : <>MOM: bu koşuda karar günü yok (karar yalnız ayın son seansında){m.last_rebalance ? ` · son karar ${String(m.last_rebalance)}` : ""}</>}
                                        {"deferred" in m && ` · ertelendi: ${String(m.deferred)}`}
                                        {"error" in m && ` · hata: ${String(m.error)}`}
                                    </div>
                                );
                            })()}
                            <div style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 4 }}>
                                Terfi otomatik değildir. “TERFİ ADAYI” yalnız gerçek para tartışmasını açar.
                                “KALDI” durumunda gölge kapatılır. Geç verilen kararlar (açılıştan sonra) ve ısınma dönemi değerlendirmeye girmez.
                            </div></>
                        : "Gölge koşusu henüz çalışmadı. İlk koşudan sonra burada veri görünecek."}
                </div>
            </div>

            {data.health && <HealthBanner h={data.health} />}

            {strategies.length > 0 && (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: 16 }}>
                    {strategies.map(r => <StrategyCard key={r.strategy} r={r} />)}
                </div>
            )}

            {data.report?.champion && <ChampionCard c={data.report.champion} h6={data.report.strategies["H6"]} />}

            <div className="card">
                <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
                    <strong>İşlemler</strong>
                    {(["all", "H6S", "H6SR", "H6", "MOM", "H5", "H4"] as const).map(s => (
                        <button key={s} className={strategy === s ? "btn-primary" : "btn-secondary"}
                            style={{ padding: "4px 10px", fontSize: 12 }} onClick={() => setStrategy(s)}>
                            {s === "all" ? "Tümü" : s}
                        </button>
                    ))}
                    <label style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 12, cursor: "pointer" }}>
                        <input type="checkbox" checked={hideSkipped} onChange={e => setHideSkipped(e.target.checked)} />
                        Atlananları gizle
                    </label>
                    <span style={{ color: "var(--text-muted)", fontSize: 12 }}>{rows.length} / {data.total_trades}</span>
                </div>
                <div style={{ overflowX: "auto" }}>
                    <table className="data-table" style={{ width: "100%", fontSize: 13 }}>
                        <thead>
                            <tr>
                                <th>Strateji</th><th>Hisse</th><th>Sinyal</th><th>Durum</th><th>Giriş</th>
                                <th>Stop</th><th>Çıkış / değerleme</th><th>Getiri</th><th>Net</th><th>Endeks</th><th>Not</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map(t => (
                                <tr key={t.id}>
                                    <td>{t.strategy}</td>
                                    <td style={{ fontWeight: 600 }}>{t.ticker}</td>
                                    <td>{t.signal_date}</td>
                                    <td>{STATUS_LABEL[t.status] || t.status}</td>
                                    <td>{t.entry_date ? `${t.entry_date} · $${(t.entry_price ?? 0).toFixed(2)}` : "—"}</td>
                                    <td>{t.stop ? `$${t.stop.toFixed(2)}` : "—"}</td>
                                    <td>{t.exit_date
                                        ? `${t.exit_date} · $${(t.exit_price ?? 0).toFixed(2)}`
                                        : t.mark_date ? `${t.mark_date} · $${(t.mark_price ?? 0).toFixed(2)}` : "—"}</td>
                                    <td style={{ color: tone(t.ret_pct) }}>{pct(t.ret_pct)}</td>
                                    <td style={{ color: tone(t.net_pct) }}>{pct(t.net_pct)}</td>
                                    <td style={{ color: tone(t.iwm_ret) }}>{pct(t.iwm_ret)}</td>
                                    <td style={{ color: "var(--text-muted)", fontSize: 12 }}>
                                        {[t.reason && (REASON_LABEL[t.reason] || t.reason),
                                          t.exit_reason && (REASON_LABEL[t.exit_reason] || t.exit_reason),
                                          t.late_decision ? "geç karar" : null,
                                          t.warmup ? "ısınma" : null,
                                          t.strategy === "H5" && t.key != null ? `sürpriz %${t.key.toFixed(0)}` : null,
                                          t.strategy === "H4" && t.key != null ? `$${Math.round(t.key / 1000)}K alım` : null,
                                          t.strategy === "MOM" && t.key != null ? `12 ay skoru %${(t.key * 100).toFixed(0)}` : null,
                                        ].filter(Boolean).join(" · ") || "—"}
                                    </td>
                                </tr>
                            ))}
                            {rows.length === 0 && (
                                <tr><td colSpan={11} style={{ color: "var(--text-muted)", textAlign: "center", padding: 18 }}>
                                    Henüz gölge işlem yok.
                                </td></tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
