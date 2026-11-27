"use client";
/**
 * MİDAS EMİRLERİ — ana kural GRW (büyüme + kırılım) paper hesabından bugünün uygulanabilir emir listesi.
 * AL: açılışta (adet + koruyucu stop) · SAT: kapanış 50 günlük ortalamanın altına indiyse açılışta · TUT: geçerli stop.
 * Kurallar GRW ile birebir (Lab v3 Ek W); bu sayfa yalnız okur (strateji kodu değişmez).
 * Kapanana kadar eski H6S pozisyonları ayrı bir kartta listelenir (kendi kuralıyla: 8. seans kapanışı).
 */
import { useEffect, useState } from "react";
import { getOrders, type OrdersData } from "@/lib/api";
import { ClipboardList, RefreshCw } from "lucide-react";

const usd = (v: number) => `$${v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const day = (iso: string) => {
    const d = new Date(iso + "T12:00:00");
    return d.toLocaleDateString("tr-TR", { day: "numeric", month: "short", weekday: "short" });
};
const ROW: React.CSSProperties = {
    display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 8, padding: "8px 0",
    borderBottom: "1px solid var(--border)",
};

function Section({ title, tone, empty, children }: { title: string; tone: string; empty: string; children: React.ReactNode }) {
    const hasRows = Array.isArray(children) ? children.length > 0 : !!children;
    return (
        <div className="card" style={{ borderLeft: `3px solid ${tone}` }}>
            <strong style={{ color: tone }}>{title}</strong>
            <div style={{ marginTop: 10 }}>
                {hasRows ? children : <div style={{ color: "var(--text-muted)", fontSize: 13 }}>{empty}</div>}
            </div>
        </div>
    );
}

export default function OrdersPage() {
    const [data, setData] = useState<OrdersData | null>(null);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(true);
    const load = async () => {
        setLoading(true);
        setError("");
        try {
            setData(await getOrders());
        } catch (e: unknown) {
            const ax = e as { response?: { status?: number; data?: { detail?: string } }; message?: string };
            setError(ax?.response?.status === 401 ? "Emirleri görmek için giriş yapın." : (ax?.response?.data?.detail || ax?.message || "Emirler alınamadı"));
        } finally {
            setLoading(false);
        }
    };
    useEffect(() => { void load(); }, []);

    if (loading) return <div className="card">Emirler hazırlanıyor…</div>;
    if (error) return <div className="card" style={{ color: "var(--danger)" }}>{error}</div>;
    if (!data) return null;

    const maxStop = data.max_stop_pct ?? 8;
    const sma = data.sma ?? 50;
    const old = data.legacy_h6s;

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
                <div>
                    <h1 style={{ display: "flex", alignItems: "center", gap: 8, margin: 0 }}>
                        <ClipboardList size={22} /> Midas Emirleri · {day(data.session)}
                    </h1>
                    <p style={{ color: "var(--text-muted)", margin: "6px 0 0", fontSize: 13, maxWidth: 860 }}>
                        Ana kuralın (büyüme + kırılım) bugünkü planı. Kurallar sabit: açılışta al, alır almaz koruyucu stop gir
                        (kırılım gününün dibi; alış fiyatının en çok %{maxStop} altı). Kapanış {sma} günlük ortalamanın altına
                        inerse ertesi açılışta sat. Kâr hedefi ve süre sınırı yok.
                        Hesap {usd(data.account)}{data.start != null && data.start !== data.account ? ` (başlangıç ${usd(data.start)} + kapanan işlemlerin net sonucu)` : ""} · bağlı {usd(data.held)} · boş {usd(data.free)}{data.idle_pct != null ? ` (%${data.idle_pct})` : ""}.
                        {data.fee_per_order ? ` Her emir ${usd(data.fee_per_order)} Midas ücreti (net sonuca dahil).` : ""}
                    </p>
                </div>
                <button className="btn-secondary" onClick={() => void load()}>
                    <RefreshCw size={15} style={{ marginRight: 6 }} /> Yenile
                </button>
            </div>

            <div className="card" style={{ borderLeft: "3px solid var(--accent)", fontSize: 13 }}>
                <strong>Bu liste sanal (paper) hesap içindir.</strong> Bu kural yalnız satışı en az %25 büyüyen ve kârı artan
                şirketlerdeki kırılımları alır ve yalnız piyasa yükselişteyken alım yapar. Geçmiş testte hesabı büyüttü
                (2016–2020 +%36, 2021–26 +%64) ama endeksin gerisinde kaldı (+%102 ve +%99) ve sonuçlar görüldükten sonra
                seçildi (Lab v3 Ek V, W, X). Gerçek değeri ileri testte ölçülüyor (Gölge sayfası).
            </div>

            {data.loss_streak_brake && (
                <div className="card" style={{ borderLeft: "3px solid #f59e0b", fontSize: 13 }}>
                    ⚠ <strong>Kayıp serisi freni aktif:</strong> son 2 işlem zararla kapandı. Yeni pozisyonlarda işlem başına
                    risk %{data.risk_pct ?? 0.5} (normalin yarısı); ilk kârlı kapanışta normale dönülür.
                </div>
            )}

            <Section title={`🟢 Açılışta AL (${data.buy.length})`} tone="var(--success)"
                empty="Bugün alınacak hisse yok (piyasa yükselişte değil ya da büyüyen şirkette kırılım yok — bu da bir karar).">
                {data.buy.map(b => (
                    <div key={b.ticker} style={ROW}>
                        <div><strong style={{ fontSize: "1.05rem" }}>{b.ticker}</strong>{b.overdue && <span style={{ color: "var(--danger)", fontSize: 12 }}> · giriş günü geçti</span>}</div>
                        <div>{b.shares} adet · ~{usd(b.est_cost)}</div>
                        <div style={{ color: "var(--danger)" }} title="Açılış kırılım gününün dibinde ya da altındaysa kural işlem yapmaz">
                            Açılış {usd(b.signal_low ?? 0)} ya da altındaysa ALMA
                        </div>
                        <div title="Alış fiyatı kesinleşince: bu iki seviyeden yüksek olanı">
                            Stop: {usd(b.signal_low ?? 0)} ile alışın %{maxStop} altından yüksek olanı{" "}
                            <span style={{ color: "var(--text-muted)" }}>(~{usd(b.est_stop)})</span>
                        </div>
                        <div style={{ color: "var(--text-muted)" }}>Al {day(b.buy_day)} açılış</div>
                    </div>
                ))}
            </Section>

            <Section title={`🔴 Bugün açılışta SAT (${data.sell.length})`} tone="var(--danger)"
                empty="Bugün satılacak pozisyon yok.">
                {data.sell.map(p => (
                    <div key={p.ticker} style={ROW}>
                        <div><strong style={{ fontSize: "1.05rem" }}>{p.ticker}</strong></div>
                        <div>{p.shares} adet · alış {usd(p.entry_price)}</div>
                        <div>Açılışta piyasa emriyle sat; stop emrini iptal et</div>
                        <div style={{ color: "var(--text-muted)" }}>Son kapanış {sma} günlük ortalamanın altında</div>
                    </div>
                ))}
            </Section>

            <Section title={`⏸ TUT (${data.hold.length})`} tone="var(--accent)" empty="Açık pozisyon yok.">
                {data.hold.map(p => (
                    <div key={p.ticker} style={ROW}>
                        <div><strong>{p.ticker}</strong></div>
                        <div>{p.shares} adet · alış {usd(p.entry_price)}</div>
                        <div>Stop: <strong>{usd(p.stop)}</strong> (Midas&apos;ta açık kalsın)</div>
                        <div style={{ color: "var(--text-muted)" }}>Satış: kapanış {sma} günlük ortalamanın altına inince</div>
                    </div>
                ))}
            </Section>

            {old && (
                <div className="card" style={{ borderLeft: "3px solid var(--text-muted)", fontSize: 13 }}>
                    <strong>Eski kuraldan (H6S) kalan pozisyonlar</strong>
                    <div style={{ color: "var(--text-muted)", margin: "4px 0 8px" }}>
                        H6S 28 Kasım 2026&apos;da ana kural olmaktan çıkarıldı ve yeni pozisyon açmıyor. Aşağıdakiler kendi
                        kuralıyla kapanır: stop = alış − {old.stop_atr}×ATR, {old.hold_sessions}. seans kapanışında satış.
                    </div>
                    {old.buy.map(b => (
                        <div key={`b-${b.ticker}`} style={ROW}>
                            <div><strong>{b.ticker}</strong> · AL</div>
                            <div>{b.shares} adet · ~{usd(b.est_cost)}</div>
                            <div>Stop: alış − {usd(b.stop_distance)}</div>
                            <div style={{ color: "var(--text-muted)" }}>Al {day(b.buy_day)}{b.sell_day ? ` · Sat ${day(b.sell_day)} kapanış` : ""}</div>
                        </div>
                    ))}
                    {old.sell.map(p => (
                        <div key={`s-${p.ticker}`} style={ROW}>
                            <div><strong>{p.ticker}</strong> · <span style={{ color: "var(--danger)" }}>SAT</span></div>
                            <div>{p.shares} adet · alış {usd(p.entry_price)}</div>
                            <div>{p.overdue ? "Gecikti, hemen piyasa emriyle sat" : "Bugün kapanışa yakın piyasa emriyle sat"}</div>
                            <div style={{ color: "var(--text-muted)" }}>{day(p.entry_day)}{p.sell_day ? ` → ${day(p.sell_day)}` : ""}</div>
                        </div>
                    ))}
                    {old.hold.map(p => (
                        <div key={`h-${p.ticker}`} style={ROW}>
                            <div><strong>{p.ticker}</strong> · TUT</div>
                            <div>{p.shares} adet · alış {usd(p.entry_price)}</div>
                            <div>Stop: <strong>{usd(p.stop)}</strong></div>
                            <div style={{ color: "var(--text-muted)" }}>{p.sell_day ? `Sat ${day(p.sell_day)} kapanış` : ""}</div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
