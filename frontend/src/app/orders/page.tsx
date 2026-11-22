"use client";
/**
 * MİDAS EMİRLERİ — H6S paper hesabından bugünün uygulanabilir emir listesi.
 * AL: açılışta (adet + koruyucu stop) · SAT: 8. seans kapanışında · TUT: geçerli stop.
 * Kurallar H6S ile birebir; bu sayfa yalnız okur (strateji kodu değişmez).
 */
import { useEffect, useState } from "react";
import { getOrders, type OrdersData } from "@/lib/api";
import { ClipboardList, RefreshCw } from "lucide-react";

const usd = (v: number) => `$${v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const day = (iso: string) => {
    const d = new Date(iso + "T12:00:00");
    return d.toLocaleDateString("tr-TR", { day: "numeric", month: "short", weekday: "short" });
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

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
                <div>
                    <h1 style={{ display: "flex", alignItems: "center", gap: 8, margin: 0 }}>
                        <ClipboardList size={22} /> Midas Emirleri · {day(data.session)}
                    </h1>
                    <p style={{ color: "var(--text-muted)", margin: "6px 0 0", fontSize: 13, maxWidth: 860 }}>
                        H6S hesabının bugünkü planı. Kurallar sabit: açılışta al, alır almaz koruyucu stop gir
                        (alış − {data.stop_atr}×ATR), {data.hold_sessions}. seans kapanışında sat. Hedef ve iz süren stop yok.
                        Hesap {usd(data.account)}{data.start != null && data.start !== data.account ? ` (başlangıç ${usd(data.start)} + kapanan işlemlerin net sonucu)` : ""} · bağlı {usd(data.held)} · boş {usd(data.free)}{data.idle_pct != null ? ` (%${data.idle_pct})` : ""}.
                        {data.fee_per_order ? ` Her emir ${usd(data.fee_per_order)} Midas ücreti (net sonuca dahil).` : ""}
                    </p>
                </div>
                <button className="btn-secondary" onClick={() => void load()}>
                    <RefreshCw size={15} style={{ marginRight: 6 }} /> Yenile
                </button>
            </div>

            {data.loss_streak_brake && (
                <div className="card" style={{ borderLeft: "3px solid #f59e0b", fontSize: 13 }}>
                    ⚠ <strong>Kayıp serisi freni aktif:</strong> son 3 H6S işlemi zararla kapandı. Yeni pozisyonlar yarım boyut
                    (hesabın %5&apos;i); ilk kârlı kapanışta tam boyuta dönülür. Geçmiş testte hem getiriyi hem düşüşü iyileştirdi.
                </div>
            )}

            <Section title={`🟢 Açılışta AL (${data.buy.length})`} tone="var(--success)"
                empty="Bugün alınacak H6S sinyali yok (piyasa freni ya da kırılım yok — bu da bir karar).">
                {data.buy.map(b => (
                    <div key={b.ticker} style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 8, padding: "8px 0", borderBottom: "1px solid var(--border)" }}>
                        <div><strong style={{ fontSize: "1.05rem" }}>{b.ticker}</strong>{b.overdue && <span style={{ color: "var(--danger)", fontSize: 12 }}> · giriş günü geçti</span>}</div>
                        <div>{b.shares} adet · ~{usd(b.est_cost)}</div>
                        <div title="Alış fiyatı kesinleşince: stop = alış − bu mesafe">Stop: alış − {usd(b.stop_distance)} <span style={{ color: "var(--text-muted)" }}>(~{usd(b.est_stop)})</span></div>
                        <div style={{ color: "var(--text-muted)" }}>Al {day(b.buy_day)} · Sat {day(b.sell_day)} kapanış</div>
                    </div>
                ))}
            </Section>

            <Section title={`🔴 Bugün kapanışta SAT (${data.sell.length})`} tone="var(--danger)"
                empty="Bugün süresi dolan pozisyon yok.">
                {data.sell.map(p => (
                    <div key={p.ticker} style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 8, padding: "8px 0", borderBottom: "1px solid var(--border)" }}>
                        <div><strong style={{ fontSize: "1.05rem" }}>{p.ticker}</strong>{p.overdue && <span style={{ color: "var(--danger)", fontSize: 12 }}> · gecikti, hemen sat</span>}</div>
                        <div>{p.shares} adet · alış {usd(p.entry_price)}</div>
                        <div>{p.overdue ? "Hemen piyasa emriyle" : "Kapanışa yakın piyasa emriyle"}</div>
                        <div style={{ color: "var(--text-muted)" }}>{day(p.entry_day)} → {day(p.sell_day)}</div>
                    </div>
                ))}
            </Section>

            <Section title={`⏸ TUT (${data.hold.length})`} tone="var(--accent)" empty="Açık H6S pozisyonu yok.">
                {data.hold.map(p => (
                    <div key={p.ticker} style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 8, padding: "8px 0", borderBottom: "1px solid var(--border)" }}>
                        <div><strong>{p.ticker}</strong></div>
                        <div>{p.shares} adet · alış {usd(p.entry_price)}</div>
                        <div>Stop: <strong>{usd(p.stop)}</strong> (Midas&apos;ta açık kalsın)</div>
                        <div style={{ color: "var(--text-muted)" }}>Sat {day(p.sell_day)} kapanış</div>
                    </div>
                ))}
            </Section>
        </div>
    );
}
