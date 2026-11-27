/** İşlemi açan kuralın rozeti: GRW (ana kural), H6S (eski ana kural), diğerleri (eski kurallar / manuel). */
export function StrategyBadge({ strategy }: { strategy?: string | null }) {
    const s = String(strategy || "").toUpperCase();
    const style = { marginLeft: 6, fontSize: "0.65rem" };
    if (s === "GRW") return <span className="badge badge-green" style={style} title="GRW büyüme + kırılım (ana strateji)">GRW</span>;
    if (s === "H6S") return <span className="badge badge-blue" style={style} title="H6S (eski ana kural; 28 Kasım 2026'dan beri yeni pozisyon açmıyor)">H6S</span>;
    return <span className="badge badge-yellow" style={style} title="Eski kurallar (kontrol grubu / manuel)">Eski</span>;
}
