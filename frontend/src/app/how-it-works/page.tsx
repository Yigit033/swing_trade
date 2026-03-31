import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";
import { ArrowDown, BookOpen, Lightbulb, GitBranch, AlertCircle } from "lucide-react";

export const metadata: Metadata = {
    title: "Nasıl çalışır? | Swing Trade",
    description: "Small-cap scanner akışı, paper trade ve ayarlarda iyileştirme rehberi.",
};

/** Ayarlar sayfasındaki bölüm `id` ile aynı (settings/page.tsx). */
function SettingsSectionLink({
    id,
    children,
}: {
    id: string;
    children: ReactNode;
}) {
    return (
        <Link href={`/settings#${id}`} style={{ color: "var(--accent)", fontWeight: 600 }}>
            {children}
        </Link>
    );
}

function ArrowDivider() {
    return (
        <div
            aria-hidden
            style={{
                display: "flex",
                justifyContent: "center",
                padding: "6px 0",
                color: "var(--accent)",
                opacity: 0.85,
            }}
        >
            <ArrowDown size={22} strokeWidth={2.5} />
        </div>
    );
}

function PipelineStep({
    n,
    title,
    children,
}: {
    n: number;
    title: string;
    children: ReactNode;
}) {
    return (
        <div className="glass-card" style={{ padding: "16px 20px" }}>
            <div style={{ display: "flex", alignItems: "flex-start", gap: 14 }}>
                <div
                    style={{
                        flexShrink: 0,
                        width: 32,
                        height: 32,
                        borderRadius: 10,
                        background: "linear-gradient(135deg, rgba(59,130,246,0.35), rgba(139,92,246,0.25))",
                        border: "1px solid rgba(59,130,246,0.35)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontWeight: 800,
                        fontSize: "0.85rem",
                        color: "var(--text-primary)",
                    }}
                >
                    {n}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                    <h3 style={{ margin: "0 0 8px", fontSize: "0.95rem", fontWeight: 700, color: "var(--text-primary)" }}>
                        {title}
                    </h3>
                    <div style={{ fontSize: "0.82rem", lineHeight: 1.65, color: "var(--text-secondary)" }}>{children}</div>
                </div>
            </div>
        </div>
    );
}

function PlaybookCard({
    title,
    symptom,
    focus,
    children,
}: {
    title: string;
    symptom: string;
    focus: string;
    children: ReactNode;
}) {
    return (
        <div className="glass-card" style={{ padding: "18px 20px", borderLeft: "3px solid var(--yellow)" }}>
            <div style={{ fontWeight: 800, fontSize: "0.88rem", color: "var(--text-primary)", marginBottom: 6 }}>{title}</div>
            <p style={{ margin: "0 0 10px", fontSize: "0.78rem", color: "var(--text-muted)", fontStyle: "italic" }}>{symptom}</p>
            <p style={{ margin: "0 0 12px", fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.55 }}>
                <strong style={{ color: "var(--accent)" }}>Pipeline odağı:</strong> {focus}
            </p>
            <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.65 }}>{children}</div>
        </div>
    );
}

export default function HowItWorksPage() {
    return (
        <div style={{ maxWidth: 820, margin: "0 auto", padding: "24px 18px 48px" }}>
            <header style={{ marginBottom: 28 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
                    <div
                        style={{
                            width: 44,
                            height: 44,
                            borderRadius: 12,
                            background: "linear-gradient(135deg, rgba(59,130,246,0.2), rgba(139,92,246,0.15))",
                            border: "1px solid var(--border)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                        }}
                    >
                        <BookOpen size={22} color="var(--accent)" />
                    </div>
                    <div>
                        <h1 style={{ margin: 0, fontSize: "1.55rem", fontWeight: 800, letterSpacing: "-0.02em" }}>Nasıl çalışır?</h1>
                        <p style={{ margin: "4px 0 0", fontSize: "0.8rem", color: "var(--text-muted)" }}>
                            Small-cap scanner’dan paper trade’e kadar akış ve kötü sonuçta ayar rehberi
                        </p>
                    </div>
                </div>
                <p style={{ fontSize: "0.84rem", lineHeight: 1.65, color: "var(--text-secondary)", margin: 0 }}>
                    Bu sayfa, uygulamanın <strong style={{ color: "var(--text-primary)" }}>canlı small-cap tarama</strong> hattını özetler.
                    Backtest aynı motoru tarihsel veriyle simüle eder;{" "}
                    <Link href="/lookup" style={{ color: "var(--accent)" }}>
                        Manual Lookup
                    </Link>{" "}
                    ve{" "}
                    <Link href="/pending" style={{ color: "var(--accent)" }}>
                        Pending
                    </Link>{" "}
                    ekranları ayrı giriş noktalarıdır.
                </p>
            </header>

            {/* Pipeline */}
            <section style={{ marginBottom: 40 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
                    <GitBranch size={18} color="var(--purple)" />
                    <h2 style={{ margin: 0, fontSize: "1.05rem", fontWeight: 800 }}>Scanner → Track pipeline</h2>
                </div>
                <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", margin: "0 0 18px", lineHeight: 1.55 }}>
                    <Link href="/scanner" style={{ color: "var(--accent)" }}>
                        Scanner
                    </Link>{" "}
                    sayfasında <strong>Scan</strong> ile başlar; sonuç kartlarında <strong>Track</strong> ile paper trade listesine eklenir.
                </p>

                <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
                    <PipelineStep n={1} title="Scan ve parametreler">
                        Tarayıcı, seçtiğin <code style={{ fontSize: "0.76rem", background: "rgba(255,255,255,0.06)", padding: "2px 6px", borderRadius: 4 }}>min_quality</code>,{" "}
                        <code style={{ fontSize: "0.76rem", background: "rgba(255,255,255,0.06)", padding: "2px 6px", borderRadius: 4 }}>top_n</code>,{" "}
                        <code style={{ fontSize: "0.76rem", background: "rgba(255,255,255,0.06)", padding: "2px 6px", borderRadius: 4 }}>portfolio_value</code> ile arka plan tarama
                        işini başlatır. Sayfayı değiştirsen bile işlem sürer; durum periyodik olarak sunucudan okunur.
                    </PipelineStep>
                    <ArrowDivider />
                    <PipelineStep n={2} title="API: job kuyruğu">
                        Sunucu hemen bir <strong>job_id</strong> döner; tarama ayrı bir iş parçacığında çalışır. Aynı anda ikinci bir tarama
                        istenirse mevcut işe bağlanma veya çakışma yanıtı dönebilir.
                    </PipelineStep>
                    <ArrowDivider />
                    <PipelineStep n={3} title="Universe (hisse listesi)">
                        Motor, ayarlardaki kurallara göre small-cap evrenini oluşturur (Finviz veya statik liste, ticker üst sınırı, önbellek süresi).
                    </PipelineStep>
                    <ArrowDivider />
                    <PipelineStep n={4} title="Fiyat verisi">
                        Her ticker için yakın geçmiş OHLCV çekilir; yeterli barı olmayan semboller atlanır.
                    </PipelineStep>
                    <ArrowDivider />
                    <PipelineStep n={5} title="Piyasa rejimi (bir kez)">
                        Tüm hisseler için ortak <strong>BULL / BEAR / CAUTION</strong> rejimi ve skor çarpanı hesaplanır; sonuçlar kalıcı depoya da yazılabilir.
                    </PipelineStep>
                    <ArrowDivider />
                    <PipelineStep n={6} title="Hisse başına motor (scan_stock)">
                        <ul style={{ margin: "8px 0 0", paddingLeft: 18 }}>
                            <li><strong>Filtreler</strong> — float, ATR%, kazanç penceresi vb.</li>
                            <li><strong>Tetikleyiciler</strong> — hacim patlaması, kırılım vb.</li>
                            <li><strong>Swing hazır</strong> — örn. 5g momentum ve MA20 üstü; geçmezse sinyal üretilmez.</li>
                            <li><strong>Canlı modda</strong> — sektör RS, kısa pozisyon / insider / haber gibi katalizörler.</li>
                            <li><strong>Swing tipi</strong> (A/B/C/S), sert RSI ve geç-giriş kapıları.</li>
                            <li><strong>Kalite skoru</strong> ve <strong>risk</strong> — stop, hedefler, pozisyon büyüklüğü, tutma aralığı.</li>
                            <li><strong>Anlatı</strong> — teknik özet metin (LLM), backtest modunda kapalıdır.</li>
                        </ul>
                    </PipelineStep>
                    <ArrowDivider />
                    <PipelineStep n={7} title="Rejim çarpanı ve sıralama">
                        Gösterilen skor rejim çarpanıyla düşürülebilir; <strong>orijinal skor</strong> ayrı saklanır. Adaylar kaliteye göre sıralanır.
                    </PipelineStep>
                    <ArrowDivider />
                    <PipelineStep n={8} title="API son filtresi">
                        İstekteki eşiklere ve rejime göre <strong>etkili min kalite</strong> ve <strong>etkili top_n</strong> uygulanır; liste kesilir.
                    </PipelineStep>
                    <ArrowDivider />
                    <PipelineStep n={9} title="Sonuç ekranı ve Track">
                        Tamamlanınca sinyaller tarayıcıda gösterilir; isteğe bağlı <strong>otomatik Track</strong> eşiği açıksa uygun kartlar paper trade’e eklenir.
                        Elle <strong>Track</strong> aynı API ile <Link href="/trades" style={{ color: "var(--accent)" }}>Paper Trades</Link>’e kayıt açar (mükerrer güvenli).
                    </PipelineStep>
                </div>
            </section>

            {/* Other flows */}
            <section style={{ marginBottom: 40 }}>
                <h2 style={{ margin: "0 0 12px", fontSize: "1.05rem", fontWeight: 800 }}>Diğer akışlar (kısa)</h2>
                <div className="glass-card" style={{ padding: "16px 20px", fontSize: "0.82rem", lineHeight: 1.65, color: "var(--text-secondary)" }}>
                    <ul style={{ margin: 0, paddingLeft: 18 }}>
                        <li style={{ marginBottom: 8 }}>
                            <Link href="/lookup" style={{ color: "var(--accent)", fontWeight: 600 }}>Manual Lookup</Link> — Belirttiğin semboller için analiz; evren taraması değil.
                        </li>
                        <li style={{ marginBottom: 8 }}>
                            <Link href="/pending" style={{ color: "var(--accent)", fontWeight: 600 }}>Pending</Link> — Giriş onayı bekleyen işlemler (scanner akışına bağlı alt süreç).
                        </li>
                        <li>
                            <Link href="/backtest" style={{ color: "var(--accent)", fontWeight: 600 }}>Backtest</Link> — Aynı small-cap motorunu geçmişte yürütür; canlı veri / anlatı farkları olabilir.
                        </li>
                    </ul>
                </div>
            </section>

            {/* Tuning playbook */}
            <section style={{ marginBottom: 24 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                    <Lightbulb size={18} color="var(--yellow)" />
                    <h2 style={{ margin: 0, fontSize: "1.05rem", fontWeight: 800 }}>Performans kötüyse nereye bakmalı?</h2>
                </div>

                <div
                    className="glass-card"
                    style={{
                        padding: "14px 18px",
                        marginBottom: 18,
                        display: "flex",
                        gap: 12,
                        alignItems: "flex-start",
                        borderLeft: "3px solid var(--accent)",
                    }}
                >
                    <AlertCircle size={20} style={{ flexShrink: 0, color: "var(--accent)", marginTop: 2 }} />
                    <div style={{ fontSize: "0.8rem", lineHeight: 1.6, color: "var(--text-secondary)" }}>
                        <strong style={{ color: "var(--text-primary)" }}>10–15 paper trade</strong> küçük bir örneklem; eğilim okumak için faydalıdır ama istatistiksel olarak gürültülüdür.
                        Mümkünse aynı dönemde{" "}
                        <Link href="/backtest" style={{ color: "var(--accent)" }}>backtest</Link> çıktısıyla karşılaştır; ikisi aynı yönü gösteriyorsa ayar değişikliği daha güvenilir.
                        Aşırı ince ayar tek veri setinde <strong>overfitting</strong> riskini artırır.
                    </div>
                </div>

                <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", margin: "0 0 16px", lineHeight: 1.55 }}>
                    Aşağıdaki satırlar <strong>Ayarlar</strong> sayfasındaki bölüm başlıklarına doğrudan gider (#anchor). Özet kutu:{" "}
                    <Link href="/settings#settings-ayar-rehberi-giris" style={{ color: "var(--accent)", fontWeight: 600 }}>
                        Zarar ediyorsanız — hangi ayar neyi etkiler?
                    </Link>
                </p>

                <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                    <PlaybookCard
                        title="Çok az sinyal veya liste boş"
                        symptom="Tarama sürekli boş veya çok az kart dönüyor."
                        focus="Evren genişliği, veri eksikliği, motor içi filtreler ve son API eşiği."
                    >
                        <ul style={{ margin: 0, paddingLeft: 18 }}>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-finviz-evren">Finviz evren taraması (~N hisse)</SettingsSectionLink> — tarama tavanı, Finviz açık/kapalı,
                                önbellek, sıralama ve chase cezası.
                            </li>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-evren-filtreleri">Evren filtreleri</SettingsSectionLink> — bar verisiyle elenen mcap/hacim/fiyat/float.
                            </li>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-tarama-gecitleri">Tarama geçitleri</SettingsSectionLink> — parabol / ekstrem / geç giriş eşikleri çok sıkıysa aday azalır.
                            </li>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-sinyal-filtresi">Sinyal filtresi</SettingsSectionLink> — max RSI, volume surge, min ATR%.
                            </li>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-rejim-min-kalite">Rejim (min kalite / top N tavanı)</SettingsSectionLink> — BEAR/CAUTION’da liste kısabilir.
                            </li>
                            <li style={{ marginBottom: 0 }}>
                                <Link href="/scanner" style={{ color: "var(--accent)", fontWeight: 600 }}>Scanner</Link> ekranındaki{" "}
                                <code style={{ fontSize: "0.76rem" }}>min_quality</code> / <code style={{ fontSize: "0.76rem" }}>top_n</code>{" "}
                                istek parametreleri (JSON dosyasında değil).
                            </li>
                        </ul>
                    </PlaybookCard>
                    <PlaybookCard
                        title="Sık stop, kısa sürede zarar"
                        symptom="Paper trade’lerde çoğu işlem stop ile kapanıyor."
                        focus="Giriş kalitesi mi dar stop mu — ikisini ayırarak bak."
                    >
                        <ul style={{ margin: 0, paddingLeft: 18 }}>
                            <li style={{ marginBottom: 8 }}>
                                Giriş: <SettingsSectionLink id="settings-section-tarama-gecitleri">Tarama geçitleri</SettingsSectionLink>,{" "}
                                <SettingsSectionLink id="settings-section-sinyal-filtresi">Sinyal filtresi</SettingsSectionLink>,{" "}
                                <SettingsSectionLink id="settings-section-swing-hazirlik">Swing hazırlık / aşırı uzama (sinyal)</SettingsSectionLink>.
                            </li>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-risk-yonetimi">Risk yönetimi</SettingsSectionLink> — stop ATR, min/max stop %, tip bazlı tavanlar.
                            </li>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-hedefler-atr">Hedefler (ATR + tavan %)</SettingsSectionLink> — T1/T2 mesafeleri; stop ile birlikte R:R hissi verir.
                            </li>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-backtest-giris-yurutme">Backtest / giriş yürütme</SettingsSectionLink> — gap limitleri, max kayıp/işlem, gap risk bütçesi.
                            </li>
                            <li style={{ marginBottom: 0 }}>
                                Simülasyonda çıkış:{" "}
                                <SettingsSectionLink id="settings-section-backtest-cikis-trailing">Backtest çıkış (zaman stop / trailing)</SettingsSectionLink>.
                            </li>
                        </ul>
                    </PlaybookCard>
                    <PlaybookCard
                        title="İşlemler hep ters dönüyor (whipsaw)"
                        symptom="Giriş sonrası hemen ters hareket, düşük kazanma oranı."
                        focus="Erken / geç giriş ve trend uyumu."
                    >
                        <ul style={{ margin: 0, paddingLeft: 18 }}>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-tarama-gecitleri">Tarama geçitleri</SettingsSectionLink> — geç giriş ve parabol satırları.
                            </li>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-swing-hazirlik">Swing hazırlık / aşırı uzama (sinyal)</SettingsSectionLink> — MA20 ve 5g aşırı uzama.
                            </li>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-backtest-giris-ema-gap">Backtest giriş (EMA / gap)</SettingsSectionLink> — trend EMA hizalama, Tip C açılış oranı.
                            </li>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-skorlama">Skorlama ayarları</SettingsSectionLink> —{" "}
                                <SettingsSectionLink id="settings-details-skorlama-kademe-tablolari">Kademe tabloları</SettingsSectionLink>,{" "}
                                <SettingsSectionLink id="settings-details-skorlama-momentum-ham">Momentum ve risk — ham alt puanlar</SettingsSectionLink>, bonus/ceza ızgarası (aynı blokta).
                            </li>
                            <li style={{ marginBottom: 0 }}>
                                Tip karışıklığı:{" "}
                                <SettingsSectionLink id="settings-section-swing-siniflandirma-gelismis">Swing sınıflandırma (S / C / B / A) — gelişmiş</SettingsSectionLink>.
                            </li>
                        </ul>
                    </PlaybookCard>
                    <PlaybookCard
                        title="Ayı veya CAUTION’da batma"
                        symptom="Sadece zayıf piyasa günlerinde veya rejim uyarısı varken kötü sonuç."
                        focus="Rejim çarpanı ve rejime özel kalite tabanları."
                    >
                        <ul style={{ margin: 0, paddingLeft: 18 }}>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-rejim-min-kalite">Rejim (min kalite / top N tavanı)</SettingsSectionLink> — BULL/BEAR/CAUTION tabanları.
                            </li>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-backtest-tip-kalitesi">Backtest tip kalitesi (BEAR / CAUTION tabanları)</SettingsSectionLink> — tip C/A/B için ayı ve CAUTION zeminleri.
                            </li>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-risk-hedefleri-rejim">Risk hedefleri (rejim / kalite)</SettingsSectionLink> — CAUTION/BEAR T2 çarpanları ve R:R zeminleri.
                            </li>
                            <li style={{ marginBottom: 0 }}>
                                Scanner’da görünen <code style={{ fontSize: "0.76rem" }}>regime_multiplier</code> cezası: önce{" "}
                                <SettingsSectionLink id="settings-section-sinyal-filtresi">Sinyal filtresi</SettingsSectionLink> /{" "}
                                <SettingsSectionLink id="settings-section-tarama-gecitleri">tarama geçitleri</SettingsSectionLink>{" "}
                                sıkılaştırmak, min kaliteyi körü körüne düşürmekten çoğu zaman daha güvenli bir denemedir.
                            </li>
                        </ul>
                    </PlaybookCard>
                    <PlaybookCard
                        title="Backtest genel olarak zayıf"
                        symptom="Walk-forward veya uzun dönem metrikler düşük (getiri, drawdown, win rate)."
                        focus="Önce evren ve giriş kuralları, sonra risk; tek seferde onlarca kaydırma yapma."
                    >
                        <ul style={{ margin: 0, paddingLeft: 18 }}>
                            <li style={{ marginBottom: 8 }}>
                                Evren: <SettingsSectionLink id="settings-section-finviz-evren">Finviz evren taraması</SettingsSectionLink>,{" "}
                                <SettingsSectionLink id="settings-section-evren-filtreleri">Evren filtreleri</SettingsSectionLink> (manuel liste vs canlı Finviz farkı).
                            </li>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-skorlama">Skorlama ayarları</SettingsSectionLink> — tek seferde tek eksen seç (ör. sadece geç giriş cezaları veya sadece ağırlıklar).
                            </li>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-backtest-dongu">Backtest döngü (rejim / drawdown)</SettingsSectionLink> — ayıda giriş kapatma, drawdown durdurma.
                            </li>
                            <li style={{ marginBottom: 8 }}>
                                Çıkış:{" "}
                                <SettingsSectionLink id="settings-section-backtest-cikis-trailing">Backtest çıkış (zaman stop / trailing)</SettingsSectionLink>.
                            </li>
                            <li style={{ marginBottom: 0 }}>
                                Özet: <Link href="/performance" style={{ color: "var(--accent)", fontWeight: 600 }}>Performance</Link> ve trade günlüğü ile birlikte oku.
                            </li>
                        </ul>
                    </PlaybookCard>
                    <PlaybookCard
                        title="Belirli swing tipinde kayıp"
                        symptom="Örneğin sadece Tip B veya sadece Tip C kötü."
                        focus="İlgili tipin skor ve tutma parametreleri."
                    >
                        <ul style={{ margin: 0, paddingLeft: 18 }}>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-swing-siniflandirma-gelismis">Swing sınıflandırma (S / C / B / A) — gelişmiş</SettingsSectionLink> — tip başına RSI, 5g, vol, tutma, min skor.
                            </li>
                            <li style={{ marginBottom: 8 }}>
                                <SettingsSectionLink id="settings-section-skorlama">Skorlama ayarları</SettingsSectionLink> — tip bazlı cezalar ve ağırlıklar; alt başlıklar:{" "}
                                <SettingsSectionLink id="settings-details-skorlama-kademe-tablolari">Kademe tabloları</SettingsSectionLink>,{" "}
                                <SettingsSectionLink id="settings-details-skorlama-momentum-ham">Momentum ham puanlar</SettingsSectionLink>.
                            </li>
                            <li style={{ marginBottom: 0 }}>
                                Liste kesimi: <SettingsSectionLink id="settings-section-rejim-min-kalite">Rejim</SettingsSectionLink> + Scanner{" "}
                                <code style={{ fontSize: "0.76rem" }}>min_quality</code> /{" "}
                                <code style={{ fontSize: "0.76rem" }}>top_n</code>.
                            </li>
                        </ul>
                    </PlaybookCard>
                </div>
            </section>

            <footer style={{ marginTop: 32, paddingTop: 20, borderTop: "1px solid var(--border)", fontSize: "0.75rem", color: "var(--text-muted)" }}>
                Bu sayfa proje davranışının özetidir; kod değiştikçe motor ayrıntıları güncellenebilir. Sorun yaşarsan önce API logları ve{" "}
                <Link href="/performance" style={{ color: "var(--accent)" }}>Performance</Link> metriklerine bak.
            </footer>
        </div>
    );
}
