"use client";
import { useState, useRef, useEffect } from "react";
import { chatWithAI, getWeeklyReportAI, getModelStatus, trainModel, predictSignal } from "@/lib/api";
import { Send, Bot, User, FileText, Brain, Crosshair, Activity } from "lucide-react";

interface Message {
    role: "user" | "ai";
    content: string;
}

type Tab = "chat" | "weekly" | "model" | "predict";

const QUICK_PROMPTS = [
    "Hangi sektörlerde fırsat var?",
    "Win rate'imi nasıl arttırabilirim?",
    "Risk yönetimi için önerilerin neler?",
    "Bu hafta kaç trade açmalıyım?",
];

/* ─── Tab Button ──────────────────────────────────────── */
function TabBtn({ active, icon, label, onClick }: { active: boolean; icon: React.ReactNode; label: string; onClick: () => void }) {
    return (
        <button onClick={onClick} style={{
            padding: "8px 16px", borderRadius: 8, border: "1px solid",
            borderColor: active ? "var(--accent)" : "var(--border)",
            background: active ? "rgba(59,130,246,0.15)" : "transparent",
            color: active ? "var(--accent)" : "var(--text-secondary)",
            cursor: "pointer", fontSize: "0.8rem", fontWeight: 600,
            display: "flex", alignItems: "center", gap: 6, transition: "all 0.15s",
        }}>
            {icon}{label}
        </button>
    );
}

/* ─── Chat Tab ────────────────────────────────────────── */
function ChatTab() {
    const [messages, setMessages] = useState<Message[]>([
        { role: "ai", content: "Merhaba! Ben Swing Trade AI asistanınım. Trade stratejilerin, risk yönetimin veya pozisyonların hakkında sorular sorabilirsin. 📊" }
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const bottomRef = useRef<HTMLDivElement>(null);
    useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

    const send = async (text?: string) => {
        const message = text || input.trim();
        if (!message || loading) return;
        setInput("");
        setMessages(prev => [...prev, { role: "user", content: message }]);
        setLoading(true);
        const history = messages.map(m => ({ role: m.role, content: m.content }));
        try {
            const res = await chatWithAI(message, history);
            setMessages(prev => [...prev, { role: "ai", content: res.answer || "Yanıt alınamadı." }]);
        } catch {
            setMessages(prev => [...prev, { role: "ai", content: "❌ API bağlantısı kurulamadı." }]);
        } finally { setLoading(false); }
    };

    return (
        <>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
                {QUICK_PROMPTS.map(p => (
                    <button key={p} className="btn-secondary" style={{ fontSize: "0.78rem", padding: "5px 12px" }}
                        onClick={() => send(p)}>{p}</button>
                ))}
            </div>
            <div className="glass-card" style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
                <div style={{ flex: 1, overflowY: "auto", padding: "20px 22px", display: "flex", flexDirection: "column", gap: 14 }}>
                    {messages.map((m, i) => (
                        <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start", flexDirection: m.role === "user" ? "row-reverse" : "row" }}>
                            <div style={{
                                width: 32, height: 32, borderRadius: "50%", flexShrink: 0,
                                background: m.role === "user" ? "linear-gradient(135deg, #3b82f6, #8b5cf6)" : "linear-gradient(135deg, #10b981, #3b82f6)",
                                display: "flex", alignItems: "center", justifyContent: "center",
                            }}>
                                {m.role === "user" ? <User size={16} color="#fff" /> : <Bot size={16} color="#fff" />}
                            </div>
                            <div className={m.role === "user" ? "chat-user" : "chat-ai"}
                                style={{ fontSize: "0.875rem", lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
                                {m.content}
                            </div>
                        </div>
                    ))}
                    {loading && (
                        <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                            <div style={{ width: 32, height: 32, borderRadius: "50%", background: "linear-gradient(135deg, #10b981, #3b82f6)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                                <Bot size={16} color="#fff" />
                            </div>
                            <div className="chat-ai" style={{ display: "flex", gap: 6, alignItems: "center" }}>
                                <span className="spinner" style={{ width: 14, height: 14 }} />
                                <span style={{ color: "var(--text-secondary)", fontSize: "0.8rem" }}>Analyzing...</span>
                            </div>
                        </div>
                    )}
                    <div ref={bottomRef} />
                </div>
                <div style={{ padding: "14px 18px", borderTop: "1px solid var(--border)", display: "flex", gap: 10 }}>
                    <input className="input" value={input} onChange={e => setInput(e.target.value)}
                        onKeyDown={e => e.key === "Enter" && !e.shiftKey && send()}
                        placeholder="Strateji, risk yönetimi, trade fikirleri hakkında sor..." />
                    <button className="btn-primary" onClick={() => send()} disabled={loading || !input.trim()}
                        style={{ padding: "9px 16px", flexShrink: 0 }}><Send size={15} /></button>
                </div>
            </div>
        </>
    );
}

/* ─── Weekly Report Tab ───────────────────────────────── */
function WeeklyReportTab() {
    const [report, setReport] = useState<string | null>(null);
    const [context, setContext] = useState<Record<string, unknown> | null>(null);
    const [loading, setLoading] = useState(false);
    const [meta, setMeta] = useState<{ llm_available?: boolean; from_cache?: boolean; generated_at?: string }>({});

    const generate = async () => {
        setLoading(true);
        try {
            const res = await getWeeklyReportAI();
            if (res.success && res.report) {
                setReport(res.report);
                setContext(res.context || null);
                setMeta({ llm_available: res.llm_available, from_cache: res.from_cache, generated_at: res.generated_at });
            } else {
                setReport(res.error || "Rapor oluşturulamadı.");
            }
        } catch { setReport("❌ API bağlantı hatası."); }
        finally { setLoading(false); }
    };

    return (
        <div className="glass-card" style={{ padding: 24 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                <div>
                    <h2 style={{ margin: 0, fontSize: "1.1rem" }}>🤖 AI Haftalık Performans Analizi</h2>
                    <p style={{ margin: "4px 0 0", fontSize: "0.8rem", color: "var(--text-muted)" }}>
                        Geçmiş trade verilerini analiz ederek strateji öngörüleri ve iyileştirme önerileri sunar.
                    </p>
                </div>
                <button className="btn-primary" onClick={generate} disabled={loading}
                    style={{ padding: "8px 18px", fontSize: "0.85rem" }}>
                    {loading ? <span className="spinner" style={{ width: 14, height: 14 }} /> : "📝"} Rapor Oluştur
                </button>
            </div>

            {report && (
                <>
                    <div style={{ display: "flex", gap: 12, marginBottom: 14, flexWrap: "wrap" }}>
                        {meta.llm_available && <span className="badge badge-green">🤖 AI Analizi</span>}
                        {!meta.llm_available && <span className="badge badge-yellow">📊 İstatistik Rapor</span>}
                        {meta.from_cache && <span className="badge badge-blue">📦 Önbellekten</span>}
                        {meta.generated_at && <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>📅 {meta.generated_at}</span>}
                    </div>
                    <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.7, fontSize: "0.875rem" }}>{report}</div>
                    {context && (
                        <details style={{ marginTop: 16 }}>
                            <summary style={{ cursor: "pointer", fontSize: "0.8rem", color: "var(--accent)" }}>📊 Ham İstatistikler</summary>
                            <pre style={{ fontSize: "0.75rem", marginTop: 8, padding: 12, background: "rgba(0,0,0,0.2)", borderRadius: 8, overflow: "auto" }}>
                                {JSON.stringify(context, null, 2)}
                            </pre>
                        </details>
                    )}
                </>
            )}
            {!report && !loading && (
                <div style={{ padding: 32, textAlign: "center", color: "var(--text-muted)" }}>
                    Rapor oluşturmak için butona bas. Hesaplamalar deterministik sistemde yapılır, LLM sadece sonuçları yorumlar.
                </div>
            )}
        </div>
    );
}

/* ─── Model Status & Training Tab ─────────────────────── */
function ModelTab() {
    /* eslint-disable @typescript-eslint/no-explicit-any */
    const [status, setStatus] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [training, setTraining] = useState(false);
    const [trainResult, setTrainResult] = useState<any>(null);

    const load = async () => {
        setLoading(true);
        try { const r = await getModelStatus(); setStatus(r); } catch { /* ignore */ }
        finally { setLoading(false); }
    };

    useEffect(() => { load(); }, []);

    const handleTrain = async () => {
        setTraining(true);
        setTrainResult(null);
        try {
            const r = await trainModel();
            setTrainResult(r);
            if (r.success) load(); // Refresh status
        } catch { setTrainResult({ success: false, error: "API bağlantı hatası" }); }
        finally { setTraining(false); }
    };

    if (loading) return <div style={{ padding: 48, textAlign: "center" }}><span className="spinner" /></div>;

    const MILESTONE = 15;
    const real = status?.real_count || 0;
    const demo = status?.demo_count || 0;
    const meta = status?.meta;

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Milestone Banner */}
            <div className="glass-card" style={{ padding: 16 }}>
                <h3 style={{ margin: "0 0 12px", fontSize: "1rem" }}>🤖 AI Signal Quality Predictor</h3>
                <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", margin: "0 0 12px" }}>
                    Geçmiş paper trade sonuçlarından öğrenerek yeni sinyallerin kazanma ihtimalini tahmin eder (XGBoost).
                </p>
                {real === 0 && (
                    <div style={{ padding: "10px 14px", background: "rgba(59,130,246,0.1)", borderRadius: 8, fontSize: "0.8rem", color: "var(--accent)" }}>
                        📭 Henüz kapatılmış gerçek trade yok. İlk trade'leri kapatarak model eğitimine başla.
                    </div>
                )}
                {real > 0 && real < MILESTONE && (
                    <div style={{ padding: "10px 14px", background: "rgba(245,158,11,0.1)", borderRadius: 8, fontSize: "0.8rem" }}>
                        <span style={{ color: "rgb(245,158,11)" }}>⏳ {real}/{MILESTONE} gerçek trade — modeli güvenilir eğitmek için {MILESTONE - real} trade daha kapat.</span>
                        <div style={{ height: 6, background: "rgba(255,255,255,0.1)", borderRadius: 3, marginTop: 8, overflow: "hidden" }}>
                            <div style={{ height: "100%", width: `${(real / MILESTONE) * 100}%`, background: "rgb(245,158,11)", borderRadius: 3 }} />
                        </div>
                    </div>
                )}
                {real >= 30 && (
                    <div style={{ padding: "10px 14px", background: "rgba(34,197,94,0.1)", borderRadius: 8, fontSize: "0.8rem", color: "var(--green)" }}>
                        🏆 {real} gerçek trade! Model güçlü veri tabanına sahip.
                    </div>
                )}
                {real >= MILESTONE && real < 30 && (
                    <div style={{ padding: "10px 14px", background: "rgba(34,197,94,0.1)", borderRadius: 8, fontSize: "0.8rem", color: "var(--green)" }}>
                        🎉 {real} gerçek trade! Modeli yeniden eğitmek için harika bir an.
                    </div>
                )}
            </div>

            {/* Model Status */}
            <div className="glass-card" style={{ padding: 16 }}>
                <h3 style={{ margin: "0 0 12px", fontSize: "1rem" }}>📋 Model Durumu</h3>
                {meta ? (
                    <>
                        <div style={{ padding: "8px 14px", background: "rgba(34,197,94,0.1)", borderRadius: 8, marginBottom: 12, fontSize: "0.8rem", color: "var(--green)" }}>
                            ✅ Model eğitilmiş ve hazır
                        </div>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 10 }}>
                            {[
                                { label: "Accuracy", val: `${(meta.accuracy * 100).toFixed(1)}%` },
                                { label: "ROC-AUC", val: meta.roc_auc?.toFixed(3), sub: meta.roc_auc >= 0.7 ? "🟢 İyi" : meta.roc_auc >= 0.55 ? "🟡 Makul" : "🔴 Zayıf" },
                                { label: "F1 Score", val: meta.f1?.toFixed(3) },
                                { label: "Eğitim Verisi", val: `${meta.total_trades} trade` },
                            ].map(m => (
                                <div key={m.label} style={{ padding: "10px 12px", background: "rgba(255,255,255,0.04)", borderRadius: 8, textAlign: "center" }}>
                                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: 4 }}>{m.label}</div>
                                    <div style={{ fontSize: "1rem", fontWeight: 700 }}>{m.val}</div>
                                    {m.sub && <div style={{ fontSize: "0.7rem", marginTop: 2 }}>{m.sub}</div>}
                                </div>
                            ))}
                        </div>
                        <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 10 }}>
                            📅 Son eğitim: {meta.trained_at?.slice(0, 19)} | 5-Fold CV ROC-AUC: {meta.cv_roc_auc_mean?.toFixed(3)} ± {meta.cv_roc_auc_std?.toFixed(3)}
                        </p>
                    </>
                ) : (
                    <div style={{ padding: "10px 14px", background: "rgba(239,68,68,0.1)", borderRadius: 8, fontSize: "0.8rem", color: "var(--red)" }}>
                        ❌ Model henüz eğitilmedi. Aşağıdaki butonu kullanarak eğit.
                    </div>
                )}
            </div>

            {/* Train Model */}
            <div className="glass-card" style={{ padding: 16 }}>
                <h3 style={{ margin: "0 0 8px", fontSize: "1rem" }}>🏋️ Modeli Eğit / Güncelle</h3>
                <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: 12 }}>
                    Yeni trade'ler kapatıldıkça modeli güncelleyebilirsin. Eğitim birkaç saniye sürer.
                </p>
                <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: 12 }}>
                    📊 Mevcut veri: {real} gerçek trade + {demo} demo trade
                </p>
                <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                    <button className="btn-primary" onClick={handleTrain} disabled={training}
                        style={{ padding: "8px 20px" }}>
                        {training ? <><span className="spinner" style={{ width: 14, height: 14 }} /> Eğitiliyor...</> : "🚀 Modeli Eğit"}
                    </button>
                    <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", lineHeight: 1.5 }}>
                        <strong>Ne zaman yeniden eğitmeli?</strong><br />
                        • Her 10 yeni trade kapandığında<br />
                        • Strateji değişikliklerinden sonra<br />
                        • Model skoru düştüğünde
                    </div>
                </div>
                {trainResult && (
                    <div style={{
                        marginTop: 12, padding: "10px 14px", borderRadius: 8, fontSize: "0.8rem",
                        background: trainResult.success ? "rgba(34,197,94,0.1)" : "rgba(239,68,68,0.1)",
                        color: trainResult.success ? "var(--green)" : "var(--red)",
                    }}>
                        {trainResult.success
                            ? `✅ Eğitim tamamlandı! Accuracy: ${(trainResult.accuracy * 100).toFixed(1)}% | ROC-AUC: ${trainResult.roc_auc?.toFixed(3)} | F1: ${trainResult.f1?.toFixed(3)}`
                            : `❌ ${trainResult.error || "Eğitim başarısız"}`}
                    </div>
                )}
            </div>
        </div>
    );
}

/* ─── Live Signal Prediction Tab ──────────────────────── */
function PredictTab() {
    const [entry, setEntry] = useState("10.00");
    const [stop, setStop] = useState("9.00");
    const [target, setTarget] = useState("13.00");
    const [atr, setAtr] = useState("0.50");
    const [quality, setQuality] = useState(7);
    const [swingType, setSwingType] = useState("A");
    const [holdDays, setHoldDays] = useState(7);
    const [loading, setLoading] = useState(false);
    /* eslint-disable @typescript-eslint/no-explicit-any */
    const [result, setResult] = useState<any>(null);

    const handlePredict = async () => {
        setLoading(true);
        setResult(null);
        try {
            const r = await predictSignal({
                entry_price: parseFloat(entry),
                stop_loss: parseFloat(stop),
                target: parseFloat(target),
                atr: parseFloat(atr),
                quality_score: quality,
                swing_type: swingType,
                max_hold_days: holdDays,
            });
            setResult(r);
        } catch { setResult({ success: false, error: "API bağlantı hatası" }); }
        finally { setLoading(false); }
    };

    const entryF = parseFloat(entry) || 0;
    const stopF = parseFloat(stop) || 0;
    const targetF = parseFloat(target) || 0;
    const risk = entryF > 0 ? ((entryF - stopF) / entryF * 100) : 0;
    const reward = entryF > 0 ? ((targetF - entryF) / entryF * 100) : 0;
    const rr = risk > 0 ? reward / risk : 0;

    const labelStyle: React.CSSProperties = { fontSize: "0.75rem", fontWeight: 700, display: "block", marginBottom: 4 };
    const inputStyle: React.CSSProperties = { marginBottom: 12 };

    return (
        <div className="glass-card" style={{ padding: 24 }}>
            <h2 style={{ margin: "0 0 6px", fontSize: "1.1rem" }}>🔬 Canlı Sinyal Testi</h2>
            <p style={{ margin: "0 0 18px", fontSize: "0.8rem", color: "var(--text-muted)" }}>
                Bir sinyal gir, AI kazanma ihtimalini tahmin etsin.
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 16 }}>
                <div>
                    <label style={{ ...labelStyle, color: "var(--text-secondary)" }}>Entry Price ($)</label>
                    <input className="input" type="number" step="0.01" value={entry} onChange={e => setEntry(e.target.value)} style={inputStyle} />
                    <label style={{ ...labelStyle, color: "var(--red)" }}>🛑 Stop Loss ($)</label>
                    <input className="input" type="number" step="0.01" value={stop} onChange={e => setStop(e.target.value)} style={inputStyle} />
                    <label style={{ ...labelStyle, color: "var(--green)" }}>🎯 Target ($)</label>
                    <input className="input" type="number" step="0.01" value={target} onChange={e => setTarget(e.target.value)} style={inputStyle} />
                </div>
                <div>
                    <label style={labelStyle}>ATR</label>
                    <input className="input" type="number" step="0.05" value={atr} onChange={e => setAtr(e.target.value)} style={inputStyle} />
                    <label style={labelStyle}>Kalite Skoru (0-10)</label>
                    <input className="input" type="range" min={0} max={10} value={quality} onChange={e => setQuality(parseInt(e.target.value))} style={inputStyle} />
                    <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--accent)", marginBottom: 12 }}>{quality}/10</div>
                    <label style={labelStyle}>Swing Tipi</label>
                    <select className="input" value={swingType} onChange={e => setSwingType(e.target.value)} style={inputStyle}>
                        {["A", "B", "C", "S"].map(t => <option key={t} value={t}>{t}</option>)}
                    </select>
                </div>
                <div>
                    <label style={labelStyle}>⏱️ Max Hold (gün)</label>
                    <input className="input" type="number" min={1} max={60} value={holdDays} onChange={e => setHoldDays(parseInt(e.target.value) || 7)} style={inputStyle} />
                    <div style={{ padding: "10px 12px", background: "rgba(255,255,255,0.04)", borderRadius: 8, fontSize: "0.78rem", lineHeight: 1.6 }}>
                        <div>Risk: <span style={{ color: "var(--red)", fontWeight: 700 }}>{risk.toFixed(1)}%</span></div>
                        <div>Reward: <span style={{ color: "var(--green)", fontWeight: 700 }}>{reward.toFixed(1)}%</span></div>
                        <div>R/R: <span style={{ fontWeight: 700 }}>1:{rr.toFixed(1)}</span></div>
                    </div>
                </div>
            </div>

            <button className="btn-primary" onClick={handlePredict} disabled={loading}
                style={{ marginTop: 16, padding: "10px 24px", fontSize: "0.9rem" }}>
                {loading ? <span className="spinner" style={{ width: 14, height: 14 }} /> : "🤖"} Tahmin Al
            </button>

            {result && (
                <div style={{
                    marginTop: 16, padding: "14px 18px", borderRadius: 10, fontSize: "0.85rem",
                    background: result.success
                        ? (result.win_probability >= 0.6 ? "rgba(34,197,94,0.1)" : result.win_probability >= 0.45 ? "rgba(245,158,11,0.1)" : "rgba(239,68,68,0.1)")
                        : "rgba(239,68,68,0.1)",
                }}>
                    {result.success ? (
                        <>
                            <div style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: 6 }}>
                                {result.win_probability >= 0.6 ? "🟢" : result.win_probability >= 0.45 ? "🟡" : "🔴"}{" "}
                                Kazanma: %{Math.round(result.win_probability * 100)} — {result.confidence}
                            </div>
                            <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
                                Risk: {risk.toFixed(1)}% | Reward: {reward.toFixed(1)}% | R/R: 1:{rr.toFixed(1)}
                            </div>
                            {result.top_features && (
                                <div style={{ marginTop: 10 }}>
                                    <div style={{ fontSize: "0.78rem", fontWeight: 600, marginBottom: 4 }}>En etkili özellikler:</div>
                                    {result.top_features.slice(0, 3).map((f: { feature: string; importance: number }) => (
                                        <div key={f.feature} style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                                            {f.feature}: {"█".repeat(Math.round(f.importance * 50))} ({f.importance.toFixed(3)})
                                        </div>
                                    ))}
                                </div>
                            )}
                        </>
                    ) : (
                        <span style={{ color: "var(--red)" }}>❌ {result.error}</span>
                    )}
                </div>
            )}
        </div>
    );
}

/* ─── Main Page ───────────────────────────────────────── */
export default function ChatPage() {
    const [tab, setTab] = useState<Tab>("chat");

    return (
        <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 80px)" }}>
            <div style={{ marginBottom: 16 }}>
                <h1 className="page-title gradient-text">AI & Modeller</h1>
                <p className="page-subtitle">Strategy Chat · Haftalık Rapor · ML Model · Canlı Test</p>
            </div>

            {/* Tabs */}
            <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
                <TabBtn active={tab === "chat"} icon={<Bot size={14} />} label="Strategy Chat" onClick={() => setTab("chat")} />
                <TabBtn active={tab === "weekly"} icon={<FileText size={14} />} label="Haftalık Rapor" onClick={() => setTab("weekly")} />
                <TabBtn active={tab === "model"} icon={<Brain size={14} />} label="AI Model" onClick={() => setTab("model")} />
                <TabBtn active={tab === "predict"} icon={<Crosshair size={14} />} label="Canlı Sinyal Testi" onClick={() => setTab("predict")} />
            </div>

            {/* Tab content */}
            <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
                {tab === "chat" && <ChatTab />}
                {tab === "weekly" && <WeeklyReportTab />}
                {tab === "model" && <ModelTab />}
                {tab === "predict" && <PredictTab />}
            </div>
        </div>
    );
}
