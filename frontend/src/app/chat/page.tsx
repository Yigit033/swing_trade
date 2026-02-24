"use client";
import { useState, useRef, useEffect } from "react";
import { chatWithAI } from "@/lib/api";
import { Send, Bot, User, MessageSquare } from "lucide-react";

interface Message {
    role: "user" | "ai";
    content: string;
}

const QUICK_PROMPTS = [
    "Hangi sektörlerde fırsat var?",
    "Win rate'imi nasıl artırabilirim?",
    "Risk yönetimi için önerilerin?",
    "Bu hafta kaç trade açmalıyım?",
];

export default function ChatPage() {
    const [messages, setMessages] = useState<Message[]>([
        { role: "ai", content: "Merhaba! Ben Swing Trade AI asistanınım. Trade stratejilerin, risk yönetimin veya mevcut pozisyonların hakkında sorular sorabilirsin. Geçmiş trade verilerin bağlamında cevap vereceğim. 📊" }
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

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
            setMessages(prev => [...prev, { role: "ai", content: "❌ API bağlantısı kurulamadı. Sunucunun çalıştığından emin ol." }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 80px)" }}>
            {/* Header */}
            <div style={{ marginBottom: 20 }}>
                <h1 className="page-title gradient-text">AI Strategy Chat</h1>
                <p className="page-subtitle">Gemini AI · Trade history context aware</p>
            </div>

            {/* Quick prompts */}
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
                {QUICK_PROMPTS.map(p => (
                    <button key={p} className="btn-secondary" style={{ fontSize: "0.78rem", padding: "5px 12px" }}
                        onClick={() => send(p)}>
                        {p}
                    </button>
                ))}
            </div>

            {/* Chat area */}
            <div className="glass-card" style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
                {/* Messages */}
                <div style={{ flex: 1, overflowY: "auto", padding: "20px 22px", display: "flex", flexDirection: "column", gap: 14 }}>
                    {messages.map((m, i) => (
                        <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start", flexDirection: m.role === "user" ? "row-reverse" : "row" }}>
                            {/* Avatar */}
                            <div style={{
                                width: 32, height: 32, borderRadius: "50%", flexShrink: 0,
                                background: m.role === "user"
                                    ? "linear-gradient(135deg, #3b82f6, #8b5cf6)"
                                    : "linear-gradient(135deg, #10b981, #3b82f6)",
                                display: "flex", alignItems: "center", justifyContent: "center",
                            }}>
                                {m.role === "user" ? <User size={16} color="#fff" /> : <Bot size={16} color="#fff" />}
                            </div>

                            {/* Bubble */}
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

                {/* Input */}
                <div style={{ padding: "14px 18px", borderTop: "1px solid var(--border)", display: "flex", gap: 10 }}>
                    <input
                        className="input"
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={e => e.key === "Enter" && !e.shiftKey && send()}
                        placeholder="Ask about strategy, risk management, trade ideas..."
                    />
                    <button className="btn-primary" onClick={() => send()} disabled={loading || !input.trim()}
                        style={{ padding: "9px 16px", flexShrink: 0 }}>
                        <Send size={15} />
                    </button>
                </div>
            </div>
        </div>
    );
}
