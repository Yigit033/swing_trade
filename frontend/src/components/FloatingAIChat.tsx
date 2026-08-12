"use client";

import { useState, useRef, useEffect } from "react";
import { MessageCircle, X, Send, Bot, User, Maximize2, Minimize2 } from "lucide-react";
import { chatWithAI } from "@/lib/api";

interface Message {
    role: "user" | "ai";
    content: string;
}

export default function FloatingAIChat() {
    const [isOpen, setIsOpen] = useState(false);
    const [isExpanded, setIsExpanded] = useState(false);
    const [messages, setMessages] = useState<Message[]>([
        { role: "ai", content: "Merhaba! Ben Swing Trade Strategy asistanınım. Trade stratejilerin, risk yönetimin veya pozisyonların hakkında sorular sorabilirsin. 📊" }
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const bottomRef = useRef<HTMLDivElement>(null);

    // Otomatik aşağı kaydırma
    useEffect(() => {
        if (isOpen) {
            bottomRef.current?.scrollIntoView({ behavior: "smooth" });
        }
    }, [messages, isOpen]);

    const handleSend = async () => {
        const text = input.trim();
        if (!text || loading) return;

        setInput("");
        setMessages((prev) => [...prev, { role: "user", content: text }]);
        setLoading(true);

        const history = messages.map(m => ({ role: m.role, content: m.content }));

        try {
            const res = await chatWithAI(text, history);
            setMessages((prev) => [...prev, { role: "ai", content: res.answer || "Yanıt alınamadı." }]);
        } catch {
            setMessages((prev) => [...prev, { role: "ai", content: "❌ API bağlantısı kurulamadı." }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="floating-chat-container">
            {/* Sohbet Paneli */}
            <div className={`floating-chat-panel ${isOpen ? "open" : "closed"} ${isExpanded ? "expanded" : ""}`}>
                {/* Header */}
                <div className="floating-chat-header">
                    <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                        <div className="floating-chat-avatar">
                            <Bot size={18} color="#fff" />
                        </div>
                        <div style={{ display: "flex", flexDirection: "column" }}>
                            <span style={{ fontSize: "0.95rem", fontWeight: 600, color: "#fff", lineHeight: 1.1 }}>AI Asistan</span>
                            <span style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.7)" }}>Sana özel analiz</span>
                        </div>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                        <button onClick={() => setIsExpanded(!isExpanded)} className="floating-chat-action-btn">
                            {isExpanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
                        </button>
                        <button onClick={() => setIsOpen(false)} className="floating-chat-action-btn">
                            <X size={18} />
                        </button>
                    </div>
                </div>

                {/* Mesaj Alanı */}
                <div className="floating-chat-messages">
                    {messages.map((m, i) => (
                        <div key={i} className={`floating-chat-message-row ${m.role}`}>
                            {m.role === "ai" && (
                                <div className="floating-chat-avatar-small ai">
                                    <Bot size={14} color="#fff" />
                                </div>
                            )}
                            <div className={`floating-chat-bubble ${m.role}`}>
                                {m.content}
                            </div>
                            {m.role === "user" && (
                                <div className="floating-chat-avatar-small user">
                                    <User size={14} color="#fff" />
                                </div>
                            )}
                        </div>
                    ))}
                    {loading && (
                        <div className="floating-chat-message-row ai">
                            <div className="floating-chat-avatar-small ai">
                                <Bot size={14} color="#fff" />
                            </div>
                            <div className="floating-chat-bubble ai" style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                <span className="spinner" style={{ width: "14px", height: "14px" }} />
                                <span style={{ fontSize: "0.8rem", opacity: 0.8 }}>Düşünüyor...</span>
                            </div>
                        </div>
                    )}
                    <div ref={bottomRef} />
                </div>

                {/* Girdi Alanı */}
                <div className="floating-chat-input-area">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
                        placeholder="Bir şeyler sor..."
                        className="floating-chat-input"
                    />
                    <button
                        onClick={handleSend}
                        disabled={loading || !input.trim()}
                        className="floating-chat-send-btn"
                    >
                        <Send size={16} />
                    </button>
                </div>
            </div>

            {/* Tetikleyici Buton */}
            <button
                className={`floating-chat-trigger ${isOpen ? "hidden" : ""}`}
                onClick={() => setIsOpen(true)}
                aria-label="AI ile konuş"
            >
                <MessageCircle size={28} color="#fff" />
            </button>
        </div>
    );
}
