"""
prompts.py — Tüm LLM Prompt Şablonları

Her özellik için:
  1. SYSTEM_PROMPT: "LLM'e kim olduğunu söyle"
  2. build_*_prompt(): "Ne yapmasını istedin?" → hazır string

Her yeni özellik aynı 4 adımı izler:
  VERİ → PROMPT → LLM (client.complete) → ÇIKTI

v2.0 İyileştirmeler:
  - Türkçe karakter desteği (UTF-8)
  - Açık pozisyonlar prompt'a eklendi
  - Market regime bilgisi prompt'a eklendi
  - 300 kelime limiti kaldırıldı (soru karmaşıklığına göre uzasın)
"""

from typing import Dict


# ════════════════════════════════════════════════════
# A: HAFTALIK RAPOR
# ════════════════════════════════════════════════════

SYSTEM_PROMPT = """Sen bir profesyonel swing trading performans analistisin.
Görevin:
- Paper trading sisteminin haftalık sonuçlarını analiz etmek
- Hangi kurulumların (A/B/C tipi) daha başarılı olduğunu yorumlamak
- İyileştirme önerileri sunmak

KURALLAR:
1. Asla "şu hisseyi al/sat" gibi doğrudan yatırım tavsiyesi verme
2. Tüm analiz sağlanan veriye dayansın
3. Sade Türkçe kullan (teknik terimler İngilizce olabilir)
4. Markdown formatında yaz
5. Önerilerde somut ve ölçülebilir ol
"""


def build_weekly_report_prompt(context: Dict) -> str:
    """
    WeeklyDataCollector.collect() çıktısını haftalık rapor promptuna dönüştür.
    """
    period   = context.get("period", {})
    weekly   = context.get("weekly_summary", {})
    all_time = context.get("all_time_summary", {})
    trades   = context.get("weekly_trades", [])
    by_type  = context.get("by_swing_type", {})
    top_win  = context.get("top_win")
    top_loss = context.get("top_loss")
    open_pos = context.get("open_positions", [])
    regime   = context.get("market_regime", {})

    period_str = f"{period.get('start', '?')} - {period.get('end', '?')}"

    if trades:
        trade_lines = []
        for t in trades:
            emoji = "✅" if t["outcome"] == "WIN" else "❌"
            trade_lines.append(
                f"  [{emoji}] {t['ticker']:6} | {t['status']:8} | "
                f"P/L:{t['pnl_pct']:+.2f}% | R/R:1:{t['rr_ratio']:.1f} | "
                f"Tip:{t['swing_type']} | {t['exit_date']}"
            )
        trade_block = "\n".join(trade_lines)
    else:
        trade_block = "  (Bu dönemde kapanan trade yok)"

    type_lines = []
    for st in sorted(by_type.keys()):
        d = by_type[st]
        type_lines.append(
            f"  Tip {st}: {d['count']} trade | Win Rate:%{d['win_rate']:.0f} | Ort.P/L:{d['avg_pnl']:+.2f}%"
        )
    type_block = "\n".join(type_lines) if type_lines else "  (Veri yok)"

    extremes = ""
    if top_win:
        extremes += f"  En İyi: {top_win['ticker']} -> {top_win['pnl_pct']:+.2f}% ({top_win['status']})\n"
    if top_loss:
        extremes += f"  En Kötü: {top_loss['ticker']} -> {top_loss['pnl_pct']:+.2f}% ({top_loss['status']})"

    # Açık pozisyonlar bloğu
    open_block = _build_open_positions_block(open_pos)

    # Market regime bloğu
    regime_block = _build_regime_block(regime)

    return (
        f"Aşağıdaki paper trading verilerini analiz et ve haftalık performans raporu yaz.\n\n"
        f"DÖNEM: {period_str}\n\n"
        f"{regime_block}\n"
        f"HAFTALIK ÖZET:\n"
        f"  Toplam Trade  : {weekly.get('total', 0)}\n"
        f"  Kazanılan     : {weekly.get('wins', 0)} (%{weekly.get('win_rate', 0):.1f})\n"
        f"  Kaybedilen    : {weekly.get('losses', 0)}\n"
        f"  Toplam P/L    : {weekly.get('total_pnl_pct', 0):+.2f}%\n"
        f"  Ort. P/L/Trade: {weekly.get('avg_pnl_pct', 0):+.2f}%\n"
        f"  Ort. Kazanç   : {weekly.get('avg_win_pct', 0):+.2f}%\n"
        f"  Ort. Kayıp    : {weekly.get('avg_loss_pct', 0):+.2f}%\n"
        f"  Profit Factor : {weekly.get('profit_factor', 0):.2f}x\n\n"
        f"BU DÖNEM KAPANAN TRADE'LER:\n{trade_block}\n\n"
        f"{open_block}\n"
        f"TÜM ZAMANLARIN ÖZETİ (kıyaslama için):\n"
        f"  Toplam Trade  : {all_time.get('total', 0)}\n"
        f"  Win Rate      : %{all_time.get('win_rate', 0):.1f}\n"
        f"  Ort. P/L      : {all_time.get('avg_pnl_pct', 0):+.2f}%\n\n"
        f"SWING TİPİ BAZINDA:\n{type_block}\n\n"
        f"ÖNE ÇIKANLAR:\n{extremes if extremes else '  (Veri yok)'}\n\n"
        f"Lütfen şu başlıklar altında bir Türkçe rapor yaz:\n\n"
        f"## Haftalık Özet\n"
        f"## Bu Hafta Neler İyi Gitti?\n"
        f"## Neler İyileştirilebilir?\n"
        f"## Setup Analizi\n"
        f"## Önümüzdeki Hafta İçin 3 Öneri\n\n"
        f"Rapor 300-400 kelime olsun. Doğrudan yatırım tavsiyesi verme."
    )


# ════════════════════════════════════════════════════
# B: SİNYAL BRİFİNGİ (A özelliği)
# ════════════════════════════════════════════════════
#
# VERİ AKIŞI (adım adım):
#   1. VERİ    → scanner signal dict (entry, stop, target, atr, quality...)
#   2. PROMPT  → build_signal_briefing_prompt() bunu metin haline getirir
#   3. LLM     → client.complete() — her zaman aynı satır
#   4. ÇIKTI   → sinyal kartı altındaki bilgi kutusu
#
# Haftalık rapordan farkı: çok kısa (2-3 cümle), tek sinyal için, anlık.

SIGNAL_BRIEFING_SYSTEM = (
    "Sen bir swing trading sinyal yorumcususun.\n"
    "Görev: Verilen teknik kurulum verisini 2-3 cümleyle Türkçe değerlendir.\n"
    "KURALLAR:\n"
    "- Asla 'al' veya 'sat' deme\n"
    "- Sadece verilen sayılara bak\n"
    "- Maksimum 3 cümle\n"
    "- Türkçe yaz, teknik terimler İngilizce olabilir (ATR, R/R vs.)"
)


def build_signal_briefing_prompt(signal: Dict) -> str:
    """
    Tek bir scanner sinyali için kısa AI brifingi üreten prompt.

    Nasıl çalışır:
      scanner dict gelir -> sayılar metin haline gelir -> LLM 2-3 cümle yazar

    Args:
        signal: Scanner signal dict
                (entry_price, stop_loss, target_1, atr, quality_score, swing_type...)
    """
    entry   = signal.get("entry_price", 0) or 0
    stop    = signal.get("stop_loss", 0) or 0
    target  = signal.get("target_1") or signal.get("target", 0) or 0
    atr     = signal.get("atr", 0) or 0
    quality = signal.get("quality_score", 0) or 0
    stype   = signal.get("swing_type", "?")
    ticker  = signal.get("ticker", "?")

    # ÖNEMLİ: Bu hesaplamalar LLM yapmıyor — biz yapıp LLM'e hazır veriyoruz.
    # Bu Hybrid Architecture'ın özü: deterministik hesap + LLM yorum.
    risk_pct   = abs(entry - stop) / entry * 100 if entry else 0
    reward_pct = abs(target - entry) / entry * 100 if entry else 0
    rr_ratio   = reward_pct / risk_pct if risk_pct > 0 else 0
    atr_pct    = atr / entry * 100 if entry else 0

    rr_label  = "güçlü" if rr_ratio >= 3.0 else "orta" if rr_ratio >= 2.0 else "düşük"
    atr_label = "yüksek volatilite" if atr_pct > 5 else "normal volatilite"
    q_label   = "yüksek kaliteli" if quality >= 8 else "orta kaliteli" if quality >= 6 else "düşük kaliteli"
    return (
        f"Aşağıdaki swing trade sinyalini 2-3 cümleyle değerlendir:\n\n"
        f"HİSSE: {ticker} | Tip: {stype} | Kalite: {quality}/10 ({q_label})\n\n"
        f"TEKNİK KURULUM:\n"
        f"  Entry     : ${entry:.2f}\n"
        f"  Stop Loss : ${stop:.2f}  -> Risk: %{risk_pct:.1f}\n"
        f"  Target    : ${target:.2f} -> Kazanç Potansiyeli: %{reward_pct:.1f}\n"
        f"  R/R Oranı : 1:{rr_ratio:.1f} ({rr_label})\n"
        f"  ATR       : ${atr:.2f} (%{atr_pct:.1f} - {atr_label})\n\n"
        f"Kurulumun güçlü ve zayıf yönlerini belirt. "
        f"Doğrudan al/sat tavsiyesi verme. "
        f"3 cümleyi geçme."
    )


# ════════════════════════════════════════════════════
# C: STRATEJİ SORU-CEVAP (D özelliği)
# ════════════════════════════════════════════════════
#
# Bu "RAG-lite" mimarisi:
#   Gerçek RAG: harici belgeler vektör DB'de tutulur, sorgu anında getirilir.
#   Burada: trade geçmişi zaten küçük -> direkt prompt'a koyuyoruz.
#
# VERİ AKIŞI:
#   1. VERİ    → data_collector.collect() → tüm istatistikler
#   2. PROMPT  → kullanıcının sorusu + bu context birleştirilir
#   3. LLM     → client.complete() — aynı satır
#   4. ÇIKTI   → chat kutusunda cevap

STRATEGY_CHAT_SYSTEM = (
    "Sen, Swing Trade sisteminin profesyonel, bilge ve destekleyici baş stratejistisin (AI Mentor).\n"
    "Görevin, kullanıcının trade verilerini analiz etmenin YANI SIRA, ona genel finans, piyasa koşulları, "
    "teknik analiz ve risk yönetimi konularında üst düzey koçluk yapmaktır.\n\n"
    "KURALLAR:\n"
    "1. KİŞİLİK (PERSONA): Soğuk bir bot gibi değil; deneyimli, zeki ve motive edici bir mentor gibi konuş. "
    "Kullanıcı kâr ettiğinde başarısını pekiştir, zarar ettiğinde ise psikolojisini yönetip ders çıkarmasını sağla.\n"
    "2. VERİ KULLANIMI: Kullanıcı kendi performansı, açık pozisyonları veya geçmiş işlemleri hakkında soru sorarsa, "
    "MUTLAKA sana sağlanan sistem verilerine (sayılara, P/L'ye) dayanarak analiz yap.\n"
    "3. GENEL BİLGİ DAĞARCIĞI: Kullanıcı 'Fed faiz indirirse ne olur?', 'RSI nedir?', 'Zarar psikolojisi' gibi "
    "genel sorular sorarsa, kendi devasa bilgi dağarcığını kullanarak detaylıca cevapla. Bu tür sorularda 'Sağlanan veride bu bilgi yok' DEME.\n"
    "4. TAVSİYE SINIRI: Strateji, eğitim ve piyasa yorumu yap; ancak asla 'Şu hisseyi al/sat' şeklinde "
    "kesin yatırım danışmanlığı (YTD) yapma.\n"
    "5. FORMAT: Kusursuz, akıcı ve samimi bir Türkçe kullan. Cevaplarını kalın yazılar, maddeler ve emojilerle (gerekirse) "
    "görsel olarak zenginleştirerek (Markdown) ver."
)


def build_strategy_chat_prompt(question: str, context: Dict) -> str:
    """
    Kullanıcının sorusunu ve tüm trade geçmişini birleştirip LLM'e
    göndermek için prompt.

    VERİ AKIŞI:
      data_collector.collect() -> context dict  (Supabase/PostgreSQL istatistikler)
      kullanıcı sorusu         -> question str
      ikisi BİRLEŞTİRİLİYOR   -> tek prompt string

    Bu RAG-lite: trade geçmişi küçük olduğu için direkt prompt'a koyuyoruz.
    Gerçek RAG'da binlerce belge için vektör DB kullanılır.

    Args:
        question: "Bu hafta neden kaybettik?" gibi soru
        context:  WeeklyDataCollector.collect() çıktısı
    """
    all_s    = context.get("all_time_summary", {})
    by_type  = context.get("by_swing_type", {})
    trades   = context.get("weekly_trades", [])
    top_win  = context.get("top_win")
    top_loss = context.get("top_loss")
    open_pos = context.get("open_positions", [])
    regime   = context.get("market_regime", {})

    # Son 25 trade
    trade_lines = []
    for t in trades[:25]:
        emoji = "✅" if t["outcome"] == "WIN" else "❌"
        trade_lines.append(
            f"[{emoji}] {t['ticker']} | {t['status']} | "
            f"P/L:{t['pnl_pct']:+.1f}% | Tip:{t['swing_type']} | R/R:1:{t['rr_ratio']:.1f} | Tarih: {t['exit_date']}"
        )
    trade_block = "\n".join(trade_lines) or "(Trade yok)"

    type_lines = [
        f"Tip {st}: {d['count']} trade | %{d['win_rate']:.0f} win | {d['avg_pnl']:+.1f}% ort."
        for st, d in sorted(by_type.items())
    ]
    type_block = "\n".join(type_lines) or "(Veri yok)"

    win_ticker  = top_win["ticker"] if top_win else "?"
    win_pnl     = top_win["pnl_pct"] if top_win else 0
    win_status  = top_win["status"] if top_win else ""
    loss_ticker = top_loss["ticker"] if top_loss else "?"
    loss_pnl    = top_loss["pnl_pct"] if top_loss else 0
    loss_status = top_loss["status"] if top_loss else ""

    # Açık pozisyonlar bloğu
    open_block = _build_open_positions_block(open_pos)

    # Market regime bloğu
    regime_block = _build_regime_block(regime)

    return (
        f"SİSTEM VERİSİ (bu veriye dayanarak cevap ver):\n\n"
        f"{regime_block}\n"
        f"GENEL İSTATİSTİKLER:\n"
        f"  Toplam Trade : {all_s.get('total', 0)}\n"
        f"  Win Rate     : %{all_s.get('win_rate', 0):.1f}\n"
        f"  Ort. P/L     : {all_s.get('avg_pnl_pct', 0):+.2f}%\n"
        f"  Profit Factor: {all_s.get('profit_factor', 0):.2f}x\n\n"
        f"SETUP BAZINDA:\n{type_block}\n\n"
        f"{open_block}\n"
        f"SON 25 TRADE:\n{trade_block}\n\n"
        f"EN İYİ : {win_ticker} ({win_pnl:+.1f}% - {win_status})\n"
        f"EN KÖTÜ: {loss_ticker} ({loss_pnl:+.1f}% - {loss_status})\n\n"
        f"---\n"
        f"KULLANICININ SORUSU: {question}\n"
        f"---\n\n"
        f"Kullanıcının sorusunu yukarıdaki kurallara (Persona'na) uygun şekilde yanıtla."
    )


# ════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR (Ortak bloklar)
# ════════════════════════════════════════════════════

def _build_open_positions_block(open_pos: list) -> str:
    """Açık pozisyonları prompt'a eklenecek metin bloğuna dönüştürür."""
    if not open_pos:
        return "AÇIK POZİSYONLAR:\n  (Şu an açık pozisyon yok)\n"

    lines = ["AÇIK POZİSYONLAR:"]
    for p in open_pos:
        status_label = "⏳ PENDING" if p["status"] == "PENDING" else "🟢 OPEN"
        lines.append(
            f"  {status_label} {p['ticker']} | Tip:{p['swing_type']} | "
            f"Giriş:${p['entry_price']:.2f} | Güncel:${p['current_price']:.2f} | "
            f"P/L:{p['unrealized_pnl_pct']:+.1f}% | "
            f"Stop:${p['stop_loss']:.2f} | Hedef:${p['target']:.2f} | "
            f"Giriş Tarihi: {p['entry_date']}"
        )
    return "\n".join(lines) + "\n"


def _build_regime_block(regime: dict) -> str:
    """Market regime bilgisini prompt'a eklenecek metin bloğuna dönüştürür."""
    if not regime or regime.get("regime") == "UNKNOWN":
        return "PİYASA DURUMU: Bilinmiyor\n"

    regime_labels = {
        "BULL": "🟢 BOĞA (Yükseliş Trendi)",
        "CAUTION": "🟡 DİKKAT (Belirsiz)",
        "BEAR": "🔴 AYI (Düşüş Trendi)",
    }
    label = regime_labels.get(regime["regime"], regime["regime"])
    confidence = regime.get("confidence", "?")

    parts = [f"PİYASA DURUMU: {label} (Güven: {confidence})"]
    if regime.get("spy_price"):
        parts.append(f"  SPY: ${regime['spy_price']:.2f}")
    if regime.get("vix"):
        parts.append(f"  VIX: {regime['vix']:.1f}")
    return "\n".join(parts) + "\n"
