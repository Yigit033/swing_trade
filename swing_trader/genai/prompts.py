"""
prompts.py — Tüm LLM Prompt Şablonları

Her özellik için:
  1. SYSTEM_PROMPT: "LLM'e kim olduğunu söyle"
  2. build_*_prompt(): "Ne yapmasını istedin?" → hazır string

Her yeni özellik aynı 4 adımı izler:
  VERİ → PROMPT → LLM (client.complete) → ÇIKTI
"""

from typing import Dict


# ════════════════════════════════════════════════════
# A: HAFTALIK RAPOR
# ════════════════════════════════════════════════════

SYSTEM_PROMPT = """Sen bir profesyonel swing trading performans analistisın.
Görevin:
- Paper trading sisteminin haftalık sonuçlarını analiz etmek
- Hangi kurulumların (A/B/C/S tipi) daha başarılı olduğunu yorumlamak
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

    period_str = f"{period.get('start', '?')} - {period.get('end', '?')}"

    if trades:
        trade_lines = []
        for t in trades:
            emoji = "V" if t["outcome"] == "WIN" else "X"
            trade_lines.append(
                f"  [{emoji}] {t['ticker']:6} | {t['status']:8} | "
                f"P/L:{t['pnl_pct']:+.2f}% | R/R:1:{t['rr_ratio']:.1f} | "
                f"Tip:{t['swing_type']} | {t['exit_date']}"
            )
        trade_block = "\n".join(trade_lines)
    else:
        trade_block = "  (Bu donemde kapanan trade yok)"

    type_lines = []
    for st in sorted(by_type.keys()):
        d = by_type[st]
        type_lines.append(
            f"  Tip {st}: {d['count']} trade | Win Rate:%{d['win_rate']:.0f} | Ort.P/L:{d['avg_pnl']:+.2f}%"
        )
    type_block = "\n".join(type_lines) if type_lines else "  (Veri yok)"

    extremes = ""
    if top_win:
        extremes += f"  En Iyi: {top_win['ticker']} -> {top_win['pnl_pct']:+.2f}% ({top_win['status']})\n"
    if top_loss:
        extremes += f"  En Kotu: {top_loss['ticker']} -> {top_loss['pnl_pct']:+.2f}% ({top_loss['status']})"

    return (
        f"Asagidaki paper trading verilerini analiz et ve haftalik performans raporu yaz.\n\n"
        f"DONEM: {period_str}\n\n"
        f"HAFTALIK OZET:\n"
        f"  Toplam Trade  : {weekly.get('total', 0)}\n"
        f"  Kazanilan     : {weekly.get('wins', 0)} (%{weekly.get('win_rate', 0):.1f})\n"
        f"  Kaybedilen    : {weekly.get('losses', 0)}\n"
        f"  Toplam P/L    : {weekly.get('total_pnl_pct', 0):+.2f}%\n"
        f"  Ort. P/L/Trade: {weekly.get('avg_pnl_pct', 0):+.2f}%\n"
        f"  Ort. Kazanc   : {weekly.get('avg_win_pct', 0):+.2f}%\n"
        f"  Ort. Kayip    : {weekly.get('avg_loss_pct', 0):+.2f}%\n"
        f"  Profit Factor : {weekly.get('profit_factor', 0):.2f}x\n\n"
        f"BU DONEM KAPANAN TRADELER:\n{trade_block}\n\n"
        f"TUM ZAMANLARIN OZETI (bagbam icin):\n"
        f"  Toplam Trade  : {all_time.get('total', 0)}\n"
        f"  Win Rate      : %{all_time.get('win_rate', 0):.1f}\n"
        f"  Ort. P/L      : {all_time.get('avg_pnl_pct', 0):+.2f}%\n\n"
        f"SWING TIPI BAZINDA:\n{type_block}\n\n"
        f"ONE CIKANLAR:\n{extremes if extremes else '  (Veri yok)'}\n\n"
        f"Lutfen su basliklar altinda bir Turkce rapor yaz:\n\n"
        f"## Haftalik Ozet\n"
        f"## Bu Hafta Neler Iyi Gitti?\n"
        f"## Neler Iyilestirilebilir?\n"
        f"## Setup Analizi\n"
        f"## Onumüzdeki Hafta Icin 3 Oneri\n\n"
        f"Rapor 300-400 kelime olsun. Dogrudan yatirim tavsiyesi verme."
    )


# ════════════════════════════════════════════════════
# B: SİNYAL BRİFİNGİ (A özelliği)
# ════════════════════════════════════════════════════
#
# VERİ AKIŞI (adım adım):
#   1. VERİ    → scanner signal dict (entry, stop, target, atr, quality...)
#   2. PROMPT  → build_signal_briefing_prompt() bunu metin haline getirir
#   3. LLM     → client.complete() — her zaman aynı satır
#   4. ÇIKTI   → sinyal kartı altındaki st.info() kutusu
#
# Haftalık rapordan farkı: çok kısa (2-3 cümle), tek sinyal için, anlık.

SIGNAL_BRIEFING_SYSTEM = (
    "Sen bir swing trading sinyal yorumcususun.\n"
    "Gorev: Verilen teknik kurulum verisini 2-3 cumleyle Turkce degerlendir.\n"
    "KURALLAR:\n"
    "- Asla 'al' veya 'sat' deme\n"
    "- Sadece verilen sayilara bak\n"
    "- Maksimum 3 cumle\n"
    "- Turkce yaz, teknik terimler Ingilizce olabilir (ATR, R/R vs.)"
)


def build_signal_briefing_prompt(signal: Dict) -> str:
    """
    Tek bir scanner sinyali icin kisa AI brifingi ureten prompt.

    Nasil calisir:
      scanner dict gelir -> sayilar metin haline gelir -> LLM 2-3 cumle yazar

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

    # ONEMLI: Bu hesaplamalar LLM yapmıyor — biz yapip LLM'e hazir veriyoruz.
    # Bu Hybrid Architecture'in ozü: deterministik hesap + LLM yorum.
    risk_pct   = abs(entry - stop) / entry * 100 if entry else 0
    reward_pct = abs(target - entry) / entry * 100 if entry else 0
    rr_ratio   = reward_pct / risk_pct if risk_pct > 0 else 0
    atr_pct    = atr / entry * 100 if entry else 0

    rr_label  = "guclu" if rr_ratio >= 3.0 else "orta" if rr_ratio >= 2.0 else "dusuk"
    atr_label = "yuksek volatilite" if atr_pct > 5 else "normal volatilite"
    q_label   = "yuksek kaliteli" if quality >= 8 else "orta kaliteli" if quality >= 6 else "dusuk kaliteli"

    return (
        f"Asagidaki swing trade sinyalini 2-3 cumleyle degerlendir:\n\n"
        f"HISSE: {ticker} | Tip: {stype} | Kalite: {quality}/10 ({q_label})\n\n"
        f"TEKNIK KURULUM:\n"
        f"  Entry     : ${entry:.2f}\n"
        f"  Stop Loss : ${stop:.2f}  -> Risk: %{risk_pct:.1f}\n"
        f"  Target    : ${target:.2f} -> Kazanc Potansiyeli: %{reward_pct:.1f}\n"
        f"  R/R Orani : 1:{rr_ratio:.1f} ({rr_label})\n"
        f"  ATR       : ${atr:.2f} (%{atr_pct:.1f} - {atr_label})\n\n"
        f"Kurulumun guclu ve zayif yonlerini belirt. "
        f"Dogrudan al/sat tavsiyesi verme. "
        f"3 cumleyi gecme."
    )


# ════════════════════════════════════════════════════
# C: STRATEJİ SORU-CEVAP (D özelliği)
# ════════════════════════════════════════════════════
#
# Bu "RAG-lite" mimarisi:
#   Gercek RAG: harici belgeler vektor DB'de tutulur, sorgu aninda getirilir.
#   Burada: trade gecmisi zaten kucuk -> direkt prompt'a koyuyoruz.
#
# VERİ AKIŞI:
#   1. VERİ    → data_collector.collect() → tum istatistikler
#   2. PROMPT  → kullanicinin sorusu + bu context birlestirilir
#   3. LLM     → client.complete() — ayni satir
#   4. ÇIKTI   → chat kutusunda cevap

STRATEGY_CHAT_SYSTEM = (
    "Sen bir paper trading sistem danismanisın.\n"
    "Sana verilen trade gecmisi ve istatistiklere dayanarak soru cevapla.\n"
    "KURALLAR:\n"
    "- Sadece saglanan veriye dayan\n"
    "- Bilmiyorsan 'Bu veriden cikaramiiyorum' de\n"
    "- Dogrudan 'su hisseyi al' gibi tavsiye verme\n"
    "- Turkce ve sade konus\n"
    "- Somut sayilar kullan\n"
    "- Maksimum 200 kelime"
)


def build_strategy_chat_prompt(question: str, context: Dict) -> str:
    """
    Kullanicinin sorusunu ve tum trade gecmisini birlestirip LLM'e gondermek icin prompt.

    VERİ AKIŞI:
      data_collector.collect() -> context dict  (SQLite istatistikler)
      kullanici sorusu          -> question str
      ikisi BIRLESTIRILIYOR    -> tek prompt string

    Bu RAG-lite: trade gecmisi kucuk oldugu icin direkt prompt'a koyuyoruz.
    Gercek RAG'da binlerce belge icin vektor DB kullanilir.

    Args:
        question: "Bu hafta neden kaybettik?" gibi soru
        context:  WeeklyDataCollector.collect() ciktisi
    """
    all_s    = context.get("all_time_summary", {})
    by_type  = context.get("by_swing_type", {})
    trades   = context.get("weekly_trades", [])
    top_win  = context.get("top_win")
    top_loss = context.get("top_loss")

    # Son 10 trade (token limiti icin)
    trade_lines = []
    for t in trades[:10]:
        emoji = "V" if t["outcome"] == "WIN" else "X"
        trade_lines.append(
            f"[{emoji}] {t['ticker']} | {t['status']} | "
            f"P/L:{t['pnl_pct']:+.1f}% | Tip:{t['swing_type']} | R/R:1:{t['rr_ratio']:.1f}"
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

    return (
        f"SISTEM VERISI (bu veriye dayanarak cevap ver):\n\n"
        f"GENEL ISTATISTIKLER:\n"
        f"  Toplam Trade : {all_s.get('total', 0)}\n"
        f"  Win Rate     : %{all_s.get('win_rate', 0):.1f}\n"
        f"  Ort. P/L     : {all_s.get('avg_pnl_pct', 0):+.2f}%\n"
        f"  Profit Factor: {all_s.get('profit_factor', 0):.2f}x\n\n"
        f"SETUP BAZINDA:\n{type_block}\n\n"
        f"SON TRADELER:\n{trade_block}\n\n"
        f"EN IYI : {win_ticker} ({win_pnl:+.1f}% - {win_status})\n"
        f"EN KOTU: {loss_ticker} ({loss_pnl:+.1f}% - {loss_status})\n\n"
        f"---\n"
        f"KULLANICININ SORUSU: {question}\n"
        f"---\n\n"
        f"Yukaridaki sisteme dayanarak soruyu cevapla. "
        f"Veri yoksa 'Bu soruyu mevcut veriden cevaplayamiyorum' de."
    )
