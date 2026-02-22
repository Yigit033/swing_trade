"""
prompts.py — LLM Prompt Şablonları

NEDEN PROMPT MÜHENDİSLİĞİ ÖNEMLİ?
────────────────────────────────────
LLM'den kaliteli çıktı almak için onu doğru yönlendirmen gerekir.
"Ne sorduğun" kadar "nasıl sorduğun" da önemlidir.

İyi bir prompt:
  ✅ Net bir rol tanımı içerir (system prompt)
  ✅ Yapılandırılmış veri sunar (JSON/tablolar)
  ✅ Beklenen formatı belirtir
  ✅ Kısıtları açıklar ("alım satım tavsiyesi verme")
"""

from typing import Dict


# ─────────────────────────────────────────────────────
# System Prompt — LLM'e kim olduğunu tanımla
# ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """Sen bir profesyonel swing trading performans analistisın.

Görevin:
- Paper trading sisteminin haftalık sonuçlarını analiz etmek
- Hangi kurulumların (A/B/C/S tipi) daha başarılı olduğunu yorumlamak
- İyileştirme önerileri sunmak

KURALLAR (çiğneme):
1. Asla "şu hisseyi al/sat" gibi doğrudan yatırım tavsiyesi verme
2. Tüm analizin sağlanan veriye dayansın, tahmin yapma
3. Sade ve net Türkçe kullan (teknik terimler için İngilizce kabul edilir)
4. Markdown formatında yaz (##, **, - )
5. Her öneride somut ve ölçülebilir ol ("stop'u biraz geniş tut" DEĞİL, "stop'u 1.5×ATR olarak ayarla")
"""

# ─────────────────────────────────────────────────────
# Haftalık Rapor Prompt Builder
# ─────────────────────────────────────────────────────

def build_weekly_report_prompt(context: Dict) -> str:
    """
    data_collector.py'ın ürettiği context dict'ini
    LLM'e gönderilecek prompt'a dönüştür.
    
    Args:
        context: WeeklyDataCollector.collect() çıktısı
    
    Returns:
        Hazır prompt string
    """
    period   = context.get("period", {})
    weekly   = context.get("weekly_summary", {})
    all_time = context.get("all_time_summary", {})
    trades   = context.get("weekly_trades", [])
    by_type  = context.get("by_swing_type", {})
    top_win  = context.get("top_win")
    top_loss = context.get("top_loss")

    # Dönem başlığı
    period_str = f"{period.get('start', '?')} → {period.get('end', '?')}"

    # Haftalık trade listesi
    if trades:
        trade_lines = []
        for t in trades:
            emoji = "✅" if t["outcome"] == "WIN" else "❌"
            trade_lines.append(
                f"  {emoji} {t['ticker']:6} | {t['status']:8} | "
                f"P/L: {t['pnl_pct']:+.2f}% | "
                f"R/R: 1:{t['rr_ratio']:.1f} | "
                f"Tip: {t['swing_type']} | "
                f"Çıkış: {t['exit_date']}"
            )
        trade_block = "\n".join(trade_lines)
    else:
        trade_block = "  (Bu dönemde kapanan trade yok)"

    # Swing type özeti
    type_lines = []
    for st in sorted(by_type.keys()):
        d = by_type[st]
        type_lines.append(
            f"  Tip {st}: {d['count']} trade | "
            f"Win Rate: %{d['win_rate']:.0f} | "
            f"Ort. P/L: {d['avg_pnl']:+.2f}%"
        )
    type_block = "\n".join(type_lines) if type_lines else "  (Veri yok)"

    # Öne çıkan tradeler
    extremes_block = ""
    if top_win:
        extremes_block += f"  🏆 En İyi: {top_win['ticker']} → {top_win['pnl_pct']:+.2f}% ({top_win['status']})\n"
    if top_loss:
        extremes_block += f"  🔴 En Kötü: {top_loss['ticker']} → {top_loss['pnl_pct']:+.2f}% ({top_loss['status']})"

    prompt = f"""Aşağıdaki paper trading verilerini analiz et ve haftalık performans raporu yaz.

═══════════════════════════════════════════════
📅 DÖNEM: {period_str}
═══════════════════════════════════════════════

📊 HAFTALIK ÖZET:
  Toplam Trade  : {weekly.get('total', 0)}
  Kazanılan     : {weekly.get('wins', 0)} (%{weekly.get('win_rate', 0):.1f})
  Kaybedilen    : {weekly.get('losses', 0)}
  Toplam P/L    : {weekly.get('total_pnl_pct', 0):+.2f}%
  Ort. P/L/Trade: {weekly.get('avg_pnl_pct', 0):+.2f}%
  Ort. Kazanç   : {weekly.get('avg_win_pct', 0):+.2f}%
  Ort. Kayıp    : {weekly.get('avg_loss_pct', 0):+.2f}%
  Profit Factor : {weekly.get('profit_factor', 0):.2f}x

📋 BU DÖNEM KAPANAN TRADELER:
{trade_block}

📈 TÜM ZAMANLARIN ÖZETI (bağlam için):
  Toplam Trade  : {all_time.get('total', 0)}
  Win Rate      : %{all_time.get('win_rate', 0):.1f}
  Ort. P/L      : {all_time.get('avg_pnl_pct', 0):+.2f}%

🎯 SWİNG TİPİ BAZINDA (tüm zaman):
{type_block}

⭐ ÖNE ÇIKANLAR:
{extremes_block if extremes_block else "  (Veri yok)"}

═══════════════════════════════════════════════

Lütfen şu başlıkları içeren bir Türkçe rapor yaz:

## 📊 Haftalık Özet
(2-3 cümleyle genel tablo)

## ✅ Bu Hafta Neler İyi Gitti?
(Varsa başarılı kurulumlar ve nedenleri)

## ⚠️ Neler İyileştirilebilir?
(Kayıpların analizi, tekrarlayan hatalar)

## 🎯 Setup Analizi
(Hangi swing tipi (A/B/C/S) daha iyi performans gösterdi ve neden?)

## 💡 Önümüzdeki Hafta İçin 3 Öneri
(Somut, ölçülebilir öneriler — "risk yönetimini iyileştir" değil, spesifik ol)

Raporun 300-400 kelime arasında olsun. Doğrudan yatırım tavsiyesi verme.
"""
    return prompt
