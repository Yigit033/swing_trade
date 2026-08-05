"""
Signal Narrative Generator - Human-readable analysis for swing trade signals.
Generates Cuma Çevik style commentary explaining why a stock was selected.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


def _t1_frac_pct() -> float:
    """T1'de satılan pozisyon yüzdesi — AYARDAN okunur.

    2026-08-04: metin "pozisyonun yarısını sat" diye sabitti; T1 oranı ölçümle
    %50'den %33'e indirilince açıklama YANLIŞ bilgi vermeye başladı. Kullanıcıya
    yanlış talimat vermek, hiç açıklama vermemekten kötüdür.
    """
    try:
        from .settings_config import load_settings

        return float(load_settings().partial_at_t1_fraction) * 100
    except Exception:
        return 33.0


class SignalNarrative:
    """
    Generate human-readable narrative analysis for swing trade signals.
    
    Output Format:
    - Why this stock? (Volume, trend, momentum)
    - Technical status (RSI, MA, 5-day return)
    - Risk/Reward (Entry, stop, target with percentages)
    - Extra factors (Float, Short Interest, Sector RS)
    - Recommendation (Hold duration, warnings)
    """
    
    # Swing type descriptions
    TYPE_DESCRIPTIONS = {
        'C': {
            'name': 'Early Stage',
            'tr': 'Erken Aşama',
            'description': 'Early momentum building - catching the move early',
            'tr_desc': 'Erken momentum oluşumu - hareketi erken yakalama'
        },
        'B': {
            'name': 'Momentum',
            'tr': 'Momentum',
            'description': 'Strong momentum breakout - ride the wave',
            'tr_desc': 'Güçlü momentum kırılımı - dalgayı sür'
        },
        'A': {
            'name': 'Continuation',
            'tr': 'Devam',
            'description': 'Established trend continuation - steady gains',
            'tr_desc': 'Kurulmuş trend devamı - istikrarlı kazanç'
        }
    }
    
    @classmethod
    def generate_narrative(cls, signal: Dict) -> Dict:
        """
        Generate comprehensive narrative for a signal.
        
        Args:
            signal: Signal dict from SmallCapEngine
        
        Returns:
            Dict with narrative sections
        """
        ticker = signal.get('ticker', 'UNKNOWN')
        
        try:
            # Extract signal data
            entry = signal.get('entry_price', 0)
            stop = signal.get('stop_loss', 0)
            target_1 = signal.get('target_1', 0)
            target_2 = signal.get('target_2', 0)
            quality = signal.get('quality_score', 0)
            swing_type = signal.get('swing_type', 'A')
            
            volume_surge = signal.get('volume_surge', 1.0)
            atr_percent = signal.get('atr_percent', 0)
            rsi = signal.get('rsi', 50)
            five_day_return = signal.get('five_day_return', 0)
            float_millions = signal.get('float_millions', 0)
            
            sector_rs = signal.get('sector_rs_score', 0)
            is_sector_leader = signal.get('is_sector_leader', False)
            
            hold_min = signal.get('hold_days_min', signal.get('expected_hold_min', 2))
            hold_max = signal.get('hold_days_max', signal.get('expected_hold_max', 5))
            
            # Calculate percentages
            stop_pct = ((entry - stop) / entry * 100) if entry > 0 else 0
            t1_pct = ((target_1 - entry) / entry * 100) if entry > 0 else 0
            t2_pct = ((target_2 - entry) / entry * 100) if entry > 0 and target_2 > 0 else 0
            
            # Technical levels (may be None if not calculated)
            tech_levels = signal.get('technical_levels', None)
            
            # MACD and divergence info
            macd_bullish = signal.get('macd_bullish', False)
            rsi_divergence = signal.get('rsi_divergence', False)
            higher_lows = signal.get('higher_lows', False)
            # Skoru gercekten kaydiran iki VCE isareti (+8 / +5)
            vce_premium = signal.get('vce_premium', False)
            vce_tight_coil = signal.get('vce_tight_coil', False)
            
            return cls._generate_turkish(
                ticker, entry, stop, target_1, target_2,
                stop_pct, t1_pct, t2_pct,
                quality, swing_type, volume_surge, atr_percent, rsi,
                five_day_return, float_millions, sector_rs,
                is_sector_leader, hold_min, hold_max,
                tech_levels, macd_bullish, rsi_divergence, higher_lows,
                vce_premium, vce_tight_coil
            )
            
        except Exception as e:
            logger.error(f"Error generating narrative for {ticker}: {e}")
            return {
                'ticker': ticker,
                'headline': f"{ticker} - Analiz hatası",
                'full_text': "Analiz oluşturulamadı.",
                'sections': {}
            }
    
    @classmethod
    def _generate_turkish(
        cls, ticker, entry, stop, target_1, target_2,
        stop_pct, t1_pct, t2_pct,
        quality, swing_type, volume_surge, atr_percent, rsi,
        five_day_return, float_millions, sector_rs,
        is_sector_leader, hold_min, hold_max,
        tech_levels, macd_bullish, rsi_divergence, higher_lows,
        vce_premium=False, vce_tight_coil=False
    ) -> Dict:
        """Generate professional Turkish narrative (Cuma Çevik style)."""
        
        type_info = cls.TYPE_DESCRIPTIONS.get(swing_type, cls.TYPE_DESCRIPTIONS['A'])
        
        # ── HEADLINE ──
        if quality >= 75:
            quality_text = "🔥 Güçlü sinyal"
        elif quality >= 60:
            quality_text = "✅ İyi sinyal"
        else:
            quality_text = "⚠️ Takip edilebilir"
        
        headline = f"{ticker} - {type_info['tr']} ({hold_min}-{hold_max} gün) | {quality_text}"
        
        # ── SECTION 1: SETUP AÇIKLAMASI (Trendline + Volume + Pattern) ──
        setup_parts = []
        
        # Trendline break detection
        trendline = tech_levels.get('trendline', {}) if tech_levels else {}
        trendline_desc = trendline.get('description', '')
        
        if trendline_desc == 'düşen_trend_kırıldı':
            break_pct = trendline.get('break_pct', 0)
            setup_parts.append(f"📐 **Düşen trendi yukarı kırmış** (+%{break_pct:.1f} üstünde).")
        elif trendline_desc == 'yükselen_trend':
            setup_parts.append("📈 Yükselen trend devam ediyor.")
        
        # Volume pattern
        vol_pattern = tech_levels.get('volume_pattern', '') if tech_levels else ''
        if vol_pattern:
            setup_parts.append(f"🔊 {vol_pattern}")
        elif volume_surge >= 2.0:
            setup_parts.append(f"🔊 Güçlü hacim patlaması ({volume_surge:.1f}x ortalama).")
        elif volume_surge >= 1.5:
            setup_parts.append(f"📊 Hacim ortalamanın {volume_surge:.1f} katı.")
        
        # Type-specific setup description
        if swing_type == 'C':
            if rsi < 45:
                setup_parts.append("⭐ Erken aşama — dipten dönüş sinyali. Hacim yeni artmaya başlamış.")
            else:
                setup_parts.append("⭐ Erken momentum oluşumu. Henüz kalabalık bu hisseyi keşfetmedi.")
        elif swing_type == 'B':
            setup_parts.append(f"🚀 Güçlü momentum hareketiyle gelen hisse. Son 5 günde +%{five_day_return:.0f} yükselmiş.")
        else:
            if higher_lows:
                setup_parts.append("🐢 Higher lows yaparak trend devam ediyor.")
            else:
                setup_parts.append("🐢 Trend devam formasyonu.")
        
        # Technical confirmations
        tech_confirms = []
        if macd_bullish:
            tech_confirms.append("MACD bullish cross")
        if rsi_divergence:
            tech_confirms.append("RSI divergence")
        if higher_lows:
            tech_confirms.append("higher lows")
        if tech_confirms:
            setup_parts.append(f"✅ Teknik onaylar: {', '.join(tech_confirms)}.")
        
        setup_text = " ".join(setup_parts) if setup_parts else type_info.get('tr_desc', 'Teknik kriterler karşılandı.')
        
        # ── SECTION 2: FİYAT SEVİYELERİ VE KOŞULLU HEDEFLER ──
        levels_parts = []
        
        # Nearest resistance — conditional target
        nearest_res = tech_levels.get('nearest_resistance') if tech_levels else None
        nearest_res_pct = tech_levels.get('nearest_resistance_pct', 0) if tech_levels else 0
        
        if nearest_res and nearest_res_pct > 0:
            if nearest_res < target_1:
                # Resistance is before T1 — mention it as intermediate level
                levels_parts.append(
                    f"📍 **${nearest_res:.2f}** seviyesi ilk karşılaşacağı direnç (+%{nearest_res_pct:.1f}). "
                    f"Burayı yukarı kırarsa T1: **${target_1:.2f}** (+%{t1_pct:.1f}) hedeflenebilir."
                )
            else:
                # Resistance is at/above T1
                levels_parts.append(
                    f"📍 **${nearest_res:.2f}** seviyesi güçlü direnç (+%{nearest_res_pct:.1f})."
                )
            
            # All resistance levels for context
            all_res = tech_levels.get('resistance_levels', []) if tech_levels else []
            if len(all_res) >= 2:
                res_prices = [f"${r['price']:.2f}" for r in all_res[:3]]
                levels_parts.append(f"Direnç seviyeleri sırasıyla: {', '.join(res_prices)}.")
        
        # T1 → T2 conditional flow
        if target_1 > 0 and target_2 > 0:
            levels_parts.append(
                f"🎯 **${target_1:.2f}** doları yukarı kırarsa **${target_2:.2f}'ye** (+%{t2_pct:.1f}) gidebilir."
            )
        
        # Support level
        nearest_sup = tech_levels.get('nearest_support') if tech_levels else None
        nearest_sup_pct = tech_levels.get('nearest_support_pct', 0) if tech_levels else 0
        
        if nearest_sup and nearest_sup_pct < 0:
            levels_parts.append(
                f"Destek: **${nearest_sup:.2f}** ({nearest_sup_pct:+.1f}%)."
            )
        
        levels_text = " ".join(levels_parts) if levels_parts else ""
        
        # ── SECTION 3: ENTRY / STOP / HEDEFLER ──
        rr_ratio_t1 = t1_pct / stop_pct if stop_pct > 0 else 0
        rr_ratio_t2 = t2_pct / stop_pct if stop_pct > 0 else 0
        
        rr_lines = [
            f"📍 **Entry:** ${entry:.2f}",
            f"🛑 **Stop:** ${stop:.2f} (-%{stop_pct:.1f})",
            f"🎯 **T1:** ${target_1:.2f} (+%{t1_pct:.1f}) — pozisyonun %{_t1_frac_pct():.0f}'ini sat",
        ]
        if target_2 > 0:
            rr_lines.append(f"🎯 **T2:** ${target_2:.2f} (+%{t2_pct:.1f}) — kalanı sat veya trail")
        rr_lines.append(f"⚖️ **Risk/Ödül:** T1 → 1:{rr_ratio_t1:.1f} | T2 → 1:{rr_ratio_t2:.1f}")
        
        rr_text = "\n".join(rr_lines)
        
        # ── SECTION 4: EKSTRA FAKTÖRLER ──
        extra_parts = []
        
        if float_millions > 0:
            if float_millions <= 20:
                extra_parts.append(f"🔥 Çok sıkı float ({float_millions:.0f}M) — patlama potansiyeli yüksek")
            elif float_millions <= 50:
                extra_parts.append(f"💎 Sıkı float ({float_millions:.0f}M)")
        
        if is_sector_leader:
            extra_parts.append(f"👑 **Sektör lideri** (+{sector_rs:.0f} RS)")
        elif sector_rs > 10:
            extra_parts.append(f"Sektör performansı: +{sector_rs:.0f}")

        # VCE kalite işaretleri — skoru gerçekten kaydıran iki şey. Kullanıcının
        # "bu neden listenin başında?" sorusunun cevabı bunlar, o yüzden burada.
        if vce_premium:
            extra_parts.append(
                "⭐ **Premium sıkışma** — kırılım hacim onaylı (≥1.5x) ve "
                "kapanış günün üst %40'ında (+8 kalite)"
            )
        if vce_tight_coil:
            extra_parts.append(
                "🌀 **Sıkı yay** — oynaklık baz seviyenin %65 altına sıkışmış; "
                "bu, dışörneklemde ayakta kalan tek seçim özelliği (+5 kalite)"
            )

        extra_text = " | ".join(extra_parts) if extra_parts else ""
        
        # ── SECTION 5: ÖNERİ VE UYARILAR ──
        rec_parts = []
        rec_parts.append(f"⏱️ **{hold_min}-{hold_max} gün** hold önerisi")
        
        if swing_type == 'B':
            rec_parts.append("🏃 Momentum tarafında kal, trailing stop ile kâr koru")
        elif swing_type == 'C':
            rec_parts.append("🎯 Erken girişin avantajını kullan, sabırlı ol")
        else:
            rec_parts.append("📊 Trend devamını takip et")
        
        # Risk warnings
        if atr_percent > 10:
            rec_parts.append(f"⚠️ Volatilite yüksek (%{atr_percent:.1f}) — riskli hisse olduğunu unutma")
        if rsi > 75:
            rec_parts.append("⚠️ RSI yüksek — kademeli kâr al, tamamını tutma")
        if float_millions > 0 and float_millions <= 15:
            rec_parts.append("⚠️ Çok düşük float — spread geniş olabilir, limit emir kullan")
        
        rec_text = " | ".join(rec_parts)
        
        # ── RSI / TEKNIK DURUM ──
        tech_parts = []
        if rsi > 80:
            tech_parts.append(f"⚠️ RSI aşırı alım ({rsi:.0f})")
        elif rsi > 70:
            tech_parts.append(f"RSI {rsi:.0f} — momentum güçlü ama dikkat")
        elif rsi > 50:
            tech_parts.append(f"RSI {rsi:.0f} — sağlıklı seviye")
        else:
            tech_parts.append(f"RSI {rsi:.0f} — erken giriş fırsatı")
        
        if atr_percent > 8:
            tech_parts.append(f"Volatilite: %{atr_percent:.1f}")
        
        tech_text = " | ".join(tech_parts)
        
        # ── FULL TEXT (Cuma Çevik style) ──
        sections = []
        sections.append(f"**{headline}**")
        sections.append("")
        sections.append(f"📌 **Setup:**")
        sections.append(setup_text)
        
        if levels_text:
            sections.append("")
            sections.append(f"📊 **Fiyat Seviyeleri:**")
            sections.append(levels_text)
        
        sections.append("")
        sections.append(rr_text)
        
        if tech_text:
            sections.append("")
            sections.append(f"📈 **Teknik:** {tech_text}")
        
        if extra_text:
            sections.append("")
            sections.append(f"💡 **Ekstra:** {extra_text}")
        
        sections.append("")
        sections.append(f"🎯 **Öneri:** {rec_text}")
        
        full_text = "\n".join(sections)
        
        return {
            'ticker': ticker,
            'headline': headline,
            'full_text': full_text,
            'sections': {
                'setup': setup_text,
                'levels': levels_text,
                'risk_reward': rr_text,
                'technical': tech_text,
                'extras': extra_text,
                'recommendation': rec_text
            },
            'quality_emoji': '🔥' if quality >= 75 else ('✅' if quality >= 60 else '⚠️'),
            'type_name': type_info['tr']
        }
    
def generate_signal_narrative(signal: Dict) -> Dict:
    """
    Convenience function to generate narrative for a signal.
    
    Usage:
        from swing_trader.small_cap.narrative import generate_signal_narrative
        narrative = generate_signal_narrative(signal)
        print(narrative['full_text'])
    """
    return SignalNarrative.generate_narrative(signal)
