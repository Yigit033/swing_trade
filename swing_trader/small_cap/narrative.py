"""
Signal Narrative Generator - Human-readable analysis for swing trade signals.
Generates Cuma Çevik style commentary explaining why a stock was selected.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


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
        'S': {
            'name': 'Short Squeeze',
            'tr': 'Kısa Sıkışması',
            'description': 'High short interest with volume surge - potential squeeze play',
            'tr_desc': 'Yüksek short interest ve hacim patlaması - sıkışma potansiyeli'
        },
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
    def generate_narrative(cls, signal: Dict, language: str = 'tr') -> Dict:
        """
        Generate comprehensive narrative for a signal.
        
        Args:
            signal: Signal dict from SmallCapEngine
            language: 'tr' for Turkish, 'en' for English
        
        Returns:
            Dict with narrative sections
        """
        ticker = signal.get('ticker', 'UNKNOWN')
        
        try:
            # Extract signal data
            entry = signal.get('entry_price', 0)
            stop = signal.get('stop_loss', 0)
            target = signal.get('target_1', 0)
            quality = signal.get('quality_score', 0)
            swing_type = signal.get('swing_type', 'A')
            
            volume_surge = signal.get('volume_surge', 1.0)
            atr_percent = signal.get('atr_percent', 0)
            rsi = signal.get('rsi', 50)
            five_day_return = signal.get('five_day_return', 0)
            float_millions = signal.get('float_millions', 0)
            
            short_percent = signal.get('short_percent', 0)
            sector_rs = signal.get('sector_rs_score', 0)
            is_sector_leader = signal.get('is_sector_leader', False)
            is_squeeze = signal.get('is_squeeze_candidate', False)
            
            hold_min = signal.get('hold_days_min', signal.get('expected_hold_min', 2))
            hold_max = signal.get('hold_days_max', signal.get('expected_hold_max', 5))
            
            # Calculate percentages
            stop_pct = ((entry - stop) / entry * 100) if entry > 0 else 0
            target_pct = ((target - entry) / entry * 100) if entry > 0 else 0
            
            # Generate sections based on language
            if language == 'tr':
                narrative = cls._generate_turkish(
                    ticker, entry, stop, target, stop_pct, target_pct,
                    quality, swing_type, volume_surge, atr_percent, rsi,
                    five_day_return, float_millions, short_percent, sector_rs,
                    is_sector_leader, is_squeeze, hold_min, hold_max
                )
            else:
                narrative = cls._generate_english(
                    ticker, entry, stop, target, stop_pct, target_pct,
                    quality, swing_type, volume_surge, atr_percent, rsi,
                    five_day_return, float_millions, short_percent, sector_rs,
                    is_sector_leader, is_squeeze, hold_min, hold_max
                )
            
            return narrative
            
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
        cls, ticker, entry, stop, target, stop_pct, target_pct,
        quality, swing_type, volume_surge, atr_percent, rsi,
        five_day_return, float_millions, short_percent, sector_rs,
        is_sector_leader, is_squeeze, hold_min, hold_max
    ) -> Dict:
        """Generate Turkish narrative."""
        
        type_info = cls.TYPE_DESCRIPTIONS.get(swing_type, cls.TYPE_DESCRIPTIONS['A'])
        
        # Headline
        if quality >= 75:
            quality_text = "🔥 Güçlü sinyal"
        elif quality >= 60:
            quality_text = "✅ İyi sinyal"
        else:
            quality_text = "⚠️ Takip edilebilir"
        
        headline = f"{ticker} - {type_info['tr']} ({hold_min}-{hold_max} gün) | {quality_text}"
        
        # Section 1: Why this stock?
        why_parts = []
        
        if volume_surge >= 2.0:
            why_parts.append(f"🔊 **Güçlü hacim patlaması** ({volume_surge:.1f}x ortalama)")
        elif volume_surge >= 1.5:
            why_parts.append(f"📊 Hacim ortalamanın {volume_surge:.1f} katı")
        
        if five_day_return > 20:
            why_parts.append(f"🚀 Son 5 günde **+{five_day_return:.0f}%** yükseliş")
        elif five_day_return > 10:
            why_parts.append(f"📈 Son 5 günde +{five_day_return:.0f}% yükseliş")
        elif five_day_return > 0:
            why_parts.append(f"Momentum başlangıcı (+{five_day_return:.0f}%)")
        elif five_day_return < 0:
            why_parts.append(f"Pullback girişi ({five_day_return:.0f}%)")
        
        why_text = " ".join(why_parts) if why_parts else "Teknik kriterler karşılandı."
        
        # Section 2: Technical status
        tech_parts = []
        
        # RSI interpretation
        if rsi > 80:
            tech_parts.append(f"⚠️ RSI yüksek ({rsi:.0f}) - kısa vadeli al-sat")
        elif rsi > 70:
            tech_parts.append(f"RSI {rsi:.0f} - momentum güçlü ama dikkatli ol")
        elif rsi > 50:
            tech_parts.append(f"RSI {rsi:.0f} - sağlıklı seviyede, yukarı potansiyel var")
        else:
            tech_parts.append(f"RSI {rsi:.0f} - erken giriş fırsatı")
        
        # Volatility
        if atr_percent > 10:
            tech_parts.append(f"Volatilite yüksek (%{atr_percent:.1f})")
        elif atr_percent > 6:
            tech_parts.append(f"Volatilite orta (%{atr_percent:.1f})")
        
        tech_text = " | ".join(tech_parts)
        
        # Section 3: Risk/Reward
        rr_ratio = target_pct / stop_pct if stop_pct > 0 else 0
        rr_text = f"""
📍 **Entry:** ${entry:.2f}
🛑 **Stop:** ${stop:.2f} (-%{stop_pct:.1f})
🎯 **Hedef:** ${target:.2f} (+%{target_pct:.1f})
⚖️ **Risk/Ödül:** 1:{rr_ratio:.1f}
        """.strip()
        
        # Section 4: Extra factors
        extra_parts = []
        
        if float_millions > 0:
            if float_millions <= 20:
                extra_parts.append(f"🔥 Çok sıkı float ({float_millions:.0f}M) - patlama potansiyeli")
            elif float_millions <= 50:
                extra_parts.append(f"💎 Sıkı float ({float_millions:.0f}M)")
            else:
                extra_parts.append(f"Float: {float_millions:.0f}M")
        
        if is_squeeze:
            extra_parts.append(f"🩳 **Short squeeze adayı!** (SI: %{short_percent:.1f})")
        elif short_percent > 10:
            extra_parts.append(f"Short interest: %{short_percent:.1f}")
        
        if is_sector_leader:
            extra_parts.append(f"👑 **Sektör lideri** (+{sector_rs:.0f} RS)")
        elif sector_rs > 10:
            extra_parts.append(f"Sektör performansı: +{sector_rs:.0f}")
        
        extra_text = " | ".join(extra_parts) if extra_parts else "Standart koşullar."
        
        # Section 5: Recommendation
        rec_parts = []
        rec_parts.append(f"⏱️ **{hold_min}-{hold_max} gün** hold önerisi")
        
        if swing_type == 'S':
            rec_parts.append("⚡ Hızlı hareketlere hazır ol, ani düşüşler olabilir")
        elif swing_type == 'B':
            rec_parts.append("🏃 Momentum tarafında kal, trailing stop kullan")
        elif swing_type == 'C':
            rec_parts.append("🎯 Erken girişin avantajını kullan, sabırlı ol")
        else:
            rec_parts.append("📊 Trend devamını takip et")
        
        if rsi > 75:
            rec_parts.append("⚠️ RSI yüksek - kademeli kâr al")
        
        rec_text = " | ".join(rec_parts)
        
        # Full text
        full_text = f"""
**{headline}**

📌 **Neden bu hisse?**
{why_text}

📊 **Teknik Durum:**
{tech_text}

{rr_text}

💡 **Ekstra Faktörler:**
{extra_text}

🎯 **Öneri:**
{rec_text}
        """.strip()
        
        return {
            'ticker': ticker,
            'headline': headline,
            'full_text': full_text,
            'sections': {
                'why': why_text,
                'technical': tech_text,
                'risk_reward': rr_text,
                'extras': extra_text,
                'recommendation': rec_text
            },
            'quality_emoji': '🔥' if quality >= 75 else ('✅' if quality >= 60 else '⚠️'),
            'type_name': type_info['tr']
        }
    
    @classmethod
    def _generate_english(
        cls, ticker, entry, stop, target, stop_pct, target_pct,
        quality, swing_type, volume_surge, atr_percent, rsi,
        five_day_return, float_millions, short_percent, sector_rs,
        is_sector_leader, is_squeeze, hold_min, hold_max
    ) -> Dict:
        """Generate English narrative."""
        
        type_info = cls.TYPE_DESCRIPTIONS.get(swing_type, cls.TYPE_DESCRIPTIONS['A'])
        
        # Headline
        if quality >= 75:
            quality_text = "🔥 Strong signal"
        elif quality >= 60:
            quality_text = "✅ Good signal"
        else:
            quality_text = "⚠️ Worth watching"
        
        headline = f"{ticker} - {type_info['name']} ({hold_min}-{hold_max}d) | {quality_text}"
        
        # Section 1: Why this stock?
        why_parts = []
        
        if volume_surge >= 2.0:
            why_parts.append(f"🔊 **Strong volume surge** ({volume_surge:.1f}x average)")
        elif volume_surge >= 1.5:
            why_parts.append(f"📊 Volume {volume_surge:.1f}x vs average")
        
        if five_day_return > 20:
            why_parts.append(f"🚀 **+{five_day_return:.0f}%** in 5 days")
        elif five_day_return > 10:
            why_parts.append(f"📈 +{five_day_return:.0f}% in 5 days")
        elif five_day_return > 0:
            why_parts.append(f"Momentum starting (+{five_day_return:.0f}%)")
        elif five_day_return < 0:
            why_parts.append(f"Pullback entry ({five_day_return:.0f}%)")
        
        why_text = " ".join(why_parts) if why_parts else "Technical criteria met."
        
        # Section 2: Technical status
        tech_parts = []
        
        if rsi > 80:
            tech_parts.append(f"⚠️ RSI high ({rsi:.0f}) - short-term play")
        elif rsi > 70:
            tech_parts.append(f"RSI {rsi:.0f} - strong momentum, be careful")
        elif rsi > 50:
            tech_parts.append(f"RSI {rsi:.0f} - healthy level, upside potential")
        else:
            tech_parts.append(f"RSI {rsi:.0f} - early entry opportunity")
        
        if atr_percent > 10:
            tech_parts.append(f"High volatility ({atr_percent:.1f}%)")
        elif atr_percent > 6:
            tech_parts.append(f"Medium volatility ({atr_percent:.1f}%)")
        
        tech_text = " | ".join(tech_parts)
        
        # Section 3: Risk/Reward
        rr_ratio = target_pct / stop_pct if stop_pct > 0 else 0
        rr_text = f"""
📍 **Entry:** ${entry:.2f}
🛑 **Stop:** ${stop:.2f} (-{stop_pct:.1f}%)
🎯 **Target:** ${target:.2f} (+{target_pct:.1f}%)
⚖️ **Risk/Reward:** 1:{rr_ratio:.1f}
        """.strip()
        
        # Section 4: Extra factors
        extra_parts = []
        
        if float_millions > 0:
            if float_millions <= 20:
                extra_parts.append(f"🔥 Very tight float ({float_millions:.0f}M) - explosive potential")
            elif float_millions <= 50:
                extra_parts.append(f"💎 Tight float ({float_millions:.0f}M)")
            else:
                extra_parts.append(f"Float: {float_millions:.0f}M")
        
        if is_squeeze:
            extra_parts.append(f"🩳 **Short squeeze candidate!** (SI: {short_percent:.1f}%)")
        elif short_percent > 10:
            extra_parts.append(f"Short interest: {short_percent:.1f}%")
        
        if is_sector_leader:
            extra_parts.append(f"👑 **Sector leader** (+{sector_rs:.0f} RS)")
        elif sector_rs > 10:
            extra_parts.append(f"Sector strength: +{sector_rs:.0f}")
        
        extra_text = " | ".join(extra_parts) if extra_parts else "Standard conditions."
        
        # Section 5: Recommendation
        rec_parts = []
        rec_parts.append(f"⏱️ **{hold_min}-{hold_max} day** hold recommendation")
        
        if swing_type == 'S':
            rec_parts.append("⚡ Be ready for fast moves, sudden drops possible")
        elif swing_type == 'B':
            rec_parts.append("🏃 Stay on momentum side, use trailing stop")
        elif swing_type == 'C':
            rec_parts.append("🎯 Leverage early entry advantage, be patient")
        else:
            rec_parts.append("📊 Follow trend continuation")
        
        if rsi > 75:
            rec_parts.append("⚠️ RSI high - consider scaling out")
        
        rec_text = " | ".join(rec_parts)
        
        # Full text
        full_text = f"""
**{headline}**

📌 **Why this stock?**
{why_text}

📊 **Technical Status:**
{tech_text}

{rr_text}

💡 **Extra Factors:**
{extra_text}

🎯 **Recommendation:**
{rec_text}
        """.strip()
        
        return {
            'ticker': ticker,
            'headline': headline,
            'full_text': full_text,
            'sections': {
                'why': why_text,
                'technical': tech_text,
                'risk_reward': rr_text,
                'extras': extra_text,
                'recommendation': rec_text
            },
            'quality_emoji': '🔥' if quality >= 75 else ('✅' if quality >= 60 else '⚠️'),
            'type_name': type_info['name']
        }


def generate_signal_narrative(signal: Dict, language: str = 'tr') -> Dict:
    """
    Convenience function to generate narrative for a signal.
    
    Usage:
        from swing_trader.small_cap.narrative import generate_signal_narrative
        narrative = generate_signal_narrative(signal, language='tr')
        print(narrative['full_text'])
    """
    return SignalNarrative.generate_narrative(signal, language)
