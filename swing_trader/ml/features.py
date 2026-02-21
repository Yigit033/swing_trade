"""
features.py — Feature Engineering Modülü

Bu dosya, ham paper trade verisini ML modeli için
sayısal özelliklere (features) dönüştürür.

─────────────────────────────────────────────────
TEMEL KAVRAM: Feature Engineering
─────────────────────────────────────────────────
ML modeli ham veri göremez. Örneğin "entry_price = 12.50"
tek başına anlamsız. Ama "stop %7 aşağıda, target %18 yukarıda"
gibi ORANLAR modele anlam taşır. İşte feature engineering budur.
─────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────────

# Hangi exit statüsleri "WIN" sayılır?
WIN_STATUSES = {"TARGET", "TRAILED"}

# Model için kullanılacak feature sütunları
# (bunları değiştirirsen predictor.py ve trainer.py da etkilenir)
FEATURE_COLUMNS = [
    "risk_pct",           # Entry'den stop'a % mesafe (risk)
    "reward_pct",         # Entry'den target'a % mesafe (potansiyel kâr)
    "risk_reward_ratio",  # reward / risk oranı (ne kadar iyi setup?)
    "atr_pct",            # ATR / entry_price × 100 (volatilite)
    "quality_score",      # Sistemin hesapladığı sinyal kalite puanı
    "swing_type_enc",     # A/B/C/S → 0/1/2/3 (sayısal encoding)
    "max_hold_days",      # Max bekletme süresi
    "day_of_week",        # Giriş günü (0=Pzt, 4=Cum) — haftalık patern
    "month",              # Giriş ayı (1-12) — mevsimsel etki
]

# Swing type encoding haritası
SWING_TYPE_MAP = {"A": 0, "B": 1, "C": 2, "S": 3}


class FeatureEngineer:
    """
    Ham paper trade verisini ML feature'larına dönüştürür.
    
    Kullanım:
        engineer = FeatureEngineer()
        X, y = engineer.transform(closed_trades_list)
    """

    def transform(
        self, trades: List[Dict]
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Trade listesini feature matrix'e (X) ve label vector'e (y) çevirir.
        
        Args:
            trades: storage.get_closed_trades() çıktısı (dict listesi)
        
        Returns:
            (X, y) tuple:
                X → feature matrix (DataFrame), her satır bir trade
                y → label vector (Series): 1=WIN, 0=LOSS
                    (eğer trades kapatılmamışsa y = None döner)
        """
        if not trades:
            logger.warning("Boş trade listesi. Feature üretilemiyor.")
            return pd.DataFrame(columns=FEATURE_COLUMNS), None

        rows = []
        labels = []

        for trade in trades:
            try:
                row = self._extract_features(trade)
                rows.append(row)

                # ── Label (Hedef Değer) ──────────────────────────────
                # Sadece kapatılmış tradelerde label var
                status = trade.get("status", "")
                if status:
                    # WIN_STATUSES'ten biriyse 1 (kazandı), değilse 0 (kaybetti)
                    labels.append(1 if status in WIN_STATUSES else 0)

            except Exception as e:
                logger.warning(f"Trade feature çıkarma hatası (id={trade.get('id')}): {e}")
                continue

        X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

        # Eksik değerleri 0 ile doldur (NaN modeli bozar)
        X = X.fillna(0)

        y = pd.Series(labels, name="label") if labels else None

        logger.info(
            f"Feature engineering tamamlandı: {len(X)} trade, "
            f"{len(FEATURE_COLUMNS)} özellik"
            + (f", WIN rate: {sum(labels)/len(labels)*100:.1f}%" if labels else "")
        )

        return X, y

    def transform_signal(self, signal: Dict) -> pd.DataFrame:
        """
        Tek bir sinyali (henüz girilmemiş trade) feature'a çevirir.
        Predictor.predict() tarafından kullanılır.
        
        Args:
            signal: Scanner'dan gelen sinyal dict'i
        
        Returns:
            Tek satırlık DataFrame (modele verilecek veri)
        """
        row = self._extract_features(signal)
        X = pd.DataFrame([row], columns=FEATURE_COLUMNS).fillna(0)
        return X

    # ─────────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────────

    def _extract_features(self, trade: Dict) -> List:
        """
        Tek bir trade dict'inden özellik listesi üretir.
        
        NOT: Tüm hesaplamalar ORANSAL (%) — böylece farklı
        fiyat seviyesindeki hisseler karşılaştırılabilir olur.
        """
        # ── Temel fiyatlar ──────────────────────────────────────────
        entry = float(trade.get("entry_price") or trade.get("signal_price") or 1)
        stop = float(trade.get("stop_loss") or 0)
        target = float(trade.get("target") or 0)
        atr = float(trade.get("atr") or 0)

        # ── Risk ve Reward (%) ──────────────────────────────────────
        # risk_pct: Stop ne kadar aşağıda? (pozitif sayı = risk miktarı)
        # Örnek: entry=10, stop=9 → risk = (10-9)/10 × 100 = %10
        risk_pct = ((entry - stop) / entry * 100) if entry > 0 and stop > 0 else 0

        # reward_pct: Target ne kadar yukarıda?
        # Örnek: entry=10, target=13 → reward = (13-10)/10 × 100 = %30
        reward_pct = ((target - entry) / entry * 100) if entry > 0 and target > 0 else 0

        # risk_reward_ratio: İdeal setup'ta bu 2.0+ olmalı (1 risk = 2 reward)
        risk_reward_ratio = (reward_pct / risk_pct) if risk_pct > 0 else 0

        # ── Volatilite (ATR%) ────────────────────────────────────────
        # ATR (Average True Range) hissenin günlük hareketini gösterir
        # Bunu entry'ye bölerek normalize ediyoruz
        atr_pct = (atr / entry * 100) if entry > 0 and atr > 0 else 0

        # ── Kalite ve Tip ────────────────────────────────────────────
        quality_score = float(trade.get("quality_score") or 0)

        # Swing type'ı sayıya çevir (ML sayısal veri ister)
        swing_type_raw = str(trade.get("swing_type") or "A").strip().upper()
        swing_type_enc = SWING_TYPE_MAP.get(swing_type_raw, 0)

        max_hold_days = int(trade.get("max_hold_days") or 7)

        # ── Zaman Özellikleri ────────────────────────────────────────
        # Hangi gün/ay girildi? Piyasa günü kalıpları olabilir.
        day_of_week = 0
        month = 0
        try:
            date_str = trade.get("entry_date") or trade.get("date") or ""
            if date_str:
                import datetime
                dt = datetime.datetime.strptime(date_str[:10], "%Y-%m-%d")
                day_of_week = dt.weekday()  # 0=Pazartesi, 4=Cuma
                month = dt.month
        except Exception:
            pass  # Tarih yoksa 0 kalır

        # Özellik sırası FEATURE_COLUMNS ile aynı olmalı!
        return [
            risk_pct,
            reward_pct,
            risk_reward_ratio,
            atr_pct,
            quality_score,
            swing_type_enc,
            max_hold_days,
            day_of_week,
            month,
        ]
