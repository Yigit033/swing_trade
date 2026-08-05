# -*- coding: utf-8 -*-
"""
BARAJ KARARI — HANGİSİ DAHA ÇOK PARA KAZANDIRIR?
================================================================================
Soru: tavan dışı +8/+5, 34 sinyali Q80'in üstüne taşıyor ve o 34'ün EV'si
+0.89% (taban +4.19%). Barajı ham skora uygulayıp bunları elemek daha mı kârlı?

TUZAK: "EV arttı" cevap DEĞİL. Barajı yükseltmek işlem başı ortalamayı HER ZAMAN
artırır (zayıf olanları atıyorsun) ama TOPLAM kârı düşürebilir — çünkü atılanlar
zarar ettirmiyor, sadece az kazanıyor (+0.89% > 0).

Doğru soru: SERMAYE KISITI altında hangi kurgu daha çok para yapar?
  · Slot sınırsızsa  → her pozitif-EV işlem para ekler, çok işlem kazanır.
  · Slot kısıtlıysa  → zayıf işlem, iyi bir işlemin yerini kapatır (fırsat
    maliyeti) ve o zaman eleme kazanır.
Hangi rejimde olduğumuz slot sayısına bağlı → duyarlılık taraması yapılır.

MODEL: kronolojik sırayla ilerle, eşzamanlı pozisyon sayısı `slots` ile sınırlı,
her pozisyon ~1 ay (20 işlem günü) slot tutar, sermaye eşit bölünür ve bileşik
büyür. Slot doluysa sinyal KAÇIRILIR (canlıdaki top_n davranışı).
Maliyet: ölçülen 0.19 puan/işlem slippage+spread.
"""
import json
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE = "output/post_cap_bonuses.json"
Q = 80.0
PREMIUM_PTS = 8.0
COIL_PTS = 5.0
COIL_RATIO_MAX = 0.65
COST_PCT = 0.19          # ölçülen slippage + spread maliyeti (puan)
HOLD_MONTHS = 1.0        # 20 işlem günü ≈ 1 ay slot doluluğu
MONTHS = 21.0


def load():
    recs = json.load(open(CACHE))
    for r in recs:
        prem = bool(r.get("is_premium"))
        sq = float(r.get("squeeze_ratio", 0) or 0)
        coil = bool(0 < sq < COIL_RATIO_MAX)
        r["q_bonus"] = float(r["q"])                       # mevcut: bonuslu skor
        r["q_raw"] = float(r["q"]) - (PREMIUM_PTS if prem else 0) - (COIL_PTS if coil else 0)
        r["net"] = float(r["r"]) - COST_PCT                # maliyet düşülmüş getiri
    recs.sort(key=lambda x: x["date"])
    return recs


def simulate(recs, gate_key, slots):
    """Slot-kısıtlı portföy: eşiği geçenler sırayla girer, slot doluysa kaçırılır."""
    equity = 1.0
    open_until = deque()          # (bitiş ayı,) — slot doluluğu
    taken, missed, rets = 0, 0, []
    for r in recs:
        if r[gate_key] < Q:
            continue
        # ay indeksi (kaba): yıl*12 + ay
        y, m = int(r["date"][:4]), int(r["date"][5:7])
        t = y * 12 + m
        while open_until and open_until[0] <= t:
            open_until.popleft()
        if len(open_until) >= slots:
            missed += 1
            continue
        open_until.append(t + HOLD_MONTHS)
        alloc = 1.0 / slots                     # sermayenin slot başına payı
        equity *= (1 + alloc * r["net"] / 100)
        taken += 1
        rets.append(r["net"])
    a = np.array(rets) if rets else np.array([0.0])
    return dict(equity=equity, taken=taken, missed=missed,
                ev=a.mean(), wr=(a > 0).mean() * 100)


def main():
    recs = load()
    variants = [("MEVCUT  (baraj bonuslu skora)", "q_bonus"),
                ("ÖNERİ   (baraj ham skora)", "q_raw")]

    print("=" * 92)
    print("  BARAJ KARARI — 21 ay, maliyet %.2f puan/işlem dahil" % COST_PCT)
    print("=" * 92)

    print("\n  A) SERMAYE SINIRSIZ — her sinyal alınabiliyorsa")
    print(f"    {'kurgu':<32}{'işlem':>7}{'işlem/ay':>10}{'EV':>9}{'WR':>6}{'TOPLAM getiri':>16}")
    for label, key in variants:
        sel = [r for r in recs if r[key] >= Q]
        a = np.array([r["net"] for r in sel])
        print(f"    {label:<32}{len(a):>7}{len(a)/MONTHS:>10.1f}{a.mean():>+8.2f}%{(a>0).mean()*100:>5.0f}%"
              f"{a.sum():>+15.0f}%")
    print("    → toplam getiri = işlem sayısı × EV. Atılan işlemler POZİTİF EV'li")
    print("      olduğu için elemek toplamı DÜŞÜRÜR.")

    print("\n  B) SLOT KISITLI — eşzamanlı pozisyon sınırı varsa (gerçek durum)")
    print(f"    {'slot':<6}{'kurgu':<32}{'alınan':>8}{'kaçan':>7}{'EV':>9}{'son sermaye':>14}")
    for slots in (2, 3, 4, 5, 8, 12):
        best = None
        for label, key in variants:
            s = simulate(recs, key, slots)
            print(f"    {slots:<6}{label:<32}{s['taken']:>8}{s['missed']:>7}"
                  f"{s['ev']:>+8.2f}%{(s['equity']-1)*100:>+13.1f}%")
            if best is None:
                best = (label, s["equity"])
            else:
                win = "ÖNERİ" if s["equity"] > best[1] else "MEVCUT"
                diff = abs(s["equity"] - best[1]) * 100
                print(f"    {'':<6}{'→ kazanan: ' + win:<32}{'':>8}{'':>7}{'':>9}{diff:>+13.1f} puan fark")
        print()

    print("  YORUM: slot sayısı düşükse (sermaye dar) zayıf işlem iyi işlemin yerini")
    print("  kapatır → eleme kazanır. Slot bolsa her pozitif-EV işlem para ekler →")
    print("  mevcut kurgu kazanır. Kesişme noktası kararı belirler.")


if __name__ == "__main__":
    main()
