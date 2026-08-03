# -*- coding: utf-8 -*-
"""
BİLEŞEN ÇIKARMA — OOS DOĞRULAMASI (R2 + R4)
================================================================================
Faz 2'de bırak-birini-çıkar testi çarpıcı bir sonuç verdi: hacim bileşenini
(volume_explosion) skordan ÇIKARINCA üst-bucket EV +4.97% → +8.29% çıktı
(+3.32 puan, WR %60 → %80). Ayrıca hacim bileşeninin getiriyle ham korelasyonu
NEGATİF (-0.146).

Ama o test 88 sinyalin üst %18'i = ~16 işlem üzerindeydi. Bu büyüklükte bir
iddia için fazlasıyla küçük. Üç ek sınav:

  S1  OOS ayrımı (2025-06-01): train'de kazanan OOS'ta da kazanıyor mu?
  S2  Sabit-sayı karşılaştırması: yüzde-bucket yerine "aynı sayıda sinyal seç"
      (bucket kayması EV farkını sahte şişirebiliyor)
  S3  Rastgele-alt-örneklem kararlılığı: 200 bootstrap turunda kaç kez kazanıyor?
      (>=%80 kazanma oranı yoksa gürültü sayılır)

KABUL KRİTERİ (R2+R4, önceden yazıldı): bir bileşenin çıkarılması ancak
  (a) TRAIN ve OOS'ta AYRI AYRI pozitif iyileştirme,
  (b) sabit-sayı karşılaştırmasında da iyileştirme,
  (c) bootstrap kazanma oranı >= %80
ise ÖNERİLİR. Aksi halde gürültü kabul edilir ve MEVCUT skor korunur.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

MAXES = dict(vol=30, atr=25, float=20, mom=15, risk=15, trend=25)
CURRENT_W = dict(vol=.12, atr=.13, float=.25, mom=.25, risk=.10, trend=.15)
OOS_SPLIT = "2025-06-01"
LIVE = dict(price=7.0, mcap_m=250.0, dvol_m=5.0, avgvol_k=500.0)
RNG = np.random.default_rng(20260803)


def load(avgvol_k=LIVE["avgvol_k"]):
    recs = json.load(open("output/signal_lab.json"))
    return [r for r in recs
            if r["price"] >= LIVE["price"] and r["mcap_m"] >= LIVE["mcap_m"]
            and r["dvol_m"] >= LIVE["dvol_m"] and r["avgvol_k"] >= avgvol_k]


def rescore(r, w):
    return sum((max(r.get(f"c_{k}", 0), 0) / mx) * 100 * w.get(k, 0)
               for k, mx in MAXES.items())


def drop_weights(drop):
    w = {k: v for k, v in CURRENT_W.items() if k != drop}
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


def top_n_ev(rows, w, n):
    """Verilen ağırlıkla skorla, EN İYİ n sinyali seç, EV döndür (sabit sayı)."""
    if not rows or n <= 0:
        return None
    n = min(n, len(rows))
    scored = sorted(rows, key=lambda r: -rescore(r, w))[:n]
    a = np.array([r["r"] for r in scored])
    return float(a.mean()), float((a > 0).mean() * 100)


def main():
    rows = load()
    tr = [r for r in rows if r["date"] < OOS_SPLIT]
    te = [r for r in rows if r["date"] >= OOS_SPLIT]
    print("=" * 96)
    print(f"  BİLEŞEN ÇIKARMA — OOS DOĞRULAMASI")
    print(f"  Toplam {len(rows)} sinyal | TRAIN {len(tr)} (<{OOS_SPLIT}) | OOS {len(te)} (>={OOS_SPLIT})")
    print("=" * 96)

    # Seçim sayısı: her setin üst %30'u (küçük örneklemde %18 çok azdı)
    FRAC = 0.30
    n_all, n_tr, n_te = [max(3, int(len(x) * FRAC)) for x in (rows, tr, te)]
    print(f"  Karşılaştırma: her ağırlık setiyle EN İYİ n sinyal (sabit sayı)")
    print(f"    TÜM n={n_all}   TRAIN n={n_tr}   OOS n={n_te}\n")

    base_all = top_n_ev(rows, CURRENT_W, n_all)
    base_tr = top_n_ev(tr, CURRENT_W, n_tr)
    base_te = top_n_ev(te, CURRENT_W, n_te)
    print(f"  {'ayarlar':<22}{'TÜM':<20}{'TRAIN':<20}{'OOS':<20}{'karar'}")
    print("  " + "-" * 92)
    print(f"  {'MEVCUT (6 bileşen)':<22}"
          f"{f'EV {base_all[0]:+.2f}% WR {base_all[1]:.0f}%':<20}"
          f"{f'EV {base_tr[0]:+.2f}% WR {base_tr[1]:.0f}%':<20}"
          f"{f'EV {base_te[0]:+.2f}% WR {base_te[1]:.0f}%':<20}—")

    verdicts = {}
    for drop in MAXES:
        w = drop_weights(drop)
        a, t, o = top_n_ev(rows, w, n_all), top_n_ev(tr, w, n_tr), top_n_ev(te, w, n_te)
        d_all, d_tr, d_oos = a[0] - base_all[0], t[0] - base_tr[0], o[0] - base_te[0]

        # S3: bootstrap kararlılığı
        wins = 0
        B = 200
        for _ in range(B):
            idx = RNG.integers(0, len(rows), len(rows))
            samp = [rows[i] for i in idx]
            nb = max(3, int(len(samp) * FRAC))
            bb, dd = top_n_ev(samp, CURRENT_W, nb), top_n_ev(samp, w, nb)
            if dd and bb and dd[0] > bb[0]:
                wins += 1
        stab = wins / B * 100

        ok = (d_tr > 0) and (d_oos > 0) and (d_all > 0) and stab >= 80
        verdicts[drop] = (ok, d_all, d_tr, d_oos, stab)
        if ok:
            mark = f"✓ ÖNERİLİR (kararlılık %{stab:.0f})"
        elif d_all > 0:
            reasons = []
            if d_tr <= 0: reasons.append("TRAIN'de değil")
            if d_oos <= 0: reasons.append("OOS'ta değil")
            if stab < 80: reasons.append(f"kararlılık %{stab:.0f}")
            mark = f"✗ GÜRÜLTÜ ({', '.join(reasons)})"
        else:
            mark = f"✗ zararlı (kararlılık %{stab:.0f})"
        print(f"  {('-' + drop + ' çıkar'):<22}"
              f"{f'EV {a[0]:+.2f}% ({d_all:+.2f})':<20}"
              f"{f'EV {t[0]:+.2f}% ({d_tr:+.2f})':<20}"
              f"{f'EV {o[0]:+.2f}% ({d_oos:+.2f})':<20}{mark}")

    # ── Eşik + huni kombinasyonu (Faz 3'te Q82 öne çıkmıştı) ──────────────
    print("\n" + "=" * 96)
    print("  EŞİK KARŞILAŞTIRMASI — mevcut skorla, canlı vs genişletilmiş huni")
    print("=" * 96)
    span_all = 21.0
    print(f"  {'ayar':<38}{'n':>5}{'/ay':>7}{'EV':>9}{'WR':>7}{'PF':>7}   {'OOS EV':>9}")
    print("  " + "-" * 88)
    for label, av, th in [
        ("MEVCUT: canlı huni @Q80", 500.0, 80),
        ("canlı huni @Q82", 500.0, 82),
        ("genişletilmiş huni @Q80", 300.0, 80),
        ("genişletilmiş huni @Q82", 300.0, 82),
        ("genişletilmiş huni @Q85", 300.0, 85),
    ]:
        rr = [r for r in load(avgvol_k=av) if r["q"] >= th]
        if not rr:
            continue
        a = np.array([r["r"] for r in rr])
        w_, l_ = a[a > 0], a[a <= 0]
        pf = (w_.sum() / abs(l_.sum())) if l_.size and l_.sum() != 0 else float("inf")
        oos = [r["r"] for r in rr if r["date"] >= OOS_SPLIT]
        oev = np.mean(oos) if oos else 0.0
        print(f"  {label:<38}{len(rr):>5}{len(rr)/span_all:>7.1f}{a.mean():>+9.2f}"
              f"{(a>0).mean()*100:>6.0f}%{pf:>7.2f}   {oev:>+9.2f}")

    print("\n" + "=" * 96)
    accepted = [d for d, v in verdicts.items() if v[0]]
    if accepted:
        print(f"  SONUÇ: şu bileşen(ler) çıkarılmalı → {', '.join(accepted)}")
    else:
        print("  SONUÇ: hiçbir bileşen çıkarma üç sınavı birden geçmedi → MEVCUT SKOR KORUNUR")
        print("         (Faz 2'deki +3.32 puanlık 'hacim zararlı' bulgusu küçük-örneklem gürültüsü)")


if __name__ == "__main__":
    main()
