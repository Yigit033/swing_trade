# -*- coding: utf-8 -*-
"""
SİNYAL LABORATUVARI — ANALİZ AŞAMASI (motor koşmaz, hepsi offline)
================================================================================
collect_signal_lab.py'nin ürettiği output/signal_lab.json üzerinde dört faz:

  FAZ 1  Huni taraması      — hangi kapıyı gevşetmek KÂRLI ek sinyal getiriyor?
  FAZ 2  Skor bileşenleri   — hangi bileşen işe yarıyor? (bırak-birini-çıkar)
  FAZ 3  Eşik taraması      — Q kaçta "diz noktası"? (sayı↔EV takası)
  FAZ 4  OOS doğrulama      — bulunan en iyi ayar geleceğe tutuyor mu?

KARAR KURALLARI (baştan yazıldı — sonuca bakıp kural uydurmamak için):
  R1. Bir kapıyı gevşetmek ancak EK sinyallerin EV'si pozitif VE mevcut tabanın
      EV'sini seyreltmiyorsa kabul edilir.
  R2. Bir ağırlık seti ancak üst-bucket EV'yi MEVCUT'tan anlamlı (>1 puan)
      yükseltiyorsa VE OOS'ta da tutuyorsa kabul edilir.
  R3. Bir eşik ancak ayda >=2 sinyal bırakıyorsa aday olabilir (0.6/ay kabul
      edilemez — profesyonel pratik 4-12/ay).
  R4. Her aday OOS (2025-06-01 kesimi) ile doğrulanır; train-only kazanç
      curve-fit sayılır ve REDDEDİLİR.
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

MAXES = dict(vol=30, atr=25, float=20, mom=15, risk=15, trend=25)
OOS_SPLIT = "2025-06-01"

# Canlı (mevcut) kapı değerleri
# 2026-08-04: price_max 1000 → 200 (measure_price_band.py: $200+ EV -2.24%,
# OOS +2.51→+3.45; mekanik sebep: $2.500 pozisyon tavanında $200 üstü hissede
# 1 hisse pozisyonun >%8'i, sizing yuvarlamaya boğuluyor).
LIVE = dict(price=7.0, price_max=200.0, mcap_m=250.0, dvol_m=5.0, avgvol_k=500.0)


def load():
    recs = json.load(open("output/signal_lab.json"))
    for r in recs:
        r["r"] = float(r["r"])
        r["q"] = float(r["q"])
    return recs


def gate(recs, price=None, price_max=None, mcap_m=None, dvol_m=None,
         avgvol_k=None, mcap_max=None):
    out = recs
    if price is not None:
        out = [r for r in out if r["price"] >= price]
    if price_max is not None:
        out = [r for r in out if r["price"] <= price_max]
    if mcap_m is not None:
        out = [r for r in out if r["mcap_m"] >= mcap_m]
    if mcap_max is not None:
        out = [r for r in out if r["mcap_m"] <= mcap_max]
    if dvol_m is not None:
        out = [r for r in out if r["dvol_m"] >= dvol_m]
    if avgvol_k is not None:
        out = [r for r in out if r["avgvol_k"] >= avgvol_k]
    return out


def stats(rows):
    if not rows:
        return dict(n=0, ev=0.0, wr=0.0, pf=0.0)
    a = np.array([r["r"] for r in rows])
    w, l = a[a > 0], a[a <= 0]
    pf = (w.sum() / abs(l.sum())) if l.size and l.sum() != 0 else float("inf")
    return dict(n=len(a), ev=float(a.mean()), wr=float((a > 0).mean() * 100), pf=float(pf))


def months_span(recs):
    ds = sorted(r["date"] for r in recs)
    if not ds:
        return 1.0
    y0, m0 = int(ds[0][:4]), int(ds[0][5:7])
    y1, m1 = int(ds[-1][:4]), int(ds[-1][5:7])
    return max(1.0, (y1 - y0) * 12 + (m1 - m0) + 1)


def fmt(s, span=None):
    rate = f"{s['n']/span:>5.1f}/ay" if span else ""
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return f"n={s['n']:<5}{rate:<9} EV {s['ev']:+6.2f}%  WR {s['wr']:3.0f}%  PF {pf:>5}"


# ── FAZ 1: huni taraması ─────────────────────────────────────────────────
def phase1(recs, span):
    print("\n" + "=" * 100)
    print("  FAZ 1 — HUNİ TARAMASI: hangi kapıyı gevşetmek KÂRLI ek sinyal getiriyor?")
    print("=" * 100)
    base = gate(recs, **LIVE)
    base_keys = {(r["tk"], r["date"]) for r in base}
    print(f"  CANLI TABAN (fiyat>={LIVE['price']} mcap>={LIVE['mcap_m']:.0f}M "
          f"dvol>={LIVE['dvol_m']:.0f}M avgvol>={LIVE['avgvol_k']:.0f}K)")
    print(f"    TÜMÜ    {fmt(stats(base), span)}")
    print(f"    Q80+    {fmt(stats([r for r in base if r['q'] >= 80]), span)}")

    variants = []
    for v in (3.0, 2.0, 1.0):
        variants.append((f"dolar-hacim 5M -> {v:.0f}M", {**LIVE, "dvol_m": v}))
    for v in (150.0, 100.0):
        variants.append((f"mcap alt 250M -> {v:.0f}M", {**LIVE, "mcap_m": v}))
    for v in (5.0, 3.0):
        variants.append((f"fiyat alt 7 -> {v:.0f}", {**LIVE, "price": v}))
    for v in (300.0,):
        variants.append((f"Finviz avgvol 500K -> {v:.0f}K", {**LIVE, "avgvol_k": v}))
    variants.append(("HEPSİ GEVŞEK (3M/150M/$5/300K)",
                     dict(price=5.0, mcap_m=150.0, dvol_m=3.0, avgvol_k=300.0)))
    variants.append(("MAKSİMUM GEVŞEK (1M/100M/$3/300K)",
                     dict(price=3.0, mcap_m=100.0, dvol_m=1.0, avgvol_k=300.0)))

    print(f"\n  {'varyant':<34}{'TÜM sinyal':<44}{'EK sinyaller (R1 kararı)'}")
    print("  " + "-" * 96)
    keep = []
    for tag, g in variants:
        rows = gate(recs, **g)
        s = stats(rows)
        extra = [r for r in rows if (r["tk"], r["date"]) not in base_keys]
        se = stats(extra)
        bs = stats(base)
        # R1: ek EV pozitif VE toplam EV tabandan >0.5 puan düşmüyor
        ok = se["n"] > 0 and se["ev"] > 0 and s["ev"] >= bs["ev"] - 0.5
        mark = "✓ KABUL" if ok else ("✗ RED" if se["n"] else "— etkisiz")
        if ok:
            keep.append((tag, g, s, se))
        ex = f"+{se['n']:<4} EV {se['ev']:+6.2f}% WR {se['wr']:3.0f}%" if se["n"] else "—"
        print(f"  {tag:<34}{fmt(s, span):<44}{ex:<32}{mark}")

    print(f"\n  {'varyant':<34}{'SADECE Q80+':<44}")
    print("  " + "-" * 96)
    for tag, g in [("CANLI TABAN", LIVE)] + variants:
        rows = [r for r in gate(recs, **g) if r["q"] >= 80]
        print(f"  {tag:<34}{fmt(stats(rows), span)}")
    return keep


# ── FAZ 2: skor bileşenleri ──────────────────────────────────────────────
CURRENT_W = dict(vol=.12, atr=.13, float=.25, mom=.25, risk=.10, trend=.15)


def rescore(r, w):
    tot = 0.0
    for k, mx in MAXES.items():
        tot += (max(r.get(f"c_{k}", 0), 0) / mx) * 100 * w.get(k, 0)
    return tot


def top_bucket(rows, w, frac=0.18):
    scored = sorted(((rescore(r, w), r) for r in rows), key=lambda x: -x[0])
    n = max(1, int(len(scored) * frac))
    top = [r for _, r in scored[:n]]
    s = stats(top)
    qs = np.array([x for x, _ in scored]); rs = np.array([r["r"] for _, r in scored])
    s["corr"] = float(np.corrcoef(qs, rs)[0, 1]) if qs.std() > 0 else 0.0
    return s


def phase2(recs, span):
    print("\n" + "=" * 100)
    print("  FAZ 2 — SKOR BİLEŞENLERİ: hangisi işe yarıyor? (canlı kapılarla)")
    print("=" * 100)
    rows = gate(recs, **LIVE)
    if len(rows) < 30:
        print(f"  UYARI: yalnız {len(rows)} sinyal — bileşen testi güvenilmez, atlanıyor")
        return None

    # 2a. Tek tek bileşen↔getiri korelasyonu
    print("\n  2a) Her bileşenin getiriyle HAM korelasyonu (yüksek = bilgi taşıyor)")
    print("  " + "-" * 60)
    rr = np.array([r["r"] for r in rows])
    for k in MAXES:
        v = np.array([r.get(f"c_{k}", 0) for r in rows])
        c = float(np.corrcoef(v, rr)[0, 1]) if v.std() > 0 else 0.0
        bar = "█" * int(abs(c) * 60)
        print(f"    {k:<7}{c:+.3f}  {bar}")

    # 2b. Bırak-birini-çıkar (yeniden normalize)
    print("\n  2b) BIRAK-BİRİNİ-ÇIKAR: bileşeni sil, kalanı yeniden normalize et")
    print("  " + "-" * 78)
    base = top_bucket(rows, CURRENT_W)
    print(f"    {'MEVCUT (6 bileşen)':<28} üst-bucket EV {base['ev']:+6.2f}%  "
          f"WR {base['wr']:3.0f}%  korr {base['corr']:+.3f}")
    for drop in MAXES:
        w = {k: v for k, v in CURRENT_W.items() if k != drop}
        tot = sum(w.values())
        w = {k: v / tot for k, v in w.items()}
        s = top_bucket(rows, w)
        d = s["ev"] - base["ev"]
        verdict = "→ bileşen ZARARLI" if d > 0.5 else ("→ katkısı YOK" if abs(d) <= 0.5 else "→ bileşen FAYDALI")
        print(f"    {('-' + drop + ' çıkarıldı'):<28} üst-bucket EV {s['ev']:+6.2f}%  "
              f"WR {s['wr']:3.0f}%  korr {s['corr']:+.3f}  ({d:+.2f}) {verdict}")

    # 2c. Ağırlık setleri
    print("\n  2c) AĞIRLIK SETLERİ")
    print("  " + "-" * 78)
    sets = {
        "A) MEVCUT v33": CURRENT_W,
        "B) float/mom yarıya": dict(vol=.12, atr=.13, float=.125, mom=.125, risk=.10, trend=.40),
        "C) float/mom sıfır": dict(vol=.10, atr=.25, float=.0, mom=.0, risk=.20, trend=.45),
        "D) trend ağırlıklı": dict(vol=.15, atr=.20, float=.15, mom=.10, risk=.05, trend=.35),
        "E) eşit ağırlık": {k: 1 / 6 for k in MAXES},
        "F) sadece trend+risk": dict(vol=.0, atr=.0, float=.0, mom=.0, risk=.40, trend=.60),
    }
    best = None
    for tag, w in sets.items():
        s = top_bucket(rows, w)
        if best is None or s["ev"] > best[1]["ev"]:
            best = (tag, s, w)
        print(f"    {tag:<24} üst-bucket EV {s['ev']:+6.2f}%  WR {s['wr']:3.0f}%  korr {s['corr']:+.3f}")
    print(f"\n    → En iyi: {best[0]} ({best[1]['ev']:+.2f}%)   "
          f"MEVCUT'tan fark: {best[1]['ev'] - base['ev']:+.2f} puan")
    if best[1]["ev"] - base["ev"] <= 1.0:
        print("      R2: fark <1 puan → MEVCUT ağırlıklar korunur (değişiklik gereksiz)")
    return best


# ── FAZ 3: eşik taraması ─────────────────────────────────────────────────
def phase3(recs, span, gates=None):
    g = gates or LIVE
    print("\n" + "=" * 100)
    print("  FAZ 3 — EŞİK TARAMASI: 'diz noktası' nerede? (R3: >=2 sinyal/ay şart)")
    print("=" * 100)
    rows = gate(recs, **g)
    print(f"  {'eşik':<8}{'n':>6}{'/ay':>8}{'EV':>9}{'WR':>7}{'PF':>8}   karar")
    print("  " + "-" * 70)
    for th in (0, 60, 65, 70, 73, 75, 78, 80, 82, 85):
        sel = [r for r in rows if r["q"] >= th]
        s = stats(sel)
        rate = s["n"] / span
        pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
        note = ""
        if rate < 2:
            note = "✗ R3: çok az sinyal"
        elif s["ev"] <= 0:
            note = "✗ EV negatif"
        else:
            note = "✓ aday"
        print(f"  Q>={th:<5}{s['n']:>6}{rate:>8.1f}{s['ev']:>+9.2f}{s['wr']:>6.0f}%{pf:>8}   {note}")


# ── FAZ 4: OOS doğrulama ─────────────────────────────────────────────────
def phase4(recs, span, candidates):
    print("\n" + "=" * 100)
    print(f"  FAZ 4 — OOS DOĞRULAMA (kesim {OOS_SPLIT}): curve-fit kontrolü")
    print("=" * 100)
    def split(rows):
        return ([r for r in rows if r["date"] < OOS_SPLIT],
                [r for r in rows if r["date"] >= OOS_SPLIT])

    print(f"  {'ayar':<38}{'TRAIN':<28}{'OOS (test)':<28} karar")
    print("  " + "-" * 96)
    cands = [("CANLI TABAN", LIVE)] + [(t, g) for t, g, _, _ in candidates]
    for tag, g in cands:
        rows = gate(recs, **g)
        for th in (78, 80):
            sel = [r for r in rows if r["q"] >= th]
            tr, te = split(sel)
            st_, se = stats(tr), stats(te)
            ok = se["n"] >= 3 and se["ev"] > 0 and se["ev"] >= st_["ev"] * 0.5
            if se["n"] < 3:
                mark = "— OOS örneklem yok"
            elif ok:
                mark = "✓ tutuyor"
            else:
                mark = "✗ OOS düştü (curve-fit şüphesi)"
            train_s = f"n={st_['n']} EV {st_['ev']:+.2f}% WR {st_['wr']:.0f}%"
            oos_s = f"n={se['n']} EV {se['ev']:+.2f}% WR {se['wr']:.0f}%"
            label = f"{tag} @Q{th}"
            print(f"  {label:<38}{train_s:<28}{oos_s:<28}{mark}")


def main():
    recs = load()
    span = months_span(recs)
    a = np.array([r["r"] for r in recs])
    print("=" * 100)
    print(f"  SİNYAL LABORATUVARI — {len(recs)} sinyal | {span:.0f} ay "
          f"({min(r['date'] for r in recs)} → {max(r['date'] for r in recs)})")
    print(f"  Gevşek toplama havuzu: EV {a.mean():+.2f}%  WR {(a>0).mean()*100:.0f}%")
    print("=" * 100)

    keep = phase1(recs, span)
    phase2(recs, span)
    phase3(recs, span)
    if keep:
        print("\n  (Faz 3 tekrar — Faz 1'de KABUL edilen en gevşek kapıyla)")
        phase3(recs, span, gates=keep[-1][1])
    phase4(recs, span, keep)


if __name__ == "__main__":
    main()
