# -*- coding: utf-8 -*-
"""
FİYAT BANDI ÖLÇÜMÜ — "Hangi fiyat bandı bizim strateji için en kârlı av sahası?"
================================================================================
NEDEN: 2026-08-04'te dolar-hacim kapısını ölçerken beklenmedik bir şey çıktı —
aynı motor, aynı eşik (Q80), aynı exit ile:
    likit evren (S&P 400/600)     → EV +2.41%
    illikit evren ($5-20 fiyat)   → EV -2.36%   (dolar-hacim $5M ÜSTÜ bile!)
Yani kapıları/eşikleri kurcalamaktan önce sorulacak soru: DOĞRU HAVUZDA MI
avlanıyoruz? Canlı ayar: Finviz 'Over $7', motor $7-$1000.

METODOLOJİ — değişken izolasyonu:
İki veri setini BİRLEŞTİRMİYORUM. Birleştirsem fiyat ile likiditeyi karıştırırdım
($5-20 hisseler doğal olarak daha illikit). Bunun yerine:
  A) Her set İÇİNDE fiyat kovaları → desen var mı?
  B) İki set AYNI yönü gösteriyor mu? → gerçek fiyat etkisi
     Sadece birinde varsa → likiditeyle karışmış, sonuç yok
  C) KESİŞİM BÖLGESİ ($10-30): fiyat sabitken likit vs illikit → likiditenin
     saf etkisi (fiyattan arındırılmış)

KARAR KURALI (önceden yazıldı): bir fiyat eşiği ancak
  (a) iki sette de aynı yön, (b) OOS'ta tutuyor, (c) kalan sinyal ayda >=2
ise değiştirilir. n<10 kovaya göre eşik değişmez.

Girdi : output/signal_lab.json (likit), output/dollar_volume_gate.json (illikit)
Yeni motor koşusu GEREKMİYOR — iki dosya da fiyat + kalite + gerçek exit getirisi taşıyor.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

OOS_SPLIT = "2025-06-01"
Q_LIVE = 80.0        # canlı eşik (CAUTION/BEAR); BULL 78


def stats(rows):
    if not rows:
        return dict(n=0, ev=0.0, wr=0.0, pf=0.0)
    a = np.array([r["r"] for r in rows])
    w, l = a[a > 0], a[a <= 0]
    pf = (w.sum() / abs(l.sum())) if l.size and l.sum() != 0 else float("inf")
    return dict(n=len(a), ev=float(a.mean()), wr=float((a > 0).mean() * 100), pf=float(pf))


def fmt(s):
    if s["n"] == 0:
        return "n=0"
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return f"n={s['n']:<4} EV {s['ev']:+6.2f}%  WR {s['wr']:3.0f}%  PF {pf:>5}"


def load(path, label):
    recs = json.load(open(path))
    for r in recs:
        r["_set"] = label
        r["r"] = float(r["r"])
        r["q"] = float(r["q"])
        r["price"] = float(r["price"])
    return recs


def band(rows, lo, hi, q_min=None):
    return [r for r in rows
            if lo <= r["price"] < hi and (q_min is None or r["q"] >= q_min)]


BANDS = [(0, 10), (10, 20), (20, 35), (35, 60), (60, 100), (100, 1e9)]


def band_label(lo, hi):
    return f"${lo:.0f}-{hi:.0f}" if hi < 1e9 else f"${lo:.0f}+"


def section(title):
    print("\n" + "=" * 96)
    print(f"  {title}")
    print("=" * 96)


def main():
    liq = load("output/signal_lab.json", "likit")
    ill = load("output/dollar_volume_gate.json", "illikit")

    section("A) HER SET İÇİNDE FİYAT KOVALARI (Q80+ — canlı eşik)")
    for rows, lbl in ((liq, "LİKİT evren (995 ticker, S&P 400/600)"),
                      (ill, "İLLİKİT evren (310 ticker, $5-20 hedefli)")):
        print(f"\n  ── {lbl} ──")
        tot = stats([r for r in rows if r["q"] >= Q_LIVE])
        print(f"    {'TÜMÜ':<12}{fmt(tot)}")
        for lo, hi in BANDS:
            sel = band(rows, lo, hi, Q_LIVE)
            if sel:
                weak = "  ⚠ n<10" if len(sel) < 10 else ""
                print(f"    {band_label(lo, hi):<12}{fmt(stats(sel))}{weak}")

    section("B) İKİ SET AYNI YÖNÜ GÖSTERİYOR MU? (fiyat etkisi gerçek mi?)")
    print(f"  {'band':<12}{'LİKİT':<34}{'İLLİKİT':<34}{'aynı yön?'}")
    print("  " + "-" * 92)
    for lo, hi in BANDS:
        a, b = stats(band(liq, lo, hi, Q_LIVE)), stats(band(ill, lo, hi, Q_LIVE))
        if a["n"] == 0 and b["n"] == 0:
            continue
        if a["n"] and b["n"]:
            same = "✓ evet" if (a["ev"] > 0) == (b["ev"] > 0) else "✗ TERS"
        else:
            same = "— tek sette veri"
        print(f"  {band_label(lo, hi):<12}{fmt(a):<34}{fmt(b):<34}{same}")

    section("C) KESİŞİM BÖLGESİ ($10-30): fiyat sabit, likidite değişken")
    print("  Bu, likiditenin FİYATTAN ARINDIRILMIŞ saf etkisi.")
    print("  " + "-" * 92)
    for lbl, rows in (("LİKİT $10-30", liq), ("İLLİKİT $10-30", ill)):
        sel = [r for r in rows if 10 <= r["price"] < 30 and r["q"] >= Q_LIVE]
        print(f"    {lbl:<18}{fmt(stats(sel))}")

    section("D) ALTERNATİF MİN FİYAT EŞİKLERİ (likit evrende, Q80+)")
    print("  Canlı: Finviz 'Over $7', motor $7-$1000")
    print(f"\n  {'min fiyat':<12}{'sinyal':<34}{'ELENENLERİN getirisi':<30}")
    print("  " + "-" * 92)
    base = [r for r in liq if r["q"] >= Q_LIVE]
    for th in (7, 15, 20, 25, 35):
        keep = [r for r in base if r["price"] >= th]
        drop = [r for r in base if r["price"] < th]
        d = stats(drop)
        ds = f"-{d['n']:<4} EV {d['ev']:+6.2f}% WR {d['wr']:3.0f}%" if d["n"] else "—"
        print(f"  ${th:<11}{fmt(stats(keep)):<34}{ds:<30}")

    section("E) MAKS FİYAT — pahalı hisseler işe yarıyor mu?")
    print(f"  {'maks fiyat':<12}{'sinyal':<34}{'ELENENLERİN getirisi':<30}")
    print("  " + "-" * 92)
    for th in (1000, 200, 100, 60):
        keep = [r for r in base if r["price"] <= th]
        drop = [r for r in base if r["price"] > th]
        d = stats(drop)
        ds = f"-{d['n']:<4} EV {d['ev']:+6.2f}% WR {d['wr']:3.0f}%" if d["n"] else "—"
        print(f"  ${th:<11}{fmt(stats(keep)):<34}{ds:<30}")

    section("F) OOS DOĞRULAMA — aday eşikler gelecekte de tutuyor mu?")
    def split(rows):
        return ([r for r in rows if r["date"] < OOS_SPLIT],
                [r for r in rows if r["date"] >= OOS_SPLIT])
    print(f"  {'ayar':<24}{'TRAIN':<34}{'OOS (test)':<34}")
    print("  " + "-" * 92)
    CANDS = [
        ("MEVCUT: $7-1000", base),
        ("$7-200 (maks düşür)", [r for r in base if r["price"] <= 200]),
        ("$7-100 (maks düşür)", [r for r in base if r["price"] <= 100]),
        ("$15+", [r for r in base if r["price"] >= 15]),
        ("$20+", [r for r in base if r["price"] >= 20]),
        ("$20-200", [r for r in base if 20 <= r["price"] <= 200]),
    ]
    for lbl, sel in CANDS:
        tr, te = split(sel)
        st_, se = stats(tr), stats(te)
        verdict = ""
        if se["n"] >= 10 and st_["n"] >= 10:
            b0 = stats(split(base)[1])
            verdict = "✓ OOS'ta daha iyi" if se["ev"] > b0["ev"] + 0.3 else (
                "— fark yok" if abs(se["ev"] - b0["ev"]) <= 0.3 else "✗ OOS'ta daha kötü")
        print(f"  {lbl:<24}{fmt(st_):<34}{fmt(se):<34}{verdict}")

    section("G) PAHALI HİSSE — mekanik sebep var mı? (pozisyon granülerliği)")
    print("  Pozisyon tavanı: $10.000 portföyün %25'i = $2.500")
    print(f"\n  {'fiyat':<12}{'alınabilen hisse':<20}{'1 hisse = pozisyonun %':<26}{'granülerlik'}")
    print("  " + "-" * 92)
    for px in (20, 50, 100, 200, 400, 618):
        sh = int(2500 / px)
        pct = (px / 2500 * 100) if sh else 100.0
        note = "sağlam" if pct < 5 else ("kaba" if pct < 15 else "KULLANILAMAZ")
        print(f"  ${px:<11}{sh:<20}{pct:>6.1f}%{'':<19}{note}")
    print("\n  → Pahalı hissede stop/hedef matematiği yuvarlama hatasına boğulur:")
    print("    $618'lik hissede 4 hisse alırsın; 1 hisse pozisyonun %25'i.")
    print("    Bu, veri deseninin ARKASINDAKİ mekanik sebep olabilir.")

    section("SONUÇ")
    span_months = 21.0
    print(f"  Aylık sinyal hızı (R3 kuralı: >=2/ay şart):")
    for lbl, sel in [("$7+ (mevcut)", base),
                     ("$15+", [r for r in base if r["price"] >= 15]),
                     ("$20+", [r for r in base if r["price"] >= 20]),
                     ("$20-200", [r for r in base if 20 <= r["price"] <= 200])]:
        s = stats(sel)
        rate = s["n"] / span_months
        ok = "✓" if rate >= 2 else "✗ çok az"
        print(f"    {lbl:<16}{rate:>5.1f}/ay  EV {s['ev']:+6.2f}%  {ok}")


if __name__ == "__main__":
    main()
