"""GRW "büyüme + kırılım" — SEC XBRL verisinden çeyreklik satış / net kâr ve büyüme şartı. SAF fonksiyonlar.

Ön kayıt: lab/PREREGISTRATION.md Ek V ("Büyüme verisi", "Büyüme şartı") ve Ek W. Tanımlar lab/pro_rules.py ile
birebir aynıdır (swing_trader/tests/test_shadow_grw.py bunu doğrular); lab toplu dosyayı (companyfacts), canlı
etiket bazlı uç noktayı (companyconcept) okur — ikisi de aynı kayıtları aynı biçimde verir.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ── Büyüme verisi (SEC XBRL companyfacts) ────────────────────────────────────
REV_TAGS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
            "RevenuesNetOfInterestExpense")
NI_TAGS = ("NetIncomeLoss", "ProfitLoss")           # öncelik sırası
Q_DAYS, M9_DAYS, FY_DAYS = (80, 100), (260, 285), (350, 380)
YOY_DAYS = (350, 380)                               # Q−4'ün dönem sonu Q'dan bu kadar gün önce
MIN_SALES_GROWTH, MAX_STALE_DAYS = 0.25, 200


def _first_filed(units, pick_max):
    """(start, end) → (değer, dosya tarihi): dönemi İLK dosyalayan belgedeki değer. Aynı ilk tarihte birden çok
    değer varsa ``pick_max`` → en büyüğü (satış etiketleri), değilse ilki (etiket önceliği)."""
    best = {}
    for u in units:
        if "start" not in u or u.get("val") is None or not u.get("filed"):
            continue
        k = (u["start"], u["end"])
        cur = best.get(k)
        if cur is None or u["filed"] < cur[1] or (pick_max and u["filed"] == cur[1] and u["val"] > cur[0]):
            best[k] = (float(u["val"]), u["filed"])
    return best


def _periods(gaap, tags, pick_max):
    """Etiket listesinden (start, end) → (değer, dosya tarihi). pick_max: tüm etiketler birlikte (ilk dosyalayan,
    eşitlikte en büyük); değilse öncelik sırası (ilk etikette olmayan dönemler sonrakinden tamamlanır)."""
    if pick_max:
        units = [u for t in tags for u in ((gaap.get(t) or {}).get("units") or {}).get("USD", [])]
        return _first_filed(units, True)
    out = {}
    for t in tags:
        for k, v in _first_filed(((gaap.get(t) or {}).get("units") or {}).get("USD", []), False).items():
            out.setdefault(k, v)
    return out


def _quarters(periods):
    """Dönem sonu → (çeyreklik değer, bilinme tarihi). Doğrudan çeyrek (80–100 gün) öncelikli; yoksa
    yıllık (350–380) − aynı başlangıçlı dokuz aylık (260–285), bilinme tarihi ikisinin geç olanı."""
    span = {k: (pd.Timestamp(k[1]) - pd.Timestamp(k[0])).days for k in periods}
    q = {}
    for k, d in span.items():
        if Q_DAYS[0] <= d <= Q_DAYS[1]:
            cur = q.get(k[1])
            if cur is None or abs(d - 91) < cur[2]:
                q[k[1]] = (periods[k][0], periods[k][1], abs(d - 91))
    ends = sorted(pd.Timestamp(e) for e in q)
    nine = [(pd.Timestamp(k[0]), k) for k, d in span.items() if M9_DAYS[0] <= d <= M9_DAYS[1]]
    for k, d in span.items():
        if not (FY_DAYS[0] <= d <= FY_DAYS[1]):
            continue
        e = pd.Timestamp(k[1])
        if any(abs((e - x).days) <= 5 for x in ends):          # yıl sonu çeyreği zaten doğrudan raporlanmış
            continue
        s = pd.Timestamp(k[0])
        m = [kk for ss, kk in nine if abs((ss - s).days) <= 5 and 60 <= (e - pd.Timestamp(kk[1])).days <= 110]
        if not m:
            continue
        kk = min(m, key=lambda x: periods[x][1])               # ilk dosyalanan dokuz aylık
        cand = (periods[k][0] - periods[kk][0], max(periods[k][1], periods[kk][1]), 0)
        if k[1] not in q or cand[1] < q[k[1]][1]:
            q[k[1]] = cand
    return {e: (v[0], v[1]) for e, v in q.items()}


def quarters_table(facts: dict) -> pd.DataFrame:
    """companyfacts JSON → çeyrek tablosu [end, rev, rev_known, ni, ni_known] (yalnız ikisi de olan çeyrekler)."""
    gaap = ((facts or {}).get("facts") or {}).get("us-gaap") or {}
    rev = _quarters(_periods(gaap, REV_TAGS, True))
    ni = _quarters(_periods(gaap, NI_TAGS, False))
    rows = [dict(end=pd.Timestamp(e), rev=rev[e][0], rev_known=pd.Timestamp(rev[e][1]),
                 ni=ni[e][0], ni_known=pd.Timestamp(ni[e][1])) for e in sorted(set(rev) & set(ni))]
    return pd.DataFrame(rows, columns=["end", "rev", "rev_known", "ni", "ni_known"])


def growth_asof(q: pd.DataFrame, cal: pd.DatetimeIndex):
    """Takvimdeki her gün t için (bilgi_var, büyüme_şartı) — ön kayıt "Büyüme şartı".

    Q = dosya tarihi < t olan en yeni (dönem sonu en büyük) çeyrek; dönem sonu t'den en çok 200 gün önce.
    Q−4 = dönem sonu Q'dan 350–380 gün önce, dosya tarihi < t. bilgi_var = Q ve Q−4 mevcut ve taze.
    """
    known = np.zeros(len(cal), dtype=bool)
    ok = np.zeros(len(cal), dtype=bool)
    if q is None or len(q) == 0:
        return known, ok
    q = q.sort_values("end").reset_index(drop=True)
    kq = np.maximum(q.rev_known.values, q.ni_known.values)          # çeyreğin bilinme (dosya) tarihi
    ends = q.end.values
    order = np.argsort(kq, kind="stable")
    calv = cal.values
    best = -1                                                        # bilinenler içinde dönem sonu en büyük
    # olay günleri: dosya tarihinden SONRAKİ ilk seans
    starts = np.searchsorted(calv, kq[order], side="right")
    cur = np.full(len(cal), -1)
    for j, s in zip(order, starts):
        if best == -1 or ends[j] > ends[best]:
            best = j
        if s < len(cal):
            cur[s:] = best
    day = np.timedelta64(1, "D")
    for j in np.unique(cur[cur >= 0]):
        m = cur == j
        fresh = m & ((calv - ends[j]) / day <= MAX_STALE_DAYS)
        gap = (ends[j] - ends) / day
        prev = np.where((gap >= YOY_DAYS[0]) & (gap <= YOY_DAYS[1]))[0]
        if len(prev) == 0:
            continue
        p = prev[np.argmin(np.abs(gap[prev] - 364))]
        have = fresh & (calv > kq[p])
        known |= have
        r0, r1, n0, n1 = q.rev.iat[p], q.rev.iat[j], q.ni.iat[p], q.ni.iat[j]
        if r0 > 0 and r1 / r0 - 1 >= MIN_SALES_GROWTH and n1 > n0:
            ok |= have
    return known, ok


def growth_on(q: pd.DataFrame, day: date) -> Dict:
    """Tek gün için büyüme şartı + rapor ayrıntısı (``growth_asof`` ile aynı kural, takvim = [day]).

    Dönüş: known (veri var ve taze), ok (şart sağlandı), varsa quarter_end, sales_growth_pct, ni, ni_prev.
    """
    cal = pd.DatetimeIndex([pd.Timestamp(day)])
    known, ok = growth_asof(q, cal)
    out: Dict = dict(known=bool(known[0]), ok=bool(ok[0]))
    if q is None or len(q) == 0:
        return out
    t = pd.Timestamp(day)
    k = q[(q.rev_known < t) & (q.ni_known < t)].sort_values("end")
    if len(k) == 0:
        return out
    cur = k.iloc[-1]
    out["quarter_end"] = cur.end.date().isoformat()
    gap = (cur.end - k.end).dt.days
    prev = k[(gap >= YOY_DAYS[0]) & (gap <= YOY_DAYS[1])]
    if len(prev):
        p = prev.iloc[int(np.argmin(np.abs((cur.end - prev.end).dt.days.to_numpy() - 364)))]
        if p.rev > 0:
            out["sales_growth_pct"] = round(100 * float(cur.rev / p.rev - 1), 1)
        out["ni"], out["ni_prev"] = float(cur.ni), float(p.ni)
        out["rev"], out["rev_prev"] = float(cur.rev), float(p.rev)       # Ek Y okuma kartı (kural değişmez)
    return out
