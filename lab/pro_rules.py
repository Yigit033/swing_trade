"""Ek V kuralları — PRO: büyüyen lider hisse + VCE kırılımı + dar stop + kazananı tutma + değişken boyut.

Saf fonksiyonlar: veri kökü, ortam değişkeni ya da dosya okumaz (testler doğrudan çağırır).
Ön kayıt: lab/PREREGISTRATION.md Ek V. Koşucu: lab/pro_v.py.
"""
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

# ── Lider / tetik / işlem ────────────────────────────────────────────────────
RS_PCT, LOW_MULT, HIGH_MULT, SMA200_LAG = 0.80, 1.30, 0.75, 21
VCE_ATR, VCE_SQUEEZE, VCE_BASE, VCE_LOOK, VCE_RVOL, VCE_RVOL_DAYS = 14, 0.8, (20, 5), 20, 1.5, 50
MAX_STOP = 0.08                                      # stop = max(Low_t, (1 − 0.08) × giriş)
RISK, RISK_COLD, COLD_STREAK, MAX_FRAC = 0.01, 0.005, 2, 0.25
COLS = ["strategy", "ticker", "signal_date", "entry_day", "exit_day", "entry_price", "exit_price", "ret_pct",
        "reason", "key", "at_open", "bars", "stop_dist"]


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


# ── Tetik ve lider şartı (panel: gün × hisse) ────────────────────────────────

def vce_mask(P) -> pd.DataFrame:
    """signals.check_vce_breakout sert kapıları, vektörel (parite: test_lab_pro)."""
    O, H, L, C, V = (P[f] for f in ("Open", "High", "Low", "Close", "Volume"))
    pc = C.shift(1)
    tr = np.maximum(H - L, np.maximum((H - pc).abs(), (L - pc).abs()))
    tr = tr.where(pc.notna(), H - L)                                   # ilk bar: yalnız H − L
    atr = tr.rolling(VCE_ATR).mean() / C * 100
    far, near = VCE_BASE
    base = atr.rolling(far - near).mean().shift(near + 1)              # ATR%(t−20 … t−6) ortalaması
    squeeze = (atr.shift(1) / base) < VCE_SQUEEZE
    brk = C > H.rolling(VCE_LOOK).max().shift(1)
    rvol = V / V.rolling(VCE_RVOL_DAYS).mean()
    return squeeze & (base > 0) & brk & (C > pc) & (C > C.rolling(50).mean()) & (rvol >= VCE_RVOL)


def rs_score(P) -> pd.DataFrame:
    return P["Close"] / P["Close"].shift(252) - 1


def leader_mask(P, pool) -> pd.DataFrame:
    """Minervini trend şablonu + göreli güç yüzdeliği ≥ 80 (t günü havuz üyeleri içinde)."""
    H, L, C = P["High"], P["Low"], P["Close"]
    s50, s150, s200 = C.rolling(50).mean(), C.rolling(150).mean(), C.rolling(200).mean()
    pct = rs_score(P).where(pool).rank(axis=1, pct=True)
    return ((C > s150) & (C > s200) & (s150 > s200) & (s200 > s200.shift(SMA200_LAG)) & (s50 > s150) & (C > s50)
            & (C >= LOW_MULT * L.rolling(252).min()) & (C >= HIGH_MULT * H.rolling(252).max()) & (pct >= RS_PCT))


# ── İşlemler ─────────────────────────────────────────────────────────────────

def _walk_trail(ie, O, L, C, S50, stop, last):
    """Giriş ie açılışı. Her gün: (dün Close < SMA50 ise) açılışta sat → stop (min(Open, stop)) → bugün
    Close < SMA50 ise yarın açılışta sat. ``last`` = dönemin son seansı (kapanışla kapat)."""
    sell_next = False
    for k in range(ie, last + 1):
        if not np.isfinite(C[k]):
            continue                                                   # bar yok: bekle
        if sell_next and np.isfinite(O[k]):
            return k, float(O[k]), "TRAIL", True
        if L[k] <= stop:
            px = min(O[k], stop)
            return k, float(px), "STOP", bool(px == O[k] and k > ie)
        if np.isfinite(S50[k]) and C[k] < S50[k]:
            sell_next = True
    k = last
    while k > ie and not np.isfinite(C[k]):
        k -= 1
    return k, float(C[k]), "END", False


def _walk_time(ie, O, L, C, stop, last, hold=8):
    """Betimleyici kıyas: aynı stop, çıkış ``hold``. seansın kapanışı."""
    end = min(last, ie + hold - 1)
    for k in range(ie, end + 1):
        if not np.isfinite(C[k]):
            continue
        if L[k] <= stop:
            px = min(O[k], stop)
            return k, float(px), "STOP", bool(px == O[k] and k > ie)
    k = end
    while k > ie and not np.isfinite(C[k]):
        k -= 1
    return k, float(C[k]), ("TIME" if end == ie + hold - 1 else "END"), False


def pro_trades(P, signal, key, start, end, exit_mode="trail") -> pd.DataFrame:
    """Sinyal düzeyi işlemler (portföyden bağımsız). ``signal``: gün × hisse bool; giriş t+1 açılışı dönem içinde.

    Stop = max(Low_t, (1 − 0.08) × giriş); giriş ≤ Low_t → işlem yok. Hisse başına tek açık pozisyon.
    """
    O, L, C = P["Open"], P["Low"], P["Close"]
    cal = C.index
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    per = cal[(cal >= start) & (cal <= end)]
    i0, last = cal.get_loc(per[0]) - 1, cal.get_loc(per[-1])           # ilk sinyal günü: dönemden önceki son seans
    s50 = C.rolling(50).mean()
    sig = signal.reindex(index=cal, columns=C.columns, fill_value=False).fillna(False).values[i0:last]
    rows = []
    for jc in np.where(sig.any(axis=0))[0]:
        t = C.columns[jc]
        o, lo, c, m50 = O[t].values, L[t].values, C[t].values, s50[t].values
        busy_until = -1
        for i in i0 + np.where(sig[:, jc])[0]:
            ie = i + 1
            if ie <= busy_until:                                       # hisse başına tek açık pozisyon
                continue
            px = o[ie]
            if not np.isfinite(px) or px <= 0 or not np.isfinite(lo[i]) or px <= lo[i]:
                continue
            stop = max(lo[i], (1 - MAX_STOP) * px)
            xi, xp, why, at_open = (_walk_trail(ie, o, lo, c, m50, stop, last) if exit_mode == "trail"
                                    else _walk_time(ie, o, lo, c, stop, last))
            busy_until = xi
            rows.append(dict(strategy="PRO", ticker=t, signal_date=cal[i], entry_day=cal[ie], exit_day=cal[xi],
                             entry_price=float(px), exit_price=xp, ret_pct=100 * (xp / px - 1), reason=why,
                             key=float(key.iat[i, jc]) if np.isfinite(key.iat[i, jc]) else 0.0, at_open=at_open,
                             bars=xi - ie + 1, stop_dist=float((px - stop) / px)))
    T = pd.DataFrame(rows, columns=COLS)
    return T.sort_values(["entry_day", "key"], ascending=[True, False]).reset_index(drop=True)


# ── Portföy: risk tabanlı boyut ──────────────────────────────────────────────

def portfolio_risk(T, C, cost=0.20, fee=1.5, cap0=10_000.0, risk=RISK, cold=RISK_COLD, streak=COLD_STREAK,
                   max_frac=MAX_FRAC, fixed_frac=None):
    """Günlük değerleme; pozisyon = min(max_frac × özsermaye, risk × özsermaye ÷ stop uzaklığı). Son ``streak``
    kapanan işlem de (maliyet sonrası) zarardaysa risk ``cold``. Nakit yetmezse atla. ``fixed_frac`` verilirse
    sabit oran (betimleyici kıyas). Dönüş: (özet, günlük özsermaye serisi, alınan işlemler)."""
    if len(T) == 0:
        return dict(total_pct=0.0, max_dd_pct=0.0, taken=0, idle_cash_pct=100.0), pd.Series(dtype=float), T
    T = T.sort_values(["entry_day", "key"], ascending=[True, False])
    days = C.index[(C.index >= T.entry_day.min()) & (C.index <= T.exit_day.max())]
    by_entry = {d: g for d, g in T.groupby("entry_day")}
    cash, pos, curve, closed, idle, taken, allocs = cap0, [], [], [], [], [], []

    def close_out(p):
        nonlocal cash
        cash += p["sh"] * p["exit_price"] * (1 - cost / 200) - fee
        closed.append(p["net"])

    for d in days:
        keep = []
        for p in pos:
            (close_out(p) if p["exit_day"] == d and p["at_open"] else keep.append(p))
        pos = keep
        eq_prev = cash + sum(p["sh"] * p["last"] for p in pos)
        for t in by_entry.get(d, pd.DataFrame()).itertuples():
            is_cold = len(closed) >= streak and all(x < 0 for x in closed[-streak:])
            if fixed_frac is not None:
                alloc = fixed_frac * eq_prev
            else:
                alloc = min(max_frac * eq_prev, (cold if is_cold else risk) * eq_prev / max(t.stop_dist, 1e-9))
            if alloc > cash or alloc <= 2 * fee:
                continue
            cash -= alloc
            taken.append(t.Index)
            allocs.append(alloc / eq_prev)
            pos.append(dict(t=t.ticker, sh=(alloc - fee) / (t.entry_price * (1 + cost / 200)), exit_day=t.exit_day,
                            exit_price=t.exit_price, at_open=t.at_open, last=t.entry_price,
                            net=t.ret_pct - cost - 200 * fee / alloc))
        keep = []
        for p in pos:
            (close_out(p) if p["exit_day"] == d else keep.append(p))
        pos = keep
        for p in pos:
            v = C.at[d, p["t"]]
            if np.isfinite(v):
                p["last"] = v
        eq = cash + sum(p["sh"] * p["last"] for p in pos)
        curve.append(eq)
        idle.append(cash / eq)
    eq = pd.Series(curve, index=days)
    Tk = T.loc[taken]
    out = dict(total_pct=round(float((eq.iloc[-1] / cap0 - 1) * 100), 1),
               max_dd_pct=round(float((eq / eq.cummax() - 1).min() * 100), 1),
               taken=len(taken), idle_cash_pct=round(100 * float(np.mean(idle)), 1),
               net_per_trade=round(float((Tk.ret_pct - cost).mean()), 3) if len(Tk) else None,
               mean_bars=round(float(Tk.bars.mean()), 1) if len(Tk) else None,
               mean_position_pct=round(100 * float(np.mean(allocs)), 1) if allocs else None)
    return out, eq, Tk
