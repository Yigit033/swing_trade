"""Ek X kuralları — TRD: büyüyen şirket + yükselen eğilim, haftalık karar (kırılım kalıbı yok).

Saf fonksiyonlar: veri kökü, ortam değişkeni ya da dosya okumaz (testler doğrudan çağırır).
Ön kayıt: lab/PREREGISTRATION.md Ek X. Koşucu: lab/trend_x.py. Büyüme tanımı, stop, risk ve maliyet Ek V'den
(lab/pro_rules.py) aynen alınır.
"""
import numpy as np
import pandas as pd

from lab import pro_rules as pr

SMA_FAST, SMA_SLOW = 50, 200
STOP = pr.MAX_STOP                                   # sabit %8: stop = (1 − 0.08) × giriş
COLS = ["strategy", "ticker", "signal_date", "entry_day", "exit_day", "entry_price", "exit_price", "ret_pct",
        "reason", "key", "at_open", "bars", "stop_dist", "alloc_frac", "net"]


def trend_mask(P) -> pd.DataFrame:
    """Yükselen eğilim: Close > SMA50 VE SMA50 > SMA200."""
    C = P["Close"]
    s50, s200 = C.rolling(SMA_FAST).mean(), C.rolling(SMA_SLOW).mean()
    return (C > s50) & (s50 > s200)


def week_last(cal: pd.DatetimeIndex) -> np.ndarray:
    """Haftanın son seansı: bir sonraki seans başka bir ISO haftasında. Takvimin son günü bilinemez → False."""
    iso = cal.isocalendar()
    wk = iso["year"].values.astype(int) * 100 + iso["week"].values.astype(int)
    out = np.zeros(len(cal), dtype=bool)
    out[:-1] = wk[1:] != wk[:-1]
    return out


def growth_rate_asof(q: pd.DataFrame, cal: pd.DatetimeIndex) -> np.ndarray:
    """Takvimdeki her gün için bilinen son çeyreğin yıllık satış büyümesi (oran); bilinmiyorsa NaN.

    ``pro_rules.growth_asof`` ile aynı çeyrek seçimi (betimleyici "satış büyümesi sırası" kıyası için).
    """
    rate = np.full(len(cal), np.nan)
    if q is None or len(q) == 0:
        return rate
    q = q.sort_values("end").reset_index(drop=True)
    kq = np.maximum(q.rev_known.values, q.ni_known.values)
    ends = q.end.values
    order = np.argsort(kq, kind="stable")
    calv = cal.values
    starts = np.searchsorted(calv, kq[order], side="right")
    cur = np.full(len(cal), -1)
    best = -1
    for j, s in zip(order, starts):
        if best == -1 or ends[j] > ends[best]:
            best = j
        if s < len(cal):
            cur[s:] = best
    day = np.timedelta64(1, "D")
    for j in np.unique(cur[cur >= 0]):
        fresh = (cur == j) & ((calv - ends[j]) / day <= pr.MAX_STALE_DAYS)
        gap = (ends[j] - ends) / day
        prev = np.where((gap >= pr.YOY_DAYS[0]) & (gap <= pr.YOY_DAYS[1]))[0]
        if len(prev) == 0:
            continue
        p = prev[np.argmin(np.abs(gap[prev] - 364))]
        have = fresh & (calv > kq[p])
        if q.rev.iat[p] > 0:
            rate[have] = q.rev.iat[j] / q.rev.iat[p] - 1
    return rate


def simulate(P, eligible, key, bull, start, end, cost=0.20, fee=1.5, cap0=10_000.0, risk=pr.RISK,
             cold=pr.RISK_COLD, streak=pr.COLD_STREAK, max_frac=pr.MAX_FRAC, stop=STOP):
    """Haftalık karar portföyü (ön kayıt Ek X). Dönüş: (özet, günlük özsermaye serisi, işlemler).

    Gün sırası: açılış satışları (haftalık çıkış kararı + stopun altına boşluklu açılış) → açılış alımları
    (önceki kapanış özsermayesiyle boyut) → gün içi stop → kapanış değerlemesi → karar günüyse yarının emirleri.
    ``eligible``/``key``: gün × hisse; ``bull``: gün → bool. Giriş günleri [start, end] içinde.
    """
    O, L, C = (P[f] for f in ("Open", "Low", "Close"))
    cal, cols = C.index, C.columns
    per = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    k0, last = cal.get_loc(per[0]), cal.get_loc(per[-1])
    o, lo, c = O.values, L.values, C.values
    s50 = C.rolling(SMA_FAST).mean().values
    wl = week_last(cal)
    elig = eligible.reindex(index=cal, columns=cols, fill_value=False).fillna(False).values.astype(bool)
    kv = np.nan_to_num(key.reindex(index=cal, columns=cols).values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    bl = pd.Series(bull).reindex(cal, fill_value=False).values.astype(bool)

    cash, pos, closed, rows, curve, idle = float(cap0), [], [], [], [], []

    def decide(k):
        """Karar günü k: yarının alım sırası (rejim BULL ise uygun ve elde olmayanlar, anahtar büyükten küçüğe)."""
        if not (wl[k] and bl[k]):
            return []
        held = {p["j"] for p in pos}
        cand = [j for j in np.where(elig[k])[0] if j not in held]
        return [(j, k) for j in sorted(cand, key=lambda j: -kv[k, j])]

    def close_out(p, k, px, why, at_open):
        nonlocal cash
        cash += p["sh"] * px * (1 - cost / 200) - fee
        ret = 100 * (px / p["px"] - 1)
        net = ret - cost - 200 * fee / p["alloc"]
        closed.append(net)
        rows.append(dict(strategy="TRD", ticker=cols[p["j"]], signal_date=cal[p["sig"]], entry_day=cal[p["ie"]],
                         exit_day=cal[k], entry_price=p["px"], exit_price=float(px), ret_pct=ret, reason=why,
                         key=p["key"], at_open=bool(at_open), bars=k - p["ie"] + 1, stop_dist=stop,
                         alloc_frac=p["frac"], net=net))

    pending = decide(k0 - 1) if k0 > 0 else []
    for k in range(k0, last + 1):
        keep = []                                              # 1) açılış satışları
        for p in pos:
            ok = np.isfinite(o[k, p["j"]])
            if ok and p["sell"]:
                close_out(p, k, o[k, p["j"]], "TRAIL", True)
            elif ok and o[k, p["j"]] <= p["stop"]:
                close_out(p, k, o[k, p["j"]], "STOP", True)
            else:
                keep.append(p)
        pos = keep
        eq_prev = cash + sum(p["sh"] * p["last"] for p in pos)
        for j, sig in pending:                                 # 2) açılış alımları
            px = o[k, j]
            if not np.isfinite(px) or px <= 0:
                continue
            is_cold = len(closed) >= streak and all(x < 0 for x in closed[-streak:])
            alloc = min(max_frac * eq_prev, (cold if is_cold else risk) * eq_prev / stop)
            if alloc > cash or alloc <= 2 * fee:
                continue
            cash -= alloc
            pos.append(dict(j=j, sig=sig, ie=k, px=float(px), stop=(1 - stop) * float(px), sell=False, alloc=alloc,
                            frac=alloc / eq_prev, key=float(kv[sig, j]), last=float(px),
                            sh=(alloc - fee) / (px * (1 + cost / 200))))
        pending = []
        keep = []                                              # 3) gün içi stop
        for p in pos:
            if np.isfinite(lo[k, p["j"]]) and lo[k, p["j"]] <= p["stop"]:
                close_out(p, k, p["stop"], "STOP", False)
            else:
                keep.append(p)
        pos = keep
        for p in pos:                                          # 4) kapanış değerlemesi
            if np.isfinite(c[k, p["j"]]):
                p["last"] = float(c[k, p["j"]])
        if k == last:
            for p in pos:
                close_out(p, k, p["last"], "END", False)
            pos = []
        elif wl[k]:                                            # 5) karar günü: haftalık çıkış + yarının alımları
            for p in pos:
                j = p["j"]
                if np.isfinite(c[k, j]) and np.isfinite(s50[k, j]) and c[k, j] < s50[k, j]:
                    p["sell"] = True
            pending = decide(k)
        eq = cash + sum(p["sh"] * p["last"] for p in pos)
        curve.append(eq)
        idle.append(cash / eq if eq > 0 else 1.0)

    eq = pd.Series(curve, index=cal[k0:last + 1])
    T = pd.DataFrame(rows, columns=COLS)
    peak = np.maximum.accumulate(np.r_[cap0, eq.values])[1:]
    out = dict(total_pct=round(float((eq.iloc[-1] / cap0 - 1) * 100), 1),
               max_dd_pct=round(float((eq.values / peak - 1).min() * 100), 1),
               taken=len(T), idle_cash_pct=round(100 * float(np.mean(idle)), 1),
               net_per_trade=round(float((T.ret_pct - cost).mean()), 3) if len(T) else None,
               mean_bars=round(float(T.bars.mean()), 1) if len(T) else None,
               mean_position_pct=round(100 * float(T.alloc_frac.mean()), 1) if len(T) else None)
    return out, eq, T.sort_values(["entry_day", "key"], ascending=[True, False]).reset_index(drop=True)
