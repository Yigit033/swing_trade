"""Ek T kuralları — MOM (aylık 12-1 momentum rotasyonu) ve havuz kıyası. Saf fonksiyonlar: veri kökü,
ortam değişkeni ya da dosya okumaz (testler doğrudan çağırır). Ön kayıt: lab/PREREGISTRATION.md Ek T."""
import numpy as np
import pandas as pd

N_POS, SELL_RANK = 10, 20          # en çok 10 pozisyon; uygunlar içinde sıra > 20 → sat
LOOK, SKIP = 252, 21               # 12-1 momentum: Close[t−21] / Close[t−252] − 1


def rebalance_days(cal, start, end):
    """Dönem başından önceki son seans + dönem içindeki ay sonu seansları (dönemin son seansı hariç)."""
    per = cal[(cal >= start) & (cal <= end)]
    month_last = pd.Series(cal, index=cal).groupby(cal.to_period("M")).max()
    ends = set(pd.DatetimeIndex(month_last.values))
    return [cal[cal < start][-1]] + [d for d in per if d in ends and d < per[-1]], per[-1]


def mom_trades(P, pool, regimes, start, end, use_regime=True) -> pd.DataFrame:
    O, C = P["Open"], P["Close"]
    cal = C.index
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    sma200 = C.rolling(200).mean()
    score = C.shift(SKIP) / C.shift(LOOK) - 1
    idx = {d: i for i, d in enumerate(cal)}
    days, last = rebalance_days(cal, start, end)
    held, rows = {}, []

    def close_out(h, v, xd, px, why, at_open):
        rows.append(dict(strategy="MOM", ticker=h, signal_date=v["signal"], entry_day=v["entry_day"], exit_day=xd,
                         entry_price=v["entry_price"], exit_price=float(px),
                         ret_pct=100 * (float(px) / v["entry_price"] - 1), reason=why, key=v["score"],
                         regime=v["regime"], at_open=at_open, bars=idx[xd] - idx[v["entry_day"]]))

    for t in days:
        d1 = cal[idx[t] + 1]
        in_pool = pool.loc[t].reindex(C.columns, fill_value=False).astype(bool)
        above = C.loc[t] > sma200.loc[t]
        sc = score.loc[t]
        elig = in_pool & above & sc.notna()
        ranked = sc[elig].sort_values(ascending=False)
        rank = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
        for h in list(held):                                   # sat: uygun değil ya da sıra > 20
            if h in rank.index and rank[h] <= SELL_RANK:
                continue
            why = "POOL" if not in_pool[h] else ("TREND" if not above[h] else ("RANK" if h in rank.index else "NA"))
            j = idx[d1]
            while j <= idx[last] and not np.isfinite(O.iat[j, C.columns.get_loc(h)]):
                j += 1                                         # açılış barı yoksa ilk bulunan açılışa kadar bekle
            if j <= idx[last]:
                close_out(h, held.pop(h), cal[j], O.iat[j, C.columns.get_loc(h)], why, True)
        reg = regimes.get(t, ("UNKNOWN", ""))[0]
        if (not use_regime) or reg == "BULL":                 # al: yalnız BULL
            for h in ranked.index:
                if len(held) >= N_POS:
                    break
                if h in held:
                    continue
                px = O.at[d1, h]
                if not np.isfinite(px) or px <= 0:
                    continue
                held[h] = dict(signal=t, entry_day=d1, entry_price=float(px), score=float(ranked[h]), regime=reg)
    for h, v in held.items():                                  # dönem sonu: son seansın kapanışı
        col = C[h].loc[:last].dropna()
        close_out(h, v, col.index[-1], col.iloc[-1], "END", False)
    T = pd.DataFrame(rows, columns=["strategy", "ticker", "signal_date", "entry_day", "exit_day", "entry_price",
                                    "exit_price", "ret_pct", "reason", "key", "regime", "at_open", "bars"])
    return T.sort_values(["entry_day", "key"], ascending=[True, False]).reset_index(drop=True)


def attach_pool(T, P, pool) -> pd.DataFrame:
    """Havuz L'nin aynı giriş (açılış) → aynı çıkış zamanlaması getirisi (çıkış açılıştaysa açılış, değilse kapanış)."""
    O, C = P["Open"], P["Close"]
    pool = pool.reindex(columns=C.columns, fill_value=False)
    cache, out = {}, []
    for r in T.itertuples():
        k = (r.signal_date, r.entry_day, r.exit_day, bool(r.at_open))
        if k not in cache:
            m = pool.loc[r.signal_date].values
            x = ((O if r.at_open else C).loc[r.exit_day].values[m] / O.loc[r.entry_day].values[m] - 1) * 100
            x = x[np.isfinite(x)]
            cache[k] = float(np.mean(x)) if len(x) else np.nan
        out.append(cache[k])
    T = T.copy()
    T["pool_ret"] = out
    T["excess"] = T.ret_pct - T.pool_ret
    return T
