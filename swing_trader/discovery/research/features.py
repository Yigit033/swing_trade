"""Vectorized *diagnostics* of champion predicates; parity checked against originals.

Future columns are outcomes only, never inputs to selectors. Historical nominal
price reconstructs subsequent recorded splits; adjusted price ratios are normalized
at decision time. This mitigates ex-post adjustment, does NOT create PIT vendor data.
"""
import numpy as np
import pandas as pd
from ..bars import rows_frame
from swing_trader.small_cap.signals import SmallCapSignals
from swing_trader.small_cap.filters import SmallCapFilters


def decision_frame(obj, day):
    raw = rows_frame(obj["rows"], adjusted=False)
    stamp = pd.Timestamp(day)
    splits = raw.loc[raw.index > stamp, "Stock Splits"]
    factor = float(splits[splits > 0].prod())
    part = rows_frame(obj["rows"]).loc[:stamp].copy()
    ratio_now = float((raw["Adj Close"] / raw.Close).loc[stamp])
    part[["Open", "High", "Low", "Close"]] *= factor / ratio_now
    part["Adj Close"] = part["Close"]
    part["Volume"] /= factor
    return part


def features(obj, calendar, settings):
    raw = rows_frame(obj["rows"], adjusted=False).reindex(calendar)
    d = rows_frame(obj["rows"]).reindex(calendar)
    c, h, l, v = d.Close, d.High, d.Low, d.Volume
    sig = SmallCapSignals(settings=settings)
    x = pd.DataFrame(index=d.index)
    ratio = raw["Adj Close"] / raw.Close
    split = raw["Stock Splits"].where(raw["Stock Splits"] > 0, 1)
    after = split.iloc[::-1].cumprod().iloc[::-1] / split
    x["price"] = raw.Close * after
    x["dv"] = (c*v).rolling(20).mean() / ratio
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atrpct = tr.rolling(sig.VCE_ATR_PERIOD).mean()/c*100
    far, near = sig.VCE_BASELINE_OFFSET
    baseline = atrpct.shift(near+1).rolling(far-near).mean()
    x["squeeze"] = atrpct.shift()/baseline
    x["atr_pct"] = atrpct
    x["rvol50"] = v/v.rolling(sig.RVOL_BASELINE_DAYS).mean()
    x["rvol20"] = v/v.rolling(20).mean()
    x["change1"] = (c/c.shift()-1)*100
    x["change5"] = (c/c.shift(5)-1)*100
    x["change20"] = (c/c.shift(20)-1)*100
    ma20, ma50 = c.rolling(20).mean(), c.rolling(50).mean()
    hi20 = h.shift().rolling(sig.VCE_BREAKOUT_LOOKBACK).max()
    x["distance20high_pct"] = (c/hi20-1)*100
    x["vce_squeeze"] = x["squeeze"] < sig.VCE_SQUEEZE_RATIO
    x["vce_high20"] = c > hi20
    x["vce_green"] = c > c.shift()
    x["vce_ma50"] = c > ma50
    x["vce_volume"] = x.rvol50 >= sig.VCE_MIN_RVOL_GATE
    x["rvol_lower"] = x.rvol50 >= sig.RVOL_THRUST_MULT
    x["rvol_upper"] = x.rvol50 < sig._rvol_max
    x["rvol_green"] = c > c.shift()
    x["rvol_change"] = x.change1 < sig._rvol_max_day_change
    x["rvol_ma20"] = c > ma20
    x["vce"] = x[[k for k in x if k.startswith("vce_")]].all(axis=1)
    x["rvol"] = x[[k for k in x if k.startswith("rvol_")]].all(axis=1)
    x["champion"] = x.vce | x.rvol
    x["history"] = d.Close.notna().rolling(130).sum() == 130
    x["zero_volume52"] = (v <= 0).rolling(52).sum()
    x["price_pass"] = x.price.between(settings.universe_filters.min_price, settings.universe_filters.max_price)
    x["liquidity_pass"] = x.dv >= SmallCapFilters.MIN_DOLLAR_VOLUME
    x["base"] = x.history & x.price_pass & x.liquidity_pass
    range5 = h.rolling(5).max()-l.rolling(5).min()
    range20 = h.rolling(20).max()-l.rolling(20).min()
    x["range_ratio"] = range5/range20
    up = (c*v).where(c > c.shift(), 0).rolling(10).sum()
    down = (c*v).where(c < c.shift(), 0).rolling(10).sum()
    x["accumulation_ratio"] = up/down.replace(0, np.nan)
    x["dry_volume"] = v.rolling(5).median()/v.rolling(20).median()
    x["prebreak"] = ((ma20 > ma50) & (c/hi20).between(.97, 1) & (x.range_ratio <= .5)
                       & (x.accumulation_ratio >= 1.2) & (x.dry_volume <= .8))
    impulse = (c.shift(5)/c.shift(10)-1)*100
    prior5hi, prior5lo = h.shift().rolling(5).max(), l.shift().rolling(5).min()
    x["impulse5"] = impulse
    x["relaunch"] = (impulse.between(10, 40) & (((prior5hi/prior5lo)-1)*100 <= 8)
                       & (ma20 > ma50) & (c > prior5hi) & (x.rvol20 >= 1.2) & (x.change1 < 8))
    return x, d


def outcomes(d):
    """Same next scheduled open, same adjusted revision; never fed to features."""
    x = pd.DataFrame(index=d.index)
    entry = d.Open.shift(-1)
    x["entry_gap_pct"] = (entry/d.Close-1)*100
    for n in (5, 10, 20, 30):
        # Rows t+1..t+n, including entry day as day1.
        future = d.shift(-1)
        valid = future.Close.notna().iloc[::-1].rolling(n).sum().iloc[::-1] == n
        vol_ok = (future.Volume > 0).iloc[::-1].rolling(n).sum().iloc[::-1] == n
        end = d.Close.shift(-n)
        high = future.High.iloc[::-1].rolling(n).max().iloc[::-1]
        low = future.Low.iloc[::-1].rolling(n).min().iloc[::-1]
        x[f"valid{n}"] = valid & vol_ok & (entry > 0)
        x[f"r{n}"] = ((end/entry-1)*100).where(x[f"valid{n}"])
        x[f"mfe{n}"] = ((high/entry-1)*100).clip(lower=0).where(x[f"valid{n}"])
        x[f"mae{n}"] = ((low/entry-1)*100).clip(upper=0).where(x[f"valid{n}"])
        for bps in (20, 50, 100):
            x[f"net{n}_{bps}"] = (100*(end*(1-bps/20000)/(entry*(1+bps/20000))-1)).where(x[f"valid{n}"])
    return x


VCE_PREDICATES = ["vce_squeeze", "vce_high20", "vce_green", "vce_ma50", "vce_volume"]
RVOL_PREDICATES = ["rvol_lower", "rvol_upper", "rvol_green", "rvol_change", "rvol_ma20"]
