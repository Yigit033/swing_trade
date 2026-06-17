"""
EXIT STRATEGY LAB — "T1/T2 çok mu küçük? Kazananları koştursak daha mı çok kazanırız?"
=====================================================================================
Sistemin GERÇEKTEN seçtiği canlı Finviz evreni üzerinde (test-live tutarlılık):
  1. VCE (Variant B) sinyallerini bulur — lookahead yok, giriş ertesi gün açılış
  2. Her sinyali FARKLI çıkış stratejileriyle bar-bar simüle eder
  3. EV/trade, win rate, ort kazanç/kayıp, MFE-yakalama oranını raporlar

Amaç: "kazananı koştur" (geniş trailing, T2 cap yok) gerçekten mevcut yapıdan
(T1 kısmi + breakeven + T2 cap) daha çok mu kazandırıyor — veriyle.

Veri tek sefer çekilip output/_exit_univ.pkl'e cache'lenir.
"""
import sys, os, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.basicConfig(level=logging.ERROR)
import numpy as np, pandas as pd


def get_universe_tickers(cap=260):
    from swing_trader.small_cap.universe import SmallCapUniverse
    uni = SmallCapUniverse()
    return uni.get_universe(use_finviz=True, max_tickers=cap, force_refresh=True)


def fetch(tickers, start, end, cache):
    if os.path.exists(cache):
        with open(cache, 'rb') as f: return pickle.load(f)
    import yfinance as yf
    print(f"  {len(tickers)} ticker, {start}→{end} indiriliyor...")
    raw = yf.download(tickers, start=start, end=end, group_by='ticker',
                      auto_adjust=True, progress=False, threads=True)
    data = {}
    for t in tickers:
        try:
            df = raw[t].dropna().reset_index()
            if len(df) >= 80:
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                data[t] = df
        except Exception:
            pass
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, 'wb') as f: pickle.dump(data, f)
    print(f"  {len(data)}/{len(tickers)} ticker yeterli veri")
    return data


def add(df):
    c=df['Close'].astype(float); h=df['High'].astype(float); l=df['Low'].astype(float)
    df=df.copy()
    df['ma50']=c.rolling(50).mean(); df['hi20']=h.rolling(20).max()
    df['vol20']=df['Volume'].astype(float).rolling(20).mean()
    trr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    df['atr']=trr.rolling(14).mean(); df['atr_pct']=df['atr']/c*100
    return df


def is_vce_B(df, t):
    """Variant B: squeeze + breakout + green + MA50 (canlı sistemle aynı)."""
    a=df['atr_pct'].iloc[t-1]; b=df['atr_pct'].iloc[t-20:t-5].mean()
    if pd.isna(a) or pd.isna(b) or b<=0 or a>=b*0.8: return False
    c=df['Close'].iloc[t]; hi=df['hi20'].iloc[t-1]
    if pd.isna(hi) or not c>hi>0: return False
    if c<=df['Close'].iloc[t-1]: return False
    m=df['ma50'].iloc[t]
    return (not pd.isna(m) and c>m)


def simulate(df, t, strat):
    """
    Bar-bar çıkış simülasyonu. Giriş = t+1 açılış.
    strat: dict(stop_atr, t1_pct, t1_frac, be_after_t1, t2_pct, trail_atr, trail_after, hold)
    Döner: realized return % (pozisyon ağırlıklı).
    """
    o=df['Open'].astype(float).values; c=df['Close'].astype(float).values
    h=df['High'].astype(float).values; l=df['Low'].astype(float).values
    n=len(df); e=t+1
    if e>=n: return None
    entry=o[e]; atr=float(df['atr'].iloc[t])
    if entry<=0 or atr<=0: return None

    stop=entry-strat['stop_atr']*atr
    t1=entry*(1+strat['t1_pct']) if strat.get('t1_pct') else None
    t2=entry*(1+strat['t2_pct']) if strat.get('t2_pct') else None
    pos=1.0; realized=0.0; peak=entry; t1_done=False

    last=min(e+strat['hold'], n-1)
    for j in range(e, last+1):
        # 1. Stop / trailing stop (önce — kötü senaryo)
        if l[j]<=stop:
            px = min(o[j], stop) if o[j] < stop else stop
            realized += pos*(px/entry-1); pos=0.0; break
        # 2. T1 kısmi
        if t1 and not t1_done and h[j]>=t1:
            realized += strat['t1_frac']*(t1/entry-1)
            pos -= strat['t1_frac']; t1_done=True
            if strat.get('be_after_t1'): stop=max(stop, entry)
        # 3. T2 cap (varsa) — kalanı sat
        if t2 and h[j]>=t2:
            realized += pos*(t2/entry-1); pos=0.0; break
        # 4. Trailing güncelle (peak'ten)
        if h[j]>peak: peak=h[j]
        if strat.get('trail_atr'):
            gain_atr=(peak-entry)/atr
            if gain_atr>=strat.get('trail_after',1.0):
                stop=max(stop, peak-strat['trail_atr']*atr)
    if pos>0:
        realized += pos*(c[last]/entry-1)
    return realized*100


STRATS = {
 'MEVCUT (T1+BE+T2cap28)': dict(stop_atr=1.5, t1_pct=0.10, t1_frac=0.5, be_after_t1=True,
                                 t2_pct=0.28, trail_atr=2.5, trail_after=2.0, hold=10),
 'RUNNER (T2cap yok, trail2.0)': dict(stop_atr=1.5, t1_pct=0.10, t1_frac=0.5, be_after_t1=True,
                                 t2_pct=None, trail_atr=2.0, trail_after=1.5, hold=20),
 'GENIS RUNNER (stop2, trail2.5)': dict(stop_atr=2.0, t1_pct=0.12, t1_frac=0.33, be_after_t1=True,
                                 t2_pct=None, trail_atr=2.5, trail_after=1.5, hold=20),
 'SAF TRAIL (T1 yok, trail2.5)': dict(stop_atr=2.0, t1_pct=None, t1_frac=0.0, be_after_t1=False,
                                 t2_pct=None, trail_atr=2.5, trail_after=1.0, hold=20),
 'YUKSEK CAP (T2=45, hold15)': dict(stop_atr=1.5, t1_pct=0.10, t1_frac=0.5, be_after_t1=True,
                                 t2_pct=0.45, trail_atr=2.5, trail_after=2.0, hold=15),
}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--start', default='2024-01-01'); ap.add_argument('--end', default='2026-06-15')
    a=ap.parse_args()
    print("Finviz canlı evreni çekiliyor...")
    tickers=get_universe_tickers(260)
    print(f"  {len(tickers)} ticker")
    data=fetch(tickers, a.start, a.end, 'output/_exit_univ.pkl')
    data={t:add(d) for t,d in data.items()}

    # VCE sinyallerini topla
    sigs=[]
    for tk,df in data.items():
        for t in range(60, len(df)-21):
            if is_vce_B(df,t): sigs.append((tk,df,t))
    print(f"\n  Finviz evreninde {len(sigs)} VCE (Variant B) sinyali bulundu\n")
    if len(sigs)<20:
        print("  Yetersiz sinyal — daha geniş tarih/ evren gerek."); return

    # MFE referansı (potansiyel)
    mfes=[]
    for tk,df,t in sigs:
        h=df['High'].astype(float).values; o=df['Open'].astype(float).values; e=t+1; n=len(df)
        if e>=n: continue
        end=min(e+10,n); mfes.append((h[e:end].max()/o[e]-1)*100)
    print(f"  Sinyallerin 10-gün MFE'si (potansiyel tavan): medyan {np.median(mfes):+.1f}%  "
          f"p75 {np.percentile(mfes,75):+.1f}%  p90 {np.percentile(mfes,90):+.1f}%\n")

    print(f"  {'Strateji':<32}{'EV/trade':>10}{'WR':>7}{'ort kazanç':>12}{'ort kayıp':>11}{'medyan':>9}")
    print("  "+"-"*81)
    results={}
    for name,strat in STRATS.items():
        rets=[]
        for tk,df,t in sigs:
            r=simulate(df,t,strat)
            if r is not None: rets.append(r)
        a_=np.array(rets); wins=a_[a_>0]; losses=a_[a_<=0]
        ev=a_.mean(); wr=(a_>0).mean()*100
        aw=wins.mean() if len(wins) else 0; al=losses.mean() if len(losses) else 0
        results[name]=ev
        print(f"  {name:<32}{ev:>+9.2f}%{wr:>6.0f}%{aw:>+11.1f}%{al:>+10.1f}%{np.median(a_):>+8.1f}%")

    best=max(results, key=results.get)
    print(f"\n  → En yüksek EV/trade: '{best}' ({results[best]:+.2f}%)")
    print(f"  (mevcut: {results['MEVCUT (T1+BE+T2cap28)']:+.2f}%)")
    print("="*84)


if __name__=='__main__':
    main()
