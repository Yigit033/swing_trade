# -*- coding: utf-8 -*-
"""
DÜŞÜK-LİKİDİTE EVRENİ ÇEK — "$5M dolar-hacim kapısı bize para kaybettiriyor mu?"
================================================================================
NEDEN GEREKLİ: 2026-08-04'te dolar-hacim kapısını (5M→3M→2M→1M) gevşetme testi
"etkisiz" çıktı — ama test GEÇERSİZDİ. Sebep: mevcut önbellek (995 ticker,
S&P 400+600) neredeyse tamamen likit isimlerden oluşuyor; sadece 8 ticker (%1)
$5M altında ve sinyallerin %0'ı. Yani kapının eleyeceği popülasyon veri setinde
HİÇ YOKTU. "Etkisi yok" değil, "ölçülemedi".

Bu script tam o eksik popülasyonu çeker: Finviz keşif bantlarını GEÇEN ama
motorun $5M dolar-hacim kapısına TAKILAN hisseler.

Hedef profil: $5M/gün = fiyat × adet. Kapının kestiği bölge düşük fiyat +
mütevazı hacim: örn. $10 × 400K adet = $4M/gün (takılır), $30 × 400K = $12M (geçer).
Bu yüzden sorgu: fiyat $7-20, ortalama hacim 300K-750K adet.

⚠️ SURVIVORSHIP UYARISI (sonucu yorumlarken ZORUNLU): Finviz BUGÜNÜN listesini
verir. Düşük likiditeli small-cap'lerde delist/iflas oranı likit isimlerden çok
daha yüksektir — yani bugün hayatta olanlar, o dönemin popülasyonunun İYİMSER
bir alt kümesidir. Bu önyargı gevşetme LEHİNE çalışır. Ölçüm "gevşet" derse bile
gerçek sonuç ölçülenden kötü olur; "gevşetme" derse sonuç kesindir.

Çıktı: output/_lowliq_universe.json + output/_lowliq_data.pkl
"""
import sys, os, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings('ignore')
import logging; logging.disable(logging.CRITICAL)
import pandas as pd, yfinance as yf

from swing_trader.small_cap.universe import _ticker_safe_overview_cls

START, END = '2024-06-01', '2026-05-30'
UNIV_CACHE = 'output/_lowliq_universe.json'
DATA_CACHE = 'output/_lowliq_data.pkl'

# Motorun $5M kapısının kestiği bölgeyi hedefleyen sorgular
# Finviz yalnız hazır bantlar kabul ediyor (serbest aralık yok). Geçerli
# seçenekler içinden kapının kestiği bölgeye en yakın kombinasyonlar:
#   fiyat $5-10 / $10-20  ×  hacim 100K-500K / 500K-1M
# Dolar-hacim ($5M) filtresi zaten sonradan post-filtre olarak uygulanacak.
QUERIES = [
    ("small, $5-10, 100K-500K", {
        'Market Cap.': 'Small ($300mln to $2bln)', 'Country': 'USA',
        'Price': '$5 to $10', 'Average Volume': '100K to 500K',
    }),
    ("small, $10-20, 100K-500K", {
        'Market Cap.': 'Small ($300mln to $2bln)', 'Country': 'USA',
        'Price': '$10 to $20', 'Average Volume': '100K to 500K',
    }),
    ("small, $5-10, 500K-1M", {
        'Market Cap.': 'Small ($300mln to $2bln)', 'Country': 'USA',
        'Price': '$5 to $10', 'Average Volume': '500K to 1M',
    }),
    ("small, $10-20, 500K-1M", {
        'Market Cap.': 'Small ($300mln to $2bln)', 'Country': 'USA',
        'Price': '$10 to $20', 'Average Volume': '500K to 1M',
    }),
    ("mid, $10-20, 100K-500K", {
        'Market Cap.': 'Mid ($2bln to $10bln)', 'Country': 'USA',
        'Price': '$10 to $20', 'Average Volume': '100K to 500K',
    }),
]


def fetch_universe():
    if os.path.exists(UNIV_CACHE):
        t = json.load(open(UNIV_CACHE))
        print(f'Evren cache: {len(t)} ticker'); return t
    Overview = _ticker_safe_overview_cls()
    seen = []
    for label, filters in QUERIES:
        try:
            ov = Overview()
            ov.set_filter(filters_dict=filters)
            df = ov.screener_view(order='Market Cap.', ascend=False)
            got = list(df['Ticker']) if df is not None and len(df) else []
            print(f'  {label:<28} {len(got):>4} ticker', flush=True)
            seen.extend(got)
        except Exception as e:
            print(f'  {label:<28} HATA: {e}', flush=True)
        time.sleep(1)
    uniq = sorted(set(seen))
    json.dump(uniq, open(UNIV_CACHE, 'w'))
    print(f'Toplam tekil: {len(uniq)} ticker → {UNIV_CACHE}')
    return uniq


def fetch_data(tickers):
    if os.path.exists(DATA_CACHE):
        d = pickle.load(open(DATA_CACHE, 'rb'))
        print(f'Veri cache: {len(d)} ticker'); return d
    print(f'{len(tickers)} ticker indiriliyor ({START} → {END})...', flush=True)
    data, CHUNK = {}, 40
    chunks = [tickers[i:i + CHUNK] for i in range(0, len(tickers), CHUNK)]
    for ci, ch in enumerate(chunks):
        try:
            raw = yf.download(ch, start=START, end=END, group_by='ticker',
                              auto_adjust=True, progress=False, threads=True)
            for t in ch:
                try:
                    df = raw[t].dropna().reset_index()
                    if len(df) >= 120:
                        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                        data[t] = df
                except Exception:
                    pass
        except Exception as e:
            print(f'  chunk {ci+1} hata: {e}', flush=True)
        if (ci + 1) % 3 == 0:
            print(f'  {ci+1}/{len(chunks)} chunk, {len(data)} ticker', flush=True)
        time.sleep(0.5)
    pickle.dump(data, open(DATA_CACHE, 'wb'))
    print(f'TAMAM: {len(data)}/{len(tickers)} → {DATA_CACHE}')
    return data


def main():
    print('DÜŞÜK-LİKİDİTE EVRENİ (motorun $5M kapısının kestiği bölge)\n')
    tickers = fetch_universe()
    if not tickers:
        print('Evren boş — Finviz sorguları sonuç vermedi.'); return
    data = fetch_data(tickers)

    # Gerçekten hedef bölgede miyiz? Dolar-hacim dağılımını doğrula
    import numpy as np
    vals = []
    for tk, df in data.items():
        try:
            c = df['Close'].astype(float).tail(20); v = df['Volume'].astype(float).tail(20)
            vals.append(float((c * v).mean()) / 1e6)
        except Exception:
            pass
    if vals:
        a = np.array(vals)
        print(f'\nDolar-hacim dağılımı ({len(a)} ticker, M$/gün):')
        print(f'  min={a.min():.2f}  medyan={np.median(a):.1f}  max={a.max():.0f}')
        for th in (1, 2, 3, 5, 10):
            n = (a < th).sum()
            print(f'  <${th}M: {n} ticker ({n/len(a)*100:.0f}%)')
        below5 = (a < 5).sum()
        print(f'\n=> Hedef bölge (<$5M): {below5} ticker '
              f'({below5/len(a)*100:.0f}%) — önceki sette bu oran %1 idi')


if __name__ == '__main__':
    main()
