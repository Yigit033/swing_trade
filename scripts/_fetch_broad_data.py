# -*- coding: utf-8 -*-
"""Geniş evren (S&P 400+600) 2 yıllık veri indir → cache. Batch, dayanıklı."""
import sys, os, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings('ignore')
import logging; logging.disable(logging.CRITICAL)
import pandas as pd, yfinance as yf

tickers = json.load(open('output/_universe_broad.json'))
START, END = '2024-06-01', '2026-05-30'
CACHE = 'output/_broad_data.pkl'

if os.path.exists(CACHE):
    d = pickle.load(open(CACHE,'rb'))
    print(f'Cache zaten var: {len(d)} ticker'); sys.exit(0)

print(f'{len(tickers)} ticker indiriliyor ({START} -> {END})...', flush=True)
data = {}
CHUNK = 40
chunks = [tickers[i:i+CHUNK] for i in range(0, len(tickers), CHUNK)]
for ci, ch in enumerate(chunks):
    try:
        raw = yf.download(ch, start=START, end=END, group_by='ticker',
                          auto_adjust=True, progress=False, threads=True)
        for t in ch:
            try:
                df = raw[t].dropna().reset_index()
                if len(df) >= 120:  # en az ~6 ay veri (batmışlar/yeniler elenir — gerçekçi)
                    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                    data[t] = df
            except Exception:
                pass
    except Exception as e:
        print(f'  chunk {ci+1} hata: {e}', flush=True)
    if (ci+1) % 5 == 0:
        print(f'  {ci+1}/{len(chunks)} chunk, {len(data)} ticker veri aldı', flush=True)
    time.sleep(0.5)

pickle.dump(data, open(CACHE,'wb'))
print(f'TAMAM: {len(data)}/{len(tickers)} ticker veri aldı → {CACHE}', flush=True)
