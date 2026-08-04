# -*- coding: utf-8 -*-
"""Düşük-likidite evreni için sharesOutstanding + floatShares (mcap/float skoru gerçek olsun)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings('ignore')
import logging; logging.disable(logging.CRITICAL)
import pickle, yfinance as yf

CACHE = 'output/_shares_lowliq.json'
if os.path.exists(CACHE):
    print('shares cache zaten var'); sys.exit(0)

data = pickle.load(open('output/_lowliq_data.pkl', 'rb'))
tickers = list(data.keys())
print(f'{len(tickers)} ticker icin shares cekiliyor...', flush=True)
out = {}
for i, t in enumerate(tickers):
    try:
        info = yf.Ticker(t).info
        out[t] = {'shares': info.get('sharesOutstanding'), 'float': info.get('floatShares')}
    except Exception:
        out[t] = {'shares': None, 'float': None}
    if (i + 1) % 50 == 0:
        print(f'  {i+1}/{len(tickers)}', flush=True)
        json.dump(out, open(CACHE, 'w'))
json.dump(out, open(CACHE, 'w'))
got = sum(1 for v in out.values() if v.get('shares'))
print(f'TAMAM: {got}/{len(tickers)} ticker shares verisi aldi', flush=True)
