"""
MISSED MOVERS DIAGNOSTIC — "Midas'ta gördüğüm bu hisseleri neden bulamadık?"

Kullanıcının Midas'ta gördüğü büyük yükselenleri (NUVL, PAYO, VELO, ADUR, CECO, OIO)
bizim GERÇEK pipeline'ımızdan geçirir ve her birinin TAM OLARAK nerede öldüğünü
kanıtla gösterir:

  ADIM A — UNIVERSE: Finviz evrenimize giriyor mu? (cap/price/country/float filtreleri)
  ADIM B — ENGINE:   Eğer veri çekilebiliyorsa scan_stock'ta nerede reject ediliyor?

Hiçbir şeyi değiştirmez — sadece okur ve raporlar.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.WARNING)

# Kullanıcının gösterdiği 6 hisse + ekran görüntüsünden okunan veriler
MISSED = {
    'NUVL': {'cap_b': 7.02, 'price': 123.16, 'day_pct': 39.19, 'country': 'USA',   'note': 'biotech (Nuvalent)'},
    'PAYO': {'cap_b': 1.69, 'price': 6.42,   'day_pct': 25.00, 'country': 'USA',   'note': 'Payoneer fintech'},
    'VELO': {'cap_b': 0.472,'price': 19.98,  'day_pct': 42.31, 'country': 'USA',   'note': 'Velo3D, news spike'},
    'ADUR': {'cap_b': 0.473,'price': 16.82,  'day_pct': 19.07, 'country': 'Canada','note': 'Aduro Clean Tech (Kanada)'},
    'CECO': {'cap_b': 2.80, 'price': 93.48,  'day_pct': 13.11, 'country': 'USA',   'note': 'CECO Environmental'},
    'OIO':  {'cap_b': 0.689,'price': 2.11,   'day_pct': 8.21,  'country': 'foreign','note': 'OIO Group, ordinary shares'},
}


def analyze_universe_filters():
    """ADIM A — bizim sabit Finviz filtrelerimize karşı her hisseyi analitik test et."""
    from swing_trader.small_cap.settings_config import load_settings
    s = load_settings()
    us = s.universe_scan

    # Finviz sorgularındaki sabit kısıtlar (universe.py)
    CAP_MIN_B, CAP_MAX_B = 0.300, 2.0      # 'Small ($300mln to $2bln)'
    FINVIZ_PRICE_MIN = 7.0                  # 'Price: Over $7'
    POST_PRICE_MIN = us.post_filter_price_min
    POST_PRICE_MAX = us.post_filter_price_max
    COUNTRY = 'USA'

    print("=" * 92)
    print(" ADIM A — UNIVERSE FİLTRESİ:  Hisse bizim Finviz evrenimize GİRİYOR mu?")
    print("=" * 92)
    print(f"  Sabit kısıtlar: Cap ${CAP_MIN_B*1000:.0f}M-${CAP_MAX_B}B | "
          f"Finviz Price>${FINVIZ_PRICE_MIN} | Post-filter Price ${POST_PRICE_MIN}-${POST_PRICE_MAX} | "
          f"Country={COUNTRY}")
    print("-" * 92)

    for t, d in MISSED.items():
        reasons = []
        if d['cap_b'] > CAP_MAX_B:
            reasons.append(f"CAP ${d['cap_b']}B > ${CAP_MAX_B}B tavan ({d['cap_b']/CAP_MAX_B:.1f}x aşım)")
        if d['cap_b'] < CAP_MIN_B:
            reasons.append(f"CAP ${d['cap_b']*1000:.0f}M < ${CAP_MIN_B*1000:.0f}M taban")
        eff_price_min = max(FINVIZ_PRICE_MIN, POST_PRICE_MIN)
        if d['price'] < eff_price_min:
            reasons.append(f"PRICE ${d['price']} < ${eff_price_min} taban")
        if d['price'] > POST_PRICE_MAX:
            reasons.append(f"PRICE ${d['price']} > ${POST_PRICE_MAX} tavan")
        if d['country'] != 'USA':
            reasons.append(f"COUNTRY={d['country']} (USA-only filtre eler)")

        if reasons:
            verdict = "❌ EVRENE GİRMEZ"
        else:
            verdict = "✅ evrene girer (chase cezası downstream'de)"
        print(f"  {t:5s} | cap ${d['cap_b']}B | ${d['price']:7.2f} | gün {d['day_pct']:+.1f}% | {d['note']}")
        print(f"        → {verdict}")
        for r in reasons:
            print(f"            • {r}")
    print()


def analyze_engine():
    """ADIM B — veri çekilebilen hisseler için scan_stock rejection sebebini bul."""
    print("=" * 92)
    print(" ADIM B — ENGINE TRACE:  Veri çekilebilenleri scan_stock'tan geçir")
    print("=" * 92)
    try:
        from swing_trader.small_cap.engine import SmallCapEngine
        from swing_trader.data.fetcher import DataFetcher
    except Exception as e:
        print(f"  ⚠️ Engine/Fetcher import edilemedi: {e}")
        return

    engine = SmallCapEngine()
    fetcher = DataFetcher()

    for i, t in enumerate(MISSED.keys()):
        try:
            df = fetcher.fetch_stock_data(t, period='4mo')
        except Exception as e:
            print(f"  {t:5s} | ❌ veri çekilemedi: {str(e)[:60]}")
            if i < len(MISSED) - 1:
                time.sleep(2)
            continue

        if df is None or len(df) < 50:
            n = 0 if df is None else len(df)
            print(f"  {t:5s} | ⚠️ yetersiz veri ({n} bar) — yeni/illikit olabilir")
            if i < len(MISSED) - 1:
                time.sleep(2)
            continue

        try:
            rc = {}
            sig = engine.scan_stock(t, df, backtest_mode=True, portfolio_value=10000,
                                    reject_counts=rc)
            # Ek teşhis metrikleri (ekran görüntüsündeki günlük % ile karşılaştır)
            close = df['Close'].astype(float)
            r5 = (float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else 0
            ma20 = float(close.tail(20).mean())
            ma20_dist = (float(close.iloc[-1]) / ma20 - 1) * 100 if ma20 > 0 else 0
            if sig:
                print(f"  {t:5s} | ✅ SİNYAL | Tip:{sig.get('swing_type')} "
                      f"QS:{sig.get('quality_score',0):.1f} RSI:{sig.get('rsi',0):.0f} "
                      f"5D:{sig.get('five_day_return',0):+.1f}%")
            else:
                reason = list(rc.keys())[0] if rc else 'unknown'
                print(f"  {t:5s} | ❌ REJECT → '{reason}'  "
                      f"(5D:{r5:+.1f}%  MA20'den:{ma20_dist:+.1f}%)")
        except Exception as e:
            print(f"  {t:5s} | ❌ scan_stock hata: {str(e)[:70]}")

        if i < len(MISSED) - 1:
            time.sleep(2)
    print()


def check_live_universe():
    """ADIM C — bugünkü gerçek Finviz evrenimizi çek, 6 hisse içinde mi bak."""
    print("=" * 92)
    print(" ADIM C — CANLI EVREN:  Bugünkü gerçek Finviz evrenimizde bu 6'sı var mı?")
    print("=" * 92)
    try:
        from swing_trader.small_cap.universe import SmallCapUniverse
        uni = SmallCapUniverse()
        tickers = uni.get_universe(use_finviz=True, max_tickers=200, force_refresh=True)
        print(f"  Evren boyutu: {len(tickers)} ticker")
        found = [t for t in MISSED if t in tickers]
        missing = [t for t in MISSED if t not in tickers]
        print(f"  ✅ Evrende OLAN  : {found if found else '(hiçbiri)'}")
        print(f"  ❌ Evrende OLMAYAN: {missing}")
    except Exception as e:
        print(f"  ⚠️ Canlı evren çekilemedi: {str(e)[:80]}")
    print()


if __name__ == '__main__':
    print("\n" + "█" * 92)
    print(" MISSED MOVERS DIAGNOSTIC — 'Midas'taki bu hisseleri neden bulamadık?'")
    print("█" * 92 + "\n")
    analyze_universe_filters()
    check_live_universe()
    analyze_engine()
    print("=" * 92)
    print(" BİTTİ — yukarıdaki tablo her hissenin nerede öldüğünü gösterir.")
    print("=" * 92)
