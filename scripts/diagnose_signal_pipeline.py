"""
SIGNAL PIPELINE DIAGNOSTIC — "Sistem tam olarak ne buluyor?"

Bu script:
1. Gerçek small-cap momentum ticker'ları kullanır (market cap $250M-$2.5B)
2. DataFetcher üzerinden veri çeker (API ile aynı yol — rate limit sorunsuz)
3. Pipeline'ı adım adım izler: filter → trigger → swing_confirmation → score
4. Her adımda neyin geçtiğini, neyin elemlendiğini raporlar
5. Geçen hisseler için forward return (R5, R10) hesaplar

v2.0: Proper small-cap universe, DataFetcher entegrasyonu, robust rate-limit handling
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import time

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from swing_trader.small_cap.engine import SmallCapEngine
from swing_trader.data.fetcher import DataFetcher

# ============================================================
# GERÇEK SMALL-CAP TICKERS — Market Cap $250M-$2.5B, likit, aktif
# ============================================================
# NOT: CAVA ($19B), IONQ ($8B), HIMS ($8B), CRDO ($10B) gibi
# mid/large-cap hisseler listeden çıkartıldı. Bu hisseler
# market cap filtresi tarafından zaten reject ediliyordu.
#
# Bu liste gerçek small-cap momentum hisselerinden oluşur.
DIAG_TICKERS = [
    # Uzay/Savunma — true small-cap
    'RKLB', 'LUNR', 'ACHR',
    # AI/Tech — small-cap aralığında
    'BBAI', 'SOUN',
    # Nükleer/Enerji
    'OKLO', 'SMR', 'NNE',
    # EV / eVTOL / Mobility
    'JOBY', 'QS', 'BLNK',
    # Data Center / Infra
    'APLD', 'AEHR',
    # Biotech/Healthcare (true small-cap)
    'AVXL', 'GERN', 'BTAI',
    # Ek small-cap momentum
    'UUUU', 'NMRA', 'NVCR',
    'RDW', 'LILM', 'ARQQ',
]


def fetch_all_data(tickers: list, fetcher: DataFetcher) -> dict:
    """
    DataFetcher ile veri çek — API ve Manual Lookup ile TAM AYNI yol.
    Yahoo session yönetimi, retry, validation hepsi DataFetcher'da.

    Rate limit koruması:
    - Her ticker arası 2 saniye bekleme
    - Rate limit hatası alırsa 10-30 saniye bekleyip tekrar dene
    - Max 3 retry per ticker
    """
    data_dict = {}
    failed = []

    print(f"  DataFetcher ile çekiliyor ({len(tickers)} ticker)...")
    print(f"  (Rate limit koruması: ticker arası 2s, retry arası 10-30s)")

    for i, ticker in enumerate(tickers):
        success = False
        for attempt in range(3):
            try:
                df = fetcher.fetch_stock_data(ticker, period='4mo')
                if df is not None and len(df) >= 20:
                    data_dict[ticker] = df
                    print(f"    [{i+1:2d}/{len(tickers)}] ✅ {ticker}: {len(df)} bar")
                    success = True
                    break
                else:
                    bar_count = len(df) if df is not None else 0
                    if attempt < 2:
                        print(f"    [{i+1:2d}/{len(tickers)}] ⏳ {ticker}: {bar_count} bar, retry {attempt+1}...")
                        time.sleep(5)
                    else:
                        print(f"    [{i+1:2d}/{len(tickers)}] ⚠️ {ticker}: {bar_count} bar (yetersiz)")
                        failed.append(ticker)
                        success = True  # Don't retry further
                    break
            except Exception as e:
                err_str = str(e)
                if 'Rate' in err_str or 'Too Many' in err_str or '429' in err_str:
                    wait = 10 * (attempt + 1)  # 10s, 20s, 30s
                    if attempt < 2:
                        print(f"    [{i+1:2d}/{len(tickers)}] ⏳ {ticker}: Rate limited, {wait}s bekleniyor...")
                        time.sleep(wait)
                    else:
                        print(f"    [{i+1:2d}/{len(tickers)}] ❌ {ticker}: Rate limited (3 retry başarısız)")
                        failed.append(ticker)
                else:
                    print(f"    [{i+1:2d}/{len(tickers)}] ❌ {ticker}: {err_str[:50]}")
                    failed.append(ticker)
                    break  # Non-rate-limit errors don't retry

        # Rate limit koruması: her ticker arası 2 saniye
        if i < len(tickers) - 1:
            time.sleep(2)

    if failed:
        print(f"\n  ⚠️ {len(failed)} ticker başarısız: {', '.join(failed)}")

    return data_dict


def check_filters_manually(ticker: str, df: pd.DataFrame) -> dict:
    """Filter kontrolünü elle yap ve neyin geçtiğini raporla."""
    result = {
        'has_data': len(df) >= 50,
        'price': 0,
        'volume_avg': 0,
        'atr_pct': 0,
        'close_above_ma20': False,
        'five_day_return': 0,
        'rsi_14': 0,
        'obv_trend': 'unknown',
    }

    if len(df) < 50:
        return result

    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    volume = df['Volume'].astype(float)

    result['price'] = round(float(close.iloc[-1]), 2)
    result['volume_avg'] = int(volume.tail(20).mean())

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_14 = float(tr.tail(14).mean())
    result['atr_pct'] = round(atr_14 / float(close.iloc[-1]) * 100, 2) if close.iloc[-1] > 0 else 0

    # MA20
    ma20 = float(close.tail(20).mean())
    result['close_above_ma20'] = float(close.iloc[-1]) > ma20
    result['ma20'] = round(ma20, 2)
    result['ma20_distance_pct'] = round((float(close.iloc[-1]) / ma20 - 1) * 100, 2) if ma20 > 0 else 0

    # MA50
    if len(close) >= 50:
        ma50 = float(close.tail(50).mean())
        result['ma50'] = round(ma50, 2)
        result['close_above_ma50'] = float(close.iloc[-1]) > ma50
        result['ma50_distance_pct'] = round((float(close.iloc[-1]) / ma50 - 1) * 100, 2) if ma50 > 0 else 0

    # 5-day return
    if len(close) >= 6:
        result['five_day_return'] = round((float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100, 2)

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    result['rsi_14'] = round(float(rsi.iloc[-1]), 1) if not pd.isna(rsi.iloc[-1]) else 0

    # Volume surge (bugünkü hacim / 20 günlük ort)
    vol_avg_20 = float(volume.tail(20).mean())
    vol_today = float(volume.iloc[-1])
    result['volume_surge'] = round(vol_today / vol_avg_20, 2) if vol_avg_20 > 0 else 0

    # OBV trend
    obv = (np.sign(close.diff()) * volume).cumsum()
    obv_5d_slope = float(obv.iloc[-1] - obv.iloc[-6]) if len(obv) >= 6 else 0
    result['obv_trend'] = 'accumulation' if obv_5d_slope > 0 else 'distribution'
    result['obv_5d_slope'] = round(obv_5d_slope, 0)

    # MA20 slope (son 5 gün)
    ma20_series = close.rolling(20).mean()
    if len(ma20_series.dropna()) >= 5:
        ma20_slope = float(ma20_series.iloc[-1] - ma20_series.iloc[-5])
        result['ma20_slope'] = 'rising' if ma20_slope > 0 else 'falling'

    # Trend phase
    if len(close) >= 20:
        vol_recent = float(volume.iloc[-10:].mean())
        vol_prior = float(volume.iloc[-20:-10].mean())
        vol_expanding = vol_recent > vol_prior * 1.1 if vol_prior > 0 else False
        price_rising = float(close.iloc[-1]) > float(close.iloc[-10])
        if price_rising and vol_expanding:
            result['trend_phase'] = 'markup'
        elif price_rising and not vol_expanding:
            result['trend_phase'] = 'late_markup'
        elif not price_rising and vol_expanding:
            result['trend_phase'] = 'distribution'
        else:
            result['trend_phase'] = 'markdown'

    return result


def run_engine_scan(engine: SmallCapEngine, ticker: str, df: pd.DataFrame) -> dict:
    """Engine scan_stock'u çalıştır ve sonucu döndür."""
    try:
        signal = engine.scan_stock(ticker, df, backtest_mode=True, portfolio_value=10000)
        if signal:
            return {
                'passed': True,
                'quality_score': signal.get('quality_score', 0),
                'swing_type': signal.get('swing_type', '?'),
                'rsi': signal.get('rsi', 0),
                'five_day_return': signal.get('five_day_return', 0),
                'volume_surge': signal.get('volume_surge', 0),
                'entry_price': signal.get('entry_price', 0),
                'stop_loss': signal.get('stop_loss', 0),
                'target_1': signal.get('target_1', 0),
                'target_2': signal.get('target_2', 0),
                'obv_accumulation': signal.get('obv_accumulation', False),
                'obv_distribution': signal.get('obv_distribution', False),
                'trend_phase': signal.get('swing_details', {}).get(
                    'trend_quality', {}
                ).get('trend_phase', '?'),
                'trend_strength': signal.get('swing_details', {}).get(
                    'trend_quality', {}
                ).get('trend_strength', 0),
            }
        return {'passed': False, 'reason': 'scan_stock returned None'}
    except Exception as e:
        return {'passed': False, 'reason': str(e)}


def calculate_forward_returns(df: pd.DataFrame) -> dict:
    """Sinyal gününden sonraki 5 ve 10 günlük getirileri hesapla."""
    result = {'R5': None, 'R10': None, 'MFE_10': None, 'MAE_10': None}

    close = df['Close'].astype(float).values
    high = df['High'].astype(float).values
    low = df['Low'].astype(float).values

    if len(close) < 25:
        return result

    signal_idx = len(close) - 15  # 15 gün önceki bar
    entry_price = close[signal_idx]

    if entry_price <= 0:
        return result

    # R5
    if signal_idx + 5 < len(close):
        result['R5'] = round((close[signal_idx + 5] / entry_price - 1) * 100, 2)

    # R10
    if signal_idx + 10 < len(close):
        result['R10'] = round((close[signal_idx + 10] / entry_price - 1) * 100, 2)

    # MFE (10 gün içinde en yüksek)
    fwd_window = min(10, len(high) - signal_idx - 1)
    if fwd_window > 0:
        max_high = high[signal_idx + 1: signal_idx + 1 + fwd_window].max()
        result['MFE_10'] = round((max_high / entry_price - 1) * 100, 2)

    # MAE (10 gün içinde en düşük)
    if fwd_window > 0:
        min_low = low[signal_idx + 1: signal_idx + 1 + fwd_window].min()
        result['MAE_10'] = round((min_low / entry_price - 1) * 100, 2)

    result['signal_date'] = str(df['Date'].iloc[signal_idx])[:10] if 'Date' in df.columns else '?'
    result['signal_price'] = round(entry_price, 2)

    return result


def main():
    print("=" * 90)
    print(" 🔍 SIGNAL PIPELINE DIAGNOSTIC v2.0 — Gerçek Small-Cap Analizi")
    print("=" * 90)

    engine = SmallCapEngine()
    fetcher = DataFetcher()

    # Veri topla
    print(f"\n📥 {len(DIAG_TICKERS)} gerçek small-cap ticker için veri çekiliyor...")
    print(f"   (DataFetcher kullanılıyor — Manual Lookup ile aynı yol)")
    data_dict = fetch_all_data(DIAG_TICKERS, fetcher)

    print(f"\n📊 Veri başarılı: {len(data_dict)}/{len(DIAG_TICKERS)} ticker")

    if len(data_dict) == 0:
        print("\n  ❌ HİÇBİR TICKER İÇİN VERİ ÇEKİLEMEDİ!")
        print("     İnternet bağlantınızı kontrol edin.")
        return

    # ============================================================
    # ADIM 1: Manuel teknik analiz
    # ============================================================
    print("\n" + "=" * 90)
    print(" 📈 ADIM 1: TEKNİK ANALİZ TARAMASI")
    print("=" * 90)

    tech_results = {}
    for ticker, df in sorted(data_dict.items()):
        ta = check_filters_manually(ticker, df)
        tech_results[ticker] = ta

        status_icons = []
        if ta.get('close_above_ma20'):
            status_icons.append("MA20✅")
        else:
            status_icons.append("MA20❌")
        if ta.get('close_above_ma50'):
            status_icons.append("MA50✅")
        else:
            status_icons.append("MA50❌")
        if ta.get('obv_trend') == 'accumulation':
            status_icons.append("OBV↑")
        else:
            status_icons.append("OBV↓")
        if ta.get('ma20_slope') == 'rising':
            status_icons.append("Slope↑")
        else:
            status_icons.append("Slope↓")

        phase = ta.get('trend_phase', '?')
        phase_icon = {'markup': '🟢', 'late_markup': '🟡', 'distribution': '🔴', 'markdown': '⬛'}.get(phase, '❓')

        vol_surge_icon = "🔥" if ta.get('volume_surge', 0) >= 1.5 else "  "

        print(f"  {ticker:5s} | ${ta['price']:7.2f} | RSI:{ta['rsi_14']:5.1f} | "
              f"5D:{ta['five_day_return']:+6.1f}% | ATR:{ta['atr_pct']:4.1f}% | "
              f"VolSurge:{ta.get('volume_surge', 0):4.1f}x {vol_surge_icon} | "
              f"{' '.join(status_icons)} | {phase_icon}{phase}")

    # ============================================================
    # ADIM 2: Engine Scan
    # ============================================================
    print("\n" + "=" * 90)
    print(" 🎯 ADIM 2: ENGINE SCAN — SmallCapEngine.scan_stock() sonuçları")
    print("=" * 90)

    signals_found = []
    signals_rejected = []

    for ticker, df in sorted(data_dict.items()):
        result = run_engine_scan(engine, ticker, df)
        if result['passed']:
            signals_found.append({'ticker': ticker, **result})
            qs = result['quality_score']
            st = result['swing_type']
            phase = result.get('trend_phase', '?')
            ts = result.get('trend_strength', 0)
            print(f"  ✅ {ticker:5s} | Tip:{st} | QS:{qs:5.1f} | RSI:{result.get('rsi', 0):.0f} | "
                  f"5D:{result.get('five_day_return', 0):+.1f}% | "
                  f"Phase:{phase}({ts})")
        else:
            signals_rejected.append({'ticker': ticker, **result})
            reason = result.get('reason', 'unknown')[:65]
            print(f"  ❌ {ticker:5s} | {reason}")

    pass_rate = len(signals_found) / len(data_dict) * 100 if data_dict else 0
    print(f"\n  📊 Sonuç: {len(signals_found)} sinyal / {len(data_dict)} ticker "
          f"({pass_rate:.0f}% pass rate)")

    # ============================================================
    # ADIM 3: Forward Return Analizi
    # ============================================================
    print("\n" + "=" * 90)
    print(" 💰 ADIM 3: FORWARD RETURN ANALİZİ — '15 gün önce sinyal versek ne olurdu?'")
    print("=" * 90)

    fwd_results = []
    for ticker, df in sorted(data_dict.items()):
        fwd = calculate_forward_returns(df)
        if fwd['R5'] is not None:
            fwd_results.append({'ticker': ticker, **fwd})
            r5_icon = "✅" if fwd['R5'] > 0 else "❌"
            r10_icon = "✅" if (fwd.get('R10') or 0) > 0 else "❌"
            print(f"  {ticker:5s} | Sinyal: {fwd.get('signal_date', '?')} @ ${fwd.get('signal_price', 0):.2f} | "
                  f"R5:{fwd['R5']:+6.1f}% {r5_icon} | R10:{fwd.get('R10', 'n/a')} {r10_icon} | "
                  f"MFE:{fwd.get('MFE_10', 'n/a')}% | MAE:{fwd.get('MAE_10', 'n/a')}%")

    # İstatistikler
    if fwd_results:
        r5_vals = [f['R5'] for f in fwd_results if f['R5'] is not None]
        r10_vals = [f['R10'] for f in fwd_results if f.get('R10') is not None]
        mfe_vals = [f['MFE_10'] for f in fwd_results if f.get('MFE_10') is not None]
        mae_vals = [f['MAE_10'] for f in fwd_results if f.get('MAE_10') is not None]

        print(f"\n  📊 ÖZET İSTATİSTİKLER:")
        if r5_vals:
            r5_positive = sum(1 for v in r5_vals if v > 0)
            print(f"     R5  Medyan: {np.median(r5_vals):+.1f}% | Ort: {np.mean(r5_vals):+.1f}% | "
                  f"Pozitif: {r5_positive}/{len(r5_vals)} ({r5_positive/len(r5_vals)*100:.0f}%)")
        if r10_vals:
            r10_positive = sum(1 for v in r10_vals if v > 0)
            print(f"     R10 Medyan: {np.median(r10_vals):+.1f}% | Ort: {np.mean(r10_vals):+.1f}% | "
                  f"Pozitif: {r10_positive}/{len(r10_vals)} ({r10_positive/len(r10_vals)*100:.0f}%)")
        if mfe_vals:
            print(f"     MFE Medyan: {np.median(mfe_vals):+.1f}% | (10 gün içinde max yükseliş)")
        if mae_vals:
            print(f"     MAE Medyan: {np.median(mae_vals):+.1f}% | (10 gün içinde max düşüş)")

    # ============================================================
    # ADIM 4: Sinyal verilen hisseler
    # ============================================================
    if signals_found:
        print("\n" + "=" * 90)
        print(" 🏆 ADIM 4: SİNYAL VERİLEN HİSSELERİN DETAYLI ANALİZİ")
        print("=" * 90)

        for sig in sorted(signals_found, key=lambda x: x['quality_score'], reverse=True):
            ticker = sig['ticker']
            ta = tech_results.get(ticker, {})
            fwd = next((f for f in fwd_results if f['ticker'] == ticker), {})

            print(f"\n  {'─'*60}")
            print(f"  {ticker} | Tip: {sig['swing_type']} | Quality Score: {sig['quality_score']:.1f}")
            print(f"  {'─'*60}")
            print(f"    Fiyat: ${ta.get('price', 0):.2f} | RSI: {ta.get('rsi_14', 0):.1f}")
            print(f"    5D Return: {ta.get('five_day_return', 0):+.1f}% | ATR: {ta.get('atr_pct', 0):.1f}%")
            print(f"    Volume Surge: {ta.get('volume_surge', 0):.1f}x | OBV: {ta.get('obv_trend', '?')}")
            print(f"    MA20: {ta.get('close_above_ma20', '?')} (dist: {ta.get('ma20_distance_pct', 0):+.1f}%)")
            print(f"    MA50: {ta.get('close_above_ma50', '?')} (dist: {ta.get('ma50_distance_pct', 0):+.1f}%)")
            print(f"    MA20 Slope: {ta.get('ma20_slope', '?')}")
            print(f"    Trend Phase: {sig.get('trend_phase', '?')} (strength: {sig.get('trend_strength', 0)})")

            # Forward return
            if fwd:
                print(f"    📈 Forward: R5={fwd.get('R5', 'n/a')}% | R10={fwd.get('R10', 'n/a')}% | "
                      f"MFE={fwd.get('MFE_10', 'n/a')}% | MAE={fwd.get('MAE_10', 'n/a')}%")

            # Potansiyel sorun tespiti
            issues = []
            if ta.get('obv_trend') == 'distribution':
                issues.append("⚠️ OBV Distribution — akıllı para çıkıyor olabilir")
            if ta.get('ma20_slope') == 'falling':
                issues.append("⚠️ MA20 düşüşte — trend zayıflıyor")
            if ta.get('rsi_14', 0) > 70:
                issues.append("⚠️ RSI > 70 — overbought")
            if ta.get('five_day_return', 0) > 20:
                issues.append("⚠️ 5D > +20% — late entry riski")
            if not ta.get('close_above_ma50'):
                issues.append("⚠️ Close < MA50 — uzun vadeli trend down")
            if ta.get('trend_phase') in ('distribution', 'markdown'):
                issues.append("⚠️ Trend phase: " + ta.get('trend_phase', '?'))

            if issues:
                print(f"    🔴 SORUNLAR:")
                for issue in issues:
                    print(f"       {issue}")
            else:
                print(f"    🟢 Sorun tespit edilmedi")

    # ============================================================
    # ADIM 5: SONUÇ VE ÖNERİLER
    # ============================================================
    print("\n" + "=" * 90)
    print(" 📋 SONUÇ VE ÖNERİLER")
    print("=" * 90)

    total = len(data_dict)
    found = len(signals_found)

    print(f"\n  Taranan: {total} ticker (gerçek small-cap, $250M-$2.5B)")
    print(f"  Sinyal: {found} ({found/total*100:.0f}%)" if total > 0 else f"  Sinyal: {found}")

    if found > 0:
        avg_qs = np.mean([s['quality_score'] for s in signals_found])
        type_dist = {}
        for s in signals_found:
            st = s['swing_type']
            type_dist[st] = type_dist.get(st, 0) + 1
        print(f"  Ortalama QS: {avg_qs:.1f}")
        print(f"  Tip dağılımı: {dict(sorted(type_dist.items()))}")

        # Gate effectiveness
        dist_signals = sum(1 for s in signals_found
                          if tech_results.get(s['ticker'], {}).get('obv_trend') == 'distribution')
        fall_signals = sum(1 for s in signals_found
                          if tech_results.get(s['ticker'], {}).get('ma20_slope') == 'falling')
        below_ma50 = sum(1 for s in signals_found
                        if not tech_results.get(s['ticker'], {}).get('close_above_ma50', True))

        print(f"\n  🛡️ GATE ETKİNLİĞİ:")
        if dist_signals > 0:
            print(f"    ⚠️ {dist_signals}/{found} sinyal OBV DISTRIBUTION — "
                  f"OBV hard gate'i atlatmış olabilir!")
        else:
            print(f"    ✅ OBV Distribution hard gate: Tüm distribution hisseleri engellendi")

        if fall_signals > 0:
            print(f"    ⚠️ {fall_signals}/{found} sinyal MA20 FALLING — "
                  f"trend quality gate atlatmış!")
        else:
            print(f"    ✅ MA20 slope gate: Tüm falling trend hisseleri engellendi")

        if below_ma50 > 0:
            print(f"    ⚠️ {below_ma50}/{found} sinyal MA50 ALTINDA")
        else:
            print(f"    ✅ MA50 gate: Tüm uzun vadeli downtrend hisseleri engellendi")
    else:
        print("\n  ℹ️ Hiç sinyal yok — mevcut piyasa koşullarında normal olabilir.")
        print("     Bu mutlaka bir sorun DEĞİL — sistem sadece yüksek kaliteli setuplara giriyor.")
        print("     Kontrol noktaları:")
        print("     1. Filtreler: min_atr=3%, vol_surge=1.5x, market_cap=$250M-$2.5B")
        print("     2. Swing confirmation: 5d_mom>0, Close>MA20, MA20 slope↑, MA50 OK")
        print("     3. Hard gates: OBV distribution, trend phase, RSI overbought")

    # JSON çıktı
    output = {
        'timestamp': datetime.now().isoformat(),
        'version': '2.0',
        'tickers_scanned': total,
        'signals_found': found,
        'signals': [{
            'ticker': s['ticker'],
            'quality_score': s['quality_score'],
            'swing_type': s['swing_type'],
            'rsi': s.get('rsi', 0),
            'five_day_return': s.get('five_day_return', 0),
            'volume_surge': s.get('volume_surge', 0),
            'trend_phase': s.get('trend_phase', '?'),
            'trend_strength': s.get('trend_strength', 0),
        } for s in signals_found],
        'rejected': [{
            'ticker': r['ticker'],
            'reason': r.get('reason', 'unknown')[:100],
        } for r in signals_rejected],
        'tech_analysis': {
            t: {k: v for k, v in ta.items() if k != 'has_data'}
            for t, ta in tech_results.items()
        },
        'forward_returns': fwd_results,
    }

    os.makedirs('output', exist_ok=True)
    with open('output/signal_diagnostic.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  📁 Detaylı sonuçlar: output/signal_diagnostic.json")
    print("=" * 90)


if __name__ == '__main__':
    main()
