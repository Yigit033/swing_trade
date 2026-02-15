"""
ALL-SEASON STRESS TEST — "Bu sistem her mevsimde para kazandirir mi?"

4 FARKLI MARKET KOSULU:
1. BULL  : 2024 Oct-Dec (Election rally, AI hype, small-cap boom)
2. BEAR  : 2022 Jun-Aug (Fed rate hikes, S&P -20%, small-cap crash)
3. SIDEWAYS: 2023 Aug-Oct (Range-bound, low VIX, S&P choppy)
4. CURRENT : 2025 Nov - 2026 Feb (our baseline test)

HER SEZON AYNI SITEM, AYNI PARAMETRELER.
Tek fark: market kosullari.
"""
import sys, time
sys.path.insert(0, '.')

from swing_trader.small_cap.smallcap_backtest import SmallCapBacktester

# Genis small-cap evreni — farkli dönemlerde farkli hisseler pass edecek
# Önemli: filtreler (market cap, float, ATR, volume) otomatik olarak
# o dönemde uygun olmayanları eleyecek
TICKERS = [
    # Momentum small-caps (mevcut test listesi)
    'AEHR', 'AXTI', 'VELO', 'NMRA', 'NVCR',
    'SOUN', 'BBAI', 'RKLB', 'ASTS', 'IONQ',
    'ACHR', 'BTAI', 'GERN', 'LUNR',
    'UUUU', 'SMR', 'OKLO', 'NNE',
    'QS', 'BLNK', 'APLD',
    # EK: 2022-2023'te aktif olmus small-cap momentum hisseleri
    'CLOV', 'WISH', 'IRNT', 'ATER',
    'SKLZ', 'OPEN', 'SOFI', 'PLTR',
    'DNA', 'MVST', 'LAZR', 'STEM',
    'RDW', 'JOBY', 'LILM', 'ARQQ',
]

SEASONS = [
    {
        'name': 'BEAR (2022 Jun-Aug)',
        'label': 'BEAR',
        'start': '2022-06-01',
        'end': '2022-08-31',
        'emoji': '🐻',
        'context': 'Fed aggressive rate hikes, S&P -20%, crypto crash, small-cap massacre'
    },
    {
        'name': 'SIDEWAYS (2023 Aug-Oct)',
        'label': 'SIDEWAYS',
        'start': '2023-08-01',
        'end': '2023-10-31',
        'emoji': '😐',
        'context': 'Range-bound S&P 4300-4600, low VIX, sector rotation, no clear direction'
    },
    {
        'name': 'BULL (2024 Oct-Dec)',
        'label': 'BULL',
        'start': '2024-10-01',
        'end': '2024-12-31',
        'emoji': '🐂',
        'context': 'Election rally, AI hype, rate cut expectations, small-cap boom'
    },
    {
        'name': 'CURRENT (2025 Nov - 2026 Feb)',
        'label': 'CURRENT',
        'start': '2025-11-15',
        'end': '2026-02-12',
        'emoji': '📊',
        'context': 'Baseline V3 test period'
    }
]

all_results = []
lines = []
lines.append("=" * 90)
lines.append(" ALL-SEASON STRESS TEST — Her Mevsimde Para Kazandirir mi?")
lines.append(" Ayni sistem, ayni parametreler, 4 farkli market kosulu")
lines.append("=" * 90)

for i, season in enumerate(SEASONS):
    print(f"\n{'='*70}")
    print(f" {season['emoji']} SEZON {i+1}/4: {season['name']}")
    print(f" {season['context']}")
    print(f"{'='*70}")
    
    bt = SmallCapBacktester()
    
    try:
        results = bt.run_backtest(
            tickers=TICKERS,
            start_date=season['start'],
            end_date=season['end'],
            initial_capital=10000,
            max_concurrent=3,
            progress_callback=lambda p, m: print(f"  [{p:3d}%] {m}")
        )
        
        M = results['metrics']
        trades = results['trades']
        
        # Per-type stats
        type_summary = {}
        for stype, stats in M.get('type_stats', {}).items():
            total_t = stats['wins'] + stats['losses']
            wr = stats['wins'] / total_t if total_t > 0 else 0
            total_pnl = stats['total_pnl']
            type_summary[stype] = {'trades': total_t, 'wr': wr, 'total_pnl': total_pnl}
        
        # Exit stats
        exit_summary = {}
        for reason, stats in M.get('exit_stats', {}).items():
            exit_summary[reason] = {'count': stats['count'], 'avg_pnl': stats['avg_pnl']}
        
        # Per-ticker summary
        tk_stats = {}
        for t in trades:
            tk = t['ticker']
            if tk not in tk_stats:
                tk_stats[tk] = {'trades': 0, 'wins': 0, 'pnl_dollar': 0, 'pnl_pct': 0}
            tk_stats[tk]['trades'] += 1
            if t['pnl_pct'] > 0:
                tk_stats[tk]['wins'] += 1
            tk_stats[tk]['pnl_dollar'] += t['pnl_dollar']
            tk_stats[tk]['pnl_pct'] += t['pnl_pct']
        
        season_data = {
            'season': season,
            'metrics': M,
            'type_summary': type_summary,
            'exit_summary': exit_summary,
            'tk_stats': tk_stats,
            'trades': trades
        }
        all_results.append(season_data)
        
    except Exception as e:
        print(f"  HATA: {e}")
        all_results.append({
            'season': season,
            'metrics': {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                       'total_pnl_dollar': 0, 'total_return': 0, 'max_drawdown': 0,
                       'winning_trades': 0, 'losing_trades': 0, 'avg_win_pct': 0,
                       'avg_loss_pct': 0, 'avg_hold_days': 0, 'final_capital': 10000},
            'type_summary': {},
            'exit_summary': {},
            'tk_stats': {},
            'trades': [],
            'error': str(e)
        })
    
    time.sleep(1)  # Rate limit korunma

# ============================================================
# SONUC RAPORU
# ============================================================
lines.append("")
lines.append("-" * 90)
lines.append(f"  {'':22s}  {'BEAR':>10s}  {'SIDEWAYS':>10s}  {'BULL':>10s}  {'CURRENT':>10s}")
lines.append(f"  {'':22s}  {'2022Q3':>10s}  {'2023Q3':>10s}  {'2024Q4':>10s}  {'2025Q4':>10s}")
lines.append("-" * 90)

metrics_keys = [
    ('Toplam Trade', 'total_trades', 'd', ''),
    ('Win Rate', 'win_rate', '.0f', '%', lambda x: x*100),
    ('Kazanan', 'winning_trades', 'd', ''),
    ('Kaybeden', 'losing_trades', 'd', ''),
    ('Ort Kazanc %', 'avg_win_pct', '+.1f', '%'),
    ('Ort Kayip %', 'avg_loss_pct', '+.1f', '%'),
    ('Profit Factor', 'profit_factor', '.2f', ''),
    ('Toplam P/L $', 'total_pnl_dollar', '+,.0f', ''),
    ('Return %', 'total_return', '+.1f', '%', lambda x: x*100),
    ('Max Drawdown', 'max_drawdown', '.1f', '%'),
    ('Bitis Sermaye $', 'final_capital', ',.0f', ''),
]

for label, key, fmt, suffix, *transform in metrics_keys:
    vals = []
    for r in all_results:
        v = r['metrics'].get(key, 0)
        if transform:
            v = transform[0](v)
        vals.append(f"{v:{fmt}}{suffix}")
    lines.append(f"  {label:22s}  {vals[0]:>10s}  {vals[1]:>10s}  {vals[2]:>10s}  {vals[3]:>10s}")

lines.append("-" * 90)

# TYPE BREAKDOWN per season
lines.append("")
lines.append("--- TIP PERFORMANSI (Sezon Bazli) ---")
for i, r in enumerate(all_results):
    s = r['season']
    lines.append(f"  {s['emoji']} {s['label']}:")
    if r['type_summary']:
        for stype in ['A', 'C', 'B', 'S']:
            if stype in r['type_summary']:
                d = r['type_summary'][stype]
                lines.append(f"    Tip {stype}: {d['trades']} trade, WR: {d['wr']:.0%}, P/L: {d['total_pnl']:+.1f}%")
    else:
        lines.append(f"    (Trade yok)")

# EXIT BREAKDOWN per season
lines.append("")
lines.append("--- CIKIS TIPI (Sezon Bazli) ---")
for i, r in enumerate(all_results):
    s = r['season']
    lines.append(f"  {s['emoji']} {s['label']}:")
    if r['exit_summary']:
        for reason in ['TARGET', 'TRAILED', 'TIMEOUT', 'STOPPED', 'FORCED']:
            if reason in r['exit_summary']:
                d = r['exit_summary'][reason]
                lines.append(f"    {reason}: {d['count']} trade, ort: {d['avg_pnl']:+.1f}%")
    else:
        lines.append(f"    (Trade yok)")

# TOP PERFORMERS per season
lines.append("")
lines.append("--- HISSE BAZLI (Sezon Bazli) ---")
for i, r in enumerate(all_results):
    s = r['season']
    lines.append(f"  {s['emoji']} {s['label']}:")
    if r['tk_stats']:
        sorted_tk = sorted(r['tk_stats'].items(), key=lambda x: x[1]['pnl_dollar'], reverse=True)
        for tk, st in sorted_tk[:5]:  # Top 5
            wr = st['wins'] / st['trades'] if st['trades'] > 0 else 0
            icon = "[+]" if st['pnl_dollar'] > 0 else "[-]"
            lines.append(f"    {icon} {tk:5s}: {st['trades']} trade, WR: {wr:.0%}, P/L: ${st['pnl_dollar']:+.0f}")
    else:
        lines.append(f"    (Trade yok)")

# TRADE LOGS per season
for i, r in enumerate(all_results):
    s = r['season']
    lines.append("")
    lines.append(f"--- TRADE LOG: {s['emoji']} {s['name']} ---")
    if r['trades']:
        for t in r['trades']:
            icon = "W" if t['pnl_pct'] > 0 else "L"
            sp = t.get('stop_pct', 0)
            tp = t.get('target_pct', 0)
            shr = t.get('shares', 0)
            sm = t.get('stop_method', '?')
            lines.append(f"  [{icon}] {t['ticker']:5s} Tip{t.get('swing_type','?')} | "
                        f"{t['entry_date']} -> {t['exit_date']:10s} | "
                        f"${t['entry_price']:7.2f} -> ${t['exit_price']:7.2f} | "
                        f"{t['pnl_pct']:+6.1f}% ${t['pnl_dollar']:+8.1f} | "
                        f"stop:{sp:.0f}% tgt:{tp:.0f}% {sm} | {t['exit_reason']}")
    else:
        lines.append("  (Trade yok)")

# FINAL VERDICT
lines.append("")
lines.append("=" * 90)
lines.append(" FINAL SONUC")
lines.append("=" * 90)

profitable_seasons = sum(1 for r in all_results if r['metrics']['total_pnl_dollar'] > 0)
total_seasons = len(all_results)
total_pnl = sum(r['metrics']['total_pnl_dollar'] for r in all_results)
total_trades = sum(r['metrics']['total_trades'] for r in all_results)
total_wins = sum(r['metrics']['winning_trades'] for r in all_results)
overall_wr = total_wins / total_trades if total_trades > 0 else 0

lines.append(f"  Karli Sezon: {profitable_seasons}/{total_seasons}")
lines.append(f"  Toplam P/L (tum sezonlar): ${total_pnl:+,.0f}")
lines.append(f"  Toplam Trade: {total_trades}, Toplam Kazanan: {total_wins}, WR: {overall_wr:.0%}")

# Season-by-season summary
for r in all_results:
    s = r['season']
    M = r['metrics']
    icon = "✅" if M['total_pnl_dollar'] > 0 else "⚠️" if M['total_trades'] == 0 else "❌"
    lines.append(f"  {icon} {s['label']:10s}: ${M['total_pnl_dollar']:+,.0f} "
                f"({M['total_return']*100:+.1f}%) | {M['total_trades']} trade, "
                f"WR: {M['win_rate']*100:.0f}%, PF: {M['profit_factor']:.2f}")

if profitable_seasons == total_seasons:
    lines.append(f"\n  >>> HER MEVSIMDE KARLI! SISTEM CALISIYOR! <<<")
elif profitable_seasons >= total_seasons * 0.75:
    lines.append(f"\n  >>> COGU MEVSIMDE KARLI ({profitable_seasons}/{total_seasons}) - IYI SISTEM <<<")
elif profitable_seasons >= total_seasons * 0.5:
    lines.append(f"\n  >>> YARISINDA KARLI ({profitable_seasons}/{total_seasons}) - IYILESTIRME LAZIM <<<")
else:
    lines.append(f"\n  >>> COK AZ MEVSIMDE KARLI ({profitable_seasons}/{total_seasons}) - CIDDI SORUN <<<")

lines.append("=" * 90)

output = "\n".join(lines)
with open("all_season_results.txt", "w", encoding="utf-8") as f:
    f.write(output)

print(output)
print(f"\nSaved to all_season_results.txt")
