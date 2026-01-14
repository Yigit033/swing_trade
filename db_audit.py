"""
Comprehensive Database Quality Audit Script
Performs all validation checks on swing trading database
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import sys
import io

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Database path
DB_PATH = r"c:\swing_trade\data\stocks.db"

# US Market holidays 2024-2025
US_HOLIDAYS = [
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25"
]

def get_connection():
    return sqlite3.connect(DB_PATH)

def run_audit():
    conn = get_connection()
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    issues = []
    
    print("=" * 60)
    print("DATABASE QUALITY AUDIT REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Get table structure
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print(f"\nTables found: {tables['name'].tolist()}")
    
    # Get all data
    df = pd.read_sql("SELECT * FROM stock_data", conn)
    print(f"Total records: {len(df):,}")
    print(f"Unique tickers: {df['ticker'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Ensure proper column names
    df.columns = [c.lower() for c in df.columns]
    
    # Convert date
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    print("\n" + "=" * 60)
    print("[1] TIME & DATE INTEGRITY")
    print("=" * 60)
    
    # 1a. Future dates
    future_dates = df[df['date'] > datetime.now().date()]
    if len(future_dates) > 0:
        print(f"\n[X] FUTURE DATES FOUND: {len(future_dates)} records")
        for _, row in future_dates.head(10).iterrows():
            issues.append({
                'ticker': row['ticker'], 'date': str(row['date']),
                'issue_category': 'TIME_INTEGRITY', 'issue_description': 'Future date record',
                'severity': 'HIGH', 'trade_impact': 'Corrupts backtest timeline'
            })
    else:
        print("\n[OK] Future dates: NO ISSUES FOUND")
    
    # 1b. Weekend records
    df['weekday'] = pd.to_datetime(df['date']).dt.dayofweek
    weekend = df[df['weekday'] >= 5]
    if len(weekend) > 0:
        print(f"[X] WEEKEND RECORDS FOUND: {len(weekend)} records")
        for _, row in weekend.head(10).iterrows():
            issues.append({
                'ticker': row['ticker'], 'date': str(row['date']),
                'issue_category': 'TIME_INTEGRITY', 'issue_description': f"Weekend record (day {row['weekday']})",
                'severity': 'MEDIUM', 'trade_impact': 'Invalid trading day'
            })
    else:
        print("[OK] Weekend records: NO ISSUES FOUND")
    
    # 1c. Holiday records
    holiday_dates = [datetime.strptime(d, '%Y-%m-%d').date() for d in US_HOLIDAYS]
    holidays = df[df['date'].isin(holiday_dates)]
    if len(holidays) > 0:
        print(f"[!] HOLIDAY RECORDS FOUND: {len(holidays)} records")
        for _, row in holidays.head(5).iterrows():
            issues.append({
                'ticker': row['ticker'], 'date': str(row['date']),
                'issue_category': 'TIME_INTEGRITY', 'issue_description': 'US market holiday record',
                'severity': 'LOW', 'trade_impact': 'Minor - may be partial trading day'
            })
    else:
        print("[OK] Holiday records: NO ISSUES FOUND")
    
    # 1d. Duplicate records
    duplicates = df[df.duplicated(subset=['ticker', 'date'], keep=False)]
    if len(duplicates) > 0:
        print(f"[X] DUPLICATE RECORDS FOUND: {len(duplicates)} records")
        for _, row in duplicates.head(10).iterrows():
            issues.append({
                'ticker': row['ticker'], 'date': str(row['date']),
                'issue_category': 'TIME_INTEGRITY', 'issue_description': 'Duplicate ticker+date',
                'severity': 'HIGH', 'trade_impact': 'Corrupts signals and backtest'
            })
    else:
        print("[OK] Duplicate records: NO ISSUES FOUND")
    
    # 1e. Date gaps
    gap_issues = []
    sample_tickers = df['ticker'].unique()[:20]
    for ticker in sample_tickers:
        ticker_df = df[df['ticker'] == ticker].sort_values('date')
        if len(ticker_df) > 10:
            dates = pd.to_datetime(ticker_df['date'])
            date_diffs = dates.diff().dt.days
            large_gaps = date_diffs[date_diffs > 7]
            if len(large_gaps) > 0:
                for idx in large_gaps.index[:2]:
                    gap_issues.append({
                        'ticker': ticker, 'date': str(ticker_df.loc[idx, 'date']),
                        'issue_category': 'TIME_INTEGRITY', 'issue_description': f'Gap of {int(large_gaps[idx])} days',
                        'severity': 'MEDIUM', 'trade_impact': 'May miss trading opportunities'
                    })
    
    if gap_issues:
        print(f"[!] DATE GAPS FOUND: {len(gap_issues)} issues (sample)")
        issues.extend(gap_issues[:5])
    else:
        print("[OK] Date gaps: NO ISSUES FOUND")
    
    print("\n" + "=" * 60)
    print("[2] OHLC LOGIC VALIDATION")
    print("=" * 60)
    
    # 2a. High < Open or High < Close
    high_invalid = df[(df['high'] < df['open']) | (df['high'] < df['close'])]
    if len(high_invalid) > 0:
        print(f"\n[X] HIGH < OPEN/CLOSE: {len(high_invalid)} records")
        for _, row in high_invalid.head(5).iterrows():
            issues.append({
                'ticker': row['ticker'], 'date': str(row['date']),
                'issue_category': 'OHLC_LOGIC', 'issue_description': f"High({row['high']:.2f}) < Open({row['open']:.2f})/Close({row['close']:.2f})",
                'severity': 'HIGH', 'trade_impact': 'Invalid price data - corrupt indicators'
            })
    else:
        print("\n[OK] High >= Open/Close: NO ISSUES FOUND")
    
    # 2b. Low > Open or Low > Close
    low_invalid = df[(df['low'] > df['open']) | (df['low'] > df['close'])]
    if len(low_invalid) > 0:
        print(f"[X] LOW > OPEN/CLOSE: {len(low_invalid)} records")
        for _, row in low_invalid.head(10).iterrows():
            issues.append({
                'ticker': row['ticker'], 'date': str(row['date']),
                'issue_category': 'OHLC_LOGIC', 'issue_description': f"Low({row['low']:.2f}) > Open({row['open']:.2f})/Close({row['close']:.2f})",
                'severity': 'HIGH', 'trade_impact': 'Invalid price data - corrupt indicators'
            })
    else:
        print("[OK] Low <= Open/Close: NO ISSUES FOUND")
    
    # 2c. High == Low
    doji_bars = df[(df['high'] == df['low']) & (df['volume'] > 0)]
    if len(doji_bars) > 0:
        print(f"[!] HIGH == LOW BARS: {len(doji_bars)} records")
        for _, row in doji_bars.head(3).iterrows():
            issues.append({
                'ticker': row['ticker'], 'date': str(row['date']),
                'issue_category': 'OHLC_LOGIC', 'issue_description': f"Zero range bar at ${row['close']:.2f}",
                'severity': 'LOW', 'trade_impact': 'May indicate data issue or halted trading'
            })
    else:
        print("[OK] High == Low bars: NO ISSUES FOUND")
    
    # 2d. OHLC <= 0
    zero_prices = df[(df['open'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0) | (df['close'] <= 0)]
    if len(zero_prices) > 0:
        print(f"[X] ZERO/NEGATIVE PRICES: {len(zero_prices)} records")
        for _, row in zero_prices.head(5).iterrows():
            issues.append({
                'ticker': row['ticker'], 'date': str(row['date']),
                'issue_category': 'OHLC_LOGIC', 'issue_description': 'Zero or negative price',
                'severity': 'HIGH', 'trade_impact': 'Critical - breaks all calculations'
            })
    else:
        print("[OK] Zero/negative prices: NO ISSUES FOUND")
    
    print("\n" + "=" * 60)
    print("[3] VOLUME CONSISTENCY & ANOMALIES")
    print("=" * 60)
    
    # 3a. Volume = 0 with price movement
    df['price_change'] = abs(df['close'] - df['open'])
    zero_vol_move = df[(df['volume'] == 0) & (df['price_change'] > 0.01)]
    if len(zero_vol_move) > 0:
        print(f"\n[!] ZERO VOLUME WITH PRICE MOVEMENT: {len(zero_vol_move)} records")
        for _, row in zero_vol_move.head(5).iterrows():
            issues.append({
                'ticker': row['ticker'], 'date': str(row['date']),
                'issue_category': 'VOLUME_ANOMALY', 'issue_description': f"Volume=0 but price moved ${row['price_change']:.2f}",
                'severity': 'MEDIUM', 'trade_impact': 'Volume indicators unreliable'
            })
    else:
        print("\n[OK] Zero volume with movement: NO ISSUES FOUND")
    
    # 3b. Volume spikes
    volume_spikes = []
    for ticker in sample_tickers[:10]:
        ticker_df = df[df['ticker'] == ticker].sort_values('date').copy()
        if len(ticker_df) > 5:
            ticker_df['prev_vol'] = ticker_df['volume'].shift(1)
            ticker_df['vol_ratio'] = ticker_df['volume'] / ticker_df['prev_vol'].replace(0, 1)
            spikes = ticker_df[ticker_df['vol_ratio'] >= 10]
            for _, row in spikes.head(2).iterrows():
                volume_spikes.append({
                    'ticker': ticker, 'date': str(row['date']),
                    'issue_category': 'VOLUME_ANOMALY', 'issue_description': f"Volume spike {row['vol_ratio']:.1f}x",
                    'severity': 'LOW', 'trade_impact': 'May be legitimate news event'
                })
    
    if volume_spikes:
        print(f"[!] VOLUME SPIKES (10x+): {len(volume_spikes)} occurrences")
        issues.extend(volume_spikes[:5])
    else:
        print("[OK] Volume spikes 10x+: NO ISSUES FOUND")
    
    print("\n" + "=" * 60)
    print("[4] SPLIT & ADJUSTED DATA CONSISTENCY")
    print("=" * 60)
    
    # 4a. Large price changes
    split_candidates = []
    for ticker in sample_tickers[:15]:
        ticker_df = df[df['ticker'] == ticker].sort_values('date').copy()
        if len(ticker_df) > 5:
            ticker_df['pct_change'] = ticker_df['close'].pct_change().abs()
            big_moves = ticker_df[ticker_df['pct_change'] >= 0.30]
            for _, row in big_moves.head(2).iterrows():
                split_candidates.append({
                    'ticker': ticker, 'date': str(row['date']),
                    'issue_category': 'SPLIT_DATA', 'issue_description': f"Price change {row['pct_change']*100:.1f}%",
                    'severity': 'MEDIUM', 'trade_impact': 'May be split or legitimate move'
                })
    
    if split_candidates:
        print(f"\n[!] LARGE PRICE CHANGES (30%+): {len(split_candidates)} occurrences")
        issues.extend(split_candidates[:5])
    else:
        print("\n[OK] Large price changes: NO ISSUES FOUND")
    
    print("\n" + "=" * 60)
    print("[5] DATA FRESHNESS (TIMELINESS)")
    print("=" * 60)
    
    # Latest date per ticker
    latest_dates = df.groupby('ticker')['date'].max().reset_index()
    latest_dates.columns = ['ticker', 'latest_date']
    
    most_recent = latest_dates['latest_date'].max()
    stale_tickers = latest_dates[latest_dates['latest_date'] < most_recent - timedelta(days=5)]
    
    if len(stale_tickers) > 0:
        print(f"\n[!] STALE TICKERS (>5 days behind): {len(stale_tickers)} tickers")
        for _, row in stale_tickers.head(5).iterrows():
            issues.append({
                'ticker': row['ticker'], 'date': str(row['latest_date']),
                'issue_category': 'DATA_FRESHNESS', 'issue_description': f"Last update: {row['latest_date']}",
                'severity': 'MEDIUM', 'trade_impact': 'Missing recent price action'
            })
    else:
        print("\n[OK] Data freshness: NO ISSUES FOUND")
    
    print(f"[i] Most recent date in database: {most_recent}")
    
    # Check if data is current
    today = datetime.now().date()
    days_since_update = (today - most_recent).days
    if days_since_update > 4:
        print(f"[!] Database may be stale: {days_since_update} days since last update")
        issues.append({
            'ticker': 'ALL', 'date': str(most_recent),
            'issue_category': 'DATA_FRESHNESS', 'issue_description': f'{days_since_update} days since last update',
            'severity': 'HIGH', 'trade_impact': 'Signals based on old data'
        })
    else:
        print(f"[OK] Database is current ({days_since_update} days old)")
    
    print("\n" + "=" * 60)
    print("[6] DATABASE-LEVEL VALIDATIONS")
    print("=" * 60)
    
    # Duplicates
    dup_count = df.duplicated(subset=['ticker', 'date']).sum()
    print(f"\n{'[X]' if dup_count > 0 else '[OK]'} Duplicate records: {dup_count}")
    
    # Row count per ticker
    row_counts = df.groupby('ticker').size()
    
    # Tickers with few records
    low_record_tickers = row_counts[row_counts < 50]
    if len(low_record_tickers) > 0:
        print(f"[!] TICKERS WITH <50 RECORDS: {len(low_record_tickers)}")
        for ticker, count in low_record_tickers.head(5).items():
            issues.append({
                'ticker': ticker, 'date': 'N/A',
                'issue_category': 'DATABASE_LEVEL', 'issue_description': f'Only {count} records',
                'severity': 'MEDIUM', 'trade_impact': 'Insufficient data for indicators'
            })
    else:
        print("[OK] All tickers have sufficient records")
    
    # Inconsistent record counts
    median_records = row_counts.median()
    very_low = row_counts[row_counts < median_records * 0.5]
    if len(very_low) > 0:
        print(f"[!] INCOMPLETE DATA SUSPECTED: {len(very_low)} tickers have <50% of median records")
    else:
        print("[OK] Record counts are consistent")
    
    conn.close()
    
    # Count issues by severity
    high_issues = sum(1 for i in issues if i['severity'] == 'HIGH')
    medium_issues = sum(1 for i in issues if i['severity'] == 'MEDIUM')
    low_issues = sum(1 for i in issues if i['severity'] == 'LOW')
    
    print("\n" + "=" * 60)
    print("ISSUES SUMMARY TABLE")
    print("=" * 60)
    
    if issues:
        print(f"\nHigh: {high_issues} | Medium: {medium_issues} | Low: {low_issues}")
        print("\n" + "-" * 120)
        print(f"{'Ticker':<8} {'Date':<12} {'Category':<18} {'Description':<45} {'Severity':<8} {'Impact':<30}")
        print("-" * 120)
        for issue in issues[:30]:  # Limit output
            print(f"{issue['ticker']:<8} {issue['date']:<12} {issue['issue_category']:<18} {issue['issue_description'][:44]:<45} {issue['severity']:<8} {issue['trade_impact'][:29]:<30}")
    else:
        print("\n[OK] NO ISSUES FOUND IN DATABASE!")
    
    # Calculate reliability score
    score = 10
    score -= high_issues * 1.5
    score -= medium_issues * 0.3
    score -= low_issues * 0.05
    score = max(0, min(10, score))
    
    print("\n" + "=" * 60)
    print("FINAL ASSESSMENT")
    print("=" * 60)
    
    print(f"\nDATA RELIABILITY SCORE: {score:.1f}/10")
    
    # Top 3 critical issues
    print("\nTOP 3 MOST CRITICAL ISSUES:")
    high_priority = [i for i in issues if i['severity'] == 'HIGH'][:3]
    if high_priority:
        for i, issue in enumerate(high_priority, 1):
            print(f"  {i}. [{issue['ticker']}] {issue['issue_description']}")
    else:
        medium_priority = [i for i in issues if i['severity'] == 'MEDIUM'][:3]
        if medium_priority:
            for i, issue in enumerate(medium_priority, 1):
                print(f"  {i}. [{issue['ticker']}] {issue['issue_description']}")
        else:
            print("  None - data quality is excellent!")
    
    # Can data be used?
    print("\nCAN THIS DATA BE USED FOR BACKTESTING?")
    if high_issues == 0:
        print("  YES - No critical issues found")
    elif high_issues <= 3:
        print(f"  YES WITH CAUTION - {high_issues} critical issue(s) to investigate")
    else:
        print(f"  NO - {high_issues} critical issues must be resolved first")
    
    return issues, score

if __name__ == "__main__":
    issues, score = run_audit()
