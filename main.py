"""
Main execution script for swing trading system.
"""

import argparse
import logging
import yaml
import sys
from pathlib import Path
from datetime import datetime

from swing_trader.data.fetcher import DataFetcher
from swing_trader.data.storage import DatabaseManager
from swing_trader.data.updater import DataUpdater
from swing_trader.strategy.signals import SignalGenerator
from swing_trader.strategy.scoring import SignalScorer
from swing_trader.strategy.risk_manager import RiskManager
from swing_trader.dashboard.alerts import AlertSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)


def download_data(config: dict, days: int = 250):
    """
    Download initial historical data.
    
    Args:
        config: Configuration dictionary
        days: Number of days of historical data
    """
    logger.info("="*60)
    logger.info("DOWNLOADING INITIAL DATA")
    logger.info("="*60)
    
    try:
        fetcher = DataFetcher(config['data']['source'])
        updater = DataUpdater(config)
        
        # Get S&P 500 tickers
        logger.info("Fetching S&P 500 ticker list...")
        tickers = fetcher.get_sp500_tickers()
        logger.info(f"Found {len(tickers)} tickers")
        
        # Download data
        results = updater.initial_download(tickers, days=days)
        
        successful = sum(1 for v in results.values() if v)
        logger.info(f"Successfully downloaded: {successful}/{len(tickers)} stocks")
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}", exc_info=True)


def daily_scan(config: dict, portfolio_value: float = 10000):
    """
    Perform daily stock scan and generate signals.
    
    Args:
        config: Configuration dictionary
        portfolio_value: Current portfolio value for position sizing
    """
    logger.info("="*60)
    logger.info("DAILY STOCK SCAN")
    logger.info("="*60)
    
    try:
        # Initialize components
        db = DatabaseManager(config['data']['database_path'])
        updater = DataUpdater(config)
        signal_generator = SignalGenerator(config)
        scorer = SignalScorer(config)
        risk_manager = RiskManager(config)
        alerts = AlertSystem(config)
        
        # Update data
        logger.info("Updating stock data...")
        update_results = updater.daily_update()
        successful_updates = sum(1 for v in update_results.values() if v)
        logger.info(f"Updated {successful_updates} stocks")
        
        # Get filtered tickers
        logger.info("Filtering stocks...")
        tickers = updater.get_filtered_tickers()
        logger.info(f"Scanning {len(tickers)} stocks")
        
        # Get data for all tickers
        data_dict = {}
        for ticker in tickers:
            df = db.get_stock_data(ticker, limit=250)
            if df is not None and len(df) >= 50:
                data_dict[ticker] = df
        
        logger.info(f"Loaded data for {len(data_dict)} stocks")
        
        # Generate signals
        logger.info("Generating signals...")
        signals = signal_generator.scan_stocks(tickers, data_dict)
        logger.info(f"Found {len(signals)} potential signals")
        
        if not signals:
            logger.info("No signals found today")
            return
        
        # Add risk management
        logger.info("Calculating position sizing...")
        for signal in signals:
            risk_manager.add_risk_management_to_signal(signal, portfolio_value)
        
        # Rank signals
        signals = scorer.rank_signals(signals)
        
        # Filter by minimum score
        min_score = config['alerts'].get('min_signal_score', 7)
        high_quality_signals = [s for s in signals if s['score'] >= min_score]
        
        logger.info(f"High quality signals (score >= {min_score}): {len(high_quality_signals)}")
        
        # Save signals to database
        today = datetime.now().strftime('%Y-%m-%d')
        for signal in high_quality_signals:
            signal['date'] = today
            db.insert_trading_signal(signal)
        
        # Display results
        print("\n" + "="*60)
        print(f"SCAN RESULTS - {today}")
        print("="*60)
        print(f"Total signals found: {len(signals)}")
        print(f"High quality signals: {len(high_quality_signals)}")
        
        if high_quality_signals:
            print("\nTOP 10 SIGNALS:")
            print("-"*72)
            print(f"{'Ticker':<8} {'Score':<6} {'Entry':<10} {'Stop':<10} {'Target':<10} {'R:R':<6} {'Hold':<8}")
            print("-"*72)
            
            for signal in high_quality_signals[:10]:
                ticker = signal['ticker']
                score = signal['score']
                entry = signal['entry_price']
                stop = signal['stop_loss']
                target = signal.get('target_1', 0)
                rr = (target - entry) / (entry - stop) if (entry - stop) > 0 else 0
                hold_min = signal.get('expected_hold_min', 5)
                hold_max = signal.get('expected_hold_max', 15)
                hold_display = f"{hold_min}-{hold_max}d"
                
                print(f"{ticker:<8} {score:<6} ${entry:<9.2f} ${stop:<9.2f} ${target:<9.2f} {rr:<6.1f} {hold_display:<8}")
            
            print("-"*72)
        
        # Send alerts
        if high_quality_signals:
            logger.info("Sending alerts...")
            alerts.send_daily_summary(high_quality_signals, today)
            
            # Send individual alerts for top signals
            for signal in high_quality_signals[:3]:
                alerts.send_signal_alert(signal)
        
        logger.info("Daily scan complete!")
        
    except Exception as e:
        logger.error(f"Error during daily scan: {e}", exc_info=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Swing Trading System')
    
    parser.add_argument(
        '--download-data',
        action='store_true',
        help='Download initial historical data'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=250,
        help='Number of days of historical data (default: 250)'
    )
    
    parser.add_argument(
        '--daily-scan',
        action='store_true',
        help='Perform daily stock scan'
    )
    
    parser.add_argument(
        '--portfolio-value',
        type=float,
        default=10000,
        help='Portfolio value for position sizing (default: 10000)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Execute requested action
    if args.download_data:
        download_data(config, args.days)
    elif args.daily_scan:
        daily_scan(config, args.portfolio_value)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

