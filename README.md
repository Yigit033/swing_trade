# 📈 Professional Swing Trading System

A comprehensive, production-ready Python-based stock swing trading system with technical analysis, risk management, backtesting, and real-time monitoring capabilities.

## 🎯 Features

### Core Functionality
- ✅ **Automated Stock Scanning**: Scan 200+ stocks in under 15 seconds
- ✅ **Technical Analysis**: 15+ indicators (RSI, MACD, EMA, ADX, OBV, VWAP, etc.)
- ✅ **Smart Signal Generation**: Multi-factor scoring system (0-150 scale)
- ✅ **Risk Management**: Position sizing, stop-loss, take-profit automation
- ✅ **Backtesting Engine**: Test strategies on historical data
- ✅ **Interactive Dashboard**: Beautiful Streamlit web interface
- ✅ **Alert System**: Email and Telegram notifications
- ✅ **Performance Metrics**: Sharpe ratio, drawdown, win rate, profit factor

### 🚀 SmallCap Momentum System (Senior Trader v2.0)
- ✅ **4-Type Classification**: Type S (Squeeze), C (Early), B (Momentum), A (Continuation)
- ✅ **Float Tiering**: Atomic (≤15M), Micro (15-30M), Small (30-45M), Tight (45-60M)
- ✅ **RSI Bullish Divergence**: Game-changing early reversal detection
- ✅ **MACD & VWAP Analysis**: Technical confluence confirmation
- ✅ **Short Squeeze Detection**: SI ≥20%, Days-to-Cover ≥5
- ✅ **Finviz Integration**: Real-time momentum universe from Finviz screener
- ✅ **Pullback Entry**: Allows -5% to +15% 5-day return for early entries

### Safety Features
- 🛡️ **Maximum 2% risk per trade** (0.5% for SmallCaps)
- 🛡️ **Maximum 20% portfolio allocation per stock**
- 🛡️ **Maximum 5 open positions**
- 🛡️ **Maximum 30% sector allocation**
- 🛡️ **ATR-based stop losses** (10-period ATR for SmallCaps)
- 🛡️ **Earnings exclusion**: ±7 days for SmallCaps

## 📋 Requirements

- Python 3.8+
- Windows/Linux/Mac
- Internet connection (for data download)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd swing_trade

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup

```bash
# Run full setup
python setup.py --full-setup
```

This will:
- Create necessary directories
- Initialize SQLite database
- Create `.env` file template
- Verify configuration

### 3. Configure (Optional)

Edit `config.yaml` to customize:
- Risk parameters
- Indicator settings
- Alert preferences
- Backtest parameters

Edit `.env` for alerts (optional):
```
ALPHA_VANTAGE_API_KEY=your_key_here
EMAIL_PASSWORD=your_email_app_password
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 4. Download Data

```bash
# Download S&P 500 historical data (last 250 days)
python main.py --download-data --days=250
```

This takes 10-20 minutes for ~500 stocks.

### 5. Run Daily Scan

```bash
# Scan for trading signals
python main.py --daily-scan --portfolio-value=10000
```

Output:
```
SCAN RESULTS - 2024-01-15
==================================================
Total signals found: 23
High quality signals: 8

TOP 10 SIGNALS:
--------------------------------------------------
Ticker   Score  Entry      Stop       Target     R:R
--------------------------------------------------
AAPL     8      $185.50    $179.80    $195.20    1.7
MSFT     8      $375.20    $368.40    $389.60    2.1
...
```

### 6. Launch Dashboard

```bash
streamlit run swing_trader/dashboard/app.py
```

Opens web dashboard at `http://localhost:8501`

## 📊 Dashboard Features

### 🔍 Scan Stocks Page
- Run live stock scans
- Filter by minimum score
- View top N results
- Interactive charts with indicators
- Candlestick + EMA + Support/Resistance
- RSI, MACD, Volume panels

### 📉 Backtest Page
- Test strategy on historical data
- Customize date range and capital
- View performance metrics
- Equity curve visualization
- Detailed trade log

### ⚙️ Settings Page
- Download S&P 500 data
- Update existing data
- View database status
- System information

## 🎓 Trading Strategy

### Entry Conditions (ALL must be TRUE for signal)

1. **Trend**: Price > EMA_200 (uptrend)
2. **RSI**: 30 < RSI < 50 AND rising (oversold recovery)
3. **MACD**: Histogram crosses above 0 (bullish momentum)
4. **Volume**: Volume > 1.2x average (confirmation)
5. **OBV**: Positive slope (buying pressure)
6. **Support**: Price within 3% of 20-day support

### Bonus Scoring (+1 to +2 each)

- Strong trend alignment (EMA_20 > EMA_50 > EMA_200): +2
- Very high volume (>1.5x average): +1
- Strong ADX (>25): +1
- **Penalty**: Weak ADX (<20): -2

### Exit Conditions (ANY triggers exit)

1. **Stop Loss**: Price hits ATR-based stop (2x ATR below entry)
2. **Take Profit 1**: 1.5:1 reward/risk ratio
3. **Take Profit 2**: 2.5:1 reward/risk ratio
4. **RSI Overbought**: RSI > 70
5. **MACD Cross Down**: Histogram crosses below 0

## 📈 Performance Expectations

### Realistic Targets (based on backtesting)

- **Win Rate**: 45-55%
- **Profit Factor**: 1.5-2.5
- **Annual Return**: 15-35% (varies by market conditions)
- **Maximum Drawdown**: 10-20%
- **Average Hold Time**: 5-15 days
- **Sharpe Ratio**: 0.8-1.5

### What This System DOES

✅ Systematically identifies opportunities  
✅ Removes emotional decision-making  
✅ Enforces strict risk management  
✅ Provides real-time monitoring  
✅ Maintains trading discipline

### What This System DOESN'T DO

❌ Guarantee profits (no system does)  
❌ Win every trade (losses are normal)  
❌ Prevent market crashes  
❌ Replace due diligence  
❌ Eliminate all risks

## ⚠️ Important Warnings

### Before Using Real Money

1. **Paper Trade First**: Simulate for 3+ months
2. **Complete 50+ Trades**: Gain experience
3. **Understand the Strategy**: Know why signals are generated
4. **Risk Management**: Never risk more than you can afford to lose
5. **Market Knowledge**: Understand basic market mechanics

### Risk Disclaimer

⚠️ **Trading stocks involves substantial risk of loss. Past performance does not guarantee future results. This software is for educational purposes. Always consult a financial advisor before making investment decisions.**

## 📖 Usage Guide

### Daily Workflow

```bash
# 1. Update data (after market close)
python main.py --daily-scan

# 2. Review signals in dashboard
streamlit run swing_trader/dashboard/app.py

# 3. Manual verification
# - Check news for selected stocks
# - Review charts and indicators
# - Verify risk/reward ratios

# 4. Execute trades (on your brokerage platform)
# - Enter positions for approved signals
# - Set stop-loss and take-profit orders
# - Track in spreadsheet or journal

# 5. Monitor positions
# - Check dashboard daily
# - Update stop-loss if needed (trailing stops)
# - Exit when conditions are met
```

### Scheduled Automation (Optional)

**Windows Task Scheduler** or **Linux cron**:

```bash
# Run daily at 5:30 PM (after market close)
30 17 * * 1-5 cd /path/to/swing_trade && python main.py --daily-scan
```

## 🧪 Running Backtests

### Command Line

```python
from swing_trader.backtesting.engine import BacktestEngine
from swing_trader.backtesting.metrics import PerformanceMetrics
import yaml

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Load data (dictionary of ticker -> DataFrame)
# ... load your data ...

# Run backtest
engine = BacktestEngine(config)
results = engine.run_backtest('2022-01-01', '2024-12-31', data_dict)

# Calculate metrics
metrics = PerformanceMetrics.calculate_metrics(results)
PerformanceMetrics.print_metrics(metrics)
```

### Via Dashboard

1. Open dashboard: `streamlit run swing_trader/dashboard/app.py`
2. Navigate to "Backtest" page
3. Select date range and capital
4. Click "Run Backtest"
5. View results and equity curve

## 🔧 Configuration

### Key Parameters in `config.yaml`

```yaml
risk:
  max_risk_per_trade: 0.02       # 2% risk per trade
  max_position_size: 0.20        # 20% max per stock
  max_open_positions: 5          # Maximum positions
  stop_loss_atr_multiplier: 2.0  # Stop loss distance

strategy:
  min_signal_score: 6            # Minimum score to generate signal
  rsi_entry_min: 30              # RSI lower bound
  rsi_entry_max: 50              # RSI upper bound
  volume_surge_threshold: 1.2    # Volume confirmation

filters:
  min_volume: 500000             # Daily volume filter
  min_price: 10.0                # Minimum stock price
  max_price: 500.0               # Maximum stock price
  min_market_cap: 300000000      # $300M market cap
```

## 📁 Project Structure

```
swing_trade/
├── swing_trader/
│   ├── data/                  # Data fetching and storage
│   │   ├── fetcher.py         # API data fetching
│   │   ├── storage.py         # SQLite database
│   │   └── updater.py         # Daily updates
│   ├── indicators/            # Technical indicators
│   │   ├── trend.py           # EMA, ADX
│   │   ├── momentum.py        # RSI, MACD
│   │   └── volume.py          # OBV, Volume analysis
│   ├── strategy/              # Trading strategy
│   │   ├── signals.py         # Signal generation
│   │   ├── scoring.py         # Signal scoring
│   │   └── risk_manager.py    # Risk management
│   ├── backtesting/           # Backtesting engine
│   │   ├── engine.py          # Backtest execution
│   │   └── metrics.py         # Performance metrics
│   └── dashboard/             # Web interface
│       ├── app.py             # Streamlit dashboard
│       └── alerts.py          # Email/Telegram alerts
├── data/                      # SQLite database
├── logs/                      # Log files
├── output/                    # Output files
├── main.py                    # Main script
├── setup.py                   # Setup script
├── config.yaml                # Configuration
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🐛 Troubleshooting

### Common Issues

**1. Database not found**
```bash
python setup.py --init-db
```

**2. Missing dependencies**
```bash
pip install -r requirements.txt
```

**3. No data in database**
```bash
python main.py --download-data --days=250
```

**4. TA-Lib installation error (Windows)**
```bash
# Download TA-Lib wheel from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.XX-cpXX-cpXX-win_amd64.whl
```

**5. Streamlit port already in use**
```bash
streamlit run swing_trader/dashboard/app.py --server.port 8502
```

## 📚 Additional Resources

### Learning Materials
- **Technical Analysis**: Study RSI, MACD, moving averages
- **Risk Management**: Position sizing, stop-losses
- **Trading Psychology**: Discipline, emotion control
- **Backtesting**: Statistical significance, overfitting

### Recommended Reading
- "Technical Analysis of Financial Markets" - John Murphy
- "Trading for a Living" - Dr. Alexander Elder
- "Market Wizards" - Jack Schwager
- "The New Trading for a Living" - Dr. Alexander Elder

## 🤝 Contributing

This is an educational project. Feel free to:
- Improve algorithms
- Add new indicators
- Enhance visualizations
- Fix bugs
- Add tests

## 📄 License

This project is for educational purposes only. Use at your own risk.

## 📞 Support

For issues or questions:
1. Check documentation and troubleshooting section
2. Review code comments and docstrings
3. Test with paper trading first
4. Consult financial advisors for investment decisions

## 🔄 Version History

### v1.0.0 (Initial Release)
- Complete trading system implementation
- Multi-indicator signal generation
- Risk management system
- Backtesting engine
- Streamlit dashboard
- Alert system (Email/Telegram)
- Comprehensive documentation

---

**Remember**: This tool assists in analysis but doesn't replace human judgment. Always do your own research, manage risk carefully, and never invest more than you can afford to lose.

**Happy Trading! 📈**

