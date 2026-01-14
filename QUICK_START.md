# ⚡ Quick Start Guide - 5 Minutes to First Scan

Get up and running with your first stock scan in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] Internet connection
- [ ] 2GB free disk space
- [ ] 5-10 minutes of time

## 🚀 Installation (2 minutes)

### Step 1: Open Terminal/Command Prompt

**Windows**: Press `Win + R`, type `cmd`, press Enter  
**Mac**: Press `Cmd + Space`, type `terminal`, press Enter  
**Linux**: Press `Ctrl + Alt + T`

### Step 2: Navigate to Project

```bash
cd C:\swing_trade  # Windows
cd ~/swing_trade   # Mac/Linux
```

### Step 3: Install & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run setup
python setup.py --full-setup
```

**⏱️ Takes 1-2 minutes**

Expected output:
```
✅ SETUP COMPLETE!
```

## 📥 Download Data (10-15 minutes)

Download historical stock data:

```bash
python main.py --download-data --days=250
```

**⏱️ Takes 10-15 minutes for ~500 stocks**

While waiting, you can:
- ☕ Get coffee
- 📖 Read STRATEGY_GUIDE.md
- ⚙️ Review config.yaml settings

## 🔍 Your First Scan (30 seconds)

```bash
python main.py --daily-scan --portfolio-value=10000
```

You should see:
```
====================================================
SCAN RESULTS - 2024-01-15
====================================================
Total signals found: 23
High quality signals: 8

TOP 10 SIGNALS:
--------------------------------------------------
Ticker   Score  Entry      Stop       Target     R:R
--------------------------------------------------
AAPL     8      $185.50    $179.80    $195.20    1.7
MSFT     8      $375.20    $368.40    $389.60    2.1
GOOGL    7      $142.30    $138.90    $147.40    1.5
...
```

**🎉 Congratulations! You just ran your first scan!**

## 🖥️ Launch Dashboard (Instant)

```bash
streamlit run swing_trader/dashboard/app.py
```

Browser opens automatically to: `http://localhost:8501`

### Dashboard Features:
- 📊 **Scan Stocks**: Run interactive scans
- 📉 **Backtest**: Test strategy on history
- ⚙️ **Settings**: Manage data and configuration

## 📚 What's Next?

### Beginner Path (Recommended)

1. **📖 Learn the Strategy** (15 min)
   ```bash
   # Read strategy guide
   cat STRATEGY_GUIDE.md  # Linux/Mac
   type STRATEGY_GUIDE.md  # Windows
   ```

2. **🧪 Run Backtest** (5 min)
   - Open dashboard
   - Click "Backtest" page
   - Set dates: 2022-01-01 to 2024-12-31
   - Initial capital: $10,000
   - Click "Run Backtest"
   - Review results

3. **📝 Paper Trade** (3+ months)
   - Run daily scans
   - Track signals in spreadsheet
   - Simulate trades (don't use real money yet!)
   - Goal: 50+ simulated trades
   - Learn from mistakes

4. **📊 Review Performance** (Weekly)
   - Track win rate
   - Analyze losing trades
   - Refine your execution
   - Build confidence

5. **💰 Go Live** (After 3+ months paper trading)
   - Start small (10-20% of intended capital)
   - Follow rules strictly
   - Increase size gradually
   - Journal every trade

### Advanced Path

1. **🔧 Customize Strategy**
   - Edit `config.yaml`
   - Adjust risk parameters
   - Change indicator settings
   - Backtest changes

2. **🤖 Automate Scans**
   - Set up daily automated scans
   - Configure email/Telegram alerts
   - Schedule updates

3. **📈 Portfolio Tracking**
   - Track open positions
   - Monitor exits
   - Calculate real performance

## 🎯 Daily Workflow (5 minutes/day)

### After Market Close (5:30 PM ET)

```bash
# 1. Run daily scan (2 min)
python main.py --daily-scan

# 2. Review signals in dashboard (2 min)
streamlit run swing_trader/dashboard/app.py

# 3. Research top signals (1 min)
# - Check news for stocks
# - Verify charts
# - Decide which to trade
```

### Next Morning (Before Market Open)

```
1. Place orders for approved signals
2. Set stop-loss orders
3. Set take-profit orders (or alerts)
```

### During Market Day

```
- Monitor positions (check 2-3 times/day)
- Don't overtrade
- Don't panic on small moves
- Trust your stops
```

## 💡 Pro Tips for Beginners

### Do's ✅

- ✅ Start with paper trading
- ✅ Follow rules consistently
- ✅ Track every trade in journal
- ✅ Accept losses as part of trading
- ✅ Review weekly performance
- ✅ Keep position sizes small initially
- ✅ Use stop losses always
- ✅ Learn from mistakes

### Don'ts ❌

- ❌ Trade real money immediately
- ❌ Risk more than 2% per trade
- ❌ Chase signals (wait for setup)
- ❌ Average down on losers
- ❌ Ignore stop losses
- ❌ Trade based on emotions
- ❌ Overtrade (quality > quantity)
- ❌ Expect to win every trade

## 🆘 Quick Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt
```

### "Database not found"
```bash
python setup.py --init-db
```

### "No signals found"
```bash
# Download more data
python main.py --download-data --days=250

# Or lower minimum score in config.yaml
min_signal_score: 5  # instead of 7
```

### "Port already in use"
```bash
streamlit run swing_trader/dashboard/app.py --server.port 8502
```

### Dashboard not opening automatically
```
Open browser manually: http://localhost:8501
```

## 📊 Understanding Your First Results

### Signal Score Meaning

- **8-10**: Excellent setup (rare, ~5% of signals)
- **7**: Good setup (common, ~15% of signals)
- **6**: Acceptable setup (common, ~25% of signals)
- **<6**: Skip (many, ~55% of scans)

### Entry/Stop/Target Example

```
Ticker: AAPL
Entry: $185.50      ← Buy here
Stop: $179.80       ← Exit if price drops here (loss)
Target: $195.20     ← Exit if price rises here (profit)
R:R: 1.7            ← Reward/Risk ratio

Risk: $185.50 - $179.80 = $5.70 per share
Reward: $195.20 - $185.50 = $9.70 per share
Ratio: $9.70 / $5.70 = 1.7:1
```

### How Many Signals is Normal?

- **Bull Market**: 20-40 signals/day
- **Neutral Market**: 10-20 signals/day
- **Bear Market**: 0-10 signals/day

**Quality > Quantity**: It's better to have 3 great signals than 30 mediocre ones.

## 🎓 Learning Resources

### Included Documentation

1. **README.md**: Complete system overview
2. **STRATEGY_GUIDE.md**: Strategy deep dive
3. **INSTALLATION_GUIDE.md**: Detailed installation
4. **This file**: Quick start

### Recommended Reading

- "Technical Analysis of Financial Markets" - John Murphy
- "Trading for a Living" - Dr. Alexander Elder
- Free resource: Investopedia.com (learn indicators)

### Practice Tools

- TradingView.com (free charts)
- Yahoo Finance (stock data)
- Stock Simulators (paper trading)

## 📅 30-Day Beginner Plan

### Week 1: Learn & Setup
- ✅ Install system
- ✅ Read documentation
- ✅ Understand indicators
- ✅ Run backtests

### Week 2: Paper Trade
- ✅ Daily scans
- ✅ Pick 2-3 signals
- ✅ Track in spreadsheet
- ✅ Analyze results

### Week 3: Refine
- ✅ Review winning/losing trades
- ✅ Identify patterns
- ✅ Practice discipline
- ✅ Build confidence

### Week 4: Consistency
- ✅ Continue paper trading
- ✅ Track performance metrics
- ✅ Prepare for live trading
- ✅ Set up brokerage account

## 🎯 Success Metrics (Paper Trading)

Track these in a spreadsheet:

```
Date | Ticker | Entry | Exit | P&L | P&L% | Reason
-----|--------|-------|------|-----|------|-------
1/15 | AAPL   | 185.5 | 195.2| +9.7| +5.2%| Target 1
1/16 | MSFT   | 375.2 | 368.4| -6.8| -1.8%| Stop loss
...
```

**Goals after 50 trades**:
- Win rate: > 45%
- Avg win: > Avg loss × 1.5
- Max drawdown: < 15%
- Profit factor: > 1.5

## 🚀 You're Ready!

You now have:
- ✅ Working system
- ✅ Historical data
- ✅ Daily scan capability
- ✅ Dashboard access
- ✅ Strategy knowledge

**Next Step**: Run your first paper trade!

```bash
# Daily routine starts now:
python main.py --daily-scan
```

**Remember**: 
- 📚 Learn constantly
- 📝 Journal every trade
- 🎯 Focus on process, not profits
- ⏳ Be patient (success takes time)
- 🧘 Stay disciplined

---

**Questions?** Review README.md and STRATEGY_GUIDE.md

**Good luck! 📈**

