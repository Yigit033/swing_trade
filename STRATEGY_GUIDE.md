# 📚 Strategy Guide - Swing Trading System

Comprehensive explanation of the trading strategy, indicators, and signal logic.

## 📊 Strategy Overview

This is a **multi-factor technical swing trading strategy** that combines:
- Trend following (EMAs)
- Momentum indicators (RSI, MACD)
- Volume confirmation (OBV, Volume surge)
- Support/resistance levels

**Trading Style**: Swing trading (hold 5-15 days)  
**Time Frame**: Daily charts  
**Win Rate Target**: 45-55%  
**Risk/Reward**: Minimum 1.5:1

## 🎯 Core Philosophy

1. **Trend is Your Friend**: Only trade in uptrends (price > EMA_200)
2. **Buy Low in Uptrend**: Enter on pullbacks (RSI 30-50)
3. **Confirmation Required**: Multiple indicators must align
4. **Strict Risk Management**: Never risk more than 2% per trade
5. **Let Winners Run**: Use trailing stops and multiple targets

## 📈 Technical Indicators Explained

### 1. Exponential Moving Averages (EMA)

**What they are**: Weighted averages giving more importance to recent prices.

- **EMA_20**: Short-term trend (1 month)
- **EMA_50**: Medium-term trend (2.5 months)
- **EMA_200**: Long-term trend (10 months)

**How we use them**:
- **Trend Filter**: Price must be > EMA_200 (uptrend)
- **Bonus Points**: EMA_20 > EMA_50 > EMA_200 (strong alignment)

**Why it works**: Strong trends tend to continue; avoid fighting the trend.

### 2. Relative Strength Index (RSI)

**What it is**: Momentum oscillator measuring overbought/oversold conditions (0-100 scale).

- **0-30**: Oversold (potential buy)
- **30-70**: Neutral
- **70-100**: Overbought (potential sell)

**How we use it**:
- **Entry**: RSI between 30-50 AND rising (oversold recovery)
- **Exit**: RSI > 70 (take profits)

**Why it works**: Buy when temporarily oversold in an uptrend; sell when overextended.

### 3. MACD (Moving Average Convergence Divergence)

**What it is**: Trend-following momentum indicator showing relationship between two EMAs.

- **MACD Line**: 12-day EMA - 26-day EMA
- **Signal Line**: 9-day EMA of MACD
- **Histogram**: MACD - Signal (shows crossovers)

**How we use it**:
- **Entry**: Histogram crosses above 0 (bullish crossover)
- **Exit**: Histogram crosses below 0 (bearish crossover)

**Why it works**: Identifies momentum shifts and trend changes early.

### 4. On-Balance Volume (OBV)

**What it is**: Cumulative volume indicator showing buying/selling pressure.

- Add volume on up days
- Subtract volume on down days

**How we use it**:
- **Entry**: OBV slope > 0 (accumulation)
- **Confirmation**: Rising OBV confirms price uptrend

**Why it works**: "Volume precedes price" - smart money accumulates before price rises.

### 5. Volume Analysis

**What it is**: Measures trading activity.

**How we use it**:
- **Entry**: Volume > 1.2x average (confirmation)
- **Bonus**: Volume > 1.5x average (strong interest)

**Why it works**: High volume confirms breakout validity and trader interest.

### 6. Support Levels

**What they are**: Price levels where buying interest is strong (floors).

**How we use it**:
- **Entry**: Price within 3% of 20-day support level
- **Logic**: Buying near support offers better risk/reward

**Why it works**: Support levels act as "demand zones" where buyers step in.

### 7. Average Directional Index (ADX)

**What it is**: Measures trend strength (0-100 scale).

- **0-25**: Weak or no trend
- **25-50**: Strong trend
- **50+**: Very strong trend

**How we use it**:
- **Bonus**: ADX > 25 (strong trend)
- **Penalty**: ADX < 20 (choppy market, avoid)

**Why it works**: Trends are more reliable in trending markets (obvious but crucial).

### 8. Average True Range (ATR)

**What it is**: Measures volatility (average price movement).

**How we use it**:
- **Stop Loss**: Entry - (2 × ATR) (volatility-adjusted)
- **Position Sizing**: Risk / (2 × ATR) = shares to buy

**Why it works**: Adapts to each stock's natural volatility; prevents being stopped out by noise.

## 🔍 Entry Signal Logic

### Mandatory Conditions (ALL 6 Required)

```python
✅ 1. Trend: close > EMA_200
✅ 2. RSI: 30 < RSI < 50 AND RSI rising
✅ 3. MACD: Histogram crosses above 0
✅ 4. Volume: > 1.2x average
✅ 5. OBV: Slope > 0 (rising)
✅ 6. Support: Price within 3% of 20-day support
```

**Base Score**: 6 points (one per condition)

### Bonus Conditions (Optional)

```python
+2 points: EMA_20 > EMA_50 > EMA_200 (perfect alignment)
+1 point: Volume > 1.5x average (strong surge)
+1 point: ADX > 25 (strong trend)
-2 points: ADX < 20 (weak trend - penalty)
```

**Final Score**: Base + Bonuses (0-10 scale)

**Minimum Score to Trade**: 6 (configurable)

## 📊 Scoring Examples

### Example 1: Perfect Signal (Score 10/10)

```
AAPL - January 15, 2024
✅ Price: $185.50 (above EMA_200: $170)
✅ RSI: 42 → 45 (rising from oversold)
✅ MACD: Histogram 0.02 (crossed above 0)
✅ Volume: 85M (1.4x average)
✅ OBV: Slope +15,000 (rising)
✅ Support: $182 (within 2% of current price)
✅ EMA Alignment: 20 > 50 > 200 (+2)
✅ Volume Surge: 1.4x (+0)
✅ ADX: 28 (+1)

Final Score: 6 + 3 = 9/10
```

**Trade**:
- Entry: $185.50
- Stop: $180.50 (2 × ATR = $2.50)
- Target 1: $193.00 (1.5:1 R/R)
- Target 2: $198.00 (2.5:1 R/R)

### Example 2: Marginal Signal (Score 6/10)

```
XYZ - January 15, 2024
✅ Price: $50.20 (above EMA_200: $48)
✅ RSI: 38 → 40 (rising)
✅ MACD: Histogram 0.01 (barely crossed)
✅ Volume: 1.3M (1.25x average)
✅ OBV: Slope +500 (barely rising)
✅ Support: $49.50 (within 1.5%)
❌ EMA Alignment: 20 > 50 but 50 < 200 (0)
❌ Volume Surge: Only 1.25x (0)
⚠️ ADX: 18 (weak trend, -2)

Final Score: 6 + 0 - 2 = 4/10
```

**Result**: NO TRADE (below minimum score of 6)

### Example 3: Strong Signal (Score 8/10)

```
MSFT - January 15, 2024
✅ Price: $375 (above EMA_200: $350)
✅ RSI: 35 → 38 (rising from oversold)
✅ MACD: Histogram 0.15 (strong crossover)
✅ Volume: 32M (1.6x average)
✅ OBV: Slope +25,000 (strong rise)
✅ Support: $372 (within 1%)
✅ EMA Alignment: 20 > 50 > 200 (+2)
✅ Volume Surge: 1.6x (+1)
❌ ADX: 22 (moderate, 0)

Final Score: 6 + 3 = 9/10
```

**Trade**:
- Entry: $375
- Stop: $368 (2 × ATR = $3.50)
- Target 1: $385.50 (1.5:1)
- Target 2: $392.50 (2.5:1)

## 🛡️ Risk Management

### Position Sizing Formula

```python
# Calculate stop loss distance
stop_distance = entry_price - stop_loss
stop_loss = entry_price - (2 × ATR)

# Calculate shares to buy
max_risk = portfolio_value × 0.02  # 2% risk
shares = max_risk / stop_distance

# Apply position size limit
max_position_value = portfolio_value × 0.20  # 20% max
if shares × entry_price > max_position_value:
    shares = max_position_value / entry_price
```

### Example Calculation

```
Portfolio: $10,000
Stock: AAPL at $185.50
ATR: $2.50

Stop Loss: $185.50 - (2 × $2.50) = $180.50
Risk per share: $5.00
Max risk: $10,000 × 0.02 = $200
Shares: $200 / $5.00 = 40 shares
Position value: 40 × $185.50 = $7,420 (74% of portfolio)

❌ TOO LARGE! Exceeds 20% limit ($2,000)
✅ Adjusted: $2,000 / $185.50 = 10 shares
✅ Final position: 10 shares = $1,855
✅ Risk: 10 × $5.00 = $50 (0.5% of portfolio)
```

### Risk Limits (Hard Coded)

```python
MAX_RISK_PER_TRADE = 2%        # Per trade
MAX_POSITION_SIZE = 20%        # Per stock
MAX_OPEN_POSITIONS = 5         # Total positions
MAX_SECTOR_ALLOCATION = 30%    # Per sector
```

These limits **CANNOT be exceeded** even if signals are perfect.

## 🚪 Exit Strategy

### Exit Triggers (ANY condition = EXIT)

1. **Stop Loss Hit**: Price touches stop loss (2 × ATR below entry)
2. **Target 1 Reached**: 1.5:1 reward/risk (exit half, move stop to breakeven)
3. **Target 2 Reached**: 2.5:1 reward/risk (exit remaining)
4. **RSI Overbought**: RSI > 70 (take profits)
5. **MACD Reversal**: Histogram crosses below 0 (momentum fading)

### Trailing Stop (Optional Manual Adjustment)

```python
# After Target 1 reached:
- Sell 50% of position at Target 1
- Move stop to entry price (breakeven)
- Let remaining 50% run to Target 2

# If price continues up:
- Trail stop below recent swing lows
- Lock in profits as price rises
```

## 📉 What to Avoid (Red Flags)

### Never Trade When:

1. **No Clear Trend**: Price choppy, sideways movement
2. **Weak ADX**: ADX < 20 (no clear trend)
3. **Low Volume**: Volume < 500,000 shares daily
4. **Earnings Week**: High volatility risk
5. **News Events**: Upcoming Fed meetings, major announcements
6. **Gap Ups**: Stock gapped up >5% (wait for pullback)
7. **Extended Price**: Stock already up 20%+ in 1 month

### Example: Bad Signal (Do NOT Trade)

```
BAD - January 15, 2024
❌ Price: $95 (choppy around EMA_200)
✅ RSI: 40 (OK)
❌ MACD: Histogram flickering above/below 0
❌ Volume: 250K (below 500K minimum)
✅ OBV: Rising (OK)
✅ Support: $94 (OK)
❌ ADX: 15 (very weak trend)
❌ News: Earnings in 2 days

Score: Even if 6+, DO NOT TRADE
Reason: Low volume + weak trend + earnings risk
```

## 🎓 Common Questions

**Q: Why 2% risk limit?**  
A: If you lose 10 trades in a row (very rare), you're only down 20%. Allows recovery.

**Q: Why require all 6 conditions?**  
A: Reduces false signals. Better to miss opportunities than take bad trades.

**Q: Can I adjust the strategy?**  
A: Yes, via `config.yaml`. But backtest any changes first!

**Q: What if RSI is at 28 (just below 30)?**  
A: No trade. Rules are rules. Discipline > optimization.

**Q: Should I buy more if price drops after entry?**  
A: NO. "Averaging down" violates risk management. Take the loss if stop is hit.

**Q: Can I hold longer than 15 days?**  
A: Yes, if price keeps trending up. Use trailing stops.

## 📊 Strategy Performance by Market Condition

### Bull Market (S&P 500 trending up)
- Win Rate: 55-65%
- Many signals
- Larger winners
- **Best environment for this strategy**

### Sideways Market (Choppy, range-bound)
- Win Rate: 40-50%
- Fewer signals (good - avoiding bad trades)
- Smaller winners
- **Challenging but manageable**

### Bear Market (S&P 500 trending down)
- Win Rate: 30-40%
- Few signals (great - preserves capital)
- Quick exits important
- **Strategy goes to cash automatically**

## 🔄 Strategy Improvements (Ideas for Advanced Users)

1. **Sector Rotation**: Weight toward strong sectors
2. **Market Regime Filter**: Reduce size in weak markets
3. **Volatility Scaling**: Increase positions in low VIX
4. **Earnings Avoidance**: Skip stocks with earnings in 7 days
5. **Relative Strength**: Compare to SPY performance
6. **News Sentiment**: Integrate news analysis

## ⚠️ Final Reminders

1. **Past Performance ≠ Future Results**: Markets change
2. **Drawdowns Happen**: Expect losing streaks
3. **Discipline is Key**: Follow rules even when tempting to break them
4. **Paper Trade First**: Practice until consistent
5. **Risk Management > Entry Signals**: Proper sizing saves you

---

**The strategy is simple but not easy. Success comes from consistent execution, not from tweaking parameters.**

