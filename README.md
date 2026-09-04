# 📈 Swing Trading System — AI-Powered

A production-grade Python-based swing trade system developed for personal use. It features technical analysis, risk management, paper trading, backtesting, and a **multi-layered artificial intelligence** architecture (XGBoost + Agentic AI).

---

## 🧠 AI Architecture — Two-Tiered

This system combines two different AI approaches into a **hybrid architecture**:

| Layer | Technology | Function |
|--------|-----------|----------|
| **Layer 1** | XGBoost (Classical ML) | Predicts win probability from historical closed trades (e.g., **72% win probability**) |
| **Layer 2** | LLM / Agentic AI | Translates numbers into human-readable insights — generates weekly reports, signal briefings, and acts as an autonomous Strategy Advisor with live market access |

> **Hybrid Principle:** All numerical calculations (R/R, ATR%, profit factor, risk%) are executed in Python. The LLM handles interpretation, reporting, and live data retrieval — it never makes direct numerical trading decisions on its own.

---

## 🎯 Features

### 🔬 Classical Technical Analysis
- **Automated Stock Scanner**: Scans 200+ stocks in under 15 seconds
- **Indicators Used**: ATR (10/14), RVOL + volume spikes, SMA 20/50, RSI (14) + bullish divergence, MACD (12/26/9), OBV trend, higher-low structure, MA20 slope. **Not Used**: ADX, Bollinger, VWAP, stochastic — none are in the codebase (omitted as they don't provide a measurable edge).
- **Multi-Factor Scoring**: 0–100+ scale
- **Risk Management**: Automated position sizing, stop-loss, and take-profit
- **Backtesting Engine**: Tests strategies against historical data

### 🚀 SmallCap Momentum System (Senior Trader v2.1)
- **3-Tier Classification**: C (Early/Continuation), B (Momentum), A (Continuation)
- **Float Tiering**: Atomic (≤15M), Micro (15-30M), Small (30-50M), Tight (50-60M)
- **RSI Bullish Divergence**: Early reversal detection
- **Sector RS Analysis**: Highlights sector leaders (+12 bonus score)
- **Finviz Integration**: Live momentum universe

### 🤖 Agentic & Generative AI Features
| Feature | Trigger | Output |
|---------|-----------|-------|
| **📝 Weekly Report** | Performance tab → "Generate Report" | Trader-style summary + strategic advice |
| **🤖 Signal Briefing** | Manual Lookup → when a stock is scanned | 2-3 sentence setup interpretation ("Strong R/R, high volatility...") |
| **💬 Strategy Advisor (Agentic)** | Performance tab → free text chat | Autonomous AI that can fetch **live stock data** (Price, Volume, MA20/50) via Tool Calling, combined with your full trade history |

> Without an API key, all AI features operate in **deterministic fallback** mode.

### 🧮 XGBoost ML Signal Prediction
- Trained on historical closed trades (`ml/trainer.py`)
- Calculates **win probability** (0–100%) for each signal
- Feature importance explanation via **SHAP**
- Displayed as a badge in Manual Lookup: `🤖 AI Forecast: 72% — High Confidence`
- Unlocks after 50+ closed trades

### 📊 Paper Trading System
- **Next-Day Confirmation Mechanism**: Signals enter as PENDING, confirmed at the Open price the next day
- **Gap Filter**: Gap-up > +5% or Gap-down < -3% → Auto-REJECT
- **Modern Card UI**: Entry / Stop / Target / R/R metrics for each pending signal
- **Manual Approve/Reject Buttons**: One-click ✅ Approve or ❌ Cancel
- **Trailing Stop**: ATR-based, activates as the position matures
- **Auto-Close Triggers**: Target hit, Stop hit, Timeout, Trailing stop hit

---

## 🚀 Installation

### 1. Requirements
- Python 3.8+
- Internet connection (for market data)

### 2. Setup
```bash
git clone https://github.com/Yigit033/swing_trade.git
cd swing_trade

python -m venv venv
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### 3. `.env` File (Optional — for AI Features)
```env
# AI Provider (choose one)
LLM_PROVIDER=groq          # or: openai, gemini

# API Keys
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
```
> The system will continue to work without an API key — AI features will gracefully fall back to deterministic responses.

### 4. Start the Application
```bash
# Backend (FastAPI)
uvicorn api.main:app --reload --port 8000

# Frontend (Next.js) — in a separate terminal
cd frontend && npm run dev
```
Opens `http://localhost:5000` in your browser. (On Windows, `SwingTrade_Dashboard.bat` starts both simultaneously.)

---

## 🖥️ Dashboard Pages (Next.js)

| Page | Content |
|-------|--------|
| **🚀 Scanner** | SmallCap momentum scan (background job + live progress) |
| **🗂 Scanner History** | Historical scans + forward-return tracking |
| **📝 Manual Lookup** | Single stock, step-by-step diagnosis (filter → trigger → score) |
| **📊 Paper Trades** | Active/Pending/Closed trade tracking, pending approval flow |
| **📉 Performance** | Win rate, profit factor, weekly report |
| **⚙️ Settings** | Engine parameters (JSON based, editable via UI) |
| **🤖 AI / Chat** | XGBoost train/predict + autonomous strategy chat |

---

## 📁 Project Structure

```
swing_trade/
├── api/                   # FastAPI backend (9 routers)
├── frontend/              # Next.js dashboard
├── swing_trader/
│   ├── genai/             # Generative & Agentic AI modules
│   │   ├── llm_client.py      # Provider-agnostic client (Groq/OpenAI/Gemini)
│   │   ├── prompts.py         # Prompt builders
│   │   ├── reporter.py        # Weekly report orchestrator
│   │   ├── signal_briefer.py  # Signal briefing orchestrator
│   │   ├── strategy_chat.py   # Strategy Q&A orchestrator
│   │   ├── tools.py           # LLM Tool Calling definitions and callbacks
│   │   └── data_collector.py  # Deterministic data collection from DB
│   ├── ml/                # Classical ML
│   │   ├── trainer.py         # XGBoost training
│   │   ├── predictor.py       # Win probability prediction
│   │   └── features.py        # Feature engineering
│   ├── small_cap/         # SmallCap Momentum engine
│   │   ├── engine.py          # Core scanning engine
│   │   ├── scoring.py         # Quality scoring (0-100+)
│   │   ├── narrative.py       # Textual analysis generation
│   │   └── risk.py            # ATR-based risk calculations
│   ├── paper_trading/     # Paper Trade system
│   │   ├── tracker.py         # Trade tracking, gap filters, trailing stop
│   │   ├── storage.py         # SQLite CRUD operations
│   │   └── reporter.py        # Performance summary
│   └── data/              # Fetcher + scan history storage
├── data/
│   └── paper_trades.db    # SQLite database
├── config.yaml            # Global configurations
├── requirements.txt
└── .env                   # API keys (ignored by git)
```

---

## 📋 Daily Workflow

```bash
# 1. Start Dashboard (backend + frontend)
SwingTrade_Dashboard.bat   # or: uvicorn api.main:app --port 8000  +  cd frontend && npm run dev

# 2. Scan via SmallCap or Manual Lookup
# → Pin favorable signals to PENDING using "📌 Track"

# 3. Next day, click Update Prices
# → System fetches next day's open price, applies gap filter
# → Approved trades become OPEN, rejected trades are logged as REJECTED

# 4. Monitor Active Trades
# → Auto-closes on Stop/Target/Timeout
# → Trailing stop activates (ATR-based, once position matures)

# 5. Get Weekly Report in Performance Tab
# → "Generate Report" → LLM analyzes trade history
# → Ask the Strategy Advisor free-form questions (e.g., about specific tickers)
```

---

## 🔧 ML Model Training

The XGBoost model can be trained after 50+ closed trades:

```bash
# In Dashboard → 🤖 AI Model tab → "Train Model"
# Or via CLI:
python -c "from swing_trader.ml.trainer import ModelTrainer; ModelTrainer().train()"
```

After training, the `🤖 AI Forecast: XX% — Confidence` badge will appear in Manual Lookup.

---

## 🛡️ Risk Parameters

| Parameter | Large Cap | SmallCap |
|-----------|-----------|---------|
| Max risk / trade | 2% | 0.5% |
| Max position size | 20% portfolio | 5% portfolio |
| Stop loss | ATR × 2.0 | ATR × 1.0 (capped at 12%) |
| Gap-up reject | — | > 5% |
| Gap-down reject | — | < -3% |
| Trailing stop | — | ATR-based (at 2+ ATR profit) |

---

## ⚠️ Risk Warning

> Trading stocks carries significant risk of loss. Past performance does not guarantee future results. This software is for educational purposes only. Always consult a financial advisor before making investment decisions.

**Paper trade for at least 3 months before using real capital.**

---

## 📄 Version History

| Version | Features |
|----------|--------|
| v3.1 | **Agentic AI**: Added Tool Calling to Strategy Advisor for live market data retrieval (Groq/OpenAI) |
| v3.0 | GenAI features (Signal Briefer, Strategy Chat, Weekly Reporter), modern Pending UI, quality score fixes |
| v2.1 | SmallCap Senior Trader: sector RS, insider bonus, short squeeze, trailing stop |
| v2.0 | Paper Trading system, gap filter, next-day confirmation |
| v1.5 | XGBoost ML signal prediction, SHAP feature importance |
| v1.0 | Basic scanning, technical analysis, backtesting, initial web dashboard |
