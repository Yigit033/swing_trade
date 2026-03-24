"""
Streamlit dashboard for swing trading system.
Run with: streamlit run swing_trader/dashboard/app.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yaml
import logging

from swing_trader.data.storage import DatabaseManager
from swing_trader.data.fetcher import DataFetcher
from swing_trader.data.updater import DataUpdater
from swing_trader.strategy.signals import SignalGenerator
from swing_trader.strategy.scoring import SignalScorer
from swing_trader.strategy.risk_manager import RiskManager
from swing_trader.backtesting.engine import BacktestEngine
from swing_trader.backtesting.metrics import PerformanceMetrics
from swing_trader.small_cap import SmallCapEngine  # NEW: Independent SmallCap Engine
from swing_trader.paper_trading.storage import PaperTradeStorage
from swing_trader.paper_trading.tracker import PaperTradeTracker
from swing_trader.paper_trading.reporter import PaperTradeReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Page config
st.set_page_config(
    page_title="Swing Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_resource
def load_config():
    """Load configuration file."""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return None

# ============================================================
# SESSION STATE INITIALIZATION - Persist results across pages
# ============================================================
def init_session_state():
    """Initialize session state for result persistence."""
    # SmallCap scan results
    if 'smallcap_results' not in st.session_state:
        st.session_state.smallcap_results = []
    if 'smallcap_stats' not in st.session_state:
        st.session_state.smallcap_stats = None
    
    # Manual Lookup results
    if 'manual_results' not in st.session_state:
        st.session_state.manual_results = None
    if 'manual_tickers' not in st.session_state:
        st.session_state.manual_tickers = ""
    if 'manual_scan_time' not in st.session_state:
        st.session_state.manual_scan_time = None
    
    # Auto-track settings
    if 'auto_track_enabled' not in st.session_state:
        st.session_state.auto_track_enabled = True
    if 'auto_track_min_quality' not in st.session_state:
        st.session_state.auto_track_min_quality = 65
    if 'last_auto_tracked' not in st.session_state:
        st.session_state.last_auto_tracked = []
    
    # Scan history tracking
    if 'scan_history' not in st.session_state:
        st.session_state.scan_history = []  # List of {'timestamp': dt, 'type': 'smallcap'/'manual', 'signals': N}
    
    # LargeCap scan results
    if 'largecap_results' not in st.session_state:
        st.session_state.largecap_results = None
    if 'largecap_scan_time' not in st.session_state:
        st.session_state.largecap_scan_time = None

# Initialize session state
init_session_state()

# Initialize components
@st.cache_resource
def init_components():
    """Initialize system components."""
    config = load_config()
    if not config:
        return None
    
    db = DatabaseManager(config['data']['database_path'])
    return {
        'config': config,
        'db': db,
        'fetcher': DataFetcher(config['data']['source']),
        'updater': DataUpdater(config),
        'signal_generator': SignalGenerator(config),
        'scorer': SignalScorer(config),
        'risk_manager': RiskManager(config),
        'small_cap_engine': SmallCapEngine(config)  # NEW: Independent SmallCap Engine
    }

def create_candlestick_chart(df: pd.DataFrame, ticker: str):
    """Create candlestick chart with indicators."""
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ))
    
    # Add EMAs if available
    for ema in ['EMA_20', 'EMA_50', 'EMA_200']:
        if ema in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df[ema],
                name=ema,
                line=dict(width=1)
            ))
    
    # Add support/resistance if available
    if 'Support_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Support_20'],
            name='Support',
            line=dict(color='green', dash='dash', width=1)
        ))
    
    if 'Resistance_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Resistance_20'],
            name='Resistance',
            line=dict(color='red', dash='dash', width=1)
        ))
    
    fig.update_layout(
        title=f'{ticker} - Price Chart',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_indicator_chart(df: pd.DataFrame, indicator: str, ticker: str):
    """Create indicator chart."""
    fig = go.Figure()
    
    if indicator == 'RSI' and indicator in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['RSI'],
            name='RSI',
            line=dict(color='purple')
        ))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig.update_yaxes(range=[0, 100])
        
    elif indicator == 'MACD' and 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_signal'], name='Signal', line=dict(color='orange')))
        fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_hist'], name='Histogram'))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
    elif indicator == 'Volume' and 'Volume' in df.columns:
        colors = ['red' if close < open_ else 'green' 
                  for close, open_ in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color=colors))
        
        if 'Volume_MA' in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Volume_MA'], name='Volume MA', line=dict(color='orange')))
    
    fig.update_layout(
        title=f'{ticker} - {indicator}',
        height=250,
        showlegend=True
    )
    
    return fig

def scan_page(components):
    """Scan Results page with persistent results."""
    st.title("📊 Daily Stock Scan")
    
    # Initialize session state for scan results
    if 'scan_results' not in st.session_state:
        st.session_state.scan_results = None
    if 'scan_stats' not in st.session_state:
        st.session_state.scan_stats = None
    if 'scan_data_dict' not in st.session_state:
        st.session_state.scan_data_dict = {}
    # NEW: Store raw (unfiltered) signals for dynamic filtering
    if 'raw_signals' not in st.session_state:
        st.session_state.raw_signals = []
    
    # Scan parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        portfolio_value = st.number_input("Portfolio Value ($)", value=10000, step=1000)
    
    with col2:
        min_quality = st.slider("Minimum Quality Score", 30, 80, 45, help="Quality score 0-100 (higher = better)")
    
    with col3:
        top_n = st.number_input("Top N Results", value=6, min_value=1, max_value=20)
    
    # Run Scan button
    if st.button("🔍 Run Scan", type="primary"):
        with st.spinner("Scanning stocks..."):
            try:
                from datetime import datetime
                
                # Get filtered tickers
                tickers = components['updater'].get_filtered_tickers()
                
                if not tickers:
                    st.session_state.scan_results = []
                    st.session_state.scan_stats = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'stocks_scanned': 0,
                        'raw_signals': 0,
                        'filtered_signals': 0,
                        'reason': 'no_data'
                    }
                else:
                    st.info(f"Scanning {len(tickers)} stocks...")
                    
                    # Get data for all tickers
                    data_dict = {}
                    stocks_with_data = 0
                    stocks_insufficient = 0
                    
                    for ticker in tickers:
                        df = components['db'].get_stock_data(ticker, limit=250)
                        if df is not None and len(df) >= 50:
                            data_dict[ticker] = df
                            stocks_with_data += 1
                        else:
                            stocks_insufficient += 1
                    
                    # Get SPY data for market regime
                    spy_data = components['db'].get_stock_data('SPY', limit=250)
                    if spy_data is not None and len(spy_data) >= 200:
                        data_dict['SPY'] = spy_data
                    
                    # Store data_dict in session state for chart display
                    st.session_state.scan_data_dict = data_dict
                    
                    # Generate signals (with SPY for market regime)
                    signals = components['signal_generator'].scan_stocks(tickers, data_dict, spy_data)
                    raw_signal_count = len(signals)
                    market_regime = components['signal_generator']._market_regime
                    
                    # Add risk management
                    for signal in signals:
                        components['risk_manager'].add_risk_management_to_signal(signal, portfolio_value)
                    
                    # Rank signals
                    signals = components['scorer'].rank_signals(signals)
                    
                    # Store RAW (unfiltered) signals for dynamic filtering later
                    st.session_state.raw_signals = signals.copy()
                    
                    # Get regime-adjusted parameters
                    regime_params = components['signal_generator'].get_regime_adjusted_params()
                    effective_min_quality = max(min_quality, regime_params['min_quality_score'])
                    effective_top_n = min(top_n, regime_params['top_signals_count'])
                    
                    # Filter by quality score
                    filtered_signals = [s for s in signals if s.get('quality_score', 0) >= effective_min_quality]
                    filtered_count = len(filtered_signals)
                    filtered_signals = filtered_signals[:effective_top_n]
                    
                    # Store filtered results in session state
                    st.session_state.scan_results = filtered_signals
                    st.session_state.scan_stats = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'stocks_scanned': len(tickers),
                        'stocks_with_data': stocks_with_data,
                        'stocks_insufficient': stocks_insufficient,
                        'raw_signals': raw_signal_count,
                        'total_qualified': len(signals),  # Before quality filter
                        'market_regime': market_regime,
                        'filtered_signals': filtered_count,
                        'min_quality_used': min_quality,
                        'reason': 'success' if filtered_count > 0 else 'no_qualifying'
                    }
                    
            except Exception as e:
                st.error(f"Error during scan: {e}")
                logging.exception("Scan error")
                st.session_state.scan_results = []
                st.session_state.scan_stats = {'reason': 'error', 'error': str(e)}
    
    # Display results (from session state - persists across page switches)
    st.divider()
    
    # DYNAMIC FILTERING: Apply current slider values to raw_signals
    if st.session_state.raw_signals and len(st.session_state.raw_signals) > 0:
        # Filter raw signals with CURRENT slider values (not scan-time values)
        dynamically_filtered = [
            s for s in st.session_state.raw_signals 
            if s.get('quality_score', 0) >= min_quality
        ]
        dynamically_filtered = dynamically_filtered[:top_n]
        
        # Update scan_results with dynamically filtered signals
        st.session_state.scan_results = dynamically_filtered
    
    if st.session_state.scan_stats is not None:
        stats = st.session_state.scan_stats
        signals = st.session_state.scan_results or []
        
        # Show scan info with dynamic filter indicator
        if 'timestamp' in stats:
            raw_count = len(st.session_state.raw_signals) if st.session_state.raw_signals else 0
            st.caption(f"📅 Last scan: {stats['timestamp']} | 📊 Total qualified: {raw_count} → Showing: {len(signals)} (Min Quality: {min_quality}, Top N: {top_n})")
        
        # Results summary
        if stats.get('reason') == 'no_data':
            st.error("❌ **No Data in Database**")
            st.markdown("""
            ### Why no signals?
            - **No stock data found in database**
            
            ### How to fix:
            1. Go to **⚙️ Settings** → **📥 Download Data**
            2. Click **Download S&P 500 Data**
            3. Wait for download to complete
            4. Return here and scan again
            """)
            
        elif stats.get('reason') == 'no_qualifying':
            st.warning(f"⚠️ **Found {stats['raw_signals']} signals, but 0 passed your filters**")
            
            st.markdown(f"""
            ### Why 0 qualifying signals?
            
            | Metric | Value |
            |--------|-------|
            | Stocks scanned | {stats.get('stocks_scanned', 0)} |
            | Stocks with data | {stats.get('stocks_with_data', 0)} |
            | Raw signals found | {stats['raw_signals']} |
            | Passed score ≥{stats['min_score_used']} filter | 0 |
            
            ### Possible reasons:
            1. **Score threshold too high** - Try lowering "Minimum Signal Score" to 5 or 6
            2. **Market conditions** - No stocks currently meet all entry criteria
            3. **RSI filter** - Stocks may be overbought (RSI > 70), rejected for safety
            4. **Data quality** - Some stock data may be stale or corrupted
            
            ### Try these fixes:
            - Lower the minimum score slider
            - Update data in Settings → Download Data
            - Check Database Viewer for data freshness
            """)
            
        elif stats.get('reason') == 'error':
            st.error(f"❌ Scan failed: {stats.get('error', 'Unknown error')}")
            
        elif len(signals) > 0:
            # Success - show results
            market_regime = stats.get('market_regime', 'RISK_ON')
            regime_emoji = "🟢" if market_regime == 'RISK_ON' else "🔴"
            st.success(f"✅ Found {len(signals)} signals! | Market: {regime_emoji} {market_regime}")
            
            # Summary metrics
            df_signals = components['scorer'].create_signals_dataframe(signals)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Signals", len(signals))
            avg_quality = df_signals['quality_score'].mean() if 'quality_score' in df_signals else df_signals['score'].mean()
            col2.metric("Avg Quality", f"{avg_quality:.0f}")
            col3.metric("Scanned", stats.get('stocks_scanned', 0))
            col4.metric("Raw Signals", stats.get('raw_signals', 0))
            col5.metric("Regime", f"{regime_emoji} {market_regime}")
            
            # Signals table
            st.subheader("📋 Top Signals")
            
            # Create display dataframe with time columns
            display_df = df_signals.copy()
            
            # Create Hold Period column (e.g., "5-15d")
            display_df['hold_period'] = display_df.apply(
                lambda x: f"{int(x.get('expected_hold_min', 5))}-{int(x.get('expected_hold_max', 15))}d", 
                axis=1
            )
            
            # Create Expires column with color indicator
            from datetime import datetime
            today = datetime.now().date()
            
            def format_expires(exp_date):
                if not exp_date or exp_date == '':
                    return "N/A"
                try:
                    exp_dt = datetime.strptime(exp_date, '%Y-%m-%d').date()
                    days_left = (exp_dt - today).days
                    if days_left < 0:
                        return "❌ Expired"
                    elif days_left == 0:
                        return "⚠️ Today"
                    elif days_left == 1:
                        return "🟡 1 day"
                    elif days_left == 2:
                        return "🟢 2 days"
                    else:
                        return f"🟢 {days_left} days"
                except:
                    return "N/A"
            
            display_df['expires'] = display_df['expiration_date'].apply(format_expires)
            
            # Format other columns
            display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
            display_df['stop_loss'] = display_df['stop_loss'].apply(lambda x: f"${x:.2f}")
            display_df['target_1'] = display_df['target_1'].apply(lambda x: f"${x:.2f}")
            display_df['rsi'] = display_df['rsi'].apply(lambda x: f"{x:.1f}")
            display_df['volume_surge'] = display_df['volume_surge'].apply(lambda x: f"{x:.2f}x")
            
            # Add quality score formatting
            if 'quality_score' in display_df.columns:
                display_df['quality'] = display_df['quality_score'].apply(lambda x: f"{x:.0f}")
            else:
                display_df['quality'] = display_df['score'].apply(lambda x: f"{x*10:.0f}")
            
            # Select and reorder columns for display
            final_display = display_df[['ticker', 'quality', 'entry_price', 'stop_loss', 
                                         'target_1', 'hold_period', 'expires', 'rsi', 'volume_surge']]
            
            st.dataframe(final_display, use_container_width=True, height=400)
            
            # Signal explanation
            with st.expander("📚 How to Use These Signals"):
                st.markdown("""
                | Column | Meaning |
                |--------|---------|
                | **ticker** | Stock symbol to trade |
                | **quality** | Trade quality score 0-100 (50+ good, 70+ excellent) |
                | **entry_price** | Buy at or below this price |
                | **stop_loss** | Sell immediately if price drops to this level |
                | **target_1** | Take profit when price reaches this level |
                | **hold_period** | Expected days to reach target (based on volatility) |
                | **expires** | Signal validity - don't enter after it expires |
                | **rsi** | Relative Strength - lower is better for entry |
                | **volume_surge** | Volume vs average - higher = more interest |
                
                ### Trading Rules:
                1. 🟢 **Enter** when price is at or near entry_price
                2. 🔴 **Stop Loss** - Always set stop loss immediately after buying
                3. 🎯 **Take Profit** - Sell when target_1 is reached
                4. ⏰ **Expires** - Don't enter if signal is expired
                """)
            
            # Chart for selected stock
            st.subheader("📈 Chart Analysis")
            selected_ticker = st.selectbox("Select stock to analyze:", df_signals['ticker'].tolist())
            
            data_dict = st.session_state.scan_data_dict
            if selected_ticker in data_dict:
                ticker_df = data_dict[selected_ticker].copy()
                ticker_df = components['signal_generator'].calculate_all_indicators(ticker_df)
                
                # Price chart
                fig_price = create_candlestick_chart(ticker_df.tail(60), selected_ticker)
                st.plotly_chart(fig_price, use_container_width=True)
                
                # Indicator charts
                col1, col2 = st.columns(2)
                with col1:
                    fig_rsi = create_indicator_chart(ticker_df.tail(60), 'RSI', selected_ticker)
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                with col2:
                    fig_macd = create_indicator_chart(ticker_df.tail(60), 'MACD', selected_ticker)
                    st.plotly_chart(fig_macd, use_container_width=True)
                
                # Volume chart
                fig_vol = create_indicator_chart(ticker_df.tail(60), 'Volume', selected_ticker)
                st.plotly_chart(fig_vol, use_container_width=True)
    else:
        # No scan run yet
        st.info("👆 Click **Run Scan** to find trading signals")
        
        with st.expander("ℹ️ What does the scan do?"):
            st.markdown("""
            The scan analyzes S&P 500 stocks looking for swing trading opportunities:
            
            1. **Checks trend** - Price must be above 200-day moving average
            2. **Checks momentum** - RSI and MACD must show bullish signals
            3. **Checks volume** - Volume must be above average
            4. **Filters overbought** - Rejects stocks with RSI > 70
            5. **Calculates targets** - Stop-loss and take-profit levels
            6. **Ranks by score** - Best opportunities shown first
            
            **Tip:** Lower the minimum score if you want more signals (but lower quality).
            """)

def backtest_page(components):
    """Backtest page."""
    st.title("📉 Strategy Backtest")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2022, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2024, 12, 31))
    
    initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
    
    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest... This may take several minutes."):
            try:
                # Get tickers
                tickers = components['updater'].get_filtered_tickers()[:50]  # Limit for performance
                
                if not tickers:
                    st.warning("No tickers in database.")
                    return
                
                st.info(f"Backtesting on {len(tickers)} stocks from {start_date} to {end_date}")
                
                # Get data
                data_dict = {}
                for ticker in tickers:
                    df = components['db'].get_stock_data(
                        ticker,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    )
                    if df is not None and len(df) >= 50:
                        data_dict[ticker] = df
                
                # Run backtest
                engine = BacktestEngine(components['config'])
                engine.initial_capital = initial_capital
                
                results = engine.run_backtest(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    data_dict
                )
                
                # Calculate metrics
                metrics = PerformanceMetrics.calculate_metrics(results)
                
                # Display results
                st.success("Backtest complete!")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return", f"{metrics['total_return']:.2%}", 
                           f"${metrics['total_pnl']:,.2f}")
                col2.metric("Win Rate", f"{metrics['win_rate']:.1%}")
                col3.metric("Total Trades", metrics['total_trades'])
                col4.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                
                # Detailed metrics
                st.subheader("📊 Performance Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Return Metrics**")
                    st.write(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
                    st.write(f"Final Value: ${metrics['final_value']:,.2f}")
                    st.write(f"CAGR: {metrics['cagr']:.2%}")
                    st.write(f"Max Drawdown: {metrics['max_drawdown_percent']:.1f}%")
                    
                with col2:
                    st.markdown("**Trade Statistics**")
                    st.write(f"Winning Trades: {metrics['winning_trades']}")
                    st.write(f"Losing Trades: {metrics['losing_trades']}")
                    st.write(f"Avg Win: ${metrics['avg_win']:,.2f}")
                    st.write(f"Avg Loss: ${metrics['avg_loss']:,.2f}")
                    st.write(f"Profit Factor: {metrics['profit_factor']:.2f}")
                
                # Equity curve
                if results['equity_curve']:
                    st.subheader("📈 Equity Curve")
                    equity_df = pd.DataFrame(results['equity_curve'])
                    equity_df['date'] = pd.to_datetime(equity_df['date'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity_df['date'],
                        y=equity_df['portfolio_value'],
                        name='Portfolio Value',
                        line=dict(color='blue', width=2)
                    ))
                    fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray", 
                                 annotation_text="Initial Capital")
                    fig.update_layout(
                        title='Portfolio Value Over Time',
                        yaxis_title='Portfolio Value ($)',
                        xaxis_title='Date',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Trade log
                if results['trades']:
                    st.subheader("📝 Trade Log")
                    trades_df = pd.DataFrame(results['trades'])
                    st.dataframe(trades_df, use_container_width=True, height=300)
                
            except Exception as e:
                st.error(f"Error during backtest: {e}")
                logging.exception("Backtest error")

def settings_page(components):
    """Settings page with data management and database viewer."""
    st.title("⚙️ Settings & Data Management")
    
    # Create tabs for different settings sections
    tab1, tab2, tab3 = st.tabs(["📥 Download Data", "📊 Database Viewer", "🔧 Configuration"])
    
    # =====================================================
    # TAB 1: DOWNLOAD DATA
    # =====================================================
    with tab1:
        st.subheader("Download Stock Data")
        st.write("Download historical data for S&P 500 stocks from Yahoo Finance")
        
        col1, col2 = st.columns(2)
        with col1:
            days = st.number_input("Days of history to download", value=500, min_value=100, max_value=2000, step=50)
        with col2:
            st.write("")
            st.write(f"**Estimated time:** ~{days//100 * 3} minutes")
        
        if st.button("🚀 Download S&P 500 Data (Full Refresh)", type="primary"):
            try:
                # Get tickers
                tickers = components['fetcher'].get_sp500_tickers()
                st.info(f"Found {len(tickers)} S&P 500 stocks to download")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Download with progress
                successful = 0
                failed = 0
                
                for i, ticker in enumerate(tickers):
                    try:
                        status_text.text(f"Downloading {ticker}... ({i+1}/{len(tickers)})")
                        df = components['fetcher'].fetch_stock_data(ticker, period=f"{days}d")
                        
                        if df is not None and len(df) > 0:
                            components['db'].insert_stock_data(ticker, df)
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                        logging.warning(f"Failed {ticker}: {e}")
                    
                    progress_bar.progress((i + 1) / len(tickers))
                
                status_text.empty()
                progress_bar.empty()
                st.success(f"✅ Download complete! {successful} successful, {failed} failed")
                st.balloons()
                
            except Exception as e:
                st.error(f"Error during download: {e}")
                logging.exception("Download error")
        
        st.divider()
        
        # Quick update option
        st.subheader("Update Existing Data")
        st.write("Update only the latest prices (faster than full download)")
        
        if st.button("🔄 Update All Stocks to Today"):
            with st.spinner("Updating stock data..."):
                try:
                    results = components['updater'].daily_update()
                    successful = sum(1 for v in results.values() if v)
                    st.success(f"Updated {successful}/{len(results)} stocks to latest data")
                except Exception as e:
                    st.error(f"Error updating: {e}")
    
    # =====================================================
    # TAB 2: DATABASE VIEWER
    # =====================================================
    with tab2:
        st.subheader("Database Overview")
        
        # Get all tickers
        tickers = components['db'].get_all_tickers()
        
        if not tickers:
            st.warning("⚠️ No data in database. Go to 'Download Data' tab to get started.")
        else:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Stocks", len(tickers))
            
            # Get sample dates for freshness check
            from datetime import datetime, timedelta
            today = datetime.now().date()
            
            # Build database overview
            db_data = []
            fresh_count = 0
            stale_count = 0
            
            with st.spinner("Loading database info..."):
                for ticker in tickers[:100]:  # Limit to first 100 for speed
                    try:
                        df = components['db'].get_stock_data(ticker, limit=1)
                        if df is not None and len(df) > 0:
                            last_date = df.iloc[-1]['Date']
                            last_price = df.iloc[-1]['Close']
                            
                            # Parse date
                            if isinstance(last_date, str):
                                last_dt = datetime.strptime(last_date[:10], '%Y-%m-%d').date()
                            else:
                                last_dt = last_date.date() if hasattr(last_date, 'date') else today
                            
                            days_old = (today - last_dt).days
                            is_fresh = days_old <= 3
                            
                            if is_fresh:
                                fresh_count += 1
                            else:
                                stale_count += 1
                            
                            db_data.append({
                                'Ticker': ticker,
                                'Last Price': f"${last_price:.2f}",
                                'Last Date': last_dt.strftime('%Y-%m-%d'),
                                'Days Old': days_old,
                                'Status': '🟢 Fresh' if is_fresh else '🔴 Stale'
                            })
                    except:
                        pass
            
            col2.metric("Fresh Data (≤3 days)", fresh_count)
            col3.metric("Stale Data (>3 days)", stale_count)
            col4.metric("Data Quality", f"{fresh_count/(fresh_count+stale_count)*100:.0f}%" if (fresh_count+stale_count) > 0 else "N/A")
            
            # Display database table
            if db_data:
                st.subheader("Stock Data Status")
                db_df = pd.DataFrame(db_data)
                
                # Filter options
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    status_filter = st.selectbox("Filter by status", ["All", "Fresh Only", "Stale Only"])
                with filter_col2:
                    search = st.text_input("Search ticker", "")
                
                # Apply filters
                display_df = db_df.copy()
                if status_filter == "Fresh Only":
                    display_df = display_df[display_df['Status'].str.contains('Fresh')]
                elif status_filter == "Stale Only":
                    display_df = display_df[display_df['Status'].str.contains('Stale')]
                
                if search:
                    display_df = display_df[display_df['Ticker'].str.contains(search.upper())]
                
                st.dataframe(display_df, use_container_width=True, height=400)
            
            # Individual stock viewer
            st.divider()
            st.subheader("View Individual Stock Data")
            
            selected_ticker = st.selectbox("Select a stock to view details:", tickers)
            
            if selected_ticker:
                df = components['db'].get_stock_data(selected_ticker, limit=30)
                if df is not None and len(df) > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"{selected_ticker} Latest Price", f"${df.iloc[-1]['Close']:.2f}")
                    with col2:
                        st.metric("Total Records", len(components['db'].get_stock_data(selected_ticker, limit=10000)))
                    
                    st.write("**Last 30 Days:**")
                    st.dataframe(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(30), use_container_width=True)
    
    # =====================================================
    # TAB 3: CONFIGURATION
    # =====================================================
    with tab3:
        st.subheader("System Configuration")
        
        config = components['config']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Settings**")
            st.write(f"Max Risk per Trade: {config['risk']['max_risk_per_trade']:.1%}")
            st.write(f"Max Position Size: {config['risk']['max_position_size']:.1%}")
            st.write(f"Max Open Positions: {config['risk']['max_open_positions']}")
            st.write(f"Stop-Loss ATR Multiplier: {config['risk']['stop_loss_atr_multiplier']}x")
        
        with col2:
            st.markdown("**Strategy Settings**")
            st.write(f"Min Signal Score: {config['strategy']['min_signal_score']}")
            st.write(f"RSI Entry Range: {config['strategy']['rsi_entry_min']}-{config['strategy']['rsi_entry_max']}")
            st.write(f"Max Holding Days: {config['strategy']['max_holding_days']}")
            st.write(f"Signal Expiration: {config['strategy']['signal_expiration_days']} days")
        
        st.divider()
        st.info("💡 To modify settings, edit the `config.yaml` file in the project root.")


def manual_lookup_page(components):
    """
    Manual Ticker Lookup - Analyze any ticker against swing trade strategies.
    Routes to SmallCap or LargeCap engine based on market cap.
    """
    st.title("📝 Manual Ticker Lookup")
    st.markdown("*Analyze specific tickers against swing trade strategies*")
    
    # Input section
    st.markdown("### Enter Tickers")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ticker_input = st.text_input(
            "Tickers (comma separated)",
            value=st.session_state.manual_tickers,
            placeholder="AAPL, TSLA, PLUG, NTLA",
            help="Enter one or more tickers separated by commas"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    # Portfolio value for position sizing
    portfolio_value = st.number_input("Portfolio Value ($)", value=10000, step=1000, key="ml_portfolio")
    
    st.divider()
    
    if analyze_button and ticker_input.strip():
        # Parse tickers
        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
        
        if not tickers:
            st.warning("Please enter at least one ticker.")
            return
        
        st.markdown(f"### Analyzing {len(tickers)} ticker(s)...")
        
        # Get engines
        small_cap_engine = components['small_cap_engine']
        signal_generator = components['signal_generator']
        fetcher = components['fetcher']
        
        progress = st.progress(0)
        results = []
        
        for i, ticker in enumerate(tickers):
            progress.progress((i + 1) / len(tickers))
            
            try:
                # Fetch data
                df = fetcher.fetch_stock_data(ticker, period='3mo')
                
                if df is None or len(df) < 21:
                    results.append({
                        'ticker': ticker,
                        'status': 'error',
                        'message': 'Insufficient data or invalid ticker'
                    })
                    continue
                
                # Get stock info
                import yfinance as yf
                stock = yf.Ticker(ticker)
                info = stock.info
                
                market_cap = info.get('marketCap', 0)
                float_shares = info.get('floatShares', 0)
                sector = info.get('sector', 'Unknown')
                company_name = info.get('shortName', ticker)
                
                # Route to correct engine based on market cap
                if market_cap < 3_000_000_000:  # < $3B = SmallCap
                    result = analyze_smallcap_ticker(
                        ticker, df, info, small_cap_engine, portfolio_value
                    )
                    result['strategy'] = 'SmallCap'
                else:  # >= $3B = LargeCap
                    result = analyze_largecap_ticker(
                        ticker, df, info, signal_generator, components, portfolio_value
                    )
                    result['strategy'] = 'LargeCap'
                
                result['ticker'] = ticker
                result['company_name'] = company_name
                result['sector'] = sector
                result['market_cap'] = market_cap
                result['float_shares'] = float_shares
                results.append(result)
                
            except Exception as e:
                results.append({
                    'ticker': ticker,
                    'status': 'error',
                    'message': str(e)
                })
        
        progress.empty()
        
        # Store results in session state
        st.session_state.manual_results = results
        st.session_state.manual_tickers = ticker_input
        st.session_state.manual_scan_time = datetime.now()
        
        # Log scan to history
        swing_ready_count = sum(1 for r in results if r.get('swing_ready', False))
        st.session_state.scan_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'manual',
            'signals': swing_ready_count
        })
        
        # AUTO-TRACK for Manual Lookup
        if st.session_state.auto_track_enabled and results:
            auto_quality = st.session_state.auto_track_min_quality
            paper_storage = PaperTradeStorage()
            paper_tracker = PaperTradeTracker(paper_storage)
            
            auto_tracked = []
            auto_skipped = []
            
            for result in results:
                if (result.get('status') == 'analyzed' 
                    and result.get('swing_ready', False) 
                    and result.get('quality_score', 0) >= auto_quality):
                    
                    # Build signal dict for tracking
                    track_signal = {
                        'ticker': result.get('ticker', ''),
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'entry_price': result.get('entry_price', 0),
                        'stop_loss': result.get('stop_loss', 0),
                        'target_1': result.get('target_1', 0),
                        'swing_type': result.get('swing_type', 'A'),
                        'quality_score': result.get('quality_score', 0),
                        'position_size': result.get('position_size', 100),
                        'hold_days_max': result.get('hold_days', (2, 7))[1] if isinstance(result.get('hold_days'), tuple) else 7,
                        'type_reason': result.get('type_reason', '')
                    }
                    
                    trade_id = paper_tracker.add_trade_from_signal(track_signal)
                    if trade_id > 0:
                        auto_tracked.append(result['ticker'])
                    else:
                        auto_skipped.append(result['ticker'])
            
            if auto_tracked:
                st.success(
                    f"📌 **Auto-Track:** {len(auto_tracked)} sinyal paper trade'e eklendi → "
                    f"{', '.join(auto_tracked)}"
                )
            if auto_skipped:
                st.caption(f"⏭️ Zaten takipte: {', '.join(auto_skipped)}")
    
    # Display persisted results
    if st.session_state.manual_results:
        st.markdown("---")
        scan_time = st.session_state.get('manual_scan_time', '')
        if scan_time:
            st.caption(f"📊 Last scan: {scan_time.strftime('%H:%M:%S')} | Tickers: {st.session_state.manual_tickers}")
        st.markdown("## Analysis Results")
        
        for result in st.session_state.manual_results:
            display_ticker_result(result)


def analyze_smallcap_ticker(ticker, df, info, engine, portfolio_value):
    """Analyze ticker using SmallCap swing trade criteria."""
    from datetime import datetime
    
    result = {'status': 'analyzed'}
    
    # Get stock info dict
    stock_info = {
        'ticker': ticker,
        'marketCap': info.get('marketCap', 0),
        'floatShares': info.get('floatShares', 0),
        'shortName': info.get('shortName', ticker),
        'sector': info.get('sector', 'Unknown')
    }
    
    # Step 1: Apply filters
    filter_passed, filter_results = engine.filters.apply_all_filters(
        ticker, df, stock_info, datetime.now()
    )
    result['filter_passed'] = filter_passed
    result['filter_details'] = filter_results
    
    if not filter_passed:
        result['swing_ready'] = False
        result['rejection_reason'] = 'Failed universe filters'
        # Still calculate RSI for display even on rejection
        result['rsi'] = engine.signals.calculate_rsi(df)
        # Calculate 5-day return
        if len(df) >= 6:
            result['five_day_return'] = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100
        else:
            result['five_day_return'] = 0
        return result
    
    # Step 2: Check triggers
    triggered, trigger_details = engine.signals.check_all_triggers(df)
    result['trigger_passed'] = triggered
    result['trigger_details'] = trigger_details
    
    if not triggered:
        result['swing_ready'] = False
        result['rejection_reason'] = 'No signal trigger (Volume or ATR too low)'
        # Calculate RSI and 5-day return for display
        result['rsi'] = engine.signals.calculate_rsi(df)
        if len(df) >= 6:
            result['five_day_return'] = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100
        else:
            result['five_day_return'] = 0
        # Also add trigger details for visibility
        vol_surge = trigger_details.get('volume_surge', 0)
        atr_pct = trigger_details.get('atr_percent', 0) * 100
        result['rejection_reason'] += f' | VolSurge: {vol_surge:.1f}x | ATR: {atr_pct:.1f}%'
        return result
    
    # Step 3: Check boosters and swing confirmation
    boosters = engine.signals.check_boosters(df)
    swing_ready = boosters.get('swing_ready', False)
    swing_details = boosters.get('swing_details', {})
    
    result['swing_ready'] = swing_ready
    result['boosters'] = boosters
    result['swing_details'] = swing_details
    
    if not swing_ready:
        result['rejection_reason'] = 'Failed swing confirmation'
        return result
    
    # Step 4: Calculate score
    volume_surge = trigger_details.get('volume_surge', 2.0)
    atr_percent = trigger_details.get('atr_percent', 0.06)
    float_shares = filter_results.get('float_shares', 0)
    
    quality_score = engine.scoring.calculate_quality_score(
        df, volume_surge, atr_percent, float_shares, boosters
    )
    result['quality_score'] = quality_score
    
    # Step 5: Get swing type
    five_day_return = swing_details.get('five_day_momentum', {}).get('return', 0)
    ma20_distance = swing_details.get('above_ma20', {}).get('distance', 0)
    rsi = boosters.get('rsi', 50)
    higher_lows = boosters.get('higher_lows', False)
    
    # Calculate close position
    today_high = float(df['High'].iloc[-1])
    today_low = float(df['Low'].iloc[-1])
    today_close = float(df['Close'].iloc[-1])
    day_range = today_high - today_low
    close_position = (today_close - today_low) / day_range if day_range > 0 else 0.5
    
    swing_type, hold_days, type_reason = engine._classify_swing_type(
        five_day_return, rsi, volume_surge, higher_lows,
        close_position=close_position, ma20_distance=ma20_distance
    )
    
    result['swing_type'] = swing_type
    result['hold_days'] = hold_days
    result['type_reason'] = type_reason
    result['five_day_return'] = five_day_return
    result['rsi'] = rsi
    result['volume_surge'] = volume_surge
    result['atr_percent'] = atr_percent
    result['entry_price'] = today_close
    
    # Add risk management
    signal = {
        'entry_price': today_close,
        'atr_percent': atr_percent / 100 if atr_percent > 1 else atr_percent,
        'date': datetime.now().strftime('%Y-%m-%d')
    }
    signal = engine.risk.add_risk_management(signal, df, portfolio_value)
    result['stop_loss'] = signal.get('stop_loss')
    result['target_1'] = signal.get('target_1')
    result['position_size'] = signal.get('position_size')
    
    # Generate narrative for swing-ready signals
    if result.get('swing_ready'):
        try:
            from swing_trader.small_cap.narrative import generate_signal_narrative
            from swing_trader.small_cap.technical_levels import calculate_technical_levels
            
            # Calculate technical levels from OHLCV data
            tech_levels = None
            try:
                tech_levels = calculate_technical_levels(df, today_close, volume_surge)
            except Exception:
                pass
            
            # Build full signal dict for narrative
            narrative_signal = {
                'ticker': ticker,
                'entry_price': today_close,
                'stop_loss': result.get('stop_loss', 0),
                'target_1': result.get('target_1', 0),
                'target_2': signal.get('target_2', 0),
                'quality_score': result.get('quality_score', 0),
                'swing_type': result.get('swing_type', 'A'),
                'volume_surge': result.get('volume_surge', 1.0),
                'atr_percent': result.get('atr_percent', 0),
                'rsi': result.get('rsi', 50),
                'five_day_return': result.get('five_day_return', 0),
                'float_millions': result.get('float_shares', 0) / 1_000_000 if result.get('float_shares') else 0,
                'short_percent': 0,
                'sector_rs_score': 0,
                'is_sector_leader': False,
                'is_squeeze_candidate': False,
                'expected_hold_min': result.get('hold_days', (2, 5))[0],
                'expected_hold_max': result.get('hold_days', (2, 5))[1],
                'type_reason': result.get('type_reason', ''),
                'technical_levels': tech_levels,
                'macd_bullish': result.get('macd_bullish', False),
                'rsi_divergence': result.get('rsi_divergence', False),
                'higher_lows': result.get('higher_lows', False)
            }
            
            narrative = generate_signal_narrative(narrative_signal, language='tr')
            result['narrative'] = narrative
            result['narrative_text'] = narrative.get('full_text', '')
            result['narrative_headline'] = narrative.get('headline', f"{ticker}")
        except Exception as e:
            result['narrative_text'] = ''
            result['narrative_headline'] = f"{ticker} - {result.get('swing_type', 'A')}"
    
    return result


def analyze_largecap_ticker(ticker, df, info, signal_generator, components, portfolio_value):
    """Analyze ticker using LargeCap swing trade criteria."""
    result = {'status': 'analyzed'}
    
    try:
        # First, calculate indicators on the dataframe
        df_with_indicators = signal_generator.calculate_all_indicators(df)
        
        # Generate signal using existing signal generator (singular, not plural)
        signal = signal_generator.generate_signal(ticker, df_with_indicators)
        
        if signal:
            result['swing_ready'] = True
            result['quality_score'] = signal.get('quality_score', 50)
            result['swing_type'] = 'LargeCap'
            result['hold_days'] = (signal.get('hold_min', 5), signal.get('hold_max', 10))
            result['type_reason'] = f"LargeCap swing: {signal.get('entry_type', 'pullback')} setup"
            result['entry_price'] = signal.get('entry_price', float(df['Close'].iloc[-1]))
            result['stop_loss'] = signal.get('stop_loss')
            result['target_1'] = signal.get('target_1')
            result['five_day_return'] = 0  # Calculate manually
            
            # Get metrics from signal
            result['rsi'] = signal.get('rsi', 50)
            result['volume_surge'] = 1.0
            
            # Calculate 5-day return
            if len(df) >= 6:
                result['five_day_return'] = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100
        else:
            result['swing_ready'] = False
            result['rejection_reason'] = 'No LargeCap signal triggered'
            
            # Calculate 5-day return properly
            if len(df) >= 6:
                result['five_day_return'] = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100
            else:
                result['five_day_return'] = 0
            
            # Still calculate RSI for display
            if len(df) >= 14:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
                rs = gain / loss if loss != 0 else 0
                result['rsi'] = 100 - (100 / (1 + rs)) if rs != 0 else 50
            else:
                result['rsi'] = 50
                
    except Exception as e:
        result['swing_ready'] = False
        result['rejection_reason'] = f'Analysis error: {str(e)}'
    
    return result


def display_ticker_result(result):
    """Display analysis result card for a ticker."""
    ticker = result.get('ticker', 'UNKNOWN')
    status = result.get('status', 'analyzed')
    
    if status == 'error':
        st.error(f"**{ticker}**: ❌ {result.get('message', 'Unknown error')}")
        return
    
    swing_ready = result.get('swing_ready', False)
    strategy = result.get('strategy', 'Unknown')
    company_name = result.get('company_name', ticker)
    sector = result.get('sector', 'Unknown')
    market_cap = result.get('market_cap', 0)
    
    # Header with status
    if swing_ready:
        swing_type = result.get('swing_type', 'A')
        hold_days = result.get('hold_days', (3, 7))
        quality_score = result.get('quality_score', 50)
        
        type_labels = {'A': 'Continuation', 'B': 'Momentum', 'C': 'Early Stage', 'LargeCap': 'LargeCap Swing'}
        type_emojis = {'A': '🐢', 'B': '🚀', 'C': '⭐', 'LargeCap': '📈'}
        
        st.success(f"""
        ### {type_emojis.get(swing_type, '📊')} {ticker} - ✅ SWING CANDIDATE
        **{company_name}** | {sector} | ${market_cap/1e9:.1f}B
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Strategy", strategy)
        with col2:
            st.metric("Type", f"{swing_type} ({type_labels.get(swing_type, 'Unknown')})")
        with col3:
            st.metric("Hold", f"{hold_days[0]}-{hold_days[1]} days")
        with col4:
            st.metric("Quality", f"{quality_score:.0f}")
        
        # Metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            entry = result.get('entry_price', 0)
            st.metric("Entry", f"${entry:.2f}")
        with col2:
            stop = result.get('stop_loss', 0)
            if stop:
                st.metric("Stop Loss", f"${stop:.2f}")
        with col3:
            target = result.get('target_1', 0)
            if target:
                st.metric("Target", f"${target:.2f}")
        with col4:
            rsi = result.get('rsi', 50)
            st.metric("RSI", f"{rsi:.0f}")
        with col5:
            five_d = result.get('five_day_return', 0)
            st.metric("5-Day", f"{five_d:+.0f}%")
        
        # Type reason
        type_reason = result.get('type_reason', '')
        if type_reason:
            st.info(f"📋 {type_reason}")
        
        # Position size
        pos_size = result.get('position_size', 0)
        if pos_size:
            st.caption(f"📊 Suggested Position: {pos_size} shares")
        
        # ── AI Sinyal Kalite Tahmini (opsiyonel — model yoksa gizlenir) ──
        # Bu blok tamamen izole: model yüklü değilse hiçbir şey göstermez.
        # swing_trader/ml/trainer.py ile modeli eğittikten sonra burada görünür.
        try:
            from swing_trader.ml.predictor import SignalPredictor
            _ai_predictor = SignalPredictor()
            if _ai_predictor.is_ready:
                _ai_signal = {
                    'entry_price': result.get('entry_price', 0),
                    'stop_loss':   result.get('stop_loss', 0),
                    'target':      result.get('target_1', 0),
                    'atr':         result.get('atr', 0),
                    'quality_score': result.get('quality_score', 0),
                    'swing_type':  result.get('swing_type', 'A'),
                    'max_hold_days': result.get('hold_days', (2, 7))[1] if isinstance(result.get('hold_days'), tuple) else 7,
                    'entry_date':  datetime.now().strftime('%Y-%m-%d'),
                }
                _pred = _ai_predictor.predict(_ai_signal)
                if _pred:
                    _win_pct = int(_pred['win_probability'] * 100)
                    _conf = _pred['confidence']
                    st.info(f"🤖 **AI Tahmin:** Kazanma ihtimali **%{_win_pct}** — {_conf}")
        except Exception:
            pass  # Model yoksa veya hata olursa sessizce geç

        # ── AI Sinyal Brifingi (GenAI — LLM yorumu) ──────────────────
        # XGBoost kazanma ihtimalini sayısal verdi; LLM bunu Türkçe yorumlar.
        # LLM bağlı değilse deterministik özet gösterir.
        try:
            from swing_trader.genai.signal_briefer import SignalBriefer
            _briefer = SignalBriefer()
            _brief_signal = {
                "ticker":        ticker,
                "entry_price":   result.get("entry_price", 0),
                "stop_loss":     result.get("stop_loss", 0),
                "target_1":      result.get("target_1", 0),
                "atr":           result.get("atr", 0),
                "quality_score": result.get("quality_score", 0),
                "swing_type":    result.get("swing_type", "A"),
            }
            _brief = _briefer.brief(_brief_signal)
            if _brief.get("success") and _brief.get("text"):
                _icon = "🤖" if not _brief.get("fallback") else "📊"
                with st.expander(f"{_icon} AI Kurulum Değerlendirmesi", expanded=True):
                    st.markdown(_brief["text"])
                    if _brief.get("fallback"):
                        st.caption("API key ekleyince GPT/Gemini ile daha derin analiz gelir.")
        except Exception:
            pass  # GenAI yoksa veya hata olursa sessizce geç


        narrative_text = result.get('narrative_text', '')
        if narrative_text:
            with st.expander("📝 Detaylı Analiz (Cuma Çevik Tarzı)", expanded=True):
                st.markdown(narrative_text)
                
                # Add Track button
                if st.button(f"📌 Track {ticker}", key=f"track_manual_{ticker}"):
                    from swing_trader.paper_trading.storage import PaperTradeStorage
                    from swing_trader.paper_trading.tracker import PaperTradeTracker
                    
                    paper_storage = PaperTradeStorage()
                    paper_tracker = PaperTradeTracker(paper_storage)
                    
                    # Build signal dict for tracking
                    track_signal = {
                        'ticker': ticker,
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'entry_price': result.get('entry_price', 0),
                        'stop_loss': result.get('stop_loss', 0),
                        'target_1': result.get('target_1', 0),
                        'swing_type': result.get('swing_type', 'A'),
                        'quality_score': result.get('quality_score', 0),
                        'position_size': pos_size,
                        'hold_days_max': result.get('hold_days', (2, 7))[1],
                        'type_reason': result.get('type_reason', '')
                    }
                    
                    trade_id = paper_tracker.add_trade_from_signal(track_signal)
                    if trade_id > 0:
                        st.success(f"✅ {ticker} paper trade'e eklendi!")
                    else:
                        st.warning(f"⚠️ {ticker} zaten takipte")
    
    else:
        rejection = result.get('rejection_reason', 'Does not meet swing criteria')
        
        st.warning(f"""
        ### ❌ {ticker} - NOT SWING READY
        **{company_name}** | {sector} | ${market_cap/1e9:.2f}B
        """)
        
        # Show more metrics including 5-day return
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Strategy", strategy)
        with col2:
            rsi = result.get('rsi', 50)
            rsi_color = "🔴" if rsi > 70 else ("🟢" if rsi < 30 else "⚪")
            st.metric("RSI", f"{rsi:.0f} {rsi_color}")
        with col3:
            five_d = result.get('five_day_return', 0)
            st.metric("5-Day", f"{five_d:+.0f}%")
        with col4:
            float_shares = result.get('float_shares', 0)
            if float_shares and float_shares > 0:
                st.metric("Float", f"{float_shares/1e6:.0f}M")
            else:
                st.metric("Status", "❌ No Signal")
        
        # ── DETAILED REJECTION BREAKDOWN (Turkish) ──
        with st.expander("🔍 Neden Reddedildi? — Detaylı Analiz", expanded=True):
            # Determine which stage failed
            filter_passed = result.get('filter_passed', False)
            trigger_passed = result.get('trigger_passed', False)
            swing_ready = result.get('swing_ready', False)
            
            # Stage 1: Filters
            stage1_icon = "✅" if filter_passed else "❌"
            # Stage 2: Triggers (only reached if filters pass)
            stage2_icon = "✅" if trigger_passed else ("❌" if filter_passed else "⬜")
            # Stage 3: Swing Confirmation (only reached if triggers pass)
            stage3_icon = "✅" if swing_ready else ("❌" if trigger_passed else "⬜")
            
            st.markdown(f"""
**Analiz Aşamaları:**
{stage1_icon} Aşama 1: Filtreler (Market Cap, Volume, ATR%, Float, Earnings)
{stage2_icon} Aşama 2: Sinyal Tetikleyiciler (Volume Patlama, Volatilite, Breakout)
{stage3_icon} Aşama 3: Swing Onayı (5-Gün Momentum, MA20 Üzerinde, Higher Lows)
            """)
            
            st.markdown("---")
            
            # ── STAGE 1: Filters ──
            filter_details = result.get('filter_details', {})
            if filter_details:
                st.markdown("**📊 Aşama 1 — Filtreler:**")
                filters = filter_details.get('filters', {})
                filter_labels = {
                    'market_cap': 'Piyasa Değeri',
                    'avg_volume': 'Ortalama Hacim',
                    'atr_percent': 'Volatilite (ATR%)',
                    'float': 'Float (Halka Açık Pay)',
                    'earnings': 'Kazanç Raporu',
                    'price': 'Fiyat'
                }
                for key, val in filters.items():
                    passed = val.get('passed', False)
                    reason = val.get('reason', '')
                    icon = "✅" if passed else "❌"
                    label = filter_labels.get(key, key)
                    st.write(f"{icon} **{label}**: {reason}")
                
                if not filter_passed:
                    st.error("⛔ Filtrelerden geçemedi — sonraki aşamalar test edilmedi.")
                    # Show explanation for common failures
                    for key, val in filters.items():
                        if not val.get('passed', False):
                            if 'Float' in val.get('reason', ''):
                                st.caption("💡 Float (halka açık pay sayısı) çok yüksek. Düşük float'lu hisseler daha patlayıcı hareket eder.")
                            elif 'Earnings' in val.get('reason', ''):
                                st.caption("💡 Kazanç raporu yakında açıklanacak. Rapor sonrası büyük düşüş riski var, bu yüzden bloklanıyor.")
                            elif 'Market cap' in val.get('reason', ''):
                                st.caption("💡 Piyasa değeri small-cap aralığının ($250M-$2.5B) dışında.")
                            elif 'Volume' in val.get('reason', ''):
                                st.caption("💡 Ortalama işlem hacmi çok düşük — likidite riski.")
                            elif 'ATR' in val.get('reason', ''):
                                st.caption("💡 Volatilite çok düşük — swing trade için yeterli hareket potansiyeli yok.")
            
            # ── STAGE 2: Triggers ──
            trigger_details = result.get('trigger_details', {})
            if trigger_details and filter_passed:
                st.markdown("---")
                st.markdown("**🎯 Aşama 2 — Sinyal Tetikleyiciler:**")
                
                triggers = trigger_details.get('triggers', {})
                trigger_labels = {
                    'volume_surge': 'Hacim Patlaması (Volume Surge)',
                    'atr_percent': 'Volatilite Genişlemesi (ATR%)',
                    'breakout': 'Breakout (Direnç Kırılımı)'
                }
                
                for key, val in triggers.items():
                    passed = val.get('passed', False)
                    reason = val.get('reason', '')
                    icon = "✅" if passed else "❌"
                    label = trigger_labels.get(key, key)
                    st.write(f"{icon} **{label}**: {reason}")
                
                if not trigger_passed:
                    vol_surge = trigger_details.get('volume_surge', 0)
                    st.error(f"⛔ Sinyal tetiklenmedi — hacim ortalamanın altında ({vol_surge:.1f}x). En az 1.5x olmalı.")
                    st.caption("💡 Bu hisse şu an yeterli alım ilgisi görmüyor. Hacim patlaması olmadan girmek riskli — momentum olmadan fiyat hareket etmez.")
            
            # ── STAGE 3: Swing Confirmation ──
            swing_details = result.get('swing_details', {})
            if swing_details and trigger_passed:
                st.markdown("---")
                st.markdown("**✨ Aşama 3 — Swing Onayı:**")
                
                swing_checks = {
                    'five_day_momentum': '5-Günlük Momentum',
                    'above_ma20': '20-Gün Ort. Üzerinde',
                    'higher_lows': 'Yükselen Dipler',
                    'multi_day_volume': 'Çok-Gün Hacim Trendi',
                    'overextension': 'Aşırı Uzama Kontrolü'
                }
                
                for key, label in swing_checks.items():
                    val = swing_details.get(key, {})
                    if isinstance(val, dict):
                        passed = val.get('passed')
                        if passed is None:
                            st.write(f"⬜ **{label}**: Kontrol edilmedi")
                        else:
                            icon = "✅" if passed else "❌"
                            ret = val.get('return', val.get('value', ''))
                            extra = f" ({ret:.1f}%)" if isinstance(ret, (int, float)) else ""
                            st.write(f"{icon} **{label}**{extra}")
                
                if not swing_ready:
                    st.error("⛔ Swing onayı başarısız — trend teyit edilemedi.")
                    five_day = swing_details.get('five_day_momentum', {})
                    ma20 = swing_details.get('above_ma20', {})
                    if isinstance(five_day, dict) and not five_day.get('passed', True):
                        st.caption("💡 Son 5 günde negatif momentum — hisse düşüşte.")
                    if isinstance(ma20, dict) and not ma20.get('passed', True):
                        st.caption("💡 Fiyat 20-günlük ortalamanın altında — hisse henüz toparlanmadı.")
    
    st.markdown("---")

def small_cap_page(components):
    """
    SmallCap Momentum page - COMPLETELY SEPARATE from LargeCap scanning.
    Uses independent SmallCapEngine with different philosophy and rules.
    """
    st.title("🚀 SmallCap Momentum Scanner")
    st.markdown("*High-volatility small caps for short-term swing trades (2-8 days)*")
    
    # Get risk constants from SmallCapRisk for dynamic display
    from swing_trader.small_cap.risk import SmallCapRisk
    risk_config = SmallCapRisk()
    position_pct_min = int(risk_config.POSITION_SIZE_FACTOR * 100 - 5)  # 25%
    position_pct_max = int(risk_config.POSITION_SIZE_FACTOR * 100 + 10)  # 40%
    max_risk_pct = risk_config.MAX_RISK_PER_TRADE * 100  # 0.5%
    max_hold_days = risk_config.MAX_HOLDING_DAYS  # 7 days
    
    # Warning banner - DYNAMIC values
    st.warning(f"""
    ⚠️ **HIGH RISK STRATEGY**
    - Small caps have extreme volatility
    - Position sizes are **{position_pct_min}-{position_pct_max}%** of normal
    - All signals have `volatility_warning = TRUE`
    - Max holding period: **{max_hold_days} days**
    """)
    
    # Session state already initialized globally via init_session_state()
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        portfolio_value = st.number_input("Portfolio Value ($)", value=10000, step=1000, key="sc_portfolio")
    
    with col2:
        min_quality = st.slider("Min Quality Score", 40, 90, 60, key="sc_quality",
                                help="Higher = stricter filter (recommended: 60+)")
    
    with col3:
        top_n = st.number_input("Top N Results", value=5, min_value=1, max_value=15, key="sc_topn")
    
    # Auto-Track Settings
    with st.expander("⚙️ Auto-Track Ayarları", expanded=False):
        at_col1, at_col2 = st.columns(2)
        with at_col1:
            auto_track = st.checkbox(
                "📌 Otomatik Paper Trade Takibi",
                value=st.session_state.auto_track_enabled,
                help="Scan sonucu kaliteli sinyalleri otomatik paper trade'e ekler"
            )
            st.session_state.auto_track_enabled = auto_track
        with at_col2:
            auto_track_quality = st.slider(
                "Min kalite (Auto-Track)", 50, 100, 
                st.session_state.auto_track_min_quality,
                key="sc_auto_quality",
                help="Bu puanın üzerindeki sinyaller otomatik takibe alınır"
            )
            st.session_state.auto_track_min_quality = auto_track_quality
        
        if auto_track:
            st.info(f"✅ Kalite ≥ **{auto_track_quality}** olan sinyaller otomatik paper trade'e eklenecek")
        else:
            st.caption("⏸️ Otomatik takip kapalı — sinyalleri manuel olarak eklemeniz gerekir")
    
    # Scan button
    if st.button("🚀 Scan SmallCaps", type="primary"):
        with st.spinner("Scanning small cap momentum stocks..."):
            try:
                engine = components['small_cap_engine']
                fetcher = components['fetcher']
                
                # Get small cap universe
                tickers = engine.get_small_cap_universe()
                st.info(f"Scanning {len(tickers)} potential small cap stocks...")
                
                # Fetch data for all tickers
                data_dict = {}
                for ticker in tickers:
                    try:
                        df = fetcher.fetch_stock_data(ticker, period='3mo')
                        if df is not None and len(df) >= 20:
                            data_dict[ticker] = df
                    except Exception:
                        continue
                
                # Run scan
                signals = engine.scan_universe(tickers, data_dict, portfolio_value)
                
                # Filter by quality
                signals = [s for s in signals if s.get('quality_score', 0) >= min_quality]
                signals = signals[:top_n]
                
                st.session_state.smallcap_results = signals
                st.session_state.smallcap_stats = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'tickers_scanned': len(tickers),
                    'data_fetched': len(data_dict),
                    'signals_found': len(signals)
                }
                
                # Log scan to history
                st.session_state.scan_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'smallcap',
                    'signals': len(signals)
                })
                
                # ============================================================
                # AUTO-TRACK: Automatically add qualifying signals to paper trades
                # ============================================================
                if st.session_state.auto_track_enabled and signals:
                    auto_quality = st.session_state.auto_track_min_quality
                    paper_storage = PaperTradeStorage()
                    paper_tracker = PaperTradeTracker(paper_storage)
                    
                    auto_tracked = []
                    auto_skipped = []
                    
                    for signal in signals:
                        if signal.get('quality_score', 0) >= auto_quality:
                            trade_id = paper_tracker.add_trade_from_signal(signal)
                            if trade_id > 0:
                                auto_tracked.append(signal['ticker'])
                            else:
                                auto_skipped.append(signal['ticker'])
                    
                    st.session_state.last_auto_tracked = auto_tracked
                    
                    if auto_tracked:
                        st.success(
                            f"📌 **Auto-Track:** {len(auto_tracked)} sinyal paper trade'e eklendi → "
                            f"{', '.join(auto_tracked)}"
                        )
                    if auto_skipped:
                        st.caption(
                            f"⏭️ Zaten takipte: {', '.join(auto_skipped)}"
                        )
                
            except Exception as e:
                st.error(f"Error during scan: {e}")
                logging.exception("SmallCap scan error")
    
    # ============================================================
    # 📖 TECHNICAL GLOSSARY (Always visible, above results)
    # ============================================================
    with st.expander("📖 **Teknik Terimler Sözlüğü** — Terimlerin ne anlama geldiğini öğren", expanded=False):
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("""
#### 📊 Temel Göstergeler

| Terim | Açıklama |
|-------|----------|
| **RSI** | *Relative Strength Index* — 0-100 arası momentum ölçer. **< 30** aşırı satım (ucuz), **> 70** aşırı alım (pahalı). Swing için **40-65** ideal giriş bölgesi. |
| **ATR** | *Average True Range* — Hissenin günlük ortalama hareket aralığı (dolar). Stop-loss hesabında kullanılır. |
| **ATR%** | ATR'nin fiyata oranı. **%5+** yüksek oynaklık demek — daha fazla kâr potansiyeli ama daha fazla risk. |
| **Volume Surge** | Günlük işlem hacminin 20 günlük ortalamaya oranı. **2x+** güçlü ilgi, **4x+** çok güçlü. |
| **MACD** | *Moving Average Convergence Divergence* — Trend yönünü ve momentum gücünü gösterir. Sinyal çizgisini yukarı keserse **alım sinyali**. |
| **MA20** | 20 günlük hareketli ortalama. Fiyat MA20'nin üstündeyse **kısa vadeli trend yukarı**. |
            """)
            
            st.markdown("""
#### 🎯 Risk & Hedefler

| Terim | Açıklama |
|-------|----------|
| **Stop Loss** | Zarar kes seviyesi. Fiyat buraya düşerse **hemen sat**. Genelde 1-1.5 ATR altında, Type'a göre max %8-12. |
| **T1 (Hedef 1)** | İlk kâr alma noktası. Pozisyonun yarısını burada sat. Type'a göre +%18 ile +%30 arası. |
| **T2 (Hedef 2)** | İkinci hedef. Kalan pozisyonu burada sat veya trail stop ile devam et. +%30 ile +%60 arası. |
| **R/R Oranı** | *Risk/Reward Ratio* — Riske ettiğin her $1 için kazanma potansiyeli. **1:3+** iyi, **1:2** minimum. |
| **Trailing Stop** | Fiyat yükseldikçe stop seviyesini de yukarı çeken dinamik zarar kes. Kârı korur. |
            """)
        
        with col_right:
            st.markdown("""
#### 🏢 Hisse Bilgileri

| Terim | Açıklama |
|-------|----------|
| **Float** | Piyasada serbestçe alınıp satılabilen hisse sayısı. **< 20M** sıkı float = daha keskin hareketler. |
| **SI%** | *Short Interest* — Açığa satılmış hisselerin float'a oranı. **> 20%** squeeze (sıkışma) potansiyeli. |
| **Days to Cover** | Açığa satılan hisselerin ortalama hacimle kaç günde kapatılacağı. **> 5** = squeeze riski. |
| **RS** | *Relative Strength* — Hissenin sektörüne göre performansı. **+15+** sektör lideri. |
| **Catalyst** | Fiyatı tetikleyecek olay: kazanç raporu, FDA onayı, anlaşma haberi vb. |
| **Quality** | Toplam kalite skoru (0-100+). Tüm metriklerin birleşimi. **70+** güçlü, **55-70** orta. |
            """)
            
            st.markdown("""
#### 🏷️ Swing Tipleri

| Tip | İsim | Süre | Açıklama |
|-----|------|------|----------|
| 🔥 **S** | Squeeze | 1-4 gün | Short sıkışması. SI ≥ %20, çok riskli ama çok kârlı. |
| ⭐ **C** | Erken Aşama | 3-8 gün | **En iyi R/R.** RSI düşük, hareket yeni başlıyor. Pullback girişi mümkün. |
| 🚀 **B** | Momentum | 2-6 gün | Hisse zaten +%30-70 yükselmiş. Sadece catalyst + yüksek volume ile gir. |
| 🐢 **A** | Devam | 5-14 gün | Trend devamı. En güvenli ama en yavaş. Higher lows yapısı önemli. |
            """)
    
    # Display results
    st.divider()
    
    if st.session_state.smallcap_stats:
        stats = st.session_state.smallcap_stats
        st.caption(f"📅 Last scan: {stats['timestamp']} | Scanned: {stats['tickers_scanned']} | Signals: {stats['signals_found']}")
    
    signals = st.session_state.smallcap_results
    
    if signals and len(signals) > 0:
        st.success(f"🚀 Found {len(signals)} SmallCap Momentum signals!")
        
        # Create DataFrame for display
        display_data = []
        type_emojis = {'S': '🔥', 'C': '⭐', 'B': '🚀', 'A': '🐢'}
        type_labels_tr = {'S': 'Squeeze', 'C': 'Erken', 'B': 'Momentum', 'A': 'Devam'}
        for s in signals:
            swing_type = s.get('swing_type', 'A')
            hold_min = s.get('expected_hold_min', s.get('hold_days_min', 2))
            hold_max = s.get('expected_hold_max', s.get('hold_days_max', 5))
            display_data.append({
                'Ticker': s['ticker'],
                'Tip': f"{type_emojis.get(swing_type, '📊')} {swing_type}",
                'Quality': f"{s['quality_score']:.0f}",
                'Entry': f"${s['entry_price']:.2f}",
                'Stop': f"${s.get('stop_loss', 0):.2f} ({s.get('stop_loss_pct', 0):.0f}%)",
                'T1': f"${s.get('target_1', 0):.2f} (+{s.get('target_1_pct', 0):.0f}%)",
                'T2': f"${s.get('target_2', 0):.2f} (+{s.get('target_2_pct', 0):.0f}%)",
                'Vol Surge': f"{s['volume_surge']:.1f}x",
                'ATR%': f"{s['atr_percent']:.1f}%",
                'Float': f"{s['float_millions']:.0f}M",
                # NEW COLUMNS: Senior Trader v2.1
                'SI%': f"{s.get('short_percent', 0):.1f}%" if s.get('short_percent', 0) > 0 else '-',
                'RS': f"+{s.get('sector_rs_score', 0):.0f}" if s.get('sector_rs_score', 0) > 0 else f"{s.get('sector_rs_score', 0):.0f}",
                'Cat': '🔥' if s.get('total_catalyst_bonus', 0) >= 10 else ('✨' if s.get('total_catalyst_bonus', 0) >= 5 else '-'),
                'Hold': f"{hold_min}-{hold_max}d",
                '⚠️': '🔴' if s.get('volatility_warning') else '🟢'
            })

        
        df_display = pd.DataFrame(display_data)
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # Legend
        with st.expander("📚 Column Explanation"):
            st.markdown("""
            | Column | Meaning |
            |--------|---------|
            | **Tip** | Swing tipi: 🔥 S=Squeeze, ⭐ C=Erken Aşama, 🚀 B=Momentum, 🐢 A=Devam |
            | **Quality** | Momentum quality score (0-135) |
            | **Entry** | Current close price (enter here) |
            | **Stop** | Stop loss — type-specific cap: C=%8, A/B=%10, S=%12 |
            | **T1** | 1. hedef: pozisyonun yarısını sat. Type'a göre +%18 ile +%30 |
            | **T2** | 2. hedef: kalanı sat veya trail. Type'a göre +%30 ile +%60 |
            | **Vol Surge** | Volume vs 20-day avg |
            | **ATR%** | Volatility (ATR/Price) |
            | **Float** | Shares floating (smaller = more explosive) |
            | **SI%** | Short Interest % of Float (>20% = squeeze candidate) |
            | **RS** | Sector Relative Strength (>+15 = sector leader) |
            | **Cat** | Catalyst: 🔥 Strong, ✨ Moderate, - None |
            | **Hold** | Expected holding period |
            | **⚠️** | Volatility warning (always 🔴 for small caps) |
            """)
        
        # ============================================================
        # NARRATIVE ANALYSIS SECTION (Cuma Çevik Style)
        # ============================================================
        st.markdown("---")
        st.markdown("### 📝 Sinyal Analizleri")
        st.caption("Her sinyal için detaylı yorum ve öneri")
        
        for signal in signals:
            ticker = signal['ticker']
            headline = signal.get('narrative_headline', f"{ticker}")
            narrative_text = signal.get('narrative_text', '')
            quality = signal.get('quality_score', 0)
            
            # Quality badge
            if quality >= 75:
                badge = "🔥"
            elif quality >= 60:
                badge = "✅"
            else:
                badge = "⚠️"
            
            with st.expander(f"{badge} **{headline}**", expanded=False):
                if narrative_text:
                    st.markdown(narrative_text)
                else:
                    # Fallback if narrative not generated
                    st.markdown(f"""
**{ticker}** - Type {signal.get('swing_type', 'A')}

📍 **Entry:** ${signal.get('entry_price', 0):.2f}
🛑 **Stop:** ${signal.get('stop_loss', 0):.2f}
🎯 **Target:** ${signal.get('target_1', 0):.2f}

📊 **Metrikler:**
- Volume: {signal.get('volume_surge', 1):.1f}x
- RSI: {signal.get('rsi', 50):.0f}
- 5 Günlük: +{signal.get('five_day_return', 0):.0f}%
                    """)
                
                # Track button inside expander
                if st.button(f"📌 Track {ticker}", key=f"track_exp_{ticker}"):
                    paper_storage = PaperTradeStorage()
                    paper_tracker = PaperTradeTracker(paper_storage)
                    trade_id = paper_tracker.add_trade_from_signal(signal)
                    if trade_id > 0:
                        st.success(f"✅ {ticker} paper trade'e eklendi!")
                    else:
                        st.warning(f"⚠️ {ticker} zaten takipte")
        
        st.markdown("---")
        
        # Risk reminder - DYNAMIC values
        st.error(f"""
        **⚠️ RISK RULES:**
        - Position size: **{position_pct_min}-{position_pct_max}%** of normal
        - Max risk: **{max_risk_pct}%** per trade
        - ALWAYS use stop loss
        - Max hold: **{max_hold_days} days** (exit even if not at target)
        - Accept gap risk as part of strategy
        """)
        
    elif st.session_state.smallcap_stats:
        st.warning("No signals found matching criteria. Try lowering the quality threshold.")
    else:
        st.info("👆 Click 'Scan SmallCaps' to find momentum opportunities")


# ================================================================
# PAPER TRADES PAGE
# ================================================================

def paper_trades_page(components: dict):
    """Paper Trading Tracker page - Track signals without real money."""
    
    st.title("📊 Paper Trading Tracker")
    st.markdown("*Track your SmallCap signals without risking real money*")
    
    # Initialize paper trading components
    storage = PaperTradeStorage()
    tracker = PaperTradeTracker(storage)
    reporter = PaperTradeReporter(storage)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Active Trades", "📜 Closed Trades", "📊 Performance", "🤖 AI Model"
    ])
    
    # ============================================================
    # TAB 1: ACTIVE TRADES
    # ============================================================
    with tab1:
        st.subheader("Active Paper Trades")
        
        # Update button
        if st.button("🔄 Update Prices", key="update_prices"):
            with st.spinner("Fetching latest prices..."):
                # This also confirms pending trades via reporter
                tracker.confirm_pending_trades()
                tracker.update_all_open_trades()
                st.success("Prices updated & pending trades confirmed!")
                st.rerun()
        
        # Get open trades summary
        open_summary = reporter.get_open_trades_summary()
        
        if open_summary['count'] > 0:
            # ── SUMMARY METRICS (with percentage-based total) ──
            total_unrealized = open_summary['total_unrealized_pnl']
            trades_list = open_summary['trades']
            
            # Calculate weighted average P/L percentage
            pnl_pcts = [t.get('unrealized_pnl_pct', 0) for t in trades_list]
            avg_pnl_pct = sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0
            
            # Individual counts
            winning = sum(1 for t in trades_list if t.get('unrealized_pnl', 0) > 0)
            losing = sum(1 for t in trades_list if t.get('unrealized_pnl', 0) < 0)
            breakeven = sum(1 for t in trades_list if t.get('unrealized_pnl', 0) == 0)
            
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Açık Pozisyon", open_summary['count'])
            with m2:
                st.metric(
                    "Ort. P/L",
                    f"{avg_pnl_pct:+.2f}%",
                    delta="kârda" if avg_pnl_pct >= 0 else "zararda",
                    delta_color="normal" if avg_pnl_pct >= 0 else "inverse"
                )
            with m3:
                total_pnl_pct_sum = sum(pnl_pcts)
                st.metric(
                    "Toplam P/L",
                    f"{total_pnl_pct_sum:+.2f}%",
                    delta=f"${total_unrealized:+.2f}",
                    delta_color="normal" if total_unrealized >= 0 else "inverse"
                )
            with m4:
                st.metric("W / L / B", f"{winning} / {losing} / {breakeven}")
            
            st.markdown("---")
            
            # ── ACTIVE TRADE CARDS ──
            for trade in trades_list:
                ticker = trade['ticker']
                trade_id = trade['id']
                entry = trade['entry_price']
                current = trade['current_price']
                stop = trade['stop_loss']
                target = trade['target']
                trailing = trade.get('trailing_stop', stop)
                initial_stop = trade.get('initial_stop', stop)
                trail_moved = trailing > initial_stop
                pnl = trade.get('unrealized_pnl', 0)
                pnl_pct = trade.get('unrealized_pnl_pct', 0)
                days_held = trade.get('days_held', 0)
                max_hold = trade.get('max_hold_days', 7)
                swing_type = trade.get('swing_type', 'A')
                
                # Color based on P/L
                if pnl_pct > 5:
                    pnl_emoji = "🟢"
                elif pnl_pct > 0:
                    pnl_emoji = "🟩"
                elif pnl_pct == 0:
                    pnl_emoji = "⚪"
                elif pnl_pct > -3:
                    pnl_emoji = "🟧"
                else:
                    pnl_emoji = "🔴"
                
                # Hold days warning
                hold_pct = days_held / max_hold if max_hold > 0 else 0
                if hold_pct >= 0.85:
                    hold_emoji = "⏰"
                elif hold_pct >= 0.6:
                    hold_emoji = "⏱️"
                else:
                    hold_emoji = "📅"
                
                # Type labels
                type_labels = {'A': '🔥Momentum', 'B': '💎Breakout', 'C': '🌱Early', 'S': '🩳Squeeze'}
                type_label = type_labels.get(swing_type, swing_type)
                
                with st.expander(
                    f"{pnl_emoji} **{ticker}** | {pnl_pct:+.1f}% | "
                    f"${current:.2f} | {type_label} | "
                    f"{hold_emoji} {days_held}/{max_hold}g",
                    expanded=True
                ):
                    # ── ROW 1: Key metrics with st.metric ──
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    
                    with mc1:
                        price_change = current - entry
                        st.metric(
                            "Fiyat",
                            f"${current:.2f}",
                            delta=f"{pnl_pct:+.1f}%",
                            delta_color="normal" if pnl_pct >= 0 else "inverse"
                        )
                    
                    with mc2:
                        dist_to_target = ((target / current) - 1) * 100 if current > 0 else 0
                        st.metric(
                            "🎯 Hedef",
                            f"${target:.2f}",
                            delta=f"{dist_to_target:+.1f}% kaldı"
                        )
                    
                    with mc3:
                        active_stop = trailing if trail_moved else stop
                        dist_to_stop = ((active_stop / current) - 1) * 100 if current > 0 else 0
                        stop_label = "🔒 Trailing" if trail_moved else "🛑 Stop"
                        st.metric(
                            stop_label,
                            f"${active_stop:.2f}",
                            delta=f"{dist_to_stop:+.1f}%",
                            delta_color="inverse" if dist_to_stop < -5 else "normal"
                        )
                    
                    with mc4:
                        st.metric("⏱️ Hold", f"{days_held} / {max_hold} gün")
                        st.progress(min(hold_pct, 1.0))
                    
                    # ── Risk/Reward mini bar ──
                    risk_from_entry = ((entry - active_stop) / entry) * 100 if entry > 0 else 0
                    reward_from_entry = ((target - entry) / entry) * 100 if entry > 0 else 0
                    rr = reward_from_entry / risk_from_entry if risk_from_entry > 0 else 0
                    
                    st.caption(
                        f"📍 Giriş: ${entry:.2f} | "
                        f"📅 {trade.get('entry_date', '-')} | "
                        f"R/R: 1:{rr:.1f} | "
                        f"Risk: {risk_from_entry:.1f}% → Reward: {reward_from_entry:.1f}%"
                    )
                    
                    st.markdown("---")
                    
                    # ── ROW 2: Stop + Target + Actions ──
                    stop_col, target_col, action_col = st.columns(3)
                    
                    with stop_col:
                        st.markdown("**🛑 Stop Loss Yönetimi**")
                        
                        # Show current stop info
                        if trail_moved:
                            st.markdown(
                                f"Orijinal: ~~${initial_stop:.2f}~~ → "
                                f"Trailing: **${trailing:.2f}** 🔒",
                                unsafe_allow_html=True
                            )
                        
                        # Dynamic stop loss input
                        current_stop = trailing if trail_moved else stop
                        min_stop = round(current * 0.50, 2)
                        max_stop = round(current * 0.995, 2)
                        
                        new_stop = st.number_input(
                            f"Stop (${min_stop:.2f} - ${max_stop:.2f})",
                            min_value=min_stop,
                            max_value=max_stop,
                            value=round(current_stop, 2),
                            step=0.05,
                            key=f"stop_input_{trade_id}",
                            format="%.2f"
                        )
                        
                        if abs(new_stop - current_stop) > 0.01:
                            stop_pct_from_current = ((new_stop / current) - 1) * 100
                            if new_stop > current_stop:
                                st.info(f"⬆️ ${current_stop:.2f} → ${new_stop:.2f} ({stop_pct_from_current:+.1f}%)")
                            else:
                                st.warning(f"⬇️ ${current_stop:.2f} → ${new_stop:.2f} ({stop_pct_from_current:+.1f}%)")
                            
                            if st.button("✅ Stop Güncelle", key=f"update_stop_{trade_id}"):
                                storage.update_trade(trade_id, {
                                    'stop_loss': new_stop,
                                    'trailing_stop': new_stop
                                })
                                st.success(f"✅ {ticker} stop → ${new_stop:.2f}")
                                st.rerun()
                    
                    with target_col:
                        st.markdown("**🎯 Hedef Yönetimi**")
                        
                        # Dynamic target input
                        min_target = round(current * 1.005, 2)  # At least 0.5% above current
                        max_target = round(current * 3.0, 2)    # Max 3x current price
                        
                        new_target = st.number_input(
                            f"Target (${min_target:.2f} - ${max_target:.2f})",
                            min_value=min_target,
                            max_value=max_target,
                            value=round(target, 2),
                            step=0.10,
                            key=f"target_input_{trade_id}",
                            format="%.2f"
                        )
                        
                        if abs(new_target - target) > 0.01:
                            target_pct_from_entry = ((new_target / entry) - 1) * 100
                            old_pct = ((target / entry) - 1) * 100
                            if new_target > target:
                                st.info(f"⬆️ Hedef yükseltiliyor: ${target:.2f} → ${new_target:.2f} (+{target_pct_from_entry:.1f}%)")
                            else:
                                st.warning(f"⬇️ Hedef düşürülüyor: ${target:.2f} → ${new_target:.2f} (+{target_pct_from_entry:.1f}%)")
                            
                            if st.button("✅ Hedef Güncelle", key=f"update_target_{trade_id}"):
                                storage.update_trade(trade_id, {'target': new_target})
                                st.success(f"✅ {ticker} hedef → ${new_target:.2f}")
                                st.rerun()
                    
                    with action_col:
                        st.markdown("**⚙️ Aksiyonlar**")
                        
                        # Hold days extension
                        hold_col1, hold_col2 = st.columns(2)
                        with hold_col1:
                            if st.button("⏱️ +3 Gün", key=f"extend_{trade_id}"):
                                new_max = max_hold + 3
                                storage.update_trade(trade_id, {'max_hold_days': new_max})
                                st.success(f"⏱️ {ticker} → {new_max} gün")
                                st.rerun()
                        with hold_col2:
                            if max_hold > days_held + 1:
                                if st.button("⏱️ -3 Gün", key=f"shorten_{trade_id}"):
                                    new_max = max(days_held + 1, max_hold - 3)
                                    storage.update_trade(trade_id, {'max_hold_days': new_max})
                                    st.info(f"⏱️ {ticker} → {new_max} gün")
                                    st.rerun()
                        
                        st.markdown("")  # spacer
                        
                        # Close and delete
                        if st.button("🔒 Close Trade", key=f"close_{trade_id}", type="primary"):
                            tracker.manual_close_trade(trade_id)
                            st.success(f"Closed {ticker}")
                            st.rerun()
                        
                        if st.button("🗑️ Sil", key=f"delete_{trade_id}"):
                            storage.delete_trade(trade_id)
                            st.warning(f"Deleted {ticker}")
                            st.rerun()
        else:
            st.info("No active paper trades. Add trades from SmallCap Momentum page!")
        
        # PENDING trades section — Modern Kart Arayüzü
        if open_summary.get('pending_count', 0) > 0:
            st.markdown("---")

            n_pending = open_summary['pending_count']
            st.markdown(f"## ⏳ Bekleyen Sinyaller — {n_pending} onay bekleniyor")

            # Kural kutusu
            with st.expander("ℹ️ Onay Kuralları", expanded=False):
                st.markdown("""
| Durum | Kural |
|---|---|
| ✅ **Onaylandı** | Ertesi gün Open fiyatı sinyal fiyatından **≤ +5%** veya **≥ -3%** arasında |
| ❌ **Gap-up Red** | Open **> +5%** → momentum bitti, girmiyoruz |
| ❌ **Gap-down Red** | Open **< -3%** → kötü haber, girmiyoruz |
| ⏳ **Bekleniyor** | Henüz ertesi gün verisi yok (piyasa açılmadı) |

*Sayfa her yenilendiğinde de otomatik kontrol edilir — "🔄 Update Prices" butonu ayrıca açık trade fiyatlarını da günceller.*
                """)

            # Her pending sinyal için kart
            for trade in open_summary['pending_trades']:
                trade_id   = trade['id']
                ticker     = trade['ticker']
                stype      = trade.get('swing_type', '?')
                sig_price  = trade.get('signal_price') or trade['entry_price']
                stop       = trade.get('stop_loss', 0)
                target     = trade.get('target', 0)
                added_date = trade.get('entry_date', '?')
                quality    = trade.get('quality_score', 0)

                risk_pct   = abs(sig_price - stop) / sig_price * 100 if sig_price else 0
                reward_pct = abs(target - sig_price) / sig_price * 100 if sig_price else 0
                rr         = reward_pct / risk_pct if risk_pct > 0 else 0

                # Kalite badge (0-100+ skala, /10 değil)
                if quality >= 75:
                    q_badge = "🔥"
                elif quality >= 55:
                    q_badge = "✅"
                elif quality > 0:
                    q_badge = "⚠️"
                else:
                    q_badge = "—"  # Skor kaydedilmemiş

                with st.container(border=True):
                    # Başlık satırı
                    h_col1, h_col2 = st.columns([3, 1])
                    with h_col1:
                        q_str = f"{q_badge} {quality:.0f}/100" if quality > 0 else "Skor yok"
                        st.markdown(f"### ⏳ {ticker} — Tip {stype} | Kalite: {q_str}")
                        st.caption(f"Sinyal tarihi: {added_date} | Sinyal fiyatı: ${sig_price:.2f}")
                    with h_col2:
                        st.markdown("")
                        st.markdown(f"**R/R:** 1:{rr:.1f}")

                    # Fiyat metrikleri
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("📍 Entry (Sinyal)", f"${sig_price:.2f}")
                    with m2:
                        st.metric("🛑 Stop Loss", f"${stop:.2f}",
                                  delta=f"-{risk_pct:.1f}%", delta_color="inverse")
                    with m3:
                        st.metric("🎯 Target", f"${target:.2f}",
                                  delta=f"+{reward_pct:.1f}%")
                    with m4:
                        st.metric("⚡ Gap Limiti", "+5% / -3%")

                    # Eylem butonları
                    b1, b2, b3 = st.columns([2, 2, 4])
                    with b1:
                        if st.button(f"✅ Şimdi Onayla", key=f"confirm_now_{trade_id}",
                                     type="primary"):
                            with st.spinner(f"{ticker} onaylanıyor..."):
                                results = tracker.confirm_pending_trades()
                            matched = next((r for r in results if r.get('id') == trade_id), None)
                            if matched:
                                if matched.get('confirm_status') == 'confirmed':
                                    gp = matched.get('gap_pct', 0)
                                    ep = matched.get('entry_price', 0)
                                    st.success(f"✅ {ticker} onaylandı! Open: ${ep:.2f} (gap {gp:+.1f}%)")
                                elif matched.get('confirm_status') == 'rejected':
                                    st.error(f"❌ {ticker} reddedildi: {matched.get('reject_reason', 'Gap filtresi')}")
                                elif matched.get('confirm_status') == 'waiting':
                                    st.info(f"⏳ {ticker} için henüz ertesi gün verisi yok.")
                            else:
                                st.warning("Sonuç alınamadı, Update Prices'ı dene.")
                            st.rerun()
                    with b2:
                        if st.button(f"❌ İptal Et", key=f"reject_manual_{trade_id}"):
                            storage.close_trade(
                                trade_id, sig_price, added_date,
                                'REJECTED', 'Manuel iptal edildi'
                            )
                            st.warning(f"{ticker} iptal edildi.")
                            st.rerun()
                    with b3:
                        st.caption("💡 'Şimdi Onayla' API'dan ertesi gün Open fiyatını çekip gap filtresini uygular.")

        # Confirm result sonuçları (Update Prices sonrası)
        confirm_results = open_summary.get('confirm_results', [])
        if confirm_results:
            st.markdown("---")
            st.markdown("### 📋 Son Onay Sonuçları")
            for cr in confirm_results:
                if cr.get('confirm_status') == 'confirmed':
                    st.success(
                        f"✅ **{cr['ticker']}** onaylandı — "
                        f"Open ${cr.get('entry_price', 0):.2f} "
                        f"(gap {cr.get('gap_pct', 0):+.1f}%)"
                    )
                elif cr.get('confirm_status') == 'rejected':
                    st.error(
                        f"❌ **{cr['ticker']}** reddedildi — "
                        f"{cr.get('reject_reason', 'Gap filtresi')}"
                    )
                elif cr.get('confirm_status') == 'waiting':
                    st.info(f"⏳ **{cr['ticker']}** bekleniyor — ertesi gün verisi henüz yok")



        st.markdown("---")


        
        # Manual add trade form
        with st.expander("➕ Manually Add Trade"):
            with st.form("add_trade_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    ticker = st.text_input("Ticker", placeholder="AAPL")
                    entry_price = st.number_input("Entry Price ($)", min_value=0.01, value=10.0, format="%.2f")
                    stop_loss = st.number_input("Stop Loss ($)", min_value=0.01, value=9.0, format="%.2f")
                
                with col2:
                    entry_date = st.date_input("Entry Date")
                    target = st.number_input("Target ($)", min_value=0.01, value=13.0, format="%.2f")
                    swing_type = st.selectbox("Swing Type", ["A", "B", "C", "S"])
                
                with col3:
                    investment_amount = st.number_input(
                        "Yatırım Tutarı ($)", 
                        min_value=100.0, max_value=100000.0, 
                        value=1000.0, step=100.0,
                        help="Bu hisseye ne kadarlık yatırım yapmak istiyorsun?"
                    )
                    max_hold_days = st.number_input("Max Hold (gün)", min_value=1, max_value=30, value=7)
                    
                    # Preview shares
                    shares = int(investment_amount / entry_price) if entry_price > 0 else 0
                    st.caption(f"📊 {shares} hisse × ${entry_price:.2f} = ${shares * entry_price:.2f}")
                
                if st.form_submit_button("➕ Trade Ekle", type="primary"):
                    calc_shares = int(investment_amount / entry_price) if entry_price > 0 else 100
                    trade = {
                        'ticker': ticker.upper(),
                        'entry_date': entry_date.strftime('%Y-%m-%d'),
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'swing_type': swing_type,
                        'quality_score': 0,
                        'position_size': calc_shares,
                        'max_hold_days': max_hold_days
                    }
                    trade_id = storage.add_trade(trade)
                    if trade_id > 0:
                        st.success(f"✅ {ticker.upper()} eklendi! ({calc_shares} hisse × ${entry_price:.2f} = ${calc_shares * entry_price:.2f})")
                        st.rerun()
                    else:
                        st.error("⚠️ Trade eklenemedi")
    
    # ============================================================
    # TAB 2: CLOSED TRADES
    # ============================================================
    with tab2:
        st.subheader("Closed Trade History")
        
        closed_trades = storage.get_closed_trades(limit=50)
        
        if closed_trades:
            # Build display data
            display_data = []
            for t in closed_trades:
                pnl_pct = t.get('realized_pnl_pct', 0) or 0
                display_data.append({
                    'Ticker': t['ticker'],
                    'Type': t.get('swing_type', '-'),
                    'Entry': f"${t['entry_price']:.2f}",
                    'Exit': f"${t.get('exit_price', 0):.2f}",
                    'P/L %': f"{pnl_pct:+.1f}%",
                    'P/L $': f"${t.get('realized_pnl', 0):+.2f}",
                    'Status': t['status'],
                    'Entry Date': t['entry_date'],
                    'Exit Date': t.get('exit_date', '-')
                })
            
            df = pd.DataFrame(display_data)
            st.dataframe(df, use_container_width=True, height=400)
            
            # ── Delete Section ──
            st.markdown("---")
            st.markdown("### 🗑️ Trade Silme")
            
            # Build options for multiselect
            trade_labels = [
                f"{t['ticker']} | {t['entry_date']} → {t.get('exit_date', '-')} | {t['status']} | ${(t.get('realized_pnl', 0) or 0):+.2f}"
                for t in closed_trades
            ]
            trade_id_map = {label: t['id'] for label, t in zip(trade_labels, closed_trades)}
            
            selected_trades = st.multiselect(
                "Silinecek trade'leri seç (birden fazla seçebilirsin):",
                options=trade_labels,
                key="delete_closed_multi"
            )
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            
            with col_btn1:
                if selected_trades:
                    if st.button(f"🗑️ Seçilenleri Sil ({len(selected_trades)})", key="del_selected", type="primary"):
                        for label in selected_trades:
                            storage.delete_trade(trade_id_map[label])
                        st.success(f"✅ {len(selected_trades)} trade silindi!")
                        st.rerun()
            
            with col_btn2:
                if st.button("💣 Tüm Geçmişi Sil", key="del_all_closed"):
                    st.session_state['confirm_delete_all'] = True
            
            with col_btn3:
                if st.session_state.get('confirm_delete_all'):
                    st.warning(f"⚠️ {len(closed_trades)} trade silinecek. Bu işlem geri alınamaz!")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✅ Evet, hepsini sil", key="confirm_yes", type="primary"):
                            for t in closed_trades:
                                storage.delete_trade(t['id'])
                            st.session_state['confirm_delete_all'] = False
                            st.success(f"✅ {len(closed_trades)} trade silindi!")
                            st.rerun()
                    with c2:
                        if st.button("❌ İptal", key="confirm_no"):
                            st.session_state['confirm_delete_all'] = False
                            st.rerun()
            
            # Color legend
            st.markdown("---")
            st.markdown("""
            **Exit Types:**
            - 🔴 **STOPPED**: Hit stop loss
            - 🔒 **TRAILED**: Trailing stop triggered (profit protected)
            - 🎯 **TARGET**: Hit target price  
            - ⏰ **TIMEOUT**: Max hold days exceeded
            - ✋ **MANUAL**: Manually closed
            - ❌ **REJECTED**: Gap filter rejected (not entered)
            """)
        else:
            st.info("No closed trades yet.")
    
    # ============================================================
    # TAB 3: PERFORMANCE
    # ============================================================
    with tab3:
        st.subheader("📊 Performance Dashboard")
        
        summary = reporter.get_performance_summary()
        
        if summary['closed_trades'] > 0:
            # ── TOP METRICS ROW ──
            m1, m2, m3, m4, m5 = st.columns(5)
            
            with m1:
                st.metric("Toplam Trade", summary['closed_trades'])
            
            with m2:
                wr = summary['win_rate']
                st.metric(
                    "Win Rate", f"{wr:.1f}%",
                    delta="iyi" if wr >= 50 else "düşük",
                    delta_color="normal" if wr >= 50 else "inverse"
                )
            
            with m3:
                pf = summary['profit_factor']
                pf_str = f"{pf:.2f}" if pf < 100 else "∞"
                st.metric(
                    "Profit Factor", pf_str,
                    delta="iyi" if pf >= 1.5 else "düşük",
                    delta_color="normal" if pf >= 1.5 else "inverse"
                )
            
            with m4:
                # Average P/L % per trade
                closed_trades_data = storage.get_closed_trades(limit=1000)
                pnl_pcts = [t.get('realized_pnl_pct', 0) or 0 for t in closed_trades_data]
                avg_pnl_pct = sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0
                st.metric(
                    "Ort. P/L %", f"{avg_pnl_pct:+.2f}%",
                    delta=f"{avg_pnl_pct:+.2f}%",
                    delta_color="normal"
                )
            
            with m5:
                st.metric("Ort. Hold", f"{summary['avg_hold_days']:.1f} gün")
            
            st.markdown("---")
            
            # ── WIN / LOSS VISUAL SUMMARY ──
            wins_count = summary['wins']
            losses_count = summary['losses']
            be_count = summary.get('breakeven', 0)
            total = wins_count + losses_count + be_count
            
            col_wl1, col_wl2 = st.columns([2, 3])
            
            with col_wl1:
                st.markdown("### 📊 Win / Loss Dağılımı")
                
                if total > 0:
                    win_pct = wins_count / total
                    loss_pct = losses_count / total
                    
                    st.markdown(
                        f"🟢 **{wins_count} Win** ({win_pct:.0%}) | "
                        f"🔴 **{losses_count} Loss** ({loss_pct:.0%}) | "
                        f"⚪ **{be_count} B/E**"
                    )
                    st.progress(win_pct)
                
                st.markdown("")  # spacer
                
                # Avg win vs avg loss (% based)
                win_pcts = [p for p in pnl_pcts if p > 0]
                loss_pcts = [p for p in pnl_pcts if p < 0]
                avg_win_pct = sum(win_pcts) / len(win_pcts) if win_pcts else 0
                avg_loss_pct = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0
                
                rwc1, rwc2 = st.columns(2)
                with rwc1:
                    st.metric("Ort. Win %", f"+{avg_win_pct:.2f}%")
                    st.metric("Ort. Win $", f"${summary['avg_win']:.2f}")
                with rwc2:
                    st.metric("Ort. Loss %", f"{avg_loss_pct:.2f}%")
                    st.metric("Ort. Loss $", f"${summary['avg_loss']:.2f}")
                
                # Risk/Reward ratio
                if abs(avg_loss_pct) > 0:
                    rr_ratio = abs(avg_win_pct / avg_loss_pct)
                    rr_color = "🟢" if rr_ratio >= 1.5 else "🟡" if rr_ratio >= 1.0 else "🔴"
                    st.markdown(f"{rr_color} **Risk/Reward:** 1:{rr_ratio:.2f}")
                
                # Expectancy
                if total > 0:
                    expectancy_pct = avg_pnl_pct
                    exp_emoji = "🟢" if expectancy_pct > 0 else "🔴"
                    st.markdown(f"{exp_emoji} **Trade Başına Beklenti:** {expectancy_pct:+.2f}% (${summary['total_pnl']/total:+.2f})")
            
            with col_wl2:
                st.markdown("### 🏆 Best / Worst Trades")
                
                if summary['best_trade']:
                    bt = summary['best_trade']
                    st.success(
                        f"🏆 **En İyi:** {bt['ticker']} | "
                        f"{bt['pnl_pct']:+.1f}% | ${bt['pnl']:+.2f} | {bt.get('date', '-')}"
                    )
                if summary['worst_trade']:
                    wt = summary['worst_trade']
                    st.error(
                        f"📉 **En Kötü:** {wt['ticker']} | "
                        f"{wt['pnl_pct']:+.1f}% | ${wt['pnl']:+.2f} | {wt.get('date', '-')}"
                    )
                
                st.markdown("---")
                
                # System evaluation
                st.markdown("### 💡 Sistem Değerlendirmesi")
                wr = summary['win_rate']
                pf = summary['profit_factor']
                
                # Score system
                score = 0
                feedback_items = []
                
                if wr >= 55:
                    score += 3
                    feedback_items.append("✅ Win rate güçlü")
                elif wr >= 45:
                    score += 2
                    feedback_items.append("⚠️ Win rate orta")
                else:
                    score += 0
                    feedback_items.append("🛑 Win rate düşük")
                
                if pf >= 2.0:
                    score += 3
                    feedback_items.append("✅ Profit factor mükemmel")
                elif pf >= 1.5:
                    score += 2
                    feedback_items.append("✅ Profit factor iyi")
                elif pf >= 1.0:
                    score += 1
                    feedback_items.append("⚠️ Profit factor marginal")
                else:
                    feedback_items.append("🛑 Profit factor < 1 (zarar)")
                
                if avg_pnl_pct > 2:
                    score += 2
                    feedback_items.append("✅ Ort. kâr/trade yüksek")
                elif avg_pnl_pct > 0:
                    score += 1
                    feedback_items.append("⚠️ Ort. kâr/trade düşük")
                else:
                    feedback_items.append("🛑 Ort. trade zararda")
                
                for item in feedback_items:
                    st.markdown(f"- {item}")
                
                if score >= 6:
                    st.success("🎯 **Sonuç:** Sistem kârlı çalışıyor. Devam!")
                elif score >= 3:
                    st.warning("⚠️ **Sonuç:** İyileştirme gerekiyor. Type seçim ve stop stratejisini gözden geçir.")
                else:
                    st.error("🛑 **Sonuç:** Strateji revizyonu gerekiyor.")
            
            st.markdown("---")
            
            # ── EQUITY CURVE ──
            equity_data = summary.get('equity_curve', [])
            if len(equity_data) >= 2:
                st.markdown("### 📈 Equity Curve")
                
                eq_df = pd.DataFrame(equity_data)
                eq_df['label'] = eq_df['date'] + ' | ' + eq_df['ticker']
                
                # Show both $ and % cumulative
                eq_tab1, eq_tab2 = st.tabs(["💰 Kümülatif P/L ($)", "📊 Kümülatif P/L (%)"])
                
                with eq_tab1:
                    chart_df = pd.DataFrame({
                        'Trade': eq_df['label'],
                        'Kümülatif P/L ($)': eq_df['cumulative_pnl']
                    }).set_index('Trade')
                    st.line_chart(chart_df, height=300)
                
                with eq_tab2:
                    # Calculate cumulative % from individual trade %s
                    cum_pct = []
                    running = 0
                    for _, row in eq_df.iterrows():
                        pnl_val = row.get('pnl', 0)
                        entry_val = row.get('entry_price', 1)
                        trade_pct = (pnl_val / entry_val * 100) if entry_val > 0 else 0
                        running += trade_pct
                        cum_pct.append(running)
                    
                    chart_pct_df = pd.DataFrame({
                        'Trade': eq_df['label'],
                        'Kümülatif P/L (%)': cum_pct
                    }).set_index('Trade')
                    st.line_chart(chart_pct_df, height=300)
                
                # Trade-by-trade detail
                with st.expander("📋 Trade Detayları"):
                    detail_df = pd.DataFrame({
                        'Tarih': eq_df['date'],
                        'Ticker': eq_df['ticker'],
                        'P/L $': eq_df['pnl'].apply(lambda x: f"${x:+.2f}"),
                        'P/L %': [f"{p:+.2f}%" for p in ([0] * len(eq_df) if not cum_pct else [cum_pct[0]] + [cum_pct[i] - cum_pct[i-1] for i in range(1, len(cum_pct))])],
                        'Kümülatif $': eq_df['cumulative_pnl'].apply(lambda x: f"${x:+.2f}"),
                        'Kümülatif %': [f"{p:+.2f}%" for p in cum_pct]
                    })
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
            
            # ── PERFORMANCE BY EXIT TYPE ──
            col_exit, col_type = st.columns(2)
            
            with col_exit:
                st.markdown("### 🚪 Exit Type Bazında")
                exit_data = []
                exit_emojis = {
                    'STOPPED': '🔴', 'TRAILED': '🔒', 'TARGET': '🎯',
                    'TIMEOUT': '⏰', 'MANUAL': '✋', 'REJECTED': '❌'
                }
                for status, data in summary['by_exit_type'].items():
                    emoji = exit_emojis.get(status, '📊')
                    exit_data.append({
                        'Tür': f"{emoji} {status}",
                        'Adet': data['count'],
                        'Ort. %': f"{data['avg_pnl_pct']:+.1f}%",
                        'Toplam $': f"${data['total_pnl']:+.2f}"
                    })
                if exit_data:
                    st.dataframe(pd.DataFrame(exit_data), use_container_width=True, hide_index=True)
            
            with col_type:
                st.markdown("### 🏷️ Swing Type Bazında")
                type_data = []
                type_descs = {
                    'A': '🔥 Momentum', 'B': '💎 Breakout', 
                    'C': '🌱 Early', 'S': '🩳 Squeeze'
                }
                for swing_type, data in summary['by_swing_type'].items():
                    desc = type_descs.get(swing_type, f'📊 Type {swing_type}')
                    type_data.append({
                        'Tür': desc,
                        'Adet': data['count'],
                        'Win Rate': f"{data['win_rate']:.0f}%",
                        'Ort. %': f"{data['avg_pnl_pct']:+.1f}%",
                        'Toplam $': f"${data['total_pnl']:+.2f}"
                    })
                if type_data:
                    st.dataframe(pd.DataFrame(type_data), use_container_width=True, hide_index=True)
        else:
            st.info("📊 Henüz kapatılmış trade yok. İlk trade'leri kapattıktan sonra performans istatistikleri burada görünecek.")

        # ── AI Haftalık Rapor Bölümü ──────────────────────────────
        st.markdown("---")
        st.markdown("## 🤖 AI Haftalık Performans Analizi")

        ai_col1, ai_col2 = st.columns([2, 1])
        with ai_col1:
            st.markdown(
                "Geçmiş trade verilerini analiz ederek strateji öngörüleri ve "
                "iyileştirme önerileri sunar. **Hesaplamalar deterministik sistemde yapılır**, "
                "LLM sadece sonuçları yorumlar."
            )
        with ai_col2:
            import os
            has_llm = bool(
                os.getenv("OPENAI_API_KEY", "").startswith("sk-") or
                os.getenv("GEMINI_API_KEY", "") not in ("", "your_gemini_api_key_here")
            )
            if has_llm:
                st.success("🟢 LLM bağlı — AI analiz aktif")
            else:
                st.warning("🟡 API key yok — istatistik rapor modu")

        # Session state ile raporu tut (sayfa yenilenmesinde kaybolmasın)
        if "weekly_report_result" not in st.session_state:
            st.session_state.weekly_report_result = None

        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
        with btn_col1:
            generate_clicked = st.button(
                "📝 Rapor Oluştur", type="primary", key="gen_weekly_report"
            )
        with btn_col2:
            refresh_clicked = st.button(
                "🔄 Yenile", key="refresh_weekly_report",
                help="Önbelleği temizle ve yeniden oluştur"
            )
        with btn_col3:
            st.caption("⏱ Günde bir kez oluşturulur, önbellekten hızlıca yüklenir")

        if generate_clicked or refresh_clicked:
            with st.spinner("Analiz yapılıyor... ⏳"):
                try:
                    from swing_trader.genai.reporter import WeeklyReporter
                    reporter = WeeklyReporter(storage, days=7)
                    if refresh_clicked:
                        reporter.clear_cache()
                    result = reporter.generate(force_refresh=refresh_clicked)
                    st.session_state.weekly_report_result = result
                except Exception as e:
                    st.session_state.weekly_report_result = {
                        "success": False, "error": str(e), "context": {}
                    }

        # Raporu göster
        result = st.session_state.weekly_report_result
        if result:
            if result.get("success") and result.get("report"):
                # Meta bilgi
                meta_parts = []
                if result.get("from_cache"):
                    meta_parts.append("📦 Önbellekten")
                if result.get("llm_available"):
                    meta_parts.append("🤖 GPT/Gemini analizi")
                else:
                    meta_parts.append("📊 İstatistik raporu (API key ekleyince AI analiz gelir)")
                if result.get("generated_at"):
                    meta_parts.append(f"📅 {result['generated_at']}")
                st.caption(" | ".join(meta_parts))

                # Raporun kendisi
                st.markdown(result["report"])

                # Ham istatistikleri gizlenebilir bölümde göster
                ctx = result.get("context", {})
                with st.expander("📊 Ham İstatistikler (Deterministik Katman)"):
                    weekly_s = ctx.get("weekly_summary", {})
                    all_s    = ctx.get("all_time_summary", {})
                    hc1, hc2 = st.columns(2)
                    with hc1:
                        st.markdown("**Bu Dönem**")
                        st.json(weekly_s)
                    with hc2:
                        st.markdown("**Tüm Zamanlar**")
                        st.json(all_s)
            else:
                st.error(f"❌ Rapor oluşturulamadı: {result.get('error', 'Bilinmeyen hata')}")
        else:
            st.info("👆 Rapor oluşturmak için butona bas")

        # ── Strateji Soru-Cevap (RAG-lite) ───────────────────────
        st.markdown("---")
        st.markdown("## 💬 Strateji Danışmanı")
        st.markdown(
            "Trade geçmişine dayanarak strateji sorularını cevaplar. "
            "**'Neden kaybettik?', 'Hangi tip daha iyi?'** gibi sorular sor."
        )

        with st.expander("💬 Soruyu Sor", expanded=False):
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            chat_q = st.text_input(
                "Sorunuzu yazın:",
                placeholder="Örnek: Bu hafta neden kayıplarımız arttı?",
                key="strategy_chat_input"
            )
            ask_btn = st.button("🤔 Sor", key="strategy_chat_btn", type="primary")

            if ask_btn and chat_q.strip():
                with st.spinner("Analiz yapılıyor..."):
                    try:
                        from swing_trader.genai.strategy_chat import StrategyChat
                        chat_engine = StrategyChat(storage)
                        chat_result = chat_engine.ask(chat_q)
                        st.session_state.chat_history.append({
                            "question": chat_q,
                            "answer": chat_result.get("answer", "Cevap alınamadı"),
                            "llm": chat_result.get("llm_available", False),
                        })
                    except Exception as e:
                        st.session_state.chat_history.append({
                            "question": chat_q,
                            "answer": f"Hata: {e}",
                            "llm": False,
                        })

            if st.button("🗑️ Sohbeti Temizle", key="clear_chat"):
                st.session_state.chat_history = []

            for item in reversed(st.session_state.chat_history):
                st.markdown(f"**❓ {item['question']}**")
                tag = "🤖 AI" if item["llm"] else "📊 İstatistik"
                st.caption(tag)
                st.markdown(item["answer"])
                st.markdown("---")



    # ============================================================
    # TAB 4: AI MODEL
    # ============================================================

    with tab4:
        st.markdown("## 🤖 AI Signal Quality Predictor")
        st.markdown(
            "Bu sekme, geçmiş paper trade sonuçlarından öğrenerek "
            "yeni sinyallerin kazanma ihtimalini tahmin eden XGBoost modelini yönetir."
        )

        # ── Milestone Banner: Gerçek Trade Sayacı ─────────────────
        # DEMO ve REJECTED tradeler hariç, gerçek kapanan trade sayısını izle.
        # Belirli eşiklerde kullanıcıya yeniden eğitim hatırlatması yap.
        _real_trades = [
            t for t in storage.get_closed_trades(limit=9999)
            if "[DEMO]" not in (t.get("notes") or "")
            and t.get("status") not in ("REJECTED", "PENDING")
        ]
        _real_count = len(_real_trades)
        _MILESTONE = 15   # Minimum eğitim eşiği

        if _real_count == 0:
            st.info(
                "📭 Henüz kapatılmış gerçek trade yok. "
                "Paper trade'leri kapat ve model eğitimine başla."
            )
        elif _real_count < _MILESTONE:
            _remaining = _MILESTONE - _real_count
            st.warning(
                f"⏳ **{_real_count}/{_MILESTONE} gerçek trade** — "
                f"Modeli güvenilir şekilde eğitmek için **{_remaining} trade daha** kapat."
            )
            st.progress(_real_count / _MILESTONE)
        elif _real_count >= 30:
            st.success(
                f"🏆 **{_real_count} gerçek trade!** Model güçlü bir veri tabanına sahip. "
                "Hâlâ en az 2 haftada bir yeniden eğitmeyi unutma."
            )
        else:
            # 15–29 arası: milestone geçildi, yeniden eğitim öner
            st.success(
                f"🎉 **{_real_count} gerçek trade ulaştı!** "
                "Modeli yeniden eğitmek için harika bir an. "
                "Aşağıdaki **🚀 Modeli Eğit** butonunu kullan."
            )


        from pathlib import Path
        import json

        META_PATH = Path("data/ml_models/signal_predictor_meta.json")
        MODEL_PATH = Path("data/ml_models/signal_predictor.pkl")

        st.markdown("### 📋 Model Durumu")

        if MODEL_PATH.exists() and META_PATH.exists():
            with open(META_PATH) as f:
                meta = json.load(f)

            # Renkli durum göstergesi
            st.success("✅ Model eğitilmiş ve hazır")

            # Metrik kartları
            ms1, ms2, ms3, ms4 = st.columns(4)
            with ms1:
                st.metric("Accuracy", f"{meta.get('accuracy', 0):.1%}")
            with ms2:
                auc = meta.get('roc_auc', 0)
                auc_label = "🟢 İyi" if auc >= 0.70 else "🟡 Makul" if auc >= 0.55 else "🔴 Zayıf"
                st.metric("ROC-AUC", f"{auc:.3f}", delta=auc_label)
            with ms3:
                st.metric("F1 Score", f"{meta.get('f1', 0):.3f}")
            with ms4:
                st.metric("Eğitim Verisi", f"{meta.get('total_trades', 0)} trade")

            # CV bilgisi
            cv_mean = meta.get('cv_roc_auc_mean', 0)
            cv_std = meta.get('cv_roc_auc_std', 0)
            st.caption(
                f"📅 Son eğitim: {meta.get('trained_at', '?')[:19]} | "
                f"5-Fold CV ROC-AUC: {cv_mean:.3f} ± {cv_std:.3f} | "
                f"Train: {meta.get('train_size', '?')} / Test: {meta.get('test_size', '?')}"
            )

            # Küçük veri uyarısı
            if meta.get('total_trades', 0) < 30:
                st.warning(
                    "⚠️ **Az veri:** Şu an model sentetik + gerçek karışık veriyle eğitildi. "
                    "Gerçek paper trade geçmişi arttıkça model daha güvenilir olacak. "
                    "30+ gerçek trade'den sonra yeniden eğitmeyi unutma."
                )
        else:
            st.error("❌ Model henüz eğitilmedi")
            st.info("Aşağıdaki butonu tıkla — sistem otomatik eğitecek.")

        st.markdown("---")

        # ── Model Eğitim Butonu ──────────────────────────────────
        st.markdown("### 🏋️ Modeli Eğit / Güncelle")
        st.markdown(
            "Yeni paper trade'ler kapatıldıkça buradan modeli güncelleyebilirsin. "
            "Eğitim birkaç saniye sürer."
        )

        # Demo trade sayısını göster
        demo_count = len([t for t in storage.get_closed_trades(limit=9999)
                         if '[DEMO]' in (t.get('notes') or '')])
        real_count = len([t for t in storage.get_closed_trades(limit=9999)
                         if '[DEMO]' not in (t.get('notes') or '')
                         and t.get('status') not in ('REJECTED',)])
        st.caption(f"📊 Mevcut veri: {real_count} gerçek trade + {demo_count} demo trade")

        train_col, info_col = st.columns([1, 3])
        with train_col:
            if st.button("🚀 Modeli Eğit", type="primary", key="train_ml_btn"):
                with st.spinner("Model eğitiliyor... (XGBoost + 5-Fold CV)"):
                    try:
                        from swing_trader.ml.trainer import SignalTrainer
                        trainer = SignalTrainer()
                        result = trainer.run()

                        if result.get('success'):
                            st.success(
                                f"✅ Eğitim tamamlandı!\n\n"
                                f"Accuracy: {result['accuracy']:.1%} | "
                                f"ROC-AUC: {result['roc_auc']:.3f} | "
                                f"F1: {result['f1']:.3f}"
                            )
                            st.rerun()
                        else:
                            st.error(f"❌ Eğitim başarısız: {result.get('error', 'Bilinmeyen hata')}")
                    except Exception as e:
                        st.error(f"❌ Hata: {e}")

        with info_col:
            st.info(
                "**Ne zaman yeniden eğitmeli?**\n"
                "- Her 10 yeni trade kapandığında\n"
                "- Strateji değişikliklerinden sonra\n"
                "- Model skoru düştüğünde"
            )

        st.markdown("---")

        # ── Feature Importance ──────────────────────────────────
        st.markdown("### 📊 Feature Importance")
        st.markdown("Model hangi özelliklere ne kadar önem veriyor?")

        if MODEL_PATH.exists():
            try:
                import joblib
                import plotly.graph_objects as go
                from swing_trader.ml.features import FEATURE_COLUMNS

                model = joblib.load(MODEL_PATH)
                importances = model.feature_importances_

                # Ada göre sırala (büyükten küçüğe)
                sorted_pairs = sorted(
                    zip(FEATURE_COLUMNS, importances),
                    key=lambda x: x[1]  # küçükten büyüğe (plotly yukarı çizer)
                )
                feat_names = [p[0] for p in sorted_pairs]
                feat_vals  = [p[1] for p in sorted_pairs]

                # Türkçe etiketler
                label_map = {
                    "risk_pct":           "Risk % (entry→stop)",
                    "reward_pct":         "Reward % (entry→target)",
                    "risk_reward_ratio":  "R/R Oranı",
                    "atr_pct":            "Volatilite (ATR%)",
                    "quality_score":      "Kalite Skoru",
                    "swing_type_enc":     "Swing Tipi (A/B/C/S)",
                    "max_hold_days":      "Max Hold Süresi",
                    "day_of_week":        "Giriş Günü",
                    "month":              "Giriş Ayı",
                }
                feat_labels = [label_map.get(n, n) for n in feat_names]

                fig = go.Figure(go.Bar(
                    x=feat_vals,
                    y=feat_labels,
                    orientation='h',
                    marker_color=[
                        f"rgba(99, 202, 183, {0.4 + v * 3})" for v in feat_vals
                    ],
                    text=[f"{v:.3f}" for v in feat_vals],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="XGBoost Feature Importance",
                    xaxis_title="Önem Skoru",
                    height=380,
                    margin=dict(l=10, r=30, t=40, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Açıklama kutusu
                top_feature = label_map.get(FEATURE_COLUMNS[int(importances.argmax())], "?")
                st.caption(
                    f"💡 **En önemli özellik:** {top_feature} — "
                    f"Model, tahmin yaparken en çok bu özelliğe bakıyor."
                )

            except Exception as e:
                st.warning(f"Feature importance yüklenemedi: {e}")
        else:
            st.info("Model eğitildikten sonra feature importance burada görünecek.")

        st.markdown("---")

        # ── Canlı Sinyal Testi ──────────────────────────────────
        st.markdown("### 🔬 Canlı Sinyal Testi")
        st.markdown("Bir sinyal gir, AI kazanma ihtimalini tahmin etsin:")

        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            test_entry  = st.number_input("Entry Price ($)", value=10.0, step=0.1, key="test_entry")
            test_stop   = st.number_input("Stop Loss ($)",   value=9.0,  step=0.1, key="test_stop")
            test_target = st.number_input("Target ($)",      value=13.0, step=0.1, key="test_target")
        with tc2:
            test_atr     = st.number_input("ATR",           value=0.5,  step=0.05, key="test_atr")
            test_quality = st.slider("Kalite Skoru",        min_value=0, max_value=10, value=7, key="test_quality")
            test_type    = st.selectbox("Swing Tipi",       ["A", "B", "C", "S"], key="test_type")
        with tc3:
            test_hold    = st.slider("Max Hold (gün)",      min_value=3, max_value=20, value=7, key="test_hold")

        if st.button("🤖 Tahmin Al", type="primary", key="predict_btn"):
            if MODEL_PATH.exists():
                try:
                    from swing_trader.ml.predictor import SignalPredictor
                    from datetime import datetime

                    predictor = SignalPredictor()
                    if predictor.is_ready:
                        test_signal = {
                            'entry_price':   test_entry,
                            'stop_loss':     test_stop,
                            'target':        test_target,
                            'atr':           test_atr,
                            'quality_score': test_quality,
                            'swing_type':    test_type,
                            'max_hold_days': test_hold,
                            'entry_date':    datetime.now().strftime('%Y-%m-%d'),
                        }
                        pred = predictor.predict(test_signal)
                        if pred:
                            win_pct  = int(pred['win_probability'] * 100)
                            loss_pct = 100 - win_pct
                            conf     = pred['confidence']

                            # Risk/Reward hızlı hesap
                            risk   = (test_entry - test_stop) / test_entry * 100
                            reward = (test_target - test_entry) / test_entry * 100
                            rr     = reward / risk if risk > 0 else 0

                            pr1, pr2 = st.columns(2)
                            with pr1:
                                if win_pct >= 60:
                                    st.success(f"🤖 **AI Tahmin:** Kazanma: **%{win_pct}** — {conf}")
                                elif win_pct >= 45:
                                    st.warning(f"🤖 **AI Tahmin:** Kazanma: **%{win_pct}** — {conf}")
                                else:
                                    st.error(f"🤖 **AI Tahmin:** Kazanma: **%{win_pct}** — {conf}")

                                st.caption(
                                    f"Risk: {risk:.1f}% | Reward: {reward:.1f}% | R/R: 1:{rr:.1f}"
                                )

                            with pr2:
                                # Feature önem sırası
                                st.markdown("**En etkili özellikler:**")
                                for feat in pred.get('top_features', [])[:3]:
                                    bar = "█" * int(feat['importance'] * 50)
                                    st.caption(f"{feat['feature']}: {bar} ({feat['importance']:.3f})")
                except Exception as e:
                    st.error(f"Tahmin hatası: {e}")
            else:
                st.warning("Önce modeli eğit!")



def dashboard_daily_summary():
    """
    Daily summary widget for sidebar.
    Shows open trades, P/L, and scan activity.
    Always visible regardless of current page.
    """
    try:
        paper_storage = PaperTradeStorage()
        paper_tracker = PaperTradeTracker(paper_storage)
        
        # Update and get open trades
        open_trades = paper_tracker.update_all_open_trades()
        closed_trades = paper_storage.get_closed_trades(limit=100)
        
        # ── OPEN TRADES ──
        st.sidebar.markdown("### 📋 Günlük Özet")
        
        if open_trades:
            st.sidebar.success(f"📌 **{len(open_trades)} açık trade**")
            
            for trade in open_trades:
                ticker = trade['ticker']
                entry = trade['entry_price']
                current = trade.get('current_price', entry)
                stop = trade['stop_loss']
                target = trade['target']
                pnl_pct = trade.get('unrealized_pnl_pct', 0)
                days = trade.get('days_held', 0)
                max_days = trade.get('max_hold_days', 7)
                
                # P/L emoji
                if pnl_pct >= 10:
                    pnl_icon = "🔥"
                elif pnl_pct >= 0:
                    pnl_icon = "🟢"
                else:
                    pnl_icon = "🔴"
                
                # Distance to stop and target
                stop_dist = ((current - stop) / current * 100) if current > 0 else 0
                target_dist = ((target - current) / current * 100) if current > 0 else 0
                
                # Proximity warning
                if stop_dist < 3:
                    proximity = "⚠️ Stop'a çok yakın!"
                elif target_dist < 5:
                    proximity = "🎯 Hedefe yakın!"
                elif days >= max_days - 1:
                    proximity = "⏰ Timeout yakın!"
                else:
                    proximity = ""
                
                st.sidebar.markdown(
                    f"{pnl_icon} **{ticker}** {pnl_pct:+.1f}% "
                    f"({days}/{max_days}g)"
                )
                if proximity:
                    st.sidebar.caption(f"  {proximity}")
        else:
            st.sidebar.info("📭 Açık trade yok")
        
        # ── GENEL İSTATİSTİK ──
        total_closed = len(closed_trades)
        if total_closed > 0:
            wins = sum(1 for t in closed_trades if (t.get('realized_pnl', 0) or 0) >= 0)
            total_pnl = sum((t.get('realized_pnl', 0) or 0) for t in closed_trades)
            win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
            
            pnl_color = "🟢" if total_pnl >= 0 else "🔴"
            st.sidebar.markdown(
                f"📊 {total_closed} kapalı | "
                f"WR: {win_rate:.0f}% | "
                f"{pnl_color} ${total_pnl:+.0f}"
            )
        
        # ── SCAN AKTİVİTESİ ──
        scan_history = st.session_state.get('scan_history', [])
        
        # Count scans in last 7 days
        from datetime import timedelta
        week_ago = datetime.now() - timedelta(days=7)
        recent_scans = [s for s in scan_history 
                       if datetime.fromisoformat(s['timestamp']) > week_ago]
        
        if recent_scans:
            last_scan = max(recent_scans, key=lambda x: x['timestamp'])
            last_dt = datetime.fromisoformat(last_scan['timestamp'])
            hours_ago = (datetime.now() - last_dt).total_seconds() / 3600
            
            if hours_ago < 1:
                time_str = f"{int(hours_ago * 60)}dk önce"
            elif hours_ago < 24:
                time_str = f"{int(hours_ago)}sa önce"
            else:
                time_str = f"{int(hours_ago / 24)}g önce"
            
            st.sidebar.caption(
                f"🔍 Son scan: {time_str} | "
                f"7 günde {len(recent_scans)} scan"
            )
        else:
            st.sidebar.warning("🔍 Son 7 günde scan yapılmadı!")
        
    except Exception as e:
        st.sidebar.caption(f"⚠️ Özet yüklenemedi")


def smallcap_backtest_page(components):
    """SmallCap Walk-Forward Backtest page."""
    st.title("📊 SmallCap Walk-Forward Backtest")
    st.markdown("*Geçmiş verilerle SmallCap sinyal kalitesini test edin*")
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        period = st.selectbox(
            "Test Periyodu",
            ["1 Ay", "2 Ay", "3 Ay", "6 Ay"],
            index=2
        )
        period_days = {"1 Ay": 30, "2 Ay": 60, "3 Ay": 90, "6 Ay": 180}[period]
    with col2:
        max_concurrent = st.number_input("Maks Eş Zamanlı Trade", min_value=1, max_value=10, value=3)
    with col3:
        initial_capital = st.number_input("Başlangıç Sermaye ($)", min_value=1000, value=10000, step=1000)
    
    # Ticker source
    ticker_source = st.radio(
        "Hisse Kaynağı",
        ["🔍 Finviz Tarama (Gerçek Small-Cap)", "📝 Manuel Listele"],
        horizontal=True
    )
    
    custom_tickers = []
    if ticker_source == "📝 Manuel Listele":
        ticker_input = st.text_input("Tickerları virgülle girin", "AEHR,AXTI,VELO,FSLY,NMRA,NVCR,SGRY")
        custom_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    
    if st.button("🚀 Backtest Başlat", type="primary"):
        from swing_trader.small_cap.smallcap_backtest import SmallCapBacktester
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        backtester = SmallCapBacktester(components['config'])
        
        # Get tickers
        if custom_tickers:
            tickers = custom_tickers
        else:
            with st.spinner("Finviz'den hisse listesi alınıyor..."):
                try:
                    from swing_trader.small_cap.engine import SmallCapEngine
                    engine = SmallCapEngine(components['config'])
                    tickers = engine.get_small_cap_universe(use_finviz=True, max_tickers=50)
                except Exception:
                    tickers = []
            
            if not tickers:
                st.warning("Hisse bulunamadı. Manuel liste kullanın.")
                return
        
        st.info(f"📊 {len(tickers)} hisse | {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')} | Maks {max_concurrent} eş zamanlı trade")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(pct, msg):
            progress_bar.progress(min(pct, 100))
            status_text.text(msg)
        
        # Run backtest
        results = backtester.run_backtest(
            tickers=tickers,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            initial_capital=initial_capital,
            max_concurrent=max_concurrent,
            min_quality=65,
            top_n=10,
            progress_callback=update_progress,
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Backtest tamamlandı!")
        
        metrics = results['metrics']
        trades = results['trades']
        
        if metrics['total_trades'] == 0:
            st.warning("⚠️ Bu dönemde hiç sinyal üretilmedi. Daha uzun periyot veya daha fazla hisse deneyin.")
            return
        
        # Store results in session for persistence
        st.session_state['backtest_results'] = results
        
        st.success(f"✅ **{metrics['total_trades']} trade** tamamlandı!")
        
        # ── KEY METRICS ──
        st.markdown("---")
        st.subheader("📈 Temel Metrikler")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Toplam Trade", metrics['total_trades'])
        
        wr = metrics['win_rate']
        wr_color = "normal" if wr >= 0.5 else "inverse"
        col2.metric("Win Rate", f"{wr:.0%}", delta=f"{metrics['winning_trades']}W / {metrics['losing_trades']}L")
        
        col3.metric("Profit Factor", f"{metrics['profit_factor']:.2f}", 
                    delta="İyi" if metrics['profit_factor'] > 1.5 else "Zayıf")
        
        col4.metric("Toplam P/L", f"${metrics['total_pnl_dollar']:+,.0f}",
                    delta=f"{metrics['total_return']:+.1%}")
        
        col5.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f}%")
        
        # ── AVG WIN/LOSS ──
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**💰 Kazanç Analizi**")
            st.write(f"Ort. Kazanç: **+{metrics['avg_win_pct']:.1f}%** (${metrics['avg_win_dollar']:+.0f})")
            st.write(f"Ort. Kayıp: **{metrics['avg_loss_pct']:.1f}%** (${metrics['avg_loss_dollar']:+.0f})")
            rr = abs(metrics['avg_win_pct'] / metrics['avg_loss_pct']) if metrics['avg_loss_pct'] != 0 else 0
            st.write(f"Risk/Reward: **{rr:.1f}:1**")
        
        with col2:
            st.markdown("**⏱️ Süre Analizi**")
            st.write(f"Ort. Tutma Süresi: **{metrics['avg_hold_days']:.1f} gün**")
            st.write(f"Başlangıç: ${metrics['initial_capital']:,.0f}")
            st.write(f"Bitiş: ${metrics['final_capital']:,.0f}")
        
        with col3:
            st.markdown("**📊 Swing Tipi Dağılımı**")
            type_stats = metrics.get('type_stats', {})
            for stype, stats in type_stats.items():
                total_t = stats['wins'] + stats['losses']
                wr_t = stats['wins'] / total_t if total_t > 0 else 0
                st.write(f"**Tip {stype}**: {total_t} trade, WR: {wr_t:.0%}, P/L: {stats['total_pnl']:+.1f}%")
        
        # ── EXIT ANALYSIS ──
        st.markdown("---")
        st.subheader("🚪 Çıkış Analizi")
        exit_stats = metrics.get('exit_stats', {})
        
        exit_labels = {
            'STOPPED': '🛑 Stop Loss',
            'TARGET': '🎯 Target Hit',
            'TIMEOUT': '⏰ Timeout',
            'FORCED': '🔚 Backtest Sonu'
        }
        
        exit_cols = st.columns(len(exit_stats))
        for i, (reason, stats) in enumerate(exit_stats.items()):
            with exit_cols[i] if i < len(exit_cols) else st.columns(1)[0]:
                label = exit_labels.get(reason, reason)
                st.metric(label, stats['count'], delta=f"Ort: {stats['avg_pnl']:+.1f}%")
        
        # ── EQUITY CURVE ──
        equity_curve = results.get('equity_curve', [])
        if equity_curve:
            st.markdown("---")
            st.subheader("📈 Equity Curve (Portföy Değeri)")
            
            eq_df = pd.DataFrame(equity_curve)
            eq_df['date'] = pd.to_datetime(eq_df['date'])
            
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq_df['date'],
                y=eq_df['portfolio_value'],
                name='Portföy Değeri',
                line=dict(color='#2196F3', width=2),
                fill='tozeroy',
                fillcolor='rgba(33,150,243,0.1)'
            ))
            fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray",
                         annotation_text="Başlangıç Sermaye")
            fig.update_layout(
                yaxis_title='Portföy Değeri ($)',
                xaxis_title='Tarih',
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # ── TRADE LOG ──
        if trades:
            st.markdown("---")
            st.subheader("📝 Trade Geçmişi")
            
            log_data = []
            for t in trades:
                log_data.append({
                    'Hisse': t['ticker'],
                    'Tip': t.get('swing_type', '-'),
                    'Giriş': f"${t['entry_price']:.2f}",
                    'Çıkış': f"${t['exit_price']:.2f}",
                    'P/L %': f"{t['pnl_pct']:+.1f}%",
                    'P/L $': f"${t['pnl_dollar']:+.0f}",
                    'Giriş Tarihi': t['entry_date'],
                    'Çıkış Tarihi': t.get('exit_date', '-'),
                    'Süre (gün)': t.get('days_held', '-'),
                    'Çıkış Nedeni': t.get('exit_reason', '-'),
                    'Skor': t.get('quality_score', 0)
                })
            
            log_df = pd.DataFrame(log_data)
            st.dataframe(log_df, use_container_width=True, height=400)


def main():
    """Main dashboard application."""
    # Initialize components
    components = init_components()
    
    if not components:
        st.error("Failed to initialize system. Please check config.yaml")
        return
    
    # Sidebar
    st.sidebar.title("📈 Swing Trader")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🔍 Scan Stocks (Large Cap)", "🚀 SmallCap Momentum", "📝 Manual Lookup", "📊 Paper Trades", "📉 Backtest (LargeCap)", "📊 Backtest (SmallCap)", "⚙️ Settings"]
    )
    
    # ── DAILY SUMMARY (Always visible) ──
    st.sidebar.markdown("---")
    dashboard_daily_summary()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Info")
    config = components['config']
    st.sidebar.write(f"Data Source: {config['data']['source']}")
    st.sidebar.write(f"Max Positions: {config['risk']['max_open_positions']}")
    st.sidebar.write(f"Risk per Trade: {config['risk']['max_risk_per_trade']:.1%}")
    
    # Route to page
    if page == "🔍 Scan Stocks (Large Cap)":
        scan_page(components)
    elif page == "🚀 SmallCap Momentum":
        small_cap_page(components)
    elif page == "📝 Manual Lookup":
        manual_lookup_page(components)
    elif page == "📊 Paper Trades":
        paper_trades_page(components)
    elif page == "📉 Backtest (LargeCap)":
        backtest_page(components)
    elif page == "📊 Backtest (SmallCap)":
        smallcap_backtest_page(components)
    elif page == "⚙️ Settings":
        settings_page(components)

if __name__ == "__main__":
    main()
