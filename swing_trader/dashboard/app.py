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
    
    else:
        rejection = result.get('rejection_reason', 'Does not meet swing criteria')
        
        st.warning(f"""
        ### ❌ {ticker} - NOT SWING READY
        **{company_name}** | {sector} | ${market_cap/1e9:.2f}B
        
        **Reason:** {rejection}
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
        
        # Show filter details if available - AUTO EXPANDED for rejection
        filter_details = result.get('filter_details', {})
        if filter_details:
            st.markdown("**🔍 Filter Details:**")
            filters = filter_details.get('filters', {})
            for key, val in filters.items():
                passed = val.get('passed', False)
                reason = val.get('reason', '')
                icon = "✅" if passed else "❌"
                st.write(f"{icon} **{key}**: {reason}")
    
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
                
            except Exception as e:
                st.error(f"Error during scan: {e}")
                logging.exception("SmallCap scan error")
    
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
        for s in signals:
            display_data.append({
                'Ticker': s['ticker'],
                'Quality': f"{s['quality_score']:.0f}",
                'Entry': f"${s['entry_price']:.2f}",
                'Stop': f"${s['stop_loss']:.2f}",
                'Target (3R)': f"${s['target_1']:.2f}",
                'Vol Surge': f"{s['volume_surge']:.1f}x",
                'ATR%': f"{s['atr_percent']:.1f}%",
                'Float': f"{s['float_millions']:.0f}M",
                'Hold': f"{s['expected_hold_min']}-{s['expected_hold_max']}d",
                '⚠️': '🔴' if s.get('volatility_warning') else '🟢'
            })
        
        df_display = pd.DataFrame(display_data)
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # Legend
        with st.expander("📚 Column Explanation"):
            st.markdown("""
            | Column | Meaning |
            |--------|---------|
            | **Quality** | Momentum quality score (0-100) |
            | **Entry** | Current close price (enter here) |
            | **Stop** | Stop loss (1-1.5 ATR) |
            | **Target (3R)** | Minimum 3:1 reward target |
            | **Vol Surge** | Volume vs 20-day avg |
            | **ATR%** | Volatility (ATR/Price) |
            | **Float** | Shares floating (smaller = more explosive) |
            | **Hold** | Expected holding period |
            | **⚠️** | Volatility warning (always 🔴 for small caps) |
            """)
        
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
        ["🔍 Scan Stocks (Large Cap)", "🚀 SmallCap Momentum", "📝 Manual Lookup", "📉 Backtest", "⚙️ Settings"]
    )
    
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
    elif page == "📉 Backtest":
        backtest_page(components)
    elif page == "⚙️ Settings":
        settings_page(components)

if __name__ == "__main__":
    main()

