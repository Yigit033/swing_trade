"""
Performance metrics calculation for backtest results.
"""

import logging
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calculate performance metrics from backtest results.
    """
    
    @staticmethod
    def calculate_metrics(backtest_results: Dict) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            backtest_results: Dictionary from BacktestEngine.run_backtest()
        
        Returns:
            Dictionary with all performance metrics
        
        Example:
            >>> metrics = PerformanceMetrics.calculate_metrics(backtest_results)
            >>> print(f"Win Rate: {metrics['win_rate']:.1%}")
            >>> print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        """
        try:
            trades = backtest_results.get('trades', [])
            equity_curve = backtest_results.get('equity_curve', [])
            
            if not trades:
                logger.warning("No trades to analyze")
                return PerformanceMetrics._empty_metrics()
            
            # Convert trades to DataFrame for easier analysis
            trades_df = pd.DataFrame(trades)
            
            # Basic metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            breakeven_trades = len(trades_df[trades_df['pnl'] == 0])
            
            # Win rate
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # P&L metrics
            total_pnl = trades_df['pnl'].sum()
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            
            total_wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
            total_losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
            
            # Profit factor
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Best and worst trades
            best_trade = trades_df['pnl'].max()
            worst_trade = trades_df['pnl'].min()
            best_trade_pct = trades_df['pnl_percent'].max()
            worst_trade_pct = trades_df['pnl_percent'].min()
            
            # Holding period
            avg_hold_days = trades_df['hold_days'].mean()
            
            # Drawdown calculation
            if equity_curve:
                equity_df = pd.DataFrame(equity_curve)
                drawdown_metrics = PerformanceMetrics._calculate_drawdown(equity_df)
            else:
                drawdown_metrics = {
                    'max_drawdown': 0,
                    'max_drawdown_percent': 0,
                    'longest_drawdown_days': 0
                }
            
            # Sharpe ratio
            sharpe_ratio = PerformanceMetrics._calculate_sharpe_ratio(trades_df, equity_df if equity_curve else None)
            
            # NEW: Sortino ratio
            sortino_ratio = PerformanceMetrics._calculate_sortino_ratio(equity_df if equity_curve else None)
            
            # Monthly returns
            monthly_returns = PerformanceMetrics._calculate_monthly_returns(trades_df)
            
            # Return metrics
            initial_capital = backtest_results.get('initial_capital', 0)
            final_value = backtest_results.get('final_portfolio_value', 0)
            total_return = (final_value - initial_capital) / initial_capital if initial_capital > 0 else 0
            
            # Calculate CAGR
            start_date = backtest_results.get('start_date', '')
            end_date = backtest_results.get('end_date', '')
            years = PerformanceMetrics._calculate_years(start_date, end_date)
            cagr = ((final_value / initial_capital) ** (1 / years) - 1) if years > 0 and initial_capital > 0 else 0
            
            # NEW: Calmar ratio (CAGR / Max Drawdown)
            calmar_ratio = PerformanceMetrics._calculate_calmar_ratio(cagr, drawdown_metrics['max_drawdown_percent'])
            
            # NEW: Exposure % (days with positions / total days)
            exposure_pct = PerformanceMetrics._calculate_exposure(equity_curve)
            
            # NEW: Trades per year
            trades_per_year = total_trades / years if years > 0 else 0
            
            # NEW: Worst consecutive losses
            worst_consecutive_losses = PerformanceMetrics._calculate_worst_consecutive_losses(trades_df)
            
            # Compile all metrics
            metrics = {
                # Trade metrics
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'breakeven_trades': breakeven_trades,
                'win_rate': win_rate,
                
                # P&L metrics
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_wins': total_wins,
                'total_losses': total_losses,
                'profit_factor': profit_factor,
                
                # Best/Worst
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'best_trade_pct': best_trade_pct,
                'worst_trade_pct': worst_trade_pct,
                
                # Holding period
                'avg_hold_days': avg_hold_days,
                
                # Drawdown
                'max_drawdown': drawdown_metrics['max_drawdown'],
                'max_drawdown_percent': drawdown_metrics['max_drawdown_percent'],
                'longest_drawdown_days': drawdown_metrics['longest_drawdown_days'],
                
                # Risk-adjusted returns
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,  # NEW
                'calmar_ratio': calmar_ratio,    # NEW
                
                # Returns
                'total_return': total_return,
                'cagr': cagr,
                'initial_capital': initial_capital,
                'final_value': final_value,
                
                # NEW metrics
                'exposure_percent': exposure_pct,
                'trades_per_year': trades_per_year,
                'worst_consecutive_losses': worst_consecutive_losses,
                
                # Monthly data
                'monthly_returns': monthly_returns
            }
            
            logger.info("Performance metrics calculated successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}", exc_info=True)
            return PerformanceMetrics._empty_metrics()
    
    @staticmethod
    def _calculate_drawdown(equity_df: pd.DataFrame) -> Dict:
        """Calculate maximum drawdown metrics."""
        try:
            equity = equity_df['portfolio_value'].values
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(equity)
            
            # Calculate drawdown
            drawdown = equity - running_max
            drawdown_pct = (drawdown / running_max) * 100
            
            # Maximum drawdown
            max_dd = drawdown.min()
            max_dd_pct = drawdown_pct.min()
            
            # Find longest drawdown period
            in_drawdown = drawdown < 0
            drawdown_lengths = []
            current_length = 0
            
            for is_dd in in_drawdown:
                if is_dd:
                    current_length += 1
                else:
                    if current_length > 0:
                        drawdown_lengths.append(current_length)
                    current_length = 0
            
            if current_length > 0:
                drawdown_lengths.append(current_length)
            
            longest_dd_days = max(drawdown_lengths) if drawdown_lengths else 0
            
            return {
                'max_drawdown': max_dd,
                'max_drawdown_percent': max_dd_pct,
                'longest_drawdown_days': longest_dd_days
            }
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return {
                'max_drawdown': 0,
                'max_drawdown_percent': 0,
                'longest_drawdown_days': 0
            }
    
    @staticmethod
    def _calculate_sharpe_ratio(trades_df: pd.DataFrame, equity_df: pd.DataFrame = None, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        try:
            if equity_df is not None and len(equity_df) > 1:
                # Calculate returns from equity curve
                equity_df = equity_df.copy()
                equity_df['returns'] = equity_df['portfolio_value'].pct_change()
                
                # Remove NaN
                returns = equity_df['returns'].dropna()
                
                if len(returns) == 0:
                    return 0.0
                
                # Annualize
                mean_return = returns.mean() * 252  # 252 trading days
                std_return = returns.std() * np.sqrt(252)
                
                if std_return == 0:
                    return 0.0
                
                sharpe = (mean_return - risk_free_rate) / std_return
                return sharpe
            else:
                # Calculate from trade returns
                if len(trades_df) == 0:
                    return 0.0
                
                returns = trades_df['pnl_percent'] / 100
                mean_return = returns.mean()
                std_return = returns.std()
                
                if std_return == 0:
                    return 0.0
                
                sharpe = (mean_return - risk_free_rate) / std_return
                return sharpe
                
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    @staticmethod
    def _calculate_monthly_returns(trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate monthly returns."""
        try:
            trades_df = trades_df.copy()
            trades_df['exit_month'] = pd.to_datetime(trades_df['exit_date']).dt.to_period('M')
            
            monthly = trades_df.groupby('exit_month')['pnl'].sum().to_dict()
            
            # Convert Period keys to strings
            monthly_str = {str(k): v for k, v in monthly.items()}
            
            return monthly_str
            
        except Exception as e:
            logger.error(f"Error calculating monthly returns: {e}")
            return {}
    
    @staticmethod
    def _calculate_years(start_date: str, end_date: str) -> float:
        """Calculate years between dates."""
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            days = (end - start).days
            return days / 365.25
        except:
            return 1.0
    
    @staticmethod
    def _calculate_sortino_ratio(equity_df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (uses downside deviation instead of std)."""
        try:
            if equity_df is None or len(equity_df) < 2:
                return 0.0
            
            equity_df = equity_df.copy()
            equity_df['returns'] = equity_df['portfolio_value'].pct_change()
            returns = equity_df['returns'].dropna()
            
            if len(returns) == 0:
                return 0.0
            
            # Annualize mean return
            mean_return = returns.mean() * 252
            
            # Downside deviation (only negative returns)
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return 10.0  # All positive returns
            
            downside_std = downside_returns.std() * np.sqrt(252)
            
            if downside_std == 0:
                return 0.0
            
            sortino = (mean_return - risk_free_rate) / downside_std
            return sortino
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    @staticmethod
    def _calculate_calmar_ratio(cagr: float, max_drawdown_pct: float) -> float:
        """Calculate Calmar ratio (CAGR / Max Drawdown)."""
        try:
            if max_drawdown_pct == 0:
                return 0.0
            # max_drawdown_pct is negative, so we use abs
            return cagr / abs(max_drawdown_pct / 100) if max_drawdown_pct != 0 else 0.0
        except:
            return 0.0
    
    @staticmethod
    def _calculate_exposure(equity_curve: List[Dict]) -> float:
        """Calculate exposure % (days with positions / total days)."""
        try:
            if not equity_curve:
                return 0.0
            
            days_with_positions = sum(1 for e in equity_curve if e.get('num_positions', 0) > 0)
            total_days = len(equity_curve)
            
            return days_with_positions / total_days if total_days > 0 else 0.0
        except:
            return 0.0
    
    @staticmethod
    def _calculate_worst_consecutive_losses(trades_df: pd.DataFrame) -> int:
        """Calculate worst streak of consecutive losing trades."""
        try:
            if len(trades_df) == 0:
                return 0
            
            max_consecutive = 0
            current_consecutive = 0
            
            for pnl in trades_df['pnl']:
                if pnl < 0:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
        except:
            return 0
    
    @staticmethod
    def _empty_metrics() -> Dict:
        """Return empty metrics dictionary."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'breakeven_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'total_wins': 0.0,
            'total_losses': 0.0,
            'profit_factor': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'best_trade_pct': 0.0,
            'worst_trade_pct': 0.0,
            'avg_hold_days': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_percent': 0.0,
            'longest_drawdown_days': 0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,  # NEW
            'calmar_ratio': 0.0,   # NEW
            'total_return': 0.0,
            'cagr': 0.0,
            'initial_capital': 0.0,
            'final_value': 0.0,
            'exposure_percent': 0.0,          # NEW
            'trades_per_year': 0.0,           # NEW
            'worst_consecutive_losses': 0,    # NEW
            'monthly_returns': {}
        }
    
    @staticmethod
    def print_metrics(metrics: Dict):
        """
        Print formatted performance metrics.
        
        Args:
            metrics: Metrics dictionary from calculate_metrics()
        
        Example:
            >>> metrics = PerformanceMetrics.calculate_metrics(results)
            >>> PerformanceMetrics.print_metrics(metrics)
        """
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        
        print("\n📊 RETURN METRICS")
        print(f"Initial Capital:      ${metrics['initial_capital']:>12,.2f}")
        print(f"Final Value:          ${metrics['final_value']:>12,.2f}")
        print(f"Total P&L:            ${metrics['total_pnl']:>12,.2f}")
        print(f"Total Return:         {metrics['total_return']:>12.2%}")
        print(f"CAGR:                 {metrics['cagr']:>12.2%}")
        
        print("\n📈 TRADE STATISTICS")
        print(f"Total Trades:         {metrics['total_trades']:>12}")
        print(f"Winning Trades:       {metrics['winning_trades']:>12} ({metrics['winning_trades']/metrics['total_trades']*100 if metrics['total_trades'] > 0 else 0:.1f}%)")
        print(f"Losing Trades:        {metrics['losing_trades']:>12} ({metrics['losing_trades']/metrics['total_trades']*100 if metrics['total_trades'] > 0 else 0:.1f}%)")
        print(f"Win Rate:             {metrics['win_rate']:>12.1%}")
        
        print("\n💰 P&L ANALYSIS")
        print(f"Average Win:          ${metrics['avg_win']:>12,.2f}")
        print(f"Average Loss:         ${metrics['avg_loss']:>12,.2f}")
        print(f"Profit Factor:        {metrics['profit_factor']:>12.2f}")
        print(f"Best Trade:           ${metrics['best_trade']:>12,.2f} ({metrics['best_trade_pct']:.1f}%)")
        print(f"Worst Trade:          ${metrics['worst_trade']:>12,.2f} ({metrics['worst_trade_pct']:.1f}%)")
        
        print("\n⏱️  HOLDING PERIOD")
        print(f"Avg Hold Days:        {metrics['avg_hold_days']:>12.1f}")
        
        print("\n📉 RISK METRICS")
        print(f"Max Drawdown:         ${metrics['max_drawdown']:>12,.2f} ({metrics['max_drawdown_percent']:.1f}%)")
        print(f"Longest DD Period:    {metrics['longest_drawdown_days']:>12} days")
        print(f"Sharpe Ratio:         {metrics['sharpe_ratio']:>12.2f}")
        
        print("\n" + "="*60)

