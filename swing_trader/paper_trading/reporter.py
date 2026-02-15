"""
Paper Trading Reporter - Performance metrics and statistics.
"""

import logging
from typing import Dict, List
from datetime import datetime

from .storage import PaperTradeStorage

logger = logging.getLogger(__name__)


class PaperTradeReporter:
    """
    Calculate performance metrics for paper trades.
    
    Metrics:
    - Total trades
    - Win rate
    - Average win / Average loss
    - Profit factor
    - Total P/L
    - Best/Worst trade
    - Average hold time
    - Performance by swing type
    """
    
    def __init__(self, storage: PaperTradeStorage = None):
        """Initialize reporter with storage."""
        self.storage = storage or PaperTradeStorage()
    
    def get_performance_summary(self) -> Dict:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dict with all performance metrics
        """
        closed_trades = self.storage.get_closed_trades(limit=1000)
        open_trades = self.storage.get_open_trades()
        
        summary = {
            'total_trades': len(closed_trades) + len(open_trades),
            'open_trades': len(open_trades),
            'closed_trades': len(closed_trades),
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'best_trade': None,
            'worst_trade': None,
            'avg_hold_days': 0.0,
            'by_exit_type': {},
            'by_swing_type': {}
        }
        
        if not closed_trades:
            return summary
        
        # Calculate metrics
        wins = []
        losses = []
        hold_days_list = []
        
        for trade in closed_trades:
            pnl = trade.get('realized_pnl', 0) or 0
            pnl_pct = trade.get('realized_pnl_pct', 0) or 0
            
            if pnl >= 0:
                wins.append(pnl)
                summary['wins'] += 1
            else:
                losses.append(abs(pnl))
                summary['losses'] += 1
            
            # Track best/worst
            if summary['best_trade'] is None or pnl_pct > summary['best_trade']['pnl_pct']:
                summary['best_trade'] = {
                    'ticker': trade['ticker'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'date': trade['exit_date']
                }
            
            if summary['worst_trade'] is None or pnl_pct < summary['worst_trade']['pnl_pct']:
                summary['worst_trade'] = {
                    'ticker': trade['ticker'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'date': trade['exit_date']
                }
            
            # Hold days
            if trade.get('entry_date') and trade.get('exit_date'):
                try:
                    entry_dt = datetime.strptime(trade['entry_date'], '%Y-%m-%d')
                    exit_dt = datetime.strptime(trade['exit_date'], '%Y-%m-%d')
                    hold_days = (exit_dt - entry_dt).days
                    hold_days_list.append(hold_days)
                except:
                    pass
            
            # By exit type
            status = trade.get('status', 'UNKNOWN')
            if status not in summary['by_exit_type']:
                summary['by_exit_type'][status] = {
                    'count': 0, 'total_pnl': 0, 'avg_pnl_pct': 0, 'pnl_pcts': []
                }
            summary['by_exit_type'][status]['count'] += 1
            summary['by_exit_type'][status]['total_pnl'] += pnl
            summary['by_exit_type'][status]['pnl_pcts'].append(pnl_pct)
            
            # By swing type
            swing_type = trade.get('swing_type', 'A') or 'A'
            if swing_type not in summary['by_swing_type']:
                summary['by_swing_type'][swing_type] = {
                    'count': 0, 'wins': 0, 'total_pnl': 0, 'pnl_pcts': []
                }
            summary['by_swing_type'][swing_type]['count'] += 1
            summary['by_swing_type'][swing_type]['total_pnl'] += pnl
            summary['by_swing_type'][swing_type]['pnl_pcts'].append(pnl_pct)
            if pnl >= 0:
                summary['by_swing_type'][swing_type]['wins'] += 1
        
        # Calculate averages
        total_closed = len(closed_trades)
        summary['win_rate'] = (summary['wins'] / total_closed * 100) if total_closed > 0 else 0
        summary['total_pnl'] = sum(wins) - sum(losses)
        summary['avg_win'] = sum(wins) / len(wins) if wins else 0
        summary['avg_loss'] = sum(losses) / len(losses) if losses else 0
        
        # Profit factor
        total_wins = sum(wins)
        total_losses = sum(losses)
        summary['profit_factor'] = (total_wins / total_losses) if total_losses > 0 else float('inf')
        
        # Avg hold days
        summary['avg_hold_days'] = sum(hold_days_list) / len(hold_days_list) if hold_days_list else 0
        
        # Calculate avg pnl_pct for exit types
        for status, data in summary['by_exit_type'].items():
            pcts = data['pnl_pcts']
            data['avg_pnl_pct'] = sum(pcts) / len(pcts) if pcts else 0
            del data['pnl_pcts']  # Clean up
        
        # Calculate win rate and avg for swing types
        for swing_type, data in summary['by_swing_type'].items():
            data['win_rate'] = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
            pcts = data['pnl_pcts']
            data['avg_pnl_pct'] = sum(pcts) / len(pcts) if pcts else 0
            del data['pnl_pcts']  # Clean up
        
        return summary
    
    def get_open_trades_summary(self) -> Dict:
        """
        Get summary of open trades.
        V3: Includes trailing_stop info and separates PENDING trades.
        
        Returns:
            Dict with open trade and pending trade statistics
        """
        # Import tracker here to avoid circular import
        from .tracker import PaperTradeTracker
        
        tracker = PaperTradeTracker(self.storage)
        
        # First confirm any pending trades
        confirm_results = tracker.confirm_pending_trades()
        
        # Then update all open trades
        open_trades = tracker.update_all_open_trades()
        
        summary = {
            'count': 0,
            'pending_count': 0,
            'total_unrealized_pnl': 0,
            'trades': [],
            'pending_trades': [],
            'confirm_results': confirm_results
        }
        
        for trade in open_trades:
            trade_info = {
                'id': trade['id'],
                'ticker': trade['ticker'],
                'entry_date': trade['entry_date'],
                'entry_price': trade['entry_price'],
                'current_price': trade.get('current_price', trade['entry_price']),
                'stop_loss': trade['stop_loss'],
                'trailing_stop': trade.get('trailing_stop') or trade['stop_loss'],
                'initial_stop': trade.get('initial_stop') or trade['stop_loss'],
                'target': trade['target'],
                'atr': trade.get('atr') or 0,
                'unrealized_pnl': trade.get('unrealized_pnl', 0),
                'unrealized_pnl_pct': trade.get('unrealized_pnl_pct', 0),
                'days_held': trade.get('days_held', 0),
                'max_hold_days': trade.get('max_hold_days', 7),
                'swing_type': trade.get('swing_type', 'A'),
                'signal_price': trade.get('signal_price'),
                'status': trade.get('status', 'OPEN')
            }
            
            if trade.get('status') == 'PENDING':
                summary['pending_count'] += 1
                summary['pending_trades'].append(trade_info)
            else:
                summary['count'] += 1
                summary['total_unrealized_pnl'] += trade.get('unrealized_pnl', 0)
                summary['trades'].append(trade_info)
        
        return summary
    
    def format_summary_text(self) -> str:
        """
        Format performance summary as text.
        
        Returns:
            Formatted string for display
        """
        summary = self.get_performance_summary()
        
        lines = [
            "=" * 50,
            "PAPER TRADING PERFORMANCE SUMMARY",
            "=" * 50,
            "",
            f"Total Trades: {summary['total_trades']}",
            f"  Open: {summary['open_trades']}",
            f"  Closed: {summary['closed_trades']}",
            "",
            f"Win Rate: {summary['win_rate']:.1f}%",
            f"  Wins: {summary['wins']} | Losses: {summary['losses']}",
            "",
            f"Total P/L: ${summary['total_pnl']:+.2f}",
            f"  Avg Win: ${summary['avg_win']:.2f}",
            f"  Avg Loss: ${summary['avg_loss']:.2f}",
            f"  Profit Factor: {summary['profit_factor']:.2f}",
            "",
            f"Avg Hold Days: {summary['avg_hold_days']:.1f}",
            ""
        ]
        
        if summary['best_trade']:
            bt = summary['best_trade']
            lines.append(f"Best Trade: {bt['ticker']} ({bt['pnl_pct']:+.1f}%)")
        
        if summary['worst_trade']:
            wt = summary['worst_trade']
            lines.append(f"Worst Trade: {wt['ticker']} ({wt['pnl_pct']:+.1f}%)")
        
        lines.append("")
        lines.append("BY EXIT TYPE:")
        for status, data in summary['by_exit_type'].items():
            lines.append(f"  {status}: {data['count']} trades, avg {data['avg_pnl_pct']:+.1f}%")
        
        lines.append("")
        lines.append("BY SWING TYPE:")
        for swing_type, data in summary['by_swing_type'].items():
            lines.append(
                f"  Type {swing_type}: {data['count']} trades, "
                f"{data['win_rate']:.0f}% win rate, avg {data['avg_pnl_pct']:+.1f}%"
            )
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
