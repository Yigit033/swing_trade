"""
Catalyst Detection Module - Short Interest, Insider Buying, News.
Senior Trader v2.1 Feature
"""

import logging
from typing import Dict, Optional
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CatalystDetector:
    """
    Detect catalysts that can drive momentum:
    - Short Interest & Days to Cover
    - Insider Buying
    - News Sentiment (basic)
    
    Type S requires: SI ≥ 20%, DTC ≥ 5
    Others: SI ≥ 10% → +5 bonus
    Insider Buying > $500K in 30d → +8 bonus
    """
    
    @classmethod
    def get_short_interest_data(cls, ticker: str) -> Dict:
        """
        Get Short Interest data from yfinance.
        
        Returns:
            {
                'short_percent': float,  # Short % of float
                'shares_short': int,
                'days_to_cover': float,
                'is_squeeze_candidate': bool,
                'bonus': int
            }
        """
        result = {
            'short_percent': 0.0,
            'shares_short': 0,
            'days_to_cover': 0.0,
            'is_squeeze_candidate': False,
            'bonus': 0
        }
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get short data
            shares_short = info.get('sharesShort', 0) or 0
            float_shares = info.get('floatShares', 0) or 0
            avg_volume = info.get('averageVolume', 0) or 0
            short_ratio = info.get('shortRatio', 0) or 0  # Days to cover
            
            # Calculate short percent of float
            if float_shares > 0:
                result['short_percent'] = (shares_short / float_shares) * 100
            
            result['shares_short'] = shares_short
            result['days_to_cover'] = short_ratio
            
            # Check for squeeze candidate (Type S criteria)
            if result['short_percent'] >= 20 and result['days_to_cover'] >= 5:
                result['is_squeeze_candidate'] = True
                result['bonus'] = 10  # Strong squeeze setup
            elif result['short_percent'] >= 15 and result['days_to_cover'] >= 3:
                result['bonus'] = 7   # Moderate squeeze potential
            elif result['short_percent'] >= 10:
                result['bonus'] = 5   # Elevated short interest
            
            return result
            
        except Exception as e:
            logger.debug(f"Error getting short interest for {ticker}: {e}")
            return result
    
    @classmethod
    def get_insider_activity(cls, ticker: str) -> Dict:
        """
        Get insider buying activity from yfinance.
        
        Returns:
            {
                'net_insider_purchases': float,  # Net $ value
                'has_insider_buying': bool,
                'bonus': int
            }
        """
        result = {
            'net_insider_purchases': 0.0,
            'has_insider_buying': False,
            'insider_count': 0,
            'bonus': 0
        }
        
        try:
            stock = yf.Ticker(ticker)
            
            # Try to get insider transactions
            try:
                insider_txns = stock.insider_transactions
                if insider_txns is not None and len(insider_txns) > 0:
                    # FIX v2.3: Actually enforce date filter (was defined but never used)
                    cutoff = datetime.now() - timedelta(days=90)  # 90 days (was 30, expanded for SEC filing delays)
                    
                    # Calculate net purchases
                    net_value = 0
                    buy_count = 0
                    
                    for _, row in insider_txns.iterrows():
                        # DATE FILTER — skip old transactions
                        try:
                            txn_date = pd.to_datetime(
                                row.get('Start Date', row.get('Date', None)),
                                errors='coerce'
                            )
                            if txn_date is not None and not pd.isna(txn_date):
                                if txn_date.to_pydatetime().replace(tzinfo=None) < cutoff:
                                    continue  # Too old, skip
                        except Exception:
                            pass  # If date parsing fails, include the transaction (safe fallback)
                        
                        # Check transaction type
                        txn_type = str(row.get('Transaction', '')).lower()
                        value = row.get('Value', 0) or 0
                        
                        if 'purchase' in txn_type or 'buy' in txn_type:
                            net_value += value
                            buy_count += 1
                        elif 'sale' in txn_type or 'sell' in txn_type:
                            net_value -= value
                    
                    result['net_insider_purchases'] = net_value
                    result['insider_count'] = buy_count
                    
                    # Award bonus for significant insider buying
                    if net_value >= 1_000_000:
                        result['has_insider_buying'] = True
                        result['bonus'] = 8  # >$1M insider buying
                    elif net_value >= 500_000:
                        result['has_insider_buying'] = True
                        result['bonus'] = 6  # >$500K
                    elif net_value >= 100_000:
                        result['has_insider_buying'] = True
                        result['bonus'] = 3  # >$100K
                        
            except Exception:
                pass  # Insider data not available
            
            return result
            
        except Exception as e:
            logger.debug(f"Error getting insider activity for {ticker}: {e}")
            return result
    
    @classmethod
    def get_news_sentiment(cls, ticker: str) -> Dict:
        """
        Basic news sentiment from yfinance news.
        
        Note: This is a simplified version. For production,
        use NewsAPI or Finnhub for better sentiment analysis.
        
        Returns:
            {
                'news_count': int,
                'has_recent_news': bool,
                'bonus': int
            }
        """
        result = {
            'news_count': 0,
            'has_recent_news': False,
            'bonus': 0
        }
        
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if news and len(news) > 0:
                # Count news in last 48 hours
                cutoff = datetime.now() - timedelta(hours=48)
                recent_count = 0
                
                for item in news[:10]:  # Check first 10
                    pub_time = item.get('providerPublishTime', 0)
                    if pub_time:
                        pub_date = datetime.fromtimestamp(pub_time)
                        if pub_date > cutoff:
                            recent_count += 1
                
                result['news_count'] = recent_count
                result['has_recent_news'] = recent_count > 0
                
                # Bonus for recent news activity
                if recent_count >= 5:
                    result['bonus'] = 5  # High news activity
                elif recent_count >= 3:
                    result['bonus'] = 3  # Moderate activity
                elif recent_count >= 1:
                    result['bonus'] = 1  # Some activity
            
            return result
            
        except Exception as e:
            logger.debug(f"Error getting news sentiment for {ticker}: {e}")
            return result
    
    @classmethod
    def get_all_catalysts(cls, ticker: str) -> Dict:
        """
        Get all catalyst data for a ticker.
        
        Returns combined dict with total bonus.
        """
        short_data = cls.get_short_interest_data(ticker)
        insider_data = cls.get_insider_activity(ticker)
        news_data = cls.get_news_sentiment(ticker)
        
        total_bonus = short_data['bonus'] + insider_data['bonus'] + news_data['bonus']
        
        return {
            'short_interest': short_data,
            'insider': insider_data,
            'news': news_data,
            'total_catalyst_bonus': total_bonus,
            'has_catalyst': total_bonus > 0
        }


def get_catalyst_bonus(ticker: str) -> int:
    """
    Convenience function to get total catalyst bonus.
    
    Usage in scoring.py:
        from swing_trader.small_cap.catalysts import get_catalyst_bonus
        bonus += get_catalyst_bonus(ticker)
    """
    result = CatalystDetector.get_all_catalysts(ticker)
    return result['total_catalyst_bonus']


def is_squeeze_candidate(ticker: str) -> bool:
    """Check if ticker is a short squeeze candidate."""
    short_data = CatalystDetector.get_short_interest_data(ticker)
    return short_data['is_squeeze_candidate']
