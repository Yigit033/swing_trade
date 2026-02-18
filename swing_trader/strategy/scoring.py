"""
Signal scoring module for ranking trading opportunities.
"""

import logging
from typing import List, Dict
import pandas as pd

logger = logging.getLogger(__name__)


class SignalScorer:
    """
    Scores and ranks trading signals.
    
    Attributes:
        config (Dict): Configuration dictionary
    """
    
    def __init__(self, config: Dict):
        """
        Initialize SignalScorer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        logger.info("SignalScorer initialized")
    
    def score_signal(self, signal: Dict) -> float:
        """
        Calculate a comprehensive score for a trading signal.
        
        Score components:
        - Base score from signal generation (0-10)
        - Risk/reward ratio adjustment
        - Trend strength adjustment
        - Volume surge adjustment
        
        Args:
            signal: Signal dictionary
        
        Returns:
            Comprehensive score (0-100)
        
        Example:
            >>> scorer = SignalScorer(config)
            >>> final_score = scorer.score_signal(signal)
            >>> print(f"Final score: {final_score:.1f}/100")
        """
        try:
            # Start with base score (0-10)
            base_score = signal.get('score', 0) * 10  # Convert to 0-100 scale
            
            # Adjust based on technical strength
            adjustments = 0
            
            # RSI in ideal range (35-45)
            rsi = signal.get('rsi', 50)
            if 35 <= rsi <= 45:
                adjustments += 5
            
            # Strong ADX (trend strength)
            adx = signal.get('adx', 0)
            if adx > 25:
                adjustments += 10
            elif adx < 20:
                adjustments -= 10
            
            # High volume surge
            volume_surge = signal.get('volume_surge', 1.0)
            if volume_surge > 1.5:
                adjustments += 10
            elif volume_surge > 1.3:
                adjustments += 5
            
            # Strong MACD
            macd_hist = signal.get('macd_hist', 0)
            if macd_hist > 0.5:
                adjustments += 5
            
            # Calculate final score
            final_score = base_score + adjustments
            
            # Clamp to 0-100
            final_score = max(0, min(100, final_score))
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error scoring signal: {e}", exc_info=True)
            return 0.0
    
    def rank_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Rank signals by score.
        
        Args:
            signals: List of signal dictionaries
        
        Returns:
            List of signals sorted by score (highest first)
        
        Example:
            >>> scorer = SignalScorer(config)
            >>> ranked = scorer.rank_signals(signals)
            >>> print(f"Top signal: {ranked[0]['ticker']} (score: {ranked[0]['final_score']:.1f})")
        """
        if not signals:
            return []
        
        try:
            # Calculate comprehensive score for each signal
            for signal in signals:
                signal['final_score'] = self.score_signal(signal)
            
            # Sort by score (descending)
            ranked = sorted(signals, key=lambda x: x['final_score'], reverse=True)
            
            logger.info(f"Ranked {len(ranked)} signals")
            return ranked
            
        except Exception as e:
            logger.error(f"Error ranking signals: {e}", exc_info=True)
            return signals
    
    def filter_signals_by_score(self, signals: List[Dict], min_score: float = 60) -> List[Dict]:
        """
        Filter signals by minimum score.
        
        Args:
            signals: List of signal dictionaries
            min_score: Minimum final score to keep (0-100)
        
        Returns:
            Filtered list of signals
        
        Example:
            >>> scorer = SignalScorer(config)
            >>> high_quality = scorer.filter_signals_by_score(signals, min_score=70)
        """
        filtered = [s for s in signals if s.get('final_score', 0) >= min_score]
        logger.info(f"Filtered {len(signals)} signals -> {len(filtered)} above score {min_score}")
        return filtered
    
    def get_top_signals(self, signals: List[Dict], top_n: int = 10) -> List[Dict]:
        """
        Get top N signals by score.
        
        Args:
            signals: List of signal dictionaries
            top_n: Number of top signals to return
        
        Returns:
            List of top N signals
        
        Example:
            >>> scorer = SignalScorer(config)
            >>> top_10 = scorer.get_top_signals(signals, top_n=10)
        """
        ranked = self.rank_signals(signals)
        return ranked[:top_n]
    
    def create_signals_dataframe(self, signals: List[Dict]) -> pd.DataFrame:
        """
        Convert signals to pandas DataFrame for easy analysis.
        
        Args:
            signals: List of signal dictionaries
        
        Returns:
            DataFrame with signal data
        
        Example:
            >>> scorer = SignalScorer(config)
            >>> df = scorer.create_signals_dataframe(signals)
            >>> print(df[['ticker', 'final_score', 'entry_price']].head())
        """
        if not signals:
            return pd.DataFrame()
        
        try:
            # Extract key fields
            data = []
            for signal in signals:
                row = {
                    'ticker': signal.get('ticker', ''),
                    'date': signal.get('date', ''),
                    'score': signal.get('score', 0),
                    'final_score': signal.get('final_score', 0),
                    'entry_price': signal.get('entry_price', 0),
                    'stop_loss': signal.get('stop_loss', 0),
                    'target_1': signal.get('target_1', 0),
                    'target_2': signal.get('target_2', 0),
                    'rsi': signal.get('rsi', 0),
                    'macd_hist': signal.get('macd_hist', 0),
                    'adx': signal.get('adx', 0),
                    'volume_surge': signal.get('volume_surge', 0),
                    'atr': signal.get('atr', 0),
                    # Time-based fields
                    'expected_hold_min': signal.get('expected_hold_min', 5),
                    'expected_hold_max': signal.get('expected_hold_max', 15),
                    'expiration_date': signal.get('expiration_date', ''),
                    'max_hold_date': signal.get('max_hold_date', ''),
                    # SmallCap-specific fields
                    'swing_type': signal.get('swing_type', ''),
                    'quality_score': signal.get('quality_score', 0),
                    'atr_percent': signal.get('atr_percent', 0),
                    'float_shares': signal.get('float_shares', 0),
                    'short_percent': signal.get('short_percent', 0),
                    'sector_rs_score': signal.get('sector_rs_score', 0),
                    'has_catalyst': signal.get('has_catalyst', False),
                    'catalyst_emoji': signal.get('catalyst_emoji', ''),
                    'type_reason': signal.get('type_reason', ''),
                    'narrative_headline': signal.get('narrative_headline', ''),
                    'narrative_text': signal.get('narrative_text', ''),
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Sort by final score
            df = df.sort_values('final_score', ascending=False).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}", exc_info=True)
            return pd.DataFrame()

