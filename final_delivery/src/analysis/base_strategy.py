"""
Base strategy class for technical analysis strategies.
"""

import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from ..data.data_pipeline import DataPipeline
from .config import SIGNAL_TYPES, SIGNAL_STRENGTH

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/technical_analysis.log',
    filemode='a'
)
logger = logging.getLogger('base_strategy')

class BaseStrategy(ABC):
    """
    Abstract base class for all technical analysis strategies.
    Defines the interface that all strategy classes must implement.
    """
    
    def __init__(self, name: str, strategy_type: str, data_pipeline: DataPipeline):
        """
        Initialize the BaseStrategy.
        
        Args:
            name: Strategy name
            strategy_type: Type of strategy (e.g., TREND_FOLLOWING, MEAN_REVERSION)
            data_pipeline: DataPipeline instance for accessing market data
        """
        self.name = name
        self.strategy_type = strategy_type
        self.data_pipeline = data_pipeline
        self.performance_metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_return': 0.0,
            'return_volatility': 0.0,
            'num_trades': 0,
            'avg_trade_duration': 0.0
        }
        self.signals_history = []
        
        logger.info(f"Initialized {name} strategy of type {strategy_type}")
    
    @abstractmethod
    def generate_signals(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Generate trading signals for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            
        Returns:
            DataFrame with trading signals
        """
        pass
    
    def calculate_signal_strength(self, conditions: Dict[str, bool], weights: Dict[str, float]) -> float:
        """
        Calculate signal strength based on multiple conditions.
        
        Args:
            conditions: Dictionary of condition names and boolean values
            weights: Dictionary of condition names and their weights
            
        Returns:
            Signal strength as a float between 0 and 1
        """
        if not conditions or not weights:
            return 0.0
        
        total_weight = sum(weights.values())
        weighted_sum = sum(weights[cond] for cond, value in conditions.items() if value and cond in weights)
        
        if total_weight == 0:
            return 0.0
        
        return min(1.0, weighted_sum / total_weight)
    
    def apply_filters(self, signals_df: pd.DataFrame, filters: Dict[str, callable]) -> pd.DataFrame:
        """
        Apply filters to trading signals.
        
        Args:
            signals_df: DataFrame with trading signals
            filters: Dictionary of filter names and filter functions
            
        Returns:
            DataFrame with filtered trading signals
        """
        filtered_df = signals_df.copy()
        
        for filter_name, filter_func in filters.items():
            try:
                filtered_df = filter_func(filtered_df)
                logger.debug(f"Applied {filter_name} filter")
            except Exception as e:
                logger.error(f"Error applying {filter_name} filter: {str(e)}")
        
        return filtered_df
    
    def update_performance_metrics(self, trades: pd.DataFrame):
        """
        Update strategy performance metrics based on completed trades.
        
        Args:
            trades: DataFrame with completed trades
        """
        if trades.empty:
            logger.warning("No trades provided for performance update")
            return
        
        try:
            # Calculate win rate
            winning_trades = trades[trades['profit'] > 0]
            self.performance_metrics['win_rate'] = len(winning_trades) / len(trades) if len(trades) > 0 else 0.0
            
            # Calculate profit factor
            gross_profit = winning_trades['profit'].sum() if not winning_trades.empty else 0.0
            losing_trades = trades[trades['profit'] < 0]
            gross_loss = abs(losing_trades['profit'].sum()) if not losing_trades.empty else 0.0
            self.performance_metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate average return
            self.performance_metrics['avg_return'] = trades['profit'].mean() if not trades.empty else 0.0
            
            # Calculate return volatility
            self.performance_metrics['return_volatility'] = trades['profit'].std() if len(trades) > 1 else 0.0
            
            # Calculate Sharpe ratio (simplified)
            if self.performance_metrics['return_volatility'] > 0:
                self.performance_metrics['sharpe_ratio'] = (
                    self.performance_metrics['avg_return'] / self.performance_metrics['return_volatility']
                )
            else:
                self.performance_metrics['sharpe_ratio'] = 0.0
            
            # Calculate max drawdown
            if 'cumulative_profit' in trades.columns:
                cumulative_profit = trades['cumulative_profit']
                max_drawdown = 0.0
                peak = cumulative_profit.iloc[0]
                
                for value in cumulative_profit:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak if peak > 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)
                
                self.performance_metrics['max_drawdown'] = max_drawdown
            
            # Update trade count
            self.performance_metrics['num_trades'] = len(trades)
            
            # Calculate average trade duration
            if 'duration' in trades.columns:
                self.performance_metrics['avg_trade_duration'] = trades['duration'].mean()
            
            logger.info(f"Updated performance metrics for {self.name} strategy")
            logger.debug(f"Performance metrics: {self.performance_metrics}")
        
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    def get_performance_score(self) -> float:
        """
        Calculate an overall performance score for the strategy.
        
        Returns:
            Performance score as a float between 0 and 1
        """
        try:
            # Define weights for each metric
            weights = {
                'win_rate': 0.25,
                'profit_factor': 0.25,
                'sharpe_ratio': 0.2,
                'max_drawdown': 0.15,
                'num_trades': 0.15
            }
            
            # Normalize metrics to 0-1 scale
            normalized_metrics = {}
            
            # Win rate is already 0-1
            normalized_metrics['win_rate'] = self.performance_metrics['win_rate']
            
            # Profit factor: 0 is bad, 2+ is good
            profit_factor = min(2.0, self.performance_metrics['profit_factor'])
            normalized_metrics['profit_factor'] = profit_factor / 2.0
            
            # Sharpe ratio: 0 is bad, 3+ is good
            sharpe_ratio = min(3.0, max(0.0, self.performance_metrics['sharpe_ratio']))
            normalized_metrics['sharpe_ratio'] = sharpe_ratio / 3.0
            
            # Max drawdown: 0 is good, 0.5+ is bad
            max_drawdown = min(0.5, self.performance_metrics['max_drawdown'])
            normalized_metrics['max_drawdown'] = 1.0 - (max_drawdown / 0.5)
            
            # Number of trades: 0 is bad, 100+ is good
            num_trades = min(100, self.performance_metrics['num_trades'])
            normalized_metrics['num_trades'] = num_trades / 100.0
            
            # Calculate weighted score
            score = sum(normalized_metrics[metric] * weight for metric, weight in weights.items())
            
            return score
        
        except Exception as e:
            logger.error(f"Error calculating performance score: {str(e)}")
            return 0.0
    
    def record_signal(self, timestamp: int, symbol: str, timeframe: str, 
                     signal_type: int, signal_strength: float, price: float):
        """
        Record a trading signal for performance tracking.
        
        Args:
            timestamp: Signal timestamp
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            signal_type: Type of signal (from SIGNAL_TYPES)
            signal_strength: Strength of signal (0-1)
            price: Price at signal generation
        """
        signal = {
            'timestamp': timestamp,
            'symbol': symbol,
            'timeframe': timeframe,
            'strategy': self.name,
            'signal_type': signal_type,
            'signal_strength': signal_strength,
            'price': price
        }
        
        self.signals_history.append(signal)
        logger.debug(f"Recorded signal: {signal}")
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.name} ({self.strategy_type})"
    
    def __repr__(self) -> str:
        """Detailed representation of the strategy."""
        return f"Strategy(name='{self.name}', type='{self.strategy_type}', performance={self.get_performance_score():.2f})"
