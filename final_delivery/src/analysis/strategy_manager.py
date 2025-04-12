"""
Strategy manager for combining and managing multiple trading strategies.
"""

import logging
import pandas as pd
import numpy as np
import sqlite3
import os
from typing import Dict, List, Optional, Tuple, Union

from .base_strategy import BaseStrategy
from .trend_following_strategy import TrendFollowingStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .breakout_strategy import BreakoutStrategy
from ..data.data_pipeline import DataPipeline
from .config import (
    SIGNAL_TYPES, SIGNAL_STRENGTH, STRATEGY_TYPES, STRATEGY_WEIGHTS,
    TIMEFRAME_WEIGHTS, SIGNAL_COMBINATION, MARKET_REGIME, DATABASE
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/technical_analysis.log',
    filemode='a'
)
logger = logging.getLogger('strategy_manager')

class StrategyManager:
    """
    Manager class for combining and managing multiple trading strategies.
    Handles strategy initialization, signal combination, and performance tracking.
    """
    
    def __init__(self, data_pipeline: DataPipeline, db_path: str = None):
        """
        Initialize the StrategyManager.
        
        Args:
            data_pipeline: DataPipeline instance for accessing market data
            db_path: Path to the SQLite database file
        """
        self.data_pipeline = data_pipeline
        self.db_path = db_path or '/home/ubuntu/crypto_trading_model/data/market_data.db'
        self.strategies = {}
        self.strategy_weights = STRATEGY_WEIGHTS.copy()
        self.timeframe_weights = TIMEFRAME_WEIGHTS.copy()
        self.current_market_regime = {}
        
        # Initialize database
        self._init_database()
        
        # Initialize strategies
        self._init_strategies()
        
        logger.info("StrategyManager initialized")
    
    def _init_database(self):
        """Initialize the SQLite database with required tables if they don't exist."""
        try:
            # Ensure database directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create signals table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['signals_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                signal_type INTEGER NOT NULL,
                signal_strength REAL NOT NULL,
                price REAL NOT NULL,
                combined_signal INTEGER,
                combined_strength REAL,
                executed BOOLEAN DEFAULT 0,
                profit REAL,
                UNIQUE(timestamp, symbol, timeframe, strategy)
            )
            ''')
            
            # Create strategy performance table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['strategies_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                win_rate REAL,
                profit_factor REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                avg_return REAL,
                return_volatility REAL,
                num_trades INTEGER,
                avg_trade_duration REAL,
                performance_score REAL,
                weight REAL,
                UNIQUE(timestamp, strategy)
            )
            ''')
            
            # Create market regimes table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['market_regimes_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                regime TEXT NOT NULL,
                volatility REAL,
                trend_strength REAL,
                UNIQUE(timestamp, symbol, timeframe)
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database tables initialized")
        
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def _init_strategies(self):
        """Initialize all trading strategies."""
        try:
            # Initialize trend following strategy
            self.strategies['TREND_FOLLOWING'] = TrendFollowingStrategy(self.data_pipeline)
            
            # Initialize mean reversion strategy
            self.strategies['MEAN_REVERSION'] = MeanReversionStrategy(self.data_pipeline)
            
            # Initialize breakout strategy
            self.strategies['BREAKOUT'] = BreakoutStrategy(self.data_pipeline)
            
            # Additional strategies can be added here
            
            logger.info(f"Initialized {len(self.strategies)} trading strategies")
        
        except Exception as e:
            logger.error(f"Error initializing strategies: {str(e)}")
    
    def detect_market_regime(self, symbol: str, timeframe: str) -> str:
        """
        Detect the current market regime for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            
        Returns:
            Market regime type (e.g., TRENDING_UP, RANGING, VOLATILE)
        """
        try:
            # Get data from pipeline
            data = self.data_pipeline.get_latest_data(symbol, timeframe)
            df = data['complete']
            
            if df.empty:
                logger.warning(f"No data available for {symbol} {timeframe}")
                return "UNKNOWN"
            
            # Calculate volatility
            volatility_lookback = MARKET_REGIME['volatility_lookback']
            returns = df['close'].pct_change()
            volatility = returns.rolling(volatility_lookback).std().iloc[-1]
            
            # Calculate trend strength
            trend_lookback = MARKET_REGIME['trend_lookback']
            
            # Use ADX if available
            if 'ADX_14' in df.columns:
                trend_strength = df['ADX_14'].iloc[-1] / 100.0  # Normalize to 0-1
            else:
                # Alternative trend strength calculation
                price_direction = np.sign(df['close'].diff())
                trend_consistency = abs(price_direction.rolling(trend_lookback).sum()) / trend_lookback
                trend_strength = trend_consistency.iloc[-1]
            
            # Determine price direction
            price_direction = 1 if df['close'].iloc[-1] > df['close'].iloc[-trend_lookback] else -1
            
            # Determine regime
            if volatility > 0.03:  # High volatility threshold
                regime = "VOLATILE"
            elif trend_strength > 0.6:  # Strong trend threshold
                regime = "TRENDING_UP" if price_direction > 0 else "TRENDING_DOWN"
            else:
                regime = "RANGING"
            
            # Check for breakout
            if 'BB_WIDTH' in df.columns:
                bb_width = df['BB_WIDTH'].iloc[-1]
                bb_width_avg = df['BB_WIDTH'].rolling(20).mean().iloc[-1]
                
                if bb_width > bb_width_avg * 1.5:  # Significant expansion in Bollinger Band width
                    regime = "BREAKOUT"
            
            # Store regime information
            self.current_market_regime[f"{symbol}_{timeframe}"] = {
                'regime': regime,
                'volatility': volatility,
                'trend_strength': trend_strength
            }
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
            INSERT OR REPLACE INTO {DATABASE['market_regimes_table']}
            (timestamp, symbol, timeframe, regime, volatility, trend_strength)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                int(df['timestamp'].iloc[-1]),
                symbol,
                timeframe,
                regime,
                volatility,
                trend_strength
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Detected {regime} regime for {symbol} {timeframe}")
            return regime
        
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return "UNKNOWN"
    
    def adjust_strategy_weights(self, symbol: str, timeframe: str):
        """
        Adjust strategy weights based on the current market regime.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
        """
        try:
            # Detect market regime
            regime = self.detect_market_regime(symbol, timeframe)
            
            if regime == "UNKNOWN":
                return
            
            # Get regime-specific weights
            regime_weights = MARKET_REGIME['regime_types'].get(regime, {})
            
            if not regime_weights:
                return
            
            # Adjust weights based on regime
            for strategy_type in self.strategies.keys():
                if strategy_type in ['TREND_FOLLOWING', 'BREAKOUT']:
                    # Trend strategies get higher weight in trending markets
                    self.strategy_weights[strategy_type] = regime_weights.get('weight_trend', 0.5)
                elif strategy_type in ['MEAN_REVERSION']:
                    # Mean reversion strategies get higher weight in ranging markets
                    self.strategy_weights[strategy_type] = regime_weights.get('weight_mean_reversion', 0.5)
            
            # Normalize weights to sum to 1
            total_weight = sum(self.strategy_weights.values())
            if total_weight > 0:
                for strategy_type in self.strategy_weights:
                    self.strategy_weights[strategy_type] /= total_weight
            
            logger.info(f"Adjusted strategy weights for {regime} regime: {self.strategy_weights}")
        
        except Exception as e:
            logger.error(f"Error adjusting strategy weights: {str(e)}")
    
    def generate_signals(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Generate trading signals from all strategies and combine them.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            
        Returns:
            DataFrame with combined trading signals
        """
        try:
            # Adjust strategy weights based on market regime
            self.adjust_strategy_weights(symbol, timeframe)
            
            # Generate signals from each strategy
            strategy_signals = {}
            for strategy_name, strategy in self.strategies.items():
                signals_df = strategy.generate_signals(symbol, timeframe)
                if not signals_df.empty:
                    strategy_signals[strategy_name] = signals_df
            
            if not strategy_signals:
                logger.warning(f"No signals generated for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Get the base dataframe from the first strategy
            first_strategy = list(strategy_signals.keys())[0]
            combined_df = strategy_signals[first_strategy][['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            combined_df['combined_signal'] = SIGNAL_TYPES['NONE']
            combined_df['combined_strength'] = 0.0
            
            # Add individual strategy signals and strengths
            for strategy_name, signals_df in strategy_signals.items():
                combined_df[f'{strategy_name}_signal'] = signals_df['signal']
                combined_df[f'{strategy_name}_strength'] = signals_df['signal_strength']
            
            # Combine signals
            for i in range(len(combined_df)):
                # Count strategies agreeing on long signal
                long_count = 0
                long_strength = 0.0
                long_weight_sum = 0.0
                
                # Count strategies agreeing on short signal
                short_count = 0
                short_strength = 0.0
                short_weight_sum = 0.0
                
                # Process each strategy's signal
                for strategy_name in strategy_signals.keys():
                    signal = combined_df[f'{strategy_name}_signal'].iloc[i]
                    strength = combined_df[f'{strategy_name}_strength'].iloc[i]
                    weight = self.strategy_weights.get(strategy_name, 0.0)
                    
                    if signal == SIGNAL_TYPES['ENTRY']['LONG']:
                        long_count += 1
                        long_strength += strength * weight
                        long_weight_sum += weight
                    elif signal == SIGNAL_TYPES['ENTRY']['SHORT']:
                        short_count += 1
                        short_strength += strength * weight
                        short_weight_sum += weight
                
                # Calculate agreement percentages
                strategy_count = len(strategy_signals)
                long_agreement = long_count / strategy_count if strategy_count > 0 else 0
                short_agreement = short_count / strategy_count if strategy_count > 0 else 0
                
                # Calculate weighted average strengths
                long_avg_strength = long_strength / long_weight_sum if long_weight_sum > 0 else 0
                short_avg_strength = short_strength / short_weight_sum if short_weight_sum > 0 else 0
                
                # Determine combined signal
                if (long_agreement >= SIGNAL_COMBINATION['strategy_agreement_threshold'] and 
                    long_avg_strength >= SIGNAL_COMBINATION['min_signal_strength'] and
                    long_avg_strength > short_avg_strength):
                    
                    combined_df.loc[combined_df.index[i], 'combined_signal'] = SIGNAL_TYPES['ENTRY']['LONG']
                    combined_df.loc[combined_df.index[i], 'combined_strength'] = long_avg_strength
                    
                elif (short_agreement >= SIGNAL_COMBINATION['strategy_agreement_threshold'] and 
                      short_avg_strength >= SIGNAL_COMBINATION['min_signal_strength'] and
                      short_avg_strength > long_avg_strength):
                    
                    combined_df.loc[combined_df.index[i], 'combined_signal'] = SIGNAL_TYPES['ENTRY']['SHORT']
                    combined_df.loc[combined_df.index[i], 'combined_strength'] = short_avg_strength
            
            # Generate exit signals
            for i in range(1, len(combined_df)):
                # Exit long position
                if combined_df['combined_signal'].iloc[i-1] == SIGNAL_TYPES['ENTRY']['LONG']:
                    # Check if any strategy suggests exiting
                    exit_signals = []
                    for strategy_name in strategy_signals.keys():
                        if combined_df[f'{strategy_name}_signal'].iloc[i] == SIGNAL_TYPES['EXIT']['LONG']:
                            exit_signals.append(combined_df[f'{strategy_name}_strength'].iloc[i])
                    
                    if exit_signals:
                        # Use the strongest exit signal
                        exit_strength = max(exit_signals)
                        combined_df.loc[combined_df.index[i], 'combined_signal'] = SIGNAL_TYPES['EXIT']['LONG']
                        combined_df.loc[combined_df.index[i], 'combined_strength'] = exit_strength
                
                # Exit short position
                elif combined_df['combined_signal'].iloc[i-1] == SIGNAL_TYPES['ENTRY']['SHORT']:
                    # Check if any strategy suggests exiting
                    exit_signals = []
                    for strategy_name in strategy_signals.keys():
                        if combined_df[f'{strategy_name}_signal'].iloc[i] == SIGNAL_TYPES['EXIT']['SHORT']:
                            exit_signals.append(combined_df[f'{strategy_name}_strength'].iloc[i])
                    
                    if exit_signals:
                        # Use the strongest exit signal
                        exit_strength = max(exit_signals)
                        combined_df.loc[combined_df.index[i], 'combined_signal'] = SIGNAL_TYPES['EXIT']['SHORT']
                        combined_df.loc[combined_df.index[i], 'combined_strength'] = exit_strength
            
            # Save combined signals to database
            self._save_combined_signals(combined_df, symbol, timeframe)
            
            logger.info(f"Generated combined signals for {symbol} {timeframe}")
            return combined_df
        
        except Exception as e:
            logger.error(f"Error generating combined signals: {str(e)}")
            return pd.DataFrame()
    
    def _save_combined_signals(self, combined_df: pd.DataFrame, symbol: str, timeframe: str):
        """
        Save combined signals to the database.
        
        Args:
            combined_df: DataFrame with combined signals
            symbol: Trading pair symbol
            timeframe: Timeframe interval
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get only rows with non-zero signals
            signal_rows = combined_df[combined_df['combined_signal'] != SIGNAL_TYPES['NONE']]
            
            for _, row in signal_rows.iterrows():
                # Update signals from individual strategies
                for strategy_name in self.strategies.keys():
                    if f'{strategy_name}_signal' in row and f'{strategy_name}_strength' in row:
                        signal_type = row[f'{strategy_name}_signal']
                        signal_strength = row[f'{strategy_name}_strength']
                        
                        if signal_type != SIGNAL_TYPES['NONE']:
                            cursor.execute(f'''
                            INSERT OR REPLACE INTO {DATABASE['signals_table']}
                            (timestamp, symbol, timeframe, strategy, signal_type, signal_strength, price, combined_signal, combined_strength)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                int(row['timestamp']),
                                symbol,
                                timeframe,
                                strategy_name,
                                int(signal_type),
                                float(signal_strength),
                                float(row['close']),
                                int(row['combined_signal']),
                                float(row['combined_strength'])
                            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved {len(signal_rows)} combined signals to database")
        
        except Exception as e:
            logger.error(f"Error saving combined signals: {str(e)}")
    
    def update_strategy_performance(self, trades: pd.DataFrame):
        """
        Update performance metrics for all strategies based on completed trades.
        
        Args:
            trades: DataFrame with completed trades
        """
        try:
            if trades.empty:
                logger.warning("No trades provided for performance update")
                return
            
            # Update performance for each strategy
            for strategy_name, strategy in self.strategies.items():
                # Filter trades for this strategy
                strategy_trades = trades[trades['strategy'] == strategy_name]
                
                if not strategy_trades.empty:
                    # Update strategy performance metrics
                    strategy.update_performance_metrics(strategy_trades)
                    
                    # Calculate performance score
                    performance_score = strategy.get_performance_score()
                    
                    # Save performance metrics to database
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute(f'''
                    INSERT INTO {DATABASE['strategies_table']}
                    (timestamp, strategy, win_rate, profit_factor, sharpe_ratio, max_drawdown, 
                     avg_return, return_volatility, num_trades, avg_trade_duration, performance_score, weight)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        int(time.time() * 1000),
                        strategy_name,
                        strategy.performance_metrics['win_rate'],
                        strategy.performance_metrics['profit_factor'],
                        strategy.performance_metrics['sharpe_ratio'],
                        strategy.performance_metrics['max_drawdown'],
                        strategy.performance_metrics['avg_return'],
                        strategy.performance_metrics['return_volatility'],
                        strategy.performance_metrics['num_trades'],
                        strategy.performance_metrics['avg_trade_duration'],
                        performance_score,
                        self.strategy_weights.get(strategy_name, 0.0)
                    ))
                    
                    conn.commit()
                    conn.close()
                    
                    logger.info(f"Updated performance metrics for {strategy_name} strategy")
            
            # Adjust strategy weights based on performance
            self._adjust_weights_by_performance()
        
        except Exception as e:
            logger.error(f"Error updating strategy performance: {str(e)}")
    
    def _adjust_weights_by_performance(self):
        """Adjust strategy weights based on performance scores."""
        try:
            # Get performance scores for all strategies
            performance_scores = {}
            for strategy_name, strategy in self.strategies.items():
                performance_scores[strategy_name] = strategy.get_performance_score()
            
            # Calculate total score
            total_score = sum(performance_scores.values())
            
            if total_score > 0:
                # Calculate new weights based on performance
                new_weights = {strategy: score / total_score for strategy, score in performance_scores.items()}
                
                # Apply gradual adjustment
                update_factor = PERFORMANCE_TRACKING['weight_update_factor']
                for strategy in self.strategy_weights:
                    if strategy in new_weights:
                        self.strategy_weights[strategy] = (
                            (1 - update_factor) * self.strategy_weights[strategy] + 
                            update_factor * new_weights[strategy]
                        )
                
                logger.info(f"Adjusted strategy weights based on performance: {self.strategy_weights}")
        
        except Exception as e:
            logger.error(f"Error adjusting weights by performance: {str(e)}")
    
    def get_multi_timeframe_signals(self, symbol: str, timeframes: List[str] = None) -> Dict:
        """
        Generate signals across multiple timeframes and combine them.
        
        Args:
            symbol: Trading pair symbol
            timeframes: List of timeframes to analyze. If None, uses all configured timeframes.
            
        Returns:
            Dictionary with combined signals for each timeframe and overall recommendation
        """
        try:
            if timeframes is None:
                timeframes = list(self.timeframe_weights.keys())
            
            # Generate signals for each timeframe
            timeframe_signals = {}
            for timeframe in timeframes:
                signals_df = self.generate_signals(symbol, timeframe)
                if not signals_df.empty:
                    # Get the most recent signal
                    latest_signal = signals_df.iloc[-1]
                    timeframe_signals[timeframe] = {
                        'signal': latest_signal['combined_signal'],
                        'strength': latest_signal['combined_strength'],
                        'price': latest_signal['close'],
                        'timestamp': latest_signal['timestamp']
                    }
            
            if not timeframe_signals:
                logger.warning(f"No signals generated for {symbol} across timeframes")
                return {}
            
            # Count signals by type
            long_count = 0
            long_strength = 0.0
            long_weight_sum = 0.0
            
            short_count = 0
            short_strength = 0.0
            short_weight_sum = 0.0
            
            # Process each timeframe's signal
            for timeframe, signal_data in timeframe_signals.items():
                signal = signal_data['signal']
                strength = signal_data['strength']
                weight = self.timeframe_weights.get(timeframe, 0.0)
                
                if signal == SIGNAL_TYPES['ENTRY']['LONG']:
                    long_count += 1
                    long_strength += strength * weight
                    long_weight_sum += weight
                elif signal == SIGNAL_TYPES['ENTRY']['SHORT']:
                    short_count += 1
                    short_strength += strength * weight
                    short_weight_sum += weight
            
            # Calculate agreement percentages
            timeframe_count = len(timeframe_signals)
            long_agreement = long_count / timeframe_count if timeframe_count > 0 else 0
            short_agreement = short_count / timeframe_count if timeframe_count > 0 else 0
            
            # Calculate weighted average strengths
            long_avg_strength = long_strength / long_weight_sum if long_weight_sum > 0 else 0
            short_avg_strength = short_strength / short_weight_sum if short_weight_sum > 0 else 0
            
            # Determine overall recommendation
            recommendation = {
                'symbol': symbol,
                'timestamp': int(time.time() * 1000),
                'timeframes_analyzed': list(timeframe_signals.keys()),
                'signal': SIGNAL_TYPES['NONE'],
                'strength': 0.0,
                'confidence': 0.0,
                'timeframe_signals': timeframe_signals
            }
            
            if (long_agreement >= SIGNAL_COMBINATION['timeframe_agreement_threshold'] and 
                long_avg_strength >= SIGNAL_COMBINATION['min_signal_strength'] and
                long_avg_strength > short_avg_strength):
                
                recommendation['signal'] = SIGNAL_TYPES['ENTRY']['LONG']
                recommendation['strength'] = long_avg_strength
                recommendation['confidence'] = long_agreement
                
            elif (short_agreement >= SIGNAL_COMBINATION['timeframe_agreement_threshold'] and 
                  short_avg_strength >= SIGNAL_COMBINATION['min_signal_strength'] and
                  short_avg_strength > long_avg_strength):
                
                recommendation['signal'] = SIGNAL_TYPES['ENTRY']['SHORT']
                recommendation['strength'] = short_avg_strength
                recommendation['confidence'] = short_agreement
            
            logger.info(f"Generated multi-timeframe signals for {symbol}")
            return recommendation
        
        except Exception as e:
            logger.error(f"Error generating multi-timeframe signals: {str(e)}")
            return {}
    
    def get_signals_for_all_symbols(self, timeframe: str = '1h') -> Dict:
        """
        Generate signals for all configured symbols at a specific timeframe.
        
        Args:
            timeframe: Timeframe to analyze
            
        Returns:
            Dictionary with signals for each symbol
        """
        try:
            from ..data.config import TRADING_PAIRS
            
            all_signals = {}
            for symbol in TRADING_PAIRS:
                signals_df = self.generate_signals(symbol, timeframe)
                if not signals_df.empty:
                    # Get the most recent signal
                    latest_signal = signals_df.iloc[-1]
                    all_signals[symbol] = {
                        'signal': latest_signal['combined_signal'],
                        'strength': latest_signal['combined_strength'],
                        'price': latest_signal['close'],
                        'timestamp': latest_signal['timestamp']
                    }
            
            logger.info(f"Generated signals for {len(all_signals)} symbols at {timeframe} timeframe")
            return all_signals
        
        except Exception as e:
            logger.error(f"Error generating signals for all symbols: {str(e)}")
            return {}
