"""
Mean reversion strategy implementation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from .base_strategy import BaseStrategy
from ..data.data_pipeline import DataPipeline
from .config import SIGNAL_TYPES, SIGNAL_STRENGTH, MEAN_REVERSION

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/technical_analysis.log',
    filemode='a'
)
logger = logging.getLogger('mean_reversion_strategy')

class MeanReversionStrategy(BaseStrategy):
    """
    Strategy that trades based on the principle that prices tend to revert to their mean.
    Uses RSI, Bollinger Bands, and Stochastic oscillators to identify overbought and oversold conditions.
    """
    
    def __init__(self, data_pipeline: DataPipeline, config: Dict = None):
        """
        Initialize the MeanReversionStrategy.
        
        Args:
            data_pipeline: DataPipeline instance for accessing market data
            config: Optional custom configuration (defaults to MEAN_REVERSION from config)
        """
        super().__init__("Mean Reversion", "MEAN_REVERSION", data_pipeline)
        self.config = config or MEAN_REVERSION
        logger.info(f"Initialized Mean Reversion strategy with config: {self.config}")
    
    def generate_signals(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Generate trading signals based on mean reversion indicators.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            
        Returns:
            DataFrame with trading signals
        """
        try:
            # Get data from pipeline
            data = self.data_pipeline.get_latest_data(symbol, timeframe)
            df = data['complete']
            
            if df.empty:
                logger.warning(f"No data available for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Create a copy of the dataframe for signals
            signals_df = df.copy()
            signals_df['signal'] = SIGNAL_TYPES['NONE']
            signals_df['signal_strength'] = 0.0
            
            # Check if required columns exist
            rsi_col = f"RSI_{self.config['rsi_period']}"
            bb_upper_col = 'BB_Upper'
            bb_middle_col = 'BB_Middle'
            bb_lower_col = 'BB_Lower'
            stoch_k_col = 'STOCH_K'
            stoch_d_col = 'STOCH_D'
            
            required_columns = [rsi_col, bb_upper_col, bb_middle_col, bb_lower_col, stoch_k_col, stoch_d_col]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns for mean reversion strategy: {missing_columns}")
                return signals_df
            
            # Calculate mean reversion conditions
            # 1. RSI conditions
            signals_df['rsi_oversold'] = df[rsi_col] < self.config['rsi_oversold']
            signals_df['rsi_overbought'] = df[rsi_col] > self.config['rsi_overbought']
            signals_df['rsi_exiting_oversold'] = (df[rsi_col] > self.config['rsi_oversold']) & (df[rsi_col].shift(1) <= self.config['rsi_oversold'])
            signals_df['rsi_exiting_overbought'] = (df[rsi_col] < self.config['rsi_overbought']) & (df[rsi_col].shift(1) >= self.config['rsi_overbought'])
            
            # 2. Bollinger Band conditions
            signals_df['price_below_lower_band'] = df['close'] < df[bb_lower_col]
            signals_df['price_above_upper_band'] = df['close'] > df[bb_upper_col]
            signals_df['price_crossing_lower_band'] = (df['close'] > df[bb_lower_col]) & (df['close'].shift(1) <= df[bb_lower_col])
            signals_df['price_crossing_upper_band'] = (df['close'] < df[bb_upper_col]) & (df['close'].shift(1) >= df[bb_upper_col])
            signals_df['price_reverting_to_mean'] = (
                (df['close'] > df[bb_middle_col]) & (df['close'].shift(1) < df[bb_middle_col]) |
                (df['close'] < df[bb_middle_col]) & (df['close'].shift(1) > df[bb_middle_col])
            )
            
            # 3. Stochastic conditions
            signals_df['stoch_oversold'] = df[stoch_k_col] < self.config['stoch_oversold']
            signals_df['stoch_overbought'] = df[stoch_k_col] > self.config['stoch_overbought']
            signals_df['stoch_k_crossing_d_up'] = (df[stoch_k_col] > df[stoch_d_col]) & (df[stoch_k_col].shift(1) <= df[stoch_d_col].shift(1))
            signals_df['stoch_k_crossing_d_down'] = (df[stoch_k_col] < df[stoch_d_col]) & (df[stoch_k_col].shift(1) >= df[stoch_d_col].shift(1))
            
            # 4. Price deviation from mean
            signals_df['price_deviation'] = (df['close'] - df['close'].rolling(self.config['mean_period']).mean()) / df['close'].rolling(self.config['mean_period']).std()
            signals_df['extreme_deviation_low'] = signals_df['price_deviation'] < -2.0
            signals_df['extreme_deviation_high'] = signals_df['price_deviation'] > 2.0
            signals_df['reverting_from_low'] = (signals_df['price_deviation'] > signals_df['price_deviation'].shift(1)) & (signals_df['extreme_deviation_low'].shift(1))
            signals_df['reverting_from_high'] = (signals_df['price_deviation'] < signals_df['price_deviation'].shift(1)) & (signals_df['extreme_deviation_high'].shift(1))
            
            # Generate long entry signals (buy when oversold and showing signs of reversal)
            long_entry_conditions = {
                'rsi_oversold': signals_df['rsi_oversold'],
                'rsi_exiting_oversold': signals_df['rsi_exiting_oversold'],
                'price_below_lower_band': signals_df['price_below_lower_band'],
                'price_crossing_lower_band': signals_df['price_crossing_lower_band'],
                'stoch_oversold': signals_df['stoch_oversold'],
                'stoch_k_crossing_d_up': signals_df['stoch_k_crossing_d_up'],
                'extreme_deviation_low': signals_df['extreme_deviation_low'],
                'reverting_from_low': signals_df['reverting_from_low']
            }
            
            long_entry_weights = {
                'rsi_oversold': 0.15,
                'rsi_exiting_oversold': 0.2,
                'price_below_lower_band': 0.1,
                'price_crossing_lower_band': 0.15,
                'stoch_oversold': 0.1,
                'stoch_k_crossing_d_up': 0.15,
                'extreme_deviation_low': 0.05,
                'reverting_from_low': 0.1
            }
            
            # Generate short entry signals (sell when overbought and showing signs of reversal)
            short_entry_conditions = {
                'rsi_overbought': signals_df['rsi_overbought'],
                'rsi_exiting_overbought': signals_df['rsi_exiting_overbought'],
                'price_above_upper_band': signals_df['price_above_upper_band'],
                'price_crossing_upper_band': signals_df['price_crossing_upper_band'],
                'stoch_overbought': signals_df['stoch_overbought'],
                'stoch_k_crossing_d_down': signals_df['stoch_k_crossing_d_down'],
                'extreme_deviation_high': signals_df['extreme_deviation_high'],
                'reverting_from_high': signals_df['reverting_from_high']
            }
            
            short_entry_weights = {
                'rsi_overbought': 0.15,
                'rsi_exiting_overbought': 0.2,
                'price_above_upper_band': 0.1,
                'price_crossing_upper_band': 0.15,
                'stoch_overbought': 0.1,
                'stoch_k_crossing_d_down': 0.15,
                'extreme_deviation_high': 0.05,
                'reverting_from_high': 0.1
            }
            
            # Calculate signal strengths for each row
            for i in range(len(signals_df)):
                # Extract conditions for current row
                current_long_conditions = {k: v.iloc[i] for k, v in long_entry_conditions.items()}
                current_short_conditions = {k: v.iloc[i] for k, v in short_entry_conditions.items()}
                
                # Calculate signal strengths
                long_strength = self.calculate_signal_strength(current_long_conditions, long_entry_weights)
                short_strength = self.calculate_signal_strength(current_short_conditions, short_entry_weights)
                
                # Determine signal based on strengths
                if long_strength > 0.6 and long_strength > short_strength:
                    signals_df.loc[signals_df.index[i], 'signal'] = SIGNAL_TYPES['ENTRY']['LONG']
                    signals_df.loc[signals_df.index[i], 'signal_strength'] = long_strength
                    
                    # Record signal for performance tracking
                    self.record_signal(
                        timestamp=signals_df['timestamp'].iloc[i],
                        symbol=symbol,
                        timeframe=timeframe,
                        signal_type=SIGNAL_TYPES['ENTRY']['LONG'],
                        signal_strength=long_strength,
                        price=signals_df['close'].iloc[i]
                    )
                    
                elif short_strength > 0.6 and short_strength > long_strength:
                    signals_df.loc[signals_df.index[i], 'signal'] = SIGNAL_TYPES['ENTRY']['SHORT']
                    signals_df.loc[signals_df.index[i], 'signal_strength'] = short_strength
                    
                    # Record signal for performance tracking
                    self.record_signal(
                        timestamp=signals_df['timestamp'].iloc[i],
                        symbol=symbol,
                        timeframe=timeframe,
                        signal_type=SIGNAL_TYPES['ENTRY']['SHORT'],
                        signal_strength=short_strength,
                        price=signals_df['close'].iloc[i]
                    )
            
            # Generate exit signals
            for i in range(1, len(signals_df)):
                # Exit long position when price reverts to mean or becomes overbought
                if signals_df['signal'].iloc[i-1] == SIGNAL_TYPES['ENTRY']['LONG']:
                    if (signals_df['price_reverting_to_mean'].iloc[i] or 
                        signals_df['rsi_overbought'].iloc[i] or 
                        signals_df['stoch_overbought'].iloc[i]):
                        
                        signals_df.loc[signals_df.index[i], 'signal'] = SIGNAL_TYPES['EXIT']['LONG']
                        signals_df.loc[signals_df.index[i], 'signal_strength'] = 0.8
                        
                        # Record signal for performance tracking
                        self.record_signal(
                            timestamp=signals_df['timestamp'].iloc[i],
                            symbol=symbol,
                            timeframe=timeframe,
                            signal_type=SIGNAL_TYPES['EXIT']['LONG'],
                            signal_strength=0.8,
                            price=signals_df['close'].iloc[i]
                        )
                
                # Exit short position when price reverts to mean or becomes oversold
                elif signals_df['signal'].iloc[i-1] == SIGNAL_TYPES['ENTRY']['SHORT']:
                    if (signals_df['price_reverting_to_mean'].iloc[i] or 
                        signals_df['rsi_oversold'].iloc[i] or 
                        signals_df['stoch_oversold'].iloc[i]):
                        
                        signals_df.loc[signals_df.index[i], 'signal'] = SIGNAL_TYPES['EXIT']['SHORT']
                        signals_df.loc[signals_df.index[i], 'signal_strength'] = 0.8
                        
                        # Record signal for performance tracking
                        self.record_signal(
                            timestamp=signals_df['timestamp'].iloc[i],
                            symbol=symbol,
                            timeframe=timeframe,
                            signal_type=SIGNAL_TYPES['EXIT']['SHORT'],
                            signal_strength=0.8,
                            price=signals_df['close'].iloc[i]
                        )
            
            logger.info(f"Generated mean reversion signals for {symbol} {timeframe}")
            return signals_df
        
        except Exception as e:
            logger.error(f"Error generating mean reversion signals: {str(e)}")
            return pd.DataFrame()
