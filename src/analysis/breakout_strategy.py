"""
Breakout strategy implementation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from .base_strategy import BaseStrategy
from ..data.data_pipeline import DataPipeline
from .config import SIGNAL_TYPES, SIGNAL_STRENGTH, BREAKOUT

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/technical_analysis.log',
    filemode='a'
)
logger = logging.getLogger('breakout_strategy')

class BreakoutStrategy(BaseStrategy):
    """
    Strategy that identifies and trades breakouts from consolidation ranges.
    Uses ATR, volume analysis, and price action to identify valid breakouts.
    """
    
    def __init__(self, data_pipeline: DataPipeline, config: Dict = None):
        """
        Initialize the BreakoutStrategy.
        
        Args:
            data_pipeline: DataPipeline instance for accessing market data
            config: Optional custom configuration (defaults to BREAKOUT from config)
        """
        super().__init__("Breakout", "BREAKOUT", data_pipeline)
        self.config = config or BREAKOUT
        logger.info(f"Initialized Breakout strategy with config: {self.config}")
    
    def generate_signals(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Generate trading signals based on breakout patterns.
        
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
            atr_col = f"ATR_{self.config['atr_period']}"
            
            required_columns = [atr_col, 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns for breakout strategy: {missing_columns}")
                return signals_df
            
            # Calculate breakout conditions
            # 1. Identify consolidation periods
            signals_df['high_low_range'] = df['high'] - df['low']
            signals_df['avg_range'] = signals_df['high_low_range'].rolling(self.config['consolidation_periods']).mean()
            signals_df['range_atr_ratio'] = signals_df['high_low_range'] / df[atr_col]
            
            # A period is considered consolidation if the range is less than the ATR
            signals_df['is_consolidation'] = signals_df['range_atr_ratio'] < 0.8
            
            # 2. Calculate rolling highs and lows for the consolidation period
            signals_df['consolidation_high'] = df['high'].rolling(self.config['consolidation_periods']).max()
            signals_df['consolidation_low'] = df['low'].rolling(self.config['consolidation_periods']).min()
            
            # 3. Identify breakouts
            # Upward breakout: price breaks above consolidation high
            signals_df['upward_breakout'] = df['close'] > signals_df['consolidation_high'].shift(1)
            
            # Downward breakout: price breaks below consolidation low
            signals_df['downward_breakout'] = df['close'] < signals_df['consolidation_low'].shift(1)
            
            # 4. Volume confirmation
            signals_df['volume_ma'] = df['volume'].rolling(self.config['consolidation_periods']).mean()
            signals_df['volume_surge'] = df['volume'] > (signals_df['volume_ma'] * self.config['volume_surge_threshold'])
            
            # 5. Breakout strength
            # Calculate percentage breakout
            signals_df['upward_breakout_pct'] = (df['close'] - signals_df['consolidation_high'].shift(1)) / signals_df['consolidation_high'].shift(1)
            signals_df['downward_breakout_pct'] = (signals_df['consolidation_low'].shift(1) - df['close']) / signals_df['consolidation_low'].shift(1)
            
            # Strong breakout if percentage exceeds threshold
            signals_df['strong_upward_breakout'] = signals_df['upward_breakout_pct'] > self.config['breakout_threshold']
            signals_df['strong_downward_breakout'] = signals_df['downward_breakout_pct'] > self.config['breakout_threshold']
            
            # 6. False breakout filter
            if self.config['false_breakout_filter']:
                # Check if price quickly reverses after breakout
                for i in range(3, len(signals_df)):
                    # For upward breakouts, check if price falls back below the breakout level
                    if signals_df['upward_breakout'].iloc[i-3]:
                        if df['low'].iloc[i] < signals_df['consolidation_high'].iloc[i-4]:
                            signals_df.loc[signals_df.index[i-3], 'upward_breakout'] = False
                    
                    # For downward breakouts, check if price rises back above the breakout level
                    if signals_df['downward_breakout'].iloc[i-3]:
                        if df['high'].iloc[i] > signals_df['consolidation_low'].iloc[i-4]:
                            signals_df.loc[signals_df.index[i-3], 'downward_breakout'] = False
            
            # Generate long entry signals (upward breakouts)
            long_entry_conditions = {
                'upward_breakout': signals_df['upward_breakout'],
                'strong_upward_breakout': signals_df['strong_upward_breakout'],
                'volume_surge': signals_df['volume_surge'],
                'prior_consolidation': signals_df['is_consolidation'].shift(1)
            }
            
            long_entry_weights = {
                'upward_breakout': 0.3,
                'strong_upward_breakout': 0.3,
                'volume_surge': 0.2,
                'prior_consolidation': 0.2
            }
            
            # Generate short entry signals (downward breakouts)
            short_entry_conditions = {
                'downward_breakout': signals_df['downward_breakout'],
                'strong_downward_breakout': signals_df['strong_downward_breakout'],
                'volume_surge': signals_df['volume_surge'],
                'prior_consolidation': signals_df['is_consolidation'].shift(1)
            }
            
            short_entry_weights = {
                'downward_breakout': 0.3,
                'strong_downward_breakout': 0.3,
                'volume_surge': 0.2,
                'prior_consolidation': 0.2
            }
            
            # Calculate signal strengths for each row
            for i in range(len(signals_df)):
                # Skip the first few rows where we don't have enough data
                if i < self.config['consolidation_periods']:
                    continue
                
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
                # Exit long position when price falls back below breakout level or ATR-based stop is hit
                if signals_df['signal'].iloc[i-1] == SIGNAL_TYPES['ENTRY']['LONG']:
                    # Calculate stop loss level based on ATR
                    stop_level = df['close'].iloc[i-1] - (df[atr_col].iloc[i-1] * self.config['atr_multiplier'])
                    
                    if df['low'].iloc[i] <= stop_level or df['close'].iloc[i] < signals_df['consolidation_high'].iloc[i-1]:
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
                
                # Exit short position when price rises back above breakout level or ATR-based stop is hit
                elif signals_df['signal'].iloc[i-1] == SIGNAL_TYPES['ENTRY']['SHORT']:
                    # Calculate stop loss level based on ATR
                    stop_level = df['close'].iloc[i-1] + (df[atr_col].iloc[i-1] * self.config['atr_multiplier'])
                    
                    if df['high'].iloc[i] >= stop_level or df['close'].iloc[i] > signals_df['consolidation_low'].iloc[i-1]:
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
            
            logger.info(f"Generated breakout signals for {symbol} {timeframe}")
            return signals_df
        
        except Exception as e:
            logger.error(f"Error generating breakout signals: {str(e)}")
            return pd.DataFrame()
