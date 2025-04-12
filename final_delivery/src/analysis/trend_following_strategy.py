"""
Trend following strategy implementation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from .base_strategy import BaseStrategy
from ..data.data_pipeline import DataPipeline
from .config import SIGNAL_TYPES, SIGNAL_STRENGTH, TREND_FOLLOWING

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/technical_analysis.log',
    filemode='a'
)
logger = logging.getLogger('trend_following_strategy')

class TrendFollowingStrategy(BaseStrategy):
    """
    Strategy that follows established market trends using moving averages,
    MACD, and ADX indicators.
    """
    
    def __init__(self, data_pipeline: DataPipeline, config: Dict = None):
        """
        Initialize the TrendFollowingStrategy.
        
        Args:
            data_pipeline: DataPipeline instance for accessing market data
            config: Optional custom configuration (defaults to TREND_FOLLOWING from config)
        """
        super().__init__("Trend Following", "TREND_FOLLOWING", data_pipeline)
        self.config = config or TREND_FOLLOWING
        logger.info(f"Initialized Trend Following strategy with config: {self.config}")
    
    def generate_signals(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Generate trading signals based on trend following indicators.
        
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
            
            # Get EMA columns
            ema_short_col = f"EMA_{self.config['ema_short_period']}"
            ema_medium_col = f"EMA_{self.config['ema_medium_period']}"
            ema_long_col = f"EMA_{self.config['ema_long_period']}"
            
            # Check if required columns exist
            required_columns = [ema_short_col, ema_medium_col, ema_long_col, 'MACD', 'MACD_Signal', 'ADX_14']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns for trend following strategy: {missing_columns}")
                return signals_df
            
            # Calculate trend conditions
            # 1. EMA alignment (short > medium > long for uptrend, reverse for downtrend)
            signals_df['uptrend_ema_alignment'] = (
                (df[ema_short_col] > df[ema_medium_col]) & 
                (df[ema_medium_col] > df[ema_long_col])
            )
            
            signals_df['downtrend_ema_alignment'] = (
                (df[ema_short_col] < df[ema_medium_col]) & 
                (df[ema_medium_col] < df[ema_long_col])
            )
            
            # 2. MACD conditions
            signals_df['macd_above_signal'] = df['MACD'] > df['MACD_Signal']
            signals_df['macd_below_signal'] = df['MACD'] < df['MACD_Signal']
            signals_df['macd_above_zero'] = df['MACD'] > 0
            signals_df['macd_below_zero'] = df['MACD'] < 0
            
            # 3. ADX condition (strong trend)
            adx_col = 'ADX_14'
            signals_df['strong_trend'] = df[adx_col] > self.config['adx_threshold']
            
            # 4. Price relative to EMAs
            signals_df['price_above_short_ema'] = df['close'] > df[ema_short_col]
            signals_df['price_below_short_ema'] = df['close'] < df[ema_short_col]
            signals_df['price_above_long_ema'] = df['close'] > df[ema_long_col]
            signals_df['price_below_long_ema'] = df['close'] < df[ema_long_col]
            
            # 5. Trend duration
            for i in range(len(signals_df)):
                if i >= self.config['min_trend_duration']:
                    # Check uptrend duration
                    uptrend_count = 0
                    for j in range(self.config['min_trend_duration']):
                        if signals_df['uptrend_ema_alignment'].iloc[i-j]:
                            uptrend_count += 1
                    
                    signals_df.loc[signals_df.index[i], 'sustained_uptrend'] = (
                        uptrend_count >= self.config['min_trend_duration']
                    )
                    
                    # Check downtrend duration
                    downtrend_count = 0
                    for j in range(self.config['min_trend_duration']):
                        if signals_df['downtrend_ema_alignment'].iloc[i-j]:
                            downtrend_count += 1
                    
                    signals_df.loc[signals_df.index[i], 'sustained_downtrend'] = (
                        downtrend_count >= self.config['min_trend_duration']
                    )
                else:
                    signals_df.loc[signals_df.index[i], 'sustained_uptrend'] = False
                    signals_df.loc[signals_df.index[i], 'sustained_downtrend'] = False
            
            # Generate long entry signals
            long_entry_conditions = {
                'uptrend_ema_alignment': signals_df['uptrend_ema_alignment'],
                'macd_above_signal': signals_df['macd_above_signal'],
                'macd_above_zero': signals_df['macd_above_zero'],
                'strong_trend': signals_df['strong_trend'],
                'price_above_short_ema': signals_df['price_above_short_ema'],
                'sustained_uptrend': signals_df['sustained_uptrend']
            }
            
            long_entry_weights = {
                'uptrend_ema_alignment': 0.25,
                'macd_above_signal': 0.15,
                'macd_above_zero': 0.15,
                'strong_trend': 0.2,
                'price_above_short_ema': 0.1,
                'sustained_uptrend': 0.15
            }
            
            # Generate short entry signals
            short_entry_conditions = {
                'downtrend_ema_alignment': signals_df['downtrend_ema_alignment'],
                'macd_below_signal': signals_df['macd_below_signal'],
                'macd_below_zero': signals_df['macd_below_zero'],
                'strong_trend': signals_df['strong_trend'],
                'price_below_short_ema': signals_df['price_below_short_ema'],
                'sustained_downtrend': signals_df['sustained_downtrend']
            }
            
            short_entry_weights = {
                'downtrend_ema_alignment': 0.25,
                'macd_below_signal': 0.15,
                'macd_below_zero': 0.15,
                'strong_trend': 0.2,
                'price_below_short_ema': 0.1,
                'sustained_downtrend': 0.15
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
                # Exit long position
                if (signals_df['signal'].iloc[i-1] == SIGNAL_TYPES['ENTRY']['LONG'] and
                    (signals_df['macd_below_signal'].iloc[i] or 
                     signals_df['price_below_short_ema'].iloc[i])):
                    
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
                
                # Exit short position
                elif (signals_df['signal'].iloc[i-1] == SIGNAL_TYPES['ENTRY']['SHORT'] and
                      (signals_df['macd_above_signal'].iloc[i] or 
                       signals_df['price_above_short_ema'].iloc[i])):
                    
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
            
            logger.info(f"Generated trend following signals for {symbol} {timeframe}")
            return signals_df
        
        except Exception as e:
            logger.error(f"Error generating trend following signals: {str(e)}")
            return pd.DataFrame()
