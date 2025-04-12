"""
Feature engineering and data preprocessing for cryptocurrency trading.
"""

import os
import logging
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import talib
from talib import abstract

from .config import (
    TECHNICAL_INDICATORS, DATABASE, TRADING_PAIRS, TIMEFRAMES,
    NORMALIZATION, FEATURE_ENGINEERING
)
from .market_data_collector import MarketDataCollector
from .technical_indicator_calculator import TechnicalIndicatorCalculator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/data_collection.log',
    filemode='a'
)
logger = logging.getLogger('feature_engineering')

class FeatureEngineering:
    """
    Class for feature engineering and data preprocessing for cryptocurrency trading.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the FeatureEngineering.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses the path from config.
        """
        self.db_path = db_path or DATABASE['path']
        self.market_data_collector = MarketDataCollector(db_path=self.db_path)
        self.indicator_calculator = TechnicalIndicatorCalculator(db_path=self.db_path)
        
        # Initialize scalers
        self.scalers = {}
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info("FeatureEngineering initialized")
    
    def _init_database(self):
        """Initialize the SQLite database with required tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create features table
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {DATABASE['tables']['features']} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            feature_name TEXT NOT NULL,
            feature_value REAL,
            UNIQUE(symbol, timeframe, timestamp, feature_name)
        )
        ''')
        
        # Create index for faster queries
        cursor.execute(f'''
        CREATE INDEX IF NOT EXISTS idx_features 
        ON {DATABASE['tables']['features']} (symbol, timeframe, timestamp, feature_name)
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Features table initialized")
    
    def _get_combined_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Get combined OHLCV and indicator data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            
        Returns:
            DataFrame with combined data
        """
        # Get OHLCV data
        ohlcv_df = self.market_data_collector.get_ohlcv_data(symbol, timeframe)
        
        if ohlcv_df.empty:
            logger.warning(f"No OHLCV data available for {symbol} {timeframe}")
            return pd.DataFrame()
        
        # Get indicator data
        indicator_df = self.indicator_calculator.get_indicators(symbol, timeframe)
        
        if indicator_df.empty:
            logger.warning(f"No indicator data available for {symbol} {timeframe}")
            return ohlcv_df
        
        # Merge OHLCV and indicator data on timestamp
        combined_df = pd.merge(ohlcv_df, indicator_df, on='timestamp', how='left')
        
        return combined_df
    
    def normalize_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Normalize data using configured normalization methods.
        
        Args:
            df: DataFrame with combined data
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            
        Returns:
            DataFrame with normalized data
        """
        if df.empty:
            return df
        
        normalized_df = df.copy()
        lookback_periods = NORMALIZATION['lookback_periods']
        
        # Create a unique key for this symbol and timeframe
        key = f"{symbol}_{timeframe}"
        
        # Initialize scalers if not already done
        if key not in self.scalers:
            self.scalers[key] = {}
        
        # Normalize price data
        price_columns = ['open', 'high', 'low', 'close']
        price_norm_method = NORMALIZATION['price_data']
        
        if price_norm_method == 'min_max':
            if 'price' not in self.scalers[key]:
                self.scalers[key]['price'] = MinMaxScaler()
            
            # Fit scaler on the most recent lookback periods
            recent_prices = df[price_columns].iloc[-lookback_periods:].values
            self.scalers[key]['price'].fit(recent_prices)
            
            # Transform all price data
            normalized_df[price_columns] = self.scalers[key]['price'].transform(df[price_columns].values)
        
        elif price_norm_method == 'z_score':
            if 'price' not in self.scalers[key]:
                self.scalers[key]['price'] = StandardScaler()
            
            recent_prices = df[price_columns].iloc[-lookback_periods:].values
            self.scalers[key]['price'].fit(recent_prices)
            normalized_df[price_columns] = self.scalers[key]['price'].transform(df[price_columns].values)
        
        # Normalize volume data
        volume_norm_method = NORMALIZATION['volume_data']
        
        if volume_norm_method == 'log':
            # Apply log transformation to volume
            normalized_df['volume'] = np.log1p(df['volume'])
        
        elif volume_norm_method == 'min_max':
            if 'volume' not in self.scalers[key]:
                self.scalers[key]['volume'] = MinMaxScaler()
            
            recent_volume = df[['volume']].iloc[-lookback_periods:].values
            self.scalers[key]['volume'].fit(recent_volume)
            normalized_df['volume'] = self.scalers[key]['volume'].transform(df[['volume']].values)
        
        elif volume_norm_method == 'z_score':
            if 'volume' not in self.scalers[key]:
                self.scalers[key]['volume'] = StandardScaler()
            
            recent_volume = df[['volume']].iloc[-lookback_periods:].values
            self.scalers[key]['volume'].fit(recent_volume)
            normalized_df['volume'] = self.scalers[key]['volume'].transform(df[['volume']].values)
        
        # Normalize indicators
        for indicator, method in NORMALIZATION['indicators'].items():
            # Find all columns that start with this indicator
            indicator_cols = [col for col in df.columns if col.startswith(indicator)]
            
            if not indicator_cols:
                continue
            
            if method == 'min_max':
                if indicator not in self.scalers[key]:
                    self.scalers[key][indicator] = MinMaxScaler()
                
                recent_values = df[indicator_cols].iloc[-lookback_periods:].values
                self.scalers[key][indicator].fit(recent_values)
                normalized_df[indicator_cols] = self.scalers[key][indicator].transform(df[indicator_cols].values)
            
            elif method == 'z_score':
                if indicator not in self.scalers[key]:
                    self.scalers[key][indicator] = StandardScaler()
                
                recent_values = df[indicator_cols].iloc[-lookback_periods:].values
                self.scalers[key][indicator].fit(recent_values)
                normalized_df[indicator_cols] = self.scalers[key][indicator].transform(df[indicator_cols].values)
            
            elif method == 'custom':
                # Custom normalization for specific indicators
                if indicator == 'BB':
                    # For Bollinger Bands, calculate percent B
                    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns and 'BB_Middle' in df.columns:
                        normalized_df['BB_Percent_B'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
                        # Clip to handle values outside bands
                        normalized_df['BB_Percent_B'] = normalized_df['BB_Percent_B'].clip(0, 1)
                        
                        # Calculate bandwidth
                        normalized_df['BB_Bandwidth'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        return normalized_df
    
    def engineer_price_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features based on price patterns.
        
        Args:
            df: DataFrame with normalized data
            
        Returns:
            DataFrame with additional price pattern features
        """
        if df.empty or not FEATURE_ENGINEERING['price_patterns']:
            return df
        
        result_df = df.copy()
        
        # Candlestick patterns
        pattern_functions = [
            ('CDL_DOJI', talib.CDLDOJI),
            ('CDL_HAMMER', talib.CDLHAMMER),
            ('CDL_SHOOTING_STAR', talib.CDLSHOOTINGSTAR),
            ('CDL_ENGULFING', talib.CDLENGULFING),
            ('CDL_EVENING_STAR', talib.CDLEVENINGSTAR),
            ('CDL_MORNING_STAR', talib.CDLMORNINGSTAR),
            ('CDL_HARAMI', talib.CDLHARAMI),
            ('CDL_MARUBOZU', talib.CDLMARUBOZU)
        ]
        
        for pattern_name, pattern_func in pattern_functions:
            try:
                result_df[f'PATTERN_{pattern_name}'] = pattern_func(
                    result_df['open'], result_df['high'], 
                    result_df['low'], result_df['close']
                )
            except Exception as e:
                logger.error(f"Error calculating {pattern_name}: {str(e)}")
        
        # Price action features
        try:
            # Body size relative to range
            result_df['BODY_SIZE'] = abs(result_df['close'] - result_df['open']) / (result_df['high'] - result_df['low'])
            
            # Upper and lower shadows
            result_df['UPPER_SHADOW'] = (result_df['high'] - result_df[['open', 'close']].max(axis=1)) / (result_df['high'] - result_df['low'])
            result_df['LOWER_SHADOW'] = (result_df[['open', 'close']].min(axis=1) - result_df['low']) / (result_df['high'] - result_df['low'])
            
            # Bullish or bearish candle
            result_df['BULLISH'] = (result_df['close'] > result_df['open']).astype(int)
            
            # Range relative to previous candle
            result_df['RANGE_CHANGE'] = (result_df['high'] - result_df['low']) / (result_df['high'].shift(1) - result_df['low'].shift(1))
            
            # Gap up or down
            result_df['GAP_UP'] = ((result_df['low'] > result_df['high'].shift(1))).astype(int)
            result_df['GAP_DOWN'] = ((result_df['high'] < result_df['low'].shift(1))).astype(int)
        
        except Exception as e:
            logger.error(f"Error calculating price action features: {str(e)}")
        
        return result_df
    
    def engineer_indicator_crossover_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features based on indicator crossovers.
        
        Args:
            df: DataFrame with normalized data
            
        Returns:
            DataFrame with additional indicator crossover features
        """
        if df.empty or not FEATURE_ENGINEERING['indicator_crossovers']:
            return df
        
        result_df = df.copy()
        
        try:
            # EMA crossovers
            ema_periods = TECHNICAL_INDICATORS['EMA']
            for i, fast_period in enumerate(ema_periods):
                for slow_period in ema_periods[i+1:]:
                    fast_col = f'EMA_{fast_period}'
                    slow_col = f'EMA_{slow_period}'
                    
                    if fast_col in df.columns and slow_col in df.columns:
                        # Current crossover state (1 if fast > slow, 0 otherwise)
                        result_df[f'EMA_CROSS_{fast_period}_{slow_period}'] = (df[fast_col] > df[slow_col]).astype(int)
                        
                        # Crossover events (1 for golden cross, -1 for death cross, 0 for no cross)
                        cross_col = f'EMA_CROSS_EVENT_{fast_period}_{slow_period}'
                        result_df[cross_col] = 0
                        
                        # Golden cross (fast crosses above slow)
                        golden_cross = (df[fast_col] > df[slow_col]) & (df[fast_col].shift(1) <= df[slow_col].shift(1))
                        result_df.loc[golden_cross, cross_col] = 1
                        
                        # Death cross (fast crosses below slow)
                        death_cross = (df[fast_col] < df[slow_col]) & (df[fast_col].shift(1) >= df[slow_col].shift(1))
                        result_df.loc[death_cross, cross_col] = -1
            
            # MACD crossovers
            if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                # Current crossover state
                result_df['MACD_CROSS'] = (df['MACD'] > df['MACD_Signal']).astype(int)
                
                # Crossover events
                result_df['MACD_CROSS_EVENT'] = 0
                
                # Bullish crossover (MACD crosses above signal)
                bullish_cross = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
                result_df.loc[bullish_cross, 'MACD_CROSS_EVENT'] = 1
                
                # Bearish crossover (MACD crosses below signal)
                bearish_cross = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
                result_df.loc[bearish_cross, 'MACD_CROSS_EVENT'] = -1
                
                # MACD zero line crossovers
                result_df['MACD_ABOVE_ZERO'] = (df['MACD'] > 0).astype(int)
                
                result_df['MACD_ZERO_CROSS_EVENT'] = 0
                bullish_zero_cross = (df['MACD'] > 0) & (df['MACD'].shift(1) <= 0)
                bearish_zero_cross = (df['MACD'] < 0) & (df['MACD'].shift(1) >= 0)
                
                result_df.loc[bullish_zero_cross, 'MACD_ZERO_CROSS_EVENT'] = 1
                result_df.loc[bearish_zero_cross, 'MACD_ZERO_CROSS_EVENT'] = -1
            
            # Stochastic crossovers
            if 'STOCH_K' in df.columns and 'STOCH_D' in df.columns:
                result_df['STOCH_CROSS'] = (df['STOCH_K'] > df['STOCH_D']).astype(int)
                
                result_df['STOCH_CROSS_EVENT'] = 0
                bullish_cross = (df['STOCH_K'] > df['STOCH_D']) & (df['STOCH_K'].shift(1) <= df['STOCH_D'].shift(1))
                bearish_cross = (df['STOCH_K'] < df['STOCH_D']) & (df['STOCH_K'].shift(1) >= df['STOCH_D'].shift(1))
                
                result_df.loc[bullish_cross, 'STOCH_CROSS_EVENT'] = 1
                result_df.loc[bearish_cross, 'STOCH_CROSS_EVENT'] = -1
                
                # Overbought/oversold conditions
                result_df['STOCH_OVERBOUGHT'] = (df['STOCH_K'] > 80).astype(int)
                result_df['STOCH_OVERSOLD'] = (df['STOCH_K'] < 20).astype(int)
            
            # RSI conditions
            if 'RSI_14' in df.columns:
                result_df['RSI_OVERBOUGHT'] = (df['RSI_14'] > 70).astype(int)
                result_df['RSI_OVERSOLD'] = (df['RSI_14'] < 30).astype(int)
                
                # RSI divergence with price (simplified)
                result_df['RSI_BULL_DIV'] = ((df['close'] < df['close'].shift(1)) & 
                                           (df['RSI_14'] > df['RSI_14'].shift(1))).astype(int)
                
                result_df['RSI_BEAR_DIV'] = ((df['close'] > df['close'].shift(1)) & 
                                           (df['RSI_14'] < df['RSI_14'].shift(1))).astype(int)
        
        except Exception as e:
            logger.error(f"Error calculating indicator crossover features: {str(e)}")
        
        return result_df
    
    def engineer_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features based on volatility.
        
        Args:
            df: DataFrame with normalized data
            
        Returns:
            DataFrame with additional volatility features
        """
        if df.empty or not FEATURE_ENGINEERING['volatility_features']:
            return df
        
        result_df = df.copy()
        
        try:
            # Volatility measures
            # Historical volatility (standard deviation of returns)
            for period in [5, 10, 20]:
                returns = df['close'].pct_change()
                result_df[f'VOLATILITY_{period}'] = returns.rolling(period).std()
            
            # ATR relative to price
            if 'ATR_14' in df.columns:
                result_df['ATR_PERCENT'] = df['ATR_14'] / df['close']
            
            # Bollinger Band width
            if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns and 'BB_Middle' in df.columns:
                result_df['BB_WIDTH'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
                
                # Price position within Bollinger Bands
                result_df['BB_POSITION'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
                
                # Bollinger Band squeeze (narrowing bands)
                result_df['BB_SQUEEZE'] = result_df['BB_WIDTH'] < result_df['BB_WIDTH'].rolling(20).mean()
            
            # High-Low range relative to close
            result_df['HL_RANGE_PCT'] = (df['high'] - df['low']) / df['close']
            
            # Consecutive up/down days
            result_df['UP_STREAK'] = 0
            result_df['DOWN_STREAK'] = 0
            
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    result_df['UP_STREAK'].iloc[i] = result_df['UP_STREAK'].iloc[i-1] + 1
                    result_df['DOWN_STREAK'].iloc[i] = 0
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    result_df['DOWN_STREAK'].iloc[i] = result_df['DOWN_STREAK'].iloc[i-1] + 1
                    result_df['UP_STREAK'].iloc[i] = 0
        
        except Exception as e:
            logger.error(f"Error calculating volatility features: {str(e)}")
        
        return result_df
    
    def engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features based on time.
        
        Args:
            df: DataFrame with normalized data
            
        Returns:
            DataFrame with additional temporal features
        """
        if df.empty or not FEATURE_ENGINEERING['temporal_features']:
            return df
        
        result_df = df.copy()
        
        try:
            # Convert timestamp to datetime
            timestamps = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Hour of day (0-23)
            result_df['HOUR'] = timestamps.dt.hour
            
            # Day of week (0-6, 0 is Monday)
            result_df['DAY_OF_WEEK'] = timestamps.dt.dayofweek
            
            # Weekend indicator
            result_df['IS_WEEKEND'] = (timestamps.dt.dayofweek >= 5).astype(int)
            
            # Hour of day sine and cosine for cyclical representation
            hours_in_day = 24
            result_df['HOUR_SIN'] = np.sin(2 * np.pi * timestamps.dt.hour / hours_in_day)
            result_df['HOUR_COS'] = np.cos(2 * np.pi * timestamps.dt.hour / hours_in_day)
            
            # Day of week sine and cosine
            days_in_week = 7
            result_df['DAY_OF_WEEK_SIN'] = np.sin(2 * np.pi * timestamps.dt.dayofweek / days_in_week)
            result_df['DAY_OF_WEEK_COS'] = np.cos(2 * np.pi * timestamps.dt.dayofweek / days_in_week)
            
            # Month sine and cosine
            months_in_year = 12
            result_df['MONTH_SIN'] = np.sin(2 * np.pi * timestamps.dt.month / months_in_year)
            result_df['MONTH_COS'] = np.cos(2 * np.pi * timestamps.dt.month / months_in_year)
        
        except Exception as e:
            logger.error(f"Error calculating temporal features: {str(e)}")
        
        return result_df
    
    def engineer_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features based on volume.
        
        Args:
            df: DataFrame with normalized data
            
        Returns:
            DataFrame with additional volume features
        """
        if df.empty or not FEATURE_ENGINEERING['volume_features']:
            return df
        
        result_df = df.copy()
        
        try:
            # Volume moving averages
            for period in [5, 10, 20]:
                result_df[f'VOLUME_MA_{period}'] = df['volume'].rolling(period).mean()
            
            # Volume relative to moving average
            for period in [5, 10, 20]:
                vol_ma_col = f'VOLUME_MA_{period}'
                if vol_ma_col in result_df.columns:
                    result_df[f'VOLUME_RATIO_{period}'] = df['volume'] / result_df[vol_ma_col]
            
            # Volume trend (increasing or decreasing)
            result_df['VOLUME_TREND'] = (df['volume'] > df['volume'].shift(1)).astype(int)
            
            # Volume spikes
            for period in [5, 10, 20]:
                vol_ma_col = f'VOLUME_MA_{period}'
                if vol_ma_col in result_df.columns:
                    # Volume spike defined as volume > 2x its moving average
                    result_df[f'VOLUME_SPIKE_{period}'] = (df['volume'] > 2 * result_df[vol_ma_col]).astype(int)
            
            # Price-volume relationship
            # Up volume vs down volume
            result_df['UP_VOLUME'] = df['volume'] * (df['close'] > df['open']).astype(int)
            result_df['DOWN_VOLUME'] = df['volume'] * (df['close'] < df['open']).astype(int)
            
            # Accumulation/Distribution
            result_df['AD_FACTOR'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            result_df['AD_FACTOR'] = result_df['AD_FACTOR'].fillna(0)
            result_df['AD_VOLUME'] = result_df['AD_FACTOR'] * df['volume']
            result_df['AD_LINE'] = result_df['AD_VOLUME'].cumsum()
            
            # Money Flow Index components
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            
            # Positive and negative money flow
            pos_money_flow = money_flow * (typical_price > typical_price.shift(1)).astype(int)
            neg_money_flow = money_flow * (typical_price < typical_price.shift(1)).astype(int)
            
            # Calculate Money Flow Ratio and Index for different periods
            for period in [14]:
                pos_sum = pos_money_flow.rolling(period).sum()
                neg_sum = neg_money_flow.rolling(period).sum()
                
                # Avoid division by zero
                money_ratio = np.where(neg_sum != 0, pos_sum / neg_sum, 100)
                result_df[f'MFI_{period}'] = 100 - (100 / (1 + money_ratio))
        
        except Exception as e:
            logger.error(f"Error calculating volume features: {str(e)}")
        
        return result_df
    
    def engineer_all_features(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Engineer all features for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            
        Returns:
            DataFrame with all engineered features
        """
        # Get combined data
        combined_df = self._get_combined_data(symbol, timeframe)
        
        if combined_df.empty:
            logger.warning(f"No data available for {symbol} {timeframe}")
            return pd.DataFrame()
        
        # Normalize data
        normalized_df = self.normalize_data(combined_df, symbol, timeframe)
        
        # Engineer features
        result_df = normalized_df.copy()
        
        # Price pattern features
        result_df = self.engineer_price_pattern_features(result_df)
        
        # Indicator crossover features
        result_df = self.engineer_indicator_crossover_features(result_df)
        
        # Volatility features
        result_df = self.engineer_volatility_features(result_df)
        
        # Temporal features
        result_df = self.engineer_temporal_features(result_df)
        
        # Volume features
        result_df = self.engineer_volume_features(result_df)
        
        logger.info(f"Engineered {len(result_df.columns) - len(combined_df.columns)} new features for {symbol} {timeframe}")
        
        return result_df
    
    def save_features_to_db(self, features_df: pd.DataFrame, symbol: str, timeframe: str):
        """
        Save engineered features to the database.
        
        Args:
            features_df: DataFrame with engineered features
            symbol: Trading pair symbol
            timeframe: Timeframe interval
        """
        if features_df.empty:
            logger.warning("No features to save to database")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get list of feature columns (exclude original columns)
            original_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe']
            indicator_columns = [col for col in features_df.columns if col.startswith(tuple(TECHNICAL_INDICATORS.keys()))]
            
            feature_columns = [col for col in features_df.columns 
                             if col not in original_columns and col not in indicator_columns]
            
            # Prepare data for insertion
            for _, row in features_df.iterrows():
                timestamp = row['timestamp']
                
                for feature_name in feature_columns:
                    feature_value = row[feature_name]
                    
                    # Skip NaN values
                    if pd.isna(feature_value):
                        continue
                    
                    # Insert or replace feature value
                    cursor.execute(f'''
                    INSERT OR REPLACE INTO {DATABASE['tables']['features']}
                    (symbol, timeframe, timestamp, feature_name, feature_value)
                    VALUES (?, ?, ?, ?, ?)
                    ''', (symbol, timeframe, timestamp, feature_name, feature_value))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved features for {symbol} {timeframe} to database")
        
        except Exception as e:
            logger.error(f"Error saving features to database: {str(e)}")
    
    def update_all_features(self):
        """Update engineered features for all configured symbols and timeframes."""
        for symbol in TRADING_PAIRS:
            for timeframe in TIMEFRAMES:
                logger.info(f"Engineering features for {symbol} {timeframe}")
                features_df = self.engineer_all_features(symbol, timeframe)
                self.save_features_to_db(features_df, symbol, timeframe)
    
    def get_features(self, symbol: str, timeframe: str, 
                    feature_names: Optional[List[str]] = None, 
                    limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve engineered features from the database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            feature_names: List of feature names to retrieve. If None, retrieves all.
            limit: Number of records to retrieve (most recent). If None, retrieves all.
            
        Returns:
            DataFrame with engineered features
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = f'''
            SELECT timestamp, feature_name, feature_value
            FROM {DATABASE['tables']['features']}
            WHERE symbol = ? AND timeframe = ?
            '''
            
            params = [symbol, timeframe]
            
            if feature_names:
                placeholders = ','.join(['?'] * len(feature_names))
                query += f" AND feature_name IN ({placeholders})"
                params.extend(feature_names)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
            
            # Pivot the data to have features as columns
            pivot_df = df.pivot_table(
                index='timestamp',
                columns='feature_name',
                values='feature_value',
                aggfunc='first'
            )
            
            # Reset index to make timestamp a column
            pivot_df = pivot_df.reset_index()
            
            # Sort by timestamp ascending
            pivot_df = pivot_df.sort_values('timestamp')
            
            return pivot_df
        
        except Exception as e:
            logger.error(f"Error retrieving features: {str(e)}")
            return pd.DataFrame()
    
    def get_complete_dataset(self, symbol: str, timeframe: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get a complete dataset with OHLCV data, indicators, and engineered features.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            limit: Number of records to retrieve (most recent). If None, retrieves all.
            
        Returns:
            DataFrame with complete dataset
        """
        # Get OHLCV data
        ohlcv_df = self.market_data_collector.get_ohlcv_data(symbol, timeframe, limit=limit)
        
        if ohlcv_df.empty:
            logger.warning(f"No OHLCV data available for {symbol} {timeframe}")
            return pd.DataFrame()
        
        # Get indicator data
        indicator_df = self.indicator_calculator.get_indicators(symbol, timeframe, limit=limit)
        
        # Get feature data
        feature_df = self.get_features(symbol, timeframe, limit=limit)
        
        # Merge all data on timestamp
        result_df = ohlcv_df
        
        if not indicator_df.empty:
            result_df = pd.merge(result_df, indicator_df, on='timestamp', how='left')
        
        if not feature_df.empty:
            result_df = pd.merge(result_df, feature_df, on='timestamp', how='left')
        
        return result_df
    
    def prepare_ml_dataset(self, symbol: str, timeframe: str, 
                          target_column: str = 'close', 
                          prediction_horizon: int = 1,
                          sequence_length: int = 10,
                          train_test_split: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare a dataset for machine learning with sequences and targets.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            target_column: Column to predict
            prediction_horizon: Number of steps ahead to predict
            sequence_length: Length of input sequences
            train_test_split: Proportion of data to use for training
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Get complete dataset
        df = self.get_complete_dataset(symbol, timeframe)
        
        if df.empty:
            logger.warning(f"No data available for {symbol} {timeframe}")
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Create target variable (future price movement)
        if target_column == 'return':
            # Percentage return
            df['target'] = df['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
        elif target_column == 'direction':
            # Price direction (1 for up, 0 for down)
            df['target'] = (df['close'].shift(-prediction_horizon) > df['close']).astype(int)
        else:
            # Future price
            df['target'] = df[target_column].shift(-prediction_horizon)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Select features (exclude timestamp, symbol, timeframe, and target)
        feature_columns = [col for col in df.columns 
                         if col not in ['timestamp', 'symbol', 'timeframe', 'target']]
        
        # Create sequences
        X, y = [], []
        for i in range(len(df) - sequence_length):
            X.append(df[feature_columns].iloc[i:i+sequence_length].values)
            y.append(df['target'].iloc[i+sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test sets
        split_idx = int(len(X) * train_test_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('/home/ubuntu/crypto_trading_model/logs', exist_ok=True)
    
    # Test the feature engineering
    feature_eng = FeatureEngineering()
    
    # Update features for a sample symbol and timeframe
    symbol = TRADING_PAIRS[0]
    timeframe = list(TIMEFRAMES.keys())[0]
    
    features_df = feature_eng.engineer_all_features(symbol, timeframe)
    feature_eng.save_features_to_db(features_df, symbol, timeframe)
    
    # Print sample features
    retrieved_df = feature_eng.get_features(symbol, timeframe, limit=5)
    print(f"\n{symbol} {timeframe} features:")
    print(retrieved_df.head())
    
    # Prepare ML dataset
    X_train, X_test, y_train, y_test = feature_eng.prepare_ml_dataset(
        symbol, timeframe, target_column='direction', prediction_horizon=1
    )
    
    print(f"\nML dataset shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
