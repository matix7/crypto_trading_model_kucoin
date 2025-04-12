"""
Technical indicator calculator for cryptocurrency trading.
"""

import os
import logging
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import talib
from talib import abstract

from .config import TECHNICAL_INDICATORS, DATABASE, TRADING_PAIRS, TIMEFRAMES
from .market_data_collector import MarketDataCollector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/data_collection.log',
    filemode='a'
)
logger = logging.getLogger('technical_indicator_calculator')

class TechnicalIndicatorCalculator:
    """
    Class for calculating technical indicators for cryptocurrency data.
    Uses TA-Lib for indicator calculations.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the TechnicalIndicatorCalculator.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses the path from config.
        """
        self.db_path = db_path or DATABASE['path']
        self.market_data_collector = MarketDataCollector(db_path=self.db_path)
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info("TechnicalIndicatorCalculator initialized")
    
    def _init_database(self):
        """Initialize the SQLite database with required tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create technical indicators table
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {DATABASE['tables']['indicators']} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            indicator_name TEXT NOT NULL,
            indicator_value REAL,
            indicator_params TEXT,
            UNIQUE(symbol, timeframe, timestamp, indicator_name, indicator_params)
        )
        ''')
        
        # Create index for faster queries
        cursor.execute(f'''
        CREATE INDEX IF NOT EXISTS idx_indicators 
        ON {DATABASE['tables']['indicators']} (symbol, timeframe, timestamp, indicator_name)
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Technical indicators table initialized")
    
    def calculate_sma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            df: DataFrame with OHLCV data
            period: SMA period
            
        Returns:
            Series with SMA values
        """
        try:
            return talib.SMA(df['close'], timeperiod=period)
        except Exception as e:
            logger.error(f"Error calculating SMA({period}): {str(e)}")
            return pd.Series(np.nan, index=df.index)
    
    def calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            df: DataFrame with OHLCV data
            period: EMA period
            
        Returns:
            Series with EMA values
        """
        try:
            return talib.EMA(df['close'], timeperiod=period)
        except Exception as e:
            logger.error(f"Error calculating EMA({period}): {str(e)}")
            return pd.Series(np.nan, index=df.index)
    
    def calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            df: DataFrame with OHLCV data
            period: RSI period
            
        Returns:
            Series with RSI values
        """
        try:
            return talib.RSI(df['close'], timeperiod=period)
        except Exception as e:
            logger.error(f"Error calculating RSI({period}): {str(e)}")
            return pd.Series(np.nan, index=df.index)
    
    def calculate_macd(self, df: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            df: DataFrame with OHLCV data
            fast_period: Fast period
            slow_period: Slow period
            signal_period: Signal period
            
        Returns:
            Tuple of (MACD, Signal, Histogram)
        """
        try:
            macd, signal, hist = talib.MACD(
                df['close'], 
                fastperiod=fast_period, 
                slowperiod=slow_period, 
                signalperiod=signal_period
            )
            return macd, signal, hist
        except Exception as e:
            logger.error(f"Error calculating MACD({fast_period},{slow_period},{signal_period}): {str(e)}")
            nan_series = pd.Series(np.nan, index=df.index)
            return nan_series, nan_series, nan_series
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        try:
            upper, middle, lower = talib.BBANDS(
                df['close'], 
                timeperiod=period, 
                nbdevup=std_dev, 
                nbdevdn=std_dev, 
                matype=0
            )
            return upper, middle, lower
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands({period},{std_dev}): {str(e)}")
            nan_series = pd.Series(np.nan, index=df.index)
            return nan_series, nan_series, nan_series
    
    def calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            df: DataFrame with OHLCV data
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        try:
            return talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        except Exception as e:
            logger.error(f"Error calculating ATR({period}): {str(e)}")
            return pd.Series(np.nan, index=df.index)
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int, d_period: int) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            df: DataFrame with OHLCV data
            k_period: %K period
            d_period: %D period
            
        Returns:
            Tuple of (Slow %K, Slow %D)
        """
        try:
            slowk, slowd = talib.STOCH(
                df['high'], 
                df['low'], 
                df['close'], 
                fastk_period=k_period, 
                slowk_period=3, 
                slowk_matype=0, 
                slowd_period=d_period, 
                slowd_matype=0
            )
            return slowk, slowd
        except Exception as e:
            logger.error(f"Error calculating Stochastic({k_period},{d_period}): {str(e)}")
            nan_series = pd.Series(np.nan, index=df.index)
            return nan_series, nan_series
    
    def calculate_stochrsi(self, df: pd.DataFrame, period: int, k_period: int, d_period: int) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic RSI.
        
        Args:
            df: DataFrame with OHLCV data
            period: RSI period
            k_period: %K period
            d_period: %D period
            
        Returns:
            Tuple of (Stoch RSI %K, Stoch RSI %D)
        """
        try:
            fastk, fastd = talib.STOCHRSI(
                df['close'], 
                timeperiod=period, 
                fastk_period=k_period, 
                fastd_period=d_period, 
                fastd_matype=0
            )
            return fastk, fastd
        except Exception as e:
            logger.error(f"Error calculating StochRSI({period},{k_period},{d_period}): {str(e)}")
            nan_series = pd.Series(np.nan, index=df.index)
            return nan_series, nan_series
    
    def calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Average Directional Index.
        
        Args:
            df: DataFrame with OHLCV data
            period: ADX period
            
        Returns:
            Series with ADX values
        """
        try:
            return talib.ADX(df['high'], df['low'], df['close'], timeperiod=period)
        except Exception as e:
            logger.error(f"Error calculating ADX({period}): {str(e)}")
            return pd.Series(np.nan, index=df.index)
    
    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with OBV values
        """
        try:
            return talib.OBV(df['close'], df['volume'])
        except Exception as e:
            logger.error(f"Error calculating OBV: {str(e)}")
            return pd.Series(np.nan, index=df.index)
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price.
        Note: VWAP is typically calculated per session (e.g., daily).
        This is a simplified version.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with VWAP values
        """
        try:
            df = df.copy()
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['price_volume'] = df['typical_price'] * df['volume']
            df['cumulative_price_volume'] = df['price_volume'].cumsum()
            df['cumulative_volume'] = df['volume'].cumsum()
            vwap = df['cumulative_price_volume'] / df['cumulative_volume']
            return vwap
        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
            return pd.Series(np.nan, index=df.index)
    
    def calculate_all_indicators(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Calculate all configured technical indicators for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            
        Returns:
            DataFrame with all indicators
        """
        # Get OHLCV data
        df = self.market_data_collector.get_ohlcv_data(symbol, timeframe)
        
        if df.empty:
            logger.warning(f"No OHLCV data available for {symbol} {timeframe}")
            return pd.DataFrame()
        
        # Create a dictionary to store all indicators
        indicators = {}
        
        # Calculate SMA
        for period in TECHNICAL_INDICATORS['SMA']:
            indicators[f'SMA_{period}'] = self.calculate_sma(df, period)
        
        # Calculate EMA
        for period in TECHNICAL_INDICATORS['EMA']:
            indicators[f'EMA_{period}'] = self.calculate_ema(df, period)
        
        # Calculate RSI
        for period in TECHNICAL_INDICATORS['RSI']:
            indicators[f'RSI_{period}'] = self.calculate_rsi(df, period)
        
        # Calculate MACD
        macd_config = TECHNICAL_INDICATORS['MACD']
        macd, signal, hist = self.calculate_macd(
            df, 
            macd_config['fast'], 
            macd_config['slow'], 
            macd_config['signal']
        )
        indicators['MACD'] = macd
        indicators['MACD_Signal'] = signal
        indicators['MACD_Hist'] = hist
        
        # Calculate Bollinger Bands
        bb_config = TECHNICAL_INDICATORS['BB']
        upper, middle, lower = self.calculate_bollinger_bands(
            df, 
            bb_config['period'], 
            bb_config['std_dev']
        )
        indicators['BB_Upper'] = upper
        indicators['BB_Middle'] = middle
        indicators['BB_Lower'] = lower
        
        # Calculate ATR
        for period in TECHNICAL_INDICATORS['ATR']:
            indicators[f'ATR_{period}'] = self.calculate_atr(df, period)
        
        # Calculate Stochastic
        stoch_config = TECHNICAL_INDICATORS['STOCH']
        slowk, slowd = self.calculate_stochastic(
            df, 
            stoch_config['k_period'], 
            stoch_config['d_period']
        )
        indicators['STOCH_K'] = slowk
        indicators['STOCH_D'] = slowd
        
        # Calculate Stochastic RSI
        stochrsi_config = TECHNICAL_INDICATORS['STOCHRSI']
        fastk, fastd = self.calculate_stochrsi(
            df, 
            stochrsi_config['period'], 
            stochrsi_config['k_period'], 
            stochrsi_config['d_period']
        )
        indicators['STOCHRSI_K'] = fastk
        indicators['STOCHRSI_D'] = fastd
        
        # Calculate ADX
        for period in TECHNICAL_INDICATORS['ADX']:
            indicators[f'ADX_{period}'] = self.calculate_adx(df, period)
        
        # Calculate OBV
        indicators['OBV'] = self.calculate_obv(df)
        
        # Calculate VWAP
        indicators['VWAP'] = self.calculate_vwap(df)
        
        # Create a DataFrame with all indicators
        indicators_df = pd.DataFrame(indicators)
        
        # Add timestamp, symbol, and timeframe columns
        indicators_df['timestamp'] = df['timestamp'].values
        indicators_df['symbol'] = symbol
        indicators_df['timeframe'] = timeframe
        
        return indicators_df
    
    def save_indicators_to_db(self, indicators_df: pd.DataFrame):
        """
        Save technical indicators to the database.
        
        Args:
            indicators_df: DataFrame with technical indicators
        """
        if indicators_df.empty:
            logger.warning("No indicators to save to database")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get list of indicator columns (exclude timestamp, symbol, timeframe)
            indicator_columns = [col for col in indicators_df.columns 
                               if col not in ['timestamp', 'symbol', 'timeframe']]
            
            # Prepare data for insertion
            for _, row in indicators_df.iterrows():
                timestamp = row['timestamp']
                symbol = row['symbol']
                timeframe = row['timeframe']
                
                for indicator_name in indicator_columns:
                    indicator_value = row[indicator_name]
                    
                    # Skip NaN values
                    if pd.isna(indicator_value):
                        continue
                    
                    # Extract indicator parameters from name (e.g., SMA_20 -> SMA, 20)
                    if '_' in indicator_name:
                        base_name, params = indicator_name.split('_', 1)
                    else:
                        base_name, params = indicator_name, ''
                    
                    # Insert or replace indicator value
                    cursor.execute(f'''
                    INSERT OR REPLACE INTO {DATABASE['tables']['indicators']}
                    (symbol, timeframe, timestamp, indicator_name, indicator_value, indicator_params)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''', (symbol, timeframe, timestamp, base_name, indicator_value, params))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved indicators for {symbol} {timeframe} to database")
        
        except Exception as e:
            logger.error(f"Error saving indicators to database: {str(e)}")
    
    def update_all_indicators(self):
        """Update technical indicators for all configured symbols and timeframes."""
        for symbol in TRADING_PAIRS:
            for timeframe in TIMEFRAMES:
                logger.info(f"Calculating indicators for {symbol} {timeframe}")
                indicators_df = self.calculate_all_indicators(symbol, timeframe)
                self.save_indicators_to_db(indicators_df)
    
    def get_indicators(self, symbol: str, timeframe: str, 
                      indicator_names: Optional[List[str]] = None, 
                      limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve technical indicators from the database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            indicator_names: List of indicator names to retrieve. If None, retrieves all.
            limit: Number of records to retrieve (most recent). If None, retrieves all.
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = f'''
            SELECT timestamp, indicator_name, indicator_value, indicator_params
            FROM {DATABASE['tables']['indicators']}
            WHERE symbol = ? AND timeframe = ?
            '''
            
            params = [symbol, timeframe]
            
            if indicator_names:
                placeholders = ','.join(['?'] * len(indicator_names))
                query += f" AND indicator_name IN ({placeholders})"
                params.extend(indicator_names)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
            
            # Pivot the data to have indicators as columns
            pivot_df = df.pivot_table(
                index='timestamp',
                columns=['indicator_name', 'indicator_params'],
                values='indicator_value',
                aggfunc='first'
            )
            
            # Flatten the column multi-index
            pivot_df.columns = [f"{name}_{params}" if params else name 
                              for name, params in pivot_df.columns]
            
            # Reset index to make timestamp a column
            pivot_df = pivot_df.reset_index()
            
            # Sort by timestamp ascending
            pivot_df = pivot_df.sort_values('timestamp')
            
            return pivot_df
        
        except Exception as e:
            logger.error(f"Error retrieving indicators: {str(e)}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('/home/ubuntu/crypto_trading_model/logs', exist_ok=True)
    
    # Test the indicator calculator
    calculator = TechnicalIndicatorCalculator()
    
    # Update indicators for a sample symbol and timeframe
    symbol = TRADING_PAIRS[0]
    timeframe = list(TIMEFRAMES.keys())[0]
    
    indicators_df = calculator.calculate_all_indicators(symbol, timeframe)
    calculator.save_indicators_to_db(indicators_df)
    
    # Print sample indicators
    retrieved_df = calculator.get_indicators(symbol, timeframe, limit=5)
    print(f"\n{symbol} {timeframe} indicators:")
    print(retrieved_df)
