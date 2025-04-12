"""
Market data collector for retrieving cryptocurrency data from exchanges.
"""

import os
import time
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional, Tuple, Union

from .config import TIMEFRAMES, TRADING_PAIRS, DATABASE, API

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/data_collection.log',
    filemode='a'
)
logger = logging.getLogger('market_data_collector')

class MarketDataCollector:
    """
    Class for collecting market data from cryptocurrency exchanges.
    Currently supports Binance exchange.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the MarketDataCollector.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses the path from config.
        """
        self.db_path = db_path or DATABASE['path']
        self.base_url = API['binance']['base_url']
        self.api_key = API['binance']['api_key']
        self.api_secret = API['binance']['api_secret']
        self.rate_limit = API['binance']['rate_limit']
        self.last_request_time = 0
        self.request_count = 0
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info("MarketDataCollector initialized")
    
    def _init_database(self):
        """Initialize the SQLite database with required tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create OHLCV table
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {DATABASE['tables']['ohlcv']} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            UNIQUE(symbol, timeframe, timestamp)
        )
        ''')
        
        # Create index for faster queries
        cursor.execute(f'''
        CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_timestamp 
        ON {DATABASE['tables']['ohlcv']} (symbol, timeframe, timestamp)
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    
    def _respect_rate_limit(self):
        """Respect the API rate limit by adding delays if necessary."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        # Reset counter if a minute has passed
        if time_since_last_request > 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        # If approaching rate limit, wait until the minute is up
        if self.request_count >= self.rate_limit['max_requests'] - 5:
            sleep_time = 60 - time_since_last_request
            if sleep_time > 0:
                logger.warning(f"Rate limit approaching, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()
        
        # Small delay between requests to be gentle on the API
        if time_since_last_request < 0.1:
            time.sleep(0.1 - time_since_last_request)
        
        self.request_count += 1
        self.last_request_time = time.time()
    
    def fetch_historical_data(self, symbol: str, timeframe: str, 
                             limit: int = 1000, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe interval (e.g., '1m', '1h')
            limit: Number of candles to fetch (max 1000)
            end_time: End time in milliseconds. If None, fetches the most recent data.
            
        Returns:
            DataFrame with OHLCV data
        """
        self._respect_rate_limit()
        
        endpoint = f"{self.base_url}/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': timeframe,
            'limit': limit
        }
        
        if end_time:
            params['endTime'] = end_time
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # Add symbol and timeframe columns
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            logger.info(f"Fetched {len(df)} {timeframe} candles for {symbol}")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()
    
    def save_data_to_db(self, df: pd.DataFrame):
        """
        Save OHLCV data to the SQLite database.
        
        Args:
            df: DataFrame with OHLCV data
        """
        if df.empty:
            logger.warning("No data to save to database")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Select only the columns we need
            df_to_save = df[['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Save to database
            df_to_save.to_sql(
                DATABASE['tables']['ohlcv'], 
                conn, 
                if_exists='append', 
                index=False,
                method='multi'
            )
            
            conn.close()
            logger.info(f"Saved {len(df)} rows to database")
        
        except Exception as e:
            logger.error(f"Error saving data to database: {str(e)}")
    
    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        """
        Get the latest timestamp for a symbol and timeframe from the database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            
        Returns:
            Latest timestamp in milliseconds or None if no data exists
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
            SELECT MAX(timestamp) FROM {DATABASE['tables']['ohlcv']}
            WHERE symbol = ? AND timeframe = ?
            ''', (symbol, timeframe))
            
            result = cursor.fetchone()[0]
            conn.close()
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting latest timestamp: {str(e)}")
            return None
    
    def update_historical_data(self, symbol: str, timeframe: str):
        """
        Update historical data for a symbol and timeframe.
        Fetches data since the latest timestamp in the database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
        """
        latest_timestamp = self.get_latest_timestamp(symbol, timeframe)
        
        if latest_timestamp:
            # Add one candle interval to avoid duplicate
            interval_ms = self._timeframe_to_milliseconds(timeframe)
            start_time = latest_timestamp + interval_ms
            
            # Fetch data since the latest timestamp
            df = self.fetch_historical_data(symbol, timeframe, end_time=None)
        else:
            # No data exists, fetch the maximum amount
            df = self.fetch_historical_data(symbol, timeframe, limit=1000)
        
        if not df.empty:
            self.save_data_to_db(df)
    
    def _timeframe_to_milliseconds(self, timeframe: str) -> int:
        """
        Convert a timeframe string to milliseconds.
        
        Args:
            timeframe: Timeframe interval (e.g., '1m', '1h')
            
        Returns:
            Milliseconds equivalent of the timeframe
        """
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value * 60 * 1000
        elif unit == 'h':
            return value * 60 * 60 * 1000
        elif unit == 'd':
            return value * 24 * 60 * 60 * 1000
        elif unit == 'w':
            return value * 7 * 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"Unknown timeframe unit: {unit}")
    
    def update_all_data(self):
        """Update historical data for all configured symbols and timeframes."""
        for symbol in TRADING_PAIRS:
            for timeframe, config in TIMEFRAMES.items():
                logger.info(f"Updating {symbol} {timeframe} data")
                self.update_historical_data(symbol, timeframe)
    
    def get_ohlcv_data(self, symbol: str, timeframe: str, limit: int = None) -> pd.DataFrame:
        """
        Retrieve OHLCV data from the database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            limit: Number of candles to retrieve (most recent). If None, retrieves all.
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = f'''
            SELECT timestamp, open, high, low, close, volume
            FROM {DATABASE['tables']['ohlcv']}
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            '''
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
            conn.close()
            
            # Sort by timestamp ascending
            df = df.sort_values('timestamp')
            
            return df
        
        except Exception as e:
            logger.error(f"Error retrieving OHLCV data: {str(e)}")
            return pd.DataFrame()
    
    def cleanup_old_data(self):
        """Remove old data that exceeds the keep_periods configuration."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for symbol in TRADING_PAIRS:
                for timeframe, config in TIMEFRAMES.items():
                    # Calculate the cutoff timestamp
                    keep_periods = config['keep_periods']
                    interval_ms = self._timeframe_to_milliseconds(timeframe)
                    cutoff_timestamp = int(time.time() * 1000) - (keep_periods * interval_ms)
                    
                    # Delete old data
                    cursor.execute(f'''
                    DELETE FROM {DATABASE['tables']['ohlcv']}
                    WHERE symbol = ? AND timeframe = ? AND timestamp < ?
                    ''', (symbol, timeframe, cutoff_timestamp))
                    
                    deleted_count = cursor.rowcount
                    if deleted_count > 0:
                        logger.info(f"Deleted {deleted_count} old records for {symbol} {timeframe}")
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('/home/ubuntu/crypto_trading_model/logs', exist_ok=True)
    
    # Test the data collector
    collector = MarketDataCollector()
    collector.update_all_data()
    
    # Print sample data
    for symbol in TRADING_PAIRS[:1]:  # Just the first symbol
        for timeframe in list(TIMEFRAMES.keys())[:2]:  # Just the first two timeframes
            df = collector.get_ohlcv_data(symbol, timeframe, limit=5)
            print(f"\n{symbol} {timeframe} data:")
            print(df)
