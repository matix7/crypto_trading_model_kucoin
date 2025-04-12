"""
Market data collector for cryptocurrency exchanges.
Fetches historical and real-time data from KuCoin.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

# Import KuCoin API client
from kucoin.client import Market
from kucoin.client import Trade

from .config import EXCHANGE, TRADING_PAIRS, TIMEFRAMES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('market_data_collector')

class MarketDataCollector:
    """
    Market data collector for cryptocurrency exchanges.
    Fetches historical and real-time data from KuCoin.
    """
    
    def __init__(self, api_key=None, api_secret=None, api_passphrase=None, is_sandbox=True):
        """
        Initialize the market data collector.
        
        Args:
            api_key: KuCoin API key
            api_secret: KuCoin API secret
            api_passphrase: KuCoin API passphrase
            is_sandbox: Whether to use the sandbox environment
        """
        self.is_sandbox = is_sandbox
        
        # Initialize KuCoin clients
        self.market_client = Market(url='https://api-sandbox.kucoin.com' if is_sandbox else 'https://api.kucoin.com')
        
        if api_key and api_secret and api_passphrase:
            self.trade_client = Trade(key=api_key, secret=api_secret, passphrase=api_passphrase, 
                                    is_sandbox=is_sandbox)
        else:
            self.trade_client = None
        
        logger.info(f"MarketDataCollector initialized with sandbox={is_sandbox}")
    
    def get_historical_klines(self, symbol, interval, start_time=None, end_time=None, limit=500):
        """
        Get historical klines (candlesticks) for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USDT')
            interval: Timeframe interval (e.g., '1m', '5m', '1h')
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Maximum number of klines to return
            
        Returns:
            DataFrame with historical klines
        """
        try:
            # Convert interval to KuCoin format
            kucoin_interval = self._convert_interval_to_kucoin(interval)
            
            # Convert timestamps to seconds for KuCoin API
            start_time_sec = int(start_time / 1000) if start_time else None
            end_time_sec = int(end_time / 1000) if end_time else None
            
            # Get klines from KuCoin
            klines = self.market_client.get_kline(symbol, kucoin_interval, start=start_time_sec, end=end_time_sec)
            
            # Convert to DataFrame
            columns = ['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover']
            df = pd.DataFrame(klines, columns=columns)
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            for col in ['open', 'close', 'high', 'low', 'volume', 'turnover']:
                df[col] = df[col].astype(float)
            
            # Rename turnover to quote_asset_volume for consistency
            df.rename(columns={'turnover': 'quote_asset_volume'}, inplace=True)
            
            # Sort by timestamp
            df.sort_values('timestamp', inplace=True)
            
            # Limit the number of rows
            if limit and len(df) > limit:
                df = df.tail(limit)
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting historical klines: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_klines(self, symbol, interval, limit=100):
        """
        Get the latest klines (candlesticks) for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USDT')
            interval: Timeframe interval (e.g., '1m', '5m', '1h')
            limit: Maximum number of klines to return
            
        Returns:
            DataFrame with latest klines
        """
        try:
            # Convert interval to KuCoin format
            kucoin_interval = self._convert_interval_to_kucoin(interval)
            
            # Get klines from KuCoin
            klines = self.market_client.get_kline(symbol, kucoin_interval, limit=limit)
            
            # Convert to DataFrame
            columns = ['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover']
            df = pd.DataFrame(klines, columns=columns)
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            for col in ['open', 'close', 'high', 'low', 'volume', 'turnover']:
                df[col] = df[col].astype(float)
            
            # Rename turnover to quote_asset_volume for consistency
            df.rename(columns={'turnover': 'quote_asset_volume'}, inplace=True)
            
            # Sort by timestamp
            df.sort_values('timestamp', inplace=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting latest klines: {str(e)}")
            return pd.DataFrame()
    
    def get_ticker(self, symbol):
        """
        Get ticker information for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USDT')
            
        Returns:
            Dictionary with ticker information
        """
        try:
            # Get ticker from KuCoin
            ticker = self.market_client.get_ticker(symbol)
            
            # Format the response
            result = {
                'symbol': symbol,
                'price': float(ticker['price']),
                'volume': float(ticker['vol']),
                'change_percent': float(ticker['changeRate']) * 100,
                'high': float(ticker['high']),
                'low': float(ticker['low']),
                'timestamp': datetime.now()
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting ticker: {str(e)}")
            return {}
    
    def get_order_book(self, symbol, limit=100):
        """
        Get order book for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USDT')
            limit: Maximum number of bids and asks to return
            
        Returns:
            Dictionary with order book information
        """
        try:
            # Get order book from KuCoin
            order_book = self.market_client.get_part_order(symbol, limit)
            
            # Format the response
            bids = [[float(price), float(qty)] for price, qty in order_book['bids']]
            asks = [[float(price), float(qty)] for price, qty in order_book['asks']]
            
            result = {
                'symbol': symbol,
                'bids': bids,
                'asks': asks,
                'timestamp': datetime.now()
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting order book: {str(e)}")
            return {}
    
    def get_recent_trades(self, symbol, limit=100):
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USDT')
            limit: Maximum number of trades to return
            
        Returns:
            DataFrame with recent trades
        """
        try:
            # Get recent trades from KuCoin
            trades = self.market_client.get_trade_histories(symbol)
            
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            
            # Convert types
            df['time'] = pd.to_datetime(df['time'], unit='ns')
            df['price'] = df['price'].astype(float)
            df['size'] = df['size'].astype(float)
            
            # Rename columns for consistency
            df.rename(columns={
                'time': 'timestamp',
                'size': 'quantity',
                'side': 'side'
            }, inplace=True)
            
            # Limit the number of rows
            if limit and len(df) > limit:
                df = df.head(limit)
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting recent trades: {str(e)}")
            return pd.DataFrame()
    
    def get_account_balance(self):
        """
        Get account balance.
        
        Returns:
            DataFrame with account balance
        """
        try:
            if not self.trade_client:
                logger.warning("Trade client not initialized. Cannot get account balance.")
                return pd.DataFrame()
            
            # Get account balance from KuCoin
            accounts = self.trade_client.get_accounts()
            
            # Convert to DataFrame
            df = pd.DataFrame(accounts)
            
            # Convert types
            df['balance'] = df['balance'].astype(float)
            df['available'] = df['available'].astype(float)
            df['holds'] = df['holds'].astype(float)
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting account balance: {str(e)}")
            return pd.DataFrame()
    
    def get_symbol_info(self, symbol=None):
        """
        Get symbol information.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USDT')
            
        Returns:
            Dictionary with symbol information or DataFrame with all symbols
        """
        try:
            # Get symbols from KuCoin
            symbols = self.market_client.get_symbol_list()
            
            if symbol:
                # Find the specific symbol
                for sym in symbols:
                    if sym['symbol'] == symbol:
                        return sym
                
                logger.warning(f"Symbol {symbol} not found")
                return {}
            else:
                # Return all symbols as DataFrame
                return pd.DataFrame(symbols)
        
        except Exception as e:
            logger.error(f"Error getting symbol info: {str(e)}")
            return {} if symbol else pd.DataFrame()
    
    def get_exchange_info(self):
        """
        Get exchange information.
        
        Returns:
            Dictionary with exchange information
        """
        try:
            # Get exchange info from KuCoin
            symbols = self.market_client.get_symbol_list()
            
            # Format the response
            result = {
                'exchange': 'KuCoin',
                'timezone': 'UTC',
                'server_time': int(time.time() * 1000),
                'symbols': symbols
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting exchange info: {str(e)}")
            return {}
    
    def get_all_tickers(self):
        """
        Get tickers for all symbols.
        
        Returns:
            DataFrame with all tickers
        """
        try:
            # Get all tickers from KuCoin
            tickers = self.market_client.get_all_tickers()
            
            # Extract ticker data
            ticker_data = tickers['ticker']
            
            # Convert to DataFrame
            df = pd.DataFrame(ticker_data)
            
            # Convert types
            for col in ['buy', 'sell', 'last', 'vol', 'volValue', 'high', 'low', 'changeRate', 'changePrice']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            
            # Rename columns for consistency
            df.rename(columns={
                'symbol': 'symbol',
                'last': 'price',
                'vol': 'volume',
                'changeRate': 'change_percent',
                'high': 'high',
                'low': 'low'
            }, inplace=True)
            
            # Calculate change percent
            if 'change_percent' in df.columns:
                df['change_percent'] = df['change_percent'] * 100
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting all tickers: {str(e)}")
            return pd.DataFrame()
    
    def _convert_interval_to_kucoin(self, interval):
        """
        Convert interval to KuCoin format.
        
        Args:
            interval: Timeframe interval (e.g., '1m', '5m', '1h')
            
        Returns:
            KuCoin interval format
        """
        mapping = {
            '1m': '1min',
            '3m': '3min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1hour',
            '2h': '2hour',
            '4h': '4hour',
            '6h': '6hour',
            '12h': '12hour',
            '1d': '1day',
            '1w': '1week'
        }
        return mapping.get(interval, '1hour')
    
    def download_historical_data(self, symbol, interval, start_date, end_date=None, save_path=None):
        """
        Download historical data for a symbol and save to CSV.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USDT')
            interval: Timeframe interval (e.g., '1m', '5m', '1h')
            start_date: Start date (e.g., '2023-01-01')
            end_date: End date (e.g., '2023-12-31'), defaults to current date
            save_path: Path to save the CSV file, defaults to current directory
            
        Returns:
            Path to the saved CSV file
        """
        try:
            # Convert dates to timestamps
            start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            
            if end_date:
                end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            else:
                end_timestamp = int(datetime.now().timestamp() * 1000)
            
            # Get historical klines
            df = self.get_historical_klines(symbol, interval, start_timestamp, end_timestamp)
            
            if df.empty:
                logger.warning(f"No data found for {symbol} {interval} from {start_date} to {end_date}")
                return None
            
            # Save to CSV
            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                file_path = save_path
            else:
                os.makedirs('data', exist_ok=True)
                file_path = f"data/{symbol.replace('-', '')}_{interval}_{start_date}_{end_date or 'now'}.csv"
            
            df.to_csv(file_path, index=False)
            logger.info(f"Historical data saved to {file_path}")
            
            return file_path
        
        except Exception as e:
            logger.error(f"Error downloading historical data: {str(e)}")
            return None
    
    def get_data_for_all_pairs(self, interval='1h', limit=100):
        """
        Get latest data for all configured trading pairs.
        
        Args:
            interval: Timeframe interval (e.g., '1m', '5m', '1h')
            limit: Maximum number of klines to return per pair
            
        Returns:
            Dictionary with DataFrames for each trading pair
        """
        result = {}
        
        for pair in TRADING_PAIRS:
            logger.info(f"Getting data for {pair} {interval}")
            df = self.get_latest_klines(pair, interval, limit)
            
            if not df.empty:
                result[pair] = df
            else:
                logger.warning(f"No data found for {pair} {interval}")
        
        return result
    
    def get_current_prices(self):
        """
        Get current prices for all configured trading pairs.
        
        Returns:
            Dictionary with current prices
        """
        result = {}
        
        try:
            # Get all tickers
            tickers = self.get_all_tickers()
            
            if tickers.empty:
                logger.warning("No tickers found")
                return result
            
            # Filter for configured trading pairs
            for pair in TRADING_PAIRS:
                ticker_row = tickers[tickers['symbol'] == pair]
                
                if not ticker_row.empty:
                    result[pair] = float(ticker_row.iloc[0]['price'])
                else:
                    logger.warning(f"No ticker found for {pair}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting current prices: {str(e)}")
            return result
