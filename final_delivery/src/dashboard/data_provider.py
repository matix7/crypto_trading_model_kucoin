"""
Data provider for the dashboard module.
Retrieves and processes data from the database and API.
"""

import os
import json
import time
import sqlite3
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import requests

from .config import DATABASE, API, DASHBOARD

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dashboard_data_provider')

class DataProvider:
    """
    Data provider for the dashboard module.
    Retrieves and processes data from the database and API.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the data provider.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path or DATABASE['DB_PATH']
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # API configuration
        self.api_base_url = API['BASE_URL']
        self.api_endpoints = API['ENDPOINTS']
        self.api_headers = API['HEADERS']
        self.api_timeout = API['TIMEOUT'] / 1000  # Convert to seconds
        self.api_retry_count = API['RETRY_COUNT']
        self.api_retry_delay = API['RETRY_DELAY'] / 1000  # Convert to seconds
        
        logger.info("Data provider initialized")
    
    def get_system_status(self) -> Dict:
        """
        Get current system status from the API.
        
        Returns:
            Dictionary with system status
        """
        try:
            # Call API
            endpoint = self.api_endpoints['status']
            url = f"{self.api_base_url}{endpoint}"
            
            response = self._make_api_request('GET', url)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error getting system status: {response.status_code} - {response.text}")
                return {
                    'status': 'error',
                    'message': f"Error getting system status: {response.status_code}"
                }
        
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_open_positions(self) -> List[Dict]:
        """
        Get open positions from the API.
        
        Returns:
            List of open positions
        """
        try:
            # Call API
            endpoint = self.api_endpoints['positions']
            url = f"{self.api_base_url}{endpoint}"
            
            response = self._make_api_request('GET', url)
            
            if response.status_code == 200:
                return response.json().get('positions', [])
            else:
                logger.error(f"Error getting open positions: {response.status_code} - {response.text}")
                return []
        
        except Exception as e:
            logger.error(f"Error getting open positions: {str(e)}")
            return []
    
    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """
        Get trade history from the API.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of trades
        """
        try:
            # Call API
            endpoint = self.api_endpoints['trades']
            url = f"{self.api_base_url}{endpoint}?limit={limit}"
            
            response = self._make_api_request('GET', url)
            
            if response.status_code == 200:
                return response.json().get('trades', [])
            else:
                logger.error(f"Error getting trade history: {response.status_code} - {response.text}")
                return []
        
        except Exception as e:
            logger.error(f"Error getting trade history: {str(e)}")
            return []
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics from the API.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Call API
            endpoint = self.api_endpoints['metrics']
            url = f"{self.api_base_url}{endpoint}"
            
            response = self._make_api_request('GET', url)
            
            if response.status_code == 200:
                return response.json().get('metrics', {})
            else:
                logger.error(f"Error getting performance metrics: {response.status_code} - {response.text}")
                return {}
        
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {}
    
    def get_equity_curve(self, days: int = 30) -> List[Dict]:
        """
        Get equity curve data from the database.
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            List of equity data points
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            
            # Calculate start timestamp
            start_timestamp = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            # Query database
            query = f"""
            SELECT timestamp, equity
            FROM {DATABASE['TABLES']['paper_balance']}
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(query, conn, params=(start_timestamp,))
            
            # Close connection
            conn.close()
            
            # Convert timestamp to date
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Format for chart
            result = []
            for _, row in df.iterrows():
                result.append({
                    'date': row['date'].strftime('%Y-%m-%d %H:%M:%S'),
                    'equity': row['equity']
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting equity curve: {str(e)}")
            return []
    
    def get_daily_returns(self, days: int = 30) -> List[Dict]:
        """
        Get daily returns data from the database.
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            List of daily return data points
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            
            # Calculate start timestamp
            start_timestamp = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            # Query database
            query = f"""
            SELECT timestamp, metric_value
            FROM {DATABASE['TABLES']['paper_performance']}
            WHERE metric_name = 'daily_return' AND timestamp >= ?
            ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(query, conn, params=(start_timestamp,))
            
            # Close connection
            conn.close()
            
            # Convert timestamp to date
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Format for chart
            result = []
            for _, row in df.iterrows():
                result.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'return': row['metric_value'] * 100  # Convert to percentage
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting daily returns: {str(e)}")
            return []
    
    def get_drawdown(self, days: int = 30) -> List[Dict]:
        """
        Calculate and get drawdown data.
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            List of drawdown data points
        """
        try:
            # Get equity curve
            equity_data = self.get_equity_curve(days)
            
            if not equity_data:
                return []
            
            # Convert to DataFrame
            df = pd.DataFrame(equity_data)
            df['equity'] = df['equity'].astype(float)
            
            # Calculate drawdown
            df['peak'] = df['equity'].cummax()
            df['drawdown'] = (df['equity'] - df['peak']) / df['peak'] * 100  # Convert to percentage
            
            # Format for chart
            result = []
            for _, row in df.iterrows():
                result.append({
                    'date': row['date'],
                    'drawdown': row['drawdown']
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting drawdown: {str(e)}")
            return []
    
    def get_win_rate_data(self) -> List[Dict]:
        """
        Get win rate data for pie chart.
        
        Returns:
            List of win rate data points
        """
        try:
            # Get trade history
            trades = self.get_trade_history(1000)  # Get a large sample
            
            if not trades:
                return []
            
            # Count winning and losing trades
            winning_trades = sum(1 for trade in trades if trade.get('profit_loss', 0) > 0)
            losing_trades = sum(1 for trade in trades if trade.get('profit_loss', 0) < 0)
            
            # Format for chart
            result = [
                {'name': 'Winning Trades', 'value': winning_trades},
                {'name': 'Losing Trades', 'value': losing_trades}
            ]
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting win rate data: {str(e)}")
            return []
    
    def get_profit_distribution(self) -> List[Dict]:
        """
        Get profit distribution data for histogram.
        
        Returns:
            List of profit distribution data points
        """
        try:
            # Get trade history
            trades = self.get_trade_history(1000)  # Get a large sample
            
            if not trades:
                return []
            
            # Extract profit/loss percentages
            profit_loss_percentages = [
                trade.get('profit_loss_percentage', 0) * 100  # Convert to percentage
                for trade in trades
                if trade.get('status') == 'CLOSED'
            ]
            
            # Create histogram
            hist, bin_edges = np.histogram(profit_loss_percentages, bins=20)
            
            # Format for chart
            result = []
            for i in range(len(hist)):
                result.append({
                    'range': f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}",
                    'frequency': int(hist[i])
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting profit distribution: {str(e)}")
            return []
    
    def get_performance_metrics_radar(self) -> List[Dict]:
        """
        Get performance metrics data for radar chart.
        
        Returns:
            List of performance metrics data points
        """
        try:
            # Get performance metrics
            metrics = self.get_performance_metrics()
            
            if not metrics:
                return []
            
            # Select metrics for radar chart
            selected_metrics = {
                'Win Rate': metrics.get('win_rate', 0) * 100,  # Convert to percentage
                'Profit Factor': min(metrics.get('profit_factor', 0), 5),  # Cap at 5 for better visualization
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Calmar Ratio': metrics.get('calmar_ratio', 0),
                'Success Rate': metrics.get('success_rate', 0) * 100,  # Convert to percentage
                'Expectancy': metrics.get('expectancy', 0) * 100  # Scale for better visualization
            }
            
            # Format for chart
            result = []
            for metric, value in selected_metrics.items():
                result.append({
                    'metric': metric,
                    'value': value
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting performance metrics radar: {str(e)}")
            return []
    
    def get_trade_history_scatter(self) -> List[Dict]:
        """
        Get trade history data for scatter plot.
        
        Returns:
            List of trade history data points
        """
        try:
            # Get trade history
            trades = self.get_trade_history(100)
            
            if not trades:
                return []
            
            # Format for chart
            result = []
            for trade in trades:
                if trade.get('status') == 'CLOSED':
                    result.append({
                        'date': datetime.fromtimestamp(trade.get('exit_time', 0) / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                        'profit': trade.get('profit_loss', 0),
                        'trading_pair': trade.get('trading_pair', ''),
                        'side': trade.get('side', ''),
                        'duration': trade.get('trade_duration', 0) / 3600000  # Convert to hours
                    })
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting trade history scatter: {str(e)}")
            return []
    
    def get_position_size_history(self) -> List[Dict]:
        """
        Get position size history data for bar chart.
        
        Returns:
            List of position size history data points
        """
        try:
            # Get trade history
            trades = self.get_trade_history(100)
            
            if not trades:
                return []
            
            # Format for chart
            result = []
            for trade in trades:
                result.append({
                    'date': datetime.fromtimestamp(trade.get('entry_time', 0) / 1000).strftime('%Y-%m-%d'),
                    'position_size': trade.get('position_size', 0),
                    'trading_pair': trade.get('trading_pair', '')
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting position size history: {str(e)}")
            return []
    
    def get_trading_pairs_distribution(self) -> List[Dict]:
        """
        Get trading pairs distribution data for pie chart.
        
        Returns:
            List of trading pairs distribution data points
        """
        try:
            # Get trade history
            trades = self.get_trade_history(1000)  # Get a large sample
            
            if not trades:
                return []
            
            # Count trades by trading pair
            pair_counts = {}
            for trade in trades:
                pair = trade.get('trading_pair', 'Unknown')
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            
            # Format for chart
            result = []
            for pair, count in pair_counts.items():
                result.append({
                    'name': pair,
                    'value': count
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting trading pairs distribution: {str(e)}")
            return []
    
    def get_trade_duration_histogram(self) -> List[Dict]:
        """
        Get trade duration data for histogram.
        
        Returns:
            List of trade duration data points
        """
        try:
            # Get trade history
            trades = self.get_trade_history(1000)  # Get a large sample
            
            if not trades:
                return []
            
            # Extract trade durations
            durations = [
                trade.get('trade_duration', 0) / 3600000  # Convert to hours
                for trade in trades
                if trade.get('status') == 'CLOSED'
            ]
            
            # Create histogram
            hist, bin_edges = np.histogram(durations, bins=10)
            
            # Format for chart
            result = []
            for i in range(len(hist)):
                result.append({
                    'range': f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}",
                    'frequency': int(hist[i])
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting trade duration histogram: {str(e)}")
            return []
    
    def get_all_dashboard_data(self) -> Dict:
        """
        Get all data needed for the dashboard.
        
        Returns:
            Dictionary with all dashboard data
        """
        try:
            # Get system status
            system_status = self.get_system_status()
            
            # Get open positions
            open_positions = self.get_open_positions()
            
            # Get trade history
            trade_history = self.get_trade_history(DASHBOARD['MAX_TRADES_DISPLAY'])
            
            # Get performance metrics
            performance_metrics = self.get_performance_metrics()
            
            # Get chart data
            equity_curve = self.get_equity_curve()
            daily_returns = self.get_daily_returns()
            drawdown = self.get_drawdown()
            win_rate = self.get_win_rate_data()
            profit_distribution = self.get_profit_distribution()
            performance_metrics_radar = self.get_performance_metrics_radar()
            trade_history_scatter = self.get_trade_history_scatter()
            position_size_history = self.get_position_size_history()
            trading_pairs_distribution = self.get_trading_pairs_distribution()
            trade_duration_histogram = self.get_trade_duration_histogram()
            
            # Combine all data
            return {
                'system_status': system_status,
                'open_positions': open_positions,
                'trade_history': trade_history,
                'performance_metrics': performance_metrics,
                'chart_data': {
                    'equity_curve': equity_curve,
                    'daily_returns': daily_returns,
                    'drawdown': drawdown,
                    'win_rate': win_rate,
                    'profit_distribution': profit_distribution,
                    'performance_metrics_radar': performance_metrics_radar,
                    'trade_history_scatter': trade_history_scatter,
                    'position_size_history': position_size_history,
                    'trading_pairs_distribution': trading_pairs_distribution,
                    'trade_duration_histogram': trade_duration_histogram
                }
            }
        
        except Exception as e:
            logger.error(f"Error getting all dashboard data: {str(e)}")
            return {
                'error': str(e)
            }
    
    def start_trading(self) -> Dict:
        """
        Start the trading system.
        
        Returns:
            Response from the API
        """
        try:
            # Call API
            endpoint = self.api_endpoints['start']
            url = f"{self.api_base_url}{endpoint}"
            
            response = self._make_api_request('POST', url, data={})
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error starting trading: {response.status_code} - {response.text}")
                return {
                    'status': 'error',
                    'message': f"Error starting trading: {response.status_code}"
                }
        
        except Exception as e:
            logger.error(f"Error starting trading: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def stop_trading(self) -> Dict:
        """
        Stop the trading system.
        
        Returns:
            Response from the API
        """
        try:
            # Call API
            endpoint = self.api_endpoints['stop']
            url = f"{self.api_base_url}{endpoint}"
            
            response = self._make_api_request('POST', url, data={})
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error stopping trading: {response.status_code} - {response.text}")
                return {
                    'status': 'error',
                    'message': f"Error stopping trading: {response.status_code}"
                }
        
        except Exception as e:
            logger.error(f"Error stopping trading: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def update_config(self, config: Dict) -> Dict:
        """
        Update the trading system configuration.
        
        Args:
            config: New configuration
            
        Returns:
            Response from the API
        """
        try:
            # Call API
            endpoint = self.api_endpoints['config']
            url = f"{self.api_base_url}{endpoint}"
            
            response = self._make_api_request('POST', url, data={'config': config})
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error updating config: {response.status_code} - {response.text}")
                return {
                    'status': 'error',
                    'message': f"Error updating config: {response.status_code}"
                }
        
        except Exception as e:
            logger.error(f"Error updating config: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _make_api_request(self, method: str, url: str, data: Dict = None) -> requests.Response:
        """
        Make an API request with retry logic.
        
        Args:
            method: HTTP method
            url: URL
            data: Request data
            
        Returns:
            Response from the API
        """
        for attempt in range(self.api_retry_count + 1):
            try:
                if method.upper() == 'GET':
                    response = requests.get(
                        url,
                        headers=self.api_headers,
                        timeout=self.api_timeout
                    )
                elif method.upper() == 'POST':
                    response = requests.post(
                        url,
                        headers=self.api_headers,
                        json=data,
                        timeout=self.api_timeout
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                return response
            
            except Exception as e:
                if attempt < self.api_retry_count:
                    logger.warning(f"API request failed, retrying ({attempt+1}/{self.api_retry_count}): {str(e)}")
                    time.sleep(self.api_retry_delay)
                else:
                    logger.error(f"API request failed after {self.api_retry_count} retries: {str(e)}")
                    raise
