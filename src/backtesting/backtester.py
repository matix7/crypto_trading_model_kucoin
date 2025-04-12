"""
Backtester class for backtesting trading strategies.
"""

import logging
import os
import time
import json
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .config import (
    BACKTESTING, PERFORMANCE_METRICS, MARKET_CONDITIONS,
    STRESS_TEST_SCENARIOS, WALK_FORWARD_ANALYSIS,
    MONTE_CARLO_SIMULATION, REPORTING, DATABASE
)

# Import other modules
import sys
sys.path.append('/home/ubuntu/crypto_trading_model/src')
from data.market_data_collector import MarketDataCollector
from data.technical_indicator_calculator import TechnicalIndicatorCalculator
from data.feature_engineering import FeatureEngineer
from analysis.strategy_manager import StrategyManager
from sentiment.sentiment_manager import SentimentManager
from prediction.ensemble_manager import EnsembleManager
from risk.risk_manager import RiskManager
from risk.position_sizer import PositionSizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/backtesting.log',
    filemode='a'
)
logger = logging.getLogger('backtester')

class Backtester:
    """
    Backtester class for backtesting trading strategies.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the Backtester.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path or '/home/ubuntu/crypto_trading_model/data/market_data.db'
        
        # Backtesting parameters
        self.initial_capital = BACKTESTING['DEFAULT_INITIAL_CAPITAL']
        self.commission = BACKTESTING['DEFAULT_COMMISSION']
        self.slippage = BACKTESTING['DEFAULT_SLIPPAGE']
        self.timeframe = BACKTESTING['DEFAULT_TIMEFRAME']
        self.start_date = BACKTESTING['DEFAULT_START_DATE']
        self.end_date = BACKTESTING['DEFAULT_END_DATE']
        self.coins = BACKTESTING['DEFAULT_COINS']
        self.max_open_trades = BACKTESTING['MAX_OPEN_TRADES']
        
        # Feature flags
        self.enable_position_sizing = BACKTESTING['ENABLE_POSITION_SIZING']
        self.enable_risk_management = BACKTESTING['ENABLE_RISK_MANAGEMENT']
        self.enable_trailing_stop = BACKTESTING['ENABLE_TRAILING_STOP']
        self.enable_partial_take_profit = BACKTESTING['ENABLE_PARTIAL_TAKE_PROFIT']
        self.enable_compounding = BACKTESTING['ENABLE_COMPOUNDING']
        self.enable_portfolio_rebalancing = BACKTESTING['ENABLE_PORTFOLIO_REBALANCING']
        self.enable_market_condition_detection = BACKTESTING['ENABLE_MARKET_CONDITION_DETECTION']
        self.enable_sentiment_analysis = BACKTESTING['ENABLE_SENTIMENT_ANALYSIS']
        self.enable_ensemble_predictions = BACKTESTING['ENABLE_ENSEMBLE_PREDICTIONS']
        self.enable_dynamic_optimization = BACKTESTING['ENABLE_DYNAMIC_OPTIMIZATION']
        self.enable_walk_forward_analysis = BACKTESTING['ENABLE_WALK_FORWARD_ANALYSIS']
        self.enable_monte_carlo_simulation = BACKTESTING['ENABLE_MONTE_CARLO_SIMULATION']
        self.enable_stress_testing = BACKTESTING['ENABLE_STRESS_TESTING']
        self.enable_benchmark_comparison = BACKTESTING['ENABLE_BENCHMARK_COMPARISON']
        
        # Optimization settings
        self.optimization_metric = BACKTESTING['OPTIMIZATION_METRIC']
        self.optimization_parameters = BACKTESTING['OPTIMIZATION_PARAMETERS']
        self.optimization_ranges = BACKTESTING['OPTIMIZATION_RANGES']
        
        # Benchmark settings
        self.benchmark_strategy = BACKTESTING['BENCHMARK_STRATEGY']
        self.benchmark_coins = BACKTESTING['BENCHMARK_COINS']
        self.benchmark_weights = BACKTESTING['BENCHMARK_WEIGHTS']
        
        # Results storage
        self.results = {}
        self.trades = []
        self.equity_curve = []
        self.drawdowns = []
        self.metrics = {}
        self.optimization_results = []
        self.monte_carlo_results = []
        self.walk_forward_results = []
        
        # Component instances
        self.market_data_collector = None
        self.technical_indicator_calculator = None
        self.feature_engineer = None
        self.strategy_manager = None
        self.sentiment_manager = None
        self.ensemble_manager = None
        self.risk_manager = None
        self.position_sizer = None
        
        # Ensure logs directory exists
        os.makedirs('/home/ubuntu/crypto_trading_model/logs', exist_ok=True)
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Ensure reports directory exists
        os.makedirs(REPORTING['REPORT_DIRECTORY'], exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info("Backtester initialized")
    
    def _init_database(self):
        """Initialize the SQLite database with required tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create backtest results table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['backtest_results_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                coins TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                initial_capital REAL NOT NULL,
                final_capital REAL NOT NULL,
                total_return REAL NOT NULL,
                annualized_return REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                win_rate REAL NOT NULL,
                profit_factor REAL NOT NULL,
                parameters TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Create backtest trades table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['backtest_trades_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id TEXT NOT NULL,
                trade_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                coin TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                position_size REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                profit_loss REAL NOT NULL,
                profit_loss_percentage REAL NOT NULL,
                trade_duration INTEGER NOT NULL,
                market_condition TEXT,
                signal_strength REAL,
                confidence REAL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Create backtest metrics table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['backtest_metrics_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Create optimization results table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['optimization_results_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                optimization_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                parameters TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Create Monte Carlo results table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['monte_carlo_results_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                simulation_number INTEGER NOT NULL,
                final_equity REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                win_rate REAL NOT NULL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Create walk-forward results table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['walk_forward_results_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                window_number INTEGER NOT NULL,
                training_start_date TEXT NOT NULL,
                training_end_date TEXT NOT NULL,
                testing_start_date TEXT NOT NULL,
                testing_end_date TEXT NOT NULL,
                parameters TEXT NOT NULL,
                training_metric_value REAL NOT NULL,
                testing_metric_value REAL NOT NULL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database tables initialized")
        
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def initialize_components(self):
        """Initialize all required components for backtesting."""
        try:
            logger.info("Initializing components for backtesting")
            
            # Initialize market data collector
            self.market_data_collector = MarketDataCollector(
                db_path=self.db_path
            )
            
            # Initialize technical indicator calculator
            self.technical_indicator_calculator = TechnicalIndicatorCalculator(
                db_path=self.db_path
            )
            
            # Initialize feature engineer
            self.feature_engineer = FeatureEngineer(
                db_path=self.db_path
            )
            
            # Initialize strategy manager
            self.strategy_manager = StrategyManager(
                db_path=self.db_path
            )
            
            # Initialize sentiment manager if enabled
            if self.enable_sentiment_analysis:
                self.sentiment_manager = SentimentManager(
                    db_path=self.db_path
                )
            
            # Initialize ensemble manager if enabled
            if self.enable_ensemble_predictions:
                self.ensemble_manager = EnsembleManager(
                    coin=self.coins[0],  # Use first coin for initialization
                    timeframe=self.timeframe,
                    db_path=self.db_path
                )
            
            # Initialize risk manager if enabled
            if self.enable_risk_management:
                self.risk_manager = RiskManager(
                    db_path=self.db_path
                )
                
                # Set initial capital
                self.risk_manager.set_account_balance(self.initial_capital)
                
                # Initialize position sizer
                if self.enable_position_sizing:
                    self.position_sizer = PositionSizer(
                        risk_manager=self.risk_manager,
                        db_path=self.db_path
                    )
            
            logger.info("Components initialized for backtesting")
        
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def run_backtest(self, parameters: Dict = None) -> Dict:
        """
        Run a backtest with the specified parameters.
        
        Args:
            parameters: Dictionary of parameters to override defaults
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Generate backtest ID
            backtest_id = f"BT_{int(time.time())}"
            
            logger.info(f"Starting backtest {backtest_id}")
            
            # Override default parameters if provided
            if parameters:
                for key, value in parameters.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                        logger.info(f"Overriding parameter {key} with value {value}")
            
            # Initialize components
            self.initialize_components()
            
            # Fetch historical data for all coins
            all_data = {}
            for coin in self.coins:
                logger.info(f"Fetching historical data for {coin}")
                
                # Fetch OHLCV data
                data = self.market_data_collector.get_historical_data(
                    coin=coin,
                    timeframe=self.timeframe,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                
                if data.empty:
                    logger.warning(f"No data available for {coin}")
                    continue
                
                # Calculate technical indicators
                data = self.technical_indicator_calculator.calculate_indicators(data)
                
                # Engineer features
                data = self.feature_engineer.engineer_features(data)
                
                # Add sentiment data if enabled
                if self.enable_sentiment_analysis and self.sentiment_manager:
                    sentiment_data = self.sentiment_manager.get_historical_sentiment(
                        coin=coin,
                        start_date=self.start_date,
                        end_date=self.end_date
                    )
                    
                    if not sentiment_data.empty:
                        # Merge sentiment data with price data
                        data = pd.merge_asof(
                            data,
                            sentiment_data,
                            left_index=True,
                            right_index=True,
                            direction='backward'
                        )
                
                # Store data
                all_data[coin] = data
            
            # Check if we have data for at least one coin
            if not all_data:
                logger.error("No data available for any coin")
                return {
                    'status': 'error',
                    'message': 'No data available for any coin'
                }
            
            # Reset risk manager state
            if self.risk_manager:
                self.risk_manager.set_account_balance(self.initial_capital)
            
            # Initialize results storage
            self.trades = []
            self.equity_curve = [{'timestamp': pd.Timestamp(self.start_date), 'equity': self.initial_capital}]
            self.drawdowns = []
            
            # Run simulation
            self._run_simulation(backtest_id, all_data)
            
            # Calculate performance metrics
            self.metrics = self._calculate_performance_metrics()
            
            # Save results
            self._save_backtest_results(backtest_id, parameters)
            
            # Generate reports if enabled
            if REPORTING['GENERATE_HTML_REPORT']:
                self._generate_html_report(backtest_id)
            
            if REPORTING['GENERATE_PDF_REPORT']:
                self._generate_pdf_report(backtest_id)
            
            if REPORTING['GENERATE_CSV_RESULTS']:
                self._generate_csv_results(backtest_id)
            
            # Run Monte Carlo simulation if enabled
            if self.enable_monte_carlo_simulation:
                self._run_monte_carlo_simulation(backtest_id)
            
            logger.info(f"Backtest {backtest_id} completed")
            
            # Return results
            return {
                'status': 'success',
                'backtest_id': backtest_id,
                'metrics': self.metrics,
                'trades_count': len(self.trades),
                'final_equity': self.equity_curve[-1]['equity'] if self.equity_curve else self.initial_capital
            }
        
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error running backtest: {str(e)}"
            }
    
    def _run_simulation(self, backtest_id: str, all_data: Dict[str, pd.DataFrame]):
        """
        Run the backtest simulation.
        
        Args:
            backtest_id: Unique identifier for this backtest
            all_data: Dictionary of DataFrames with historical data for each coin
        """
        try:
            logger.info("Running simulation")
            
            # Get common date range across all coins
            common_dates = None
            for coin, data in all_data.items():
                if common_dates is None:
                    common_dates = set(data.index)
                else:
                    common_dates = common_dates.intersection(set(data.index))
            
            common_dates = sorted(list(common_dates))
            
            if not common_dates:
                logger.error("No common dates found across all coins")
                return
            
            # Initialize portfolio and open positions
            portfolio = {
                'cash': self.initial_capital,
                'positions': {}
            }
            
            open_positions = {}
            
            # Simulate trading day by day
            for current_date in tqdm(common_dates, desc="Simulating trading"):
                # Get data for current date
                current_data = {}
                for coin, data in all_data.items():
                    if current_date in data.index:
                        current_data[coin] = data.loc[current_date]
                
                # Skip if no data available for current date
                if not current_data:
                    continue
                
                # Detect market condition if enabled
                market_condition = 'UNKNOWN'
                if self.enable_market_condition_detection:
                    market_condition = self._detect_market_condition(current_data)
                
                # Process open positions
                self._process_open_positions(
                    open_positions, current_data, current_date, portfolio, market_condition
                )
                
                # Generate trading signals
                signals = self._generate_trading_signals(
                    current_data, current_date, market_condition
                )
                
                # Execute trading signals
                self._execute_trading_signals(
                    signals, current_data, current_date, portfolio, open_positions, market_condition
                )
                
                # Update equity curve
                total_equity = portfolio['cash']
                for coin, position in open_positions.items():
                    current_price = current_data[coin]['close']
                    position_value = position['quantity'] * current_price
                    total_equity += position_value
                
                self.equity_curve.append({
                    'timestamp': current_date,
                    'equity': total_equity
                })
                
                # Calculate drawdown
                peak_equity = max(entry['equity'] for entry in self.equity_curve)
                drawdown = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0.0
                
                self.drawdowns.append({
                    'timestamp': current_date,
                    'drawdown': drawdown
                })
            
            # Close any remaining open positions at the end of the simulation
            final_date = common_dates[-1]
            final_data = {}
            for coin, data in all_data.items():
                if final_date in data.index:
                    final_data[coin] = data.loc[final_date]
            
            for coin, position in list(open_positions.items()):
                if coin in final_data:
                    self._close_position(
                        coin, position, final_data[coin]['close'],
                        final_date, portfolio, 'End of backtest'
                    )
            
            logger.info("Simulation completed")
        
        except Exception as e:
            logger.error(f"Error running simulation: {str(e)}")
            raise
    
    def _detect_market_condition(self, current_data: Dict) -> str:
        """
        Detect current market condition based on technical indicators.
        
        Args:
            current_data: Dictionary of current data for each coin
            
        Returns:
            Market condition string
        """
        try:
            # This is a simplified implementation
            # In a real system, we would use more sophisticated methods
            
            # Calculate average trend strength across all coins
            trend_strengths = []
            volatilities = []
            volumes = []
            
            for coin, data in current_data.items():
                # Check if required indicators are available
                if 'adx' in data and 'atr' in data and 'volume' in data and 'close' in data:
                    # Use ADX as trend strength indicator
                    trend_strength = data['adx'] / 100.0
                    if 'macd' in data:
                        # Adjust trend strength direction based on MACD
                        if data['macd'] < 0:
                            trend_strength = -trend_strength
                    
                    # Use ATR relative to price as volatility indicator
                    volatility = data['atr'] / data['close']
                    
                    # Use volume relative to average volume as volume indicator
                    if 'volume_sma' in data and data['volume_sma'] > 0:
                        volume = data['volume'] / data['volume_sma']
                    else:
                        volume = 1.0
                    
                    trend_strengths.append(trend_strength)
                    volatilities.append(volatility)
                    volumes.append(volume)
            
            if not trend_strengths:
                return 'UNKNOWN'
            
            # Calculate averages
            avg_trend_strength = sum(trend_strengths) / len(trend_strengths)
            avg_volatility = sum(volatilities) / len(volatilities)
            avg_volume = sum(volumes) / len(volumes)
            
            # Determine market condition based on averages
            if avg_volatility > 0.05:  # High volatility
                if avg_trend_strength > 0.5:
                    return 'TRENDING_UP'
                elif avg_trend_strength < -0.5:
                    return 'TRENDING_DOWN'
                else:
                    return 'VOLATILE'
            else:  # Low volatility
                if abs(avg_trend_strength) < 0.2:
                    return 'RANGING'
                elif avg_volume > 1.5:
                    return 'BREAKOUT'
                elif avg_trend_strength > 0.2:
                    return 'TRENDING_UP'
                elif avg_trend_strength < -0.2:
                    return 'TRENDING_DOWN'
                else:
                    return 'RANGING'
        
        except Exception as e:
            logger.error(f"Error detecting market condition: {str(e)}")
            return 'UNKNOWN'
    
    def _process_open_positions(self, open_positions: Dict, current_data: Dict,
                              current_date: pd.Timestamp, portfolio: Dict,
                              market_condition: str):
        """
        Process open positions for the current date.
        
        Args:
            open_positions: Dictionary of open positions
            current_data: Dictionary of current data for each coin
            current_date: Current date
            portfolio: Portfolio dictionary
            market_condition: Current market condition
        """
        try:
            # Process each open position
            for coin, position in list(open_positions.items()):
                # Skip if no data available for this coin
                if coin not in current_data:
                    continue
                
                # Get current price
                current_price = current_data[coin]['close']
                
                # Check stop loss
                stop_loss_price = position['stop_loss_price']
                if position['side'] == 'BUY' and current_price <= stop_loss_price:
                    self._close_position(
                        coin, position, current_price, current_date, portfolio, 'Stop Loss'
                    )
                    continue
                elif position['side'] == 'SELL' and current_price >= stop_loss_price:
                    self._close_position(
                        coin, position, current_price, current_date, portfolio, 'Stop Loss'
                    )
                    continue
                
                # Check take profit
                take_profit_price = position['take_profit_price']
                if position['side'] == 'BUY' and current_price >= take_profit_price:
                    self._close_position(
                        coin, position, current_price, current_date, portfolio, 'Take Profit'
                    )
                    continue
                elif position['side'] == 'SELL' and current_price <= take_profit_price:
                    self._close_position(
                        coin, position, current_price, current_date, portfolio, 'Take Profit'
                    )
                    continue
                
                # Check trailing stop if enabled
                if self.enable_trailing_stop and position['trailing_stop_price'] > 0:
                    trailing_stop_price = position['trailing_stop_price']
                    if position['side'] == 'BUY' and current_price <= trailing_stop_price:
                        self._close_position(
                            coin, position, current_price, current_date, portfolio, 'Trailing Stop'
                        )
                        continue
                    elif position['side'] == 'SELL' and current_price >= trailing_stop_price:
                        self._close_position(
                            coin, position, current_price, current_date, portfolio, 'Trailing Stop'
                        )
                        continue
                
                # Update trailing stop if enabled
                if self.enable_trailing_stop:
                    self._update_trailing_stop(position, current_price)
                
                # Process partial take profits if enabled
                if self.enable_partial_take_profit:
                    self._process_partial_take_profits(
                        coin, position, current_price, current_date, portfolio
                    )
        
        except Exception as e:
            logger.error(f"Error processing open positions: {str(e)}")
    
    def _update_trailing_stop(self, position: Dict, current_price: float):
        """
        Update trailing stop price based on current price.
        
        Args:
            position: Position dictionary
            current_price: Current price
        """
        try:
            # Skip if trailing stop not enabled for this position
            if not position.get('trailing_stop_enabled', False):
                return
            
            # Get trailing stop parameters
            activation_threshold = position.get('trailing_stop_activation', 0.01)
            trail_distance = position.get('trailing_stop_distance', 0.01)
            
            # Check if trailing stop should be activated
            entry_price = position['entry_price']
            
            if position['side'] == 'BUY':
                # For long positions
                price_change_percentage = (current_price - entry_price) / entry_price
                
                if price_change_percentage >= activation_threshold:
                    # Calculate new trailing stop price
                    new_trailing_stop = current_price * (1 - trail_distance)
                    
                    # Update trailing stop price if higher than current
                    if new_trailing_stop > position.get('trailing_stop_price', 0):
                        position['trailing_stop_price'] = new_trailing_stop
                        position['trailing_stop_activated'] = True
            else:  # SELL
                # For short positions
                price_change_percentage = (entry_price - current_price) / entry_price
                
                if price_change_percentage >= activation_threshold:
                    # Calculate new trailing stop price
                    new_trailing_stop = current_price * (1 + trail_distance)
                    
                    # Update trailing stop price if lower than current
                    if position.get('trailing_stop_price', float('inf')) == 0 or new_trailing_stop < position['trailing_stop_price']:
                        position['trailing_stop_price'] = new_trailing_stop
                        position['trailing_stop_activated'] = True
        
        except Exception as e:
            logger.error(f"Error updating trailing stop: {str(e)}")
    
    def _process_partial_take_profits(self, coin: str, position: Dict, current_price: float,
                                    current_date: pd.Timestamp, portfolio: Dict):
        """
        Process partial take profits for a position.
        
        Args:
            coin: Cryptocurrency symbol
            position: Position dictionary
            current_price: Current price
            current_date: Current date
            portfolio: Portfolio dictionary
        """
        try:
            # Skip if partial take profits not enabled for this position
            if not position.get('partial_take_profits', []):
                return
            
            # Process each partial take profit level
            for level in list(position.get('partial_take_profits', [])):
                # Skip if already executed
                if level.get('executed', False):
                    continue
                
                # Check if level should be executed
                if position['side'] == 'BUY' and current_price >= level['price']:
                    self._execute_partial_take_profit(
                        coin, position, level, current_price, current_date, portfolio
                    )
                elif position['side'] == 'SELL' and current_price <= level['price']:
                    self._execute_partial_take_profit(
                        coin, position, level, current_price, current_date, portfolio
                    )
        
        except Exception as e:
            logger.error(f"Error processing partial take profits: {str(e)}")
    
    def _execute_partial_take_profit(self, coin: str, position: Dict, level: Dict,
                                   current_price: float, current_date: pd.Timestamp,
                                   portfolio: Dict):
        """
        Execute a partial take profit.
        
        Args:
            coin: Cryptocurrency symbol
            position: Position dictionary
            level: Partial take profit level
            current_price: Current price
            current_date: Current date
            portfolio: Portfolio dictionary
        """
        try:
            # Calculate quantity to sell
            quantity_to_sell = position['quantity'] * level['percentage']
            
            # Adjust for rounding
            quantity_to_sell = min(quantity_to_sell, position['quantity'])
            
            # Calculate profit/loss
            if position['side'] == 'BUY':
                profit_loss = (current_price - position['entry_price']) * quantity_to_sell
            else:  # SELL
                profit_loss = (position['entry_price'] - current_price) * quantity_to_sell
            
            # Apply commission
            commission = current_price * quantity_to_sell * self.commission
            profit_loss -= commission
            
            # Update position
            position['quantity'] -= quantity_to_sell
            
            # Update portfolio
            portfolio['cash'] += current_price * quantity_to_sell - commission
            
            # Mark level as executed
            level['executed'] = True
            level['execution_price'] = current_price
            level['execution_date'] = current_date
            level['profit_loss'] = profit_loss
            
            # Log partial take profit
            logger.info(f"Executed partial take profit for {coin} at {current_price}: {profit_loss:.2f}")
            
            # Create trade record
            trade = {
                'trade_id': f"{position['trade_id']}_partial_{level['threshold']}",
                'coin': coin,
                'side': 'SELL' if position['side'] == 'BUY' else 'BUY',  # Opposite of position side
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'quantity': quantity_to_sell,
                'position_size': position['entry_price'] * quantity_to_sell,
                'profit_loss': profit_loss,
                'profit_loss_percentage': profit_loss / (position['entry_price'] * quantity_to_sell) if position['entry_price'] > 0 else 0,
                'entry_time': position['entry_time'],
                'exit_time': current_date,
                'trade_duration': (current_date - position['entry_time']).total_seconds(),
                'exit_reason': 'Partial Take Profit',
                'market_condition': position.get('market_condition', 'UNKNOWN'),
                'signal_strength': position.get('signal_strength', 0),
                'confidence': position.get('confidence', 0)
            }
            
            # Add trade to list
            self.trades.append(trade)
        
        except Exception as e:
            logger.error(f"Error executing partial take profit: {str(e)}")
    
    def _close_position(self, coin: str, position: Dict, current_price: float,
                      current_date: pd.Timestamp, portfolio: Dict, exit_reason: str):
        """
        Close a position.
        
        Args:
            coin: Cryptocurrency symbol
            position: Position dictionary
            current_price: Current price
            current_date: Current date
            portfolio: Portfolio dictionary
            exit_reason: Reason for closing the position
        """
        try:
            # Calculate profit/loss
            if position['side'] == 'BUY':
                profit_loss = (current_price - position['entry_price']) * position['quantity']
            else:  # SELL
                profit_loss = (position['entry_price'] - current_price) * position['quantity']
            
            # Apply commission
            commission = current_price * position['quantity'] * self.commission
            profit_loss -= commission
            
            # Update portfolio
            portfolio['cash'] += current_price * position['quantity'] - commission
            
            # Remove position
            if coin in portfolio['positions']:
                del portfolio['positions'][coin]
            
            # Log position close
            logger.info(f"Closed {position['side']} position for {coin} at {current_price}: {profit_loss:.2f} ({exit_reason})")
            
            # Create trade record
            trade = {
                'trade_id': position['trade_id'],
                'coin': coin,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'quantity': position['quantity'],
                'position_size': position['position_size'],
                'profit_loss': profit_loss,
                'profit_loss_percentage': profit_loss / position['position_size'] if position['position_size'] > 0 else 0,
                'entry_time': position['entry_time'],
                'exit_time': current_date,
                'trade_duration': (current_date - position['entry_time']).total_seconds(),
                'exit_reason': exit_reason,
                'market_condition': position.get('market_condition', 'UNKNOWN'),
                'signal_strength': position.get('signal_strength', 0),
                'confidence': position.get('confidence', 0)
            }
            
            # Add trade to list
            self.trades.append(trade)
        
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
    
    def _generate_trading_signals(self, current_data: Dict, current_date: pd.Timestamp,
                                market_condition: str) -> List[Dict]:
        """
        Generate trading signals for the current date.
        
        Args:
            current_data: Dictionary of current data for each coin
            current_date: Current date
            market_condition: Current market condition
            
        Returns:
            List of trading signals
        """
        try:
            signals = []
            
            # Generate signals for each coin
            for coin, data in current_data.items():
                # Skip if required data is missing
                if not isinstance(data, pd.Series):
                    continue
                
                # Generate signals using strategy manager
                if self.strategy_manager:
                    strategy_signals = self.strategy_manager.generate_signals(
                        coin=coin,
                        data=data,
                        market_condition=market_condition
                    )
                    
                    if strategy_signals:
                        for signal in strategy_signals:
                            signal['coin'] = coin
                            signal['timestamp'] = current_date
                            signal['market_condition'] = market_condition
                            signals.append(signal)
                
                # Generate signals using ensemble manager if enabled
                if self.enable_ensemble_predictions and self.ensemble_manager:
                    # Create a DataFrame with the current data
                    df = pd.DataFrame([data])
                    df.index = [current_date]
                    
                    # Generate predictions
                    predictions = self.ensemble_manager.predict(df)
                    
                    if predictions.get('status') == 'success' and 'trading_signals' in predictions:
                        trading_signal = predictions['trading_signals']
                        
                        # Convert to signal format
                        signal = {
                            'coin': coin,
                            'timestamp': current_date,
                            'market_condition': market_condition,
                            'signal_type': trading_signal.get('signal', 'NEUTRAL'),
                            'signal_strength': trading_signal.get('strength', 0),
                            'confidence': trading_signal.get('confidence', 0),
                            'source': 'ENSEMBLE'
                        }
                        
                        signals.append(signal)
            
            return signals
        
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            return []
    
    def _execute_trading_signals(self, signals: List[Dict], current_data: Dict,
                               current_date: pd.Timestamp, portfolio: Dict,
                               open_positions: Dict, market_condition: str):
        """
        Execute trading signals.
        
        Args:
            signals: List of trading signals
            current_data: Dictionary of current data for each coin
            current_date: Current date
            portfolio: Portfolio dictionary
            open_positions: Dictionary of open positions
            market_condition: Current market condition
        """
        try:
            # Skip if no signals
            if not signals:
                return
            
            # Filter signals
            valid_signals = []
            for signal in signals:
                # Skip if coin not in current data
                if signal['coin'] not in current_data:
                    continue
                
                # Skip if signal type is NEUTRAL
                if signal['signal_type'] == 'NEUTRAL':
                    continue
                
                # Skip if signal strength or confidence is too low
                if abs(signal.get('signal_strength', 0)) < 0.5 or signal.get('confidence', 0) < 0.6:
                    continue
                
                valid_signals.append(signal)
            
            # Sort signals by strength * confidence
            valid_signals.sort(
                key=lambda x: abs(x.get('signal_strength', 0)) * x.get('confidence', 0),
                reverse=True
            )
            
            # Execute signals
            for signal in valid_signals:
                coin = signal['coin']
                signal_type = signal['signal_type']
                
                # Skip if already have position for this coin
                if coin in open_positions:
                    continue
                
                # Skip if reached maximum open trades
                if len(open_positions) >= self.max_open_trades:
                    continue
                
                # Get current price
                current_price = current_data[coin]['close']
                
                # Determine trade side
                side = 'BUY' if signal_type in ['BUY', 'STRONG_BUY'] else 'SELL'
                
                # Calculate position size
                position_size = 0.0
                
                if self.enable_risk_management and self.risk_manager and self.position_sizer:
                    # Use risk manager to calculate position size
                    position_sizing = self.risk_manager.calculate_position_size(
                        coin=coin,
                        signal_type=signal_type,
                        signal_strength=signal.get('signal_strength', 0),
                        confidence=signal.get('confidence', 0),
                        market_condition=market_condition,
                        current_price=current_price
                    )
                    
                    if position_sizing.get('status') == 'ACCEPTED':
                        position_size = position_sizing.get('position_size', 0.0)
                else:
                    # Use simple position sizing (equal allocation)
                    position_size = portfolio['cash'] * 0.1  # 10% of available cash
                
                # Skip if position size is too small
                if position_size < 100:  # Minimum position size of $100
                    continue
                
                # Ensure we don't exceed available cash
                position_size = min(position_size, portfolio['cash'])
                
                # Calculate quantity
                quantity = position_size / current_price
                
                # Apply slippage
                execution_price = current_price * (1 + self.slippage) if side == 'BUY' else current_price * (1 - self.slippage)
                
                # Calculate commission
                commission = execution_price * quantity * self.commission
                
                # Ensure we have enough cash for commission
                if portfolio['cash'] < position_size + commission:
                    position_size = portfolio['cash'] - commission
                    quantity = position_size / execution_price
                
                # Skip if quantity is too small
                if quantity <= 0:
                    continue
                
                # Calculate stop loss and take profit prices
                stop_loss_percentage = 0.02  # 2% stop loss
                take_profit_percentage = 0.04  # 4% take profit
                
                if self.enable_risk_management and self.risk_manager:
                    stop_loss_percentage = self.risk_manager._calculate_stop_loss(
                        coin=coin,
                        signal_type=signal_type,
                        market_condition=market_condition
                    )
                    
                    take_profit_percentage = self.risk_manager._calculate_take_profit(
                        coin=coin,
                        signal_type=signal_type,
                        market_condition=market_condition,
                        stop_loss=stop_loss_percentage
                    )
                
                if side == 'BUY':
                    stop_loss_price = execution_price * (1 - stop_loss_percentage)
                    take_profit_price = execution_price * (1 + take_profit_percentage)
                else:  # SELL
                    stop_loss_price = execution_price * (1 + stop_loss_percentage)
                    take_profit_price = execution_price * (1 - take_profit_percentage)
                
                # Create position
                trade_id = f"{coin}_{side}_{int(current_date.timestamp())}"
                position = {
                    'trade_id': trade_id,
                    'coin': coin,
                    'side': side,
                    'entry_price': execution_price,
                    'quantity': quantity,
                    'position_size': position_size,
                    'stop_loss_price': stop_loss_price,
                    'take_profit_price': take_profit_price,
                    'trailing_stop_price': 0.0,
                    'trailing_stop_enabled': self.enable_trailing_stop,
                    'trailing_stop_activation': 0.01,  # 1% activation threshold
                    'trailing_stop_distance': 0.005,  # 0.5% trailing distance
                    'trailing_stop_activated': False,
                    'entry_time': current_date,
                    'market_condition': market_condition,
                    'signal_strength': signal.get('signal_strength', 0),
                    'confidence': signal.get('confidence', 0)
                }
                
                # Add partial take profits if enabled
                if self.enable_partial_take_profit:
                    position['partial_take_profits'] = []
                    
                    # Add partial take profit levels
                    if side == 'BUY':
                        position['partial_take_profits'].append({
                            'threshold': 0.01,  # 1% profit
                            'percentage': 0.3,  # Sell 30% of position
                            'price': execution_price * 1.01,
                            'executed': False
                        })
                        position['partial_take_profits'].append({
                            'threshold': 0.02,  # 2% profit
                            'percentage': 0.3,  # Sell 30% of position
                            'price': execution_price * 1.02,
                            'executed': False
                        })
                        position['partial_take_profits'].append({
                            'threshold': 0.03,  # 3% profit
                            'percentage': 0.2,  # Sell 20% of position
                            'price': execution_price * 1.03,
                            'executed': False
                        })
                    else:  # SELL
                        position['partial_take_profits'].append({
                            'threshold': 0.01,  # 1% profit
                            'percentage': 0.3,  # Buy back 30% of position
                            'price': execution_price * 0.99,
                            'executed': False
                        })
                        position['partial_take_profits'].append({
                            'threshold': 0.02,  # 2% profit
                            'percentage': 0.3,  # Buy back 30% of position
                            'price': execution_price * 0.98,
                            'executed': False
                        })
                        position['partial_take_profits'].append({
                            'threshold': 0.03,  # 3% profit
                            'percentage': 0.2,  # Buy back 20% of position
                            'price': execution_price * 0.97,
                            'executed': False
                        })
                
                # Update portfolio
                portfolio['cash'] -= position_size + commission
                portfolio['positions'][coin] = position
                
                # Add to open positions
                open_positions[coin] = position
                
                # Log position open
                logger.info(f"Opened {side} position for {coin} at {execution_price}: {quantity} units, ${position_size:.2f}")
        
        except Exception as e:
            logger.error(f"Error executing trading signals: {str(e)}")
    
    def _calculate_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics for the backtest.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            logger.info("Calculating performance metrics")
            
            # Skip if no equity curve
            if not self.equity_curve:
                return {}
            
            # Create DataFrame from equity curve
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            # Create DataFrame from trades
            trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
            
            # Calculate metrics
            metrics = {}
            
            # Total return
            initial_equity = self.initial_capital
            final_equity = equity_df['equity'].iloc[-1]
            total_return = (final_equity - initial_equity) / initial_equity
            metrics['total_return'] = total_return
            
            # Annualized return
            days = (equity_df.index[-1] - equity_df.index[0]).days
            if days > 0:
                annualized_return = (1 + total_return) ** (365 / days) - 1
                metrics['annualized_return'] = annualized_return
            else:
                metrics['annualized_return'] = 0.0
            
            # Maximum drawdown
            if self.drawdowns:
                max_drawdown = max(entry['drawdown'] for entry in self.drawdowns)
                metrics['max_drawdown'] = max_drawdown
            else:
                metrics['max_drawdown'] = 0.0
            
            # Sharpe ratio (simplified)
            if len(equity_df) > 1:
                # Calculate daily returns
                equity_df['daily_return'] = equity_df['equity'].pct_change()
                
                # Calculate Sharpe ratio
                avg_return = equity_df['daily_return'].mean()
                std_dev = equity_df['daily_return'].std()
                
                if std_dev > 0:
                    sharpe_ratio = avg_return / std_dev * (252 ** 0.5)  # Annualized
                    metrics['sharpe_ratio'] = sharpe_ratio
                else:
                    metrics['sharpe_ratio'] = 0.0
            else:
                metrics['sharpe_ratio'] = 0.0
            
            # Trade metrics
            if not trades_df.empty:
                # Win rate
                winning_trades = trades_df[trades_df['profit_loss'] > 0]
                win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0.0
                metrics['win_rate'] = win_rate
                
                # Profit factor
                gross_profit = winning_trades['profit_loss'].sum() if not winning_trades.empty else 0.0
                losing_trades = trades_df[trades_df['profit_loss'] < 0]
                gross_loss = abs(losing_trades['profit_loss'].sum()) if not losing_trades.empty else 0.0
                
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                metrics['profit_factor'] = profit_factor
                
                # Average win and loss
                avg_win = winning_trades['profit_loss'].mean() if not winning_trades.empty else 0.0
                avg_loss = abs(losing_trades['profit_loss'].mean()) if not losing_trades.empty else 0.0
                
                metrics['average_win'] = avg_win
                metrics['average_loss'] = avg_loss
                
                # Average win/loss ratio
                avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
                metrics['average_win_loss_ratio'] = avg_win_loss_ratio
                
                # Expectancy
                expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
                metrics['expectancy'] = expectancy
                
                # Average holding period
                avg_holding_period = trades_df['trade_duration'].mean() / 3600  # Convert to hours
                metrics['average_holding_period'] = avg_holding_period
            else:
                metrics['win_rate'] = 0.0
                metrics['profit_factor'] = 0.0
                metrics['average_win'] = 0.0
                metrics['average_loss'] = 0.0
                metrics['average_win_loss_ratio'] = 0.0
                metrics['expectancy'] = 0.0
                metrics['average_holding_period'] = 0.0
            
            logger.info(f"Calculated {len(metrics)} performance metrics")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def _save_backtest_results(self, backtest_id: str, parameters: Dict = None):
        """
        Save backtest results to database.
        
        Args:
            backtest_id: Unique identifier for this backtest
            parameters: Dictionary of parameters used for this backtest
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Save backtest results
            cursor.execute(f'''
            INSERT INTO {DATABASE['backtest_results_table']}
            (backtest_id, timestamp, start_date, end_date, coins, timeframe,
             initial_capital, final_capital, total_return, annualized_return,
             max_drawdown, sharpe_ratio, win_rate, profit_factor, parameters, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                backtest_id,
                int(time.time() * 1000),
                self.start_date,
                self.end_date,
                json.dumps(self.coins),
                self.timeframe,
                self.initial_capital,
                self.equity_curve[-1]['equity'] if self.equity_curve else self.initial_capital,
                self.metrics.get('total_return', 0.0),
                self.metrics.get('annualized_return', 0.0),
                self.metrics.get('max_drawdown', 0.0),
                self.metrics.get('sharpe_ratio', 0.0),
                self.metrics.get('win_rate', 0.0),
                self.metrics.get('profit_factor', 0.0),
                json.dumps(parameters or {}),
                int(time.time() * 1000)
            ))
            
            # Save trades
            for trade in self.trades:
                cursor.execute(f'''
                INSERT INTO {DATABASE['backtest_trades_table']}
                (backtest_id, trade_id, timestamp, coin, side, entry_price, exit_price,
                 quantity, position_size, stop_loss, take_profit, profit_loss,
                 profit_loss_percentage, trade_duration, market_condition,
                 signal_strength, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    backtest_id,
                    trade['trade_id'],
                    int(trade['exit_time'].timestamp() * 1000),
                    trade['coin'],
                    trade['side'],
                    trade['entry_price'],
                    trade['exit_price'],
                    trade['quantity'],
                    trade['position_size'],
                    0.0,  # Stop loss percentage not stored in trade
                    0.0,  # Take profit percentage not stored in trade
                    trade['profit_loss'],
                    trade['profit_loss_percentage'],
                    trade['trade_duration'],
                    trade.get('market_condition', 'UNKNOWN'),
                    trade.get('signal_strength', 0.0),
                    trade.get('confidence', 0.0),
                    int(time.time() * 1000)
                ))
            
            # Save metrics
            for metric_name, metric_value in self.metrics.items():
                cursor.execute(f'''
                INSERT INTO {DATABASE['backtest_metrics_table']}
                (backtest_id, timestamp, metric_name, metric_value, created_at)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    backtest_id,
                    int(time.time() * 1000),
                    metric_name,
                    metric_value,
                    int(time.time() * 1000)
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved backtest results for {backtest_id}")
        
        except Exception as e:
            logger.error(f"Error saving backtest results: {str(e)}")
    
    def _generate_html_report(self, backtest_id: str):
        """
        Generate HTML report for the backtest.
        
        Args:
            backtest_id: Unique identifier for this backtest
        """
        try:
            logger.info(f"Generating HTML report for {backtest_id}")
            
            # Create report directory
            report_dir = os.path.join(REPORTING['REPORT_DIRECTORY'], backtest_id)
            os.makedirs(report_dir, exist_ok=True)
            
            # Generate equity curve chart if enabled
            if REPORTING['GENERATE_EQUITY_CURVE_CHART']:
                self._generate_equity_curve_chart(backtest_id, report_dir)
            
            # Generate drawdown chart if enabled
            if REPORTING['GENERATE_DRAWDOWN_CHART']:
                self._generate_drawdown_chart(backtest_id, report_dir)
            
            # Generate trade distribution chart if enabled
            if REPORTING['GENERATE_TRADE_DISTRIBUTION_CHART']:
                self._generate_trade_distribution_chart(backtest_id, report_dir)
            
            # Generate HTML report
            html_content = self._generate_html_content(backtest_id)
            
            # Save HTML report
            html_path = os.path.join(report_dir, f"{backtest_id}_report.html")
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Generated HTML report: {html_path}")
        
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
    
    def _generate_equity_curve_chart(self, backtest_id: str, report_dir: str):
        """
        Generate equity curve chart.
        
        Args:
            backtest_id: Unique identifier for this backtest
            report_dir: Directory to save the chart
        """
        try:
            # Skip if no equity curve
            if not self.equity_curve:
                return
            
            # Create DataFrame from equity curve
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            # Create figure
            plt.figure(figsize=(12, 6))
            plt.plot(equity_df.index, equity_df['equity'])
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity')
            plt.grid(True)
            
            # Save figure
            chart_path = os.path.join(report_dir, f"{backtest_id}_equity_curve.png")
            plt.savefig(chart_path)
            plt.close()
            
            logger.info(f"Generated equity curve chart: {chart_path}")
        
        except Exception as e:
            logger.error(f"Error generating equity curve chart: {str(e)}")
    
    def _generate_drawdown_chart(self, backtest_id: str, report_dir: str):
        """
        Generate drawdown chart.
        
        Args:
            backtest_id: Unique identifier for this backtest
            report_dir: Directory to save the chart
        """
        try:
            # Skip if no drawdowns
            if not self.drawdowns:
                return
            
            # Create DataFrame from drawdowns
            drawdown_df = pd.DataFrame(self.drawdowns)
            drawdown_df.set_index('timestamp', inplace=True)
            
            # Create figure
            plt.figure(figsize=(12, 6))
            plt.plot(drawdown_df.index, drawdown_df['drawdown'] * 100)  # Convert to percentage
            plt.title('Drawdown')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            
            # Save figure
            chart_path = os.path.join(report_dir, f"{backtest_id}_drawdown.png")
            plt.savefig(chart_path)
            plt.close()
            
            logger.info(f"Generated drawdown chart: {chart_path}")
        
        except Exception as e:
            logger.error(f"Error generating drawdown chart: {str(e)}")
    
    def _generate_trade_distribution_chart(self, backtest_id: str, report_dir: str):
        """
        Generate trade distribution chart.
        
        Args:
            backtest_id: Unique identifier for this backtest
            report_dir: Directory to save the chart
        """
        try:
            # Skip if no trades
            if not self.trades:
                return
            
            # Create DataFrame from trades
            trades_df = pd.DataFrame(self.trades)
            
            # Create figure
            plt.figure(figsize=(12, 6))
            plt.hist(trades_df['profit_loss_percentage'] * 100, bins=20)  # Convert to percentage
            plt.title('Trade Profit/Loss Distribution')
            plt.xlabel('Profit/Loss (%)')
            plt.ylabel('Number of Trades')
            plt.grid(True)
            
            # Save figure
            chart_path = os.path.join(report_dir, f"{backtest_id}_trade_distribution.png")
            plt.savefig(chart_path)
            plt.close()
            
            logger.info(f"Generated trade distribution chart: {chart_path}")
        
        except Exception as e:
            logger.error(f"Error generating trade distribution chart: {str(e)}")
    
    def _generate_html_content(self, backtest_id: str) -> str:
        """
        Generate HTML content for the report.
        
        Args:
            backtest_id: Unique identifier for this backtest
            
        Returns:
            HTML content as string
        """
        try:
            # Basic HTML template
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Backtest Report: {backtest_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .chart {{ margin: 20px 0; max-width: 100%; }}
                    .metrics {{ display: flex; flex-wrap: wrap; }}
                    .metric-card {{ background-color: #f2f2f2; border-radius: 5px; padding: 15px; margin: 10px; flex: 1; min-width: 200px; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>Backtest Report: {backtest_id}</h1>
                
                <h2>Summary</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div>Total Return</div>
                        <div class="metric-value {self._get_color_class(self.metrics.get('total_return', 0))}">
                            {self.metrics.get('total_return', 0):.2%}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div>Annualized Return</div>
                        <div class="metric-value {self._get_color_class(self.metrics.get('annualized_return', 0))}">
                            {self.metrics.get('annualized_return', 0):.2%}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div>Max Drawdown</div>
                        <div class="metric-value negative">
                            {self.metrics.get('max_drawdown', 0):.2%}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div>Sharpe Ratio</div>
                        <div class="metric-value {self._get_color_class(self.metrics.get('sharpe_ratio', 0))}">
                            {self.metrics.get('sharpe_ratio', 0):.2f}
                        </div>
                    </div>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div>Win Rate</div>
                        <div class="metric-value {self._get_color_class(self.metrics.get('win_rate', 0) - 0.5)}">
                            {self.metrics.get('win_rate', 0):.2%}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div>Profit Factor</div>
                        <div class="metric-value {self._get_color_class(self.metrics.get('profit_factor', 0) - 1.0)}">
                            {self.metrics.get('profit_factor', 0):.2f}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div>Expectancy</div>
                        <div class="metric-value {self._get_color_class(self.metrics.get('expectancy', 0))}">
                            ${self.metrics.get('expectancy', 0):.2f}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div>Avg Holding Period</div>
                        <div class="metric-value">
                            {self.metrics.get('average_holding_period', 0):.2f} hours
                        </div>
                    </div>
                </div>
                
                <h2>Charts</h2>
                <div class="chart">
                    <h3>Equity Curve</h3>
                    <img src="{backtest_id}_equity_curve.png" alt="Equity Curve" style="max-width: 100%;">
                </div>
                
                <div class="chart">
                    <h3>Drawdown</h3>
                    <img src="{backtest_id}_drawdown.png" alt="Drawdown" style="max-width: 100%;">
                </div>
                
                <div class="chart">
                    <h3>Trade Distribution</h3>
                    <img src="{backtest_id}_trade_distribution.png" alt="Trade Distribution" style="max-width: 100%;">
                </div>
                
                <h2>Parameters</h2>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Initial Capital</td>
                        <td>${self.initial_capital:.2f}</td>
                    </tr>
                    <tr>
                        <td>Timeframe</td>
                        <td>{self.timeframe}</td>
                    </tr>
                    <tr>
                        <td>Start Date</td>
                        <td>{self.start_date}</td>
                    </tr>
                    <tr>
                        <td>End Date</td>
                        <td>{self.end_date}</td>
                    </tr>
                    <tr>
                        <td>Coins</td>
                        <td>{', '.join(self.coins)}</td>
                    </tr>
                    <tr>
                        <td>Commission</td>
                        <td>{self.commission:.2%}</td>
                    </tr>
                    <tr>
                        <td>Slippage</td>
                        <td>{self.slippage:.2%}</td>
                    </tr>
                </table>
                
                <h2>Trades</h2>
                <table>
                    <tr>
                        <th>Coin</th>
                        <th>Side</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>Quantity</th>
                        <th>Profit/Loss</th>
                        <th>P/L %</th>
                        <th>Duration</th>
                        <th>Exit Reason</th>
                    </tr>
            """
            
            # Add trades
            max_trades = min(REPORTING['MAX_TRADES_IN_REPORT'], len(self.trades))
            for i in range(max_trades):
                trade = self.trades[i]
                duration_hours = trade['trade_duration'] / 3600  # Convert to hours
                
                html += f"""
                    <tr>
                        <td>{trade['coin']}</td>
                        <td>{trade['side']}</td>
                        <td>${trade['entry_price']:.2f}</td>
                        <td>${trade['exit_price']:.2f}</td>
                        <td>{trade['quantity']:.4f}</td>
                        <td class="{self._get_color_class(trade['profit_loss'])}">
                            ${trade['profit_loss']:.2f}
                        </td>
                        <td class="{self._get_color_class(trade['profit_loss_percentage'])}">
                            {trade['profit_loss_percentage']:.2%}
                        </td>
                        <td>{duration_hours:.2f} hours</td>
                        <td>{trade.get('exit_reason', 'Unknown')}</td>
                    </tr>
                """
            
            # Close HTML
            html += """
                </table>
            </body>
            </html>
            """
            
            return html
        
        except Exception as e:
            logger.error(f"Error generating HTML content: {str(e)}")
            return f"<html><body><h1>Error generating report: {str(e)}</h1></body></html>"
    
    def _get_color_class(self, value: float) -> str:
        """
        Get CSS color class based on value.
        
        Args:
            value: Numeric value
            
        Returns:
            CSS class name
        """
        if value > 0:
            return "positive"
        elif value < 0:
            return "negative"
        else:
            return ""
    
    def _generate_csv_results(self, backtest_id: str):
        """
        Generate CSV results files.
        
        Args:
            backtest_id: Unique identifier for this backtest
        """
        try:
            # Create report directory
            report_dir = os.path.join(REPORTING['REPORT_DIRECTORY'], backtest_id)
            os.makedirs(report_dir, exist_ok=True)
            
            # Save equity curve
            if self.equity_curve:
                equity_df = pd.DataFrame(self.equity_curve)
                equity_path = os.path.join(report_dir, f"{backtest_id}_equity.csv")
                equity_df.to_csv(equity_path, index=False)
            
            # Save trades
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                trades_path = os.path.join(report_dir, f"{backtest_id}_trades.csv")
                trades_df.to_csv(trades_path, index=False)
            
            # Save metrics
            if self.metrics:
                metrics_df = pd.DataFrame([self.metrics])
                metrics_path = os.path.join(report_dir, f"{backtest_id}_metrics.csv")
                metrics_df.to_csv(metrics_path, index=False)
            
            logger.info(f"Generated CSV results for {backtest_id}")
        
        except Exception as e:
            logger.error(f"Error generating CSV results: {str(e)}")
    
    def _generate_pdf_report(self, backtest_id: str):
        """
        Generate PDF report for the backtest.
        
        Args:
            backtest_id: Unique identifier for this backtest
        """
        try:
            logger.info(f"PDF report generation not implemented yet for {backtest_id}")
            # This would require additional libraries like reportlab or wkhtmltopdf
        
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
    
    def _run_monte_carlo_simulation(self, backtest_id: str):
        """
        Run Monte Carlo simulation for the backtest.
        
        Args:
            backtest_id: Unique identifier for this backtest
        """
        try:
            logger.info(f"Running Monte Carlo simulation for {backtest_id}")
            
            # Skip if no trades
            if not self.trades:
                logger.warning("No trades available for Monte Carlo simulation")
                return
            
            # Create DataFrame from trades
            trades_df = pd.DataFrame(self.trades)
            
            # Get simulation parameters
            num_simulations = MONTE_CARLO_SIMULATION['NUM_SIMULATIONS']
            confidence_level = MONTE_CARLO_SIMULATION['CONFIDENCE_LEVEL']
            
            # Run simulations
            simulation_results = []
            
            for i in range(num_simulations):
                # Randomly sample trades with replacement
                sampled_trades = trades_df.sample(n=len(trades_df), replace=True)
                
                # Calculate cumulative returns
                cumulative_return = (1 + sampled_trades['profit_loss_percentage']).prod() - 1
                
                # Calculate max drawdown
                cumulative_returns = (1 + sampled_trades['profit_loss_percentage']).cumprod()
                peak = cumulative_returns.expanding(min_periods=1).max()
                drawdown = (cumulative_returns / peak - 1)
                max_drawdown = drawdown.min()
                
                # Calculate Sharpe ratio (simplified)
                returns = sampled_trades['profit_loss_percentage']
                sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0.0
                
                # Calculate win rate
                win_rate = (sampled_trades['profit_loss'] > 0).mean()
                
                # Store results
                simulation_results.append({
                    'simulation_id': f"{backtest_id}_MC_{i}",
                    'final_equity': self.initial_capital * (1 + cumulative_return),
                    'total_return': cumulative_return,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'win_rate': win_rate
                })
            
            # Convert to DataFrame
            results_df = pd.DataFrame(simulation_results)
            
            # Calculate statistics
            stats = {
                'final_equity': {
                    'mean': results_df['final_equity'].mean(),
                    'median': results_df['final_equity'].median(),
                    'std': results_df['final_equity'].std(),
                    'min': results_df['final_equity'].min(),
                    'max': results_df['final_equity'].max(),
                    f'conf_{confidence_level:.0%}': results_df['final_equity'].quantile(1 - confidence_level)
                },
                'total_return': {
                    'mean': results_df['total_return'].mean(),
                    'median': results_df['total_return'].median(),
                    'std': results_df['total_return'].std(),
                    'min': results_df['total_return'].min(),
                    'max': results_df['total_return'].max(),
                    f'conf_{confidence_level:.0%}': results_df['total_return'].quantile(1 - confidence_level)
                },
                'max_drawdown': {
                    'mean': results_df['max_drawdown'].mean(),
                    'median': results_df['max_drawdown'].median(),
                    'std': results_df['max_drawdown'].std(),
                    'min': results_df['max_drawdown'].min(),
                    'max': results_df['max_drawdown'].max(),
                    f'conf_{confidence_level:.0%}': results_df['max_drawdown'].quantile(confidence_level)
                },
                'sharpe_ratio': {
                    'mean': results_df['sharpe_ratio'].mean(),
                    'median': results_df['sharpe_ratio'].median(),
                    'std': results_df['sharpe_ratio'].std(),
                    'min': results_df['sharpe_ratio'].min(),
                    'max': results_df['sharpe_ratio'].max(),
                    f'conf_{confidence_level:.0%}': results_df['sharpe_ratio'].quantile(1 - confidence_level)
                },
                'win_rate': {
                    'mean': results_df['win_rate'].mean(),
                    'median': results_df['win_rate'].median(),
                    'std': results_df['win_rate'].std(),
                    'min': results_df['win_rate'].min(),
                    'max': results_df['win_rate'].max(),
                    f'conf_{confidence_level:.0%}': results_df['win_rate'].quantile(1 - confidence_level)
                }
            }
            
            # Store Monte Carlo results
            self.monte_carlo_results = simulation_results
            
            # Save results to database
            self._save_monte_carlo_results(backtest_id, simulation_results)
            
            # Generate Monte Carlo chart if enabled
            if REPORTING['GENERATE_MONTE_CARLO_CHART']:
                self._generate_monte_carlo_chart(backtest_id, results_df)
            
            logger.info(f"Completed Monte Carlo simulation for {backtest_id}")
        
        except Exception as e:
            logger.error(f"Error running Monte Carlo simulation: {str(e)}")
    
    def _save_monte_carlo_results(self, backtest_id: str, simulation_results: List[Dict]):
        """
        Save Monte Carlo simulation results to database.
        
        Args:
            backtest_id: Unique identifier for this backtest
            simulation_results: List of simulation results
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for result in simulation_results:
                cursor.execute(f'''
                INSERT INTO {DATABASE['monte_carlo_results_table']}
                (simulation_id, timestamp, simulation_number, final_equity,
                 max_drawdown, sharpe_ratio, win_rate, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result['simulation_id'],
                    int(time.time() * 1000),
                    int(result['simulation_id'].split('_')[-1]),
                    result['final_equity'],
                    result['max_drawdown'],
                    result['sharpe_ratio'],
                    result['win_rate'],
                    int(time.time() * 1000)
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved Monte Carlo results for {backtest_id}")
        
        except Exception as e:
            logger.error(f"Error saving Monte Carlo results: {str(e)}")
    
    def _generate_monte_carlo_chart(self, backtest_id: str, results_df: pd.DataFrame):
        """
        Generate Monte Carlo simulation chart.
        
        Args:
            backtest_id: Unique identifier for this backtest
            results_df: DataFrame with simulation results
        """
        try:
            # Create report directory
            report_dir = os.path.join(REPORTING['REPORT_DIRECTORY'], backtest_id)
            os.makedirs(report_dir, exist_ok=True)
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot histogram of final equity
            plt.hist(results_df['final_equity'], bins=30, alpha=0.7)
            
            # Add vertical line for actual final equity
            actual_final_equity = self.equity_curve[-1]['equity'] if self.equity_curve else self.initial_capital
            plt.axvline(x=actual_final_equity, color='r', linestyle='--', label='Actual Final Equity')
            
            # Add confidence interval
            confidence_level = MONTE_CARLO_SIMULATION['CONFIDENCE_LEVEL']
            conf_level_equity = results_df['final_equity'].quantile(1 - confidence_level)
            plt.axvline(x=conf_level_equity, color='g', linestyle='--', 
                       label=f'{confidence_level:.0%} Confidence Level: ${conf_level_equity:.2f}')
            
            plt.title('Monte Carlo Simulation: Final Equity Distribution')
            plt.xlabel('Final Equity ($)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            
            # Save figure
            chart_path = os.path.join(report_dir, f"{backtest_id}_monte_carlo.png")
            plt.savefig(chart_path)
            plt.close()
            
            logger.info(f"Generated Monte Carlo chart: {chart_path}")
        
        except Exception as e:
            logger.error(f"Error generating Monte Carlo chart: {str(e)}")
    
    def run_optimization(self, parameter_grid: Dict = None) -> Dict:
        """
        Run parameter optimization.
        
        Args:
            parameter_grid: Dictionary of parameters to optimize
            
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info("Starting parameter optimization")
            
            # Use default parameter grid if not provided
            if parameter_grid is None:
                parameter_grid = BACKTESTING['OPTIMIZATION_RANGES']
            
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(parameter_grid)
            
            logger.info(f"Generated {len(param_combinations)} parameter combinations")
            
            # Run backtest for each parameter combination
            optimization_results = []
            
            for params in tqdm(param_combinations, desc="Optimizing parameters"):
                # Run backtest with these parameters
                backtest_result = self.run_backtest(params)
                
                if backtest_result.get('status') == 'success':
                    # Store result
                    optimization_results.append({
                        'parameters': params,
                        'backtest_id': backtest_result.get('backtest_id'),
                        'metrics': backtest_result.get('metrics', {})
                    })
            
            # Find best parameters based on optimization metric
            best_result = self._find_best_parameters(optimization_results)
            
            # Save optimization results
            self._save_optimization_results(optimization_results)
            
            logger.info(f"Completed parameter optimization with {len(optimization_results)} results")
            
            return {
                'status': 'success',
                'best_parameters': best_result.get('parameters', {}),
                'best_metrics': best_result.get('metrics', {}),
                'best_backtest_id': best_result.get('backtest_id'),
                'num_combinations': len(param_combinations),
                'num_results': len(optimization_results)
            }
        
        except Exception as e:
            logger.error(f"Error running optimization: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error running optimization: {str(e)}"
            }
    
    def _generate_parameter_combinations(self, parameter_grid: Dict) -> List[Dict]:
        """
        Generate all combinations of parameters.
        
        Args:
            parameter_grid: Dictionary of parameters to optimize
            
        Returns:
            List of parameter dictionaries
        """
        try:
            # Get all parameter names and values
            param_names = list(parameter_grid.keys())
            param_values = list(parameter_grid.values())
            
            # Generate all combinations
            combinations = []
            
            # Helper function for recursive combination generation
            def generate_combinations(current_params, param_index):
                if param_index >= len(param_names):
                    combinations.append(current_params.copy())
                    return
                
                param_name = param_names[param_index]
                for value in param_values[param_index]:
                    current_params[param_name] = value
                    generate_combinations(current_params, param_index + 1)
            
            # Generate combinations
            generate_combinations({}, 0)
            
            return combinations
        
        except Exception as e:
            logger.error(f"Error generating parameter combinations: {str(e)}")
            return []
    
    def _find_best_parameters(self, optimization_results: List[Dict]) -> Dict:
        """
        Find best parameters based on optimization metric.
        
        Args:
            optimization_results: List of optimization results
            
        Returns:
            Dictionary with best parameters and metrics
        """
        try:
            if not optimization_results:
                return {}
            
            # Get optimization metric
            metric = self.optimization_metric
            
            # Find best result
            best_result = None
            best_value = float('-inf')
            
            for result in optimization_results:
                metrics = result.get('metrics', {})
                
                if metric in metrics:
                    value = metrics[metric]
                    
                    # Handle metrics where lower is better
                    if metric in ['max_drawdown', 'average_loss']:
                        value = -value
                    
                    if best_result is None or value > best_value:
                        best_result = result
                        best_value = value
            
            return best_result or {}
        
        except Exception as e:
            logger.error(f"Error finding best parameters: {str(e)}")
            return {}
    
    def _save_optimization_results(self, optimization_results: List[Dict]):
        """
        Save optimization results to database.
        
        Args:
            optimization_results: List of optimization results
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Generate optimization ID
            optimization_id = f"OPT_{int(time.time())}"
            
            for result in optimization_results:
                parameters = result.get('parameters', {})
                metrics = result.get('metrics', {})
                
                for metric_name, metric_value in metrics.items():
                    cursor.execute(f'''
                    INSERT INTO {DATABASE['optimization_results_table']}
                    (optimization_id, timestamp, parameters, metric_name, metric_value, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        optimization_id,
                        int(time.time() * 1000),
                        json.dumps(parameters),
                        metric_name,
                        metric_value,
                        int(time.time() * 1000)
                    ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved optimization results for {optimization_id}")
        
        except Exception as e:
            logger.error(f"Error saving optimization results: {str(e)}")
    
    def run_walk_forward_analysis(self) -> Dict:
        """
        Run walk-forward analysis.
        
        Returns:
            Dictionary with walk-forward analysis results
        """
        try:
            logger.info("Starting walk-forward analysis")
            
            # Skip if not enabled
            if not WALK_FORWARD_ANALYSIS['ENABLED']:
                logger.warning("Walk-forward analysis is disabled")
                return {
                    'status': 'error',
                    'message': 'Walk-forward analysis is disabled'
                }
            
            # Get walk-forward parameters
            training_window = WALK_FORWARD_ANALYSIS['TRAINING_WINDOW']
            testing_window = WALK_FORWARD_ANALYSIS['TESTING_WINDOW']
            step_size = WALK_FORWARD_ANALYSIS['STEP_SIZE']
            anchored = WALK_FORWARD_ANALYSIS['ANCHORED']
            
            # Parse start and end dates
            start_date = pd.Timestamp(self.start_date)
            end_date = pd.Timestamp(self.end_date)
            
            # Calculate total days
            total_days = (end_date - start_date).days
            
            # Check if we have enough data
            min_days = training_window + testing_window
            if total_days < min_days:
                logger.warning(f"Not enough data for walk-forward analysis: {total_days} days < {min_days} days")
                return {
                    'status': 'error',
                    'message': f"Not enough data for walk-forward analysis: {total_days} days < {min_days} days"
                }
            
            # Generate windows
            windows = []
            
            current_start = start_date
            while current_start + pd.Timedelta(days=min_days) <= end_date:
                if anchored:
                    # Anchored walk-forward analysis (fixed start date)
                    training_start = start_date
                else:
                    # Rolling walk-forward analysis
                    training_start = current_start
                
                training_end = training_start + pd.Timedelta(days=training_window)
                testing_start = training_end
                testing_end = testing_start + pd.Timedelta(days=testing_window)
                
                # Ensure testing end date doesn't exceed overall end date
                testing_end = min(testing_end, end_date)
                
                windows.append({
                    'training_start': training_start.strftime('%Y-%m-%d'),
                    'training_end': training_end.strftime('%Y-%m-%d'),
                    'testing_start': testing_start.strftime('%Y-%m-%d'),
                    'testing_end': testing_end.strftime('%Y-%m-%d')
                })
                
                # Move to next window
                current_start += pd.Timedelta(days=step_size)
            
            logger.info(f"Generated {len(windows)} windows for walk-forward analysis")
            
            # Run analysis for each window
            analysis_results = []
            
            # Generate analysis ID
            analysis_id = f"WFA_{int(time.time())}"
            
            for i, window in enumerate(windows):
                logger.info(f"Processing window {i+1}/{len(windows)}")
                
                # Run optimization on training period
                self.start_date = window['training_start']
                self.end_date = window['training_end']
                
                optimization_result = self.run_optimization()
                
                if optimization_result.get('status') != 'success':
                    logger.warning(f"Optimization failed for window {i+1}")
                    continue
                
                # Get best parameters
                best_parameters = optimization_result.get('best_parameters', {})
                
                # Run backtest on testing period with best parameters
                self.start_date = window['testing_start']
                self.end_date = window['testing_end']
                
                backtest_result = self.run_backtest(best_parameters)
                
                if backtest_result.get('status') != 'success':
                    logger.warning(f"Backtest failed for window {i+1}")
                    continue
                
                # Store results
                window_result = {
                    'window_number': i + 1,
                    'training_start': window['training_start'],
                    'training_end': window['training_end'],
                    'testing_start': window['testing_start'],
                    'testing_end': window['testing_end'],
                    'parameters': best_parameters,
                    'training_metrics': optimization_result.get('best_metrics', {}),
                    'testing_metrics': backtest_result.get('metrics', {}),
                    'training_backtest_id': optimization_result.get('best_backtest_id'),
                    'testing_backtest_id': backtest_result.get('backtest_id')
                }
                
                analysis_results.append(window_result)
                
                # Save to database
                self._save_walk_forward_result(analysis_id, window_result)
            
            # Reset to original date range
            self.start_date = start_date.strftime('%Y-%m-%d')
            self.end_date = end_date.strftime('%Y-%m-%d')
            
            # Store walk-forward results
            self.walk_forward_results = analysis_results
            
            logger.info(f"Completed walk-forward analysis with {len(analysis_results)} results")
            
            return {
                'status': 'success',
                'analysis_id': analysis_id,
                'windows': len(windows),
                'results': len(analysis_results)
            }
        
        except Exception as e:
            logger.error(f"Error running walk-forward analysis: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error running walk-forward analysis: {str(e)}"
            }
    
    def _save_walk_forward_result(self, analysis_id: str, result: Dict):
        """
        Save walk-forward analysis result to database.
        
        Args:
            analysis_id: Unique identifier for this analysis
            result: Walk-forward analysis result
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get optimization metric
            metric = WALK_FORWARD_ANALYSIS['OPTIMIZATION_METRIC']
            
            cursor.execute(f'''
            INSERT INTO {DATABASE['walk_forward_results_table']}
            (analysis_id, timestamp, window_number, training_start_date,
             training_end_date, testing_start_date, testing_end_date,
             parameters, training_metric_value, testing_metric_value, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_id,
                int(time.time() * 1000),
                result['window_number'],
                result['training_start'],
                result['training_end'],
                result['testing_start'],
                result['testing_end'],
                json.dumps(result['parameters']),
                result['training_metrics'].get(metric, 0.0),
                result['testing_metrics'].get(metric, 0.0),
                int(time.time() * 1000)
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved walk-forward result for window {result['window_number']}")
        
        except Exception as e:
            logger.error(f"Error saving walk-forward result: {str(e)}")
    
    def run_stress_test(self, scenario: str = None) -> Dict:
        """
        Run stress test with specified scenario.
        
        Args:
            scenario: Stress test scenario name
            
        Returns:
            Dictionary with stress test results
        """
        try:
            logger.info(f"Starting stress test: {scenario}")
            
            # Skip if not enabled
            if not self.enable_stress_testing:
                logger.warning("Stress testing is disabled")
                return {
                    'status': 'error',
                    'message': 'Stress testing is disabled'
                }
            
            # Use all scenarios if none specified
            scenarios = [scenario] if scenario else list(STRESS_TEST_SCENARIOS.keys())
            
            # Run stress test for each scenario
            stress_test_results = {}
            
            for scenario_name in scenarios:
                if scenario_name not in STRESS_TEST_SCENARIOS:
                    logger.warning(f"Unknown scenario: {scenario_name}")
                    continue
                
                # Get scenario parameters
                scenario_config = STRESS_TEST_SCENARIOS[scenario_name]
                
                # Create modified parameters for this scenario
                scenario_params = {
                    'scenario': scenario_name,
                    'description': scenario_config['description']
                }
                
                # Modify parameters based on scenario
                if scenario_name == 'EXTREME_VOLATILITY':
                    scenario_params['slippage'] = self.slippage * scenario_config['volatility_multiplier']
                elif scenario_name == 'FLASH_CRASH':
                    # Flash crash is handled during simulation
                    pass
                elif scenario_name == 'LIQUIDITY_CRISIS':
                    scenario_params['slippage'] = self.slippage * scenario_config['slippage_multiplier']
                elif scenario_name == 'CORRELATION_BREAKDOWN':
                    # Correlation breakdown is handled during simulation
                    pass
                elif scenario_name == 'EXCHANGE_OUTAGE':
                    # Exchange outage is handled during simulation
                    pass
                
                # Run backtest with scenario parameters
                backtest_result = self.run_backtest(scenario_params)
                
                if backtest_result.get('status') == 'success':
                    # Store result
                    stress_test_results[scenario_name] = {
                        'backtest_id': backtest_result.get('backtest_id'),
                        'metrics': backtest_result.get('metrics', {}),
                        'scenario': scenario_name,
                        'description': scenario_config['description']
                    }
            
            logger.info(f"Completed stress tests for {len(stress_test_results)} scenarios")
            
            return {
                'status': 'success',
                'results': stress_test_results
            }
        
        except Exception as e:
            logger.error(f"Error running stress test: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error running stress test: {str(e)}"
            }
    
    def __str__(self) -> str:
        """String representation of the backtester."""
        return f"Backtester(timeframe={self.timeframe}, coins={self.coins}, capital=${self.initial_capital:.2f})"
