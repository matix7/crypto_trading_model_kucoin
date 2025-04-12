"""
Paper Trading Engine - Main class for paper trading execution.
"""

import os
import time
import json
import logging
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import threading
import queue
import uuid
import traceback

from .config import (
    PAPER_TRADING, API_CONFIG, DATABASE, LOGGING, PERFORMANCE_METRICS
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
os.makedirs(os.path.dirname(LOGGING['LOG_FILE']), exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOGGING['LOG_LEVEL'].upper()),
    format=LOGGING['LOG_FORMAT'],
    handlers=[
        logging.StreamHandler() if LOGGING['CONSOLE_LOG'] else logging.NullHandler(),
        logging.FileHandler(LOGGING['LOG_FILE']) if LOGGING['FILE_LOG'] else logging.NullHandler()
    ]
)
logger = logging.getLogger('paper_trading')

class PaperTradingEngine:
    """
    Paper Trading Engine - Main class for paper trading execution.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the Paper Trading Engine.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path or DATABASE['DB_PATH']
        
        # Trading parameters
        self.initial_capital = PAPER_TRADING['INITIAL_CAPITAL']
        self.trading_pairs = PAPER_TRADING['TRADING_PAIRS']
        self.timeframes = PAPER_TRADING['TIMEFRAMES']
        self.max_open_trades = PAPER_TRADING['MAX_OPEN_TRADES']
        
        # Feature flags
        self.enable_position_sizing = PAPER_TRADING['ENABLE_POSITION_SIZING']
        self.enable_risk_management = PAPER_TRADING['ENABLE_RISK_MANAGEMENT']
        self.enable_trailing_stop = PAPER_TRADING['ENABLE_TRAILING_STOP']
        self.enable_partial_take_profit = PAPER_TRADING['ENABLE_PARTIAL_TAKE_PROFIT']
        self.enable_compounding = PAPER_TRADING['ENABLE_COMPOUNDING']
        self.enable_portfolio_rebalancing = PAPER_TRADING['ENABLE_PORTFOLIO_REBALANCING']
        self.enable_market_condition_detection = PAPER_TRADING['ENABLE_MARKET_CONDITION_DETECTION']
        self.enable_sentiment_analysis = PAPER_TRADING['ENABLE_SENTIMENT_ANALYSIS']
        self.enable_ensemble_predictions = PAPER_TRADING['ENABLE_ENSEMBLE_PREDICTIONS']
        self.enable_dynamic_optimization = PAPER_TRADING['ENABLE_DYNAMIC_OPTIMIZATION']
        self.enable_self_learning = PAPER_TRADING['ENABLE_SELF_LEARNING']
        
        # Trading settings
        self.learning_rate = PAPER_TRADING['LEARNING_RATE']
        self.daily_profit_target = PAPER_TRADING['DAILY_PROFIT_TARGET']
        self.success_rate_target = PAPER_TRADING['SUCCESS_RATE_TARGET']
        self.risk_per_trade = PAPER_TRADING['RISK_PER_TRADE']
        self.stop_loss_percentage = PAPER_TRADING['STOP_LOSS_PERCENTAGE']
        self.take_profit_percentage = PAPER_TRADING['TAKE_PROFIT_PERCENTAGE']
        self.trailing_stop_activation = PAPER_TRADING['TRAILING_STOP_ACTIVATION']
        self.trailing_stop_distance = PAPER_TRADING['TRAILING_STOP_DISTANCE']
        self.update_interval = PAPER_TRADING['UPDATE_INTERVAL']
        
        # Trading hours
        self.trading_hours_enabled = PAPER_TRADING['TRADING_HOURS']['ENABLED']
        self.trading_start_hour = PAPER_TRADING['TRADING_HOURS']['START_HOUR']
        self.trading_end_hour = PAPER_TRADING['TRADING_HOURS']['END_HOUR']
        self.trading_days = PAPER_TRADING['TRADING_HOURS']['DAYS']
        
        # Notifications
        self.notifications = PAPER_TRADING['NOTIFICATIONS']
        
        # State variables
        self.is_running = False
        self.account_balance = self.initial_capital
        self.open_positions = {}
        self.trade_history = []
        self.performance_metrics = {}
        self.last_update_time = None
        self.last_optimization_time = None
        self.last_learning_time = None
        self.last_daily_summary_time = None
        
        # Thread and queue for async processing
        self.trading_thread = None
        self.signal_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Component instances
        self.market_data_collector = None
        self.technical_indicator_calculator = None
        self.feature_engineer = None
        self.strategy_manager = None
        self.sentiment_manager = None
        self.ensemble_manager = None
        self.risk_manager = None
        self.position_sizer = None
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info("Paper Trading Engine initialized")
    
    def _init_database(self):
        """Initialize the SQLite database with required tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create paper trades table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['TABLES']['paper_trades']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                trading_pair TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                position_size REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                profit_loss REAL,
                profit_loss_percentage REAL,
                status TEXT NOT NULL,
                entry_time INTEGER NOT NULL,
                exit_time INTEGER,
                trade_duration INTEGER,
                exit_reason TEXT,
                market_condition TEXT,
                signal_strength REAL,
                confidence REAL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            ''')
            
            # Create paper positions table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['TABLES']['paper_positions']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id TEXT NOT NULL,
                trade_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                trading_pair TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                quantity REAL NOT NULL,
                position_size REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                trailing_stop REAL,
                unrealized_profit_loss REAL NOT NULL,
                unrealized_profit_loss_percentage REAL NOT NULL,
                status TEXT NOT NULL,
                entry_time INTEGER NOT NULL,
                market_condition TEXT,
                signal_strength REAL,
                confidence REAL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            ''')
            
            # Create paper balance table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['TABLES']['paper_balance']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                balance REAL NOT NULL,
                equity REAL NOT NULL,
                available REAL NOT NULL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Create paper performance table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['TABLES']['paper_performance']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Create trading signals table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['TABLES']['trading_signals']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                trading_pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                signal_strength REAL NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                market_condition TEXT,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Create system logs table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['TABLES']['system_logs']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Create optimization history table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['TABLES']['optimization_history']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                parameter_name TEXT NOT NULL,
                old_value REAL NOT NULL,
                new_value REAL NOT NULL,
                reason TEXT NOT NULL,
                performance_impact REAL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Create learning history table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['TABLES']['learning_history']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                parameter_name TEXT NOT NULL,
                old_value REAL NOT NULL,
                new_value REAL NOT NULL,
                learning_rate REAL NOT NULL,
                performance_impact REAL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Insert initial balance record
            cursor.execute(f'''
            INSERT INTO {DATABASE['TABLES']['paper_balance']}
            (timestamp, balance, equity, available, created_at)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                int(time.time() * 1000),
                self.initial_capital,
                self.initial_capital,
                self.initial_capital,
                int(time.time() * 1000)
            ))
            
            conn.commit()
            conn.close()
            logger.info("Database tables initialized")
        
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            logger.error(traceback.format_exc())
    
    def initialize_components(self):
        """Initialize all required components for paper trading."""
        try:
            logger.info("Initializing components for paper trading")
            
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
                # Initialize for each trading pair and timeframe
                self.ensemble_manager = {}
                for pair in self.trading_pairs:
                    self.ensemble_manager[pair] = {}
                    for timeframe in self.timeframes:
                        self.ensemble_manager[pair][timeframe] = EnsembleManager(
                            coin=pair.replace('USDT', ''),
                            timeframe=timeframe,
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
            
            logger.info("Components initialized for paper trading")
        
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def start(self):
        """Start the paper trading engine."""
        try:
            if self.is_running:
                logger.warning("Paper trading engine is already running")
                return False
            
            logger.info("Starting paper trading engine")
            
            # Initialize components
            self.initialize_components()
            
            # Set running flag
            self.is_running = True
            self.stop_event.clear()
            
            # Load account state from database
            self._load_account_state()
            
            # Start trading thread
            self.trading_thread = threading.Thread(target=self._trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            # Log system start
            self._log_system_event("System started")
            
            # Send notification if enabled
            if self.notifications['ENABLE_WEBHOOK'] and self.notifications['NOTIFY_ON_SYSTEM_START']:
                self._send_notification("System started", "Paper trading engine has been started")
            
            logger.info("Paper trading engine started")
            return True
        
        except Exception as e:
            logger.error(f"Error starting paper trading engine: {str(e)}")
            logger.error(traceback.format_exc())
            self.is_running = False
            return False
    
    def stop(self):
        """Stop the paper trading engine."""
        try:
            if not self.is_running:
                logger.warning("Paper trading engine is not running")
                return False
            
            logger.info("Stopping paper trading engine")
            
            # Set stop event
            self.stop_event.set()
            
            # Wait for trading thread to finish
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
            
            # Set running flag
            self.is_running = False
            
            # Save account state to database
            self._save_account_state()
            
            # Calculate and save performance metrics
            self._calculate_performance_metrics()
            
            # Log system stop
            self._log_system_event("System stopped")
            
            # Send notification if enabled
            if self.notifications['ENABLE_WEBHOOK'] and self.notifications['NOTIFY_ON_SYSTEM_STOP']:
                self._send_notification("System stopped", "Paper trading engine has been stopped")
            
            logger.info("Paper trading engine stopped")
            return True
        
        except Exception as e:
            logger.error(f"Error stopping paper trading engine: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _trading_loop(self):
        """Main trading loop."""
        try:
            logger.info("Trading loop started")
            
            while not self.stop_event.is_set():
                try:
                    # Check if trading is allowed based on trading hours
                    if not self._is_trading_allowed():
                        # Sleep for a minute before checking again
                        time.sleep(60)
                        continue
                    
                    # Update market data
                    self._update_market_data()
                    
                    # Process open positions
                    self._process_open_positions()
                    
                    # Generate trading signals
                    self._generate_trading_signals()
                    
                    # Process trading signals from queue
                    self._process_trading_signals()
                    
                    # Update account state
                    self._update_account_state()
                    
                    # Run optimization if enabled
                    if self.enable_dynamic_optimization:
                        self._run_optimization()
                    
                    # Run self-learning if enabled
                    if self.enable_self_learning:
                        self._run_self_learning()
                    
                    # Generate daily summary if needed
                    self._generate_daily_summary()
                    
                    # Update last update time
                    self.last_update_time = time.time()
                    
                    # Sleep for update interval
                    time.sleep(self.update_interval)
                
                except Exception as e:
                    logger.error(f"Error in trading loop: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    # Send notification if enabled
                    if self.notifications['ENABLE_WEBHOOK'] and self.notifications['NOTIFY_ON_ERROR']:
                        self._send_notification("Trading Error", f"Error in trading loop: {str(e)}")
                    
                    # Sleep for a minute before retrying
                    time.sleep(60)
            
            logger.info("Trading loop stopped")
        
        except Exception as e:
            logger.error(f"Fatal error in trading loop: {str(e)}")
            logger.error(traceback.format_exc())
            self.is_running = False
    
    def _is_trading_allowed(self) -> bool:
        """
        Check if trading is allowed based on trading hours.
        
        Returns:
            True if trading is allowed, False otherwise
        """
        # Skip check if trading hours are not enabled
        if not self.trading_hours_enabled:
            return True
        
        # Get current time
        now = datetime.utcnow()
        
        # Check day of week (0=Monday, 6=Sunday)
        if now.weekday() not in self.trading_days:
            return False
        
        # Check hour
        if not (self.trading_start_hour <= now.hour < self.trading_end_hour):
            return False
        
        return True
    
    def _update_market_data(self):
        """Update market data for all trading pairs and timeframes."""
        try:
            logger.debug("Updating market data")
            
            for pair in self.trading_pairs:
                for timeframe in self.timeframes:
                    # Fetch latest market data
                    self.market_data_collector.fetch_latest_data(
                        coin=pair.replace('USDT', ''),
                        timeframe=timeframe
                    )
            
            logger.debug("Market data updated")
        
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _process_open_positions(self):
        """Process open positions."""
        try:
            logger.debug("Processing open positions")
            
            # Skip if no open positions
            if not self.open_positions:
                return
            
            # Process each open position
            for position_id, position in list(self.open_positions.items()):
                # Get trading pair and current price
                pair = position['trading_pair']
                current_price = self._get_current_price(pair)
                
                # Skip if price is not available
                if current_price is None:
                    continue
                
                # Update position with current price
                position['current_price'] = current_price
                
                # Calculate unrealized profit/loss
                if position['side'] == 'BUY':
                    unrealized_pl = (current_price - position['entry_price']) * position['quantity']
                else:  # SELL
                    unrealized_pl = (position['entry_price'] - current_price) * position['quantity']
                
                position['unrealized_profit_loss'] = unrealized_pl
                position['unrealized_profit_loss_percentage'] = unrealized_pl / position['position_size']
                
                # Check stop loss
                if position['side'] == 'BUY' and current_price <= position['stop_loss']:
                    self._close_position(position_id, current_price, 'Stop Loss')
                    continue
                elif position['side'] == 'SELL' and current_price >= position['stop_loss']:
                    self._close_position(position_id, current_price, 'Stop Loss')
                    continue
                
                # Check take profit
                if position['side'] == 'BUY' and current_price >= position['take_profit']:
                    self._close_position(position_id, current_price, 'Take Profit')
                    continue
                elif position['side'] == 'SELL' and current_price <= position['take_profit']:
                    self._close_position(position_id, current_price, 'Take Profit')
                    continue
                
                # Check trailing stop if enabled
                if self.enable_trailing_stop and position.get('trailing_stop') is not None:
                    if position['side'] == 'BUY' and current_price <= position['trailing_stop']:
                        self._close_position(position_id, current_price, 'Trailing Stop')
                        continue
                    elif position['side'] == 'SELL' and current_price >= position['trailing_stop']:
                        self._close_position(position_id, current_price, 'Trailing Stop')
                        continue
                
                # Update trailing stop if enabled
                if self.enable_trailing_stop:
                    self._update_trailing_stop(position, current_price)
                
                # Update position in database
                self._update_position_in_db(position)
            
            logger.debug(f"Processed {len(self.open_positions)} open positions")
        
        except Exception as e:
            logger.error(f"Error processing open positions: {str(e)}")
            logger.error(traceback.format_exc())
    
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
            activation_threshold = position.get('trailing_stop_activation', self.trailing_stop_activation)
            trail_distance = position.get('trailing_stop_distance', self.trailing_stop_distance)
            
            # Check if trailing stop should be activated
            entry_price = position['entry_price']
            
            if position['side'] == 'BUY':
                # For long positions
                price_change_percentage = (current_price - entry_price) / entry_price
                
                if price_change_percentage >= activation_threshold:
                    # Calculate new trailing stop price
                    new_trailing_stop = current_price * (1 - trail_distance)
                    
                    # Update trailing stop price if higher than current
                    if position.get('trailing_stop') is None or new_trailing_stop > position['trailing_stop']:
                        position['trailing_stop'] = new_trailing_stop
                        position['trailing_stop_activated'] = True
                        logger.debug(f"Updated trailing stop for {position['trading_pair']} to {new_trailing_stop}")
            else:  # SELL
                # For short positions
                price_change_percentage = (entry_price - current_price) / entry_price
                
                if price_change_percentage >= activation_threshold:
                    # Calculate new trailing stop price
                    new_trailing_stop = current_price * (1 + trail_distance)
                    
                    # Update trailing stop price if lower than current
                    if position.get('trailing_stop') is None or new_trailing_stop < position['trailing_stop']:
                        position['trailing_stop'] = new_trailing_stop
                        position['trailing_stop_activated'] = True
                        logger.debug(f"Updated trailing stop for {position['trading_pair']} to {new_trailing_stop}")
        
        except Exception as e:
            logger.error(f"Error updating trailing stop: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _generate_trading_signals(self):
        """Generate trading signals for all trading pairs and timeframes."""
        try:
            logger.debug("Generating trading signals")
            
            for pair in self.trading_pairs:
                for timeframe in self.timeframes:
                    # Skip if already have position for this pair
                    if self._has_position_for_pair(pair):
                        continue
                    
                    # Get latest data
                    data = self._get_latest_data(pair, timeframe)
                    
                    # Skip if data is not available
                    if data is None or data.empty:
                        continue
                    
                    # Calculate technical indicators
                    data = self.technical_indicator_calculator.calculate_indicators(data)
                    
                    # Engineer features
                    data = self.feature_engineer.engineer_features(data)
                    
                    # Add sentiment data if enabled
                    if self.enable_sentiment_analysis and self.sentiment_manager:
                        sentiment_data = self.sentiment_manager.get_latest_sentiment(
                            coin=pair.replace('USDT', '')
                        )
                        
                        if sentiment_data is not None:
                            # Add sentiment data to the last row
                            for key, value in sentiment_data.items():
                                data.loc[data.index[-1], key] = value
                    
                    # Detect market condition if enabled
                    market_condition = 'UNKNOWN'
                    if self.enable_market_condition_detection:
                        market_condition = self._detect_market_condition(data)
                    
                    # Generate signals using strategy manager
                    if self.strategy_manager:
                        strategy_signals = self.strategy_manager.generate_signals(
                            coin=pair.replace('USDT', ''),
                            data=data.iloc[-1],
                            market_condition=market_condition
                        )
                        
                        if strategy_signals:
                            for signal in strategy_signals:
                                signal['trading_pair'] = pair
                                signal['timeframe'] = timeframe
                                signal['timestamp'] = int(time.time() * 1000)
                                signal['market_condition'] = market_condition
                                
                                # Add to signal queue
                                self.signal_queue.put(signal)
                                
                                # Save signal to database
                                self._save_signal_to_db(signal)
                    
                    # Generate signals using ensemble manager if enabled
                    if self.enable_ensemble_predictions and pair in self.ensemble_manager and timeframe in self.ensemble_manager[pair]:
                        # Generate predictions
                        predictions = self.ensemble_manager[pair][timeframe].predict(data.tail(1))
                        
                        if predictions.get('status') == 'success' and 'trading_signals' in predictions:
                            trading_signal = predictions['trading_signals']
                            
                            # Convert to signal format
                            signal = {
                                'trading_pair': pair,
                                'timeframe': timeframe,
                                'timestamp': int(time.time() * 1000),
                                'market_condition': market_condition,
                                'signal_type': trading_signal.get('signal', 'NEUTRAL'),
                                'signal_strength': trading_signal.get('strength', 0),
                                'confidence': trading_signal.get('confidence', 0),
                                'source': 'ENSEMBLE'
                            }
                            
                            # Add to signal queue
                            self.signal_queue.put(signal)
                            
                            # Save signal to database
                            self._save_signal_to_db(signal)
            
            logger.debug("Trading signals generated")
        
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _process_trading_signals(self):
        """Process trading signals from the queue."""
        try:
            logger.debug("Processing trading signals")
            
            # Process signals in queue
            signals_processed = 0
            while not self.signal_queue.empty() and signals_processed < 10:  # Limit to 10 signals per cycle
                try:
                    # Get signal from queue
                    signal = self.signal_queue.get(block=False)
                    signals_processed += 1
                    
                    # Skip if signal type is NEUTRAL
                    if signal['signal_type'] == 'NEUTRAL':
                        continue
                    
                    # Skip if already have position for this pair
                    if self._has_position_for_pair(signal['trading_pair']):
                        continue
                    
                    # Skip if reached maximum open trades
                    if len(self.open_positions) >= self.max_open_trades:
                        continue
                    
                    # Skip if signal strength or confidence is too low
                    if abs(signal.get('signal_strength', 0)) < 0.5 or signal.get('confidence', 0) < 0.6:
                        continue
                    
                    # Execute signal
                    self._execute_trading_signal(signal)
                
                except queue.Empty:
                    break
            
            logger.debug(f"Processed {signals_processed} trading signals")
        
        except Exception as e:
            logger.error(f"Error processing trading signals: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _execute_trading_signal(self, signal: Dict):
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal dictionary
        """
        try:
            # Get trading pair and current price
            pair = signal['trading_pair']
            current_price = self._get_current_price(pair)
            
            # Skip if price is not available
            if current_price is None:
                return
            
            # Determine trade side
            side = 'BUY' if signal['signal_type'] in ['BUY', 'STRONG_BUY'] else 'SELL'
            
            # Calculate position size
            position_size = 0.0
            
            if self.enable_risk_management and self.risk_manager and self.position_sizer:
                # Use risk manager to calculate position size
                position_sizing = self.risk_manager.calculate_position_size(
                    coin=pair.replace('USDT', ''),
                    signal_type=signal['signal_type'],
                    signal_strength=signal.get('signal_strength', 0),
                    confidence=signal.get('confidence', 0),
                    market_condition=signal.get('market_condition', 'UNKNOWN'),
                    current_price=current_price
                )
                
                if position_sizing.get('status') == 'ACCEPTED':
                    position_size = position_sizing.get('position_size', 0.0)
            else:
                # Use simple position sizing (fixed risk)
                position_size = self.account_balance * self.risk_per_trade
            
            # Skip if position size is too small
            if position_size < 10:  # Minimum position size of $10
                return
            
            # Ensure we don't exceed available balance
            available_balance = self._get_available_balance()
            position_size = min(position_size, available_balance)
            
            # Calculate quantity
            quantity = position_size / current_price
            
            # Calculate stop loss and take profit prices
            stop_loss_percentage = self.stop_loss_percentage
            take_profit_percentage = self.take_profit_percentage
            
            if self.enable_risk_management and self.risk_manager:
                stop_loss_percentage = self.risk_manager._calculate_stop_loss(
                    coin=pair.replace('USDT', ''),
                    signal_type=signal['signal_type'],
                    market_condition=signal.get('market_condition', 'UNKNOWN')
                )
                
                take_profit_percentage = self.risk_manager._calculate_take_profit(
                    coin=pair.replace('USDT', ''),
                    signal_type=signal['signal_type'],
                    market_condition=signal.get('market_condition', 'UNKNOWN'),
                    stop_loss=stop_loss_percentage
                )
            
            if side == 'BUY':
                stop_loss_price = current_price * (1 - stop_loss_percentage)
                take_profit_price = current_price * (1 + take_profit_percentage)
            else:  # SELL
                stop_loss_price = current_price * (1 + stop_loss_percentage)
                take_profit_price = current_price * (1 - take_profit_percentage)
            
            # Create trade and position
            trade_id = f"{pair}_{side}_{int(time.time())}"
            position_id = str(uuid.uuid4())
            
            # Create trade record
            trade = {
                'trade_id': trade_id,
                'timestamp': int(time.time() * 1000),
                'trading_pair': pair,
                'side': side,
                'entry_price': current_price,
                'exit_price': None,
                'quantity': quantity,
                'position_size': position_size,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'profit_loss': None,
                'profit_loss_percentage': None,
                'status': 'OPEN',
                'entry_time': int(time.time() * 1000),
                'exit_time': None,
                'trade_duration': None,
                'exit_reason': None,
                'market_condition': signal.get('market_condition', 'UNKNOWN'),
                'signal_strength': signal.get('signal_strength', 0),
                'confidence': signal.get('confidence', 0),
                'created_at': int(time.time() * 1000),
                'updated_at': int(time.time() * 1000)
            }
            
            # Create position record
            position = {
                'position_id': position_id,
                'trade_id': trade_id,
                'timestamp': int(time.time() * 1000),
                'trading_pair': pair,
                'side': side,
                'entry_price': current_price,
                'current_price': current_price,
                'quantity': quantity,
                'position_size': position_size,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'trailing_stop': None,
                'trailing_stop_enabled': self.enable_trailing_stop,
                'trailing_stop_activation': self.trailing_stop_activation,
                'trailing_stop_distance': self.trailing_stop_distance,
                'trailing_stop_activated': False,
                'unrealized_profit_loss': 0.0,
                'unrealized_profit_loss_percentage': 0.0,
                'status': 'OPEN',
                'entry_time': int(time.time() * 1000),
                'market_condition': signal.get('market_condition', 'UNKNOWN'),
                'signal_strength': signal.get('signal_strength', 0),
                'confidence': signal.get('confidence', 0),
                'created_at': int(time.time() * 1000),
                'updated_at': int(time.time() * 1000)
            }
            
            # Update account balance
            self.account_balance -= position_size
            
            # Add to open positions
            self.open_positions[position_id] = position
            
            # Add to trade history
            self.trade_history.append(trade)
            
            # Save trade and position to database
            self._save_trade_to_db(trade)
            self._save_position_to_db(position)
            
            # Update account state in database
            self._update_account_state()
            
            # Log trade
            logger.info(f"Opened {side} position for {pair} at {current_price}: {quantity} units, ${position_size:.2f}")
            
            # Send notification if enabled
            if self.notifications['ENABLE_WEBHOOK'] and self.notifications['NOTIFY_ON_TRADE']:
                self._send_notification(
                    f"New Trade: {side} {pair}",
                    f"Opened {side} position for {pair} at {current_price}: {quantity} units, ${position_size:.2f}"
                )
        
        except Exception as e:
            logger.error(f"Error executing trading signal: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _close_position(self, position_id: str, current_price: float, exit_reason: str):
        """
        Close a position.
        
        Args:
            position_id: Position ID
            current_price: Current price
            exit_reason: Reason for closing the position
        """
        try:
            # Get position
            position = self.open_positions.get(position_id)
            
            if not position:
                logger.warning(f"Position {position_id} not found")
                return
            
            # Calculate profit/loss
            if position['side'] == 'BUY':
                profit_loss = (current_price - position['entry_price']) * position['quantity']
            else:  # SELL
                profit_loss = (position['entry_price'] - current_price) * position['quantity']
            
            profit_loss_percentage = profit_loss / position['position_size']
            
            # Update account balance
            self.account_balance += position['position_size'] + profit_loss
            
            # Update position
            position['current_price'] = current_price
            position['unrealized_profit_loss'] = profit_loss
            position['unrealized_profit_loss_percentage'] = profit_loss_percentage
            position['status'] = 'CLOSED'
            position['updated_at'] = int(time.time() * 1000)
            
            # Update trade
            trade_id = position['trade_id']
            for trade in self.trade_history:
                if trade['trade_id'] == trade_id:
                    trade['exit_price'] = current_price
                    trade['profit_loss'] = profit_loss
                    trade['profit_loss_percentage'] = profit_loss_percentage
                    trade['status'] = 'CLOSED'
                    trade['exit_time'] = int(time.time() * 1000)
                    trade['trade_duration'] = trade['exit_time'] - trade['entry_time']
                    trade['exit_reason'] = exit_reason
                    trade['updated_at'] = int(time.time() * 1000)
                    
                    # Save trade to database
                    self._update_trade_in_db(trade)
                    break
            
            # Remove from open positions
            del self.open_positions[position_id]
            
            # Update position in database
            self._update_position_in_db(position)
            
            # Update account state in database
            self._update_account_state()
            
            # Log trade
            logger.info(f"Closed {position['side']} position for {position['trading_pair']} at {current_price}: {profit_loss:.2f} ({exit_reason})")
            
            # Send notification if enabled
            if self.notifications['ENABLE_WEBHOOK']:
                if profit_loss > 0 and self.notifications['NOTIFY_ON_PROFIT']:
                    self._send_notification(
                        f"Profit: {position['side']} {position['trading_pair']}",
                        f"Closed {position['side']} position for {position['trading_pair']} at {current_price}: ${profit_loss:.2f} ({profit_loss_percentage:.2%}) - {exit_reason}"
                    )
                elif profit_loss < 0 and self.notifications['NOTIFY_ON_LOSS']:
                    self._send_notification(
                        f"Loss: {position['side']} {position['trading_pair']}",
                        f"Closed {position['side']} position for {position['trading_pair']} at {current_price}: ${profit_loss:.2f} ({profit_loss_percentage:.2%}) - {exit_reason}"
                    )
        
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _update_account_state(self):
        """Update account state in database."""
        try:
            # Calculate equity
            equity = self.account_balance
            for position in self.open_positions.values():
                equity += position['unrealized_profit_loss']
            
            # Calculate available balance
            available = self.account_balance
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
            INSERT INTO {DATABASE['TABLES']['paper_balance']}
            (timestamp, balance, equity, available, created_at)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                int(time.time() * 1000),
                self.account_balance,
                equity,
                available,
                int(time.time() * 1000)
            ))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logger.error(f"Error updating account state: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _load_account_state(self):
        """Load account state from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load account balance
            cursor.execute(f'''
            SELECT balance FROM {DATABASE['TABLES']['paper_balance']}
            ORDER BY timestamp DESC LIMIT 1
            ''')
            
            result = cursor.fetchone()
            if result:
                self.account_balance = result[0]
            
            # Load open positions
            cursor.execute(f'''
            SELECT * FROM {DATABASE['TABLES']['paper_positions']}
            WHERE status = 'OPEN'
            ''')
            
            columns = [description[0] for description in cursor.description]
            positions = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            for position in positions:
                self.open_positions[position['position_id']] = position
            
            # Load trade history
            cursor.execute(f'''
            SELECT * FROM {DATABASE['TABLES']['paper_trades']}
            ORDER BY entry_time DESC LIMIT 100
            ''')
            
            columns = [description[0] for description in cursor.description]
            trades = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            self.trade_history = trades
            
            conn.close()
            
            logger.info(f"Loaded account state: balance=${self.account_balance:.2f}, {len(self.open_positions)} open positions")
        
        except Exception as e:
            logger.error(f"Error loading account state: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _save_account_state(self):
        """Save account state to database."""
        try:
            # Update account state
            self._update_account_state()
            
            # Update all open positions
            for position in self.open_positions.values():
                self._update_position_in_db(position)
            
            logger.info(f"Saved account state: balance=${self.account_balance:.2f}, {len(self.open_positions)} open positions")
        
        except Exception as e:
            logger.error(f"Error saving account state: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _save_trade_to_db(self, trade: Dict):
        """
        Save trade to database.
        
        Args:
            trade: Trade dictionary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
            INSERT INTO {DATABASE['TABLES']['paper_trades']}
            (trade_id, timestamp, trading_pair, side, entry_price, exit_price,
             quantity, position_size, stop_loss, take_profit, profit_loss,
             profit_loss_percentage, status, entry_time, exit_time, trade_duration,
             exit_reason, market_condition, signal_strength, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade['trade_id'],
                trade['timestamp'],
                trade['trading_pair'],
                trade['side'],
                trade['entry_price'],
                trade['exit_price'],
                trade['quantity'],
                trade['position_size'],
                trade['stop_loss'],
                trade['take_profit'],
                trade['profit_loss'],
                trade['profit_loss_percentage'],
                trade['status'],
                trade['entry_time'],
                trade['exit_time'],
                trade['trade_duration'],
                trade['exit_reason'],
                trade['market_condition'],
                trade['signal_strength'],
                trade['confidence'],
                trade['created_at'],
                trade['updated_at']
            ))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logger.error(f"Error saving trade to database: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _update_trade_in_db(self, trade: Dict):
        """
        Update trade in database.
        
        Args:
            trade: Trade dictionary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
            UPDATE {DATABASE['TABLES']['paper_trades']}
            SET exit_price = ?, profit_loss = ?, profit_loss_percentage = ?,
                status = ?, exit_time = ?, trade_duration = ?, exit_reason = ?,
                updated_at = ?
            WHERE trade_id = ?
            ''', (
                trade['exit_price'],
                trade['profit_loss'],
                trade['profit_loss_percentage'],
                trade['status'],
                trade['exit_time'],
                trade['trade_duration'],
                trade['exit_reason'],
                trade['updated_at'],
                trade['trade_id']
            ))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logger.error(f"Error updating trade in database: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _save_position_to_db(self, position: Dict):
        """
        Save position to database.
        
        Args:
            position: Position dictionary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
            INSERT INTO {DATABASE['TABLES']['paper_positions']}
            (position_id, trade_id, timestamp, trading_pair, side, entry_price,
             current_price, quantity, position_size, stop_loss, take_profit,
             trailing_stop, unrealized_profit_loss, unrealized_profit_loss_percentage,
             status, entry_time, market_condition, signal_strength, confidence,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position['position_id'],
                position['trade_id'],
                position['timestamp'],
                position['trading_pair'],
                position['side'],
                position['entry_price'],
                position['current_price'],
                position['quantity'],
                position['position_size'],
                position['stop_loss'],
                position['take_profit'],
                position['trailing_stop'],
                position['unrealized_profit_loss'],
                position['unrealized_profit_loss_percentage'],
                position['status'],
                position['entry_time'],
                position['market_condition'],
                position['signal_strength'],
                position['confidence'],
                position['created_at'],
                position['updated_at']
            ))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logger.error(f"Error saving position to database: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _update_position_in_db(self, position: Dict):
        """
        Update position in database.
        
        Args:
            position: Position dictionary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
            UPDATE {DATABASE['TABLES']['paper_positions']}
            SET current_price = ?, stop_loss = ?, take_profit = ?,
                trailing_stop = ?, unrealized_profit_loss = ?,
                unrealized_profit_loss_percentage = ?, status = ?, updated_at = ?
            WHERE position_id = ?
            ''', (
                position['current_price'],
                position['stop_loss'],
                position['take_profit'],
                position['trailing_stop'],
                position['unrealized_profit_loss'],
                position['unrealized_profit_loss_percentage'],
                position['status'],
                int(time.time() * 1000),
                position['position_id']
            ))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logger.error(f"Error updating position in database: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _save_signal_to_db(self, signal: Dict):
        """
        Save trading signal to database.
        
        Args:
            signal: Trading signal dictionary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
            INSERT INTO {DATABASE['TABLES']['trading_signals']}
            (timestamp, trading_pair, timeframe, signal_type, signal_strength,
             confidence, source, market_condition, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['timestamp'],
                signal['trading_pair'],
                signal['timeframe'],
                signal['signal_type'],
                signal['signal_strength'],
                signal['confidence'],
                signal.get('source', 'STRATEGY'),
                signal.get('market_condition', 'UNKNOWN'),
                int(time.time() * 1000)
            ))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logger.error(f"Error saving signal to database: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _log_system_event(self, message: str, level: str = 'INFO'):
        """
        Log system event to database.
        
        Args:
            message: Log message
            level: Log level
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
            INSERT INTO {DATABASE['TABLES']['system_logs']}
            (timestamp, level, message, created_at)
            VALUES (?, ?, ?, ?)
            ''', (
                int(time.time() * 1000),
                level,
                message,
                int(time.time() * 1000)
            ))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logger.error(f"Error logging system event: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _send_notification(self, title: str, message: str):
        """
        Send notification.
        
        Args:
            title: Notification title
            message: Notification message
        """
        try:
            # Skip if webhook is not enabled or URL is not set
            if not self.notifications['ENABLE_WEBHOOK'] or not self.notifications['WEBHOOK_URL']:
                return
            
            # TODO: Implement webhook notification
            logger.info(f"Notification: {title} - {message}")
        
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _run_optimization(self):
        """Run dynamic optimization."""
        try:
            # Skip if not enabled
            if not self.enable_dynamic_optimization:
                return
            
            # Skip if last optimization was too recent (once per day)
            if self.last_optimization_time and time.time() - self.last_optimization_time < 86400:
                return
            
            logger.info("Running dynamic optimization")
            
            # TODO: Implement dynamic optimization
            
            # Update last optimization time
            self.last_optimization_time = time.time()
            
            logger.info("Dynamic optimization completed")
        
        except Exception as e:
            logger.error(f"Error running optimization: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _run_self_learning(self):
        """Run self-learning."""
        try:
            # Skip if not enabled
            if not self.enable_self_learning:
                return
            
            # Skip if last learning was too recent (once per day)
            if self.last_learning_time and time.time() - self.last_learning_time < 86400:
                return
            
            logger.info("Running self-learning")
            
            # TODO: Implement self-learning
            
            # Update last learning time
            self.last_learning_time = time.time()
            
            logger.info("Self-learning completed")
        
        except Exception as e:
            logger.error(f"Error running self-learning: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _generate_daily_summary(self):
        """Generate daily summary."""
        try:
            # Skip if last summary was too recent (once per day)
            if self.last_daily_summary_time and time.time() - self.last_daily_summary_time < 86400:
                return
            
            logger.info("Generating daily summary")
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics()
            
            # Generate summary message
            summary = f"Daily Trading Summary\n\n"
            summary += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
            summary += f"Account Balance: ${self.account_balance:.2f}\n"
            summary += f"Total Return: {metrics.get('total_return', 0):.2%}\n"
            summary += f"Daily Return: {metrics.get('daily_return', 0):.2%}\n"
            summary += f"Win Rate: {metrics.get('win_rate', 0):.2%}\n"
            summary += f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
            summary += f"Success Rate: {metrics.get('success_rate', 0):.2%}\n"
            summary += f"Open Positions: {len(self.open_positions)}\n"
            summary += f"Trades Today: {metrics.get('trades_per_day', 0)}\n"
            
            # Send notification if enabled
            if self.notifications['ENABLE_WEBHOOK'] and self.notifications['NOTIFY_ON_DAILY_SUMMARY']:
                self._send_notification("Daily Trading Summary", summary)
            
            # Update last summary time
            self.last_daily_summary_time = time.time()
            
            logger.info("Daily summary generated")
        
        except Exception as e:
            logger.error(f"Error generating daily summary: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _calculate_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            logger.debug("Calculating performance metrics")
            
            # Get closed trades
            closed_trades = [trade for trade in self.trade_history if trade['status'] == 'CLOSED']
            
            # Skip if no closed trades
            if not closed_trades:
                return {}
            
            # Calculate metrics
            metrics = {}
            
            # Total return
            initial_capital = self.initial_capital
            current_capital = self.account_balance
            for position in self.open_positions.values():
                current_capital += position['unrealized_profit_loss']
            
            total_return = (current_capital - initial_capital) / initial_capital
            metrics['total_return'] = total_return
            
            # Daily return (last 24 hours)
            recent_trades = [
                trade for trade in closed_trades
                if trade['exit_time'] > int((time.time() - 86400) * 1000)
            ]
            
            if recent_trades:
                daily_profit = sum(trade['profit_loss'] for trade in recent_trades)
                daily_return = daily_profit / initial_capital
                metrics['daily_return'] = daily_return
            else:
                metrics['daily_return'] = 0.0
            
            # Win rate
            winning_trades = [trade for trade in closed_trades if trade['profit_loss'] > 0]
            win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0.0
            metrics['win_rate'] = win_rate
            
            # Profit factor
            gross_profit = sum(trade['profit_loss'] for trade in winning_trades) if winning_trades else 0.0
            losing_trades = [trade for trade in closed_trades if trade['profit_loss'] < 0]
            gross_loss = abs(sum(trade['profit_loss'] for trade in losing_trades)) if losing_trades else 0.0
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            metrics['profit_factor'] = profit_factor
            
            # Average win and loss
            avg_win = gross_profit / len(winning_trades) if winning_trades else 0.0
            avg_loss = gross_loss / len(losing_trades) if losing_trades else 0.0
            
            metrics['average_win'] = avg_win
            metrics['average_loss'] = avg_loss
            
            # Average win/loss ratio
            avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
            metrics['average_win_loss_ratio'] = avg_win_loss_ratio
            
            # Expectancy
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            metrics['expectancy'] = expectancy
            
            # Average holding period
            avg_holding_period = sum(trade['trade_duration'] for trade in closed_trades) / len(closed_trades) / 3600 if closed_trades else 0.0
            metrics['average_holding_period'] = avg_holding_period
            
            # Trade count
            metrics['trade_count'] = len(closed_trades)
            metrics['winning_trades'] = len(winning_trades)
            metrics['losing_trades'] = len(losing_trades)
            
            # Consecutive wins/losses
            consecutive_wins = 0
            consecutive_losses = 0
            current_streak = 0
            
            for trade in sorted(closed_trades, key=lambda x: x['exit_time']):
                if trade['profit_loss'] > 0:
                    if current_streak > 0:
                        current_streak += 1
                    else:
                        current_streak = 1
                else:
                    if current_streak < 0:
                        current_streak -= 1
                    else:
                        current_streak = -1
                
                consecutive_wins = max(consecutive_wins, current_streak if current_streak > 0 else 0)
                consecutive_losses = min(consecutive_losses, current_streak if current_streak < 0 else 0)
            
            metrics['consecutive_wins'] = consecutive_wins
            metrics['consecutive_losses'] = abs(consecutive_losses)
            
            # Largest win/loss
            largest_win = max(trade['profit_loss'] for trade in winning_trades) if winning_trades else 0.0
            largest_loss = min(trade['profit_loss'] for trade in losing_trades) if losing_trades else 0.0
            
            metrics['largest_win'] = largest_win
            metrics['largest_loss'] = abs(largest_loss)
            
            # Trades per day
            days_trading = (max(trade['exit_time'] for trade in closed_trades) - min(trade['entry_time'] for trade in closed_trades)) / (1000 * 86400) if closed_trades else 1.0
            days_trading = max(days_trading, 1.0)  # Ensure at least 1 day
            
            trades_per_day = len(closed_trades) / days_trading
            metrics['trades_per_day'] = trades_per_day
            
            # Profit per day
            profit_per_day = sum(trade['profit_loss'] for trade in closed_trades) / days_trading
            metrics['profit_per_day'] = profit_per_day
            
            # Success rate (compared to target)
            daily_return_target = self.daily_profit_target
            success_rate = min(1.0, metrics['daily_return'] / daily_return_target) if daily_return_target > 0 else 0.0
            metrics['success_rate'] = success_rate
            
            # Save metrics to database
            self._save_metrics_to_db(metrics)
            
            logger.debug(f"Calculated {len(metrics)} performance metrics")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def _save_metrics_to_db(self, metrics: Dict):
        """
        Save performance metrics to database.
        
        Args:
            metrics: Dictionary with performance metrics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = int(time.time() * 1000)
            
            for metric_name, metric_value in metrics.items():
                cursor.execute(f'''
                INSERT INTO {DATABASE['TABLES']['paper_performance']}
                (timestamp, metric_name, metric_value, created_at)
                VALUES (?, ?, ?, ?)
                ''', (
                    timestamp,
                    metric_name,
                    metric_value,
                    int(time.time() * 1000)
                ))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logger.error(f"Error saving metrics to database: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _get_current_price(self, trading_pair: str) -> Optional[float]:
        """
        Get current price for a trading pair.
        
        Args:
            trading_pair: Trading pair
            
        Returns:
            Current price or None if not available
        """
        try:
            # Get latest data
            data = self._get_latest_data(trading_pair, self.timeframes[0])
            
            if data is not None and not data.empty:
                return data['close'].iloc[-1]
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting current price: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _get_latest_data(self, trading_pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get latest data for a trading pair and timeframe.
        
        Args:
            trading_pair: Trading pair
            timeframe: Timeframe
            
        Returns:
            DataFrame with latest data or None if not available
        """
        try:
            # Get coin symbol
            coin = trading_pair.replace('USDT', '')
            
            # Get latest data
            data = self.market_data_collector.get_latest_data(
                coin=coin,
                timeframe=timeframe,
                limit=100
            )
            
            return data
        
        except Exception as e:
            logger.error(f"Error getting latest data: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _detect_market_condition(self, data: pd.DataFrame) -> str:
        """
        Detect market condition based on technical indicators.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Market condition string
        """
        try:
            # This is a simplified implementation
            # In a real system, we would use more sophisticated methods
            
            # Calculate trend strength using ADX
            if 'adx' in data.columns:
                adx = data['adx'].iloc[-1]
                
                # Calculate volatility using ATR
                atr = data['atr'].iloc[-1] if 'atr' in data.columns else 0.0
                close = data['close'].iloc[-1]
                volatility = atr / close if close > 0 else 0.0
                
                # Calculate volume relative to average
                volume = data['volume'].iloc[-1] if 'volume' in data.columns else 0.0
                volume_sma = data['volume_sma'].iloc[-1] if 'volume_sma' in data.columns else 0.0
                volume_ratio = volume / volume_sma if volume_sma > 0 else 1.0
                
                # Determine trend direction
                trend_direction = 0.0
                if 'macd' in data.columns:
                    trend_direction = 1.0 if data['macd'].iloc[-1] > 0 else -1.0
                
                # Determine market condition
                if adx > 25:  # Strong trend
                    if trend_direction > 0:
                        return 'TRENDING_UP'
                    elif trend_direction < 0:
                        return 'TRENDING_DOWN'
                    else:
                        return 'TRENDING'
                elif volatility > 0.03:  # High volatility
                    return 'VOLATILE'
                elif volume_ratio > 1.5:  # High volume
                    return 'BREAKOUT'
                else:  # Low volatility, low volume
                    return 'RANGING'
            
            return 'UNKNOWN'
        
        except Exception as e:
            logger.error(f"Error detecting market condition: {str(e)}")
            logger.error(traceback.format_exc())
            return 'UNKNOWN'
    
    def _has_position_for_pair(self, trading_pair: str) -> bool:
        """
        Check if there is an open position for a trading pair.
        
        Args:
            trading_pair: Trading pair
            
        Returns:
            True if there is an open position, False otherwise
        """
        for position in self.open_positions.values():
            if position['trading_pair'] == trading_pair:
                return True
        
        return False
    
    def _get_available_balance(self) -> float:
        """
        Get available balance.
        
        Returns:
            Available balance
        """
        return self.account_balance
    
    def get_status(self) -> Dict:
        """
        Get current status of the paper trading engine.
        
        Returns:
            Dictionary with current status
        """
        try:
            # Calculate equity
            equity = self.account_balance
            for position in self.open_positions.values():
                equity += position['unrealized_profit_loss']
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics()
            
            return {
                'status': 'running' if self.is_running else 'stopped',
                'account_balance': self.account_balance,
                'equity': equity,
                'open_positions': len(self.open_positions),
                'total_trades': len(self.trade_history),
                'total_return': metrics.get('total_return', 0.0),
                'daily_return': metrics.get('daily_return', 0.0),
                'win_rate': metrics.get('win_rate', 0.0),
                'success_rate': metrics.get('success_rate', 0.0),
                'last_update': self.last_update_time
            }
        
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_open_positions(self) -> List[Dict]:
        """
        Get list of open positions.
        
        Returns:
            List of open positions
        """
        return list(self.open_positions.values())
    
    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """
        Get trade history.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of trades
        """
        return self.trade_history[:limit]
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        return self._calculate_performance_metrics()
    
    def __str__(self) -> str:
        """String representation of the paper trading engine."""
        return f"PaperTradingEngine(balance=${self.account_balance:.2f}, positions={len(self.open_positions)})"
