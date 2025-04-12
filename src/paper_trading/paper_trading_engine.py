"""
Paper trading engine for cryptocurrency trading.
Simulates trading without real money.
"""

import os
import time
import json
import logging
import pandas as pd
import numpy as np
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

from ..data.market_data_collector import MarketDataCollector
from ..analysis.strategy_manager import StrategyManager
from ..prediction.ensemble_manager import EnsembleManager
from ..risk.risk_manager import RiskManager
from ..risk.position_sizer import PositionSizer
from .config import PAPER_TRADING, EXCHANGE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('paper_trading_engine')

class PaperTradingEngine:
    """
    Paper trading engine for cryptocurrency trading.
    Simulates trading without real money.
    """
    
    def __init__(self, config=None):
        """
        Initialize the paper trading engine.
        
        Args:
            config: Configuration dictionary (optional)
        """
        # Load configuration
        self.config = config or PAPER_TRADING
        
        # Initialize components
        self.market_data_collector = MarketDataCollector(
            api_key=EXCHANGE['API_KEY'],
            api_secret=EXCHANGE['API_SECRET'],
            api_passphrase=EXCHANGE['API_PASSPHRASE'],
            is_sandbox=EXCHANGE['USE_SANDBOX']
        )
        self.strategy_manager = StrategyManager()
        self.ensemble_manager = EnsembleManager()
        self.risk_manager = RiskManager()
        self.position_sizer = PositionSizer()
        
        # Initialize database
        self._init_database()
        
        # Initialize state
        self.is_running = False
        self.trading_thread = None
        self.last_update_time = None
        
        # Load initial state
        self._load_state()
        
        logger.info("Paper trading engine initialized")
    
    def _init_database(self):
        """
        Initialize the database.
        """
        try:
            # Create database directory if it doesn't exist
            db_dir = os.path.dirname(self.config['DB_PATH'])
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            
            # Connect to database
            conn = sqlite3.connect(self.config['DB_PATH'])
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_balance (
                timestamp INTEGER PRIMARY KEY,
                balance REAL,
                equity REAL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_positions (
                position_id TEXT PRIMARY KEY,
                timestamp INTEGER,
                trading_pair TEXT,
                side TEXT,
                entry_price REAL,
                quantity REAL,
                position_size REAL,
                stop_loss REAL,
                take_profit REAL,
                trailing_stop REAL,
                status TEXT,
                entry_time INTEGER,
                exit_time INTEGER,
                exit_price REAL,
                profit_loss REAL,
                profit_loss_percentage REAL,
                exit_reason TEXT
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_trades (
                trade_id TEXT PRIMARY KEY,
                timestamp INTEGER,
                trading_pair TEXT,
                side TEXT,
                price REAL,
                quantity REAL,
                cost REAL,
                fee REAL,
                position_id TEXT
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_performance (
                timestamp INTEGER,
                metric_name TEXT,
                metric_value REAL,
                PRIMARY KEY (timestamp, metric_name)
            )
            ''')
            
            # Commit changes
            conn.commit()
            
            # Close connection
            conn.close()
            
            logger.info("Database initialized")
        
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def _load_state(self):
        """
        Load the initial state from the database.
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.config['DB_PATH'])
            cursor = conn.cursor()
            
            # Get the latest balance
            cursor.execute('''
            SELECT balance, equity FROM paper_balance
            ORDER BY timestamp DESC
            LIMIT 1
            ''')
            
            result = cursor.fetchone()
            
            if result:
                self.balance = result[0]
                self.equity = result[1]
            else:
                # Initialize with default values
                self.balance = self.config['INITIAL_CAPITAL']
                self.equity = self.config['INITIAL_CAPITAL']
                
                # Insert initial balance
                cursor.execute('''
                INSERT INTO paper_balance (timestamp, balance, equity)
                VALUES (?, ?, ?)
                ''', (int(time.time() * 1000), self.balance, self.equity))
            
            # Get open positions
            cursor.execute('''
            SELECT * FROM paper_positions
            WHERE status = 'OPEN'
            ''')
            
            columns = [description[0] for description in cursor.description]
            self.open_positions = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            # Commit changes
            conn.commit()
            
            # Close connection
            conn.close()
            
            logger.info(f"State loaded: balance={self.balance}, equity={self.equity}, open_positions={len(self.open_positions)}")
        
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            # Initialize with default values
            self.balance = self.config['INITIAL_CAPITAL']
            self.equity = self.config['INITIAL_CAPITAL']
            self.open_positions = []
    
    def start(self):
        """
        Start the paper trading engine.
        """
        if self.is_running:
            logger.warning("Paper trading engine is already running")
            return
        
        logger.info("Starting paper trading engine")
        
        self.is_running = True
        self.last_update_time = time.time()
        
        # Start trading thread
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        logger.info("Paper trading engine started")
    
    def stop(self):
        """
        Stop the paper trading engine.
        """
        if not self.is_running:
            logger.warning("Paper trading engine is not running")
            return
        
        logger.info("Stopping paper trading engine")
        
        self.is_running = False
        
        # Wait for trading thread to stop
        if self.trading_thread:
            self.trading_thread.join(timeout=10)
        
        logger.info("Paper trading engine stopped")
    
    def _trading_loop(self):
        """
        Main trading loop.
        """
        while self.is_running:
            try:
                # Update market data
                self._update_market_data()
                
                # Update positions
                self._update_positions()
                
                # Generate trading signals
                signals = self._generate_signals()
                
                # Execute signals
                self._execute_signals(signals)
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Update state
                self._update_state()
                
                # Sleep until next update
                time.sleep(self.config['UPDATE_INTERVAL'])
            
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(10)  # Sleep for a while before retrying
    
    def _update_market_data(self):
        """
        Update market data.
        """
        try:
            # Get current prices
            self.current_prices = self.market_data_collector.get_current_prices()
            
            # Get latest klines for all trading pairs and timeframes
            self.latest_klines = {}
            
            for pair in self.config['TRADING_PAIRS']:
                self.latest_klines[pair] = {}
                
                for timeframe in self.config['TIMEFRAMES']:
                    klines = self.market_data_collector.get_latest_klines(
                        pair, timeframe, self.config['KLINES_LIMIT']
                    )
                    
                    if not klines.empty:
                        self.latest_klines[pair][timeframe] = klines
            
            self.last_update_time = time.time()
            
            logger.debug("Market data updated")
        
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
    
    def _update_positions(self):
        """
        Update open positions.
        """
        try:
            if not self.open_positions:
                return
            
            # Connect to database
            conn = sqlite3.connect(self.config['DB_PATH'])
            cursor = conn.cursor()
            
            # Update each position
            for position in self.open_positions:
                # Get current price
                current_price = self.current_prices.get(position['trading_pair'])
                
                if not current_price:
                    logger.warning(f"No current price for {position['trading_pair']}")
                    continue
                
                # Calculate unrealized profit/loss
                if position['side'] == 'BUY':
                    unrealized_pl = (current_price - position['entry_price']) * position['quantity']
                    unrealized_pl_percentage = (current_price - position['entry_price']) / position['entry_price']
                else:  # SELL
                    unrealized_pl = (position['entry_price'] - current_price) * position['quantity']
                    unrealized_pl_percentage = (position['entry_price'] - current_price) / position['entry_price']
                
                # Check stop loss
                stop_triggered = False
                exit_reason = None
                
                if position['stop_loss'] and (
                    (position['side'] == 'BUY' and current_price <= position['stop_loss']) or
                    (position['side'] == 'SELL' and current_price >= position['stop_loss'])
                ):
                    stop_triggered = True
                    exit_reason = 'Stop Loss'
                
                # Check take profit
                elif position['take_profit'] and (
                    (position['side'] == 'BUY' and current_price >= position['take_profit']) or
                    (position['side'] == 'SELL' and current_price <= position['take_profit'])
                ):
                    stop_triggered = True
                    exit_reason = 'Take Profit'
                
                # Check trailing stop
                elif position['trailing_stop'] and (
                    (position['side'] == 'BUY' and current_price <= position['trailing_stop']) or
                    (position['side'] == 'SELL' and current_price >= position['trailing_stop'])
                ):
                    stop_triggered = True
                    exit_reason = 'Trailing Stop'
                
                # Update trailing stop if price moved in favorable direction
                if not stop_triggered and position['trailing_stop']:
                    if position['side'] == 'BUY' and current_price > position['entry_price']:
                        # Calculate new trailing stop
                        new_trailing_stop = current_price * (1 - self.config['TRAILING_STOP_PERCENTAGE'])
                        
                        # Update if new trailing stop is higher
                        if new_trailing_stop > position['trailing_stop']:
                            position['trailing_stop'] = new_trailing_stop
                            
                            # Update in database
                            cursor.execute('''
                            UPDATE paper_positions
                            SET trailing_stop = ?
                            WHERE position_id = ?
                            ''', (position['trailing_stop'], position['position_id']))
                    
                    elif position['side'] == 'SELL' and current_price < position['entry_price']:
                        # Calculate new trailing stop
                        new_trailing_stop = current_price * (1 + self.config['TRAILING_STOP_PERCENTAGE'])
                        
                        # Update if new trailing stop is lower
                        if new_trailing_stop < position['trailing_stop']:
                            position['trailing_stop'] = new_trailing_stop
                            
                            # Update in database
                            cursor.execute('''
                            UPDATE paper_positions
                            SET trailing_stop = ?
                            WHERE position_id = ?
                            ''', (position['trailing_stop'], position['position_id']))
                
                # Close position if stop triggered
                if stop_triggered:
                    # Update position
                    position['status'] = 'CLOSED'
                    position['exit_time'] = int(time.time() * 1000)
                    position['exit_price'] = current_price
                    position['profit_loss'] = unrealized_pl
                    position['profit_loss_percentage'] = unrealized_pl_percentage
                    position['exit_reason'] = exit_reason
                    
                    # Update in database
                    cursor.execute('''
                    UPDATE paper_positions
                    SET status = ?, exit_time = ?, exit_price = ?, profit_loss = ?, profit_loss_percentage = ?, exit_reason = ?
                    WHERE position_id = ?
                    ''', (
                        position['status'],
                        position['exit_time'],
                        position['exit_price'],
                        position['profit_loss'],
                        position['profit_loss_percentage'],
                        position['exit_reason'],
                        position['position_id']
                    ))
                    
                    # Update balance
                    self.balance += position['position_size'] + position['profit_loss']
                    
                    # Record trade
                    trade_id = f"{position['trading_pair']}_{position['side']}_CLOSE_{position['exit_time']}"
                    
                    cursor.execute('''
                    INSERT INTO paper_trades (trade_id, timestamp, trading_pair, side, price, quantity, cost, fee, position_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        trade_id,
                        position['exit_time'],
                        position['trading_pair'],
                        'SELL' if position['side'] == 'BUY' else 'BUY',  # Opposite side for closing
                        position['exit_price'],
                        position['quantity'],
                        position['quantity'] * position['exit_price'],
                        0,  # No fee in paper trading
                        position['position_id']
                    ))
                    
                    logger.info(f"Position closed: {position['position_id']}, {exit_reason}, PL: {position['profit_loss']:.2f} ({position['profit_loss_percentage']:.2%})")
            
            # Remove closed positions
            self.open_positions = [p for p in self.open_positions if p['status'] == 'OPEN']
            
            # Commit changes
            conn.commit()
            
            # Close connection
            conn.close()
            
            logger.debug("Positions updated")
        
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
    
    def _generate_signals(self):
        """
        Generate trading signals.
        
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        try:
            # Skip if no market data
            if not hasattr(self, 'latest_klines') or not self.latest_klines:
                return signals
            
            # Generate signals for each trading pair
            for pair in self.config['TRADING_PAIRS']:
                # Skip if no klines for this pair
                if pair not in self.latest_klines or not self.latest_klines[pair]:
                    continue
                
                # Get klines for all timeframes
                pair_klines = self.latest_klines[pair]
                
                # Skip if no klines for any timeframe
                if not pair_klines:
                    continue
                
                # Calculate technical indicators
                indicators = {}
                
                for timeframe, klines in pair_klines.items():
                    indicators[timeframe] = self.strategy_manager.calculate_indicators(klines)
                
                # Generate signals from technical analysis
                ta_signals = self.strategy_manager.generate_signals(pair, pair_klines, indicators)
                
                # Generate signals from machine learning
                ml_signals = self.ensemble_manager.generate_predictions(pair, pair_klines, indicators)
                
                # Combine signals
                combined_signal = self._combine_signals(ta_signals, ml_signals)
                
                if combined_signal:
                    signals.append(combined_signal)
            
            logger.debug(f"Generated {len(signals)} signals")
            
            return signals
        
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return signals
    
    def _combine_signals(self, ta_signals, ml_signals):
        """
        Combine technical analysis and machine learning signals.
        
        Args:
            ta_signals: Technical analysis signals
            ml_signals: Machine learning signals
            
        Returns:
            Combined signal dictionary or None
        """
        try:
            # Skip if no signals
            if not ta_signals and not ml_signals:
                return None
            
            # Initialize combined signal
            combined_signal = {
                'timestamp': int(time.time() * 1000),
                'trading_pair': ta_signals.get('trading_pair') or ml_signals.get('trading_pair'),
                'signal': 'NEUTRAL',
                'confidence': 0,
                'timeframe': ta_signals.get('timeframe') or ml_signals.get('timeframe'),
                'price': ta_signals.get('price') or ml_signals.get('price'),
                'indicators': {},
                'ml_prediction': {},
                'market_condition': ta_signals.get('market_condition') or ml_signals.get('market_condition', 'UNKNOWN')
            }
            
            # Add technical indicators
            if ta_signals and 'indicators' in ta_signals:
                combined_signal['indicators'] = ta_signals['indicators']
            
            # Add machine learning prediction
            if ml_signals and 'prediction' in ml_signals:
                combined_signal['ml_prediction'] = ml_signals['prediction']
            
            # Calculate combined signal
            ta_weight = self.config['TA_WEIGHT']
            ml_weight = self.config['ML_WEIGHT']
            
            ta_signal_value = 0
            ml_signal_value = 0
            
            if ta_signals and 'signal' in ta_signals:
                ta_signal_map = {
                    'STRONG_BUY': 2,
                    'BUY': 1,
                    'NEUTRAL': 0,
                    'SELL': -1,
                    'STRONG_SELL': -2
                }
                ta_signal_value = ta_signal_map.get(ta_signals['signal'], 0)
            
            if ml_signals and 'signal' in ml_signals:
                ml_signal_map = {
                    'STRONG_BUY': 2,
                    'BUY': 1,
                    'NEUTRAL': 0,
                    'SELL': -1,
                    'STRONG_SELL': -2
                }
                ml_signal_value = ml_signal_map.get(ml_signals['signal'], 0)
            
            # Calculate weighted average
            combined_value = (ta_signal_value * ta_weight + ml_signal_value * ml_weight) / (ta_weight + ml_weight)
            
            # Map back to signal
            if combined_value >= 1.5:
                combined_signal['signal'] = 'STRONG_BUY'
            elif combined_value >= 0.5:
                combined_signal['signal'] = 'BUY'
            elif combined_value <= -1.5:
                combined_signal['signal'] = 'STRONG_SELL'
            elif combined_value <= -0.5:
                combined_signal['signal'] = 'SELL'
            else:
                combined_signal['signal'] = 'NEUTRAL'
            
            # Calculate confidence
            ta_confidence = ta_signals.get('confidence', 0.5)
            ml_confidence = ml_signals.get('confidence', 0.5)
            
            combined_signal['confidence'] = (ta_confidence * ta_weight + ml_confidence * ml_weight) / (ta_weight + ml_weight)
            
            return combined_signal
        
        except Exception as e:
            logger.error(f"Error combining signals: {str(e)}")
            return None
    
    def _execute_signals(self, signals):
        """
        Execute trading signals.
        
        Args:
            signals: List of signal dictionaries
        """
        try:
            # Skip if no signals
            if not signals:
                return
            
            # Connect to database
            conn = sqlite3.connect(self.config['DB_PATH'])
            cursor = conn.cursor()
            
            # Process each signal
            for signal in signals:
                # Skip neutral signals
                if signal['signal'] == 'NEUTRAL':
                    continue
                
                # Get trading pair
                pair = signal['trading_pair']
                
                # Get current price
                current_price = self.current_prices.get(pair)
                
                if not current_price:
                    logger.warning(f"No current price for {pair}")
                    continue
                
                # Check if we already have an open position for this pair
                existing_position = next((p for p in self.open_positions if p['trading_pair'] == pair), None)
                
                # Skip if we already have a position in the same direction
                if existing_position:
                    if (signal['signal'] in ['BUY', 'STRONG_BUY'] and existing_position['side'] == 'BUY') or \
                       (signal['signal'] in ['SELL', 'STRONG_SELL'] and existing_position['side'] == 'SELL'):
                        continue
                    
                    # Close existing position if signal is in opposite direction
                    if (signal['signal'] in ['SELL', 'STRONG_SELL'] and existing_position['side'] == 'BUY') or \
                       (signal['signal'] in ['BUY', 'STRONG_BUY'] and existing_position['side'] == 'SELL'):
                        # Update position
                        existing_position['status'] = 'CLOSED'
                        existing_position['exit_time'] = int(time.time() * 1000)
                        existing_position['exit_price'] = current_price
                        
                        # Calculate profit/loss
                        if existing_position['side'] == 'BUY':
                            profit_loss = (current_price - existing_position['entry_price']) * existing_position['quantity']
                            profit_loss_percentage = (current_price - existing_position['entry_price']) / existing_position['entry_price']
                        else:  # SELL
                            profit_loss = (existing_position['entry_price'] - current_price) * existing_position['quantity']
                            profit_loss_percentage = (existing_position['entry_price'] - current_price) / existing_position['entry_price']
                        
                        existing_position['profit_loss'] = profit_loss
                        existing_position['profit_loss_percentage'] = profit_loss_percentage
                        existing_position['exit_reason'] = 'Signal Reversal'
                        
                        # Update in database
                        cursor.execute('''
                        UPDATE paper_positions
                        SET status = ?, exit_time = ?, exit_price = ?, profit_loss = ?, profit_loss_percentage = ?, exit_reason = ?
                        WHERE position_id = ?
                        ''', (
                            existing_position['status'],
                            existing_position['exit_time'],
                            existing_position['exit_price'],
                            existing_position['profit_loss'],
                            existing_position['profit_loss_percentage'],
                            existing_position['exit_reason'],
                            existing_position['position_id']
                        ))
                        
                        # Update balance
                        self.balance += existing_position['position_size'] + existing_position['profit_loss']
                        
                        # Record trade
                        trade_id = f"{existing_position['trading_pair']}_{existing_position['side']}_CLOSE_{existing_position['exit_time']}"
                        
                        cursor.execute('''
                        INSERT INTO paper_trades (trade_id, timestamp, trading_pair, side, price, quantity, cost, fee, position_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            trade_id,
                            existing_position['exit_time'],
                            existing_position['trading_pair'],
                            'SELL' if existing_position['side'] == 'BUY' else 'BUY',  # Opposite side for closing
                            existing_position['exit_price'],
                            existing_position['quantity'],
                            existing_position['quantity'] * existing_position['exit_price'],
                            0,  # No fee in paper trading
                            existing_position['position_id']
                        ))
                        
                        logger.info(f"Position closed: {existing_position['position_id']}, Signal Reversal, PL: {existing_position['profit_loss']:.2f} ({existing_position['profit_loss_percentage']:.2%})")
                        
                        # Remove from open positions
                        self.open_positions.remove(existing_position)
                
                # Check if we have reached the maximum number of open positions
                if len(self.open_positions) >= self.config['MAX_OPEN_POSITIONS']:
                    logger.warning(f"Maximum number of open positions reached: {self.config['MAX_OPEN_POSITIONS']}")
                    continue
                
                # Check if we have enough balance
                if self.balance <= 0:
                    logger.warning("Insufficient balance")
                    continue
                
                # Determine position size
                risk_params = {
                    'balance': self.balance,
                    'risk_per_trade': self.config['RISK_PER_TRADE'],
                    'signal_strength': 1.0 if signal['signal'] in ['STRONG_BUY', 'STRONG_SELL'] else 0.5,
                    'confidence': signal['confidence'],
                    'market_condition': signal['market_condition']
                }
                
                position_size = self.position_sizer.calculate_position_size(risk_params)
                
                # Ensure position size is within limits
                position_size = min(position_size, self.balance, self.config['MAX_POSITION_SIZE'])
                position_size = max(position_size, self.config['MIN_POSITION_SIZE'])
                
                # Calculate quantity
                quantity = position_size / current_price
                
                # Round quantity to appropriate precision
                quantity = round(quantity, 6)
                
                # Skip if quantity is too small
                if quantity <= 0:
                    logger.warning(f"Quantity too small: {quantity}")
                    continue
                
                # Determine side
                side = 'BUY' if signal['signal'] in ['BUY', 'STRONG_BUY'] else 'SELL'
                
                # Calculate stop loss and take profit
                if side == 'BUY':
                    stop_loss = current_price * (1 - self.config['STOP_LOSS_PERCENTAGE'])
                    take_profit = current_price * (1 + self.config['TAKE_PROFIT_PERCENTAGE'])
                    trailing_stop = current_price * (1 - self.config['TRAILING_STOP_PERCENTAGE'])
                else:  # SELL
                    stop_loss = current_price * (1 + self.config['STOP_LOSS_PERCENTAGE'])
                    take_profit = current_price * (1 - self.config['TAKE_PROFIT_PERCENTAGE'])
                    trailing_stop = current_price * (1 + self.config['TRAILING_STOP_PERCENTAGE'])
                
                # Create position
                timestamp = int(time.time() * 1000)
                position_id = f"{pair}_{side}_{timestamp}"
                
                position = {
                    'position_id': position_id,
                    'timestamp': timestamp,
                    'trading_pair': pair,
                    'side': side,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'position_size': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'trailing_stop': trailing_stop,
                    'status': 'OPEN',
                    'entry_time': timestamp,
                    'exit_time': None,
                    'exit_price': None,
                    'profit_loss': None,
                    'profit_loss_percentage': None,
                    'exit_reason': None
                }
                
                # Insert into database
                cursor.execute('''
                INSERT INTO paper_positions (
                    position_id, timestamp, trading_pair, side, entry_price, quantity, position_size,
                    stop_loss, take_profit, trailing_stop, status, entry_time, exit_time,
                    exit_price, profit_loss, profit_loss_percentage, exit_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    position['position_id'],
                    position['timestamp'],
                    position['trading_pair'],
                    position['side'],
                    position['entry_price'],
                    position['quantity'],
                    position['position_size'],
                    position['stop_loss'],
                    position['take_profit'],
                    position['trailing_stop'],
                    position['status'],
                    position['entry_time'],
                    position['exit_time'],
                    position['exit_price'],
                    position['profit_loss'],
                    position['profit_loss_percentage'],
                    position['exit_reason']
                ))
                
                # Record trade
                trade_id = f"{pair}_{side}_OPEN_{timestamp}"
                
                cursor.execute('''
                INSERT INTO paper_trades (trade_id, timestamp, trading_pair, side, price, quantity, cost, fee, position_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_id,
                    timestamp,
                    pair,
                    side,
                    current_price,
                    quantity,
                    position_size,
                    0,  # No fee in paper trading
                    position_id
                ))
                
                # Update balance
                self.balance -= position_size
                
                # Add to open positions
                self.open_positions.append(position)
                
                logger.info(f"Position opened: {position_id}, {side} {quantity} {pair} at {current_price}")
            
            # Commit changes
            conn.commit()
            
            # Close connection
            conn.close()
            
            logger.debug("Signals executed")
        
        except Exception as e:
            logger.error(f"Error executing signals: {str(e)}")
    
    def _update_performance_metrics(self):
        """
        Update performance metrics.
        """
        try:
            # Skip if no positions have been opened
            if not self.open_positions and not self._get_closed_positions_count():
                return
            
            # Connect to database
            conn = sqlite3.connect(self.config['DB_PATH'])
            cursor = conn.cursor()
            
            # Calculate equity
            equity = self.balance
            
            for position in self.open_positions:
                # Get current price
                current_price = self.current_prices.get(position['trading_pair'])
                
                if not current_price:
                    continue
                
                # Calculate unrealized profit/loss
                if position['side'] == 'BUY':
                    unrealized_pl = (current_price - position['entry_price']) * position['quantity']
                else:  # SELL
                    unrealized_pl = (position['entry_price'] - current_price) * position['quantity']
                
                equity += position['position_size'] + unrealized_pl
            
            self.equity = equity
            
            # Update balance and equity
            timestamp = int(time.time() * 1000)
            
            cursor.execute('''
            INSERT INTO paper_balance (timestamp, balance, equity)
            VALUES (?, ?, ?)
            ''', (timestamp, self.balance, self.equity))
            
            # Calculate daily return
            cursor.execute('''
            SELECT equity FROM paper_balance
            WHERE timestamp < ?
            ORDER BY timestamp DESC
            LIMIT 1
            ''', (timestamp - 24 * 60 * 60 * 1000,))
            
            result = cursor.fetchone()
            
            if result:
                previous_equity = result[0]
                daily_return = (self.equity - previous_equity) / previous_equity
                
                cursor.execute('''
                INSERT INTO paper_performance (timestamp, metric_name, metric_value)
                VALUES (?, ?, ?)
                ''', (timestamp, 'daily_return', daily_return))
            
            # Calculate win rate
            cursor.execute('''
            SELECT COUNT(*) FROM paper_positions
            WHERE status = 'CLOSED' AND profit_loss > 0
            ''')
            
            winning_trades = cursor.fetchone()[0]
            
            cursor.execute('''
            SELECT COUNT(*) FROM paper_positions
            WHERE status = 'CLOSED'
            ''')
            
            total_closed_trades = cursor.fetchone()[0]
            
            if total_closed_trades > 0:
                win_rate = winning_trades / total_closed_trades
                
                cursor.execute('''
                INSERT INTO paper_performance (timestamp, metric_name, metric_value)
                VALUES (?, ?, ?)
                ''', (timestamp, 'win_rate', win_rate))
            
            # Calculate profit factor
            cursor.execute('''
            SELECT SUM(profit_loss) FROM paper_positions
            WHERE status = 'CLOSED' AND profit_loss > 0
            ''')
            
            gross_profit = cursor.fetchone()[0] or 0
            
            cursor.execute('''
            SELECT SUM(ABS(profit_loss)) FROM paper_positions
            WHERE status = 'CLOSED' AND profit_loss < 0
            ''')
            
            gross_loss = cursor.fetchone()[0] or 0
            
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
                
                cursor.execute('''
                INSERT INTO paper_performance (timestamp, metric_name, metric_value)
                VALUES (?, ?, ?)
                ''', (timestamp, 'profit_factor', profit_factor))
            
            # Calculate maximum drawdown
            cursor.execute('''
            SELECT equity FROM paper_balance
            ORDER BY timestamp ASC
            ''')
            
            equity_values = [row[0] for row in cursor.fetchall()]
            
            if equity_values:
                max_drawdown = self._calculate_max_drawdown(equity_values)
                
                cursor.execute('''
                INSERT INTO paper_performance (timestamp, metric_name, metric_value)
                VALUES (?, ?, ?)
                ''', (timestamp, 'max_drawdown', max_drawdown))
            
            # Commit changes
            conn.commit()
            
            # Close connection
            conn.close()
            
            logger.debug("Performance metrics updated")
        
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    def _calculate_max_drawdown(self, equity_values):
        """
        Calculate maximum drawdown from equity values.
        
        Args:
            equity_values: List of equity values
            
        Returns:
            Maximum drawdown as a decimal
        """
        max_drawdown = 0
        peak = equity_values[0]
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            
            drawdown = (peak - equity) / peak
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def _get_closed_positions_count(self):
        """
        Get the number of closed positions.
        
        Returns:
            Number of closed positions
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.config['DB_PATH'])
            cursor = conn.cursor()
            
            # Get count
            cursor.execute('''
            SELECT COUNT(*) FROM paper_positions
            WHERE status = 'CLOSED'
            ''')
            
            count = cursor.fetchone()[0]
            
            # Close connection
            conn.close()
            
            return count
        
        except Exception as e:
            logger.error(f"Error getting closed positions count: {str(e)}")
            return 0
    
    def _update_state(self):
        """
        Update the state.
        """
        # Nothing to do here, state is updated in other methods
        pass
    
    def get_status(self):
        """
        Get the current status of the paper trading engine.
        
        Returns:
            Dictionary with status information
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.config['DB_PATH'])
            cursor = conn.cursor()
            
            # Get trade counts
            cursor.execute('''
            SELECT COUNT(*) FROM paper_positions
            ''')
            
            total_trades = cursor.fetchone()[0]
            
            cursor.execute('''
            SELECT COUNT(*) FROM paper_positions
            WHERE status = 'CLOSED' AND profit_loss > 0
            ''')
            
            winning_trades = cursor.fetchone()[0]
            
            cursor.execute('''
            SELECT COUNT(*) FROM paper_positions
            WHERE status = 'CLOSED'
            ''')
            
            total_closed_trades = cursor.fetchone()[0]
            
            # Calculate win rate
            win_rate = winning_trades / total_closed_trades if total_closed_trades > 0 else 0
            
            # Get latest performance metrics
            cursor.execute('''
            SELECT metric_value FROM paper_performance
            WHERE metric_name = 'daily_return'
            ORDER BY timestamp DESC
            LIMIT 1
            ''')
            
            result = cursor.fetchone()
            daily_return = result[0] if result else 0
            
            cursor.execute('''
            SELECT metric_value FROM paper_performance
            WHERE metric_name = 'profit_factor'
            ORDER BY timestamp DESC
            LIMIT 1
            ''')
            
            result = cursor.fetchone()
            profit_factor = result[0] if result else 0
            
            cursor.execute('''
            SELECT metric_value FROM paper_performance
            WHERE metric_name = 'max_drawdown'
            ORDER BY timestamp DESC
            LIMIT 1
            ''')
            
            result = cursor.fetchone()
            max_drawdown = result[0] if result else 0
            
            # Calculate total return
            cursor.execute('''
            SELECT equity FROM paper_balance
            ORDER BY timestamp ASC
            LIMIT 1
            ''')
            
            result = cursor.fetchone()
            initial_equity = result[0] if result else self.config['INITIAL_CAPITAL']
            
            total_return = (self.equity - initial_equity) / initial_equity if initial_equity > 0 else 0
            
            # Close connection
            conn.close()
            
            # Create status dictionary
            status = {
                'status': 'running' if self.is_running else 'stopped',
                'account_balance': self.balance,
                'equity': self.equity,
                'open_positions': len(self.open_positions),
                'total_trades': total_trades,
                'total_return': total_return,
                'daily_return': daily_return,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'last_update': int(self.last_update_time * 1000) if self.last_update_time else None
            }
            
            return status
        
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_open_positions(self):
        """
        Get open positions.
        
        Returns:
            List of open positions
        """
        try:
            # Update positions with current prices
            positions = []
            
            for position in self.open_positions:
                # Get current price
                current_price = self.current_prices.get(position['trading_pair'])
                
                if not current_price:
                    continue
                
                # Calculate unrealized profit/loss
                if position['side'] == 'BUY':
                    unrealized_pl = (current_price - position['entry_price']) * position['quantity']
                    unrealized_pl_percentage = (current_price - position['entry_price']) / position['entry_price']
                else:  # SELL
                    unrealized_pl = (position['entry_price'] - current_price) * position['quantity']
                    unrealized_pl_percentage = (position['entry_price'] - current_price) / position['entry_price']
                
                # Create position dictionary
                pos = dict(position)
                pos['current_price'] = current_price
                pos['unrealized_profit_loss'] = unrealized_pl
                pos['unrealized_profit_loss_percentage'] = unrealized_pl_percentage
                
                positions.append(pos)
            
            return positions
        
        except Exception as e:
            logger.error(f"Error getting open positions: {str(e)}")
            return []
    
    def get_trade_history(self, limit=100):
        """
        Get trade history.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of trades
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.config['DB_PATH'])
            cursor = conn.cursor()
            
            # Get closed positions
            cursor.execute('''
            SELECT * FROM paper_positions
            WHERE status = 'CLOSED'
            ORDER BY exit_time DESC
            LIMIT ?
            ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            trades = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            # Close connection
            conn.close()
            
            return trades
        
        except Exception as e:
            logger.error(f"Error getting trade history: {str(e)}")
            return []
    
    def get_performance_metrics(self):
        """
        Get performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.config['DB_PATH'])
            cursor = conn.cursor()
            
            # Get latest performance metrics
            cursor.execute('''
            SELECT metric_name, metric_value FROM paper_performance
            WHERE timestamp = (
                SELECT MAX(timestamp) FROM paper_performance
            )
            ''')
            
            metrics = dict(cursor.fetchall())
            
            # Get trade counts
            cursor.execute('''
            SELECT COUNT(*) FROM paper_positions
            WHERE status = 'CLOSED' AND profit_loss > 0
            ''')
            
            winning_trades = cursor.fetchone()[0]
            
            cursor.execute('''
            SELECT COUNT(*) FROM paper_positions
            WHERE status = 'CLOSED'
            ''')
            
            total_closed_trades = cursor.fetchone()[0]
            
            # Calculate win rate
            metrics['win_rate'] = winning_trades / total_closed_trades if total_closed_trades > 0 else 0
            
            # Calculate average win and loss
            cursor.execute('''
            SELECT AVG(profit_loss) FROM paper_positions
            WHERE status = 'CLOSED' AND profit_loss > 0
            ''')
            
            result = cursor.fetchone()
            metrics['average_win'] = result[0] if result[0] is not None else 0
            
            cursor.execute('''
            SELECT AVG(profit_loss) FROM paper_positions
            WHERE status = 'CLOSED' AND profit_loss < 0
            ''')
            
            result = cursor.fetchone()
            metrics['average_loss'] = result[0] if result[0] is not None else 0
            
            # Calculate largest win and loss
            cursor.execute('''
            SELECT MAX(profit_loss) FROM paper_positions
            WHERE status = 'CLOSED'
            ''')
            
            result = cursor.fetchone()
            metrics['largest_win'] = result[0] if result[0] is not None else 0
            
            cursor.execute('''
            SELECT MIN(profit_loss) FROM paper_positions
            WHERE status = 'CLOSED'
            ''')
            
            result = cursor.fetchone()
            metrics['largest_loss'] = result[0] if result[0] is not None else 0
            
            # Calculate profit factor
            cursor.execute('''
            SELECT SUM(profit_loss) FROM paper_positions
            WHERE status = 'CLOSED' AND profit_loss > 0
            ''')
            
            gross_profit = cursor.fetchone()[0] or 0
            
            cursor.execute('''
            SELECT SUM(ABS(profit_loss)) FROM paper_positions
            WHERE status = 'CLOSED' AND profit_loss < 0
            ''')
            
            gross_loss = cursor.fetchone()[0] or 0
            
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Calculate expectancy
            metrics['expectancy'] = (metrics['win_rate'] * metrics['average_win'] + (1 - metrics['win_rate']) * metrics['average_loss']) / abs(metrics['average_loss']) if metrics['average_loss'] != 0 else 0
            
            # Calculate total return
            cursor.execute('''
            SELECT equity FROM paper_balance
            ORDER BY timestamp ASC
            LIMIT 1
            ''')
            
            result = cursor.fetchone()
            initial_equity = result[0] if result else self.config['INITIAL_CAPITAL']
            
            metrics['total_return'] = (self.equity - initial_equity) / initial_equity if initial_equity > 0 else 0
            
            # Close connection
            conn.close()
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {}
    
    def reset(self):
        """
        Reset the paper trading engine.
        """
        try:
            # Stop trading
            self.stop()
            
            # Connect to database
            conn = sqlite3.connect(self.config['DB_PATH'])
            cursor = conn.cursor()
            
            # Delete all data
            cursor.execute('DELETE FROM paper_balance')
            cursor.execute('DELETE FROM paper_positions')
            cursor.execute('DELETE FROM paper_trades')
            cursor.execute('DELETE FROM paper_performance')
            
            # Commit changes
            conn.commit()
            
            # Close connection
            conn.close()
            
            # Reset state
            self.balance = self.config['INITIAL_CAPITAL']
            self.equity = self.config['INITIAL_CAPITAL']
            self.open_positions = []
            
            # Initialize database
            self._init_database()
            
            logger.info("Paper trading engine reset")
        
        except Exception as e:
            logger.error(f"Error resetting paper trading engine: {str(e)}")
    
    def update_config(self, config):
        """
        Update the configuration.
        
        Args:
            config: New configuration dictionary
        """
        try:
            # Update configuration
            self.config.update(config)
            
            logger.info("Configuration updated")
        
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
