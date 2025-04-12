"""
Base risk manager class for risk management and position sizing.
"""

import logging
import os
import time
import json
import sqlite3
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from .config import (
    RISK_STRATEGIES, POSITION_SIZING, RISK_LIMITS, STOP_LOSS, TAKE_PROFIT,
    MARKET_CONDITION_ADJUSTMENTS, COMPOUNDING, PORTFOLIO_ALLOCATION,
    RISK_MONITORING, DATABASE
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/risk_management.log',
    filemode='a'
)
logger = logging.getLogger('risk_manager')

class RiskManager:
    """
    Risk manager class for managing risk and position sizing.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the RiskManager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path or '/home/ubuntu/crypto_trading_model/data/market_data.db'
        
        # Risk strategy weights
        self.strategy_weights = {
            strategy: config.get('weight', 0.0)
            for strategy, config in RISK_STRATEGIES.items()
            if config.get('enabled', False)
        }
        
        # Normalize weights
        weight_sum = sum(self.strategy_weights.values())
        if weight_sum > 0:
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] /= weight_sum
        
        # Trading state
        self.account_balance = 10000.0  # Default starting balance
        self.current_positions = {}  # Current open positions
        self.trade_history = []  # History of trades
        self.daily_trades = 0  # Number of trades today
        self.daily_pnl = 0.0  # Daily profit and loss
        self.consecutive_wins = 0  # Number of consecutive winning trades
        self.consecutive_losses = 0  # Number of consecutive losing trades
        self.max_drawdown = 0.0  # Maximum drawdown
        self.peak_balance = self.account_balance  # Peak account balance
        
        # Circuit breaker state
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None
        
        # Ensure logs directory exists
        os.makedirs('/home/ubuntu/crypto_trading_model/logs', exist_ok=True)
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info("RiskManager initialized")
    
    def _init_database(self):
        """Initialize the SQLite database with required tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create risk settings table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['risk_settings_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                settings_json TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Create trade log table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['trade_log_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                trade_id TEXT NOT NULL,
                coin TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                position_size REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                risk_amount REAL NOT NULL,
                risk_percentage REAL NOT NULL,
                profit_loss REAL,
                profit_loss_percentage REAL,
                trade_duration INTEGER,
                status TEXT NOT NULL,
                strategy TEXT NOT NULL,
                signal_strength REAL,
                confidence REAL,
                market_condition TEXT,
                created_at INTEGER NOT NULL,
                closed_at INTEGER,
                notes TEXT
            )
            ''')
            
            # Create position sizing table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['position_sizing_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                coin TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                signal_strength REAL NOT NULL,
                confidence REAL NOT NULL,
                position_size REAL NOT NULL,
                position_size_percentage REAL NOT NULL,
                risk_amount REAL NOT NULL,
                risk_percentage REAL NOT NULL,
                strategy_weights TEXT NOT NULL,
                market_condition TEXT,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Create risk metrics table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['risk_metrics_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                account_balance REAL NOT NULL,
                open_positions INTEGER NOT NULL,
                daily_trades INTEGER NOT NULL,
                daily_pnl REAL NOT NULL,
                daily_pnl_percentage REAL NOT NULL,
                drawdown REAL NOT NULL,
                drawdown_percentage REAL NOT NULL,
                win_rate REAL,
                profit_factor REAL,
                average_win REAL,
                average_loss REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                calmar_ratio REAL,
                expectancy REAL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Create portfolio allocation table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['portfolio_allocation_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                coin TEXT NOT NULL,
                allocation REAL NOT NULL,
                allocation_percentage REAL NOT NULL,
                target_allocation REAL NOT NULL,
                target_allocation_percentage REAL NOT NULL,
                rebalance_required BOOLEAN NOT NULL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database tables initialized")
        
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def calculate_position_size(self, coin: str, signal_type: str, signal_strength: float, 
                               confidence: float, market_condition: str, 
                               current_price: float) -> Dict:
        """
        Calculate position size based on risk parameters and trading signals.
        
        Args:
            coin: Cryptocurrency symbol
            signal_type: Type of trading signal (e.g., BUY, SELL)
            signal_strength: Strength of the trading signal
            confidence: Confidence in the trading signal
            market_condition: Current market condition
            current_price: Current price of the coin
            
        Returns:
            Dictionary with position sizing information
        """
        try:
            logger.info(f"Calculating position size for {coin} {signal_type} signal")
            
            # Check if circuit breaker is active
            if self.circuit_breaker_active:
                if self.circuit_breaker_until and datetime.now() < self.circuit_breaker_until:
                    logger.warning(f"Circuit breaker active until {self.circuit_breaker_until}")
                    return {
                        'position_size': 0.0,
                        'position_size_percentage': 0.0,
                        'risk_amount': 0.0,
                        'risk_percentage': 0.0,
                        'stop_loss': 0.0,
                        'take_profit': 0.0,
                        'status': 'REJECTED',
                        'reason': 'Circuit breaker active'
                    }
                else:
                    # Reset circuit breaker
                    self.circuit_breaker_active = False
                    self.circuit_breaker_until = None
            
            # Check if signal meets minimum thresholds
            if confidence < POSITION_SIZING['CONFIDENCE_THRESHOLD']:
                logger.info(f"Signal confidence {confidence} below threshold {POSITION_SIZING['CONFIDENCE_THRESHOLD']}")
                return {
                    'position_size': 0.0,
                    'position_size_percentage': 0.0,
                    'risk_amount': 0.0,
                    'risk_percentage': 0.0,
                    'stop_loss': 0.0,
                    'take_profit': 0.0,
                    'status': 'REJECTED',
                    'reason': 'Confidence below threshold'
                }
            
            if abs(signal_strength) < POSITION_SIZING['SIGNAL_STRENGTH_THRESHOLD']:
                logger.info(f"Signal strength {signal_strength} below threshold {POSITION_SIZING['SIGNAL_STRENGTH_THRESHOLD']}")
                return {
                    'position_size': 0.0,
                    'position_size_percentage': 0.0,
                    'risk_amount': 0.0,
                    'risk_percentage': 0.0,
                    'stop_loss': 0.0,
                    'take_profit': 0.0,
                    'status': 'REJECTED',
                    'reason': 'Signal strength below threshold'
                }
            
            # Check daily trade limit
            if self.daily_trades >= RISK_LIMITS['MAX_DAILY_TRADES']:
                logger.warning(f"Daily trade limit reached: {self.daily_trades}/{RISK_LIMITS['MAX_DAILY_TRADES']}")
                return {
                    'position_size': 0.0,
                    'position_size_percentage': 0.0,
                    'risk_amount': 0.0,
                    'risk_percentage': 0.0,
                    'stop_loss': 0.0,
                    'take_profit': 0.0,
                    'status': 'REJECTED',
                    'reason': 'Daily trade limit reached'
                }
            
            # Check daily loss limit
            if self.daily_pnl < -RISK_LIMITS['MAX_DAILY_LOSS'] * self.account_balance:
                logger.warning(f"Daily loss limit reached: {self.daily_pnl}/{-RISK_LIMITS['MAX_DAILY_LOSS'] * self.account_balance}")
                return {
                    'position_size': 0.0,
                    'position_size_percentage': 0.0,
                    'risk_amount': 0.0,
                    'risk_percentage': 0.0,
                    'stop_loss': 0.0,
                    'take_profit': 0.0,
                    'status': 'REJECTED',
                    'reason': 'Daily loss limit reached'
                }
            
            # Check drawdown limit
            current_drawdown = (self.peak_balance - self.account_balance) / self.peak_balance if self.peak_balance > 0 else 0.0
            if current_drawdown > RISK_LIMITS['MAX_DRAWDOWN']:
                logger.warning(f"Drawdown limit reached: {current_drawdown}/{RISK_LIMITS['MAX_DRAWDOWN']}")
                return {
                    'position_size': 0.0,
                    'position_size_percentage': 0.0,
                    'risk_amount': 0.0,
                    'risk_percentage': 0.0,
                    'stop_loss': 0.0,
                    'take_profit': 0.0,
                    'status': 'REJECTED',
                    'reason': 'Drawdown limit reached'
                }
            
            # Check open trade limits
            open_trades = len(self.current_positions)
            if open_trades >= RISK_LIMITS['MAX_OPEN_TRADES']:
                logger.warning(f"Max open trades limit reached: {open_trades}/{RISK_LIMITS['MAX_OPEN_TRADES']}")
                return {
                    'position_size': 0.0,
                    'position_size_percentage': 0.0,
                    'risk_amount': 0.0,
                    'risk_percentage': 0.0,
                    'stop_loss': 0.0,
                    'take_profit': 0.0,
                    'status': 'REJECTED',
                    'reason': 'Max open trades limit reached'
                }
            
            # Check open trades per coin limit
            open_trades_for_coin = sum(1 for pos in self.current_positions.values() if pos['coin'] == coin)
            if open_trades_for_coin >= RISK_LIMITS['MAX_OPEN_TRADES_PER_COIN']:
                logger.warning(f"Max open trades per coin limit reached: {open_trades_for_coin}/{RISK_LIMITS['MAX_OPEN_TRADES_PER_COIN']}")
                return {
                    'position_size': 0.0,
                    'position_size_percentage': 0.0,
                    'risk_amount': 0.0,
                    'risk_percentage': 0.0,
                    'stop_loss': 0.0,
                    'take_profit': 0.0,
                    'status': 'REJECTED',
                    'reason': 'Max open trades per coin limit reached'
                }
            
            # Calculate position size using each strategy
            strategy_sizes = {}
            
            # Fixed risk strategy
            if 'FIXED_RISK' in self.strategy_weights and self.strategy_weights['FIXED_RISK'] > 0:
                fixed_risk_config = RISK_STRATEGIES['FIXED_RISK']
                risk_per_trade = fixed_risk_config['risk_per_trade']
                risk_amount = self.account_balance * risk_per_trade
                strategy_sizes['FIXED_RISK'] = risk_amount / STOP_LOSS['DEFAULT_STOP_LOSS'] if STOP_LOSS['DEFAULT_STOP_LOSS'] > 0 else 0.0
            
            # Kelly Criterion strategy
            if 'KELLY_CRITERION' in self.strategy_weights and self.strategy_weights['KELLY_CRITERION'] > 0:
                kelly_config = RISK_STRATEGIES['KELLY_CRITERION']
                # Calculate win rate from trade history
                win_rate = self._calculate_win_rate()
                # Calculate average win/loss ratio
                avg_win_loss_ratio = self._calculate_avg_win_loss_ratio()
                
                # Kelly formula: f* = (p * b - q) / b
                # where f* is the fraction of the bankroll to wager
                # p is the probability of winning
                # q is the probability of losing (1 - p)
                # b is the odds received on the wager (average win / average loss)
                
                if win_rate > 0 and avg_win_loss_ratio > 0:
                    kelly_percentage = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
                    # Apply Kelly fraction for more conservative sizing
                    kelly_percentage *= kelly_config['kelly_fraction']
                    # Apply limits
                    kelly_percentage = max(kelly_config['min_allocation'], min(kelly_config['max_allocation'], kelly_percentage))
                    strategy_sizes['KELLY_CRITERION'] = self.account_balance * kelly_percentage
                else:
                    # Fallback to default position size if not enough history
                    strategy_sizes['KELLY_CRITERION'] = self.account_balance * POSITION_SIZING['DEFAULT_POSITION_SIZE']
            
            # Volatility-based strategy
            if 'VOLATILITY_BASED' in self.strategy_weights and self.strategy_weights['VOLATILITY_BASED'] > 0:
                volatility_config = RISK_STRATEGIES['VOLATILITY_BASED']
                # In a real implementation, we would calculate ATR from price data
                # For this simulation, we'll use a placeholder ATR value
                atr = current_price * 0.02  # Placeholder: 2% of current price as ATR
                
                risk_amount = self.account_balance * volatility_config['max_risk_per_trade']
                position_size = risk_amount / (atr * volatility_config['atr_multiplier'])
                strategy_sizes['VOLATILITY_BASED'] = position_size
            
            # Dynamic position sizing strategy
            if 'DYNAMIC_POSITION_SIZING' in self.strategy_weights and self.strategy_weights['DYNAMIC_POSITION_SIZING'] > 0:
                dynamic_config = RISK_STRATEGIES['DYNAMIC_POSITION_SIZING']
                
                # Base risk
                base_risk = dynamic_config['base_risk']
                
                # Adjust based on confidence
                confidence_adjustment = confidence * dynamic_config['confidence_multiplier']
                
                # Adjust based on consecutive wins/losses
                consecutive_adjustment = 0.0
                if self.consecutive_wins > 0:
                    consecutive_adjustment = min(
                        self.consecutive_wins * dynamic_config['consecutive_wins_factor'],
                        dynamic_config['max_consecutive_adjustment']
                    )
                elif self.consecutive_losses > 0:
                    consecutive_adjustment = -min(
                        self.consecutive_losses * dynamic_config['consecutive_losses_factor'],
                        dynamic_config['max_consecutive_adjustment']
                    )
                
                # Calculate adjusted risk percentage
                adjusted_risk = base_risk + (base_risk * confidence_adjustment) + (base_risk * consecutive_adjustment)
                adjusted_risk = max(base_risk, min(dynamic_config['max_risk'], adjusted_risk))
                
                risk_amount = self.account_balance * adjusted_risk
                strategy_sizes['DYNAMIC_POSITION_SIZING'] = risk_amount / STOP_LOSS['DEFAULT_STOP_LOSS'] if STOP_LOSS['DEFAULT_STOP_LOSS'] > 0 else 0.0
            
            # Combine strategies using weighted average
            position_size = 0.0
            for strategy, size in strategy_sizes.items():
                position_size += size * self.strategy_weights.get(strategy, 0.0)
            
            # Apply market condition adjustments
            if market_condition in MARKET_CONDITION_ADJUSTMENTS:
                position_size *= MARKET_CONDITION_ADJUSTMENTS[market_condition]['position_size_multiplier']
            
            # Apply position size limits
            position_size_percentage = position_size / self.account_balance if self.account_balance > 0 else 0.0
            position_size_percentage = max(
                POSITION_SIZING['MIN_POSITION_SIZE'],
                min(POSITION_SIZING['MAX_POSITION_SIZE'], position_size_percentage)
            )
            position_size = self.account_balance * position_size_percentage
            
            # Calculate stop loss and take profit levels
            stop_loss_percentage = self._calculate_stop_loss(coin, signal_type, market_condition)
            take_profit_percentage = self._calculate_take_profit(coin, signal_type, market_condition, stop_loss_percentage)
            
            # Calculate risk amount
            risk_amount = position_size * stop_loss_percentage
            risk_percentage = risk_amount / self.account_balance if self.account_balance > 0 else 0.0
            
            # Calculate quantity
            quantity = position_size / current_price if current_price > 0 else 0.0
            
            # Save position sizing information
            self._save_position_sizing(
                coin, signal_type, signal_strength, confidence,
                position_size, position_size_percentage,
                risk_amount, risk_percentage,
                market_condition
            )
            
            logger.info(f"Calculated position size: {position_size:.2f} ({position_size_percentage:.2%})")
            
            return {
                'position_size': position_size,
                'position_size_percentage': position_size_percentage,
                'quantity': quantity,
                'risk_amount': risk_amount,
                'risk_percentage': risk_percentage,
                'stop_loss': stop_loss_percentage,
                'take_profit': take_profit_percentage,
                'status': 'ACCEPTED',
                'reason': 'Position size calculation successful'
            }
        
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return {
                'position_size': 0.0,
                'position_size_percentage': 0.0,
                'risk_amount': 0.0,
                'risk_percentage': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'status': 'REJECTED',
                'reason': f'Error: {str(e)}'
            }
    
    def _calculate_stop_loss(self, coin: str, signal_type: str, market_condition: str) -> float:
        """
        Calculate stop loss percentage based on risk parameters and market conditions.
        
        Args:
            coin: Cryptocurrency symbol
            signal_type: Type of trading signal (e.g., BUY, SELL)
            market_condition: Current market condition
            
        Returns:
            Stop loss percentage
        """
        try:
            # Start with default stop loss
            stop_loss = STOP_LOSS['DEFAULT_STOP_LOSS']
            
            # Adjust based on market condition
            if market_condition in MARKET_CONDITION_ADJUSTMENTS:
                stop_loss *= MARKET_CONDITION_ADJUSTMENTS[market_condition]['stop_loss_multiplier']
            
            # Apply limits
            stop_loss = max(STOP_LOSS['MIN_STOP_LOSS'], min(STOP_LOSS['MAX_STOP_LOSS'], stop_loss))
            
            return stop_loss
        
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            return STOP_LOSS['DEFAULT_STOP_LOSS']
    
    def _calculate_take_profit(self, coin: str, signal_type: str, market_condition: str, stop_loss: float) -> float:
        """
        Calculate take profit percentage based on risk parameters and market conditions.
        
        Args:
            coin: Cryptocurrency symbol
            signal_type: Type of trading signal (e.g., BUY, SELL)
            market_condition: Current market condition
            stop_loss: Stop loss percentage
            
        Returns:
            Take profit percentage
        """
        try:
            # Start with default take profit
            take_profit = TAKE_PROFIT['DEFAULT_TAKE_PROFIT']
            
            # Adjust based on risk-reward ratio
            if TAKE_PROFIT['RISK_REWARD_RATIO'] > 0:
                take_profit = stop_loss * TAKE_PROFIT['RISK_REWARD_RATIO']
            
            # Adjust based on market condition
            if market_condition in MARKET_CONDITION_ADJUSTMENTS:
                take_profit *= MARKET_CONDITION_ADJUSTMENTS[market_condition]['take_profit_multiplier']
            
            # Apply limits
            take_profit = max(TAKE_PROFIT['MIN_TAKE_PROFIT'], min(TAKE_PROFIT['MAX_TAKE_PROFIT'], take_profit))
            
            return take_profit
        
        except Exception as e:
            logger.error(f"Error calculating take profit: {str(e)}")
            return TAKE_PROFIT['DEFAULT_TAKE_PROFIT']
    
    def _calculate_win_rate(self, window: int = 100) -> float:
        """
        Calculate win rate from trade history.
        
        Args:
            window: Number of recent trades to consider
            
        Returns:
            Win rate as a fraction
        """
        try:
            # Get recent trades
            recent_trades = self.trade_history[-window:] if len(self.trade_history) > 0 else []
            
            if not recent_trades:
                return 0.5  # Default win rate if no history
            
            # Count winning trades
            winning_trades = sum(1 for trade in recent_trades if trade.get('profit_loss', 0) > 0)
            
            # Calculate win rate
            win_rate = winning_trades / len(recent_trades)
            
            return win_rate
        
        except Exception as e:
            logger.error(f"Error calculating win rate: {str(e)}")
            return 0.5
    
    def _calculate_avg_win_loss_ratio(self, window: int = 100) -> float:
        """
        Calculate average win/loss ratio from trade history.
        
        Args:
            window: Number of recent trades to consider
            
        Returns:
            Average win/loss ratio
        """
        try:
            # Get recent trades
            recent_trades = self.trade_history[-window:] if len(self.trade_history) > 0 else []
            
            if not recent_trades:
                return 1.0  # Default ratio if no history
            
            # Separate winning and losing trades
            winning_trades = [trade for trade in recent_trades if trade.get('profit_loss', 0) > 0]
            losing_trades = [trade for trade in recent_trades if trade.get('profit_loss', 0) < 0]
            
            if not winning_trades or not losing_trades:
                return 1.0  # Default ratio if no winners or losers
            
            # Calculate average win and loss
            avg_win = sum(trade.get('profit_loss', 0) for trade in winning_trades) / len(winning_trades)
            avg_loss = abs(sum(trade.get('profit_loss', 0) for trade in losing_trades) / len(losing_trades))
            
            # Calculate ratio
            ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
            
            return ratio
        
        except Exception as e:
            logger.error(f"Error calculating win/loss ratio: {str(e)}")
            return 1.0
    
    def _save_position_sizing(self, coin: str, signal_type: str, signal_strength: float,
                             confidence: float, position_size: float, position_size_percentage: float,
                             risk_amount: float, risk_percentage: float, market_condition: str):
        """
        Save position sizing information to database.
        
        Args:
            coin: Cryptocurrency symbol
            signal_type: Type of trading signal
            signal_strength: Strength of the trading signal
            confidence: Confidence in the trading signal
            position_size: Calculated position size
            position_size_percentage: Position size as percentage of account
            risk_amount: Amount at risk
            risk_percentage: Risk as percentage of account
            market_condition: Current market condition
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
            INSERT INTO {DATABASE['position_sizing_table']}
            (timestamp, coin, signal_type, signal_strength, confidence,
             position_size, position_size_percentage, risk_amount, risk_percentage,
             strategy_weights, market_condition, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                int(time.time() * 1000),
                coin,
                signal_type,
                signal_strength,
                confidence,
                position_size,
                position_size_percentage,
                risk_amount,
                risk_percentage,
                json.dumps(self.strategy_weights),
                market_condition,
                int(time.time() * 1000)
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved position sizing information for {coin}")
        
        except Exception as e:
            logger.error(f"Error saving position sizing: {str(e)}")
    
    def open_trade(self, coin: str, side: str, entry_price: float, quantity: float,
                  stop_loss: float, take_profit: float, signal_strength: float,
                  confidence: float, market_condition: str) -> Dict:
        """
        Open a new trade.
        
        Args:
            coin: Cryptocurrency symbol
            side: Trade side (BUY or SELL)
            entry_price: Entry price
            quantity: Quantity to trade
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
            signal_strength: Strength of the trading signal
            confidence: Confidence in the trading signal
            market_condition: Current market condition
            
        Returns:
            Dictionary with trade information
        """
        try:
            logger.info(f"Opening {side} trade for {coin} at {entry_price}")
            
            # Generate trade ID
            trade_id = f"{coin}_{side}_{int(time.time())}"
            
            # Calculate position size and risk
            position_size = entry_price * quantity
            position_size_percentage = position_size / self.account_balance if self.account_balance > 0 else 0.0
            risk_amount = position_size * stop_loss
            risk_percentage = risk_amount / self.account_balance if self.account_balance > 0 else 0.0
            
            # Create trade object
            trade = {
                'trade_id': trade_id,
                'coin': coin,
                'side': side,
                'entry_price': entry_price,
                'exit_price': None,
                'quantity': quantity,
                'position_size': position_size,
                'position_size_percentage': position_size_percentage,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_amount': risk_amount,
                'risk_percentage': risk_percentage,
                'profit_loss': None,
                'profit_loss_percentage': None,
                'trade_duration': None,
                'status': 'OPEN',
                'strategy': 'ENSEMBLE',
                'signal_strength': signal_strength,
                'confidence': confidence,
                'market_condition': market_condition,
                'entry_time': datetime.now(),
                'exit_time': None,
                'notes': ''
            }
            
            # Add to current positions
            self.current_positions[trade_id] = trade
            
            # Increment daily trades counter
            self.daily_trades += 1
            
            # Save trade to database
            self._save_trade(trade)
            
            logger.info(f"Opened trade {trade_id} for {coin}")
            
            return trade
        
        except Exception as e:
            logger.error(f"Error opening trade: {str(e)}")
            return {
                'status': 'ERROR',
                'message': f"Error opening trade: {str(e)}"
            }
    
    def close_trade(self, trade_id: str, exit_price: float, notes: str = '') -> Dict:
        """
        Close an existing trade.
        
        Args:
            trade_id: ID of the trade to close
            exit_price: Exit price
            notes: Additional notes about the trade
            
        Returns:
            Dictionary with trade information
        """
        try:
            logger.info(f"Closing trade {trade_id} at {exit_price}")
            
            # Check if trade exists
            if trade_id not in self.current_positions:
                logger.warning(f"Trade {trade_id} not found")
                return {
                    'status': 'ERROR',
                    'message': f"Trade {trade_id} not found"
                }
            
            # Get trade
            trade = self.current_positions[trade_id]
            
            # Update trade information
            trade['exit_price'] = exit_price
            trade['exit_time'] = datetime.now()
            trade['status'] = 'CLOSED'
            trade['notes'] = notes
            
            # Calculate profit/loss
            entry_price = trade['entry_price']
            quantity = trade['quantity']
            side = trade['side']
            
            if side == 'BUY':
                profit_loss = (exit_price - entry_price) * quantity
            else:  # SELL
                profit_loss = (entry_price - exit_price) * quantity
            
            trade['profit_loss'] = profit_loss
            trade['profit_loss_percentage'] = profit_loss / trade['position_size'] if trade['position_size'] > 0 else 0.0
            
            # Calculate trade duration
            trade['trade_duration'] = (trade['exit_time'] - trade['entry_time']).total_seconds()
            
            # Update account balance
            self.account_balance += profit_loss
            
            # Update daily P&L
            self.daily_pnl += profit_loss
            
            # Update peak balance
            if self.account_balance > self.peak_balance:
                self.peak_balance = self.account_balance
            
            # Update consecutive wins/losses
            if profit_loss > 0:
                self.consecutive_wins += 1
                self.consecutive_losses = 0
            elif profit_loss < 0:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                
                # Check if circuit breaker should be activated
                circuit_breaker = RISK_LIMITS['CIRCUIT_BREAKER']
                if (circuit_breaker['enabled'] and 
                    self.consecutive_losses >= circuit_breaker['consecutive_losses'] or
                    abs(profit_loss / trade['position_size']) >= circuit_breaker['loss_threshold']):
                    
                    self.circuit_breaker_active = True
                    self.circuit_breaker_until = datetime.now() + timedelta(minutes=circuit_breaker['cooldown_period'])
                    logger.warning(f"Circuit breaker activated until {self.circuit_breaker_until}")
            
            # Add to trade history
            self.trade_history.append(trade)
            
            # Remove from current positions
            del self.current_positions[trade_id]
            
            # Update trade in database
            self._update_trade(trade)
            
            # Update risk metrics
            self._update_risk_metrics()
            
            logger.info(f"Closed trade {trade_id} with P&L: {profit_loss:.2f} ({trade['profit_loss_percentage']:.2%})")
            
            return trade
        
        except Exception as e:
            logger.error(f"Error closing trade: {str(e)}")
            return {
                'status': 'ERROR',
                'message': f"Error closing trade: {str(e)}"
            }
    
    def _save_trade(self, trade: Dict):
        """
        Save trade information to database.
        
        Args:
            trade: Trade information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
            INSERT INTO {DATABASE['trade_log_table']}
            (timestamp, trade_id, coin, side, entry_price, exit_price,
             quantity, position_size, stop_loss, take_profit,
             risk_amount, risk_percentage, profit_loss, profit_loss_percentage,
             trade_duration, status, strategy, signal_strength, confidence,
             market_condition, created_at, closed_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                int(time.time() * 1000),
                trade['trade_id'],
                trade['coin'],
                trade['side'],
                trade['entry_price'],
                trade['exit_price'],
                trade['quantity'],
                trade['position_size'],
                trade['stop_loss'],
                trade['take_profit'],
                trade['risk_amount'],
                trade['risk_percentage'],
                trade['profit_loss'],
                trade['profit_loss_percentage'],
                trade['trade_duration'],
                trade['status'],
                trade['strategy'],
                trade['signal_strength'],
                trade['confidence'],
                trade['market_condition'],
                int(trade['entry_time'].timestamp() * 1000) if trade['entry_time'] else None,
                int(trade['exit_time'].timestamp() * 1000) if trade['exit_time'] else None,
                trade['notes']
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved trade {trade['trade_id']} to database")
        
        except Exception as e:
            logger.error(f"Error saving trade: {str(e)}")
    
    def _update_trade(self, trade: Dict):
        """
        Update trade information in database.
        
        Args:
            trade: Trade information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
            UPDATE {DATABASE['trade_log_table']}
            SET exit_price = ?, profit_loss = ?, profit_loss_percentage = ?,
                trade_duration = ?, status = ?, closed_at = ?, notes = ?
            WHERE trade_id = ?
            ''', (
                trade['exit_price'],
                trade['profit_loss'],
                trade['profit_loss_percentage'],
                trade['trade_duration'],
                trade['status'],
                int(trade['exit_time'].timestamp() * 1000) if trade['exit_time'] else None,
                trade['notes'],
                trade['trade_id']
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Updated trade {trade['trade_id']} in database")
        
        except Exception as e:
            logger.error(f"Error updating trade: {str(e)}")
    
    def _update_risk_metrics(self):
        """Update risk metrics and save to database."""
        try:
            # Calculate metrics
            open_positions = len(self.current_positions)
            drawdown = self.peak_balance - self.account_balance
            drawdown_percentage = drawdown / self.peak_balance if self.peak_balance > 0 else 0.0
            
            # Calculate win rate
            win_rate = self._calculate_win_rate()
            
            # Calculate profit factor
            profit_factor = self._calculate_profit_factor()
            
            # Calculate average win and loss
            avg_win, avg_loss = self._calculate_avg_win_loss()
            
            # Calculate Sharpe ratio (simplified)
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # Calculate Sortino ratio (simplified)
            sortino_ratio = self._calculate_sortino_ratio()
            
            # Calculate Calmar ratio (simplified)
            calmar_ratio = self._calculate_calmar_ratio()
            
            # Calculate expectancy
            expectancy = self._calculate_expectancy()
            
            # Save metrics to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
            INSERT INTO {DATABASE['risk_metrics_table']}
            (timestamp, account_balance, open_positions, daily_trades,
             daily_pnl, daily_pnl_percentage, drawdown, drawdown_percentage,
             win_rate, profit_factor, average_win, average_loss,
             sharpe_ratio, sortino_ratio, calmar_ratio, expectancy, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                int(time.time() * 1000),
                self.account_balance,
                open_positions,
                self.daily_trades,
                self.daily_pnl,
                self.daily_pnl / self.account_balance if self.account_balance > 0 else 0.0,
                drawdown,
                drawdown_percentage,
                win_rate,
                profit_factor,
                avg_win,
                avg_loss,
                sharpe_ratio,
                sortino_ratio,
                calmar_ratio,
                expectancy,
                int(time.time() * 1000)
            ))
            
            conn.commit()
            conn.close()
            logger.info("Updated risk metrics")
            
            # Check for risk alerts
            self._check_risk_alerts(
                drawdown_percentage, win_rate, profit_factor,
                self.consecutive_losses, self.daily_pnl / self.account_balance if self.account_balance > 0 else 0.0
            )
        
        except Exception as e:
            logger.error(f"Error updating risk metrics: {str(e)}")
    
    def _calculate_profit_factor(self, window: int = 100) -> float:
        """
        Calculate profit factor from trade history.
        
        Args:
            window: Number of recent trades to consider
            
        Returns:
            Profit factor
        """
        try:
            # Get recent trades
            recent_trades = self.trade_history[-window:] if len(self.trade_history) > 0 else []
            
            if not recent_trades:
                return 1.0  # Default profit factor if no history
            
            # Calculate gross profit and loss
            gross_profit = sum(trade.get('profit_loss', 0) for trade in recent_trades if trade.get('profit_loss', 0) > 0)
            gross_loss = abs(sum(trade.get('profit_loss', 0) for trade in recent_trades if trade.get('profit_loss', 0) < 0))
            
            # Calculate profit factor
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.0
            
            return profit_factor
        
        except Exception as e:
            logger.error(f"Error calculating profit factor: {str(e)}")
            return 1.0
    
    def _calculate_avg_win_loss(self, window: int = 100) -> Tuple[float, float]:
        """
        Calculate average win and loss from trade history.
        
        Args:
            window: Number of recent trades to consider
            
        Returns:
            Tuple of average win and average loss
        """
        try:
            # Get recent trades
            recent_trades = self.trade_history[-window:] if len(self.trade_history) > 0 else []
            
            if not recent_trades:
                return 0.0, 0.0  # Default values if no history
            
            # Separate winning and losing trades
            winning_trades = [trade for trade in recent_trades if trade.get('profit_loss', 0) > 0]
            losing_trades = [trade for trade in recent_trades if trade.get('profit_loss', 0) < 0]
            
            # Calculate average win
            avg_win = sum(trade.get('profit_loss', 0) for trade in winning_trades) / len(winning_trades) if winning_trades else 0.0
            
            # Calculate average loss
            avg_loss = abs(sum(trade.get('profit_loss', 0) for trade in losing_trades) / len(losing_trades)) if losing_trades else 0.0
            
            return avg_win, avg_loss
        
        except Exception as e:
            logger.error(f"Error calculating average win/loss: {str(e)}")
            return 0.0, 0.0
    
    def _calculate_sharpe_ratio(self, window: int = 100, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio from trade history.
        
        Args:
            window: Number of recent trades to consider
            risk_free_rate: Risk-free rate
            
        Returns:
            Sharpe ratio
        """
        try:
            # Get recent trades
            recent_trades = self.trade_history[-window:] if len(self.trade_history) > 0 else []
            
            if not recent_trades:
                return 0.0  # Default Sharpe ratio if no history
            
            # Calculate returns
            returns = [trade.get('profit_loss_percentage', 0) for trade in recent_trades]
            
            # Calculate mean and standard deviation
            mean_return = sum(returns) / len(returns)
            std_dev = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5
            
            # Calculate Sharpe ratio
            sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev > 0 else 0.0
            
            return sharpe_ratio
        
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
    
    def _calculate_sortino_ratio(self, window: int = 100, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio from trade history.
        
        Args:
            window: Number of recent trades to consider
            risk_free_rate: Risk-free rate
            
        Returns:
            Sortino ratio
        """
        try:
            # Get recent trades
            recent_trades = self.trade_history[-window:] if len(self.trade_history) > 0 else []
            
            if not recent_trades:
                return 0.0  # Default Sortino ratio if no history
            
            # Calculate returns
            returns = [trade.get('profit_loss_percentage', 0) for trade in recent_trades]
            
            # Calculate mean return
            mean_return = sum(returns) / len(returns)
            
            # Calculate downside deviation (only negative returns)
            downside_returns = [r - risk_free_rate for r in returns if r < risk_free_rate]
            downside_deviation = (sum(r ** 2 for r in downside_returns) / len(returns)) ** 0.5 if downside_returns else 0.0
            
            # Calculate Sortino ratio
            sortino_ratio = (mean_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0
            
            return sortino_ratio
        
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0
    
    def _calculate_calmar_ratio(self, years: float = 1.0) -> float:
        """
        Calculate Calmar ratio.
        
        Args:
            years: Number of years to consider
            
        Returns:
            Calmar ratio
        """
        try:
            # Calculate annualized return
            if not self.trade_history:
                return 0.0  # Default Calmar ratio if no history
            
            # Calculate total return
            initial_balance = 10000.0  # Default starting balance
            total_return = (self.account_balance - initial_balance) / initial_balance
            
            # Calculate annualized return
            annualized_return = (1 + total_return) ** (1 / years) - 1
            
            # Calculate maximum drawdown
            max_drawdown = self.max_drawdown / self.peak_balance if self.peak_balance > 0 else 0.0
            
            # Calculate Calmar ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
            
            return calmar_ratio
        
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {str(e)}")
            return 0.0
    
    def _calculate_expectancy(self, window: int = 100) -> float:
        """
        Calculate expectancy from trade history.
        
        Args:
            window: Number of recent trades to consider
            
        Returns:
            Expectancy
        """
        try:
            # Get recent trades
            recent_trades = self.trade_history[-window:] if len(self.trade_history) > 0 else []
            
            if not recent_trades:
                return 0.0  # Default expectancy if no history
            
            # Calculate win rate
            win_rate = self._calculate_win_rate(window)
            
            # Calculate average win and loss
            avg_win, avg_loss = self._calculate_avg_win_loss(window)
            
            # Calculate expectancy
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            return expectancy
        
        except Exception as e:
            logger.error(f"Error calculating expectancy: {str(e)}")
            return 0.0
    
    def _check_risk_alerts(self, drawdown: float, win_rate: float, profit_factor: float,
                          consecutive_losses: int, daily_loss_percentage: float):
        """
        Check for risk alerts.
        
        Args:
            drawdown: Current drawdown percentage
            win_rate: Current win rate
            profit_factor: Current profit factor
            consecutive_losses: Current consecutive losses
            daily_loss_percentage: Current daily loss percentage
        """
        try:
            alerts = RISK_MONITORING['ALERTS']
            
            # Check drawdown
            if drawdown >= alerts['drawdown_threshold']:
                logger.warning(f"ALERT: Drawdown {drawdown:.2%} exceeds threshold {alerts['drawdown_threshold']:.2%}")
            
            # Check consecutive losses
            if consecutive_losses >= alerts['consecutive_losses_threshold']:
                logger.warning(f"ALERT: Consecutive losses {consecutive_losses} exceeds threshold {alerts['consecutive_losses_threshold']}")
            
            # Check daily loss
            if daily_loss_percentage <= -alerts['daily_loss_threshold']:
                logger.warning(f"ALERT: Daily loss {daily_loss_percentage:.2%} exceeds threshold {alerts['daily_loss_threshold']:.2%}")
            
            # Check win rate
            if win_rate < alerts['win_rate_threshold']:
                logger.warning(f"ALERT: Win rate {win_rate:.2%} below threshold {alerts['win_rate_threshold']:.2%}")
            
            # Check profit factor
            if profit_factor < alerts['profit_factor_threshold']:
                logger.warning(f"ALERT: Profit factor {profit_factor:.2f} below threshold {alerts['profit_factor_threshold']:.2f}")
        
        except Exception as e:
            logger.error(f"Error checking risk alerts: {str(e)}")
    
    def reset_daily_metrics(self):
        """Reset daily metrics at the start of a new trading day."""
        try:
            logger.info("Resetting daily metrics")
            
            # Reset daily trades counter
            self.daily_trades = 0
            
            # Reset daily P&L
            self.daily_pnl = 0.0
            
            # Apply compounding if enabled
            if COMPOUNDING['ENABLED'] and COMPOUNDING['COMPOUNDING_FREQUENCY'] == 'DAILY':
                self._apply_compounding()
            
            logger.info("Daily metrics reset")
        
        except Exception as e:
            logger.error(f"Error resetting daily metrics: {str(e)}")
    
    def _apply_compounding(self):
        """Apply compounding to account balance."""
        try:
            logger.info("Applying compounding")
            
            # Calculate profit
            profit = self.account_balance - 10000.0  # Assuming 10000 is initial balance
            
            if profit <= 0:
                logger.info("No profit to compound")
                return
            
            # Calculate reinvestment amount
            reinvestment = profit * COMPOUNDING['REINVESTMENT_RATE']
            
            # Calculate withdrawal amount
            withdrawal = profit * COMPOUNDING['PROFIT_WITHDRAWAL']
            
            logger.info(f"Profit: {profit:.2f}, Reinvestment: {reinvestment:.2f}, Withdrawal: {withdrawal:.2f}")
            
            # Update account balance
            self.account_balance = 10000.0 + reinvestment
            
            logger.info(f"New account balance after compounding: {self.account_balance:.2f}")
        
        except Exception as e:
            logger.error(f"Error applying compounding: {str(e)}")
    
    def get_risk_metrics(self) -> Dict:
        """
        Get current risk metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        try:
            # Calculate metrics
            open_positions = len(self.current_positions)
            drawdown = self.peak_balance - self.account_balance
            drawdown_percentage = drawdown / self.peak_balance if self.peak_balance > 0 else 0.0
            
            # Calculate win rate
            win_rate = self._calculate_win_rate()
            
            # Calculate profit factor
            profit_factor = self._calculate_profit_factor()
            
            # Calculate average win and loss
            avg_win, avg_loss = self._calculate_avg_win_loss()
            
            # Calculate Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # Calculate Sortino ratio
            sortino_ratio = self._calculate_sortino_ratio()
            
            # Calculate Calmar ratio
            calmar_ratio = self._calculate_calmar_ratio()
            
            # Calculate expectancy
            expectancy = self._calculate_expectancy()
            
            return {
                'account_balance': self.account_balance,
                'open_positions': open_positions,
                'daily_trades': self.daily_trades,
                'daily_pnl': self.daily_pnl,
                'daily_pnl_percentage': self.daily_pnl / self.account_balance if self.account_balance > 0 else 0.0,
                'drawdown': drawdown,
                'drawdown_percentage': drawdown_percentage,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'average_win': avg_win,
                'average_loss': avg_loss,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'expectancy': expectancy,
                'consecutive_wins': self.consecutive_wins,
                'consecutive_losses': self.consecutive_losses
            }
        
        except Exception as e:
            logger.error(f"Error getting risk metrics: {str(e)}")
            return {}
    
    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """
        Get trade history.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of trades
        """
        try:
            # Get recent trades
            recent_trades = self.trade_history[-limit:] if len(self.trade_history) > 0 else []
            
            return recent_trades
        
        except Exception as e:
            logger.error(f"Error getting trade history: {str(e)}")
            return []
    
    def get_open_positions(self) -> Dict:
        """
        Get current open positions.
        
        Returns:
            Dictionary with open positions
        """
        try:
            return self.current_positions
        
        except Exception as e:
            logger.error(f"Error getting open positions: {str(e)}")
            return {}
    
    def set_account_balance(self, balance: float):
        """
        Set account balance.
        
        Args:
            balance: New account balance
        """
        try:
            logger.info(f"Setting account balance to {balance:.2f}")
            
            self.account_balance = balance
            
            # Update peak balance if necessary
            if balance > self.peak_balance:
                self.peak_balance = balance
            
            logger.info(f"Account balance set to {self.account_balance:.2f}")
        
        except Exception as e:
            logger.error(f"Error setting account balance: {str(e)}")
    
    def __str__(self) -> str:
        """String representation of the risk manager."""
        return f"RiskManager(balance={self.account_balance:.2f}, open_positions={len(self.current_positions)})"
    
    def __repr__(self) -> str:
        """Detailed representation of the risk manager."""
        return f"RiskManager(balance={self.account_balance:.2f}, open_positions={len(self.current_positions)}, daily_trades={self.daily_trades}, daily_pnl={self.daily_pnl:.2f})"
