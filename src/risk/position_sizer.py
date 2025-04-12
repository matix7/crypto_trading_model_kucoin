"""
Position sizing module for calculating optimal position sizes.
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

from .config import (
    POSITION_SIZING, RISK_LIMITS, STOP_LOSS, TAKE_PROFIT,
    MARKET_CONDITION_ADJUSTMENTS, PORTFOLIO_ALLOCATION, DATABASE
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/risk_management.log',
    filemode='a'
)
logger = logging.getLogger('position_sizer')

class PositionSizer:
    """
    Position sizer class for calculating optimal position sizes.
    """
    
    def __init__(self, risk_manager, db_path: str = None):
        """
        Initialize the PositionSizer.
        
        Args:
            risk_manager: Risk manager instance
            db_path: Path to the SQLite database file
        """
        self.risk_manager = risk_manager
        self.db_path = db_path or '/home/ubuntu/crypto_trading_model/data/market_data.db'
        
        # Position sizing parameters
        self.min_position_size = POSITION_SIZING['MIN_POSITION_SIZE']
        self.max_position_size = POSITION_SIZING['MAX_POSITION_SIZE']
        self.default_position_size = POSITION_SIZING['DEFAULT_POSITION_SIZE']
        self.scaling_method = POSITION_SIZING['SCALING_METHOD']
        self.confidence_threshold = POSITION_SIZING['CONFIDENCE_THRESHOLD']
        self.signal_strength_threshold = POSITION_SIZING['SIGNAL_STRENGTH_THRESHOLD']
        self.increase_threshold = POSITION_SIZING['INCREASE_THRESHOLD']
        self.decrease_threshold = POSITION_SIZING['DECREASE_THRESHOLD']
        
        logger.info("PositionSizer initialized")
    
    def calculate_position_size(self, account_balance: float, signal_strength: float, 
                               confidence: float, market_condition: str, 
                               volatility: float, win_rate: float) -> Dict:
        """
        Calculate optimal position size based on various factors.
        
        Args:
            account_balance: Current account balance
            signal_strength: Strength of the trading signal
            confidence: Confidence in the trading signal
            market_condition: Current market condition
            volatility: Current market volatility
            win_rate: Current win rate
            
        Returns:
            Dictionary with position sizing information
        """
        try:
            logger.info(f"Calculating position size with signal strength {signal_strength} and confidence {confidence}")
            
            # Check if signal meets minimum thresholds
            if confidence < self.confidence_threshold:
                logger.info(f"Signal confidence {confidence} below threshold {self.confidence_threshold}")
                return {
                    'position_size': 0.0,
                    'position_size_percentage': 0.0,
                    'status': 'REJECTED',
                    'reason': 'Confidence below threshold'
                }
            
            if abs(signal_strength) < self.signal_strength_threshold:
                logger.info(f"Signal strength {signal_strength} below threshold {self.signal_strength_threshold}")
                return {
                    'position_size': 0.0,
                    'position_size_percentage': 0.0,
                    'status': 'REJECTED',
                    'reason': 'Signal strength below threshold'
                }
            
            # Start with default position size
            position_size_percentage = self.default_position_size
            
            # Adjust based on signal strength and confidence
            if self.scaling_method == 'LINEAR':
                # Linear scaling based on signal strength and confidence
                signal_factor = abs(signal_strength) / 1.0  # Normalize to 0-1 range
                confidence_factor = confidence / 1.0  # Normalize to 0-1 range
                
                # Combined factor
                combined_factor = (signal_factor + confidence_factor) / 2.0
                
                # Scale position size
                position_size_percentage = self.min_position_size + combined_factor * (self.max_position_size - self.min_position_size)
            
            elif self.scaling_method == 'LOGARITHMIC':
                # Logarithmic scaling for more conservative sizing
                signal_factor = abs(signal_strength) / 1.0  # Normalize to 0-1 range
                confidence_factor = confidence / 1.0  # Normalize to 0-1 range
                
                # Combined factor with logarithmic scaling
                combined_factor = (signal_factor + confidence_factor) / 2.0
                log_factor = np.log10(9 * combined_factor + 1) / np.log10(10)  # Log scaling from 0-1
                
                # Scale position size
                position_size_percentage = self.min_position_size + log_factor * (self.max_position_size - self.min_position_size)
            
            elif self.scaling_method == 'EXPONENTIAL':
                # Exponential scaling for more aggressive sizing
                signal_factor = abs(signal_strength) / 1.0  # Normalize to 0-1 range
                confidence_factor = confidence / 1.0  # Normalize to 0-1 range
                
                # Combined factor with exponential scaling
                combined_factor = (signal_factor + confidence_factor) / 2.0
                exp_factor = (np.exp(combined_factor) - 1) / (np.exp(1) - 1)  # Exp scaling from 0-1
                
                # Scale position size
                position_size_percentage = self.min_position_size + exp_factor * (self.max_position_size - self.min_position_size)
            
            else:
                # Default to linear scaling
                signal_factor = abs(signal_strength) / 1.0  # Normalize to 0-1 range
                confidence_factor = confidence / 1.0  # Normalize to 0-1 range
                
                # Combined factor
                combined_factor = (signal_factor + confidence_factor) / 2.0
                
                # Scale position size
                position_size_percentage = self.min_position_size + combined_factor * (self.max_position_size - self.min_position_size)
            
            # Adjust based on market condition
            if market_condition in MARKET_CONDITION_ADJUSTMENTS:
                position_size_percentage *= MARKET_CONDITION_ADJUSTMENTS[market_condition]['position_size_multiplier']
            
            # Adjust based on volatility
            volatility_factor = 1.0 - min(1.0, volatility / 0.1)  # Reduce position size as volatility increases
            position_size_percentage *= volatility_factor
            
            # Adjust based on win rate
            win_rate_factor = win_rate / 0.5  # Scale based on win rate (0.5 is neutral)
            position_size_percentage *= win_rate_factor
            
            # Apply limits
            position_size_percentage = max(self.min_position_size, min(self.max_position_size, position_size_percentage))
            
            # Calculate position size
            position_size = account_balance * position_size_percentage
            
            logger.info(f"Calculated position size: {position_size:.2f} ({position_size_percentage:.2%})")
            
            return {
                'position_size': position_size,
                'position_size_percentage': position_size_percentage,
                'status': 'ACCEPTED',
                'reason': 'Position size calculation successful'
            }
        
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return {
                'position_size': 0.0,
                'position_size_percentage': 0.0,
                'status': 'REJECTED',
                'reason': f'Error: {str(e)}'
            }
    
    def calculate_optimal_quantity(self, position_size: float, current_price: float, 
                                  min_quantity: float = 0.0001) -> float:
        """
        Calculate optimal quantity based on position size and current price.
        
        Args:
            position_size: Position size in base currency
            current_price: Current price of the asset
            min_quantity: Minimum quantity allowed
            
        Returns:
            Optimal quantity
        """
        try:
            if current_price <= 0:
                logger.warning(f"Invalid current price: {current_price}")
                return 0.0
            
            # Calculate raw quantity
            quantity = position_size / current_price
            
            # Round to appropriate precision
            if quantity >= 1.0:
                # Round to 2 decimal places for quantities >= 1
                quantity = round(quantity, 2)
            elif quantity >= 0.1:
                # Round to 4 decimal places for quantities >= 0.1
                quantity = round(quantity, 4)
            else:
                # Round to 8 decimal places for small quantities
                quantity = round(quantity, 8)
            
            # Ensure minimum quantity
            quantity = max(min_quantity, quantity)
            
            logger.info(f"Calculated quantity: {quantity} at price {current_price}")
            
            return quantity
        
        except Exception as e:
            logger.error(f"Error calculating quantity: {str(e)}")
            return 0.0
    
    def calculate_stop_loss_price(self, entry_price: float, side: str, stop_loss_percentage: float) -> float:
        """
        Calculate stop loss price based on entry price and stop loss percentage.
        
        Args:
            entry_price: Entry price
            side: Trade side (BUY or SELL)
            stop_loss_percentage: Stop loss percentage
            
        Returns:
            Stop loss price
        """
        try:
            if entry_price <= 0:
                logger.warning(f"Invalid entry price: {entry_price}")
                return 0.0
            
            if side == 'BUY':
                # For long positions, stop loss is below entry price
                stop_loss_price = entry_price * (1 - stop_loss_percentage)
            else:  # SELL
                # For short positions, stop loss is above entry price
                stop_loss_price = entry_price * (1 + stop_loss_percentage)
            
            logger.info(f"Calculated stop loss price: {stop_loss_price} for {side} at {entry_price}")
            
            return stop_loss_price
        
        except Exception as e:
            logger.error(f"Error calculating stop loss price: {str(e)}")
            return 0.0
    
    def calculate_take_profit_price(self, entry_price: float, side: str, take_profit_percentage: float) -> float:
        """
        Calculate take profit price based on entry price and take profit percentage.
        
        Args:
            entry_price: Entry price
            side: Trade side (BUY or SELL)
            take_profit_percentage: Take profit percentage
            
        Returns:
            Take profit price
        """
        try:
            if entry_price <= 0:
                logger.warning(f"Invalid entry price: {entry_price}")
                return 0.0
            
            if side == 'BUY':
                # For long positions, take profit is above entry price
                take_profit_price = entry_price * (1 + take_profit_percentage)
            else:  # SELL
                # For short positions, take profit is below entry price
                take_profit_price = entry_price * (1 - take_profit_percentage)
            
            logger.info(f"Calculated take profit price: {take_profit_price} for {side} at {entry_price}")
            
            return take_profit_price
        
        except Exception as e:
            logger.error(f"Error calculating take profit price: {str(e)}")
            return 0.0
    
    def calculate_partial_take_profits(self, entry_price: float, side: str, quantity: float) -> List[Dict]:
        """
        Calculate partial take profit levels based on configuration.
        
        Args:
            entry_price: Entry price
            side: Trade side (BUY or SELL)
            quantity: Trade quantity
            
        Returns:
            List of partial take profit levels
        """
        try:
            if not TAKE_PROFIT['PARTIAL_TAKE_PROFIT']['enabled']:
                return []
            
            partial_take_profits = []
            remaining_quantity = quantity
            
            for level in TAKE_PROFIT['PARTIAL_TAKE_PROFIT']['levels']:
                threshold = level['threshold']
                percentage = level['percentage']
                
                # Calculate take profit price
                if side == 'BUY':
                    price = entry_price * (1 + threshold)
                else:  # SELL
                    price = entry_price * (1 - threshold)
                
                # Calculate quantity to take profit
                take_profit_quantity = quantity * percentage
                remaining_quantity -= take_profit_quantity
                
                partial_take_profits.append({
                    'price': price,
                    'quantity': take_profit_quantity,
                    'percentage': percentage,
                    'threshold': threshold
                })
            
            logger.info(f"Calculated {len(partial_take_profits)} partial take profit levels")
            
            return partial_take_profits
        
        except Exception as e:
            logger.error(f"Error calculating partial take profits: {str(e)}")
            return []
    
    def calculate_trailing_stop(self, entry_price: float, current_price: float, side: str) -> float:
        """
        Calculate trailing stop price based on current price and configuration.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            side: Trade side (BUY or SELL)
            
        Returns:
            Trailing stop price
        """
        try:
            if not STOP_LOSS['TRAILING_STOP']['enabled']:
                return 0.0
            
            # Get trailing stop parameters
            activation_threshold = STOP_LOSS['TRAILING_STOP']['activation_threshold']
            trail_distance = STOP_LOSS['TRAILING_STOP']['trail_distance']
            
            # Check if trailing stop should be activated
            if side == 'BUY':
                # For long positions
                price_change_percentage = (current_price - entry_price) / entry_price
                
                if price_change_percentage >= activation_threshold:
                    # Trailing stop is activated
                    trailing_stop_price = current_price * (1 - trail_distance)
                    logger.info(f"Calculated trailing stop price: {trailing_stop_price} for long position")
                    return trailing_stop_price
                else:
                    # Trailing stop not activated yet
                    return 0.0
            else:  # SELL
                # For short positions
                price_change_percentage = (entry_price - current_price) / entry_price
                
                if price_change_percentage >= activation_threshold:
                    # Trailing stop is activated
                    trailing_stop_price = current_price * (1 + trail_distance)
                    logger.info(f"Calculated trailing stop price: {trailing_stop_price} for short position")
                    return trailing_stop_price
                else:
                    # Trailing stop not activated yet
                    return 0.0
        
        except Exception as e:
            logger.error(f"Error calculating trailing stop: {str(e)}")
            return 0.0
    
    def adjust_position_for_portfolio_allocation(self, coin: str, position_size: float, 
                                               account_balance: float) -> float:
        """
        Adjust position size based on portfolio allocation limits.
        
        Args:
            coin: Cryptocurrency symbol
            position_size: Calculated position size
            account_balance: Current account balance
            
        Returns:
            Adjusted position size
        """
        try:
            if not PORTFOLIO_ALLOCATION['DIVERSIFICATION']['enabled']:
                return position_size
            
            # Get maximum allocation per coin
            max_allocation_per_coin = PORTFOLIO_ALLOCATION['DIVERSIFICATION']['max_allocation_per_coin']
            
            # Get current allocation for this coin
            current_allocation = self._get_current_allocation(coin, account_balance)
            
            # Calculate maximum allowed position size
            max_position_size = account_balance * max_allocation_per_coin - current_allocation
            
            # Adjust position size if necessary
            adjusted_position_size = min(position_size, max_position_size)
            
            if adjusted_position_size < position_size:
                logger.info(f"Adjusted position size from {position_size:.2f} to {adjusted_position_size:.2f} due to portfolio allocation limits")
            
            return adjusted_position_size
        
        except Exception as e:
            logger.error(f"Error adjusting position for portfolio allocation: {str(e)}")
            return position_size
    
    def _get_current_allocation(self, coin: str, account_balance: float) -> float:
        """
        Get current allocation for a specific coin.
        
        Args:
            coin: Cryptocurrency symbol
            account_balance: Current account balance
            
        Returns:
            Current allocation in base currency
        """
        try:
            # Get current positions from risk manager
            current_positions = self.risk_manager.get_open_positions()
            
            # Calculate current allocation for this coin
            current_allocation = sum(
                position['position_size']
                for position in current_positions.values()
                if position['coin'] == coin
            )
            
            logger.info(f"Current allocation for {coin}: {current_allocation:.2f} ({current_allocation / account_balance:.2%})")
            
            return current_allocation
        
        except Exception as e:
            logger.error(f"Error getting current allocation: {str(e)}")
            return 0.0
    
    def __str__(self) -> str:
        """String representation of the position sizer."""
        return f"PositionSizer(min={self.min_position_size:.2%}, max={self.max_position_size:.2%}, default={self.default_position_size:.2%})"
