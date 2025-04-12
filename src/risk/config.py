"""
Configuration settings for the risk management module.
"""

# Risk management strategies
RISK_STRATEGIES = {
    'FIXED_RISK': {
        'enabled': True,
        'weight': 0.2,
        'risk_per_trade': 0.01,  # 1% of capital per trade
        'max_risk_per_day': 0.05  # 5% of capital per day
    },
    'KELLY_CRITERION': {
        'enabled': True,
        'weight': 0.3,
        'max_allocation': 0.2,  # Maximum 20% of capital per trade
        'min_allocation': 0.01,  # Minimum 1% of capital per trade
        'kelly_fraction': 0.5,  # Half-Kelly for more conservative sizing
        'win_rate_window': 100  # Number of trades to consider for win rate
    },
    'VOLATILITY_BASED': {
        'enabled': True,
        'weight': 0.3,
        'atr_period': 14,
        'atr_multiplier': 2.0,
        'max_risk_per_trade': 0.02,  # 2% of capital per trade
        'position_size_atr_ratio': 0.1  # Position size as a ratio of ATR
    },
    'DYNAMIC_POSITION_SIZING': {
        'enabled': True,
        'weight': 0.2,
        'base_risk': 0.01,  # 1% base risk per trade
        'max_risk': 0.03,  # 3% maximum risk per trade
        'confidence_multiplier': 1.5,  # Multiplier for prediction confidence
        'consecutive_wins_factor': 0.1,  # Increase factor for consecutive wins
        'consecutive_losses_factor': 0.5,  # Decrease factor for consecutive losses
        'max_consecutive_adjustment': 0.5  # Maximum adjustment from consecutive trades
    }
}

# Position sizing parameters
POSITION_SIZING = {
    'MIN_POSITION_SIZE': 0.01,  # Minimum position size as fraction of capital
    'MAX_POSITION_SIZE': 0.2,  # Maximum position size as fraction of capital
    'DEFAULT_POSITION_SIZE': 0.05,  # Default position size as fraction of capital
    'SCALING_METHOD': 'LOGARITHMIC',  # LINEAR, LOGARITHMIC, EXPONENTIAL
    'CONFIDENCE_THRESHOLD': 0.65,  # Minimum confidence for taking a trade
    'SIGNAL_STRENGTH_THRESHOLD': 0.5,  # Minimum signal strength for taking a trade
    'INCREASE_THRESHOLD': 0.8,  # Threshold for increasing position size
    'DECREASE_THRESHOLD': 0.6  # Threshold for decreasing position size
}

# Risk limits
RISK_LIMITS = {
    'MAX_OPEN_TRADES': 5,  # Maximum number of concurrent open trades
    'MAX_OPEN_TRADES_PER_COIN': 2,  # Maximum number of concurrent open trades per coin
    'MAX_DAILY_TRADES': 20,  # Maximum number of trades per day
    'MAX_DAILY_LOSS': 0.05,  # Maximum daily loss as fraction of capital
    'MAX_DRAWDOWN': 0.15,  # Maximum drawdown as fraction of capital
    'TRAILING_STOP_ACTIVATION': 0.02,  # Profit level to activate trailing stop (2%)
    'TRAILING_STOP_DISTANCE': 0.01,  # Trailing stop distance (1%)
    'CIRCUIT_BREAKER': {
        'enabled': True,
        'consecutive_losses': 3,  # Number of consecutive losses to trigger circuit breaker
        'loss_threshold': 0.05,  # Loss threshold to trigger circuit breaker (5%)
        'cooldown_period': 60  # Cooldown period in minutes
    }
}

# Stop loss and take profit settings
STOP_LOSS = {
    'ENABLED': True,
    'DEFAULT_STOP_LOSS': 0.02,  # 2% default stop loss
    'MIN_STOP_LOSS': 0.01,  # 1% minimum stop loss
    'MAX_STOP_LOSS': 0.05,  # 5% maximum stop loss
    'ATR_MULTIPLIER': 2.0,  # ATR multiplier for dynamic stop loss
    'VOLATILITY_ADJUSTMENT': True,  # Adjust stop loss based on volatility
    'SUPPORT_RESISTANCE_ADJUSTMENT': True,  # Adjust stop loss based on support/resistance levels
    'TRAILING_STOP': {
        'enabled': True,
        'activation_threshold': 0.01,  # 1% profit to activate trailing stop
        'trail_distance': 0.005,  # 0.5% trailing distance
        'step_size': 0.001  # 0.1% step size for trailing stop adjustment
    }
}

TAKE_PROFIT = {
    'ENABLED': True,
    'DEFAULT_TAKE_PROFIT': 0.03,  # 3% default take profit
    'MIN_TAKE_PROFIT': 0.01,  # 1% minimum take profit
    'MAX_TAKE_PROFIT': 0.1,  # 10% maximum take profit
    'RISK_REWARD_RATIO': 1.5,  # Target risk-reward ratio
    'VOLATILITY_ADJUSTMENT': True,  # Adjust take profit based on volatility
    'RESISTANCE_ADJUSTMENT': True,  # Adjust take profit based on resistance levels
    'PARTIAL_TAKE_PROFIT': {
        'enabled': True,
        'levels': [
            {'threshold': 0.01, 'percentage': 0.3},  # At 1% profit, take 30% off
            {'threshold': 0.02, 'percentage': 0.3},  # At 2% profit, take another 30% off
            {'threshold': 0.03, 'percentage': 0.2}   # At 3% profit, take another 20% off
        ]
    }
}

# Risk adjustment based on market conditions
MARKET_CONDITION_ADJUSTMENTS = {
    'TRENDING_MARKET': {
        'position_size_multiplier': 1.2,
        'stop_loss_multiplier': 1.2,
        'take_profit_multiplier': 1.5
    },
    'RANGING_MARKET': {
        'position_size_multiplier': 0.8,
        'stop_loss_multiplier': 0.8,
        'take_profit_multiplier': 0.8
    },
    'VOLATILE_MARKET': {
        'position_size_multiplier': 0.6,
        'stop_loss_multiplier': 1.5,
        'take_profit_multiplier': 1.2
    },
    'BREAKOUT_MARKET': {
        'position_size_multiplier': 1.0,
        'stop_loss_multiplier': 1.0,
        'take_profit_multiplier': 1.3
    }
}

# Compounding settings
COMPOUNDING = {
    'ENABLED': True,
    'COMPOUNDING_FREQUENCY': 'DAILY',  # TRADE, DAILY, WEEKLY
    'TARGET_DAILY_RETURN': 0.04,  # 4% target daily return
    'REINVESTMENT_RATE': 0.8,  # Reinvest 80% of profits
    'PROFIT_WITHDRAWAL': 0.2,  # Withdraw 20% of profits
    'DYNAMIC_ADJUSTMENT': True  # Dynamically adjust based on performance
}

# Portfolio allocation
PORTFOLIO_ALLOCATION = {
    'DIVERSIFICATION': {
        'enabled': True,
        'max_allocation_per_coin': 0.3,  # Maximum 30% allocation to a single coin
        'correlation_threshold': 0.7,  # Correlation threshold for diversification
        'min_coins': 3,  # Minimum number of coins to hold
        'max_coins': 10  # Maximum number of coins to hold
    },
    'REBALANCING': {
        'enabled': True,
        'frequency': 'DAILY',  # HOURLY, DAILY, WEEKLY
        'threshold': 0.05,  # 5% deviation to trigger rebalancing
        'method': 'THRESHOLD'  # THRESHOLD, CALENDAR, HYBRID
    }
}

# Risk monitoring
RISK_MONITORING = {
    'METRICS': [
        'drawdown', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
        'win_rate', 'profit_factor', 'average_win', 'average_loss',
        'max_consecutive_wins', 'max_consecutive_losses', 'expectancy'
    ],
    'ALERTS': {
        'drawdown_threshold': 0.1,  # 10% drawdown alert
        'consecutive_losses_threshold': 3,
        'daily_loss_threshold': 0.03,  # 3% daily loss alert
        'win_rate_threshold': 0.5,  # 50% win rate alert
        'profit_factor_threshold': 1.2  # 1.2 profit factor alert
    },
    'REPORTING_FREQUENCY': 'HOURLY'  # TRADE, HOURLY, DAILY
}

# Database settings
DATABASE = {
    'risk_settings_table': 'risk_settings',
    'trade_log_table': 'trade_log',
    'position_sizing_table': 'position_sizing',
    'risk_metrics_table': 'risk_metrics',
    'portfolio_allocation_table': 'portfolio_allocation'
}
