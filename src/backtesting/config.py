"""
Configuration settings for the backtesting module.
"""

# Backtesting parameters
BACKTESTING = {
    'DEFAULT_INITIAL_CAPITAL': 10000.0,  # Default initial capital
    'DEFAULT_COMMISSION': 0.001,  # Default commission rate (0.1%)
    'DEFAULT_SLIPPAGE': 0.0005,  # Default slippage (0.05%)
    'DEFAULT_TIMEFRAME': '5m',  # Default timeframe for backtesting
    'DEFAULT_START_DATE': '2023-01-01',  # Default start date
    'DEFAULT_END_DATE': '2023-12-31',  # Default end date
    'DEFAULT_COINS': ['BTC', 'ETH', 'BNB', 'SOL', 'ADA'],  # Default coins to backtest
    'MAX_OPEN_TRADES': 5,  # Maximum number of concurrent open trades
    'ENABLE_POSITION_SIZING': True,  # Enable position sizing
    'ENABLE_RISK_MANAGEMENT': True,  # Enable risk management
    'ENABLE_TRAILING_STOP': True,  # Enable trailing stop
    'ENABLE_PARTIAL_TAKE_PROFIT': True,  # Enable partial take profit
    'ENABLE_COMPOUNDING': True,  # Enable compounding
    'ENABLE_PORTFOLIO_REBALANCING': True,  # Enable portfolio rebalancing
    'ENABLE_MARKET_CONDITION_DETECTION': True,  # Enable market condition detection
    'ENABLE_SENTIMENT_ANALYSIS': True,  # Enable sentiment analysis
    'ENABLE_ENSEMBLE_PREDICTIONS': True,  # Enable ensemble predictions
    'ENABLE_DYNAMIC_OPTIMIZATION': True,  # Enable dynamic optimization
    'ENABLE_WALK_FORWARD_ANALYSIS': True,  # Enable walk-forward analysis
    'ENABLE_MONTE_CARLO_SIMULATION': True,  # Enable Monte Carlo simulation
    'ENABLE_STRESS_TESTING': True,  # Enable stress testing
    'ENABLE_BENCHMARK_COMPARISON': True,  # Enable benchmark comparison
    'BENCHMARK_STRATEGY': 'HODL',  # Default benchmark strategy
    'BENCHMARK_COINS': ['BTC'],  # Default benchmark coins
    'BENCHMARK_WEIGHTS': [1.0],  # Default benchmark weights
    'OPTIMIZATION_METRIC': 'SHARPE_RATIO',  # Default optimization metric
    'OPTIMIZATION_PARAMETERS': [
        'risk_per_trade',
        'stop_loss',
        'take_profit',
        'trailing_stop_activation',
        'trailing_stop_distance'
    ],
    'OPTIMIZATION_RANGES': {
        'risk_per_trade': [0.005, 0.01, 0.015, 0.02],
        'stop_loss': [0.01, 0.015, 0.02, 0.025, 0.03],
        'take_profit': [0.02, 0.03, 0.04, 0.05, 0.06],
        'trailing_stop_activation': [0.01, 0.015, 0.02],
        'trailing_stop_distance': [0.005, 0.01, 0.015]
    }
}

# Performance metrics to calculate
PERFORMANCE_METRICS = [
    'total_return',
    'annualized_return',
    'max_drawdown',
    'max_drawdown_duration',
    'sharpe_ratio',
    'sortino_ratio',
    'calmar_ratio',
    'win_rate',
    'profit_factor',
    'average_win',
    'average_loss',
    'average_win_loss_ratio',
    'expectancy',
    'average_holding_period',
    'max_consecutive_wins',
    'max_consecutive_losses',
    'volatility',
    'beta',
    'alpha',
    'r_squared',
    'value_at_risk',
    'conditional_value_at_risk',
    'omega_ratio',
    'kurtosis',
    'skewness'
]

# Market conditions for testing
MARKET_CONDITIONS = {
    'TRENDING_UP': {
        'description': 'Strong upward trend',
        'parameters': {
            'trend_strength': 0.8,
            'volatility': 0.2,
            'volume': 1.2
        }
    },
    'TRENDING_DOWN': {
        'description': 'Strong downward trend',
        'parameters': {
            'trend_strength': -0.8,
            'volatility': 0.3,
            'volume': 1.5
        }
    },
    'RANGING': {
        'description': 'Sideways market with low volatility',
        'parameters': {
            'trend_strength': 0.1,
            'volatility': 0.1,
            'volume': 0.8
        }
    },
    'VOLATILE': {
        'description': 'High volatility market',
        'parameters': {
            'trend_strength': 0.2,
            'volatility': 0.5,
            'volume': 1.8
        }
    },
    'BREAKOUT': {
        'description': 'Market breakout with high volume',
        'parameters': {
            'trend_strength': 0.6,
            'volatility': 0.4,
            'volume': 2.0
        }
    },
    'CRASH': {
        'description': 'Market crash with extreme volatility',
        'parameters': {
            'trend_strength': -0.9,
            'volatility': 0.8,
            'volume': 3.0
        }
    }
}

# Stress test scenarios
STRESS_TEST_SCENARIOS = {
    'EXTREME_VOLATILITY': {
        'description': 'Extreme market volatility',
        'volatility_multiplier': 3.0,
        'duration_days': 7
    },
    'FLASH_CRASH': {
        'description': 'Sudden market crash',
        'price_drop_percentage': 0.2,
        'duration_minutes': 30
    },
    'LIQUIDITY_CRISIS': {
        'description': 'Market liquidity crisis',
        'slippage_multiplier': 5.0,
        'spread_multiplier': 3.0,
        'duration_days': 3
    },
    'CORRELATION_BREAKDOWN': {
        'description': 'Breakdown of typical market correlations',
        'correlation_shift': 0.8,
        'duration_days': 5
    },
    'EXCHANGE_OUTAGE': {
        'description': 'Exchange technical outage',
        'outage_duration_minutes': 120,
        'recovery_slippage_multiplier': 2.0
    }
}

# Walk-forward analysis settings
WALK_FORWARD_ANALYSIS = {
    'ENABLED': True,
    'TRAINING_WINDOW': 90,  # Training window in days
    'TESTING_WINDOW': 30,  # Testing window in days
    'STEP_SIZE': 30,  # Step size in days
    'MIN_TRAINING_PERIODS': 3,  # Minimum number of training periods
    'ANCHORED': False,  # Whether to use anchored walk-forward analysis
    'OPTIMIZATION_METRIC': 'SHARPE_RATIO',  # Metric to optimize in training
    'PARAMETERS_TO_OPTIMIZE': [
        'risk_per_trade',
        'stop_loss',
        'take_profit'
    ]
}

# Monte Carlo simulation settings
MONTE_CARLO_SIMULATION = {
    'ENABLED': True,
    'NUM_SIMULATIONS': 1000,  # Number of simulations to run
    'CONFIDENCE_LEVEL': 0.95,  # Confidence level for results
    'METHODS': [
        'RANDOM_RETURNS',  # Randomly sample from historical returns
        'RANDOM_TRADES',  # Randomly sample from historical trades
        'BLOCK_BOOTSTRAP'  # Block bootstrap sampling
    ],
    'METRICS_TO_ANALYZE': [
        'final_equity',
        'max_drawdown',
        'sharpe_ratio',
        'win_rate'
    ]
}

# Reporting settings
REPORTING = {
    'GENERATE_HTML_REPORT': True,
    'GENERATE_PDF_REPORT': False,
    'GENERATE_CSV_RESULTS': True,
    'GENERATE_EQUITY_CURVE_CHART': True,
    'GENERATE_DRAWDOWN_CHART': True,
    'GENERATE_MONTHLY_RETURNS_HEATMAP': True,
    'GENERATE_TRADE_DISTRIBUTION_CHART': True,
    'GENERATE_MONTE_CARLO_CHART': True,
    'GENERATE_OPTIMIZATION_HEATMAP': True,
    'INCLUDE_TRADE_LIST': True,
    'MAX_TRADES_IN_REPORT': 100,
    'REPORT_DIRECTORY': '/home/ubuntu/crypto_trading_model/reports'
}

# Database settings
DATABASE = {
    'backtest_results_table': 'backtest_results',
    'backtest_trades_table': 'backtest_trades',
    'backtest_metrics_table': 'backtest_metrics',
    'optimization_results_table': 'optimization_results',
    'monte_carlo_results_table': 'monte_carlo_results',
    'walk_forward_results_table': 'walk_forward_results'
}
