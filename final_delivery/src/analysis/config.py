"""
Configuration settings for the technical analysis module.
"""

# Signal types
SIGNAL_TYPES = {
    'ENTRY': {
        'LONG': 1,   # Long entry signal
        'SHORT': -1  # Short entry signal
    },
    'EXIT': {
        'LONG': 2,   # Long exit signal
        'SHORT': -2  # Short exit signal
    },
    'NONE': 0        # No signal
}

# Signal strength levels
SIGNAL_STRENGTH = {
    'WEAK': 0.25,    # Weak signal (25% confidence)
    'MODERATE': 0.5, # Moderate signal (50% confidence)
    'STRONG': 0.75,  # Strong signal (75% confidence)
    'VERY_STRONG': 1.0  # Very strong signal (100% confidence)
}

# Strategy types
STRATEGY_TYPES = [
    'TREND_FOLLOWING',     # Strategies that follow established trends
    'MEAN_REVERSION',      # Strategies that bet on price returning to mean
    'BREAKOUT',            # Strategies that identify breakouts from ranges
    'MOMENTUM',            # Strategies based on momentum indicators
    'VOLATILITY',          # Strategies that trade based on volatility
    'PATTERN_RECOGNITION',  # Strategies based on chart patterns
    'INDICATOR_COMBINATION' # Strategies that combine multiple indicators
]

# Strategy weights (initial)
STRATEGY_WEIGHTS = {
    'TREND_FOLLOWING': 0.2,
    'MEAN_REVERSION': 0.15,
    'BREAKOUT': 0.15,
    'MOMENTUM': 0.2,
    'VOLATILITY': 0.1,
    'PATTERN_RECOGNITION': 0.1,
    'INDICATOR_COMBINATION': 0.1
}

# Strategy performance metrics
STRATEGY_METRICS = [
    'win_rate',           # Percentage of winning trades
    'profit_factor',      # Gross profits divided by gross losses
    'sharpe_ratio',       # Risk-adjusted return
    'max_drawdown',       # Maximum peak to trough decline
    'avg_return',         # Average return per trade
    'return_volatility',  # Standard deviation of returns
    'num_trades',         # Number of trades generated
    'avg_trade_duration'  # Average duration of trades
]

# Trend following strategy settings
TREND_FOLLOWING = {
    'ema_short_period': 9,
    'ema_medium_period': 21,
    'ema_long_period': 55,
    'macd_fast_period': 12,
    'macd_slow_period': 26,
    'macd_signal_period': 9,
    'adx_period': 14,
    'adx_threshold': 25,  # ADX above this indicates strong trend
    'min_trend_duration': 5  # Minimum number of periods for valid trend
}

# Mean reversion strategy settings
MEAN_REVERSION = {
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'bb_period': 20,
    'bb_std_dev': 2,
    'stoch_k_period': 14,
    'stoch_d_period': 3,
    'stoch_overbought': 80,
    'stoch_oversold': 20,
    'mean_period': 50  # Period for calculating mean
}

# Breakout strategy settings
BREAKOUT = {
    'atr_period': 14,
    'atr_multiplier': 2.5,
    'volume_surge_threshold': 2.0,  # Volume > 2x average
    'consolidation_periods': 20,
    'breakout_threshold': 0.03,  # 3% price movement
    'false_breakout_filter': True
}

# Momentum strategy settings
MOMENTUM = {
    'rsi_period': 14,
    'rsi_threshold_high': 60,
    'rsi_threshold_low': 40,
    'macd_fast_period': 12,
    'macd_slow_period': 26,
    'macd_signal_period': 9,
    'stoch_rsi_period': 14,
    'stoch_rsi_k_period': 3,
    'stoch_rsi_d_period': 3,
    'obv_min_slope': 0.1  # Minimum slope for OBV trend
}

# Volatility strategy settings
VOLATILITY = {
    'atr_period': 14,
    'atr_multiplier': 2.0,
    'bb_period': 20,
    'bb_std_dev': 2,
    'bb_squeeze_threshold': 0.1,  # BB width < 10% of 20-day average
    'volatility_explosion_threshold': 2.0  # Volatility > 2x average
}

# Pattern recognition strategy settings
PATTERN_RECOGNITION = {
    'min_pattern_quality': 80,  # Minimum pattern quality score (0-100)
    'confirmation_periods': 2,  # Periods to confirm pattern
    'patterns': [
        'ENGULFING',
        'HAMMER',
        'SHOOTING_STAR',
        'DOJI',
        'MORNING_STAR',
        'EVENING_STAR',
        'HARAMI',
        'MARUBOZU'
    ]
}

# Indicator combination strategy settings
INDICATOR_COMBINATION = {
    'min_agreement': 3,  # Minimum number of indicators that must agree
    'indicators': [
        'EMA_CROSS',
        'MACD',
        'RSI',
        'STOCH',
        'BB',
        'ADX',
        'OBV'
    ],
    'weights': {
        'EMA_CROSS': 0.2,
        'MACD': 0.2,
        'RSI': 0.15,
        'STOCH': 0.15,
        'BB': 0.1,
        'ADX': 0.1,
        'OBV': 0.1
    }
}

# Timeframe weights for multi-timeframe analysis
TIMEFRAME_WEIGHTS = {
    '1m': 0.05,  # 1-minute timeframe (lowest weight)
    '5m': 0.1,   # 5-minute timeframe
    '15m': 0.15, # 15-minute timeframe
    '1h': 0.3,   # 1-hour timeframe
    '4h': 0.25,  # 4-hour timeframe
    '1d': 0.15   # 1-day timeframe
}

# Signal combination settings
SIGNAL_COMBINATION = {
    'strategy_agreement_threshold': 0.6,  # 60% of strategies must agree
    'timeframe_agreement_threshold': 0.7,  # 70% of timeframes must agree
    'min_signal_strength': 0.5,  # Minimum combined signal strength
    'use_weighted_average': True  # Use weighted average of signals
}

# Market regime detection settings
MARKET_REGIME = {
    'volatility_lookback': 20,
    'trend_lookback': 50,
    'regime_types': {
        'TRENDING_UP': {'weight_trend': 0.7, 'weight_mean_reversion': 0.3},
        'TRENDING_DOWN': {'weight_trend': 0.7, 'weight_mean_reversion': 0.3},
        'RANGING': {'weight_trend': 0.3, 'weight_mean_reversion': 0.7},
        'VOLATILE': {'weight_trend': 0.4, 'weight_mean_reversion': 0.6},
        'BREAKOUT': {'weight_trend': 0.6, 'weight_mean_reversion': 0.4}
    }
}

# Performance tracking settings
PERFORMANCE_TRACKING = {
    'evaluation_period': 100,  # Number of periods for performance evaluation
    'min_trades_for_evaluation': 20,  # Minimum trades for reliable evaluation
    'weight_update_factor': 0.1,  # Factor for updating strategy weights
    'track_individual_signals': True  # Track performance of individual signals
}

# Database settings
DATABASE = {
    'signals_table': 'trading_signals',
    'strategies_table': 'strategy_performance',
    'market_regimes_table': 'market_regimes'
}
