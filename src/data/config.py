"""
Configuration for the data module.
"""

import os
from typing import Dict, List, Any

# Exchange Configuration
EXCHANGE = {
    'NAME': 'KuCoin',
    'API_KEY': os.getenv('KUCOIN_API_KEY', ''),
    'API_SECRET': os.getenv('KUCOIN_API_SECRET', ''),
    'API_PASSPHRASE': os.getenv('KUCOIN_API_PASSPHRASE', ''),
    'USE_SANDBOX': os.getenv('USE_SANDBOX', 'true').lower() == 'true',
    'BASE_URL': 'https://api-sandbox.kucoin.com' if os.getenv('USE_SANDBOX', 'true').lower() == 'true' else 'https://api.kucoin.com',
}

# Trading Pairs Configuration
TRADING_PAIRS = [
    'BTC-USDT',
    'ETH-USDT',
    'SOL-USDT',
    'BNB-USDT',
    'ADA-USDT'
]

# Timeframes Configuration
TIMEFRAMES = [
    '5m',
    '15m',
    '1h',
    '4h'
]

# Database Configuration
DATABASE = {
    'DB_PATH': os.getenv('DB_PATH', 'data/trading.db'),
    'TABLES': {
        'historical_data': 'historical_data',
        'technical_indicators': 'technical_indicators',
        'sentiment_data': 'sentiment_data',
        'predictions': 'predictions',
        'trades': 'trades',
        'performance': 'performance'
    }
}

# Data Collection Configuration
DATA_COLLECTION = {
    'HISTORICAL_DATA_DAYS': 60,
    'UPDATE_INTERVAL': 60,  # seconds
    'BATCH_SIZE': 1000,
    'MAX_RETRIES': 3,
    'RETRY_DELAY': 5,  # seconds
    'TIMEOUT': 30,  # seconds
    'RATE_LIMIT': {
        'MAX_REQUESTS_PER_MINUTE': 50,
        'MAX_REQUESTS_PER_HOUR': 1000
    }
}

# Technical Indicators Configuration
TECHNICAL_INDICATORS = {
    'SMA': [20, 50, 200],
    'EMA': [9, 21, 55, 200],
    'RSI': [14],
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
    'BOLLINGER_BANDS': {'period': 20, 'std_dev': 2},
    'ATR': [14],
    'STOCHASTIC': {'k': 14, 'd': 3, 'smooth_k': 3},
    'ADX': [14],
    'OBV': True,
    'VWAP': True,
    'ICHIMOKU': {
        'tenkan': 9,
        'kijun': 26,
        'senkou_span_b': 52,
        'displacement': 26
    }
}

# Feature Engineering Configuration
FEATURE_ENGINEERING = {
    'PRICE_FEATURES': [
        'returns',
        'log_returns',
        'rolling_mean',
        'rolling_std',
        'rolling_min',
        'rolling_max'
    ],
    'VOLUME_FEATURES': [
        'volume_change',
        'volume_ma',
        'relative_volume'
    ],
    'PATTERN_FEATURES': [
        'support_resistance',
        'trend_strength',
        'volatility_ratio',
        'price_channels'
    ],
    'TEMPORAL_FEATURES': [
        'hour_of_day',
        'day_of_week',
        'week_of_year',
        'month',
        'is_weekend'
    ],
    'WINDOW_SIZES': [5, 10, 20, 50],
    'NORMALIZATION': 'min_max',  # 'min_max', 'standard', 'robust'
    'MISSING_VALUES': 'forward_fill'  # 'forward_fill', 'backward_fill', 'mean', 'median', 'zero'
}

# Data Pipeline Configuration
DATA_PIPELINE = {
    'CACHE_SIZE': 1000,
    'PREFETCH_TIMEFRAMES': True,
    'BATCH_PROCESSING': True,
    'PARALLEL_PROCESSING': True,
    'NUM_WORKERS': 4,
    'LOGGING_LEVEL': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    'SAVE_INTERMEDIATE': False
}
