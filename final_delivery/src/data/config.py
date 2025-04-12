"""
Configuration settings for the data collection and preprocessing module.
"""

# Timeframes for data collection
TIMEFRAMES = {
    '1m': {'interval': '1m', 'limit': 1000, 'keep_periods': 24 * 60},  # 1 day of 1-minute data
    '5m': {'interval': '5m', 'limit': 1000, 'keep_periods': 24 * 12 * 7},  # 1 week of 5-minute data
    '15m': {'interval': '15m', 'limit': 1000, 'keep_periods': 24 * 4 * 30},  # 1 month of 15-minute data
    '1h': {'interval': '1h', 'limit': 1000, 'keep_periods': 24 * 90},  # 3 months of 1-hour data
    '4h': {'interval': '4h', 'limit': 1000, 'keep_periods': 6 * 180},  # 6 months of 4-hour data
    '1d': {'interval': '1d', 'limit': 1000, 'keep_periods': 365},  # 1 year of daily data
}

# Trading pairs to monitor
TRADING_PAIRS = [
    'BTCUSDT',  # Bitcoin
    'ETHUSDT',  # Ethereum
    'BNBUSDT',  # Binance Coin
    'SOLUSDT',  # Solana
    'ADAUSDT',  # Cardano
    'XRPUSDT',  # Ripple
    'DOGEUSDT', # Dogecoin
    'DOTUSDT',  # Polkadot
    'AVAXUSDT', # Avalanche
    'MATICUSDT', # Polygon
]

# Technical indicators configuration
TECHNICAL_INDICATORS = {
    'SMA': [20, 50, 200],  # Simple Moving Average periods
    'EMA': [9, 21, 55, 200],  # Exponential Moving Average periods
    'RSI': [14],  # Relative Strength Index periods
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},  # Moving Average Convergence Divergence
    'BB': {'period': 20, 'std_dev': 2},  # Bollinger Bands
    'ATR': [14],  # Average True Range
    'STOCH': {'k_period': 14, 'd_period': 3},  # Stochastic Oscillator
    'STOCHRSI': {'period': 14, 'k_period': 3, 'd_period': 3},  # Stochastic RSI
    'ADX': [14],  # Average Directional Index
    'OBV': [],  # On-Balance Volume (no parameters)
    'VWAP': [],  # Volume Weighted Average Price (calculated per session)
}

# Data normalization settings
NORMALIZATION = {
    'price_data': 'min_max',  # Min-max scaling for price data
    'volume_data': 'log',  # Log transformation for volume data
    'indicators': {
        'RSI': 'min_max',  # Min-max scaling for RSI (already 0-100)
        'MACD': 'z_score',  # Z-score normalization for MACD
        'BB': 'custom',  # Custom normalization for Bollinger Bands
        'STOCH': 'min_max',  # Min-max scaling for Stochastic (already 0-100)
        'ADX': 'min_max',  # Min-max scaling for ADX
        'OBV': 'z_score',  # Z-score normalization for OBV
        'ATR': 'z_score',  # Z-score normalization for ATR
    },
    'lookback_periods': 100,  # Number of periods to use for normalization
}

# Feature engineering settings
FEATURE_ENGINEERING = {
    'price_patterns': True,  # Enable price pattern detection
    'indicator_crossovers': True,  # Enable indicator crossover detection
    'volatility_features': True,  # Enable volatility-based features
    'temporal_features': True,  # Enable time-based features
    'volume_features': True,  # Enable volume-based features
}

# Database settings
DATABASE = {
    'type': 'sqlite',  # Use SQLite for simplicity in development
    'path': '/home/ubuntu/crypto_trading_model/data/market_data.db',
    'tables': {
        'ohlcv': 'ohlcv_data',
        'indicators': 'technical_indicators',
        'features': 'engineered_features',
    }
}

# API settings
API = {
    'binance': {
        'base_url': 'https://api.binance.com',
        'api_key': '',  # To be filled at runtime or from environment
        'api_secret': '',  # To be filled at runtime or from environment
        'rate_limit': {
            'max_requests': 1200,  # Maximum requests per minute
            'weight_per_request': 1,  # Default weight per request
        }
    }
}

# Logging settings
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': '/home/ubuntu/crypto_trading_model/logs/data_collection.log',
}
