"""
Configuration for the paper trading module.
"""

import os
from typing import Dict, List, Any

# Paper Trading Configuration
PAPER_TRADING = {
    'INITIAL_CAPITAL': 10000,  # Initial capital in USDT
    'UPDATE_INTERVAL': 60,  # Update interval in seconds
    'TRADING_PAIRS': [
        'BTC-USDT',
        'ETH-USDT',
        'SOL-USDT',
        'BNB-USDT',
        'ADA-USDT'
    ],
    'TIMEFRAMES': [
        '5m',
        '15m',
        '1h',
        '4h'
    ],
    'KLINES_LIMIT': 100,  # Number of klines to fetch
    'MAX_OPEN_POSITIONS': 5,  # Maximum number of open positions
    'MIN_POSITION_SIZE': 10,  # Minimum position size in USDT
    'MAX_POSITION_SIZE': 1000,  # Maximum position size in USDT
    'RISK_PER_TRADE': 0.02,  # Risk per trade as a percentage of balance
    'STOP_LOSS_PERCENTAGE': 0.02,  # Stop loss percentage
    'TAKE_PROFIT_PERCENTAGE': 0.04,  # Take profit percentage
    'TRAILING_STOP_PERCENTAGE': 0.01,  # Trailing stop percentage
    'TA_WEIGHT': 0.6,  # Weight for technical analysis signals
    'ML_WEIGHT': 0.4,  # Weight for machine learning signals
    'DB_PATH': os.getenv('DB_PATH', 'data/paper_trading.db'),  # Database path
}

# Exchange Configuration
EXCHANGE = {
    'NAME': 'KuCoin',
    'API_KEY': os.getenv('KUCOIN_API_KEY', ''),
    'API_SECRET': os.getenv('KUCOIN_API_SECRET', ''),
    'API_PASSPHRASE': os.getenv('KUCOIN_API_PASSPHRASE', ''),
    'USE_SANDBOX': os.getenv('USE_SANDBOX', 'true').lower() == 'true',
    'BASE_URL': 'https://api-sandbox.kucoin.com' if os.getenv('USE_SANDBOX', 'true').lower() == 'true' else 'https://api.kucoin.com',
}
