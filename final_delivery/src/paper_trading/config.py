"""
Configuration settings for the paper trading module.
"""

# Paper trading parameters
PAPER_TRADING = {
    'INITIAL_CAPITAL': 10000.0,  # Initial paper trading capital
    'TRADING_PAIRS': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'],  # Trading pairs
    'TIMEFRAMES': ['5m', '15m', '1h', '4h'],  # Timeframes to monitor
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
    'ENABLE_SELF_LEARNING': True,  # Enable self-learning
    'LEARNING_RATE': 0.01,  # Learning rate for self-learning
    'DAILY_PROFIT_TARGET': 0.04,  # 4% daily profit target
    'SUCCESS_RATE_TARGET': 0.85,  # 85% success rate target
    'RISK_PER_TRADE': 0.01,  # 1% risk per trade
    'STOP_LOSS_PERCENTAGE': 0.02,  # 2% stop loss
    'TAKE_PROFIT_PERCENTAGE': 0.04,  # 4% take profit
    'TRAILING_STOP_ACTIVATION': 0.01,  # 1% trailing stop activation
    'TRAILING_STOP_DISTANCE': 0.005,  # 0.5% trailing stop distance
    'UPDATE_INTERVAL': 60,  # Update interval in seconds
    'TRADING_HOURS': {
        'ENABLED': False,  # Enable trading hours restriction
        'START_HOUR': 0,  # Start hour (UTC)
        'END_HOUR': 24,  # End hour (UTC)
        'DAYS': [0, 1, 2, 3, 4, 5, 6]  # Trading days (0=Monday, 6=Sunday)
    },
    'NOTIFICATIONS': {
        'ENABLE_EMAIL': False,  # Enable email notifications
        'ENABLE_TELEGRAM': False,  # Enable Telegram notifications
        'ENABLE_DISCORD': False,  # Enable Discord notifications
        'ENABLE_SLACK': False,  # Enable Slack notifications
        'ENABLE_WEBHOOK': True,  # Enable webhook notifications
        'WEBHOOK_URL': '',  # Webhook URL
        'NOTIFY_ON_TRADE': True,  # Notify on trade execution
        'NOTIFY_ON_PROFIT': True,  # Notify on profit
        'NOTIFY_ON_LOSS': True,  # Notify on loss
        'NOTIFY_ON_ERROR': True,  # Notify on error
        'NOTIFY_ON_SYSTEM_START': True,  # Notify on system start
        'NOTIFY_ON_SYSTEM_STOP': True,  # Notify on system stop
        'NOTIFY_ON_DAILY_SUMMARY': True,  # Notify on daily summary
    }
}

# API configuration
API_CONFIG = {
    'BINANCE_API_KEY': '',  # Binance API key (for future live trading)
    'BINANCE_API_SECRET': '',  # Binance API secret (for future live trading)
    'USE_TESTNET': True,  # Use Binance testnet
    'API_TIMEOUT': 30,  # API timeout in seconds
    'MAX_RETRIES': 3,  # Maximum number of API retries
    'RETRY_DELAY': 5,  # Delay between retries in seconds
}

# Database configuration
DATABASE = {
    'DB_PATH': '/home/ubuntu/crypto_trading_model/data/trading.db',
    'TABLES': {
        'paper_trades': 'paper_trades',
        'paper_positions': 'paper_positions',
        'paper_balance': 'paper_balance',
        'paper_performance': 'paper_performance',
        'trading_signals': 'trading_signals',
        'system_logs': 'system_logs',
        'optimization_history': 'optimization_history',
        'learning_history': 'learning_history'
    }
}

# Vercel deployment configuration
VERCEL_CONFIG = {
    'PROJECT_NAME': 'crypto-trading-model',
    'FRAMEWORK': 'nextjs',
    'BUILD_COMMAND': 'npm run build',
    'OUTPUT_DIRECTORY': '.next',
    'INSTALL_COMMAND': 'npm install',
    'NODE_VERSION': '18.x',
    'ENVIRONMENT_VARIABLES': {
        'NODE_ENV': 'production',
        'NEXT_PUBLIC_API_URL': '/api',
        'NEXT_PUBLIC_WS_URL': '',
        'DATABASE_URL': '',
    }
}

# Web interface configuration
WEB_INTERFACE = {
    'PORT': 3000,
    'HOST': '0.0.0.0',
    'ENABLE_AUTHENTICATION': True,
    'SESSION_SECRET': 'your-session-secret',
    'DEFAULT_USERNAME': 'admin',
    'DEFAULT_PASSWORD': 'admin',
    'JWT_SECRET': 'your-jwt-secret',
    'JWT_EXPIRATION': '24h',
    'ENABLE_API_KEY': True,
    'API_KEY': '',
    'CORS_ORIGINS': ['*'],
    'RATE_LIMIT': {
        'WINDOW_MS': 15 * 60 * 1000,  # 15 minutes
        'MAX_REQUESTS': 100  # 100 requests per window
    }
}

# Logging configuration
LOGGING = {
    'LOG_LEVEL': 'info',  # debug, info, warning, error, critical
    'LOG_FILE': '/home/ubuntu/crypto_trading_model/logs/paper_trading.log',
    'MAX_LOG_SIZE': 10 * 1024 * 1024,  # 10 MB
    'BACKUP_COUNT': 5,
    'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'CONSOLE_LOG': True,
    'FILE_LOG': True,
}

# Performance metrics to track
PERFORMANCE_METRICS = [
    'total_return',
    'daily_return',
    'win_rate',
    'profit_factor',
    'average_win',
    'average_loss',
    'max_drawdown',
    'sharpe_ratio',
    'sortino_ratio',
    'calmar_ratio',
    'expectancy',
    'average_holding_period',
    'trade_count',
    'winning_trades',
    'losing_trades',
    'consecutive_wins',
    'consecutive_losses',
    'largest_win',
    'largest_loss',
    'average_win_loss_ratio',
    'profit_per_day',
    'trades_per_day',
    'daily_sharpe',
    'monthly_sharpe',
    'annual_return',
    'volatility',
    'success_rate'
]
