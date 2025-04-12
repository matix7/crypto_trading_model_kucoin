"""
Configuration settings for the dashboard module.
"""

# Dashboard configuration
DASHBOARD = {
    'PORT': 3000,
    'HOST': '0.0.0.0',
    'DEBUG': False,
    'THEME': 'dark',  # 'light' or 'dark'
    'TITLE': 'Crypto Trading Analytics Dashboard',
    'REFRESH_INTERVAL': 60000,  # Refresh interval in milliseconds (1 minute)
    'MAX_TRADES_DISPLAY': 100,  # Maximum number of trades to display
    'MAX_POSITIONS_DISPLAY': 20,  # Maximum number of positions to display
    'CHART_TIMEFRAMES': ['1h', '4h', '1d', '1w'],  # Timeframes for charts
    'DEFAULT_TIMEFRAME': '1h',  # Default timeframe for charts
    'ENABLE_LIVE_UPDATES': True,  # Enable live updates
    'ENABLE_NOTIFICATIONS': True,  # Enable notifications
    'ENABLE_DARK_MODE': True,  # Enable dark mode
    'ENABLE_MOBILE_VIEW': True,  # Enable mobile view
    'ENABLE_EXPORT': True,  # Enable data export
    'ENABLE_PRINT': True,  # Enable printing
    'ENABLE_FULLSCREEN': True,  # Enable fullscreen mode
    'ENABLE_ZOOM': True,  # Enable chart zooming
    'ENABLE_TOOLTIPS': True,  # Enable tooltips
    'ENABLE_LEGENDS': True,  # Enable chart legends
    'ENABLE_GRID': True,  # Enable chart grid
    'ENABLE_CROSSHAIR': True,  # Enable chart crosshair
    'ENABLE_ANNOTATIONS': True,  # Enable chart annotations
    'CHART_COLORS': {
        'primary': '#1976D2',
        'secondary': '#FF9800',
        'success': '#4CAF50',
        'danger': '#F44336',
        'warning': '#FFC107',
        'info': '#2196F3',
        'light': '#F5F5F5',
        'dark': '#212121',
        'background': '#121212',
        'surface': '#1E1E1E',
        'text': '#FFFFFF',
        'border': '#333333',
        'profit': '#00C853',
        'loss': '#FF5252',
        'neutral': '#9E9E9E'
    }
}

# API configuration
API = {
    'BASE_URL': 'http://localhost:3001/api',  # Base URL for API
    'ENDPOINTS': {
        'status': '/status',
        'start': '/start',
        'stop': '/stop',
        'positions': '/positions',
        'trades': '/trades',
        'metrics': '/metrics',
        'config': '/config'
    },
    'HEADERS': {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer your-api-key'
    },
    'TIMEOUT': 10000,  # Timeout in milliseconds
    'RETRY_COUNT': 3,  # Number of retries
    'RETRY_DELAY': 1000,  # Delay between retries in milliseconds
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

# Chart configuration
CHARTS = {
    'EQUITY_CURVE': {
        'title': 'Equity Curve',
        'x_axis_label': 'Date',
        'y_axis_label': 'Equity ($)',
        'type': 'line',
        'height': 400,
        'show_grid': True,
        'show_tooltip': True,
        'show_legend': True,
        'show_markers': False,
        'curve_type': 'monotone',
        'stroke_width': 2,
        'fill_opacity': 0.1,
        'data_key': 'equity'
    },
    'DAILY_RETURNS': {
        'title': 'Daily Returns',
        'x_axis_label': 'Date',
        'y_axis_label': 'Return (%)',
        'type': 'bar',
        'height': 300,
        'show_grid': True,
        'show_tooltip': True,
        'show_legend': True,
        'show_markers': False,
        'data_key': 'daily_return'
    },
    'DRAWDOWN': {
        'title': 'Drawdown',
        'x_axis_label': 'Date',
        'y_axis_label': 'Drawdown (%)',
        'type': 'area',
        'height': 300,
        'show_grid': True,
        'show_tooltip': True,
        'show_legend': True,
        'show_markers': False,
        'curve_type': 'monotone',
        'stroke_width': 2,
        'fill_opacity': 0.2,
        'data_key': 'drawdown'
    },
    'WIN_RATE': {
        'title': 'Win Rate',
        'type': 'pie',
        'height': 300,
        'show_tooltip': True,
        'show_legend': True,
        'inner_radius': 60,
        'outer_radius': 80,
        'data_key': 'win_rate'
    },
    'PROFIT_DISTRIBUTION': {
        'title': 'Profit Distribution',
        'x_axis_label': 'Profit/Loss (%)',
        'y_axis_label': 'Frequency',
        'type': 'histogram',
        'height': 300,
        'show_grid': True,
        'show_tooltip': True,
        'show_legend': True,
        'bin_count': 20,
        'data_key': 'profit_loss_percentage'
    },
    'PERFORMANCE_METRICS': {
        'title': 'Performance Metrics',
        'type': 'radar',
        'height': 400,
        'show_grid': True,
        'show_tooltip': True,
        'show_legend': True,
        'data_key': 'metrics'
    },
    'TRADE_HISTORY': {
        'title': 'Trade History',
        'type': 'scatter',
        'height': 300,
        'show_grid': True,
        'show_tooltip': True,
        'show_legend': True,
        'show_markers': True,
        'data_key': 'trades'
    },
    'POSITION_SIZE': {
        'title': 'Position Size',
        'x_axis_label': 'Date',
        'y_axis_label': 'Position Size ($)',
        'type': 'bar',
        'height': 300,
        'show_grid': True,
        'show_tooltip': True,
        'show_legend': True,
        'data_key': 'position_size'
    },
    'TRADING_PAIRS': {
        'title': 'Trading Pairs',
        'type': 'pie',
        'height': 300,
        'show_tooltip': True,
        'show_legend': True,
        'inner_radius': 0,
        'outer_radius': 80,
        'data_key': 'trading_pairs'
    },
    'TRADE_DURATION': {
        'title': 'Trade Duration',
        'x_axis_label': 'Duration (hours)',
        'y_axis_label': 'Frequency',
        'type': 'histogram',
        'height': 300,
        'show_grid': True,
        'show_tooltip': True,
        'show_legend': True,
        'bin_count': 10,
        'data_key': 'trade_duration'
    }
}

# Dashboard layout configuration
LAYOUT = {
    'HEADER': {
        'show_logo': True,
        'show_title': True,
        'show_status': True,
        'show_balance': True,
        'show_equity': True,
        'show_profit': True,
        'show_win_rate': True,
        'show_theme_toggle': True,
        'show_refresh_button': True,
        'show_settings_button': True,
        'show_help_button': True,
        'show_user_menu': True
    },
    'SIDEBAR': {
        'show_sidebar': True,
        'default_open': True,
        'show_navigation': True,
        'show_trading_pairs': True,
        'show_timeframes': True,
        'show_indicators': True,
        'show_strategies': True,
        'show_risk_settings': True,
        'show_system_status': True
    },
    'MAIN': {
        'layout_type': 'grid',  # 'grid' or 'tabs'
        'grid_columns': 12,  # Number of columns in grid layout
        'grid_row_height': 30,  # Row height in pixels
        'grid_margin': [10, 10],  # Margin [x, y] in pixels
        'grid_container_padding': [10, 10],  # Container padding [x, y] in pixels
        'grid_draggable': True,  # Allow dragging widgets
        'grid_resizable': True,  # Allow resizing widgets
        'grid_layout': [
            {'i': 'status', 'x': 0, 'y': 0, 'w': 12, 'h': 2},
            {'i': 'equity_curve', 'x': 0, 'y': 2, 'w': 12, 'h': 8},
            {'i': 'daily_returns', 'x': 0, 'y': 10, 'w': 6, 'h': 6},
            {'i': 'drawdown', 'x': 6, 'y': 10, 'w': 6, 'h': 6},
            {'i': 'performance_metrics', 'x': 0, 'y': 16, 'w': 6, 'h': 8},
            {'i': 'win_rate', 'x': 6, 'y': 16, 'w': 3, 'h': 8},
            {'i': 'trading_pairs', 'x': 9, 'y': 16, 'w': 3, 'h': 8},
            {'i': 'open_positions', 'x': 0, 'y': 24, 'w': 12, 'h': 6},
            {'i': 'trade_history', 'x': 0, 'y': 30, 'w': 12, 'h': 8},
            {'i': 'profit_distribution', 'x': 0, 'y': 38, 'w': 6, 'h': 6},
            {'i': 'trade_duration', 'x': 6, 'y': 38, 'w': 6, 'h': 6}
        ]
    },
    'FOOTER': {
        'show_footer': True,
        'show_version': True,
        'show_copyright': True,
        'show_links': True,
        'show_status': True,
        'show_last_update': True
    }
}

# Vercel deployment configuration
VERCEL_CONFIG = {
    'PROJECT_NAME': 'crypto-trading-dashboard',
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
