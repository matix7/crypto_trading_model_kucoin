"""
Configuration settings for the prediction module.
"""

# Model types
MODEL_TYPES = {
    'LSTM': {
        'enabled': True,
        'weight': 0.35,
        'lookback_periods': 60,  # Number of time periods to look back
        'layers': [128, 64, 32],  # Hidden layer sizes
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'batch_size': 32,
        'epochs': 100,
        'patience': 10,  # Early stopping patience
        'validation_split': 0.2
    },
    'GRU': {
        'enabled': True,
        'weight': 0.25,
        'lookback_periods': 60,
        'layers': [128, 64],
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'batch_size': 32,
        'epochs': 100,
        'patience': 10,
        'validation_split': 0.2
    },
    'TRANSFORMER': {
        'enabled': True,
        'weight': 0.20,
        'lookback_periods': 60,
        'num_heads': 4,
        'ff_dim': 128,
        'num_transformer_blocks': 2,
        'mlp_units': [64, 32],
        'mlp_dropout': 0.2,
        'dropout': 0.2,
        'batch_size': 32,
        'epochs': 100,
        'patience': 10,
        'validation_split': 0.2
    },
    'XGB': {
        'enabled': True,
        'weight': 0.20,
        'lookback_periods': 60,
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'early_stopping_rounds': 20
    }
}

# Ensemble configuration
ENSEMBLE = {
    'METHOD': 'WEIGHTED_AVERAGE',  # WEIGHTED_AVERAGE, STACKING, VOTING
    'STACKING_MODEL': 'XGB',  # Model to use for stacking
    'DYNAMIC_WEIGHTS': True,  # Dynamically adjust weights based on performance
    'WEIGHT_UPDATE_FREQUENCY': 24,  # Hours between weight updates
    'PERFORMANCE_WINDOW': 168,  # Hours of performance history to consider
    'MIN_WEIGHT': 0.05,  # Minimum weight for any model
    'CONFIDENCE_THRESHOLD': 0.65  # Minimum confidence for a prediction to be used
}

# Feature importance tracking
FEATURE_IMPORTANCE = {
    'TRACK': True,
    'UPDATE_FREQUENCY': 24,  # Hours between updates
    'TOP_FEATURES': 20,  # Number of top features to track
    'FEATURE_SELECTION_METHOD': 'SHAP'  # SHAP, PERMUTATION, GAIN
}

# Prediction targets
PREDICTION_TARGETS = {
    'PRICE_DIRECTION': {
        'enabled': True,
        'weight': 0.4,
        'thresholds': {
            'UP': 0.001,  # 0.1% price increase
            'DOWN': -0.001  # 0.1% price decrease
        }
    },
    'PRICE_MOVEMENT': {
        'enabled': True,
        'weight': 0.3,
        'prediction_horizons': [5, 15, 30, 60]  # Minutes
    },
    'VOLATILITY': {
        'enabled': True,
        'weight': 0.15,
        'prediction_horizons': [15, 30, 60]  # Minutes
    },
    'SUPPORT_RESISTANCE': {
        'enabled': True,
        'weight': 0.15,
        'levels_to_predict': 3  # Number of support/resistance levels to predict
    }
}

# Feature groups
FEATURE_GROUPS = {
    'PRICE': {
        'enabled': True,
        'weight': 0.25,
        'features': [
            'close', 'open', 'high', 'low', 'volume',
            'close_pct_change', 'volume_pct_change',
            'high_low_ratio', 'close_open_ratio'
        ]
    },
    'TECHNICAL': {
        'enabled': True,
        'weight': 0.35,
        'features': [
            # Trend indicators
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
            'macd', 'macd_signal', 'macd_hist',
            'adx_14', 'dmi_plus_14', 'dmi_minus_14',
            
            # Momentum indicators
            'rsi_14', 'stoch_k_14', 'stoch_d_14', 'cci_20', 'mfi_14',
            'roc_10', 'williams_r_14',
            
            # Volatility indicators
            'bbands_upper_20_2', 'bbands_middle_20_2', 'bbands_lower_20_2',
            'bbands_width_20_2', 'atr_14', 'atr_percent_14',
            
            # Volume indicators
            'obv', 'vwap', 'cmf_20', 'mfi_14',
            
            # Pattern recognition
            'engulfing', 'hammer', 'shooting_star', 'doji',
            'three_white_soldiers', 'three_black_crows',
            
            # Custom indicators
            'price_distance_from_sma_200', 'sma_5_sma_20_cross',
            'rsi_divergence', 'volume_price_trend'
        ]
    },
    'SENTIMENT': {
        'enabled': True,
        'weight': 0.20,
        'features': [
            'social_media_sentiment', 'news_sentiment', 'on_chain_sentiment',
            'overall_sentiment', 'sentiment_signal_type', 'sentiment_signal_strength',
            'sentiment_confidence', 'sentiment_change_24h'
        ]
    },
    'MARKET': {
        'enabled': True,
        'weight': 0.10,
        'features': [
            'market_cap', 'market_dominance', 'market_volatility_index',
            'correlation_with_btc', 'correlation_with_eth',
            'correlation_with_sp500', 'correlation_with_gold',
            'market_regime'
        ]
    },
    'TIME': {
        'enabled': True,
        'weight': 0.05,
        'features': [
            'hour_of_day', 'day_of_week', 'day_of_month',
            'month', 'quarter', 'is_weekend',
            'time_since_last_peak', 'time_since_last_bottom'
        ]
    },
    'ONCHAIN': {
        'enabled': True,
        'weight': 0.05,
        'features': [
            'transaction_volume', 'active_addresses',
            'new_addresses', 'fees', 'hash_rate',
            'transaction_value', 'exchange_inflow', 'exchange_outflow'
        ]
    }
}

# Hyperparameter optimization
HYPERPARAMETER_OPTIMIZATION = {
    'ENABLED': True,
    'METHOD': 'BAYESIAN',  # BAYESIAN, RANDOM, GRID
    'MAX_TRIALS': 50,
    'CROSS_VALIDATION_FOLDS': 5,
    'OPTIMIZATION_METRIC': 'val_loss',
    'OPTIMIZATION_DIRECTION': 'min',
    'OPTIMIZATION_FREQUENCY': 168  # Hours between optimization runs (weekly)
}

# Self-learning configuration
SELF_LEARNING = {
    'ENABLED': True,
    'RETRAINING_FREQUENCY': 24,  # Hours between retraining
    'MINIMUM_SAMPLES_FOR_RETRAINING': 1000,
    'PERFORMANCE_THRESHOLD_FOR_RETRAINING': 0.05,  # 5% performance degradation triggers retraining
    'ADAPTIVE_FEATURE_SELECTION': True,
    'TRANSFER_LEARNING': True,
    'REINFORCEMENT_LEARNING': {
        'enabled': True,
        'reward_function': 'PROFIT_AND_LOSS',  # PROFIT_AND_LOSS, SHARPE_RATIO, CUSTOM
        'exploration_rate': 0.1,
        'discount_factor': 0.95
    }
}

# Prediction output
PREDICTION_OUTPUT = {
    'CONFIDENCE_SCORE': True,
    'PREDICTION_INTERVAL': True,
    'FEATURE_ATTRIBUTION': True,
    'ALTERNATIVE_SCENARIOS': True,
    'EXPLANATION': True
}

# Database settings
DATABASE = {
    'predictions_table': 'ml_predictions',
    'model_performance_table': 'ml_model_performance',
    'feature_importance_table': 'ml_feature_importance',
    'model_weights_table': 'ml_model_weights',
    'model_registry_table': 'ml_model_registry'
}

# Model registry
MODEL_REGISTRY = {
    'SAVE_MODELS': True,
    'MODEL_DIR': '/home/ubuntu/crypto_trading_model/models',
    'VERSION_CONTROL': True,
    'MAX_VERSIONS_TO_KEEP': 5,
    'METADATA_TRACKING': True
}

# Prediction thresholds for trading signals
TRADING_SIGNAL_THRESHOLDS = {
    'STRONG_BUY': 0.8,
    'BUY': 0.6,
    'NEUTRAL': 0.0,
    'SELL': -0.6,
    'STRONG_SELL': -0.8
}

# Performance metrics to track
PERFORMANCE_METRICS = [
    'accuracy', 'precision', 'recall', 'f1_score',
    'roc_auc', 'log_loss', 'mse', 'mae', 'rmse',
    'profit_factor', 'sharpe_ratio', 'sortino_ratio',
    'max_drawdown', 'win_rate', 'avg_win', 'avg_loss'
]
