"""
Base model class for prediction models.
"""

import logging
import os
import time
import json
import sqlite3
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import pickle
import joblib

from .config import (
    MODEL_TYPES, ENSEMBLE, FEATURE_IMPORTANCE, PREDICTION_TARGETS,
    FEATURE_GROUPS, HYPERPARAMETER_OPTIMIZATION, SELF_LEARNING,
    PREDICTION_OUTPUT, DATABASE, MODEL_REGISTRY, TRADING_SIGNAL_THRESHOLDS,
    PERFORMANCE_METRICS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/prediction.log',
    filemode='a'
)
logger = logging.getLogger('base_model')

class BaseModel(ABC):
    """
    Abstract base class for all prediction models.
    Defines the interface that all model classes must implement.
    """
    
    def __init__(self, model_type: str, coin: str, timeframe: str, db_path: str = None):
        """
        Initialize the BaseModel.
        
        Args:
            model_type: Type of model (e.g., LSTM, GRU, TRANSFORMER, XGB)
            coin: Cryptocurrency symbol
            timeframe: Timeframe for predictions (e.g., '1m', '5m', '15m', '1h')
            db_path: Path to the SQLite database file
        """
        self.model_type = model_type
        self.coin = coin
        self.timeframe = timeframe
        self.db_path = db_path or '/home/ubuntu/crypto_trading_model/data/market_data.db'
        self.config = MODEL_TYPES.get(model_type, {})
        
        # Model attributes
        self.model = None
        self.scaler = None
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_columns = []
        self.target_columns = []
        self.lookback_periods = self.config.get('lookback_periods', 60)
        
        # Performance tracking
        self.performance_metrics = {}
        self.feature_importance = {}
        
        # Model metadata
        self.model_id = f"{model_type}_{coin}_{timeframe}_{int(time.time())}"
        self.model_version = 1
        self.created_at = datetime.now()
        self.last_trained_at = None
        self.last_optimized_at = None
        
        # Ensure logs directory exists
        os.makedirs('/home/ubuntu/crypto_trading_model/logs', exist_ok=True)
        
        # Ensure model directory exists
        os.makedirs(MODEL_REGISTRY['MODEL_DIR'], exist_ok=True)
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Initialized {model_type} model for {coin} on {timeframe} timeframe")
    
    def _init_database(self):
        """Initialize the SQLite database with required tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create predictions table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['predictions_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                coin TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                prediction_horizon INTEGER NOT NULL,
                prediction_value REAL NOT NULL,
                confidence REAL NOT NULL,
                upper_bound REAL,
                lower_bound REAL,
                features_used TEXT,
                actual_value REAL,
                error REAL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Create model performance table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['model_performance_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                coin TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                evaluation_window TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Create feature importance table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['feature_importance_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                coin TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                importance_value REAL NOT NULL,
                importance_method TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Create model weights table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['model_weights_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                ensemble_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                coin TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                weight REAL NOT NULL,
                created_at INTEGER NOT NULL
            )
            ''')
            
            # Create model registry table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['model_registry_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                model_type TEXT NOT NULL,
                coin TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                version INTEGER NOT NULL,
                model_path TEXT NOT NULL,
                scaler_path TEXT,
                feature_columns TEXT,
                target_columns TEXT,
                hyperparameters TEXT,
                performance_summary TEXT,
                created_at INTEGER NOT NULL,
                last_trained_at INTEGER,
                last_optimized_at INTEGER,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                UNIQUE(model_id, version)
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database tables initialized")
        
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for model training or prediction.
        
        Args:
            data: DataFrame with features and targets
            
        Returns:
            Tuple of preprocessed features and targets
        """
        pass
    
    @abstractmethod
    def build_model(self) -> None:
        """
        Build the model architecture.
        """
        pass
    
    @abstractmethod
    def train(self, data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Train the model on the provided data.
        
        Args:
            data: Training data
            validation_data: Validation data
            
        Returns:
            Dictionary with training results
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Generate predictions using the trained model.
        
        Args:
            data: Data to predict on
            
        Returns:
            Dictionary with predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> Dict:
        """
        Evaluate the model performance.
        
        Args:
            data: Test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    @abstractmethod
    def optimize_hyperparameters(self, data: pd.DataFrame) -> Dict:
        """
        Optimize model hyperparameters.
        
        Args:
            data: Data for hyperparameter optimization
            
        Returns:
            Dictionary with optimized hyperparameters
        """
        pass
    
    @abstractmethod
    def calculate_feature_importance(self, data: pd.DataFrame) -> Dict:
        """
        Calculate feature importance.
        
        Args:
            data: Data for feature importance calculation
            
        Returns:
            Dictionary with feature importance values
        """
        pass
    
    def save_model(self) -> str:
        """
        Save the trained model to disk.
        
        Returns:
            Path to the saved model
        """
        try:
            # Create model directory if it doesn't exist
            model_dir = os.path.join(MODEL_REGISTRY['MODEL_DIR'], self.coin, self.timeframe, self.model_type)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_dir, f"{self.model_id}_v{self.model_version}.pkl")
            
            if self.model_type in ['LSTM', 'GRU', 'TRANSFORMER']:
                # Save Keras model
                self.model.save(model_path.replace('.pkl', '.h5'))
                model_path = model_path.replace('.pkl', '.h5')
            else:
                # Save scikit-learn or XGBoost model
                joblib.dump(self.model, model_path)
            
            # Save scalers
            scaler_path = os.path.join(model_dir, f"{self.model_id}_v{self.model_version}_scaler.pkl")
            scalers = {
                'scaler': self.scaler,
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler
            }
            joblib.dump(scalers, scaler_path)
            
            # Save to model registry
            self._save_to_registry(model_path, scaler_path)
            
            logger.info(f"Model saved to {model_path}")
            return model_path
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return ""
    
    def load_model(self, model_id: str = None, version: int = None) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_id: Model ID to load (defaults to current model_id)
            version: Model version to load (defaults to latest)
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            # Use current model_id if not specified
            model_id = model_id or self.model_id
            
            # Get model info from registry
            model_info = self._get_model_from_registry(model_id, version)
            
            if not model_info:
                logger.warning(f"Model {model_id} version {version} not found in registry")
                return False
            
            # Load model
            model_path = model_info['model_path']
            
            if model_path.endswith('.h5'):
                # Load Keras model
                from tensorflow.keras.models import load_model
                self.model = load_model(model_path)
            else:
                # Load scikit-learn or XGBoost model
                self.model = joblib.load(model_path)
            
            # Load scalers
            scaler_path = model_info['scaler_path']
            if scaler_path and os.path.exists(scaler_path):
                scalers = joblib.load(scaler_path)
                self.scaler = scalers.get('scaler')
                self.feature_scaler = scalers.get('feature_scaler')
                self.target_scaler = scalers.get('target_scaler')
            
            # Load metadata
            self.model_id = model_info['model_id']
            self.model_version = model_info['version']
            self.feature_columns = json.loads(model_info['feature_columns'])
            self.target_columns = json.loads(model_info['target_columns'])
            self.created_at = datetime.fromtimestamp(model_info['created_at'] / 1000)
            
            if model_info['last_trained_at']:
                self.last_trained_at = datetime.fromtimestamp(model_info['last_trained_at'] / 1000)
            
            if model_info['last_optimized_at']:
                self.last_optimized_at = datetime.fromtimestamp(model_info['last_optimized_at'] / 1000)
            
            logger.info(f"Model {model_id} version {self.model_version} loaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _save_to_registry(self, model_path: str, scaler_path: str) -> None:
        """
        Save model information to the registry.
        
        Args:
            model_path: Path to the saved model
            scaler_path: Path to the saved scalers
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if model already exists in registry
            cursor.execute(f'''
            SELECT version FROM {DATABASE['model_registry_table']}
            WHERE model_id = ? ORDER BY version DESC LIMIT 1
            ''', (self.model_id,))
            
            result = cursor.fetchone()
            
            if result:
                # Update version
                self.model_version = result[0] + 1
            
            # Prepare performance summary
            performance_summary = json.dumps(self.performance_metrics)
            
            # Prepare hyperparameters
            hyperparameters = json.dumps(self.config)
            
            # Insert into registry
            cursor.execute(f'''
            INSERT INTO {DATABASE['model_registry_table']}
            (model_id, model_type, coin, timeframe, version, model_path, scaler_path,
             feature_columns, target_columns, hyperparameters, performance_summary,
             created_at, last_trained_at, last_optimized_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.model_id,
                self.model_type,
                self.coin,
                self.timeframe,
                self.model_version,
                model_path,
                scaler_path,
                json.dumps(self.feature_columns),
                json.dumps(self.target_columns),
                hyperparameters,
                performance_summary,
                int(time.time() * 1000),
                int(self.last_trained_at.timestamp() * 1000) if self.last_trained_at else None,
                int(self.last_optimized_at.timestamp() * 1000) if self.last_optimized_at else None,
                True
            ))
            
            # If version control is enabled, deactivate old versions beyond the limit
            if MODEL_REGISTRY['VERSION_CONTROL'] and MODEL_REGISTRY['MAX_VERSIONS_TO_KEEP'] > 0:
                cursor.execute(f'''
                UPDATE {DATABASE['model_registry_table']}
                SET is_active = 0
                WHERE model_id = ? AND version NOT IN (
                    SELECT version FROM {DATABASE['model_registry_table']}
                    WHERE model_id = ?
                    ORDER BY version DESC
                    LIMIT {MODEL_REGISTRY['MAX_VERSIONS_TO_KEEP']}
                )
                ''', (self.model_id, self.model_id))
            
            conn.commit()
            conn.close()
            logger.info(f"Model {self.model_id} version {self.model_version} saved to registry")
        
        except Exception as e:
            logger.error(f"Error saving to registry: {str(e)}")
    
    def _get_model_from_registry(self, model_id: str, version: int = None) -> Dict:
        """
        Get model information from the registry.
        
        Args:
            model_id: Model ID to retrieve
            version: Model version to retrieve (defaults to latest active version)
            
        Returns:
            Dictionary with model information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if version:
                # Get specific version
                cursor.execute(f'''
                SELECT * FROM {DATABASE['model_registry_table']}
                WHERE model_id = ? AND version = ?
                ''', (model_id, version))
            else:
                # Get latest active version
                cursor.execute(f'''
                SELECT * FROM {DATABASE['model_registry_table']}
                WHERE model_id = ? AND is_active = 1
                ORDER BY version DESC LIMIT 1
                ''', (model_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return {}
            
            # Convert to dictionary
            columns = [
                'id', 'model_id', 'model_type', 'coin', 'timeframe', 'version',
                'model_path', 'scaler_path', 'feature_columns', 'target_columns',
                'hyperparameters', 'performance_summary', 'created_at',
                'last_trained_at', 'last_optimized_at', 'is_active'
            ]
            
            return dict(zip(columns, result))
        
        except Exception as e:
            logger.error(f"Error getting model from registry: {str(e)}")
            return {}
    
    def save_prediction(self, prediction_data: Dict) -> None:
        """
        Save prediction to database.
        
        Args:
            prediction_data: Dictionary with prediction information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for pred in prediction_data.get('predictions', []):
                cursor.execute(f'''
                INSERT INTO {DATABASE['predictions_table']}
                (timestamp, model_id, coin, timeframe, prediction_type, prediction_horizon,
                 prediction_value, confidence, upper_bound, lower_bound, features_used,
                 created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    int(time.time() * 1000),
                    self.model_id,
                    self.coin,
                    self.timeframe,
                    pred.get('prediction_type', ''),
                    pred.get('prediction_horizon', 0),
                    pred.get('prediction_value', 0.0),
                    pred.get('confidence', 0.0),
                    pred.get('upper_bound', None),
                    pred.get('lower_bound', None),
                    json.dumps(pred.get('features_used', [])),
                    int(time.time() * 1000)
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved {len(prediction_data.get('predictions', []))} predictions to database")
        
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
    
    def update_prediction_actual(self, prediction_id: int, actual_value: float) -> None:
        """
        Update prediction with actual value and calculate error.
        
        Args:
            prediction_id: ID of the prediction to update
            actual_value: Actual value that occurred
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get prediction value
            cursor.execute(f'''
            SELECT prediction_value FROM {DATABASE['predictions_table']}
            WHERE id = ?
            ''', (prediction_id,))
            
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Prediction {prediction_id} not found")
                conn.close()
                return
            
            prediction_value = result[0]
            
            # Calculate error
            error = actual_value - prediction_value
            
            # Update prediction
            cursor.execute(f'''
            UPDATE {DATABASE['predictions_table']}
            SET actual_value = ?, error = ?
            WHERE id = ?
            ''', (actual_value, error, prediction_id))
            
            conn.commit()
            conn.close()
            logger.info(f"Updated prediction {prediction_id} with actual value {actual_value}")
        
        except Exception as e:
            logger.error(f"Error updating prediction actual: {str(e)}")
    
    def save_performance_metrics(self, metrics: Dict, evaluation_window: str = '24h') -> None:
        """
        Save performance metrics to database.
        
        Args:
            metrics: Dictionary with performance metrics
            evaluation_window: Time window for evaluation (e.g., '24h', '7d', '30d')
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for metric_name, metric_value in metrics.items():
                cursor.execute(f'''
                INSERT INTO {DATABASE['model_performance_table']}
                (timestamp, model_id, coin, timeframe, metric_name, metric_value,
                 evaluation_window, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    int(time.time() * 1000),
                    self.model_id,
                    self.coin,
                    self.timeframe,
                    metric_name,
                    metric_value,
                    evaluation_window,
                    int(time.time() * 1000)
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved {len(metrics)} performance metrics to database")
            
            # Update instance performance metrics
            self.performance_metrics = {**self.performance_metrics, **metrics}
        
        except Exception as e:
            logger.error(f"Error saving performance metrics: {str(e)}")
    
    def save_feature_importance(self, importance_data: Dict, method: str = 'SHAP') -> None:
        """
        Save feature importance to database.
        
        Args:
            importance_data: Dictionary with feature importance values
            method: Method used to calculate feature importance
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for feature_name, importance_value in importance_data.items():
                cursor.execute(f'''
                INSERT INTO {DATABASE['feature_importance_table']}
                (timestamp, model_id, coin, timeframe, feature_name, importance_value,
                 importance_method, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    int(time.time() * 1000),
                    self.model_id,
                    self.coin,
                    self.timeframe,
                    feature_name,
                    importance_value,
                    method,
                    int(time.time() * 1000)
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved {len(importance_data)} feature importance values to database")
            
            # Update instance feature importance
            self.feature_importance = importance_data
        
        except Exception as e:
            logger.error(f"Error saving feature importance: {str(e)}")
    
    def get_performance_history(self, metric_name: str = None, days: int = 7) -> pd.DataFrame:
        """
        Get historical performance metrics.
        
        Args:
            metric_name: Specific metric to retrieve (None for all metrics)
            days: Number of days of history to retrieve
            
        Returns:
            DataFrame with historical performance metrics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate timestamp for 'days' ago
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            if metric_name:
                query = f'''
                SELECT timestamp, metric_name, metric_value, evaluation_window
                FROM {DATABASE['model_performance_table']}
                WHERE model_id = ? AND metric_name = ? AND timestamp >= ?
                ORDER BY timestamp ASC
                '''
                params = (self.model_id, metric_name, start_time)
            else:
                query = f'''
                SELECT timestamp, metric_name, metric_value, evaluation_window
                FROM {DATABASE['model_performance_table']}
                WHERE model_id = ? AND timestamp >= ?
                ORDER BY timestamp ASC
                '''
                params = (self.model_id, start_time)
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            # Convert timestamp to datetime
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        
        except Exception as e:
            logger.error(f"Error retrieving performance history: {str(e)}")
            return pd.DataFrame()
    
    def get_feature_importance_history(self, days: int = 7) -> pd.DataFrame:
        """
        Get historical feature importance.
        
        Args:
            days: Number of days of history to retrieve
            
        Returns:
            DataFrame with historical feature importance
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate timestamp for 'days' ago
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            query = f'''
            SELECT timestamp, feature_name, importance_value, importance_method
            FROM {DATABASE['feature_importance_table']}
            WHERE model_id = ? AND timestamp >= ?
            ORDER BY timestamp ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=(self.model_id, start_time))
            conn.close()
            
            # Convert timestamp to datetime
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        
        except Exception as e:
            logger.error(f"Error retrieving feature importance history: {str(e)}")
            return pd.DataFrame()
    
    def get_prediction_history(self, prediction_type: str = None, days: int = 7) -> pd.DataFrame:
        """
        Get historical predictions.
        
        Args:
            prediction_type: Specific prediction type to retrieve (None for all types)
            days: Number of days of history to retrieve
            
        Returns:
            DataFrame with historical predictions
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate timestamp for 'days' ago
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            if prediction_type:
                query = f'''
                SELECT timestamp, prediction_type, prediction_horizon, prediction_value,
                       confidence, actual_value, error
                FROM {DATABASE['predictions_table']}
                WHERE model_id = ? AND prediction_type = ? AND timestamp >= ?
                ORDER BY timestamp ASC
                '''
                params = (self.model_id, prediction_type, start_time)
            else:
                query = f'''
                SELECT timestamp, prediction_type, prediction_horizon, prediction_value,
                       confidence, actual_value, error
                FROM {DATABASE['predictions_table']}
                WHERE model_id = ? AND timestamp >= ?
                ORDER BY timestamp ASC
                '''
                params = (self.model_id, start_time)
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            # Convert timestamp to datetime
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        
        except Exception as e:
            logger.error(f"Error retrieving prediction history: {str(e)}")
            return pd.DataFrame()
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.model_type} model for {self.coin} on {self.timeframe} timeframe"
    
    def __repr__(self) -> str:
        """Detailed representation of the model."""
        return f"PredictionModel(type='{self.model_type}', coin='{self.coin}', timeframe='{self.timeframe}', id='{self.model_id}')"
