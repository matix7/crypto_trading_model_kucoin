"""
LSTM model implementation for cryptocurrency price prediction.
"""

import logging
import os
import time
import json
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import joblib

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from .base_model import BaseModel
from .config import MODEL_TYPES, PREDICTION_TARGETS, FEATURE_GROUPS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/prediction.log',
    filemode='a'
)
logger = logging.getLogger('lstm_model')

class LSTMModel(BaseModel):
    """
    LSTM model implementation for cryptocurrency price prediction.
    """
    
    def __init__(self, coin: str, timeframe: str, db_path: str = None):
        """
        Initialize the LSTM model.
        
        Args:
            coin: Cryptocurrency symbol
            timeframe: Timeframe for predictions (e.g., '1m', '5m', '15m', '1h')
            db_path: Path to the SQLite database file
        """
        super().__init__('LSTM', coin, timeframe, db_path)
        
        # LSTM-specific attributes
        self.sequence_length = self.lookback_periods
        self.batch_size = self.config.get('batch_size', 32)
        self.epochs = self.config.get('epochs', 100)
        self.patience = self.config.get('patience', 10)
        self.validation_split = self.config.get('validation_split', 0.2)
        self.layers = self.config.get('layers', [128, 64, 32])
        self.dropout = self.config.get('dropout', 0.2)
        self.recurrent_dropout = self.config.get('recurrent_dropout', 0.2)
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        
        logger.info(f"Initialized LSTM model for {coin} on {timeframe} timeframe")
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for LSTM model training or prediction.
        
        Args:
            data: DataFrame with features and targets
            
        Returns:
            Tuple of preprocessed features and targets
        """
        try:
            logger.info(f"Preprocessing data with shape {data.shape}")
            
            # Separate features and targets
            feature_columns = []
            for group_name, group_config in FEATURE_GROUPS.items():
                if group_config.get('enabled', True):
                    # Add available features from this group
                    available_features = [f for f in group_config.get('features', []) if f in data.columns]
                    feature_columns.extend(available_features)
            
            # Ensure we have at least some features
            if not feature_columns:
                logger.error("No feature columns available in data")
                return np.array([]), np.array([])
            
            # Get target columns based on enabled prediction targets
            target_columns = []
            for target_name, target_config in PREDICTION_TARGETS.items():
                if target_config.get('enabled', True):
                    if target_name == 'PRICE_DIRECTION':
                        target_columns.append('future_direction')
                    elif target_name == 'PRICE_MOVEMENT':
                        for horizon in target_config.get('prediction_horizons', [5]):
                            target_columns.append(f'future_return_{horizon}')
                    elif target_name == 'VOLATILITY':
                        for horizon in target_config.get('prediction_horizons', [15]):
                            target_columns.append(f'future_volatility_{horizon}')
            
            # Ensure we have at least some targets
            if not target_columns:
                logger.error("No target columns available in data")
                return np.array([]), np.array([])
            
            # Store column names
            self.feature_columns = feature_columns
            self.target_columns = target_columns
            
            # Extract features and targets
            X = data[feature_columns].values
            y = data[target_columns].values
            
            # Scale features and targets
            X_scaled = self.feature_scaler.fit_transform(X)
            y_scaled = self.target_scaler.fit_transform(y)
            
            # Create sequences for LSTM
            X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled)
            
            logger.info(f"Preprocessed data: X shape {X_sequences.shape}, y shape {y_sequences.shape}")
            return X_sequences, y_sequences
        
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return np.array([]), np.array([])
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM model.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple of sequence arrays for features and targets
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - self.sequence_length):
            X_sequences.append(X[i:i + self.sequence_length])
            y_sequences.append(y[i + self.sequence_length])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def build_model(self) -> None:
        """
        Build the LSTM model architecture.
        """
        try:
            # Get input shape
            n_features = len(self.feature_columns)
            n_targets = len(self.target_columns)
            
            # Create model
            model = Sequential()
            
            # Add LSTM layers
            for i, units in enumerate(self.layers):
                if i == 0:
                    # First layer
                    model.add(LSTM(
                        units=units,
                        return_sequences=(i < len(self.layers) - 1),
                        dropout=self.dropout,
                        recurrent_dropout=self.recurrent_dropout,
                        input_shape=(self.sequence_length, n_features)
                    ))
                else:
                    # Hidden layers
                    model.add(LSTM(
                        units=units,
                        return_sequences=(i < len(self.layers) - 1),
                        dropout=self.dropout,
                        recurrent_dropout=self.recurrent_dropout
                    ))
                
                # Add batch normalization after each LSTM layer
                model.add(BatchNormalization())
            
            # Add output layer
            model.add(Dense(n_targets))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Set model
            self.model = model
            
            # Print model summary
            model.summary(print_fn=logger.info)
            
            logger.info(f"Built LSTM model with {len(self.layers)} layers")
        
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
    
    def train(self, data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Train the LSTM model on the provided data.
        
        Args:
            data: Training data
            validation_data: Validation data
            
        Returns:
            Dictionary with training results
        """
        try:
            logger.info(f"Training LSTM model on {len(data)} samples")
            
            # Preprocess data
            X, y = self.preprocess_data(data)
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No data available for training after preprocessing")
                return {'status': 'error', 'message': 'No data available for training'}
            
            # Split data if validation data not provided
            if validation_data is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=self.validation_split, shuffle=False
                )
            else:
                # Preprocess validation data
                X_val, y_val = self.preprocess_data(validation_data)
                X_train, y_train = X, y
            
            # Build model if not already built
            if self.model is None:
                self.build_model()
            
            # Define callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.patience // 2,
                    min_lr=0.0001
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Update last trained timestamp
            self.last_trained_at = datetime.now()
            
            # Evaluate on validation data
            val_loss, val_mae = self.model.evaluate(X_val, y_val, verbose=0)
            
            # Save performance metrics
            metrics = {
                'val_loss': val_loss,
                'val_mae': val_mae,
                'training_epochs': len(history.history['loss'])
            }
            
            self.save_performance_metrics(metrics)
            
            # Save model
            self.save_model()
            
            logger.info(f"Trained LSTM model: val_loss={val_loss:.4f}, val_mae={val_mae:.4f}")
            
            return {
                'status': 'success',
                'metrics': metrics,
                'history': history.history
            }
        
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Generate predictions using the trained LSTM model.
        
        Args:
            data: Data to predict on
            
        Returns:
            Dictionary with predictions
        """
        try:
            logger.info(f"Generating predictions for {len(data)} samples")
            
            # Ensure model is built
            if self.model is None:
                logger.error("Model not built or trained")
                return {'status': 'error', 'message': 'Model not built or trained'}
            
            # Preprocess data
            X, _ = self.preprocess_data(data)
            
            if len(X) == 0:
                logger.error("No data available for prediction after preprocessing")
                return {'status': 'error', 'message': 'No data available for prediction'}
            
            # Generate predictions
            y_pred_scaled = self.model.predict(X)
            
            # Inverse transform predictions
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
            
            # Create prediction results
            predictions = []
            
            for i, target_col in enumerate(self.target_columns):
                prediction_type = 'PRICE_DIRECTION'
                prediction_horizon = 1
                
                if 'future_return_' in target_col:
                    prediction_type = 'PRICE_MOVEMENT'
                    prediction_horizon = int(target_col.split('_')[-1])
                elif 'future_volatility_' in target_col:
                    prediction_type = 'VOLATILITY'
                    prediction_horizon = int(target_col.split('_')[-1])
                
                # Calculate confidence based on model uncertainty
                # For simplicity, using a fixed confidence value
                confidence = 0.8
                
                # Get the last prediction (most recent)
                prediction_value = y_pred[-1, i]
                
                # Add prediction
                predictions.append({
                    'prediction_type': prediction_type,
                    'prediction_horizon': prediction_horizon,
                    'prediction_value': float(prediction_value),
                    'confidence': confidence,
                    'upper_bound': float(prediction_value * 1.1),  # Simple upper bound
                    'lower_bound': float(prediction_value * 0.9),  # Simple lower bound
                    'features_used': self.feature_columns
                })
            
            # Save predictions to database
            prediction_data = {'predictions': predictions}
            self.save_prediction(prediction_data)
            
            logger.info(f"Generated {len(predictions)} predictions")
            
            return {
                'status': 'success',
                'predictions': predictions,
                'timestamp': int(time.time() * 1000)
            }
        
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, data: pd.DataFrame) -> Dict:
        """
        Evaluate the LSTM model performance.
        
        Args:
            data: Test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            logger.info(f"Evaluating model on {len(data)} samples")
            
            # Ensure model is built
            if self.model is None:
                logger.error("Model not built or trained")
                return {'status': 'error', 'message': 'Model not built or trained'}
            
            # Preprocess data
            X, y = self.preprocess_data(data)
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No data available for evaluation after preprocessing")
                return {'status': 'error', 'message': 'No data available for evaluation'}
            
            # Evaluate model
            loss, mae = self.model.evaluate(X, y, verbose=0)
            
            # Generate predictions for additional metrics
            y_pred_scaled = self.model.predict(X)
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
            y_true = self.target_scaler.inverse_transform(y)
            
            # Calculate additional metrics
            metrics = {
                'loss': loss,
                'mae': mae
            }
            
            # Calculate metrics for each target
            for i, target_col in enumerate(self.target_columns):
                # Calculate MSE and RMSE
                mse = np.mean((y_true[:, i] - y_pred[:, i]) ** 2)
                rmse = np.sqrt(mse)
                
                metrics[f'{target_col}_mse'] = mse
                metrics[f'{target_col}_rmse'] = rmse
                
                # For direction prediction, calculate accuracy
                if 'direction' in target_col:
                    # Convert to binary direction
                    y_true_dir = (y_true[:, i] > 0).astype(int)
                    y_pred_dir = (y_pred[:, i] > 0).astype(int)
                    
                    # Calculate accuracy
                    accuracy = np.mean(y_true_dir == y_pred_dir)
                    metrics[f'{target_col}_accuracy'] = accuracy
            
            # Save performance metrics
            self.save_performance_metrics(metrics, evaluation_window='test')
            
            logger.info(f"Evaluated model: loss={loss:.4f}, mae={mae:.4f}")
            
            return {
                'status': 'success',
                'metrics': metrics
            }
        
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def optimize_hyperparameters(self, data: pd.DataFrame) -> Dict:
        """
        Optimize LSTM model hyperparameters.
        
        Args:
            data: Data for hyperparameter optimization
            
        Returns:
            Dictionary with optimized hyperparameters
        """
        try:
            logger.info("Starting hyperparameter optimization")
            
            # For simplicity, we'll implement a basic grid search
            # In a real implementation, this would use Bayesian optimization or similar
            
            # Define hyperparameter grid
            param_grid = {
                'layers': [
                    [64, 32],
                    [128, 64],
                    [128, 64, 32]
                ],
                'dropout': [0.1, 0.2, 0.3],
                'recurrent_dropout': [0.1, 0.2, 0.3],
                'batch_size': [16, 32, 64]
            }
            
            # Split data into train and validation
            train_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)
            
            # Preprocess training data once
            X_train, y_train = self.preprocess_data(train_data)
            
            # Preprocess validation data once
            X_val, y_val = self.preprocess_data(val_data)
            
            best_val_loss = float('inf')
            best_params = {}
            
            # Simple grid search
            for layers in param_grid['layers']:
                for dropout in param_grid['dropout']:
                    for recurrent_dropout in param_grid['recurrent_dropout']:
                        for batch_size in param_grid['batch_size']:
                            # Update parameters
                            self.layers = layers
                            self.dropout = dropout
                            self.recurrent_dropout = recurrent_dropout
                            self.batch_size = batch_size
                            
                            # Build new model
                            self.model = None
                            self.build_model()
                            
                            # Train model
                            callbacks = [
                                EarlyStopping(
                                    monitor='val_loss',
                                    patience=5,
                                    restore_best_weights=True
                                )
                            ]
                            
                            history = self.model.fit(
                                X_train, y_train,
                                epochs=20,  # Reduced epochs for optimization
                                batch_size=batch_size,
                                validation_data=(X_val, y_val),
                                callbacks=callbacks,
                                verbose=0
                            )
                            
                            # Get validation loss
                            val_loss = min(history.history['val_loss'])
                            
                            logger.info(f"Params: layers={layers}, dropout={dropout}, "
                                       f"recurrent_dropout={recurrent_dropout}, batch_size={batch_size}, "
                                       f"val_loss={val_loss:.4f}")
                            
                            # Update best parameters
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_params = {
                                    'layers': layers,
                                    'dropout': dropout,
                                    'recurrent_dropout': recurrent_dropout,
                                    'batch_size': batch_size
                                }
            
            # Update model with best parameters
            self.layers = best_params['layers']
            self.dropout = best_params['dropout']
            self.recurrent_dropout = best_params['recurrent_dropout']
            self.batch_size = best_params['batch_size']
            
            # Update config
            self.config.update(best_params)
            
            # Rebuild model with best parameters
            self.model = None
            self.build_model()
            
            # Update last optimized timestamp
            self.last_optimized_at = datetime.now()
            
            logger.info(f"Optimized hyperparameters: {best_params}, val_loss={best_val_loss:.4f}")
            
            return {
                'status': 'success',
                'best_params': best_params,
                'best_val_loss': best_val_loss
            }
        
        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def calculate_feature_importance(self, data: pd.DataFrame) -> Dict:
        """
        Calculate feature importance for LSTM model.
        
        Args:
            data: Data for feature importance calculation
            
        Returns:
            Dictionary with feature importance values
        """
        try:
            logger.info("Calculating feature importance")
            
            # For LSTM models, we'll use a permutation importance approach
            # This is a simplified version; a real implementation would be more comprehensive
            
            # Preprocess data
            X, y = self.preprocess_data(data)
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No data available for feature importance calculation")
                return {}
            
            # Get baseline performance
            baseline_loss, _ = self.model.evaluate(X, y, verbose=0)
            
            # Calculate importance for each feature
            importance = {}
            
            # For each feature, permute values and measure impact
            for i, feature_name in enumerate(self.feature_columns):
                # Create a copy of the data
                X_permuted = X.copy()
                
                # Permute the feature
                for j in range(X_permuted.shape[0]):
                    # Shuffle this feature across all time steps
                    np.random.shuffle(X_permuted[j, :, i])
                
                # Evaluate with permuted feature
                permuted_loss, _ = self.model.evaluate(X_permuted, y, verbose=0)
                
                # Calculate importance as increase in loss
                feature_importance = permuted_loss - baseline_loss
                
                # Store importance
                importance[feature_name] = float(feature_importance)
            
            # Normalize importance values
            max_importance = max(importance.values())
            if max_importance > 0:
                for feature_name in importance:
                    importance[feature_name] /= max_importance
            
            # Save feature importance
            self.save_feature_importance(importance, method='PERMUTATION')
            
            logger.info(f"Calculated feature importance for {len(importance)} features")
            
            return importance
        
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}
