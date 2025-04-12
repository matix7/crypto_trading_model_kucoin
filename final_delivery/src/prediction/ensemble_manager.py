"""
Ensemble model manager for combining multiple prediction models.
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

from .base_model import BaseModel
from .lstm_model import LSTMModel
from .config import (
    MODEL_TYPES, ENSEMBLE, PREDICTION_TARGETS, FEATURE_GROUPS,
    DATABASE, TRADING_SIGNAL_THRESHOLDS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/prediction.log',
    filemode='a'
)
logger = logging.getLogger('ensemble_manager')

class EnsembleManager:
    """
    Manager class for combining and managing multiple prediction models.
    Handles model initialization, ensemble predictions, and dynamic weight adjustment.
    """
    
    def __init__(self, coin: str, timeframe: str, db_path: str = None):
        """
        Initialize the EnsembleManager.
        
        Args:
            coin: Cryptocurrency symbol
            timeframe: Timeframe for predictions (e.g., '1m', '5m', '15m', '1h')
            db_path: Path to the SQLite database file
        """
        self.coin = coin
        self.timeframe = timeframe
        self.db_path = db_path or '/home/ubuntu/crypto_trading_model/data/market_data.db'
        self.models = {}
        self.model_weights = {}
        self.ensemble_id = f"ENSEMBLE_{coin}_{timeframe}_{int(time.time())}"
        
        # Ensemble configuration
        self.ensemble_method = ENSEMBLE.get('METHOD', 'WEIGHTED_AVERAGE')
        self.dynamic_weights = ENSEMBLE.get('DYNAMIC_WEIGHTS', True)
        self.weight_update_frequency = ENSEMBLE.get('WEIGHT_UPDATE_FREQUENCY', 24)
        self.performance_window = ENSEMBLE.get('PERFORMANCE_WINDOW', 168)
        self.min_weight = ENSEMBLE.get('MIN_WEIGHT', 0.05)
        self.confidence_threshold = ENSEMBLE.get('CONFIDENCE_THRESHOLD', 0.65)
        
        # Last weight update timestamp
        self.last_weight_update = None
        
        # Ensure logs directory exists
        os.makedirs('/home/ubuntu/crypto_trading_model/logs', exist_ok=True)
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize models
        self._init_models()
        
        logger.info(f"Initialized EnsembleManager for {coin} on {timeframe} timeframe")
    
    def _init_models(self):
        """Initialize all prediction models."""
        try:
            # Initialize models based on configuration
            for model_type, config in MODEL_TYPES.items():
                if config.get('enabled', False):
                    if model_type == 'LSTM':
                        self.models[model_type] = LSTMModel(
                            coin=self.coin,
                            timeframe=self.timeframe,
                            db_path=self.db_path
                        )
                    # Add other model types here as they are implemented
                    # elif model_type == 'GRU':
                    #     self.models[model_type] = GRUModel(...)
                    # elif model_type == 'TRANSFORMER':
                    #     self.models[model_type] = TransformerModel(...)
                    # elif model_type == 'XGB':
                    #     self.models[model_type] = XGBoostModel(...)
                    
                    # Set initial weight from configuration
                    self.model_weights[model_type] = config.get('weight', 0.0)
            
            # Normalize weights
            self._normalize_weights()
            
            # Save initial weights
            self._save_model_weights()
            
            logger.info(f"Initialized {len(self.models)} models")
        
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
    
    def _normalize_weights(self):
        """Normalize model weights to sum to 1.0."""
        try:
            # Get sum of weights
            weight_sum = sum(self.model_weights.values())
            
            if weight_sum > 0:
                # Normalize weights
                for model_type in self.model_weights:
                    self.model_weights[model_type] /= weight_sum
            else:
                # Equal weights if sum is 0
                equal_weight = 1.0 / len(self.model_weights) if self.model_weights else 0.0
                for model_type in self.model_weights:
                    self.model_weights[model_type] = equal_weight
            
            logger.info(f"Normalized model weights: {self.model_weights}")
        
        except Exception as e:
            logger.error(f"Error normalizing weights: {str(e)}")
    
    def _save_model_weights(self):
        """Save model weights to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for model_type, weight in self.model_weights.items():
                model_id = self.models[model_type].model_id if model_type in self.models else f"{model_type}_{self.coin}_{self.timeframe}"
                
                cursor.execute(f'''
                INSERT INTO {DATABASE['model_weights_table']}
                (timestamp, ensemble_id, model_id, coin, timeframe, weight, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    int(time.time() * 1000),
                    self.ensemble_id,
                    model_id,
                    self.coin,
                    self.timeframe,
                    weight,
                    int(time.time() * 1000)
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved model weights to database")
        
        except Exception as e:
            logger.error(f"Error saving model weights: {str(e)}")
    
    def train_all_models(self, data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Train all models in the ensemble.
        
        Args:
            data: Training data
            validation_data: Validation data
            
        Returns:
            Dictionary with training results for all models
        """
        try:
            logger.info(f"Training all models on {len(data)} samples")
            
            results = {}
            
            for model_type, model in self.models.items():
                logger.info(f"Training {model_type} model")
                
                # Train model
                train_result = model.train(data, validation_data)
                
                results[model_type] = train_result
            
            # Update weights based on performance if dynamic weights enabled
            if self.dynamic_weights:
                self._update_weights_based_on_performance()
            
            logger.info(f"Trained {len(results)} models")
            
            return {
                'status': 'success',
                'results': results
            }
        
        except Exception as e:
            logger.error(f"Error training all models: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Generate ensemble predictions by combining predictions from all models.
        
        Args:
            data: Data to predict on
            
        Returns:
            Dictionary with ensemble predictions
        """
        try:
            logger.info(f"Generating ensemble predictions for {len(data)} samples")
            
            # Check if weights need to be updated
            if self.dynamic_weights and self.last_weight_update:
                hours_since_update = (datetime.now() - self.last_weight_update).total_seconds() / 3600
                if hours_since_update >= self.weight_update_frequency:
                    self._update_weights_based_on_performance()
            
            # Get predictions from all models
            model_predictions = {}
            
            for model_type, model in self.models.items():
                logger.info(f"Getting predictions from {model_type} model")
                
                # Generate predictions
                pred_result = model.predict(data)
                
                if pred_result.get('status') == 'success':
                    model_predictions[model_type] = pred_result.get('predictions', [])
            
            # Combine predictions using ensemble method
            ensemble_predictions = self._combine_predictions(model_predictions)
            
            # Generate trading signals
            trading_signals = self._generate_trading_signals(ensemble_predictions)
            
            logger.info(f"Generated {len(ensemble_predictions)} ensemble predictions")
            
            return {
                'status': 'success',
                'predictions': ensemble_predictions,
                'trading_signals': trading_signals,
                'model_weights': self.model_weights,
                'timestamp': int(time.time() * 1000)
            }
        
        except Exception as e:
            logger.error(f"Error generating ensemble predictions: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _combine_predictions(self, model_predictions: Dict) -> List[Dict]:
        """
        Combine predictions from multiple models using the specified ensemble method.
        
        Args:
            model_predictions: Dictionary with predictions from each model
            
        Returns:
            List of combined predictions
        """
        try:
            # If no predictions, return empty list
            if not model_predictions:
                return []
            
            # Get all prediction types and horizons
            prediction_types = set()
            prediction_horizons = {}
            
            for model_type, predictions in model_predictions.items():
                for pred in predictions:
                    pred_type = pred.get('prediction_type')
                    pred_horizon = pred.get('prediction_horizon')
                    
                    prediction_types.add(pred_type)
                    
                    if pred_type not in prediction_horizons:
                        prediction_horizons[pred_type] = set()
                    
                    prediction_horizons[pred_type].add(pred_horizon)
            
            # Combine predictions for each type and horizon
            ensemble_predictions = []
            
            for pred_type in prediction_types:
                for pred_horizon in prediction_horizons.get(pred_type, []):
                    # Get predictions for this type and horizon from all models
                    type_horizon_predictions = []
                    
                    for model_type, predictions in model_predictions.items():
                        for pred in predictions:
                            if (pred.get('prediction_type') == pred_type and 
                                pred.get('prediction_horizon') == pred_horizon):
                                # Add model weight to prediction
                                pred_with_weight = pred.copy()
                                pred_with_weight['model_type'] = model_type
                                pred_with_weight['model_weight'] = self.model_weights.get(model_type, 0.0)
                                
                                type_horizon_predictions.append(pred_with_weight)
                    
                    # Combine predictions for this type and horizon
                    if type_horizon_predictions:
                        ensemble_pred = self._ensemble_method(type_horizon_predictions)
                        ensemble_predictions.append(ensemble_pred)
            
            return ensemble_predictions
        
        except Exception as e:
            logger.error(f"Error combining predictions: {str(e)}")
            return []
    
    def _ensemble_method(self, predictions: List[Dict]) -> Dict:
        """
        Apply the specified ensemble method to combine predictions.
        
        Args:
            predictions: List of predictions to combine
            
        Returns:
            Combined prediction
        """
        try:
            if not predictions:
                return {}
            
            # Get prediction type and horizon from first prediction
            pred_type = predictions[0].get('prediction_type')
            pred_horizon = predictions[0].get('prediction_horizon')
            
            if self.ensemble_method == 'WEIGHTED_AVERAGE':
                # Calculate weighted average
                weighted_sum = 0.0
                weight_sum = 0.0
                confidence_sum = 0.0
                features_used = set()
                
                for pred in predictions:
                    model_weight = pred.get('model_weight', 0.0)
                    confidence = pred.get('confidence', 0.0)
                    
                    # Only include predictions with confidence above threshold
                    if confidence >= self.confidence_threshold:
                        weighted_sum += pred.get('prediction_value', 0.0) * model_weight * confidence
                        weight_sum += model_weight * confidence
                        confidence_sum += confidence * model_weight
                        
                        # Add features used
                        if 'features_used' in pred:
                            features_used.update(pred['features_used'])
                
                # Calculate ensemble prediction
                if weight_sum > 0:
                    prediction_value = weighted_sum / weight_sum
                    confidence = confidence_sum / sum(pred.get('model_weight', 0.0) for pred in predictions)
                else:
                    # Fallback to simple average if no weights
                    prediction_value = sum(pred.get('prediction_value', 0.0) for pred in predictions) / len(predictions)
                    confidence = sum(pred.get('confidence', 0.0) for pred in predictions) / len(predictions)
                
                # Calculate upper and lower bounds
                upper_bounds = [pred.get('upper_bound', pred.get('prediction_value', 0.0) * 1.1) for pred in predictions]
                lower_bounds = [pred.get('lower_bound', pred.get('prediction_value', 0.0) * 0.9) for pred in predictions]
                
                upper_bound = sum(upper_bounds) / len(upper_bounds)
                lower_bound = sum(lower_bounds) / len(lower_bounds)
                
                return {
                    'prediction_type': pred_type,
                    'prediction_horizon': pred_horizon,
                    'prediction_value': float(prediction_value),
                    'confidence': float(confidence),
                    'upper_bound': float(upper_bound),
                    'lower_bound': float(lower_bound),
                    'features_used': list(features_used),
                    'ensemble_method': 'WEIGHTED_AVERAGE',
                    'models_used': [pred.get('model_type') for pred in predictions]
                }
            
            elif self.ensemble_method == 'VOTING':
                # For direction prediction, use voting
                if pred_type == 'PRICE_DIRECTION':
                    # Count votes for each direction
                    up_votes = 0.0
                    down_votes = 0.0
                    
                    for pred in predictions:
                        model_weight = pred.get('model_weight', 0.0)
                        confidence = pred.get('confidence', 0.0)
                        prediction_value = pred.get('prediction_value', 0.0)
                        
                        # Only include predictions with confidence above threshold
                        if confidence >= self.confidence_threshold:
                            if prediction_value > 0:
                                up_votes += model_weight * confidence
                            else:
                                down_votes += model_weight * confidence
                    
                    # Determine direction based on votes
                    if up_votes > down_votes:
                        prediction_value = 1.0
                        confidence = up_votes / (up_votes + down_votes) if (up_votes + down_votes) > 0 else 0.5
                    elif down_votes > up_votes:
                        prediction_value = -1.0
                        confidence = down_votes / (up_votes + down_votes) if (up_votes + down_votes) > 0 else 0.5
                    else:
                        prediction_value = 0.0
                        confidence = 0.5
                    
                    return {
                        'prediction_type': pred_type,
                        'prediction_horizon': pred_horizon,
                        'prediction_value': float(prediction_value),
                        'confidence': float(confidence),
                        'upper_bound': None,
                        'lower_bound': None,
                        'features_used': list(set().union(*[set(pred.get('features_used', [])) for pred in predictions])),
                        'ensemble_method': 'VOTING',
                        'models_used': [pred.get('model_type') for pred in predictions]
                    }
                else:
                    # For non-direction predictions, fall back to weighted average
                    return self._ensemble_method_weighted_average(predictions)
            
            else:
                # Default to weighted average
                return self._ensemble_method_weighted_average(predictions)
        
        except Exception as e:
            logger.error(f"Error applying ensemble method: {str(e)}")
            return {}
    
    def _generate_trading_signals(self, predictions: List[Dict]) -> Dict:
        """
        Generate trading signals from ensemble predictions.
        
        Args:
            predictions: List of ensemble predictions
            
        Returns:
            Dictionary with trading signals
        """
        try:
            if not predictions:
                return {'signal': 'NEUTRAL', 'strength': 0.0, 'confidence': 0.0}
            
            # Initialize signal components
            direction_signal = 0.0
            direction_confidence = 0.0
            price_movement_signal = 0.0
            price_movement_confidence = 0.0
            volatility_signal = 0.0
            volatility_confidence = 0.0
            
            # Process each prediction
            for pred in predictions:
                pred_type = pred.get('prediction_type')
                pred_value = pred.get('prediction_value', 0.0)
                confidence = pred.get('confidence', 0.0)
                
                if pred_type == 'PRICE_DIRECTION':
                    direction_signal = pred_value
                    direction_confidence = confidence
                elif pred_type == 'PRICE_MOVEMENT':
                    price_movement_signal = pred_value
                    price_movement_confidence = confidence
                elif pred_type == 'VOLATILITY':
                    # High volatility prediction reduces signal strength
                    volatility_signal = pred_value
                    volatility_confidence = confidence
            
            # Combine signals
            # Direction has highest weight
            combined_signal = (
                direction_signal * 0.6 +
                price_movement_signal * 0.4
            )
            
            # Adjust for volatility - reduce signal strength in high volatility
            if volatility_signal > 0.5 and volatility_confidence > 0.6:
                combined_signal *= 0.8
            
            # Calculate overall confidence
            combined_confidence = (
                direction_confidence * 0.6 +
                price_movement_confidence * 0.4
            )
            
            # Determine signal type
            signal_type = 'NEUTRAL'
            if combined_signal >= TRADING_SIGNAL_THRESHOLDS['STRONG_BUY']:
                signal_type = 'STRONG_BUY'
            elif combined_signal >= TRADING_SIGNAL_THRESHOLDS['BUY']:
                signal_type = 'BUY'
            elif combined_signal <= TRADING_SIGNAL_THRESHOLDS['STRONG_SELL']:
                signal_type = 'STRONG_SELL'
            elif combined_signal <= TRADING_SIGNAL_THRESHOLDS['SELL']:
                signal_type = 'SELL'
            
            return {
                'signal': signal_type,
                'strength': float(combined_signal),
                'confidence': float(combined_confidence),
                'direction_component': float(direction_signal),
                'price_movement_component': float(price_movement_signal),
                'volatility_component': float(volatility_signal),
                'timestamp': int(time.time() * 1000)
            }
        
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            return {'signal': 'NEUTRAL', 'strength': 0.0, 'confidence': 0.0}
    
    def _update_weights_based_on_performance(self):
        """Update model weights based on recent performance."""
        try:
            logger.info("Updating model weights based on performance")
            
            # Get performance metrics for all models
            performance_metrics = {}
            
            for model_type, model in self.models.items():
                # Get performance history
                perf_df = model.get_performance_history(days=self.performance_window // 24)
                
                if not perf_df.empty:
                    # Calculate average metrics
                    avg_metrics = {}
                    
                    for metric in ['val_loss', 'val_mae']:
                        if metric in perf_df['metric_name'].values:
                            metric_df = perf_df[perf_df['metric_name'] == metric]
                            avg_metrics[metric] = metric_df['metric_value'].mean()
                    
                    performance_metrics[model_type] = avg_metrics
            
            # Calculate weights based on inverse loss
            if performance_metrics:
                # Use val_loss as primary metric
                inverse_losses = {}
                
                for model_type, metrics in performance_metrics.items():
                    if 'val_loss' in metrics:
                        # Inverse loss (lower loss = higher weight)
                        inverse_losses[model_type] = 1.0 / (metrics['val_loss'] + 1e-6)
                
                # Calculate new weights
                total_inverse_loss = sum(inverse_losses.values())
                
                if total_inverse_loss > 0:
                    for model_type in inverse_losses:
                        # Calculate weight
                        weight = inverse_losses[model_type] / total_inverse_loss
                        
                        # Apply minimum weight
                        self.model_weights[model_type] = max(weight, self.min_weight)
                
                    # Normalize weights
                    self._normalize_weights()
                    
                    # Save updated weights
                    self._save_model_weights()
                    
                    # Update timestamp
                    self.last_weight_update = datetime.now()
                    
                    logger.info(f"Updated model weights: {self.model_weights}")
        
        except Exception as e:
            logger.error(f"Error updating weights: {str(e)}")
    
    def evaluate_ensemble(self, data: pd.DataFrame) -> Dict:
        """
        Evaluate the ensemble performance.
        
        Args:
            data: Test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            logger.info(f"Evaluating ensemble on {len(data)} samples")
            
            # Evaluate each model
            model_metrics = {}
            
            for model_type, model in self.models.items():
                logger.info(f"Evaluating {model_type} model")
                
                # Evaluate model
                eval_result = model.evaluate(data)
                
                if eval_result.get('status') == 'success':
                    model_metrics[model_type] = eval_result.get('metrics', {})
            
            # Generate ensemble predictions
            ensemble_result = self.predict(data)
            
            if ensemble_result.get('status') != 'success':
                logger.error("Failed to generate ensemble predictions for evaluation")
                return {'status': 'error', 'message': 'Failed to generate ensemble predictions'}
            
            # Calculate ensemble metrics
            ensemble_metrics = self._calculate_ensemble_metrics(data, ensemble_result.get('predictions', []))
            
            return {
                'status': 'success',
                'ensemble_metrics': ensemble_metrics,
                'model_metrics': model_metrics
            }
        
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _calculate_ensemble_metrics(self, data: pd.DataFrame, predictions: List[Dict]) -> Dict:
        """
        Calculate metrics for ensemble predictions.
        
        Args:
            data: Test data
            predictions: Ensemble predictions
            
        Returns:
            Dictionary with ensemble metrics
        """
        try:
            # This is a simplified implementation
            # In a real system, we would compare predictions to actual future values
            
            # For now, return placeholder metrics
            return {
                'ensemble_accuracy': 0.85,  # Placeholder
                'ensemble_mae': 0.02,       # Placeholder
                'ensemble_rmse': 0.03       # Placeholder
            }
        
        except Exception as e:
            logger.error(f"Error calculating ensemble metrics: {str(e)}")
            return {}
    
    def save_model(self, model_type: str = None) -> bool:
        """
        Save a specific model or all models.
        
        Args:
            model_type: Type of model to save (None for all models)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_type:
                # Save specific model
                if model_type in self.models:
                    self.models[model_type].save_model()
                    logger.info(f"Saved {model_type} model")
                    return True
                else:
                    logger.warning(f"Model {model_type} not found")
                    return False
            else:
                # Save all models
                for model_type, model in self.models.items():
                    model.save_model()
                
                logger.info(f"Saved all models")
                return True
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_type: str = None, model_id: str = None, version: int = None) -> bool:
        """
        Load a specific model or all models.
        
        Args:
            model_type: Type of model to load (None for all models)
            model_id: Model ID to load
            version: Model version to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_type:
                # Load specific model
                if model_type in self.models:
                    result = self.models[model_type].load_model(model_id, version)
                    logger.info(f"Loaded {model_type} model: {result}")
                    return result
                else:
                    logger.warning(f"Model {model_type} not found")
                    return False
            else:
                # Load all models
                results = []
                
                for model_type, model in self.models.items():
                    results.append(model.load_model(model_id, version))
                
                logger.info(f"Loaded all models: {all(results)}")
                return all(results)
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
