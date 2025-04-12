"""
On-chain data sentiment analyzer implementation.
"""

import logging
import os
import time
import json
import re
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import requests
import random

from .base_sentiment_analyzer import BaseSentimentAnalyzer
from .config import (
    SENTIMENT_SOURCES, NLP_CONFIG, SENTIMENT_SCORING, KEYWORDS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/sentiment_analysis.log',
    filemode='a'
)
logger = logging.getLogger('on_chain_analyzer')

class OnChainSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Sentiment analyzer for on-chain blockchain metrics.
    """
    
    def __init__(self, metrics_type: str = 'BLOCKCHAIN_METRICS', db_path: str = None):
        """
        Initialize the OnChainSentimentAnalyzer.
        
        Args:
            metrics_type: Type of on-chain metrics
            db_path: Path to the SQLite database file
        """
        super().__init__(metrics_type, 'ON_CHAIN', db_path)
        self.metrics_type = metrics_type
        self.config = SENTIMENT_SOURCES['ON_CHAIN'].get(metrics_type, {})
        
        logger.info(f"Initialized {metrics_type} sentiment analyzer with config: {self.config}")
    
    def collect_data(self, coin: str, limit: int = 100) -> List[Dict]:
        """
        Collect on-chain data for a specific coin.
        
        Args:
            coin: Cryptocurrency symbol
            limit: Maximum number of data points to collect
            
        Returns:
            List of dictionaries containing on-chain data
        """
        if not self.config.get('enabled', False):
            logger.warning(f"{self.metrics_type} data collection is disabled")
            return []
        
        try:
            logger.info(f"Collecting {self.metrics_type} data for {coin}")
            
            # In a real implementation, this would use blockchain APIs
            # For this simulation, we'll generate synthetic data
            
            # Get metrics to collect
            metrics = self.config.get('metrics', [])
            
            # Generate synthetic on-chain data
            data_points = []
            for _ in range(min(limit, 24)):  # 24 hours of hourly data
                # Generate random timestamp within the last 24 hours
                timestamp = int((datetime.now() - timedelta(
                    hours=random.randint(0, 24),
                    minutes=random.randint(0, 60)
                )).timestamp() * 1000)
                
                # Generate random metric values
                metric_values = {}
                for metric in metrics:
                    # Generate realistic values for each metric
                    if metric == 'transaction_volume':
                        # Transaction volume in millions
                        metric_values[metric] = random.uniform(50, 500)
                    elif metric == 'active_addresses':
                        # Active addresses in thousands
                        metric_values[metric] = random.uniform(10, 100)
                    elif metric == 'new_addresses':
                        # New addresses in thousands
                        metric_values[metric] = random.uniform(1, 20)
                    elif metric == 'fees':
                        # Average fees in USD
                        metric_values[metric] = random.uniform(0.1, 10)
                    elif metric == 'hash_rate':
                        # Hash rate (for PoW coins) in arbitrary units
                        if coin in ['BTC', 'ETH']:
                            metric_values[metric] = random.uniform(100, 200)
                        else:
                            metric_values[metric] = None
                    else:
                        # Generic metric
                        metric_values[metric] = random.uniform(0, 100)
                
                # Calculate a base sentiment from the metrics
                sentiment_indicators = self._calculate_metric_sentiment_indicators(coin, metric_values)
                
                data_point = {
                    'timestamp': timestamp,
                    'source': self.metrics_type,
                    'source_type': 'ON_CHAIN',
                    'content_id': f"onchain_{coin}_{int(time.time())}_{random.randint(1000, 9999)}",
                    'content': json.dumps(metric_values),
                    'coin': coin,
                    'metric_values': metric_values,
                    'sentiment_indicators': sentiment_indicators,
                    'credibility_score': 0.9  # On-chain data is highly credible
                }
                
                data_points.append(data_point)
            
            logger.info(f"Collected {len(data_points)} on-chain data points for {coin}")
            return data_points
        
        except Exception as e:
            logger.error(f"Error collecting on-chain data: {str(e)}")
            return []
    
    def _calculate_metric_sentiment_indicators(self, coin: str, metric_values: Dict) -> Dict:
        """
        Calculate sentiment indicators from on-chain metrics.
        
        Args:
            coin: Cryptocurrency symbol
            metric_values: Dictionary of metric values
            
        Returns:
            Dictionary of sentiment indicators
        """
        indicators = {}
        
        # Transaction volume sentiment
        if 'transaction_volume' in metric_values and metric_values['transaction_volume'] is not None:
            volume = metric_values['transaction_volume']
            if volume > 300:  # High volume
                indicators['volume_sentiment'] = random.uniform(0.6, 1.0)
            elif volume > 150:  # Medium volume
                indicators['volume_sentiment'] = random.uniform(0.3, 0.7)
            else:  # Low volume
                indicators['volume_sentiment'] = random.uniform(-0.3, 0.4)
        
        # Active addresses sentiment
        if 'active_addresses' in metric_values and metric_values['active_addresses'] is not None:
            active = metric_values['active_addresses']
            if active > 70:  # High activity
                indicators['activity_sentiment'] = random.uniform(0.6, 1.0)
            elif active > 30:  # Medium activity
                indicators['activity_sentiment'] = random.uniform(0.3, 0.7)
            else:  # Low activity
                indicators['activity_sentiment'] = random.uniform(-0.3, 0.4)
        
        # New addresses sentiment
        if 'new_addresses' in metric_values and metric_values['new_addresses'] is not None:
            new_addr = metric_values['new_addresses']
            if new_addr > 15:  # High growth
                indicators['growth_sentiment'] = random.uniform(0.6, 1.0)
            elif new_addr > 5:  # Medium growth
                indicators['growth_sentiment'] = random.uniform(0.3, 0.7)
            else:  # Low growth
                indicators['growth_sentiment'] = random.uniform(-0.3, 0.4)
        
        # Fees sentiment (high fees can be negative for sentiment)
        if 'fees' in metric_values and metric_values['fees'] is not None:
            fees = metric_values['fees']
            if fees > 5:  # High fees
                indicators['fee_sentiment'] = random.uniform(-1.0, -0.3)
            elif fees > 1:  # Medium fees
                indicators['fee_sentiment'] = random.uniform(-0.5, 0.2)
            else:  # Low fees
                indicators['fee_sentiment'] = random.uniform(0.0, 0.7)
        
        # Hash rate sentiment (higher is better for security)
        if 'hash_rate' in metric_values and metric_values['hash_rate'] is not None:
            hash_rate = metric_values['hash_rate']
            if hash_rate > 150:  # High hash rate
                indicators['security_sentiment'] = random.uniform(0.6, 1.0)
            elif hash_rate > 120:  # Medium hash rate
                indicators['security_sentiment'] = random.uniform(0.3, 0.7)
            else:  # Low hash rate
                indicators['security_sentiment'] = random.uniform(-0.3, 0.4)
        
        return indicators
    
    def analyze_sentiment(self, data: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment from collected on-chain data.
        
        Args:
            data: List of dictionaries containing on-chain data
            
        Returns:
            List of dictionaries with sentiment analysis results
        """
        try:
            if not data:
                logger.warning("No data to analyze")
                return []
            
            results = []
            
            for item in data:
                # Get sentiment indicators
                sentiment_indicators = item.get('sentiment_indicators', {})
                
                if not sentiment_indicators:
                    continue
                
                # Calculate raw sentiment as average of all indicators
                indicator_values = [v for v in sentiment_indicators.values() if v is not None]
                raw_sentiment = sum(indicator_values) / len(indicator_values) if indicator_values else 0.0
                
                # Get credibility score
                credibility_score = item.get('credibility_score', 0.9)  # On-chain data is highly credible
                
                # Calculate processed sentiment
                # On-chain data doesn't have engagement metrics, so we weight text sentiment higher
                weights = {
                    'text_sentiment': 0.8,
                    'engagement': 0.0,
                    'source_credibility': 0.2
                }
                
                processed_sentiment = (
                    raw_sentiment * weights['text_sentiment'] +
                    credibility_score * weights['source_credibility']
                )
                
                # Add sentiment analysis results to item
                result = item.copy()
                result['raw_sentiment'] = raw_sentiment
                result['processed_sentiment'] = processed_sentiment
                result['engagement_score'] = 0.0  # No engagement for on-chain data
                result['credibility_score'] = credibility_score
                
                results.append(result)
            
            logger.info(f"Analyzed sentiment for {len(results)} on-chain data points")
            return results
        
        except Exception as e:
            logger.error(f"Error analyzing on-chain sentiment: {str(e)}")
            return []
    
    def update_sentiment(self, coin: str):
        """
        Update sentiment data and scores for a specific coin.
        
        Args:
            coin: Cryptocurrency symbol
        """
        try:
            # Collect new data
            data = self.collect_data(coin)
            
            # Analyze sentiment
            sentiment_data = self.analyze_sentiment(data)
            
            # Save to database
            self.save_sentiment_data(sentiment_data)
            
            # Calculate overall sentiment score
            sentiment_score = self.calculate_sentiment_score(coin)
            
            logger.info(f"Updated {self.metrics_type} sentiment for {coin}: {sentiment_score:.2f}")
            return sentiment_score
        
        except Exception as e:
            logger.error(f"Error updating on-chain sentiment: {str(e)}")
            return 0.0
