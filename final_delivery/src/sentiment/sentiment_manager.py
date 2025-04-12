"""
Sentiment manager for combining and managing multiple sentiment analyzers.
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

from .base_sentiment_analyzer import BaseSentimentAnalyzer
from .social_media_analyzer import SocialMediaSentimentAnalyzer
from .news_analyzer import NewsSentimentAnalyzer
from .on_chain_analyzer import OnChainSentimentAnalyzer
from .config import (
    SENTIMENT_SOURCES, SENTIMENT_SIGNALS, DATABASE, KEYWORDS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/sentiment_analysis.log',
    filemode='a'
)
logger = logging.getLogger('sentiment_manager')

class SentimentManager:
    """
    Manager class for combining and managing multiple sentiment analyzers.
    Handles analyzer initialization, sentiment aggregation, and signal generation.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the SentimentManager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path or '/home/ubuntu/crypto_trading_model/data/market_data.db'
        self.analyzers = {}
        
        # Ensure logs directory exists
        os.makedirs('/home/ubuntu/crypto_trading_model/logs', exist_ok=True)
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize analyzers
        self._init_analyzers()
        
        logger.info("SentimentManager initialized")
    
    def _init_analyzers(self):
        """Initialize all sentiment analyzers."""
        try:
            # Initialize social media analyzers
            for platform, config in SENTIMENT_SOURCES['SOCIAL_MEDIA'].items():
                if config.get('enabled', False):
                    self.analyzers[f"SOCIAL_MEDIA_{platform}"] = SocialMediaSentimentAnalyzer(
                        platform=platform,
                        db_path=self.db_path
                    )
            
            # Initialize news analyzers
            for news_type, config in SENTIMENT_SOURCES['NEWS'].items():
                if config.get('enabled', False):
                    self.analyzers[f"NEWS_{news_type}"] = NewsSentimentAnalyzer(
                        news_type=news_type,
                        db_path=self.db_path
                    )
            
            # Initialize on-chain analyzers
            for metrics_type, config in SENTIMENT_SOURCES['ON_CHAIN'].items():
                if config.get('enabled', False):
                    self.analyzers[f"ON_CHAIN_{metrics_type}"] = OnChainSentimentAnalyzer(
                        metrics_type=metrics_type,
                        db_path=self.db_path
                    )
            
            logger.info(f"Initialized {len(self.analyzers)} sentiment analyzers")
        
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzers: {str(e)}")
    
    def update_all_sentiment(self, coin: str):
        """
        Update sentiment data and scores from all analyzers for a specific coin.
        
        Args:
            coin: Cryptocurrency symbol
            
        Returns:
            Dictionary with sentiment scores from all analyzers
        """
        try:
            sentiment_scores = {}
            
            for analyzer_name, analyzer in self.analyzers.items():
                sentiment_score = analyzer.update_sentiment(coin)
                sentiment_scores[analyzer_name] = sentiment_score
            
            logger.info(f"Updated all sentiment for {coin}")
            return sentiment_scores
        
        except Exception as e:
            logger.error(f"Error updating all sentiment: {str(e)}")
            return {}
    
    def calculate_overall_sentiment(self, coin: str) -> float:
        """
        Calculate the overall sentiment score for a specific coin.
        
        Args:
            coin: Cryptocurrency symbol
            
        Returns:
            Overall sentiment score between -1 and 1
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent sentiment scores for the coin from all sources
            query = f'''
            SELECT source_type, source, sentiment_score, data_points, time_decay_factor
            FROM {DATABASE['sentiment_scores_table']}
            WHERE coin = ? AND timestamp >= ?
            ORDER BY timestamp DESC
            '''
            
            # Get scores from the last 24 hours
            start_time = int((datetime.now() - timedelta(hours=24)).timestamp() * 1000)
            
            df = pd.read_sql_query(query, conn, params=(coin, start_time))
            conn.close()
            
            if df.empty:
                logger.warning(f"No sentiment scores available for {coin}")
                return 0.0
            
            # Group by source_type and source, taking the most recent score for each
            df = df.sort_values('timestamp', ascending=False)
            df = df.drop_duplicates(subset=['source_type', 'source'])
            
            # Calculate weighted sentiment score
            total_weight = 0.0
            weighted_sum = 0.0
            
            for _, row in df.iterrows():
                source_type = row['source_type']
                source = row['source']
                score = row['sentiment_score']
                data_points = row['data_points']
                time_decay = row['time_decay_factor']
                
                # Get weight for this source
                weight = self._get_source_weight(source_type, source)
                
                # Adjust weight based on data points and time decay
                adjusted_weight = weight * min(1.0, data_points / 10) * time_decay
                
                weighted_sum += score * adjusted_weight
                total_weight += adjusted_weight
            
            # Calculate overall score
            overall_sentiment = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            # Save the overall sentiment
            self._save_overall_sentiment(coin, overall_sentiment)
            
            logger.info(f"Calculated overall sentiment {overall_sentiment:.2f} for {coin}")
            return overall_sentiment
        
        except Exception as e:
            logger.error(f"Error calculating overall sentiment: {str(e)}")
            return 0.0
    
    def _get_source_weight(self, source_type: str, source: str) -> float:
        """
        Get the weight for a specific sentiment source.
        
        Args:
            source_type: Type of sentiment source (e.g., SOCIAL_MEDIA, NEWS, ON_CHAIN)
            source: Specific source name
            
        Returns:
            Weight for the source
        """
        try:
            # Get weight from configuration
            if source_type == 'SOCIAL_MEDIA':
                return SENTIMENT_SOURCES['SOCIAL_MEDIA'].get(source, {}).get('weight', 0.1)
            elif source_type == 'NEWS':
                return SENTIMENT_SOURCES['NEWS'].get(source, {}).get('weight', 0.1)
            elif source_type == 'ON_CHAIN':
                return SENTIMENT_SOURCES['ON_CHAIN'].get(source, {}).get('weight', 0.1)
            else:
                return 0.1
        
        except Exception as e:
            logger.error(f"Error getting source weight: {str(e)}")
            return 0.1
    
    def _save_overall_sentiment(self, coin: str, sentiment_score: float):
        """
        Save the overall sentiment score and generate a signal.
        
        Args:
            coin: Cryptocurrency symbol
            sentiment_score: Overall sentiment score
        """
        try:
            # Determine signal type based on sentiment score
            signal_type = self._get_signal_type(sentiment_score)
            
            # Calculate signal strength and confidence
            signal_strength = abs(sentiment_score)
            confidence = min(1.0, signal_strength * 1.5)  # Higher strength = higher confidence
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
            INSERT INTO {DATABASE['sentiment_signals_table']}
            (timestamp, coin, overall_sentiment, signal_type, signal_strength, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                int(time.time() * 1000),
                coin,
                sentiment_score,
                signal_type,
                signal_strength,
                confidence
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved overall sentiment and signal for {coin}")
        
        except Exception as e:
            logger.error(f"Error saving overall sentiment: {str(e)}")
    
    def _get_signal_type(self, sentiment_score: float) -> int:
        """
        Determine the signal type based on sentiment score.
        
        Args:
            sentiment_score: Overall sentiment score
            
        Returns:
            Signal type as integer
        """
        thresholds = SENTIMENT_SIGNALS['THRESHOLDS']
        signal_types = SENTIMENT_SIGNALS['SIGNAL_TYPES']
        
        if sentiment_score >= thresholds['VERY_BULLISH']:
            return signal_types['VERY_BULLISH']
        elif sentiment_score >= thresholds['BULLISH']:
            return signal_types['BULLISH']
        elif sentiment_score <= thresholds['VERY_BEARISH']:
            return signal_types['VERY_BEARISH']
        elif sentiment_score <= thresholds['BEARISH']:
            return signal_types['BEARISH']
        else:
            return signal_types['NEUTRAL']
    
    def get_sentiment_signal(self, coin: str) -> Dict:
        """
        Get the latest sentiment signal for a specific coin.
        
        Args:
            coin: Cryptocurrency symbol
            
        Returns:
            Dictionary with sentiment signal information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = f'''
            SELECT timestamp, overall_sentiment, signal_type, signal_strength, confidence
            FROM {DATABASE['sentiment_signals_table']}
            WHERE coin = ?
            ORDER BY timestamp DESC
            LIMIT 1
            '''
            
            cursor = conn.cursor()
            cursor.execute(query, (coin,))
            result = cursor.fetchone()
            conn.close()
            
            if result is None:
                logger.warning(f"No sentiment signal available for {coin}")
                return {
                    'coin': coin,
                    'timestamp': int(time.time() * 1000),
                    'overall_sentiment': 0.0,
                    'signal_type': SENTIMENT_SIGNALS['SIGNAL_TYPES']['NEUTRAL'],
                    'signal_strength': 0.0,
                    'confidence': 0.0,
                    'signal_name': 'NEUTRAL'
                }
            
            timestamp, overall_sentiment, signal_type, signal_strength, confidence = result
            
            # Get signal name
            signal_name = 'NEUTRAL'
            for name, value in SENTIMENT_SIGNALS['SIGNAL_TYPES'].items():
                if value == signal_type:
                    signal_name = name
                    break
            
            return {
                'coin': coin,
                'timestamp': timestamp,
                'overall_sentiment': overall_sentiment,
                'signal_type': signal_type,
                'signal_strength': signal_strength,
                'confidence': confidence,
                'signal_name': signal_name
            }
        
        except Exception as e:
            logger.error(f"Error getting sentiment signal: {str(e)}")
            return {
                'coin': coin,
                'timestamp': int(time.time() * 1000),
                'overall_sentiment': 0.0,
                'signal_type': SENTIMENT_SIGNALS['SIGNAL_TYPES']['NEUTRAL'],
                'signal_strength': 0.0,
                'confidence': 0.0,
                'signal_name': 'NEUTRAL'
            }
    
    def get_sentiment_history(self, coin: str, days: int = 7) -> pd.DataFrame:
        """
        Get historical sentiment signals for a specific coin.
        
        Args:
            coin: Cryptocurrency symbol
            days: Number of days of history to retrieve
            
        Returns:
            DataFrame with historical sentiment signals
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate timestamp for 'days' ago
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            query = f'''
            SELECT timestamp, overall_sentiment, signal_type, signal_strength, confidence
            FROM {DATABASE['sentiment_signals_table']}
            WHERE coin = ? AND timestamp >= ?
            ORDER BY timestamp ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=(coin, start_time))
            conn.close()
            
            # Convert timestamp to datetime
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Add signal name
                df['signal_name'] = df['signal_type'].apply(self._get_signal_name)
            
            return df
        
        except Exception as e:
            logger.error(f"Error retrieving sentiment history: {str(e)}")
            return pd.DataFrame()
    
    def _get_signal_name(self, signal_type: int) -> str:
        """
        Get the name of a signal type.
        
        Args:
            signal_type: Signal type as integer
            
        Returns:
            Signal name as string
        """
        for name, value in SENTIMENT_SIGNALS['SIGNAL_TYPES'].items():
            if value == signal_type:
                return name
        return 'NEUTRAL'
    
    def update_sentiment_for_all_coins(self):
        """Update sentiment for all configured coins."""
        try:
            # Get list of coins from keywords
            coins = [coin for coin in KEYWORDS.keys() if coin != 'GENERAL' and coin != 'MARKET_SENTIMENT' and coin != 'REGULATION']
            
            for coin in coins:
                logger.info(f"Updating sentiment for {coin}")
                
                # Update sentiment from all analyzers
                self.update_all_sentiment(coin)
                
                # Calculate overall sentiment
                self.calculate_overall_sentiment(coin)
            
            logger.info(f"Updated sentiment for {len(coins)} coins")
        
        except Exception as e:
            logger.error(f"Error updating sentiment for all coins: {str(e)}")
    
    def run_scheduled_update(self, interval_minutes: int = 30):
        """
        Run scheduled sentiment updates at regular intervals.
        
        Args:
            interval_minutes: Update interval in minutes
        """
        logger.info(f"Starting scheduled sentiment updates every {interval_minutes} minutes")
        
        try:
            while True:
                self.update_sentiment_for_all_coins()
                logger.info(f"Sleeping for {interval_minutes} minutes until next update")
                time.sleep(interval_minutes * 60)
        
        except KeyboardInterrupt:
            logger.info("Scheduled sentiment updates stopped by user")
        
        except Exception as e:
            logger.error(f"Error in scheduled sentiment updates: {str(e)}")


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('/home/ubuntu/crypto_trading_model/logs', exist_ok=True)
    
    # Initialize sentiment manager
    manager = SentimentManager()
    
    # Update sentiment for BTC
    manager.update_all_sentiment('BTC')
    
    # Calculate overall sentiment
    sentiment = manager.calculate_overall_sentiment('BTC')
    print(f"BTC overall sentiment: {sentiment:.2f}")
    
    # Get sentiment signal
    signal = manager.get_sentiment_signal('BTC')
    print(f"BTC sentiment signal: {signal['signal_name']} (strength: {signal['signal_strength']:.2f}, confidence: {signal['confidence']:.2f})")
