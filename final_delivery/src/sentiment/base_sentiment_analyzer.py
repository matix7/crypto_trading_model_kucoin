"""
Base sentiment analyzer class for sentiment analysis.
"""

import logging
import os
import time
import sqlite3
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from .config import (
    SENTIMENT_SCORING, TIME_DECAY, SENTIMENT_SIGNALS, DATABASE
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/sentiment_analysis.log',
    filemode='a'
)
logger = logging.getLogger('base_sentiment_analyzer')

class BaseSentimentAnalyzer(ABC):
    """
    Abstract base class for all sentiment analyzers.
    Defines the interface that all sentiment analyzer classes must implement.
    """
    
    def __init__(self, name: str, source_type: str, db_path: str = None):
        """
        Initialize the BaseSentimentAnalyzer.
        
        Args:
            name: Analyzer name
            source_type: Type of sentiment source (e.g., SOCIAL_MEDIA, NEWS, ON_CHAIN)
            db_path: Path to the SQLite database file
        """
        self.name = name
        self.source_type = source_type
        self.db_path = db_path or '/home/ubuntu/crypto_trading_model/data/market_data.db'
        
        # Ensure logs directory exists
        os.makedirs('/home/ubuntu/crypto_trading_model/logs', exist_ok=True)
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Initialized {name} sentiment analyzer of type {source_type}")
    
    def _init_database(self):
        """Initialize the SQLite database with required tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create sentiment data table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['sentiment_data_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                source TEXT NOT NULL,
                source_type TEXT NOT NULL,
                content_id TEXT NOT NULL,
                content TEXT,
                author TEXT,
                url TEXT,
                coin TEXT,
                raw_sentiment REAL,
                processed_sentiment REAL,
                engagement_score REAL,
                credibility_score REAL,
                UNIQUE(source, content_id)
            )
            ''')
            
            # Create sentiment scores table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['sentiment_scores_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                coin TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                data_points INTEGER NOT NULL,
                time_decay_factor REAL NOT NULL,
                UNIQUE(timestamp, coin, source_type, source)
            )
            ''')
            
            # Create sentiment signals table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['sentiment_signals_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                coin TEXT NOT NULL,
                overall_sentiment REAL NOT NULL,
                signal_type INTEGER NOT NULL,
                signal_strength REAL NOT NULL,
                confidence REAL NOT NULL,
                UNIQUE(timestamp, coin)
            )
            ''')
            
            # Create topics table
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE['topics_table']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                topic_id INTEGER NOT NULL,
                topic_keywords TEXT NOT NULL,
                topic_weight REAL NOT NULL,
                sentiment_impact REAL,
                UNIQUE(timestamp, topic_id)
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database tables initialized")
        
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    @abstractmethod
    def collect_data(self, coin: str, limit: int = 100) -> List[Dict]:
        """
        Collect sentiment data for a specific coin.
        
        Args:
            coin: Cryptocurrency symbol
            limit: Maximum number of data points to collect
            
        Returns:
            List of dictionaries containing sentiment data
        """
        pass
    
    @abstractmethod
    def analyze_sentiment(self, data: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment from collected data.
        
        Args:
            data: List of dictionaries containing sentiment data
            
        Returns:
            List of dictionaries with sentiment analysis results
        """
        pass
    
    def save_sentiment_data(self, sentiment_data: List[Dict]):
        """
        Save sentiment data to the database.
        
        Args:
            sentiment_data: List of dictionaries with sentiment analysis results
        """
        try:
            if not sentiment_data:
                logger.warning("No sentiment data to save")
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for data in sentiment_data:
                cursor.execute(f'''
                INSERT OR REPLACE INTO {DATABASE['sentiment_data_table']}
                (timestamp, source, source_type, content_id, content, author, url, coin, 
                 raw_sentiment, processed_sentiment, engagement_score, credibility_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.get('timestamp', int(time.time() * 1000)),
                    data.get('source', self.name),
                    data.get('source_type', self.source_type),
                    data.get('content_id', ''),
                    data.get('content', ''),
                    data.get('author', ''),
                    data.get('url', ''),
                    data.get('coin', ''),
                    data.get('raw_sentiment', 0.0),
                    data.get('processed_sentiment', 0.0),
                    data.get('engagement_score', 0.0),
                    data.get('credibility_score', 0.0)
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved {len(sentiment_data)} sentiment data points to database")
        
        except Exception as e:
            logger.error(f"Error saving sentiment data: {str(e)}")
    
    def calculate_sentiment_score(self, coin: str) -> float:
        """
        Calculate the overall sentiment score for a specific coin.
        
        Args:
            coin: Cryptocurrency symbol
            
        Returns:
            Overall sentiment score between -1 and 1
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent sentiment data for the coin from this source
            query = f'''
            SELECT timestamp, processed_sentiment, engagement_score, credibility_score
            FROM {DATABASE['sentiment_data_table']}
            WHERE coin = ? AND source_type = ? AND source = ?
            ORDER BY timestamp DESC
            LIMIT 1000
            '''
            
            df = pd.read_sql_query(query, conn, params=(coin, self.source_type, self.name))
            conn.close()
            
            if df.empty:
                logger.warning(f"No sentiment data available for {coin} from {self.name}")
                return 0.0
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate time decay factor
            now = datetime.now()
            
            def get_time_decay(dt):
                age_minutes = (now - dt).total_seconds() / 60
                
                if age_minutes <= TIME_DECAY['RECENT']['max_age']:
                    return TIME_DECAY['RECENT']['weight']
                elif age_minutes <= TIME_DECAY['INTERMEDIATE']['max_age']:
                    return TIME_DECAY['INTERMEDIATE']['weight']
                elif age_minutes <= TIME_DECAY['OLD']['max_age']:
                    return TIME_DECAY['OLD']['weight']
                elif age_minutes <= TIME_DECAY['VERY_OLD']['max_age']:
                    return TIME_DECAY['VERY_OLD']['weight']
                else:
                    return 0.0
            
            df['time_decay'] = df['datetime'].apply(get_time_decay)
            
            # Filter out entries with zero time decay
            df = df[df['time_decay'] > 0]
            
            if df.empty:
                logger.warning(f"No recent sentiment data available for {coin} from {self.name}")
                return 0.0
            
            # Calculate weighted sentiment score
            weights = SENTIMENT_SCORING['WEIGHTS']
            df['weighted_sentiment'] = (
                df['processed_sentiment'] * weights['text_sentiment'] +
                df['engagement_score'] * weights['engagement'] +
                df['credibility_score'] * weights['source_credibility']
            ) * df['time_decay']
            
            # Calculate overall score
            total_weight = df['time_decay'].sum()
            sentiment_score = df['weighted_sentiment'].sum() / total_weight if total_weight > 0 else 0.0
            
            # Save the calculated score
            self._save_sentiment_score(coin, sentiment_score, len(df), df['time_decay'].mean())
            
            logger.info(f"Calculated sentiment score {sentiment_score:.2f} for {coin} from {self.name}")
            return sentiment_score
        
        except Exception as e:
            logger.error(f"Error calculating sentiment score: {str(e)}")
            return 0.0
    
    def _save_sentiment_score(self, coin: str, score: float, data_points: int, time_decay_factor: float):
        """
        Save the calculated sentiment score to the database.
        
        Args:
            coin: Cryptocurrency symbol
            score: Sentiment score
            data_points: Number of data points used
            time_decay_factor: Average time decay factor
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
            INSERT INTO {DATABASE['sentiment_scores_table']}
            (timestamp, coin, source_type, source, sentiment_score, data_points, time_decay_factor)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                int(time.time() * 1000),
                coin,
                self.source_type,
                self.name,
                score,
                data_points,
                time_decay_factor
            ))
            
            conn.commit()
            conn.close()
            logger.debug(f"Saved sentiment score for {coin} from {self.name}")
        
        except Exception as e:
            logger.error(f"Error saving sentiment score: {str(e)}")
    
    def get_sentiment_data(self, coin: str, limit: int = 100) -> pd.DataFrame:
        """
        Get sentiment data for a specific coin.
        
        Args:
            coin: Cryptocurrency symbol
            limit: Maximum number of data points to retrieve
            
        Returns:
            DataFrame with sentiment data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = f'''
            SELECT *
            FROM {DATABASE['sentiment_data_table']}
            WHERE coin = ? AND source_type = ? AND source = ?
            ORDER BY timestamp DESC
            LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(coin, self.source_type, self.name, limit))
            conn.close()
            
            return df
        
        except Exception as e:
            logger.error(f"Error retrieving sentiment data: {str(e)}")
            return pd.DataFrame()
    
    def get_sentiment_score_history(self, coin: str, days: int = 7) -> pd.DataFrame:
        """
        Get historical sentiment scores for a specific coin.
        
        Args:
            coin: Cryptocurrency symbol
            days: Number of days of history to retrieve
            
        Returns:
            DataFrame with historical sentiment scores
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate timestamp for 'days' ago
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            query = f'''
            SELECT timestamp, sentiment_score, data_points, time_decay_factor
            FROM {DATABASE['sentiment_scores_table']}
            WHERE coin = ? AND source_type = ? AND source = ? AND timestamp >= ?
            ORDER BY timestamp ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=(coin, self.source_type, self.name, start_time))
            conn.close()
            
            # Convert timestamp to datetime
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        
        except Exception as e:
            logger.error(f"Error retrieving sentiment score history: {str(e)}")
            return pd.DataFrame()
    
    def __str__(self) -> str:
        """String representation of the sentiment analyzer."""
        return f"{self.name} ({self.source_type})"
    
    def __repr__(self) -> str:
        """Detailed representation of the sentiment analyzer."""
        return f"SentimentAnalyzer(name='{self.name}', type='{self.source_type}')"
