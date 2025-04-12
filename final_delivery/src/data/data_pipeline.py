"""
Main module for data collection and preprocessing.
Integrates all components and provides a unified interface.
"""

import os
import logging
import time
import argparse
from typing import Dict, List, Optional, Tuple, Union

from .market_data_collector import MarketDataCollector
from .technical_indicator_calculator import TechnicalIndicatorCalculator
from .feature_engineering import FeatureEngineering
from .config import TRADING_PAIRS, TIMEFRAMES, DATABASE

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/home/ubuntu/crypto_trading_model/logs/data_collection.log',
    filemode='a'
)
logger = logging.getLogger('data_pipeline')

class DataPipeline:
    """
    Main class for the data collection and preprocessing pipeline.
    Integrates market data collection, technical indicator calculation, and feature engineering.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the DataPipeline.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses the path from config.
        """
        self.db_path = db_path or DATABASE['path']
        
        # Ensure logs directory exists
        os.makedirs('/home/ubuntu/crypto_trading_model/logs', exist_ok=True)
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize components
        self.market_data_collector = MarketDataCollector(db_path=self.db_path)
        self.indicator_calculator = TechnicalIndicatorCalculator(db_path=self.db_path)
        self.feature_engineering = FeatureEngineering(db_path=self.db_path)
        
        logger.info("DataPipeline initialized")
    
    def update_all_data(self):
        """Update all data, indicators, and features for all configured symbols and timeframes."""
        start_time = time.time()
        logger.info("Starting full data pipeline update")
        
        # Step 1: Update market data
        logger.info("Updating market data")
        self.market_data_collector.update_all_data()
        
        # Step 2: Update technical indicators
        logger.info("Updating technical indicators")
        self.indicator_calculator.update_all_indicators()
        
        # Step 3: Update engineered features
        logger.info("Updating engineered features")
        self.feature_engineering.update_all_features()
        
        # Step 4: Clean up old data
        logger.info("Cleaning up old data")
        self.market_data_collector.cleanup_old_data()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Full data pipeline update completed in {elapsed_time:.2f} seconds")
    
    def update_symbol_data(self, symbol: str):
        """
        Update all data, indicators, and features for a specific symbol.
        
        Args:
            symbol: Trading pair symbol
        """
        start_time = time.time()
        logger.info(f"Starting data pipeline update for {symbol}")
        
        for timeframe in TIMEFRAMES:
            # Step 1: Update market data
            logger.info(f"Updating market data for {symbol} {timeframe}")
            self.market_data_collector.update_historical_data(symbol, timeframe)
            
            # Step 2: Update technical indicators
            logger.info(f"Updating technical indicators for {symbol} {timeframe}")
            indicators_df = self.indicator_calculator.calculate_all_indicators(symbol, timeframe)
            self.indicator_calculator.save_indicators_to_db(indicators_df)
            
            # Step 3: Update engineered features
            logger.info(f"Updating engineered features for {symbol} {timeframe}")
            features_df = self.feature_engineering.engineer_all_features(symbol, timeframe)
            self.feature_engineering.save_features_to_db(features_df, symbol, timeframe)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Data pipeline update for {symbol} completed in {elapsed_time:.2f} seconds")
    
    def get_latest_data(self, symbol: str, timeframe: str, limit: int = 100) -> Dict:
        """
        Get the latest data for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            limit: Number of records to retrieve
            
        Returns:
            Dictionary with OHLCV data, indicators, and features
        """
        # Get OHLCV data
        ohlcv_df = self.market_data_collector.get_ohlcv_data(symbol, timeframe, limit=limit)
        
        # Get indicator data
        indicator_df = self.indicator_calculator.get_indicators(symbol, timeframe, limit=limit)
        
        # Get feature data
        feature_df = self.feature_engineering.get_features(symbol, timeframe, limit=limit)
        
        # Get complete dataset
        complete_df = self.feature_engineering.get_complete_dataset(symbol, timeframe, limit=limit)
        
        return {
            'ohlcv': ohlcv_df,
            'indicators': indicator_df,
            'features': feature_df,
            'complete': complete_df
        }
    
    def prepare_ml_dataset(self, symbol: str, timeframe: str, 
                          target_column: str = 'close', 
                          prediction_horizon: int = 1,
                          sequence_length: int = 10,
                          train_test_split: float = 0.8) -> Tuple:
        """
        Prepare a dataset for machine learning.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            target_column: Column to predict
            prediction_horizon: Number of steps ahead to predict
            sequence_length: Length of input sequences
            train_test_split: Proportion of data to use for training
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return self.feature_engineering.prepare_ml_dataset(
            symbol, timeframe, target_column, prediction_horizon, sequence_length, train_test_split
        )
    
    def run_scheduled_update(self, interval_minutes: int = 5):
        """
        Run scheduled updates at regular intervals.
        
        Args:
            interval_minutes: Update interval in minutes
        """
        logger.info(f"Starting scheduled updates every {interval_minutes} minutes")
        
        try:
            while True:
                self.update_all_data()
                logger.info(f"Sleeping for {interval_minutes} minutes until next update")
                time.sleep(interval_minutes * 60)
        
        except KeyboardInterrupt:
            logger.info("Scheduled updates stopped by user")
        
        except Exception as e:
            logger.error(f"Error in scheduled updates: {str(e)}")


def main():
    """Main function to run the data pipeline from command line."""
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Model Data Pipeline')
    
    parser.add_argument('--update', action='store_true', help='Update all data')
    parser.add_argument('--symbol', type=str, help='Update data for a specific symbol')
    parser.add_argument('--schedule', type=int, help='Run scheduled updates every N minutes')
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs('/home/ubuntu/crypto_trading_model/logs', exist_ok=True)
    
    # Initialize the data pipeline
    pipeline = DataPipeline()
    
    if args.update:
        pipeline.update_all_data()
    
    elif args.symbol:
        if args.symbol in TRADING_PAIRS:
            pipeline.update_symbol_data(args.symbol)
        else:
            print(f"Symbol {args.symbol} not in configured trading pairs")
            print(f"Available symbols: {', '.join(TRADING_PAIRS)}")
    
    elif args.schedule:
        pipeline.run_scheduled_update(args.schedule)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
