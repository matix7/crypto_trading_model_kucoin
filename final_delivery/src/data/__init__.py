"""
__init__.py file for the data module.
Makes the module importable and defines the package structure.
"""

from .market_data_collector import MarketDataCollector
from .technical_indicator_calculator import TechnicalIndicatorCalculator
from .feature_engineering import FeatureEngineering
from .data_pipeline import DataPipeline

__all__ = [
    'MarketDataCollector',
    'TechnicalIndicatorCalculator',
    'FeatureEngineering',
    'DataPipeline'
]
