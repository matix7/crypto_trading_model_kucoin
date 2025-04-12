"""
__init__.py file for the analysis module.
Makes the module importable and defines the package structure.
"""

from .base_strategy import BaseStrategy
from .trend_following_strategy import TrendFollowingStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .breakout_strategy import BreakoutStrategy
from .strategy_manager import StrategyManager

__all__ = [
    'BaseStrategy',
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'BreakoutStrategy',
    'StrategyManager'
]
