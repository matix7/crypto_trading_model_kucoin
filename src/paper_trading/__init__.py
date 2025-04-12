"""
__init__.py file for the paper trading module.
Makes the module importable and defines the package structure.
"""

from .paper_trading_engine import PaperTradingEngine
from .api import app, start_api_server

__all__ = [
    'PaperTradingEngine',
    'app',
    'start_api_server'
]
