"""
__init__.py file for the dashboard module.
Makes the module importable and defines the package structure.
"""

from .data_provider import DataProvider
from .api import app

__all__ = [
    'DataProvider',
    'app'
]
