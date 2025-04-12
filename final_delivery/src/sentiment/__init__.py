"""
__init__.py file for the sentiment module.
Makes the module importable and defines the package structure.
"""

from .base_sentiment_analyzer import BaseSentimentAnalyzer
from .social_media_analyzer import SocialMediaSentimentAnalyzer
from .news_analyzer import NewsSentimentAnalyzer
from .on_chain_analyzer import OnChainSentimentAnalyzer
from .sentiment_manager import SentimentManager

__all__ = [
    'BaseSentimentAnalyzer',
    'SocialMediaSentimentAnalyzer',
    'NewsSentimentAnalyzer',
    'OnChainSentimentAnalyzer',
    'SentimentManager'
]
