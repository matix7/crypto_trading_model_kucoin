"""
Social media sentiment analyzer implementation.
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
from bs4 import BeautifulSoup
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
logger = logging.getLogger('social_media_analyzer')

class SocialMediaSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Sentiment analyzer for social media sources like Twitter and Reddit.
    """
    
    def __init__(self, platform: str, db_path: str = None):
        """
        Initialize the SocialMediaSentimentAnalyzer.
        
        Args:
            platform: Social media platform (e.g., TWITTER, REDDIT)
            db_path: Path to the SQLite database file
        """
        super().__init__(platform, 'SOCIAL_MEDIA', db_path)
        self.platform = platform
        self.config = SENTIMENT_SOURCES['SOCIAL_MEDIA'].get(platform, {})
        
        # Initialize NLP components
        self._init_nlp()
        
        logger.info(f"Initialized {platform} sentiment analyzer with config: {self.config}")
    
    def _init_nlp(self):
        """Initialize NLP components for sentiment analysis."""
        try:
            # Import transformers for sentiment analysis
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            
            # Load sentiment analysis model
            model_name = NLP_CONFIG['SENTIMENT_ANALYSIS']['model']
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Create sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1  # Use CPU
            )
            
            logger.info(f"Initialized NLP components with model: {model_name}")
        
        except Exception as e:
            logger.error(f"Error initializing NLP components: {str(e)}")
            logger.info("Falling back to simplified sentiment analysis")
            self.sentiment_pipeline = None
    
    def collect_data(self, coin: str, limit: int = 100) -> List[Dict]:
        """
        Collect social media data for a specific coin.
        
        Args:
            coin: Cryptocurrency symbol
            limit: Maximum number of data points to collect
            
        Returns:
            List of dictionaries containing social media data
        """
        if not self.config.get('enabled', False):
            logger.warning(f"{self.platform} data collection is disabled")
            return []
        
        if self.platform == 'TWITTER':
            return self._collect_twitter_data(coin, limit)
        elif self.platform == 'REDDIT':
            return self._collect_reddit_data(coin, limit)
        elif self.platform == 'TELEGRAM':
            return self._collect_telegram_data(coin, limit)
        else:
            logger.warning(f"Unsupported platform: {self.platform}")
            return []
    
    def _collect_twitter_data(self, coin: str, limit: int = 100) -> List[Dict]:
        """
        Collect Twitter data for a specific coin.
        
        Args:
            coin: Cryptocurrency symbol
            limit: Maximum number of tweets to collect
            
        Returns:
            List of dictionaries containing Twitter data
        """
        try:
            logger.info(f"Collecting Twitter data for {coin}")
            
            # In a real implementation, this would use the Twitter API
            # For this simulation, we'll generate synthetic data
            
            # Get keywords for the coin
            coin_keywords = KEYWORDS.get(coin, []) + KEYWORDS.get('GENERAL', []) + KEYWORDS.get('MARKET_SENTIMENT', [])
            
            # Generate synthetic tweets
            tweets = []
            for _ in range(min(limit, self.config.get('max_tweets_per_request', 100))):
                # Generate random timestamp within the last 24 hours
                timestamp = int((datetime.now() - timedelta(
                    hours=random.randint(0, 24),
                    minutes=random.randint(0, 60)
                )).timestamp() * 1000)
                
                # Generate random tweet content
                keyword = random.choice(coin_keywords)
                sentiment_type = random.choice(['positive', 'negative', 'neutral'])
                
                content = self._generate_synthetic_tweet(coin, keyword, sentiment_type)
                
                # Generate random engagement metrics
                likes = random.randint(0, 1000)
                retweets = random.randint(0, 200)
                replies = random.randint(0, 100)
                
                # Calculate engagement score
                engagement = {
                    'likes': likes,
                    'shares': retweets,
                    'comments': replies
                }
                
                engagement_score = self._calculate_engagement_score(engagement)
                
                # Generate random author with credibility
                author = f"crypto_user_{random.randint(1000, 9999)}"
                credibility = random.uniform(0.3, 1.0)
                
                tweet = {
                    'timestamp': timestamp,
                    'source': 'TWITTER',
                    'source_type': 'SOCIAL_MEDIA',
                    'content_id': f"tweet_{int(time.time())}_{random.randint(1000, 9999)}",
                    'content': content,
                    'author': author,
                    'url': f"https://twitter.com/{author}/status/{random.randint(1000000000000000000, 9999999999999999999)}",
                    'coin': coin,
                    'engagement': engagement,
                    'engagement_score': engagement_score,
                    'credibility_score': credibility
                }
                
                tweets.append(tweet)
            
            logger.info(f"Collected {len(tweets)} tweets for {coin}")
            return tweets
        
        except Exception as e:
            logger.error(f"Error collecting Twitter data: {str(e)}")
            return []
    
    def _collect_reddit_data(self, coin: str, limit: int = 100) -> List[Dict]:
        """
        Collect Reddit data for a specific coin.
        
        Args:
            coin: Cryptocurrency symbol
            limit: Maximum number of posts to collect
            
        Returns:
            List of dictionaries containing Reddit data
        """
        try:
            logger.info(f"Collecting Reddit data for {coin}")
            
            # In a real implementation, this would use the Reddit API
            # For this simulation, we'll generate synthetic data
            
            # Get subreddits and keywords for the coin
            subreddits = self.config.get('subreddits', [])
            coin_keywords = KEYWORDS.get(coin, []) + KEYWORDS.get('GENERAL', []) + KEYWORDS.get('MARKET_SENTIMENT', [])
            
            # Generate synthetic posts
            posts = []
            for _ in range(min(limit, len(subreddits) * self.config.get('posts_per_subreddit', 25))):
                # Generate random timestamp within the last 48 hours
                timestamp = int((datetime.now() - timedelta(
                    hours=random.randint(0, 48),
                    minutes=random.randint(0, 60)
                )).timestamp() * 1000)
                
                # Generate random post content
                subreddit = random.choice(subreddits)
                keyword = random.choice(coin_keywords)
                sentiment_type = random.choice(['positive', 'negative', 'neutral'])
                
                title = self._generate_synthetic_reddit_title(coin, keyword, sentiment_type)
                content = self._generate_synthetic_reddit_content(coin, keyword, sentiment_type)
                
                # Generate random engagement metrics
                upvotes = random.randint(0, 5000)
                downvotes = random.randint(0, 1000)
                comments = random.randint(0, 500)
                
                # Calculate engagement score
                engagement = {
                    'likes': upvotes - downvotes,
                    'shares': 0,  # Reddit doesn't have shares
                    'comments': comments
                }
                
                engagement_score = self._calculate_engagement_score(engagement)
                
                # Generate random author with credibility
                author = f"redditor_{random.randint(1000, 9999)}"
                credibility = random.uniform(0.3, 1.0)
                
                post = {
                    'timestamp': timestamp,
                    'source': 'REDDIT',
                    'source_type': 'SOCIAL_MEDIA',
                    'content_id': f"reddit_{int(time.time())}_{random.randint(1000, 9999)}",
                    'title': title,
                    'content': f"{title}\n\n{content}",
                    'author': author,
                    'subreddit': subreddit,
                    'url': f"https://reddit.com/r/{subreddit}/comments/{random.randint(100000, 999999)}/",
                    'coin': coin,
                    'engagement': engagement,
                    'engagement_score': engagement_score,
                    'credibility_score': credibility
                }
                
                posts.append(post)
            
            logger.info(f"Collected {len(posts)} Reddit posts for {coin}")
            return posts
        
        except Exception as e:
            logger.error(f"Error collecting Reddit data: {str(e)}")
            return []
    
    def _collect_telegram_data(self, coin: str, limit: int = 100) -> List[Dict]:
        """
        Collect Telegram data for a specific coin.
        
        Args:
            coin: Cryptocurrency symbol
            limit: Maximum number of messages to collect
            
        Returns:
            List of dictionaries containing Telegram data
        """
        # Telegram data collection would be implemented here
        # For now, return empty list as it's disabled in config
        logger.info(f"Telegram data collection is not implemented or disabled")
        return []
    
    def _generate_synthetic_tweet(self, coin: str, keyword: str, sentiment_type: str) -> str:
        """
        Generate synthetic tweet content for testing.
        
        Args:
            coin: Cryptocurrency symbol
            keyword: Keyword to include
            sentiment_type: Type of sentiment (positive, negative, neutral)
            
        Returns:
            Synthetic tweet content
        """
        positive_templates = [
            "Just bought more #{coin}! {keyword} looking very bullish right now. 🚀",
            "#{coin} is showing strong support at current levels. {keyword} metrics are solid. 💪",
            "{keyword} analysis suggests #{coin} could break out soon. Hodling strong! 📈",
            "The #{coin} community is growing fast! {keyword} adoption increasing daily. 🌟",
            "Technical indicators for #{coin} are all green! {keyword} momentum building. 🔥"
        ]
        
        negative_templates = [
            "Dumping my #{coin} bags. {keyword} doesn't look good at all. 📉",
            "#{coin} failing to hold support levels. {keyword} metrics deteriorating. 😞",
            "{keyword} analysis shows #{coin} in a clear downtrend. Getting out while I can. ⚠️",
            "The #{coin} project is overhyped. {keyword} fundamentals are weak. 👎",
            "Technical indicators for #{coin} are bearish. {keyword} momentum fading. 🔻"
        ]
        
        neutral_templates = [
            "Watching #{coin} closely. {keyword} could go either way from here. 🧐",
            "#{coin} trading sideways for now. Waiting for {keyword} to show direction. 📊",
            "Interesting developments with #{coin} and {keyword}. Need more data before deciding. 🤔",
            "Anyone have insights on #{coin} {keyword} metrics? Looking for analysis. 📝",
            "#{coin} volatility decreasing. {keyword} might be entering accumulation phase. 🔍"
        ]
        
        if sentiment_type == 'positive':
            template = random.choice(positive_templates)
        elif sentiment_type == 'negative':
            template = random.choice(negative_templates)
        else:
            template = random.choice(neutral_templates)
        
        return template.format(coin=coin, keyword=keyword)
    
    def _generate_synthetic_reddit_title(self, coin: str, keyword: str, sentiment_type: str) -> str:
        """
        Generate synthetic Reddit post title for testing.
        
        Args:
            coin: Cryptocurrency symbol
            keyword: Keyword to include
            sentiment_type: Type of sentiment (positive, negative, neutral)
            
        Returns:
            Synthetic Reddit post title
        """
        positive_templates = [
            "[BULLISH] {coin} showing strong {keyword} signals",
            "{coin} Technical Analysis: {keyword} indicates potential breakout",
            "Why I'm extremely bullish on {coin} - {keyword} deep dive",
            "{coin} adoption growing: {keyword} metrics at all-time high",
            "Just increased my {coin} position - {keyword} looking extremely promising"
        ]
        
        negative_templates = [
            "[BEARISH] {coin} showing concerning {keyword} signals",
            "{coin} Technical Analysis: {keyword} indicates further downside",
            "Why I've sold all my {coin} - {keyword} red flags",
            "{coin} losing momentum: {keyword} metrics declining rapidly",
            "Reducing my {coin} exposure - {keyword} outlook deteriorating"
        ]
        
        neutral_templates = [
            "[DISCUSSION] {coin} current {keyword} analysis",
            "{coin} at a crossroads: {keyword} analysis needed",
            "What's your take on {coin}'s {keyword} situation?",
            "{coin} consolidating: {keyword} metrics stable for now",
            "Seeking advice on {coin} position - {keyword} confusing signals"
        ]
        
        if sentiment_type == 'positive':
            template = random.choice(positive_templates)
        elif sentiment_type == 'negative':
            template = random.choice(negative_templates)
        else:
            template = random.choice(neutral_templates)
        
        return template.format(coin=coin, keyword=keyword)
    
    def _generate_synthetic_reddit_content(self, coin: str, keyword: str, sentiment_type: str) -> str:
        """
        Generate synthetic Reddit post content for testing.
        
        Args:
            coin: Cryptocurrency symbol
            keyword: Keyword to include
            sentiment_type: Type of sentiment (positive, negative, neutral)
            
        Returns:
            Synthetic Reddit post content
        """
        positive_content = [
            f"I've been analyzing {coin} for the past few months, and the {keyword} metrics are incredibly bullish. "
            f"The development team has been consistently delivering on their roadmap, and institutional interest is growing. "
            f"With the upcoming protocol upgrade, I expect we'll see significant price appreciation. "
            f"The technical indicators also align with this bullish outlook - RSI showing strength without being overbought, "
            f"and MACD showing a bullish crossover. I'm increasing my position by another 20% today.",
            
            f"Just wanted to share some bullish analysis on {coin}. The {keyword} data shows strong network growth, "
            f"with daily active addresses up 35% this month alone. Transaction volume is also hitting new highs, "
            f"suggesting increased adoption and usage. The recent partnership announcements will only accelerate this trend. "
            f"From a technical perspective, we've established a solid support level and volume is increasing on up days. "
            f"I believe we're in the early stages of a major bull run for {coin}."
        ]
        
        negative_content = [
            f"After careful consideration, I've decided to exit my {coin} position entirely. The {keyword} metrics "
            f"have been deteriorating for weeks now. The promised development milestones keep getting delayed, and "
            f"there's growing competition in this space from more innovative projects. "
            f"Looking at the charts, we've broken below critical support levels, and volume is increasing on down days - "
            f"a classic sign of distribution. I expect further downside from here, possibly another 30-40% drop "
            f"before finding meaningful support.",
            
            f"I need to warn the community about {coin}. The {keyword} situation is much worse than most realize. "
            f"On-chain analysis shows large holders have been consistently selling for the past month. "
            f"The development GitHub has seen minimal activity, suggesting the team may be losing interest. "
            f"Technically, we're seeing lower highs and lower lows - a clear downtrend. "
            f"The project's fundamentals no longer justify its current valuation. I've sold my entire position "
            f"and will be looking for reentry points after a substantial correction."
        ]
        
        neutral_content = [
            f"I'm trying to make sense of the current {coin} market. The {keyword} indicators are giving mixed signals. "
            f"On one hand, we're seeing increased development activity and some promising partnerships. "
            f"On the other hand, the broader market conditions remain uncertain, and some metrics suggest slowing growth. "
            f"Technically, we're trading in a range between strong support and resistance levels. "
            f"I'm maintaining my current position but not adding more until I see clearer directional signals. "
            f"Would appreciate hearing others' perspectives on this.",
            
            f"Can someone help me understand what's happening with {coin} and its {keyword} situation? "
            f"I've been holding for about 6 months now, and while I'm still in profit, the recent price action "
            f"has me concerned. The project fundamentals still seem solid, but market sentiment appears to be shifting. "
            f"I'm seeing conflicting technical analysis from different sources. Some say we're about to break out, "
            f"while others predict further consolidation or even a pullback. "
            f"I'm undecided about whether to increase my position, hold, or take some profits."
        ]
        
        if sentiment_type == 'positive':
            return random.choice(positive_content)
        elif sentiment_type == 'negative':
            return random.choice(negative_content)
        else:
            return random.choice(neutral_content)
    
    def _calculate_engagement_score(self, engagement: Dict) -> float:
        """
        Calculate engagement score based on likes, shares, and comments.
        
        Args:
            engagement: Dictionary with engagement metrics
            
        Returns:
            Engagement score between 0 and 1
        """
        try:
            # Get weights for different engagement metrics
            weights = SENTIMENT_SCORING['ENGAGEMENT_METRICS']
            
            # Calculate weighted score
            weighted_sum = (
                engagement.get('likes', 0) * weights.get('likes', 0) +
                engagement.get('shares', 0) * weights.get('shares', 0) +
                engagement.get('comments', 0) * weights.get('comments', 0)
            )
            
            # Normalize score (simple sigmoid function)
            normalized_score = 2 / (1 + np.exp(-weighted_sum / 1000)) - 1
            
            return max(0, min(1, normalized_score))
        
        except Exception as e:
            logger.error(f"Error calculating engagement score: {str(e)}")
            return 0.0
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        try:
            # Apply preprocessing steps from config
            config = NLP_CONFIG['PREPROCESSING']
            
            # Remove URLs
            if config.get('remove_urls', True):
                text = re.sub(r'http\S+', '', text)
            
            # Remove usernames
            if config.get('remove_usernames', True):
                text = re.sub(r'@\w+', '', text)
            
            # Remove hashtags
            if config.get('remove_hashtags', False):
                text = re.sub(r'#\w+', '', text)
            else:
                # Keep hashtag content but remove # symbol
                text = re.sub(r'#(\w+)', r'\1', text)
            
            # Remove numbers
            if config.get('remove_numbers', False):
                text = re.sub(r'\d+', '', text)
            
            # Remove emojis (simplified)
            if config.get('remove_emojis', False):
                text = re.sub(r'[^\w\s]', '', text)
            
            # Convert to lowercase
            if config.get('lowercase', True):
                text = text.lower()
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text
    
    def analyze_sentiment(self, data: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment from collected social media data.
        
        Args:
            data: List of dictionaries containing social media data
            
        Returns:
            List of dictionaries with sentiment analysis results
        """
        try:
            if not data:
                logger.warning("No data to analyze")
                return []
            
            results = []
            
            for item in data:
                # Get content
                content = item.get('content', '')
                
                if not content:
                    continue
                
                # Preprocess text
                preprocessed_text = self._preprocess_text(content)
                
                # Analyze sentiment
                raw_sentiment = self._analyze_text_sentiment(preprocessed_text)
                
                # Calculate credibility score
                credibility_score = item.get('credibility_score', 0.5)
                
                # Calculate engagement score if not already present
                engagement_score = item.get('engagement_score', 0.0)
                if engagement_score == 0.0 and 'engagement' in item:
                    engagement_score = self._calculate_engagement_score(item['engagement'])
                
                # Calculate processed sentiment
                weights = SENTIMENT_SCORING['WEIGHTS']
                processed_sentiment = (
                    raw_sentiment * weights['text_sentiment'] +
                    engagement_score * weights['engagement'] +
                    credibility_score * weights['source_credibility']
                )
                
                # Add sentiment analysis results to item
                result = item.copy()
                result['raw_sentiment'] = raw_sentiment
                result['processed_sentiment'] = processed_sentiment
                result['engagement_score'] = engagement_score
                result['credibility_score'] = credibility_score
                
                results.append(result)
            
            logger.info(f"Analyzed sentiment for {len(results)} items")
            return results
        
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return []
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using NLP model or fallback method.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        try:
            # Use transformer model if available
            if self.sentiment_pipeline is not None:
                # Truncate text if too long
                if len(text) > 512:
                    text = text[:512]
                
                # Get sentiment prediction
                result = self.sentiment_pipeline(text)[0]
                label = result['label']
                score = result['score']
                
                # Convert to -1 to 1 scale
                if label == 'POSITIVE':
                    return score
                elif label == 'NEGATIVE':
                    return -score
                else:
                    return 0.0
            
            # Fallback to lexicon-based approach
            else:
                return self._lexicon_based_sentiment(text)
        
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return 0.0
    
    def _lexicon_based_sentiment(self, text: str) -> float:
        """
        Simple lexicon-based sentiment analysis as fallback.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        # Simple positive and negative word lists
        positive_words = [
            'bullish', 'buy', 'uptrend', 'moon', 'rocket', 'gain', 'profit', 'growth',
            'strong', 'support', 'breakout', 'rally', 'surge', 'climb', 'rise', 'up',
            'good', 'great', 'excellent', 'amazing', 'awesome', 'positive', 'success',
            'opportunity', 'potential', 'promising', 'confident', 'hodl', 'hold',
            'accumulate', 'undervalued', 'adoption', 'partnership', 'upgrade'
        ]
        
        negative_words = [
            'bearish', 'sell', 'downtrend', 'crash', 'dump', 'loss', 'decline', 'drop',
            'weak', 'resistance', 'breakdown', 'correction', 'plunge', 'fall', 'down',
            'bad', 'terrible', 'poor', 'awful', 'negative', 'failure', 'risk',
            'threat', 'concerning', 'worried', 'uncertain', 'overvalued', 'scam',
            'fud', 'fear', 'uncertainty', 'doubt', 'problem', 'issue', 'delay'
        ]
        
        # Count positive and negative words
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Calculate sentiment score
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_count
    
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
            
            logger.info(f"Updated {self.platform} sentiment for {coin}: {sentiment_score:.2f}")
            return sentiment_score
        
        except Exception as e:
            logger.error(f"Error updating sentiment: {str(e)}")
            return 0.0
