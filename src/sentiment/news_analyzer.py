"""
News sentiment analyzer implementation.
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
logger = logging.getLogger('news_analyzer')

class NewsSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Sentiment analyzer for news sources like crypto news sites and general finance news.
    """
    
    def __init__(self, news_type: str, db_path: str = None):
        """
        Initialize the NewsSentimentAnalyzer.
        
        Args:
            news_type: Type of news source (e.g., CRYPTO_NEWS, GENERAL_FINANCE)
            db_path: Path to the SQLite database file
        """
        super().__init__(news_type, 'NEWS', db_path)
        self.news_type = news_type
        self.config = SENTIMENT_SOURCES['NEWS'].get(news_type, {})
        
        # Initialize NLP components
        self._init_nlp()
        
        logger.info(f"Initialized {news_type} sentiment analyzer with config: {self.config}")
    
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
            
            # Load named entity recognition model for extracting relevant entities
            ner_model_name = NLP_CONFIG['NAMED_ENTITY_RECOGNITION']['model']
            self.ner_pipeline = pipeline(
                "ner",
                model=ner_model_name,
                tokenizer=ner_model_name,
                device=-1  # Use CPU
            )
            
            logger.info(f"Initialized NLP components with models: {model_name}, {ner_model_name}")
        
        except Exception as e:
            logger.error(f"Error initializing NLP components: {str(e)}")
            logger.info("Falling back to simplified sentiment analysis")
            self.sentiment_pipeline = None
            self.ner_pipeline = None
    
    def collect_data(self, coin: str, limit: int = 100) -> List[Dict]:
        """
        Collect news data for a specific coin.
        
        Args:
            coin: Cryptocurrency symbol
            limit: Maximum number of articles to collect
            
        Returns:
            List of dictionaries containing news data
        """
        if not self.config.get('enabled', False):
            logger.warning(f"{self.news_type} data collection is disabled")
            return []
        
        try:
            logger.info(f"Collecting {self.news_type} data for {coin}")
            
            # In a real implementation, this would use news APIs or web scraping
            # For this simulation, we'll generate synthetic data
            
            # Get sources and keywords for the coin
            sources = self.config.get('sources', [])
            coin_keywords = KEYWORDS.get(coin, []) + KEYWORDS.get('GENERAL', []) + KEYWORDS.get('REGULATION', [])
            
            # Generate synthetic articles
            articles = []
            for _ in range(min(limit, len(sources) * self.config.get('articles_per_source', 10))):
                # Generate random timestamp within the last 72 hours
                timestamp = int((datetime.now() - timedelta(
                    hours=random.randint(0, 72),
                    minutes=random.randint(0, 60)
                )).timestamp() * 1000)
                
                # Generate random article content
                source = random.choice(sources)
                keyword = random.choice(coin_keywords)
                sentiment_type = random.choice(['positive', 'negative', 'neutral'])
                
                title = self._generate_synthetic_news_title(coin, keyword, sentiment_type)
                content = self._generate_synthetic_news_content(coin, keyword, sentiment_type)
                
                # Determine source credibility
                credibility_score = self._get_source_credibility(source)
                
                # Generate random engagement metrics
                views = random.randint(100, 10000)
                shares = random.randint(5, 500)
                comments = random.randint(0, 200)
                
                # Calculate engagement score
                engagement = {
                    'likes': views // 20,  # Approximate likes based on views
                    'shares': shares,
                    'comments': comments
                }
                
                engagement_score = self._calculate_engagement_score(engagement)
                
                # Generate random author
                author = f"{random.choice(['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily'])} {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller'])}"
                
                article = {
                    'timestamp': timestamp,
                    'source': source,
                    'source_type': 'NEWS',
                    'content_id': f"news_{int(time.time())}_{random.randint(1000, 9999)}",
                    'title': title,
                    'content': f"{title}\n\n{content}",
                    'author': author,
                    'url': f"https://{source}.com/crypto/{coin.lower()}-{keyword.replace(' ', '-')}-{random.randint(10000, 99999)}",
                    'coin': coin,
                    'engagement': engagement,
                    'engagement_score': engagement_score,
                    'credibility_score': credibility_score
                }
                
                articles.append(article)
            
            logger.info(f"Collected {len(articles)} news articles for {coin}")
            return articles
        
        except Exception as e:
            logger.error(f"Error collecting news data: {str(e)}")
            return []
    
    def _generate_synthetic_news_title(self, coin: str, keyword: str, sentiment_type: str) -> str:
        """
        Generate synthetic news article title for testing.
        
        Args:
            coin: Cryptocurrency symbol
            keyword: Keyword to include
            sentiment_type: Type of sentiment (positive, negative, neutral)
            
        Returns:
            Synthetic news article title
        """
        positive_templates = [
            "{coin} Surges as {keyword} Adoption Accelerates",
            "Bullish Outlook: {coin} Poised for Growth Due to {keyword} Developments",
            "{coin} Breaks Resistance Levels Following Positive {keyword} News",
            "Analysts Predict {coin} Rally as {keyword} Metrics Improve",
            "Major Institution Backs {coin}, Citing Strong {keyword} Fundamentals"
        ]
        
        negative_templates = [
            "{coin} Plummets Amid {keyword} Concerns",
            "Bearish Signals: {coin} Faces Pressure Due to {keyword} Issues",
            "{coin} Breaks Support Levels Following Negative {keyword} News",
            "Analysts Warn of {coin} Decline as {keyword} Metrics Deteriorate",
            "Investors Flee {coin} After Troubling {keyword} Developments"
        ]
        
        neutral_templates = [
            "{coin} Stabilizes as Market Digests {keyword} Developments",
            "{coin} Trading Sideways Despite {keyword} News",
            "Mixed Signals for {coin}: {keyword} Analysis Shows Conflicting Indicators",
            "Experts Divided on {coin}'s Future Following {keyword} Announcement",
            "{coin} Volatility Decreases as {keyword} Situation Clarifies"
        ]
        
        if sentiment_type == 'positive':
            template = random.choice(positive_templates)
        elif sentiment_type == 'negative':
            template = random.choice(negative_templates)
        else:
            template = random.choice(neutral_templates)
        
        return template.format(coin=coin, keyword=keyword)
    
    def _generate_synthetic_news_content(self, coin: str, keyword: str, sentiment_type: str) -> str:
        """
        Generate synthetic news article content for testing.
        
        Args:
            coin: Cryptocurrency symbol
            keyword: Keyword to include
            sentiment_type: Type of sentiment (positive, negative, neutral)
            
        Returns:
            Synthetic news article content
        """
        current_date = datetime.now().strftime("%B %d, %Y")
        
        positive_content = [
            f"{current_date} - {coin} has seen significant price appreciation in the past 24 hours, with analysts attributing the surge to positive developments in {keyword}. "
            f"The cryptocurrency has gained over 15% in value, outperforming the broader market by a substantial margin.\n\n"
            f"Technical analysts point to a clear breakout pattern, with {coin} successfully breaching key resistance levels that had previously capped its upward movement. "
            f"The increased trading volume accompanying this price action suggests strong conviction among buyers.\n\n"
            f"\"We're seeing institutional interest in {coin} accelerate dramatically,\" said cryptocurrency analyst Alex Thompson. \"The recent {keyword} developments have addressed "
            f"previous concerns and positioned {coin} as a leading player in the space.\"\n\n"
            f"On-chain metrics also support the bullish case, with wallet addresses holding {coin} reaching an all-time high. This accumulation pattern typically precedes "
            f"extended upward price movements, according to historical data.\n\n"
            f"Market participants are now eyeing the next resistance level, approximately 25% above current prices, as the potential target for this rally.",
            
            f"{current_date} - In a significant development for the cryptocurrency market, {coin} has secured a major partnership related to {keyword}, sending its price soaring "
            f"by over 20% in a matter of hours.\n\n"
            f"The announcement, which came early this morning, details how {coin} will be integrated into mainstream {keyword} applications, potentially exposing the cryptocurrency "
            f"to millions of new users worldwide.\n\n"
            f"\"This partnership represents a watershed moment for {coin},\" commented industry expert Maria Rodriguez. \"The {keyword} integration addresses a real-world use case "
            f"and significantly enhances {coin}'s utility proposition.\"\n\n"
            f"Trading volume for {coin} has increased by over 300% since the announcement, with several major exchanges reporting technical difficulties due to the surge in activity. "
            f"Order books show strong support at current levels, with limited selling pressure despite the rapid price appreciation.\n\n"
            f"Analysts have begun revising their price targets upward, with consensus estimates now suggesting potential for an additional 40-60% gain if the current momentum continues."
        ]
        
        negative_content = [
            f"{current_date} - {coin} has experienced a sharp decline in the past 24 hours, with prices falling by over 18% amid growing concerns related to {keyword}. "
            f"The sell-off has been broad-based, with liquidations exceeding $100 million across major exchanges.\n\n"
            f"Technical analysts note that {coin} has broken below critical support levels, potentially signaling the start of a more extended downtrend. The increased volume "
            f"on this downward move suggests strong conviction among sellers.\n\n"
            f"\"The {keyword} issues facing {coin} are more serious than many investors initially realized,\" warned cryptocurrency researcher Sarah Johnson. \"These fundamental "
            f"challenges could continue to pressure prices in the near to medium term.\"\n\n"
            f"On-chain data shows large holders reducing their positions, with several whale addresses transferring significant amounts of {coin} to exchanges - typically a "
            f"precursor to selling activity.\n\n"
            f"Market participants are now watching the next support level, approximately 15% below current prices, which could provide temporary relief if reached.",
            
            f"{current_date} - {coin} investors faced a difficult day as the cryptocurrency plummeted following negative news related to {keyword}, erasing over $2 billion in market value. "
            f"The price dropped by nearly 25% before finding tentative support.\n\n"
            f"The announcement, which emerged late yesterday, highlighted significant problems with {coin}'s approach to {keyword}, calling into question the project's long-term viability. "
            f"Several prominent developers have reportedly left the project in response to these issues.\n\n"
            f"\"This development represents a serious setback for {coin},\" stated crypto analyst James Wilson. \"The {keyword} challenges expose fundamental flaws in the project's "
            f"architecture that may be difficult to overcome without significant restructuring.\"\n\n"
            f"Trading volume has surged to yearly highs as panic selling ensued, with derivatives exchanges reporting record liquidations of long positions. Market depth has deteriorated "
            f"significantly, exacerbating price volatility.\n\n"
            f"Technical indicators have turned overwhelmingly bearish, with multiple sell signals triggered across different timeframes. Analysts suggest the correction could extend "
            f"further if the project team doesn't address the {keyword} concerns promptly and convincingly."
        ]
        
        neutral_content = [
            f"{current_date} - {coin} has been trading in a narrow range over the past week despite significant developments related to {keyword}. The cryptocurrency has shown "
            f"remarkable stability in a market known for its volatility.\n\n"
            f"Technical analysis indicates a period of consolidation, with {coin} establishing clear support and resistance levels. Trading volume has decreased during this period, "
            f"suggesting a potential accumulation phase before the next directional move.\n\n"
            f"\"The market appears to be digesting the recent {keyword} news for {coin},\" explained cryptocurrency analyst Thomas Brown. \"While the developments are significant, "
            f"investors seem to be waiting for more concrete evidence of their impact before making major moves.\"\n\n"
            f"On-chain metrics present a mixed picture, with some indicators suggesting accumulation while others point to potential distribution by larger holders. This conflicting "
            f"data may explain the current price equilibrium.\n\n"
            f"Market observers remain divided on {coin}'s near-term prospects, with price predictions ranging widely depending on how the {keyword} situation evolves in the coming weeks.",
            
            f"{current_date} - {coin} investors are facing uncertainty as conflicting signals emerge regarding the cryptocurrency's {keyword} initiatives. The price has fluctuated "
            f"within a 10% range as the market attempts to evaluate these developments.\n\n"
            f"The project team released a statement addressing the {keyword} situation, but many questions remain unanswered, leading to mixed reactions from the community and analysts. "
            f"Social media sentiment indicators show an even split between bullish and bearish outlooks.\n\n"
            f"\"We're in a wait-and-see period for {coin},\" noted industry expert Robert Chen. \"The {keyword} developments could ultimately prove positive or negative depending on "
            f"execution and market reception. This uncertainty is reflected in the current price action.\"\n\n"
            f"Trading patterns suggest large institutional players are maintaining their positions while retail activity has increased, though without a clear directional bias. "
            f"Options markets show increased demand for both calls and puts, indicating expectations of higher volatility.\n\n"
            f"Analysts recommend caution until more clarity emerges, with most suggesting investors should avoid making significant position changes until the {keyword} situation "
            f"develops further."
        ]
        
        if sentiment_type == 'positive':
            return random.choice(positive_content)
        elif sentiment_type == 'negative':
            return random.choice(negative_content)
        else:
            return random.choice(neutral_content)
    
    def _get_source_credibility(self, source: str) -> float:
        """
        Determine the credibility score of a news source.
        
        Args:
            source: News source name
            
        Returns:
            Credibility score between 0 and 1
        """
        # Check source against credibility lists
        high_credibility = SENTIMENT_SCORING['CREDIBILITY_SOURCES']['HIGH']
        medium_credibility = SENTIMENT_SCORING['CREDIBILITY_SOURCES']['MEDIUM']
        low_credibility = SENTIMENT_SCORING['CREDIBILITY_SOURCES']['LOW']
        
        if source in high_credibility:
            return random.uniform(0.8, 1.0)
        elif source in medium_credibility:
            return random.uniform(0.5, 0.8)
        elif source in low_credibility:
            return random.uniform(0.2, 0.5)
        else:
            return random.uniform(0.4, 0.7)  # Default for unknown sources
    
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
            normalized_score = 2 / (1 + np.exp(-weighted_sum / 500)) - 1
            
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
            
            # Remove numbers
            if config.get('remove_numbers', False):
                text = re.sub(r'\d+', '', text)
            
            # Convert to lowercase
            if config.get('lowercase', True):
                text = text.lower()
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text using NER model.
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of extracted entities with types
        """
        try:
            if self.ner_pipeline is None:
                return []
            
            # Truncate text if too long
            if len(text) > 1000:
                text = text[:1000]
            
            # Extract entities
            entities = self.ner_pipeline(text)
            
            # Group entities by word and type
            grouped_entities = {}
            for entity in entities:
                word = entity['word']
                entity_type = entity['entity']
                score = entity['score']
                
                if word not in grouped_entities:
                    grouped_entities[word] = {
                        'word': word,
                        'type': entity_type,
                        'score': score
                    }
                elif score > grouped_entities[word]['score']:
                    grouped_entities[word]['type'] = entity_type
                    grouped_entities[word]['score'] = score
            
            return list(grouped_entities.values())
        
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def analyze_sentiment(self, data: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment from collected news data.
        
        Args:
            data: List of dictionaries containing news data
            
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
                title = item.get('title', '')
                content = item.get('content', '')
                
                if not content:
                    continue
                
                # Preprocess text
                preprocessed_title = self._preprocess_text(title)
                preprocessed_content = self._preprocess_text(content)
                
                # Analyze sentiment
                # Title sentiment carries more weight
                title_sentiment = self._analyze_text_sentiment(preprocessed_title) * 1.5
                content_sentiment = self._analyze_text_sentiment(preprocessed_content)
                
                # Weighted average of title and content sentiment
                raw_sentiment = (title_sentiment + content_sentiment * 2) / 3
                
                # Extract entities (for more advanced analysis)
                entities = self._extract_entities(preprocessed_content)
                
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
                result['entities'] = entities
                
                results.append(result)
            
            logger.info(f"Analyzed sentiment for {len(results)} news items")
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
            'accumulate', 'undervalued', 'adoption', 'partnership', 'upgrade',
            'bullish', 'surges', 'soars', 'jumps', 'breakthrough', 'milestone',
            'revolutionary', 'innovative', 'leading', 'outperform', 'exceed',
            'beat', 'record', 'high', 'peak', 'top', 'best', 'optimistic'
        ]
        
        negative_words = [
            'bearish', 'sell', 'downtrend', 'crash', 'dump', 'loss', 'decline', 'drop',
            'weak', 'resistance', 'breakdown', 'correction', 'plunge', 'fall', 'down',
            'bad', 'terrible', 'poor', 'awful', 'negative', 'failure', 'risk',
            'threat', 'concerning', 'worried', 'uncertain', 'overvalued', 'scam',
            'fud', 'fear', 'uncertainty', 'doubt', 'problem', 'issue', 'delay',
            'bearish', 'plummets', 'tumbles', 'sinks', 'collapses', 'crisis',
            'warning', 'alert', 'danger', 'trouble', 'worrying', 'disappointing',
            'miss', 'fail', 'low', 'bottom', 'worst', 'pessimistic', 'concern'
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
            
            logger.info(f"Updated {self.news_type} sentiment for {coin}: {sentiment_score:.2f}")
            return sentiment_score
        
        except Exception as e:
            logger.error(f"Error updating sentiment: {str(e)}")
            return 0.0
