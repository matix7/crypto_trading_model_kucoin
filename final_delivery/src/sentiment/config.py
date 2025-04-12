"""
Configuration settings for the sentiment analysis module.
"""

# Sentiment data sources
SENTIMENT_SOURCES = {
    'SOCIAL_MEDIA': {
        'TWITTER': {
            'enabled': True,
            'weight': 0.3,
            'keywords_per_coin': 5,
            'max_tweets_per_request': 100,
            'sentiment_update_interval': 30  # minutes
        },
        'REDDIT': {
            'enabled': True,
            'weight': 0.25,
            'subreddits': [
                'CryptoCurrency',
                'Bitcoin',
                'Ethereum',
                'CryptoMarkets',
                'altcoin',
                'SatoshiStreetBets'
            ],
            'posts_per_subreddit': 25,
            'sentiment_update_interval': 60  # minutes
        },
        'TELEGRAM': {
            'enabled': False,
            'weight': 0.1,
            'channels': [],
            'sentiment_update_interval': 60  # minutes
        }
    },
    'NEWS': {
        'CRYPTO_NEWS': {
            'enabled': True,
            'weight': 0.2,
            'sources': [
                'cointelegraph',
                'coindesk',
                'cryptonews',
                'bitcoinist',
                'newsbtc'
            ],
            'articles_per_source': 10,
            'sentiment_update_interval': 120  # minutes
        },
        'GENERAL_FINANCE': {
            'enabled': True,
            'weight': 0.15,
            'sources': [
                'bloomberg',
                'reuters',
                'wsj',
                'ft',
                'cnbc'
            ],
            'articles_per_source': 5,
            'sentiment_update_interval': 240  # minutes
        }
    },
    'ON_CHAIN': {
        'BLOCKCHAIN_METRICS': {
            'enabled': True,
            'weight': 0.1,
            'metrics': [
                'transaction_volume',
                'active_addresses',
                'new_addresses',
                'fees',
                'hash_rate'
            ],
            'sentiment_update_interval': 360  # minutes
        }
    }
}

# NLP settings
NLP_CONFIG = {
    'SENTIMENT_ANALYSIS': {
        'model': 'distilbert-base-uncased-finetuned-sst-2-english',
        'batch_size': 32,
        'use_gpu': False,
        'threshold_positive': 0.6,
        'threshold_negative': 0.4
    },
    'NAMED_ENTITY_RECOGNITION': {
        'model': 'dbmdz/bert-large-cased-finetuned-conll03-english',
        'batch_size': 16,
        'use_gpu': False
    },
    'TOPIC_MODELING': {
        'num_topics': 10,
        'update_interval': 1440  # minutes (daily)
    },
    'PREPROCESSING': {
        'remove_urls': True,
        'remove_usernames': True,
        'remove_hashtags': False,
        'remove_numbers': False,
        'remove_emojis': False,
        'lowercase': True,
        'min_token_length': 2,
        'max_token_length': 20
    }
}

# Sentiment scoring
SENTIMENT_SCORING = {
    'SCALE': {
        'VERY_NEGATIVE': -1.0,
        'NEGATIVE': -0.5,
        'NEUTRAL': 0.0,
        'POSITIVE': 0.5,
        'VERY_POSITIVE': 1.0
    },
    'WEIGHTS': {
        'text_sentiment': 0.6,
        'engagement': 0.2,
        'source_credibility': 0.2
    },
    'ENGAGEMENT_METRICS': {
        'likes': 0.3,
        'comments': 0.4,
        'shares': 0.3
    },
    'CREDIBILITY_SOURCES': {
        'HIGH': [
            'bloomberg',
            'reuters',
            'wsj',
            'ft',
            'coindesk',
            'cointelegraph'
        ],
        'MEDIUM': [
            'cryptonews',
            'bitcoinist',
            'newsbtc',
            'cnbc'
        ],
        'LOW': []
    }
}

# Keyword configuration
KEYWORDS = {
    'BTC': [
        'bitcoin',
        'btc',
        'satoshi',
        'bitcoin halving',
        'btc usd'
    ],
    'ETH': [
        'ethereum',
        'eth',
        'vitalik',
        'buterin',
        'eth usd'
    ],
    'BNB': [
        'binance coin',
        'bnb',
        'binance',
        'cz binance',
        'bnb usd'
    ],
    'SOL': [
        'solana',
        'sol',
        'solana nft',
        'solana defi',
        'sol usd'
    ],
    'ADA': [
        'cardano',
        'ada',
        'hoskinson',
        'cardano smart contracts',
        'ada usd'
    ],
    'GENERAL': [
        'crypto',
        'cryptocurrency',
        'blockchain',
        'defi',
        'nft',
        'altcoin',
        'token',
        'mining',
        'staking',
        'exchange'
    ],
    'MARKET_SENTIMENT': [
        'bull market',
        'bear market',
        'crypto crash',
        'crypto boom',
        'to the moon',
        'hodl',
        'fud',
        'fomo',
        'buy the dip',
        'sell off'
    ],
    'REGULATION': [
        'crypto regulation',
        'sec crypto',
        'crypto ban',
        'crypto tax',
        'crypto legal'
    ]
}

# Sentiment signals
SENTIMENT_SIGNALS = {
    'THRESHOLDS': {
        'VERY_BULLISH': 0.7,
        'BULLISH': 0.3,
        'NEUTRAL': -0.3,
        'BEARISH': -0.7,
        'VERY_BEARISH': -1.0
    },
    'SIGNAL_TYPES': {
        'VERY_BULLISH': 2,
        'BULLISH': 1,
        'NEUTRAL': 0,
        'BEARISH': -1,
        'VERY_BEARISH': -2
    }
}

# Time decay factors for sentiment
TIME_DECAY = {
    'RECENT': {
        'max_age': 60,  # minutes
        'weight': 1.0
    },
    'INTERMEDIATE': {
        'max_age': 360,  # minutes (6 hours)
        'weight': 0.7
    },
    'OLD': {
        'max_age': 1440,  # minutes (24 hours)
        'weight': 0.3
    },
    'VERY_OLD': {
        'max_age': 4320,  # minutes (3 days)
        'weight': 0.1
    }
}

# Database settings
DATABASE = {
    'sentiment_data_table': 'sentiment_data',
    'sentiment_scores_table': 'sentiment_scores',
    'sentiment_signals_table': 'sentiment_signals',
    'topics_table': 'sentiment_topics'
}
