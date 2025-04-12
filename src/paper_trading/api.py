"""
API module for the paper trading system.
Provides REST API endpoints for interacting with the paper trading engine.
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Depends, Header, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .paper_trading_engine import PaperTradingEngine
from .config import PAPER_TRADING, EXCHANGE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('paper_trading_api')

# Initialize FastAPI app
app = FastAPI(title="Crypto Trading API", description="API for crypto trading bot")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize paper trading engine
paper_trading_engine = PaperTradingEngine()

# API key authentication
def verify_api_key(api_key: str = Header(None)):
    """
    Verify API key.
    
    Args:
        api_key: API key from header
        
    Returns:
        True if API key is valid
    """
    expected_api_key = os.getenv('API_KEY', 'default_api_key')
    
    if api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

# Models
class StatusResponse(BaseModel):
    status: str
    account_balance: float
    equity: float
    open_positions: int
    total_trades: int
    total_return: float
    daily_return: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    last_update: Optional[int] = None

class Position(BaseModel):
    position_id: str
    timestamp: int
    trading_pair: str
    side: str
    entry_price: float
    current_price: float
    quantity: float
    position_size: float
    stop_loss: float
    take_profit: float
    trailing_stop: float
    status: str
    entry_time: int
    unrealized_profit_loss: float
    unrealized_profit_loss_percentage: float

class Trade(BaseModel):
    position_id: str
    timestamp: int
    trading_pair: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    position_size: float
    profit_loss: float
    profit_loss_percentage: float
    entry_time: int
    exit_time: int
    exit_reason: str

class PerformanceMetrics(BaseModel):
    win_rate: float
    profit_factor: float
    expectancy: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    total_return: float
    max_drawdown: float

class ConfigUpdate(BaseModel):
    initial_capital: Optional[float] = None
    update_interval: Optional[int] = None
    trading_pairs: Optional[List[str]] = None
    timeframes: Optional[List[str]] = None
    max_open_positions: Optional[int] = None
    min_position_size: Optional[float] = None
    max_position_size: Optional[float] = None
    risk_per_trade: Optional[float] = None
    stop_loss_percentage: Optional[float] = None
    take_profit_percentage: Optional[float] = None
    trailing_stop_percentage: Optional[float] = None
    ta_weight: Optional[float] = None
    ml_weight: Optional[float] = None

# Routes
@app.get("/status", response_model=StatusResponse)
def get_status(api_key: str = Depends(verify_api_key)):
    """
    Get the current status of the paper trading engine.
    """
    try:
        status = paper_trading_engine.get_status()
        return status
    
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positions", response_model=List[Position])
def get_positions(api_key: str = Depends(verify_api_key)):
    """
    Get open positions.
    """
    try:
        positions = paper_trading_engine.get_open_positions()
        return positions
    
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trades", response_model=List[Trade])
def get_trades(limit: int = 100, api_key: str = Depends(verify_api_key)):
    """
    Get trade history.
    """
    try:
        trades = paper_trading_engine.get_trade_history(limit)
        return trades
    
    except Exception as e:
        logger.error(f"Error getting trades: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance", response_model=PerformanceMetrics)
def get_performance(api_key: str = Depends(verify_api_key)):
    """
    Get performance metrics.
    """
    try:
        metrics = paper_trading_engine.get_performance_metrics()
        return metrics
    
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start")
def start_trading(api_key: str = Depends(verify_api_key)):
    """
    Start the paper trading engine.
    """
    try:
        paper_trading_engine.start()
        return {"status": "started"}
    
    except Exception as e:
        logger.error(f"Error starting trading: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop")
def stop_trading(api_key: str = Depends(verify_api_key)):
    """
    Stop the paper trading engine.
    """
    try:
        paper_trading_engine.stop()
        return {"status": "stopped"}
    
    except Exception as e:
        logger.error(f"Error stopping trading: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
def reset_trading(api_key: str = Depends(verify_api_key)):
    """
    Reset the paper trading engine.
    """
    try:
        paper_trading_engine.reset()
        return {"status": "reset"}
    
    except Exception as e:
        logger.error(f"Error resetting trading: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config")
def update_config(config: ConfigUpdate, api_key: str = Depends(verify_api_key)):
    """
    Update the configuration.
    """
    try:
        # Convert to dict and remove None values
        config_dict = {k: v for k, v in config.dict().items() if v is not None}
        
        # Convert to snake_case
        config_dict = {k.upper(): v for k, v in config_dict.items()}
        
        paper_trading_engine.update_config(config_dict)
        return {"status": "config updated"}
    
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "timestamp": int(time.time() * 1000)}

# Root endpoint
@app.get("/")
def root():
    """
    Root endpoint.
    """
    return {
        "name": "Crypto Trading API",
        "version": "1.0.0",
        "description": "API for crypto trading bot",
        "endpoints": [
            "/status",
            "/positions",
            "/trades",
            "/performance",
            "/start",
            "/stop",
            "/reset",
            "/config",
            "/health"
        ]
    }
