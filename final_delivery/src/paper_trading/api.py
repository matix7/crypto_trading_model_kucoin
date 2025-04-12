"""
Web API for the paper trading system.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import threading

from .config import WEB_INTERFACE, PAPER_TRADING, LOGGING
from .paper_trading_engine import PaperTradingEngine

# Set up logging
os.makedirs(os.path.dirname(LOGGING['LOG_FILE']), exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOGGING['LOG_LEVEL'].upper()),
    format=LOGGING['LOG_FORMAT'],
    handlers=[
        logging.StreamHandler() if LOGGING['CONSOLE_LOG'] else logging.NullHandler(),
        logging.FileHandler(LOGGING['LOG_FILE']) if LOGGING['FILE_LOG'] else logging.NullHandler()
    ]
)
logger = logging.getLogger('paper_trading_api')

# Create FastAPI app
app = FastAPI(
    title="Crypto Trading Model API",
    description="API for the high-frequency crypto trading model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=WEB_INTERFACE['CORS_ORIGINS'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize paper trading engine
paper_trading_engine = PaperTradingEngine()

# Authentication dependency
async def authenticate(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Authenticate API request.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        True if authenticated
        
    Raises:
        HTTPException: If authentication fails
    """
    if not WEB_INTERFACE['ENABLE_AUTHENTICATION']:
        return True
    
    if WEB_INTERFACE['ENABLE_API_KEY']:
        if credentials.credentials != WEB_INTERFACE['API_KEY']:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    return True

# Request models
class StartRequest(BaseModel):
    """Request model for starting the paper trading engine."""
    pass

class StopRequest(BaseModel):
    """Request model for stopping the paper trading engine."""
    pass

class ConfigUpdateRequest(BaseModel):
    """Request model for updating configuration."""
    config: Dict

# Response models
class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    status: str
    account_balance: float
    equity: float
    open_positions: int
    total_trades: int
    total_return: float
    daily_return: float
    win_rate: float
    success_rate: float
    last_update: Optional[float]

class PositionsResponse(BaseModel):
    """Response model for positions endpoint."""
    positions: List[Dict]

class TradesResponse(BaseModel):
    """Response model for trades endpoint."""
    trades: List[Dict]

class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""
    metrics: Dict

class GenericResponse(BaseModel):
    """Generic response model."""
    status: str
    message: str

# API routes
@app.get("/", response_model=GenericResponse)
async def root():
    """Root endpoint."""
    return {
        "status": "success",
        "message": "Crypto Trading Model API is running"
    }

@app.get("/status", response_model=StatusResponse)
async def get_status(_: bool = Depends(authenticate)):
    """Get current status of the paper trading engine."""
    return paper_trading_engine.get_status()

@app.post("/start", response_model=GenericResponse)
async def start_trading(_: StartRequest, __: bool = Depends(authenticate)):
    """Start the paper trading engine."""
    if paper_trading_engine.start():
        return {
            "status": "success",
            "message": "Paper trading engine started"
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start paper trading engine"
        )

@app.post("/stop", response_model=GenericResponse)
async def stop_trading(_: StopRequest, __: bool = Depends(authenticate)):
    """Stop the paper trading engine."""
    if paper_trading_engine.stop():
        return {
            "status": "success",
            "message": "Paper trading engine stopped"
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop paper trading engine"
        )

@app.get("/positions", response_model=PositionsResponse)
async def get_positions(_: bool = Depends(authenticate)):
    """Get open positions."""
    return {
        "positions": paper_trading_engine.get_open_positions()
    }

@app.get("/trades", response_model=TradesResponse)
async def get_trades(limit: int = 100, _: bool = Depends(authenticate)):
    """Get trade history."""
    return {
        "trades": paper_trading_engine.get_trade_history(limit)
    }

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(_: bool = Depends(authenticate)):
    """Get performance metrics."""
    return {
        "metrics": paper_trading_engine.get_performance_metrics()
    }

@app.post("/config", response_model=GenericResponse)
async def update_config(request: ConfigUpdateRequest, _: bool = Depends(authenticate)):
    """Update configuration."""
    # TODO: Implement configuration update
    return {
        "status": "success",
        "message": "Configuration updated"
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "message": str(exc)}
    )

# Server startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Server startup event."""
    logger.info("API server starting")

@app.on_event("shutdown")
async def shutdown_event():
    """Server shutdown event."""
    logger.info("API server shutting down")
    if paper_trading_engine.is_running:
        paper_trading_engine.stop()

def start_api_server():
    """Start the API server."""
    uvicorn.run(
        "paper_trading.api:app",
        host=WEB_INTERFACE['HOST'],
        port=WEB_INTERFACE['PORT'],
        reload=False
    )

if __name__ == "__main__":
    start_api_server()
