"""
API routes for the dashboard module.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .data_provider import DataProvider
from .config import DASHBOARD

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dashboard_api')

# Create FastAPI app
app = FastAPI(
    title="Crypto Trading Dashboard API",
    description="API for the crypto trading dashboard",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize data provider
data_provider = DataProvider()

# Response models
class DashboardDataResponse(BaseModel):
    """Response model for dashboard data endpoint."""
    system_status: Dict
    open_positions: List[Dict]
    trade_history: List[Dict]
    performance_metrics: Dict
    chart_data: Dict

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
        "message": "Crypto Trading Dashboard API is running"
    }

@app.get("/dashboard-data", response_model=DashboardDataResponse)
async def get_dashboard_data():
    """Get all data needed for the dashboard."""
    try:
        data = data_provider.get_all_dashboard_data()
        
        if 'error' in data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=data['error']
            )
        
        return data
    
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/start-trading", response_model=GenericResponse)
async def start_trading():
    """Start the trading system."""
    try:
        result = data_provider.start_trading()
        
        if result.get('status') == 'error':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('message', 'Unknown error')
            )
        
        return {
            "status": "success",
            "message": "Trading started successfully"
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error starting trading: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/stop-trading", response_model=GenericResponse)
async def stop_trading():
    """Stop the trading system."""
    try:
        result = data_provider.stop_trading()
        
        if result.get('status') == 'error':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('message', 'Unknown error')
            )
        
        return {
            "status": "success",
            "message": "Trading stopped successfully"
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error stopping trading: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

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
    logger.info("Dashboard API server starting")

@app.on_event("shutdown")
async def shutdown_event():
    """Server shutdown event."""
    logger.info("Dashboard API server shutting down")
