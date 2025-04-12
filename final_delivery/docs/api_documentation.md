# API Documentation: Crypto Trading Model

## Overview

This document provides detailed information about the APIs available in the Crypto Trading Model. These APIs allow you to interact with the paper trading system and the analytics dashboard programmatically.

## Base URLs

- **Paper Trading API**: `https://your-deployment-url.vercel.app/api/trading`
- **Dashboard API**: `https://your-deployment-url.vercel.app/api/dashboard`

## Authentication

Most API endpoints require authentication. There are two authentication methods available:

### Bearer Token Authentication

Include the API key in the Authorization header:

```
Authorization: Bearer your-api-key
```

### API Key Parameter

Include the API key as a query parameter:

```
?api_key=your-api-key
```

## Paper Trading API

### Status Endpoint

Get the current status of the paper trading engine.

**Endpoint**: `GET /status`

**Response**:
```json
{
  "status": "running",
  "account_balance": 12345.67,
  "equity": 12567.89,
  "open_positions": 3,
  "total_trades": 42,
  "total_return": 0.25,
  "daily_return": 0.04,
  "win_rate": 0.85,
  "success_rate": 0.92,
  "last_update": 1712345678
}
```

### Start Trading Endpoint

Start the paper trading engine.

**Endpoint**: `POST /start`

**Request Body**:
```json
{}
```

**Response**:
```json
{
  "status": "success",
  "message": "Paper trading engine started"
}
```

### Stop Trading Endpoint

Stop the paper trading engine.

**Endpoint**: `POST /stop`

**Request Body**:
```json
{}
```

**Response**:
```json
{
  "status": "success",
  "message": "Paper trading engine stopped"
}
```

### Open Positions Endpoint

Get the list of open positions.

**Endpoint**: `GET /positions`

**Response**:
```json
{
  "positions": [
    {
      "position_id": "1",
      "trading_pair": "BTCUSDT",
      "side": "BUY",
      "entry_price": 65432.10,
      "current_price": 65987.65,
      "quantity": 0.15,
      "position_size": 9814.82,
      "stop_loss": 63500.00,
      "take_profit": 68000.00,
      "trailing_stop": 65000.00,
      "unrealized_profit_loss": 83.33,
      "unrealized_profit_loss_percentage": 0.0085,
      "status": "OPEN",
      "entry_time": 1712345678000,
      "market_condition": "TRENDING_UP",
      "signal_strength": 0.75,
      "confidence": 0.85
    }
  ]
}
```

### Trade History Endpoint

Get the trade history.

**Endpoint**: `GET /trades`

**Query Parameters**:
- `limit` (optional): Maximum number of trades to return (default: 100)

**Response**:
```json
{
  "trades": [
    {
      "trade_id": "BTCUSDT_BUY_1712345678",
      "timestamp": 1712345678000,
      "trading_pair": "BTCUSDT",
      "side": "BUY",
      "entry_price": 64000.00,
      "exit_price": 65500.00,
      "quantity": 0.2,
      "position_size": 12800.00,
      "stop_loss": 62000.00,
      "take_profit": 67000.00,
      "profit_loss": 300.00,
      "profit_loss_percentage": 0.0234,
      "status": "CLOSED",
      "entry_time": 1712345678000,
      "exit_time": 1712352878000,
      "trade_duration": 7200000,
      "exit_reason": "Take Profit",
      "market_condition": "TRENDING_UP",
      "signal_strength": 0.8,
      "confidence": 0.9
    }
  ]
}
```

### Performance Metrics Endpoint

Get performance metrics.

**Endpoint**: `GET /metrics`

**Response**:
```json
{
  "metrics": {
    "total_return": 0.25,
    "daily_return": 0.04,
    "win_rate": 0.85,
    "profit_factor": 3.2,
    "average_win": 150.00,
    "average_loss": -50.00,
    "max_drawdown": 0.05,
    "sharpe_ratio": 2.1,
    "sortino_ratio": 3.2,
    "calmar_ratio": 1.8,
    "expectancy": 0.75,
    "average_holding_period": 2.5,
    "trade_count": 42,
    "winning_trades": 36,
    "losing_trades": 6,
    "consecutive_wins": 8,
    "consecutive_losses": 2,
    "largest_win": 500.00,
    "largest_loss": -100.00,
    "average_win_loss_ratio": 3.0,
    "profit_per_day": 400.00,
    "trades_per_day": 10,
    "daily_sharpe": 1.8,
    "monthly_sharpe": 2.2,
    "annual_return": 0.95,
    "volatility": 0.02,
    "success_rate": 0.92
  }
}
```

### Configuration Update Endpoint

Update the trading system configuration.

**Endpoint**: `POST /config`

**Request Body**:
```json
{
  "config": {
    "INITIAL_CAPITAL": 10000.0,
    "TRADING_PAIRS": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    "TIMEFRAMES": ["5m", "15m", "1h"],
    "MAX_OPEN_TRADES": 5,
    "RISK_PER_TRADE": 0.01,
    "STOP_LOSS_PERCENTAGE": 0.02,
    "TAKE_PROFIT_PERCENTAGE": 0.04
  }
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Configuration updated"
}
```

## Dashboard API

### Dashboard Data Endpoint

Get all data needed for the dashboard.

**Endpoint**: `GET /dashboard-data`

**Response**:
```json
{
  "system_status": {
    "status": "running",
    "account_balance": 12345.67,
    "equity": 12567.89,
    "open_positions": 3,
    "total_trades": 42,
    "total_return": 0.25,
    "daily_return": 0.04,
    "win_rate": 0.85,
    "success_rate": 0.92,
    "last_update": 1712345678
  },
  "open_positions": [
    {
      "position_id": "1",
      "trading_pair": "BTCUSDT",
      "side": "BUY",
      "entry_price": 65432.10,
      "current_price": 65987.65,
      "quantity": 0.15,
      "position_size": 9814.82,
      "unrealized_profit_loss": 83.33,
      "unrealized_profit_loss_percentage": 0.0085,
      "entry_time": 1712345678000
    }
  ],
  "trade_history": [
    {
      "trade_id": "BTCUSDT_BUY_1712345678",
      "timestamp": 1712345678000,
      "trading_pair": "BTCUSDT",
      "side": "BUY",
      "entry_price": 64000.00,
      "exit_price": 65500.00,
      "quantity": 0.2,
      "position_size": 12800.00,
      "profit_loss": 300.00,
      "profit_loss_percentage": 0.0234,
      "status": "CLOSED",
      "entry_time": 1712345678000,
      "exit_time": 1712352878000,
      "exit_reason": "Take Profit"
    }
  ],
  "performance_metrics": {
    "total_return": 0.25,
    "daily_return": 0.04,
    "win_rate": 0.85,
    "profit_factor": 3.2
  },
  "chart_data": {
    "equity_curve": [
      {
        "date": "2025-04-01",
        "equity": 10000.00
      },
      {
        "date": "2025-04-02",
        "equity": 10400.00
      }
    ],
    "daily_returns": [
      {
        "date": "2025-04-01",
        "return": 0.04
      },
      {
        "date": "2025-04-02",
        "return": 0.035
      }
    ],
    "drawdown": [
      {
        "date": "2025-04-01",
        "drawdown": 0.0
      },
      {
        "date": "2025-04-02",
        "drawdown": -0.01
      }
    ],
    "win_rate": [
      {
        "name": "Winning Trades",
        "value": 36
      },
      {
        "name": "Losing Trades",
        "value": 6
      }
    ],
    "performance_metrics_radar": [
      {
        "metric": "Win Rate",
        "value": 85
      },
      {
        "metric": "Profit Factor",
        "value": 3.2
      }
    ],
    "trading_pairs_distribution": [
      {
        "name": "BTCUSDT",
        "value": 20
      },
      {
        "name": "ETHUSDT",
        "value": 15
      }
    ]
  }
}
```

### Start Trading Endpoint

Start the trading system.

**Endpoint**: `POST /start-trading`

**Response**:
```json
{
  "status": "success",
  "message": "Trading started successfully"
}
```

### Stop Trading Endpoint

Stop the trading system.

**Endpoint**: `POST /stop-trading`

**Response**:
```json
{
  "status": "success",
  "message": "Trading stopped successfully"
}
```

## Error Handling

All API endpoints return appropriate HTTP status codes:

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication failed
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error responses include a JSON body with error details:

```json
{
  "status": "error",
  "message": "Error message details"
}
```

## Rate Limiting

API requests are rate-limited to prevent abuse. The current limits are:

- 100 requests per minute per IP address
- 1000 requests per hour per API key

When rate limits are exceeded, the API returns a `429 Too Many Requests` status code.

## Websocket API

In addition to the REST API, a WebSocket API is available for real-time updates:

**WebSocket URL**: `wss://your-deployment-url.vercel.app/ws`

### Authentication

Include the API key in the connection URL:

```
wss://your-deployment-url.vercel.app/ws?api_key=your-api-key
```

### Available Events

- `status`: System status updates
- `trade`: New trade notifications
- `position`: Position updates
- `performance`: Performance metric updates

### Example WebSocket Message

```json
{
  "event": "trade",
  "data": {
    "trade_id": "BTCUSDT_BUY_1712345678",
    "timestamp": 1712345678000,
    "trading_pair": "BTCUSDT",
    "side": "BUY",
    "entry_price": 64000.00,
    "exit_price": 65500.00,
    "profit_loss": 300.00,
    "profit_loss_percentage": 0.0234,
    "status": "CLOSED",
    "exit_reason": "Take Profit"
  }
}
```

## Conclusion

This API documentation provides the information needed to integrate with the Crypto Trading Model programmatically. For additional support or questions, please contact the development team.
