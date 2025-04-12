# KuCoin Crypto Trading Model API Documentation

This document provides detailed information about the API endpoints available for interacting with the KuCoin Crypto Trading Model.

## Base URL

When deployed to Vercel, the base URL will be your deployment URL (e.g., `https://crypto-trading-bot-kucoin.vercel.app/api`).

For local development, the base URL is `http://localhost:8000/api`.

## Authentication

All API endpoints (except `/health` and `/`) require authentication using an API key. The API key should be included in the request headers:

```
X-API-Key: your_api_key
```

The API key is set in the environment variables during deployment.

## Endpoints

### Status

#### GET /status

Get the current status of the trading system.

**Response:**

```json
{
  "status": "running",
  "account_balance": 10250.75,
  "equity": 10320.50,
  "open_positions": 2,
  "total_trades": 15,
  "total_return": 0.0325,
  "daily_return": 0.0075,
  "win_rate": 0.8,
  "profit_factor": 2.5,
  "max_drawdown": 0.05,
  "last_update": 1649234567890
}
```

### Positions

#### GET /positions

Get all open positions.

**Response:**

```json
[
  {
    "position_id": "BTC-USDT_BUY_1649234567890",
    "timestamp": 1649234567890,
    "trading_pair": "BTC-USDT",
    "side": "BUY",
    "entry_price": 45000.0,
    "current_price": 45500.0,
    "quantity": 0.01,
    "position_size": 450.0,
    "stop_loss": 44100.0,
    "take_profit": 46800.0,
    "trailing_stop": 44500.0,
    "status": "OPEN",
    "entry_time": 1649234567890,
    "unrealized_profit_loss": 5.0,
    "unrealized_profit_loss_percentage": 0.0111
  }
]
```

### Trades

#### GET /trades

Get historical trades.

**Query Parameters:**

- `limit` (optional): Maximum number of trades to return (default: 100)

**Response:**

```json
[
  {
    "position_id": "ETH-USDT_BUY_1649134567890",
    "timestamp": 1649134567890,
    "trading_pair": "ETH-USDT",
    "side": "BUY",
    "entry_price": 3200.0,
    "exit_price": 3300.0,
    "quantity": 0.05,
    "position_size": 160.0,
    "profit_loss": 5.0,
    "profit_loss_percentage": 0.03125,
    "entry_time": 1649134567890,
    "exit_time": 1649144567890,
    "exit_reason": "Take Profit"
  }
]
```

### Performance

#### GET /performance

Get performance metrics.

**Response:**

```json
{
  "win_rate": 0.8,
  "profit_factor": 2.5,
  "expectancy": 1.2,
  "average_win": 15.0,
  "average_loss": -5.0,
  "largest_win": 50.0,
  "largest_loss": -20.0,
  "total_return": 0.0325,
  "max_drawdown": 0.05
}
```

### Trading Control

#### POST /start

Start the trading system.

**Response:**

```json
{
  "status": "started"
}
```

#### POST /stop

Stop the trading system.

**Response:**

```json
{
  "status": "stopped"
}
```

#### POST /reset

Reset the trading system (clears all data and starts fresh).

**Response:**

```json
{
  "status": "reset"
}
```

### Configuration

#### POST /config

Update the trading system configuration.

**Request Body:**

```json
{
  "initial_capital": 10000,
  "update_interval": 60,
  "trading_pairs": ["BTC-USDT", "ETH-USDT"],
  "timeframes": ["5m", "15m", "1h"],
  "max_open_positions": 5,
  "risk_per_trade": 0.02,
  "stop_loss_percentage": 0.02,
  "take_profit_percentage": 0.04
}
```

**Response:**

```json
{
  "status": "config updated"
}
```

### Health Check

#### GET /health

Check if the API is running.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": 1649234567890
}
```

### Root

#### GET /

Get API information.

**Response:**

```json
{
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
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- 200: Success
- 400: Bad request (invalid parameters)
- 401: Unauthorized (invalid API key)
- 500: Internal server error

Error responses include a detail message:

```json
{
  "detail": "Error message"
}
```

## Rate Limiting

The API has rate limiting to prevent abuse. Excessive requests will result in a 429 (Too Many Requests) response.

## Websocket API

In addition to the REST API, the system provides a WebSocket API for real-time updates. The WebSocket endpoint is available at `/ws` and requires the same API key for authentication.

### WebSocket Events

- `status`: System status updates
- `position`: Position updates (open, update, close)
- `trade`: New trade notifications
- `performance`: Performance metric updates
- `error`: Error notifications

### WebSocket Example

```javascript
const socket = new WebSocket('wss://your-deployment-url/ws');

socket.onopen = () => {
  socket.send(JSON.stringify({
    type: 'auth',
    apiKey: 'your_api_key'
  }));
};

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};
```
