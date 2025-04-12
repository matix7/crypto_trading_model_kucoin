# Deployment Instructions: Crypto Trading Model

## Overview

This document provides step-by-step instructions for deploying the Crypto Trading Model, including both the paper trading system and the analytics dashboard. The deployment process uses Vercel for hosting the web applications and requires setting up the necessary environment and dependencies.

## Prerequisites

Before deploying the system, ensure you have the following:

- Git installed on your local machine
- Node.js 18.x or higher installed
- Python 3.10 or higher installed
- Vercel CLI installed (`npm install -g vercel`)
- A Vercel account (sign up at https://vercel.com)
- A GitHub account (optional, for source code management)

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/crypto-trading-model.git
cd crypto-trading-model
```

### 2. Set Up Python Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Set Up Node.js Environment

```bash
# Navigate to the dashboard frontend directory
cd src/dashboard/frontend

# Install Node.js dependencies
npm install
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# API Configuration
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
USE_TESTNET=true

# Database Configuration
DB_PATH=/path/to/your/database.db

# Web Interface Configuration
PORT=3000
HOST=0.0.0.0
ENABLE_AUTHENTICATION=true
SESSION_SECRET=your_session_secret
DEFAULT_USERNAME=admin
DEFAULT_PASSWORD=admin
JWT_SECRET=your_jwt_secret
API_KEY=your_api_key

# Dashboard Configuration
DASHBOARD_PORT=3001
DASHBOARD_HOST=0.0.0.0
```

### 5. Run the System Locally

#### Start the Paper Trading API

```bash
# Navigate to the root directory
cd /path/to/crypto-trading-model

# Run the paper trading API
python -m src.paper_trading.api
```

#### Start the Dashboard API

```bash
# In a new terminal, navigate to the root directory
cd /path/to/crypto-trading-model

# Run the dashboard API
python -m src.dashboard.api
```

#### Start the Dashboard Frontend

```bash
# In a new terminal, navigate to the dashboard frontend directory
cd /path/to/crypto-trading-model/src/dashboard/frontend

# Run the frontend development server
npm run dev
```

## Deployment to Vercel

### 1. Prepare for Deployment

#### Create a `vercel.json` Configuration File

Create a `vercel.json` file in the root directory:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "src/paper_trading/api.py",
      "use": "@vercel/python"
    },
    {
      "src": "src/dashboard/api.py",
      "use": "@vercel/python"
    },
    {
      "src": "src/dashboard/frontend/package.json",
      "use": "@vercel/next"
    }
  ],
  "routes": [
    {
      "src": "/api/trading/(.*)",
      "dest": "src/paper_trading/api.py"
    },
    {
      "src": "/api/dashboard/(.*)",
      "dest": "src/dashboard/api.py"
    },
    {
      "src": "/(.*)",
      "dest": "src/dashboard/frontend/$1"
    }
  ],
  "env": {
    "BINANCE_API_KEY": "@binance-api-key",
    "BINANCE_API_SECRET": "@binance-api-secret",
    "USE_TESTNET": "true",
    "DB_PATH": "/tmp/trading.db",
    "ENABLE_AUTHENTICATION": "true",
    "SESSION_SECRET": "@session-secret",
    "JWT_SECRET": "@jwt-secret",
    "API_KEY": "@api-key"
  }
}
```

#### Update API Base URLs

Update the API base URLs in the configuration files to use the deployed URLs:

1. In `src/dashboard/config.py`, update the `API` section:

```python
API = {
    'BASE_URL': '/api/trading',  # Updated for Vercel deployment
    # ... other settings ...
}
```

2. In `src/dashboard/frontend/pages/index.js`, update the API fetch calls to use the correct paths.

### 2. Set Up Vercel Secrets

Use the Vercel CLI to set up secrets for sensitive information:

```bash
vercel secrets add binance-api-key "your_binance_api_key"
vercel secrets add binance-api-secret "your_binance_api_secret"
vercel secrets add session-secret "your_session_secret"
vercel secrets add jwt-secret "your_jwt_secret"
vercel secrets add api-key "your_api_key"
```

### 3. Deploy to Vercel

```bash
# Navigate to the root directory
cd /path/to/crypto-trading-model

# Deploy to Vercel
vercel
```

Follow the prompts to link your project to your Vercel account and complete the deployment.

### 4. Verify Deployment

Once the deployment is complete, Vercel will provide a URL for your application. Visit this URL to access the dashboard and verify that everything is working correctly.

## Continuous Deployment

For continuous deployment, you can connect your GitHub repository to Vercel:

1. Push your code to a GitHub repository
2. In the Vercel dashboard, create a new project and import your GitHub repository
3. Configure the project settings and environment variables
4. Enable automatic deployments for future updates

## Database Considerations

The SQLite database used in development is stored in a temporary location in the Vercel deployment. For production use, consider:

1. Using a persistent database service like PostgreSQL
2. Implementing database migrations for schema updates
3. Setting up regular database backups

## Scaling Considerations

For higher traffic or production use:

1. Consider using a dedicated server or cloud service for the trading engine
2. Implement a more robust database solution
3. Set up monitoring and alerting for system health
4. Implement rate limiting for API endpoints
5. Use a CDN for static assets

## Troubleshooting

### Deployment Issues

If you encounter issues during deployment:

1. Check the Vercel deployment logs for error messages
2. Verify that all environment variables are correctly set
3. Ensure that the `vercel.json` configuration is correct
4. Check that all dependencies are properly installed

### Runtime Issues

If the deployed application has runtime issues:

1. Check the application logs in the Vercel dashboard
2. Verify API connectivity
3. Check database access
4. Test API endpoints using tools like Postman or curl

## Conclusion

Following these instructions will deploy the Crypto Trading Model to Vercel, making it accessible via the web. The system includes both the paper trading engine and the analytics dashboard, allowing you to monitor and control your trading strategies remotely.

For additional support or questions, please refer to the API documentation or contact the development team.
