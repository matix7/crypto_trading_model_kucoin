# KuCoin Crypto Trading Model Deployment Instructions

This document provides step-by-step instructions for deploying the KuCoin Crypto Trading Model to Vercel for paper trading and transitioning to live trading.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Setup](#local-setup)
3. [Vercel Deployment](#vercel-deployment)
4. [Paper Trading Setup](#paper-trading-setup)
5. [Transition to Live Trading](#transition-to-live-trading)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Prerequisites

Before deploying the system, ensure you have:

- GitHub account
- Vercel account
- KuCoin account (for live trading)
- Node.js and npm installed locally
- Python 3.10 or higher installed locally

## Local Setup

### Step 1: Download and Extract the Code

1. Download the `crypto_trading_model_kucoin.zip` file
2. Extract the contents to a local directory
3. Open a terminal or command prompt and navigate to the extracted directory

### Step 2: Install Dependencies

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install KuCoin Python SDK:
   ```bash
   pip install kucoin-python
   ```

4. Install frontend dependencies:
   ```bash
   cd src/dashboard/frontend
   npm install
   cd ../../..
   ```

### Step 3: Create a GitHub Repository

1. Log in to your GitHub account
2. Create a new repository named `crypto-trading-bot-kucoin`
3. Initialize the local repository and push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/crypto-trading-bot-kucoin.git
   git push -u origin main
   ```

## Vercel Deployment

### Step 1: Connect to Vercel

1. Log in to your Vercel account: https://vercel.com/
2. Click "Add New" > "Project"
3. Import your GitHub repository (`crypto-trading-bot-kucoin`)
4. Select the repository and click "Import"

### Step 2: Configure the Project

1. Configure the project settings:
   - Framework Preset: Other
   - Root Directory: ./
   - Build Command: `pip install -r requirements.txt && cd src/dashboard/frontend && npm install && npm run build`
   - Output Directory: src/dashboard/frontend/out
   - Install Command: `pip install -r requirements.txt && cd src/dashboard/frontend && npm install`

2. Add Environment Variables:
   - KUCOIN_API_KEY: your_kucoin_api_key (use dummy values for paper trading)
   - KUCOIN_API_SECRET: your_kucoin_api_secret (use dummy values for paper trading)
   - KUCOIN_API_PASSPHRASE: your_kucoin_api_passphrase (use dummy values for paper trading)
   - USE_SANDBOX: true
   - DB_PATH: /tmp/trading.db
   - ENABLE_AUTHENTICATION: true
   - SESSION_SECRET: your_session_secret (generate a random string)
   - JWT_SECRET: your_jwt_secret (generate a random string)
   - API_KEY: your_api_key (generate a random string)

3. Click "Deploy"

### Step 3: Verify Deployment

1. Wait for the deployment to complete
2. Click on the deployment URL to open the application
3. Verify that the dashboard loads correctly
4. Check the logs for any errors

## Paper Trading Setup

### Step 1: Configure Paper Trading Settings

1. Log in to the dashboard using the default credentials:
   - Username: admin
   - Password: admin

2. Navigate to the Settings page

3. Configure the following settings:
   - Trading Pairs: Select the pairs you want to trade (e.g., BTC-USDT, ETH-USDT)
   - Timeframes: Select the timeframes to analyze (e.g., 5m, 15m, 1h, 4h)
   - Initial Capital: Set your paper trading starting capital (e.g., 10000 USDT)
   - Risk Parameters: Configure risk per trade, stop loss, take profit, etc.
   - Maximum Open Positions: Set the maximum number of positions to hold simultaneously

4. Save the settings

### Step 2: Start Paper Trading

1. On the dashboard, click the "Start Trading" button
2. Verify that the system status changes to "Running"
3. Monitor the system for a few minutes to ensure it's working correctly

### Step 3: Monitor Performance

1. Use the dashboard to monitor:
   - Open positions
   - Trade history
   - Account balance and equity
   - Performance metrics

2. Let the system run for at least 2-3 weeks to:
   - Gather sufficient performance data
   - Allow the self-learning algorithms to optimize
   - Test performance across different market conditions

## Transition to Live Trading

### Step 1: Create KuCoin API Keys

1. Log in to your KuCoin account
2. Navigate to "API Management" in your account settings
3. Click "Create API"
4. Set the following permissions:
   - "General" permissions (read-only access)
   - "Trade" permissions (ability to place and cancel orders)
   - Do NOT enable "Transfer" permissions (for security)
5. Set IP restrictions to limit access to your Vercel deployment IP
6. Complete the security verification
7. Save your API Key, Secret Key, and Passphrase securely

### Step 2: Update Vercel Environment Variables

1. Go to your Vercel project dashboard
2. Navigate to "Settings" > "Environment Variables"
3. Update the following variables:
   - KUCOIN_API_KEY: your_real_kucoin_api_key
   - KUCOIN_API_SECRET: your_real_kucoin_api_secret
   - KUCOIN_API_PASSPHRASE: your_real_kucoin_api_passphrase
   - USE_SANDBOX: false
4. Click "Save"
5. Redeploy the project for the changes to take effect

### Step 3: Configure Live Trading Settings

1. Log in to the dashboard
2. Navigate to the Settings page
3. Adjust your risk parameters for live trading:
   - Reduce risk per trade (start conservative)
   - Set smaller position sizes
   - Use tighter stop losses
   - Limit the maximum number of open positions
4. Save the settings

### Step 4: Start Live Trading

1. Ensure you have sufficient funds in your KuCoin account
2. On the dashboard, click the "Start Trading" button
3. Verify that the system status changes to "Running"
4. Monitor the first few trades closely to ensure everything is working correctly

## Monitoring and Maintenance

### Daily Monitoring

1. Check the dashboard daily to:
   - Review open positions
   - Analyze completed trades
   - Monitor account balance and equity
   - Verify system status

2. Set up alerts for:
   - Significant drawdowns
   - System errors
   - Unusual trading activity

### Weekly Maintenance

1. Review performance metrics
2. Adjust risk parameters if needed
3. Check for any system updates
4. Verify KuCoin API status

### Monthly Review

1. Perform a comprehensive performance review
2. Analyze win rate, profit factor, and other metrics
3. Consider adjusting trading pairs or timeframes
4. Optimize system parameters based on performance data

### Troubleshooting

If you encounter issues:

1. Check the Vercel logs for error messages
2. Verify KuCoin API connectivity
3. Ensure sufficient balance for trading
4. Restart the system if necessary

Remember to always monitor your live trading system and be prepared to intervene if needed.
