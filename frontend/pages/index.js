import React from 'react';
import Head from 'next/head';

export default function Home() {
  return (
    <div style={{ fontFamily: 'Arial, sans-serif' }}>
      <Head>
        <title>KuCoin Crypto Trading Bot</title>
        <meta name="description" content="High-frequency crypto trading bot for KuCoin" />
      </Head>

      <main style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
        <h1 style={{ textAlign: 'center', color: '#333', marginBottom: '30px' }}>
          KuCoin Crypto Trading Bot Dashboard
        </h1>
        
        <div style={{ 
          padding: '20px', 
          border: '1px solid #ddd', 
          borderRadius: '8px',
          backgroundColor: '#f9f9f9',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        }}>
          <h2 style={{ color: '#2c3e50' }}>Welcome to Your Trading Bot</h2>
          <p>Your self-learning, high-frequency crypto trading model is ready for setup.</p>
          
          <div style={{ marginTop: '20px' }}>
            <h3>System Features:</h3>
            <ul style={{ lineHeight: '1.6' }}>
              <li>Self-learning algorithm that optimizes strategies based on trade outcomes</li>
              <li>Technical indicators, sentiment analysis, and price action pattern recognition</li>
              <li>Dynamic risk controls to secure capital</li>
              <li>Target of 3-5% daily compounding gains with 85% success rate</li>
            </ul>
          </div>
          
          <div style={{ 
            marginTop: '30px', 
            padding: '15px', 
            backgroundColor: '#e6f7ff', 
            borderLeft: '4px solid #1890ff',
            borderRadius: '3px'
          }}>
            <p><strong>Getting Started:</strong> This is the initial deployment of your trading dashboard. The complete system with full trading functionality will be available after completing the setup process.</p>
          </div>
          
          <div style={{ marginTop: '30px' }}>
            <h3>Next Steps:</h3>
            <ol style={{ lineHeight: '1.6' }}>
              <li>Configure your KuCoin API credentials</li>
              <li>Set your risk parameters and trading pairs</li>
              <li>Start with paper trading to validate performance</li>
              <li>Transition to live trading when ready</li>
            </ol>
          </div>
        </div>
      </main>
      
      <footer style={{ textAlign: 'center', marginTop: '40px', padding: '20px', color: '#666' }}>
        <p>KuCoin Crypto Trading Bot - High-Frequency Trading System</p>
      </footer>
    </div>
  );
}
