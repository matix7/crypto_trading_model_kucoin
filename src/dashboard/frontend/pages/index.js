"""
Next.js application for the dashboard frontend.
"""

import React from 'react';
import Head from 'next/head';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import IconButton from '@mui/material/IconButton';
import MenuIcon from '@mui/icons-material/Menu';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import RefreshIcon from '@mui/icons-material/Refresh';
import SettingsIcon from '@mui/icons-material/Settings';
import HelpIcon from '@mui/icons-material/Help';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import Drawer from '@mui/material/Drawer';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
import DashboardIcon from '@mui/icons-material/Dashboard';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import HistoryIcon from '@mui/icons-material/History';
import AssessmentIcon from '@mui/icons-material/Assessment';
import SettingsApplicationsIcon from '@mui/icons-material/SettingsApplications';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Snackbar from '@mui/material/Snackbar';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, RadarChart, 
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';

// Dashboard configuration
const DASHBOARD_CONFIG = {
  TITLE: 'Crypto Trading Analytics Dashboard',
  REFRESH_INTERVAL: 60000, // 1 minute
  THEME: {
    DARK: {
      palette: {
        mode: 'dark',
        primary: {
          main: '#1976D2',
        },
        secondary: {
          main: '#FF9800',
        },
        background: {
          default: '#121212',
          paper: '#1E1E1E',
        },
      },
    },
    LIGHT: {
      palette: {
        mode: 'light',
        primary: {
          main: '#1976D2',
        },
        secondary: {
          main: '#FF9800',
        },
      },
    },
  },
  CHART_COLORS: {
    PROFIT: '#00C853',
    LOSS: '#FF5252',
    NEUTRAL: '#9E9E9E',
    PRIMARY: '#1976D2',
    SECONDARY: '#FF9800',
    ACCENT: '#E91E63',
    SERIES: ['#1976D2', '#FF9800', '#E91E63', '#4CAF50', '#9C27B0', '#00BCD4', '#FFEB3B', '#795548'],
  },
};

// Mock data for development
const MOCK_DATA = {
  system_status: {
    status: 'running',
    account_balance: 12345.67,
    equity: 12567.89,
    open_positions: 3,
    total_trades: 42,
    total_return: 0.25,
    daily_return: 0.04,
    win_rate: 0.85,
    success_rate: 0.92,
    last_update: Date.now() / 1000,
  },
  open_positions: [
    {
      position_id: '1',
      trading_pair: 'BTCUSDT',
      side: 'BUY',
      entry_price: 65432.10,
      current_price: 65987.65,
      quantity: 0.15,
      position_size: 9814.82,
      unrealized_profit_loss: 83.33,
      unrealized_profit_loss_percentage: 0.0085,
      entry_time: Date.now() - 3600000,
    },
    {
      position_id: '2',
      trading_pair: 'ETHUSDT',
      side: 'BUY',
      entry_price: 3456.78,
      current_price: 3512.34,
      quantity: 2.5,
      position_size: 8641.95,
      unrealized_profit_loss: 138.90,
      unrealized_profit_loss_percentage: 0.0161,
      entry_time: Date.now() - 7200000,
    },
    {
      position_id: '3',
      trading_pair: 'SOLUSDT',
      side: 'SELL',
      entry_price: 178.90,
      current_price: 176.45,
      quantity: 45,
      position_size: 8050.50,
      unrealized_profit_loss: 110.25,
      unrealized_profit_loss_percentage: 0.0137,
      entry_time: Date.now() - 10800000,
    },
  ],
  chart_data: {
    equity_curve: Array.from({ length: 30 }, (_, i) => ({
      date: new Date(Date.now() - (29 - i) * 86400000).toISOString().split('T')[0],
      equity: 10000 * (1 + 0.04) ** i,
    })),
    daily_returns: Array.from({ length: 30 }, (_, i) => ({
      date: new Date(Date.now() - (29 - i) * 86400000).toISOString().split('T')[0],
      return: (Math.random() * 8 - 2),
    })),
    drawdown: Array.from({ length: 30 }, (_, i) => ({
      date: new Date(Date.now() - (29 - i) * 86400000).toISOString().split('T')[0],
      drawdown: -Math.random() * 5,
    })),
    win_rate: [
      { name: 'Winning Trades', value: 85 },
      { name: 'Losing Trades', value: 15 },
    ],
    performance_metrics_radar: [
      { metric: 'Win Rate', value: 85 },
      { metric: 'Profit Factor', value: 3.2 },
      { metric: 'Sharpe Ratio', value: 2.1 },
      { metric: 'Calmar Ratio', value: 1.8 },
      { metric: 'Success Rate', value: 92 },
      { metric: 'Expectancy', value: 1.5 },
    ],
    trading_pairs_distribution: [
      { name: 'BTCUSDT', value: 40 },
      { name: 'ETHUSDT', value: 30 },
      { name: 'SOLUSDT', value: 15 },
      { name: 'BNBUSDT', value: 10 },
      { name: 'ADAUSDT', value: 5 },
    ],
  },
};

// Dashboard component
export default function Dashboard() {
  // State
  const [darkMode, setDarkMode] = React.useState(true);
  const [drawerOpen, setDrawerOpen] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [snackbar, setSnackbar] = React.useState({ open: false, message: '', severity: 'info' });
  const [dashboardData, setDashboardData] = React.useState(MOCK_DATA);

  // Theme
  const theme = React.useMemo(
    () => createTheme(darkMode ? DASHBOARD_CONFIG.THEME.DARK : DASHBOARD_CONFIG.THEME.LIGHT),
    [darkMode]
  );

  // Toggle drawer
  const toggleDrawer = () => {
    setDrawerOpen(!drawerOpen);
  };

  // Toggle theme
  const toggleTheme = () => {
    setDarkMode(!darkMode);
  };

  // Refresh data
  const refreshData = async () => {
    setLoading(true);
    try {
      // In a real app, this would fetch data from the API
      // const response = await fetch('/api/dashboard-data');
      // const data = await response.json();
      // setDashboardData(data);
      
      // For now, just simulate a delay and use mock data
      await new Promise(resolve => setTimeout(resolve, 1000));
      setDashboardData({
        ...MOCK_DATA,
        system_status: {
          ...MOCK_DATA.system_status,
          last_update: Date.now() / 1000,
        },
      });
      
      setSnackbar({
        open: true,
        message: 'Dashboard data refreshed successfully',
        severity: 'success',
      });
    } catch (err) {
      console.error('Error refreshing data:', err);
      setError(err.message);
      setSnackbar({
        open: true,
        message: `Error refreshing data: ${err.message}`,
        severity: 'error',
      });
    } finally {
      setLoading(false);
    }
  };

  // Start trading
  const startTrading = async () => {
    try {
      // In a real app, this would call the API
      // await fetch('/api/start-trading', { method: 'POST' });
      
      // For now, just simulate a delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setDashboardData({
        ...dashboardData,
        system_status: {
          ...dashboardData.system_status,
          status: 'running',
        },
      });
      
      setSnackbar({
        open: true,
        message: 'Trading started successfully',
        severity: 'success',
      });
    } catch (err) {
      console.error('Error starting trading:', err);
      setSnackbar({
        open: true,
        message: `Error starting trading: ${err.message}`,
        severity: 'error',
      });
    }
  };

  // Stop trading
  const stopTrading = async () => {
    try {
      // In a real app, this would call the API
      // await fetch('/api/stop-trading', { method: 'POST' });
      
      // For now, just simulate a delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setDashboardData({
        ...dashboardData,
        system_status: {
          ...dashboardData.system_status,
          status: 'stopped',
        },
      });
      
      setSnackbar({
        open: true,
        message: 'Trading stopped successfully',
        severity: 'success',
      });
    } catch (err) {
      console.error('Error stopping trading:', err);
      setSnackbar({
        open: true,
        message: `Error stopping trading: ${err.message}`,
        severity: 'error',
      });
    }
  };

  // Close snackbar
  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  // Format currency
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  // Format percentage
  const formatPercentage = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'percent',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  // Format date
  const formatDate = (timestamp) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  // Auto-refresh
  React.useEffect(() => {
    refreshData();
    
    const interval = setInterval(() => {
      refreshData();
    }, DASHBOARD_CONFIG.REFRESH_INTERVAL);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Head>
        <title>{DASHBOARD_CONFIG.TITLE}</title>
        <meta name="description" content="Analytics dashboard for the crypto trading model" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      {/* App Bar */}
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={toggleDrawer}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            {DASHBOARD_CONFIG.TITLE}
          </Typography>
          
          {/* System Status */}
          <Chip
            label={dashboardData.system_status.status === 'running' ? 'Running' : 'Stopped'}
            color={dashboardData.system_status.status === 'running' ? 'success' : 'error'}
            sx={{ mr: 2 }}
          />
          
          {/* Balance */}
          <Typography variant="body2" sx={{ mr: 2 }}>
            Balance: {formatCurrency(dashboardData.system_status.account_balance)}
          </Typography>
          
          {/* Equity */}
          <Typography variant="body2" sx={{ mr: 2 }}>
            Equity: {formatCurrency(dashboardData.system_status.equity)}
          </Typography>
          
          {/* Total Return */}
          <Typography variant="body2" sx={{ mr: 2 }}>
            Return: {formatPercentage(dashboardData.system_status.total_return)}
          </Typography>
          
          {/* Theme Toggle */}
          <IconButton color="inherit" onClick={toggleTheme} sx={{ mr: 1 }}>
            {darkMode ? <Brightness7Icon /> : <Brightness4Icon />}
          </IconButton>
          
          {/* Refresh Button */}
          <IconButton color="inherit" onClick={refreshData} disabled={loading} sx={{ mr: 1 }}>
            {loading ? <CircularProgress size={24} color="inherit" /> : <RefreshIcon />}
          </IconButton>
          
          {/* Settings Button */}
          <IconButton color="inherit" sx={{ mr: 1 }}>
            <SettingsIcon />
          </IconButton>
          
          {/* Help Button */}
          <IconButton color="inherit" sx={{ mr: 1 }}>
            <HelpIcon />
          </IconButton>
          
          {/* User Menu */}
          <IconButton color="inherit" edge="end">
            <AccountCircleIcon />
          </IconButton>
        </Toolbar>
      </AppBar>
      
      {/* Drawer */}
      <Drawer
        variant="persistent"
        anchor="left"
        open={drawerOpen}
        sx={{
          width: 240,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: 240,
            boxSizing: 'border-box',
            top: ['48px', '56px', '64px'],
            height: 'auto',
            bottom: 0,
          },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto' }}>
          <List>
            <ListItem button>
              <ListItemIcon>
                <DashboardIcon />
              </ListItemIcon>
              <ListItemText primary="Dashboard" />
            </ListItem>
            <ListItem button>
              <ListItemIcon>
                <ShowChartIcon />
              </ListItemIcon>
              <ListItemText primary="Charts" />
            </ListItem>
            <ListItem button>
              <ListItemIcon>
                <HistoryIcon />
              </ListItemIcon>
              <ListItemText primary="Trade History" />
            </ListItem>
            <ListItem button>
              <ListItemIcon>
                <AssessmentIcon />
              </ListItemIcon>
              <ListItemText primary="Performance" />
            </ListItem>
          </List>
          <Divider />
          <List>
            <ListItem button>
              <ListItemIcon>
                <SettingsApplicationsIcon />
              </ListItemIcon>
              <ListItemText primary="Settings" />
            </ListItem>
          </List>
          <Divider />
          <List>
            {dashboardData.system_status.status === 'running' ? (
              <ListItem button onClick={stopTrading}>
                <ListItemIcon>
                  <StopIcon />
                </ListItemIcon>
                <ListItemText primary="Stop Trading" />
              </ListItem>
            ) : (
              <ListItem button onClick={startTrading}>
                <ListItemIcon>
                  <PlayArrowIcon />
                </ListItemIcon>
                <ListItemText primary="Start Trading" />
              </ListItem>
            )}
          </List>
        </Box>
      </Drawer>
      
      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          ml: drawerOpen ? '240px' : 0,
          transition: theme.transitions.create('margin', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        <Toolbar />
        
        {/* Status Cards */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Paper
              sx={{
                p: 2,
                display: 'flex',
                flexDirection: 'column',
                height: 120,
              }}
            >
              <Typography component="h2" variant="h6" color="primary" gutterBottom>
                Account Balance
              </Typography>
              <Typography component="p" variant="h4">
                {formatCurrency(dashboardData.system_status.account_balance)}
              </Typography>
              <Typography color="text.secondary" sx={{ flex: 1 }}>
                Available for trading
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper
              sx={{
                p: 2,
                display: 'flex',
                flexDirection: 'column',
                height: 120,
              }}
            >
              <Typography component="h2" variant="h6" color="primary" gutterBottom>
                Total Return
              </Typography>
              <Typography
                component="p"
                variant="h4"
                color={dashboardData.system_status.total_return >= 0 ? 'success.main' : 'error.main'}
              >
                {formatPercentage(dashboardData.system_status.total_return)}
              </Typography>
              <Typography color="text.secondary" sx={{ flex: 1 }}>
                Since inception
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper
              sx={{
                p: 2,
                display: 'flex',
                flexDirection: 'column',
                height: 120,
              }}
            >
              <Typography component="h2" variant="h6" color="primary" gutterBottom>
                Win Rate
              </Typography>
              <Typography component="p" variant="h4">
                {formatPercentage(dashboardData.system_status.win_rate)}
              </Typography>
              <Typography color="text.secondary" sx={{ flex: 1 }}>
                Target: 85%
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper
              sx={{
                p: 2,
                display: 'flex',
                flexDirection: 'column',
                height: 120,
              }}
            >
              <Typography component="h2" variant="h6" color="primary" gutterBottom>
                Daily Return
              </Typography>
              <Typography
                component="p"
                variant="h4"
                color={dashboardData.system_status.daily_return >= 0 ? 'success.main' : 'error.main'}
              >
                {formatPercentage(dashboardData.system_status.daily_return)}
              </Typography>
              <Typography color="text.secondary" sx={{ flex: 1 }}>
                Target: 3-5%
              </Typography>
            </Paper>
          </Grid>
        </Grid>
        
        {/* Equity Curve Chart */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12}>
            <Paper
              sx={{
                p: 2,
                display: 'flex',
                flexDirection: 'column',
                height: 400,
              }}
            >
              <Typography component="h2" variant="h6" color="primary" gutterBottom>
                Equity Curve
              </Typography>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={dashboardData.chart_data.equity_curve}
                  margin={{
                    top: 10,
                    right: 30,
                    left: 0,
                    bottom: 0,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip formatter={(value) => formatCurrency(value)} />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="equity"
                    stroke={DASHBOARD_CONFIG.CHART_COLORS.PRIMARY}
                    fill={DASHBOARD_CONFIG.CHART_COLORS.PRIMARY}
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
        </Grid>
        
        {/* Daily Returns and Drawdown Charts */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={6}>
            <Paper
              sx={{
                p: 2,
                display: 'flex',
                flexDirection: 'column',
                height: 300,
              }}
            >
              <Typography component="h2" variant="h6" color="primary" gutterBottom>
                Daily Returns
              </Typography>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={dashboardData.chart_data.daily_returns}
                  margin={{
                    top: 10,
                    right: 30,
                    left: 0,
                    bottom: 0,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
                  <Legend />
                  <Bar
                    dataKey="return"
                    name="Daily Return"
                    fill={DASHBOARD_CONFIG.CHART_COLORS.SECONDARY}
                  >
                    {dashboardData.chart_data.daily_returns.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.return >= 0 ? DASHBOARD_CONFIG.CHART_COLORS.PROFIT : DASHBOARD_CONFIG.CHART_COLORS.LOSS}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Paper
              sx={{
                p: 2,
                display: 'flex',
                flexDirection: 'column',
                height: 300,
              }}
            >
              <Typography component="h2" variant="h6" color="primary" gutterBottom>
                Drawdown
              </Typography>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={dashboardData.chart_data.drawdown}
                  margin={{
                    top: 10,
                    right: 30,
                    left: 0,
                    bottom: 0,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="drawdown"
                    name="Drawdown"
                    stroke={DASHBOARD_CONFIG.CHART_COLORS.LOSS}
                    fill={DASHBOARD_CONFIG.CHART_COLORS.LOSS}
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
        </Grid>
        
        {/* Performance Metrics and Win Rate Charts */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={6}>
            <Paper
              sx={{
                p: 2,
                display: 'flex',
                flexDirection: 'column',
                height: 400,
              }}
            >
              <Typography component="h2" variant="h6" color="primary" gutterBottom>
                Performance Metrics
              </Typography>
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart
                  cx="50%"
                  cy="50%"
                  outerRadius="80%"
                  data={dashboardData.chart_data.performance_metrics_radar}
                >
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} />
                  <Radar
                    name="Performance"
                    dataKey="value"
                    stroke={DASHBOARD_CONFIG.CHART_COLORS.PRIMARY}
                    fill={DASHBOARD_CONFIG.CHART_COLORS.PRIMARY}
                    fillOpacity={0.6}
                  />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Paper
                  sx={{
                    p: 2,
                    display: 'flex',
                    flexDirection: 'column',
                    height: 190,
                  }}
                >
                  <Typography component="h2" variant="h6" color="primary" gutterBottom>
                    Win Rate
                  </Typography>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={dashboardData.chart_data.win_rate}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        <Cell key="cell-0" fill={DASHBOARD_CONFIG.CHART_COLORS.PROFIT} />
                        <Cell key="cell-1" fill={DASHBOARD_CONFIG.CHART_COLORS.LOSS} />
                      </Pie>
                      <Tooltip formatter={(value, name) => [`${value} trades`, name]} />
                    </PieChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
              <Grid item xs={12}>
                <Paper
                  sx={{
                    p: 2,
                    display: 'flex',
                    flexDirection: 'column',
                    height: 190,
                  }}
                >
                  <Typography component="h2" variant="h6" color="primary" gutterBottom>
                    Trading Pairs
                  </Typography>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={dashboardData.chart_data.trading_pairs_distribution}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {dashboardData.chart_data.trading_pairs_distribution.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={DASHBOARD_CONFIG.CHART_COLORS.SERIES[index % DASHBOARD_CONFIG.CHART_COLORS.SERIES.length]}
                          />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value, name) => [`${value} trades`, name]} />
                    </PieChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
            </Grid>
          </Grid>
        </Grid>
        
        {/* Open Positions */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12}>
            <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
              <Typography component="h2" variant="h6" color="primary" gutterBottom>
                Open Positions
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Trading Pair</TableCell>
                      <TableCell>Side</TableCell>
                      <TableCell>Entry Price</TableCell>
                      <TableCell>Current Price</TableCell>
                      <TableCell>Quantity</TableCell>
                      <TableCell>Position Size</TableCell>
                      <TableCell>Unrealized P/L</TableCell>
                      <TableCell>P/L %</TableCell>
                      <TableCell>Entry Time</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {dashboardData.open_positions.map((position) => (
                      <TableRow key={position.position_id}>
                        <TableCell>{position.trading_pair}</TableCell>
                        <TableCell>
                          <Chip
                            label={position.side}
                            color={position.side === 'BUY' ? 'success' : 'error'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>{formatCurrency(position.entry_price)}</TableCell>
                        <TableCell>{formatCurrency(position.current_price)}</TableCell>
                        <TableCell>{position.quantity}</TableCell>
                        <TableCell>{formatCurrency(position.position_size)}</TableCell>
                        <TableCell
                          sx={{
                            color: position.unrealized_profit_loss >= 0 ? 'success.main' : 'error.main',
                          }}
                        >
                          {formatCurrency(position.unrealized_profit_loss)}
                        </TableCell>
                        <TableCell
                          sx={{
                            color: position.unrealized_profit_loss_percentage >= 0 ? 'success.main' : 'error.main',
                          }}
                        >
                          {formatPercentage(position.unrealized_profit_loss_percentage)}
                        </TableCell>
                        <TableCell>{new Date(position.entry_time).toLocaleString()}</TableCell>
                      </TableRow>
                    ))}
                    {dashboardData.open_positions.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={9} align="center">
                          No open positions
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Grid>
        </Grid>
        
        {/* Footer */}
        <Box sx={{ pt: 4, pb: 4 }}>
          <Typography variant="body2" color="text.secondary" align="center">
            {'© '}
            {new Date().getFullYear()}
            {' Crypto Trading Model. All rights reserved. '}
            {'Last updated: '}
            {dashboardData.system_status.last_update
              ? formatDate(dashboardData.system_status.last_update)
              : 'Never'}
          </Typography>
        </Box>
      </Box>
      
      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </ThemeProvider>
  );
}
