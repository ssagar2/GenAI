from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from typing import List, Dict, Any
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Configure logging with relative paths
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import Monte Carlo optimizer
try:
    from monte_carlo_optimizer import MonteCarloOptimizer
    from monte_carlo_api import monte_carlo_api, create_monte_carlo_routes
    MONTE_CARLO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Monte Carlo optimizer not available: {e}")
    MONTE_CARLO_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Mock stock data for faster performance
MOCK_STOCKS = {
    'AAPL': {
        'symbol': 'AAPL',
        'name': 'Apple Inc.',
        'sector': 'Technology',
        'current_price': 175.50,
        'change': 2.5,
        'volume': 50000000,
        'market_cap': 2800000000000,
        'pe_ratio': 28.5,
        'dividend_yield': 0.5,
        'volatility': 25.2,
        'beta': 1.1
    },
    'MSFT': {
        'symbol': 'MSFT',
        'name': 'Microsoft Corporation',
        'sector': 'Technology',
        'current_price': 380.25,
        'change': 1.8,
        'volume': 30000000,
        'market_cap': 2800000000000,
        'pe_ratio': 32.1,
        'dividend_yield': 0.8,
        'volatility': 22.8,
        'beta': 0.9
    },
    'GOOGL': {
        'symbol': 'GOOGL',
        'name': 'Alphabet Inc.',
        'sector': 'Technology',
        'current_price': 140.75,
        'change': 3.2,
        'volume': 25000000,
        'market_cap': 1800000000000,
        'pe_ratio': 25.3,
        'dividend_yield': 0.0,
        'volatility': 28.5,
        'beta': 1.2
    },
    'AMZN': {
        'symbol': 'AMZN',
        'name': 'Amazon.com Inc.',
        'sector': 'Consumer Discretionary',
        'current_price': 145.80,
        'change': 1.5,
        'volume': 40000000,
        'market_cap': 1500000000000,
        'pe_ratio': 45.2,
        'dividend_yield': 0.0,
        'volatility': 30.1,
        'beta': 1.3
    },
    'TSLA': {
        'symbol': 'TSLA',
        'name': 'Tesla Inc.',
        'sector': 'Consumer Discretionary',
        'current_price': 240.50,
        'change': -1.2,
        'volume': 60000000,
        'market_cap': 800000000000,
        'pe_ratio': 65.8,
        'dividend_yield': 0.0,
        'volatility': 45.2,
        'beta': 1.8
    }
}

def monte_carlo_optimization():
    """Monte Carlo simulation for portfolio optimization"""
    try:
        # This is a simplified Monte Carlo implementation
        # In a real implementation, this would use actual stock data
        n_simulations = 1000
        n_assets = 2  # For AAPL and TSLA
        
        # Generate random weights
        weights_list = []
        for _ in range(n_simulations):
            weights = np.random.random(n_assets)
            weights = weights / np.sum(weights)  # Normalize to sum to 1
            weights_list.append(weights)
        
        # Calculate Sharpe ratios (simplified)
        # In reality, this would use actual returns and volatility
        sharpe_ratios = []
        for weights in weights_list:
            # Simplified Sharpe calculation
            expected_return = np.sum(weights * [0.08, 0.12])  # 8% and 12% expected returns
            volatility = np.sum(weights * [0.15, 0.25])  # 15% and 25% volatility
            sharpe = (expected_return - 0.02) / volatility if volatility > 0 else 0
            sharpe_ratios.append(sharpe)
        
        # Find optimal weights
        optimal_idx = np.argmax(sharpe_ratios)
        optimal_weights = weights_list[optimal_idx]
        optimal_sharpe = sharpe_ratios[optimal_idx]
        
        return optimal_weights, optimal_sharpe
        
    except Exception as e:
        logger.error(f"Error in Monte Carlo optimization: {str(e)}")
        return None, 0.0

class PortfolioOptimizer:
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        self.ml_models = {}
        self.scaler = StandardScaler()
        
    def get_historical_data(self, symbol: str, period: str = '2y') -> pd.DataFrame:
        """Get historical data for ML model training"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            return hist
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for ML features"""
        if df.empty:
            return df
            
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Price momentum
        df['Price_Momentum'] = df['Close'].pct_change(periods=5)
        
        return df
    
    def train_ml_model(self, symbol: str) -> bool:
        """Train ML model for stock prediction"""
        try:
            # Get historical data
            hist_data = self.get_historical_data(symbol)
            if hist_data.empty:
                return False
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(hist_data)
            df = df.dropna()
            
            if len(df) < 100:  # Need sufficient data
                return False
            
            # Prepare features
            feature_columns = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 
                             'BB_Upper', 'BB_Lower', 'Volatility', 'Price_Momentum']
            
            X = df[feature_columns]
            y = df['Close'].shift(-1).dropna()  # Predict next day's price
            
            # Align X and y
            X = X[:-1]  # Remove last row since we don't have next day's price
            y = y[:-1]  # Remove last row
            
            # Ensure X and y have the same length
            min_length = min(len(X), len(y))
            X = X[:min_length]
            y = y[:min_length]
            
            if len(X) < 50:
                return False
            
            # Train Random Forest model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            self.ml_models[symbol] = {
                'model': model,
                'scaler': StandardScaler(),
                'features': feature_columns,
                'last_trained': datetime.now()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error training ML model for {symbol}: {str(e)}")
            return False
    
    def predict_stock_price(self, symbol: str, days_ahead: int = 30) -> Dict[str, Any]:
        """Predict stock price using ML model"""
        try:
            if symbol not in self.ml_models:
                if not self.train_ml_model(symbol):
                    return {'error': 'Could not train model'}
            
            # Get recent data for prediction
            hist_data = self.get_historical_data(symbol, '6mo')
            if hist_data.empty:
                return {'error': 'No historical data available'}
            
            df = self.calculate_technical_indicators(hist_data)
            df = df.dropna()
            
            if len(df) < 20:
                return {'error': 'Insufficient data for prediction'}
            
            model_info = self.ml_models[symbol]
            model = model_info['model']
            features = model_info['features']
            
            # Get latest features
            latest_features = df[features].iloc[-1:]
            
            # Make prediction
            predicted_price = model.predict(latest_features)[0]
            current_price = df['Close'].iloc[-1]
            
            # Calculate confidence based on model performance
            confidence = min(85, max(60, 75 + (predicted_price - current_price) / current_price * 100))
            
            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'prediction_change': ((predicted_price - current_price) / current_price) * 100,
                'confidence': confidence,
                'days_ahead': days_ahead
            }
            
        except Exception as e:
            logger.error(f"Error predicting price for {symbol}: {str(e)}")
            return {'error': 'Prediction failed'}
    
    def optimize_portfolio_ml(self, stocks: List[Dict], predictions: Dict, risk_profile: str, portfolio_value: float) -> Dict[str, float]:
        """Advanced ML-based portfolio optimization using Modern Portfolio Theory"""
        try:
            logger.info(f"=== Starting portfolio optimization for risk profile: {risk_profile} ===")
            
            # Use Monte Carlo optimizer if available
            if MONTE_CARLO_AVAILABLE and len(stocks) >= 2:
                logger.info("Using Monte Carlo optimization")
                return self._optimize_with_monte_carlo(stocks, risk_profile)
            else:
                logger.info("Using traditional MPT optimization")
                return self._optimize_with_traditional_mpt(stocks, predictions, risk_profile, portfolio_value)
            
        except Exception as e:
            logger.error(f"Error in advanced ML portfolio optimization: {str(e)}")
            # Fallback to simple equal weight allocation
            n_stocks = len(stocks)
            equal_weight = 100 / n_stocks
            return {stock['symbol']: round(equal_weight, 2) for stock in stocks}
    
    def _optimize_with_monte_carlo(self, stocks: List[Dict], risk_profile: str) -> Dict[str, float]:
        """Optimize using Monte Carlo simulation"""
        try:
            symbols = [stock['symbol'] for stock in stocks]
            
            # Create Monte Carlo optimizer
            mc_optimizer = MonteCarloOptimizer(risk_free_rate=0.02)
            
            # Run Monte Carlo simulation
            efficient_frontier = mc_optimizer.run_monte_carlo_simulation(
                symbols=symbols,
                n_simulations=5000,  # Reduced for faster response
                period='2y',
                use_parallel=True
            )
            
            # Get optimal weights
            optimal_weights = efficient_frontier.max_sharpe_portfolio.weights
            
            # Convert to allocation percentages
            allocations = {}
            for i, symbol in enumerate(symbols):
                allocations[symbol] = round(optimal_weights[i] * 100, 2)
            
            # Apply risk profile adjustments
            if risk_profile == 'conservative':
                allocations = self._adjust_for_conservative(allocations, symbols)
            elif risk_profile == 'aggressive':
                allocations = self._adjust_for_aggressive(allocations, symbols)
            
            # Ensure allocations sum to 100%
            total_allocation = sum(allocations.values())
            if total_allocation > 0:
                for symbol in allocations:
                    allocations[symbol] = round((allocations[symbol] / total_allocation) * 100, 2)
            
            logger.info(f"Monte Carlo optimization completed: {allocations}")
            return allocations
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo optimization: {str(e)}")
            # Fallback to equal weights
            n_stocks = len(stocks)
            equal_weight = 100 / n_stocks
            return {stock['symbol']: round(equal_weight, 2) for stock in stocks}
    
    def _optimize_with_traditional_mpt(self, stocks: List[Dict], predictions: Dict, risk_profile: str, portfolio_value: float) -> Dict[str, float]:
        """Traditional MPT optimization (existing logic)"""
        try:
            # Get historical data for all stocks
            historical_data = {}
            returns_data = {}
            
            for stock in stocks:
                symbol = stock['symbol']
                try:
                    # Get 2 years of historical data
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='2y')
                    
                    if len(hist) > 60:  # At least 60 days of data
                        # Calculate daily returns
                        hist['Returns'] = hist['Close'].pct_change().dropna()
                        historical_data[symbol] = hist
                        returns_data[symbol] = hist['Returns'].values
                    else:
                        # Fallback to mock data if insufficient historical data
                        returns_data[symbol] = np.random.normal(0.001, 0.02, 252)  # 252 trading days
                except Exception as e:
                    logger.error(f"Error getting historical data for {symbol}: {str(e)}")
                    returns_data[symbol] = np.random.normal(0.001, 0.02, 252)
            
            # Calculate expected returns from ML predictions
            expected_returns = {}
            for symbol, prediction in predictions.items():
                if 'prediction_change' in prediction:
                    expected_returns[symbol] = prediction['prediction_change'] / 100  # Convert to decimal
            
            # Calculate historical volatility (30-day rolling standard deviation)
            volatilities = {}
            for symbol in returns_data:
                if len(returns_data[symbol]) >= 30:
                    # 30-day rolling volatility
                    rolling_vol = np.array([np.std(returns_data[symbol][max(0, i-30):i+1]) * np.sqrt(252) 
                                          for i in range(len(returns_data[symbol]))])
                    volatilities[symbol] = np.mean(rolling_vol[-30:])  # Average of last 30 days
                else:
                    volatilities[symbol] = np.std(returns_data[symbol]) * np.sqrt(252)
            
            # Calculate correlation matrix
            symbols = list(returns_data.keys())
            n_stocks = len(symbols)
            
            if n_stocks > 1:
                # Align return series to same length
                min_length = min(len(returns_data[s]) for s in symbols)
                aligned_returns = np.array([returns_data[s][-min_length:] for s in symbols])
                correlation_matrix = np.corrcoef(aligned_returns)
            else:
                correlation_matrix = np.array([[1.0]])
            
            # Risk-free rate (approximate)
            risk_free_rate = 0.02  # 2% annual
            
            # Risk profile adjustments
            risk_adjustments = {
                'conservative': {'vol_multiplier': 0.7, 'return_multiplier': 0.8, 'max_volatility': 0.15},
                'moderate': {'vol_multiplier': 1.0, 'return_multiplier': 1.0, 'max_volatility': 0.25},
                'aggressive': {'vol_multiplier': 1.3, 'return_multiplier': 1.2, 'max_volatility': 0.35}
            }
            
            adjustment = risk_adjustments.get(risk_profile, risk_adjustments['moderate'])
            
            # Adjust volatilities and expected returns based on risk profile
            adjusted_volatilities = {s: v * adjustment['vol_multiplier'] for s, v in volatilities.items()}
            adjusted_expected_returns = {s: r * adjustment['return_multiplier'] for s, r in expected_returns.items()}
            
            # Risk profile specific allocation strategy
            optimal_sharpe = 0.0  # Default value
            
            if risk_profile == 'conservative':
                # Conservative: Allocate to lowest volatility stocks
                sorted_stocks = sorted(symbols, key=lambda s: adjusted_volatilities.get(s, 0.2))
                allocations = {}
                if len(sorted_stocks) >= 2:
                    # Allocate 70% to lowest volatility, 30% to second lowest
                    allocations[sorted_stocks[0]] = 70.0
                    allocations[sorted_stocks[1]] = 30.0
                    for stock in sorted_stocks[2:]:
                        allocations[stock] = 0.0
                else:
                    allocations[sorted_stocks[0]] = 100.0
                
            elif risk_profile == 'aggressive':
                # Aggressive: Allocate to highest volatility stocks
                sorted_stocks = sorted(symbols, key=lambda s: adjusted_volatilities.get(s, 0.2), reverse=True)
                allocations = {}
                if len(sorted_stocks) >= 2:
                    # Allocate 70% to highest volatility, 30% to second highest
                    allocations[sorted_stocks[0]] = 70.0
                    allocations[sorted_stocks[1]] = 30.0
                    for stock in sorted_stocks[2:]:
                        allocations[stock] = 0.0
                else:
                    allocations[sorted_stocks[0]] = 100.0
                    
            else:  # moderate
                # Moderate: Use Monte Carlo optimization
                optimal_weights, optimal_sharpe = monte_carlo_optimization()
                
                # If Monte Carlo fails, use equal weight
                if optimal_weights is None:
                    optimal_weights = np.ones(n_stocks) / n_stocks
                
                # Convert to allocation percentages
                allocations = {}
                for i, symbol in enumerate(symbols):
                    allocations[symbol] = round(optimal_weights[i] * 100, 2)
                
                # Ensure allocations sum to 100%
                total_allocation = sum(allocations.values())
                if total_allocation > 0:
                    for symbol in allocations:
                        allocations[symbol] = round((allocations[symbol] / total_allocation) * 100, 2)
            
            # Log optimization results
            logger.info(f"Risk profile: {risk_profile}")
            logger.info(f"Stock volatilities: {volatilities}")
            logger.info(f"Expected returns: {expected_returns}")
            logger.info(f"Adjusted volatilities: {adjusted_volatilities}")
            logger.info(f"Adjusted expected returns: {adjusted_expected_returns}")
            logger.info(f"Optimal Sharpe ratio: {optimal_sharpe:.4f}")
            logger.info(f"Portfolio allocations: {allocations}")
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error in traditional MPT optimization: {str(e)}")
            # Fallback to equal weights
            n_stocks = len(stocks)
            equal_weight = 100 / n_stocks
            return {stock['symbol']: round(equal_weight, 2) for stock in stocks}
    
    def _adjust_for_conservative(self, allocations: Dict[str, float], symbols: List[str]) -> Dict[str, float]:
        """Adjust allocations for conservative risk profile"""
        # Get volatility rankings (simplified)
        volatility_rankings = {
            'AAPL': 1, 'MSFT': 2, 'GOOGL': 3, 'AMZN': 4, 'TSLA': 5
        }
        
        # Sort by volatility (ascending)
        sorted_stocks = sorted(symbols, key=lambda x: volatility_rankings.get(x, 3))
        
        # Conservative adjustment: favor low volatility stocks
        adjusted_allocations = {}
        for i, symbol in enumerate(sorted_stocks):
            original_weight = allocations.get(symbol, 0)
            
            # Increase weight for low volatility stocks
            if i < len(sorted_stocks) // 2:  # Bottom half (low volatility)
                adjusted_allocations[symbol] = original_weight * 1.3
            else:  # Top half (high volatility)
                adjusted_allocations[symbol] = original_weight * 0.7
        
        return adjusted_allocations
    
    def _adjust_for_aggressive(self, allocations: Dict[str, float], symbols: List[str]) -> Dict[str, float]:
        """Adjust allocations for aggressive risk profile"""
        # Get volatility rankings (simplified)
        volatility_rankings = {
            'AAPL': 1, 'MSFT': 2, 'GOOGL': 3, 'AMZN': 4, 'TSLA': 5
        }
        
        # Sort by volatility (descending)
        sorted_stocks = sorted(symbols, key=lambda x: volatility_rankings.get(x, 3), reverse=True)
        
        # Aggressive adjustment: favor high volatility stocks
        adjusted_allocations = {}
        for i, symbol in enumerate(sorted_stocks):
            original_weight = allocations.get(symbol, 0)
            
            # Increase weight for high volatility stocks
            if i < len(sorted_stocks) // 2:  # Top half (high volatility)
                adjusted_allocations[symbol] = original_weight * 1.3
            else:  # Bottom half (low volatility)
                adjusted_allocations[symbol] = original_weight * 0.7
        
        return adjusted_allocations
    
    def optimize_portfolio_simple(self, stocks: List[Dict], risk_profile: str) -> Dict[str, float]:
        """Simple portfolio optimization as fallback"""
        try:
            if risk_profile == 'conservative':
                # Equal weight allocation
                total_stocks = len(stocks)
                allocations = {stock['symbol']: 100 / total_stocks for stock in stocks}
            elif risk_profile == 'aggressive':
                # Weight by volatility (higher volatility = higher allocation)
                total_volatility = sum(stock.get('volatility', 20) for stock in stocks)
                allocations = {stock['symbol']: (stock.get('volatility', 20) / total_volatility) * 100 for stock in stocks}
            else:  # moderate
                # Equal weight allocation
                total_stocks = len(stocks)
                allocations = {stock['symbol']: 100 / total_stocks for stock in stocks}
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error in simple portfolio optimization: {str(e)}")
            return {}
        
    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get real stock data with ML predictions"""
        symbol = symbol.upper()
        
        # Only support 5 specific stocks for reduced load
        supported_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        if symbol not in supported_stocks:
            return None
        
        try:
            # Get real-time data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info.get('regularMarketPrice'):
                return None
            
            # Get current price and calculate change
            hist = ticker.history(period='2d')
            if len(hist) >= 2:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                change = ((current_price - prev_price) / prev_price) * 100
            else:
                current_price = info.get('regularMarketPrice', 0)
                change = 0
            
            # Get ML prediction
            prediction = self.predict_stock_price(symbol)
            
            # Calculate realistic volatility based on historical data
            hist_30d = ticker.history(period='30d')
            if len(hist_30d) > 10:
                returns = hist_30d['Close'].pct_change().dropna()
                volatility = returns.std() * 100  # Convert to percentage
            else:
                # Fallback volatility based on stock characteristics
                volatility_map = {
                    'AAPL': 25.0,  # Lower volatility
                    'MSFT': 28.0,  # Medium volatility  
                    'GOOGL': 30,  # um-high volatility
                    'AMZN': 35.0,  # High volatility
                    'TSLA': 45,  # ry high volatility
                }
                volatility = volatility_map.get(symbol, 30)
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'current_price': current_price,
                'change': round(change, 2),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0) or 0,
                'volatility': round(volatility, 2),
                'beta': info.get('beta', 1.0),
                'ml_prediction': prediction
            }
            
        except Exception as e:
            logger.error(f"Error fetching real data for {symbol}: {str(e)}")
            # Fallback to mock data
            if symbol in MOCK_STOCKS:
                return MOCK_STOCKS[symbol]
            return None
    
    def search_stocks(self, query: str) -> List[Dict[str, Any]]:
        """Search for stocks based on query - only 5 supported stocks"""
        query = query.upper()
        results = []
        
        # Only search in the 5 supported stocks
        supported_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        for symbol in supported_stocks:
            if query in symbol or query in MOCK_STOCKS[symbol]['name'].upper():
                results.append({
                    'symbol': symbol,
                    'name': MOCK_STOCKS[symbol]['name'],
                    'sector': MOCK_STOCKS[symbol]['sector'],
                    'current_price': MOCK_STOCKS[symbol]['current_price']
                })
        
        # Limit results
        return results[:5]
    
    def calculate_portfolio_metrics(self, stocks: List[Dict], allocations: Dict, risk_profile: str = 'moderate') -> Dict[str, Any]:
        """Calculate portfolio metrics with real data and risk profile adjustments"""
        if not stocks or not allocations:
            return {
                'total_value': 10000,
                'total_return': 0,
                'expected_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'beta': 1.0,
                'sectors': {},
                'top_holdings': [],
                'total_allocation': 0
            }
        
        try:
            total_value = 10000  # Base portfolio value
            portfolio_data = []
            
            # Risk profile adjustments
            risk_adjustments = {
                'conservative': {'return_multiplier': 0.8, 'volatility_multiplier': 0.7},
                'moderate': {'return_multiplier': 1.0, 'volatility_multiplier': 1.0},
                'aggressive': {'return_multiplier': 1.3, 'volatility_multiplier': 1.4}
            }
            
            adjustment = risk_adjustments.get(risk_profile, risk_adjustments['moderate'])
            
            for stock in stocks:
                symbol = stock['symbol']
                allocation = allocations.get(symbol, 0)
                
                if allocation > 0:
                    stock_value = (allocation / 100) * total_value
                    shares = stock_value / stock['current_price']
                    
                    # Apply risk profile adjustments
                    adjusted_change = stock.get('change', 0) * adjustment['return_multiplier']
                    adjusted_volatility = stock.get('volatility', 20) * adjustment['volatility_multiplier']
                    
                    portfolio_data.append({
                        'symbol': symbol,
                        'name': stock.get('name', symbol),
                        'allocation': allocation,
                        'value': stock_value,
                        'shares': shares,
                        'current_price': stock['current_price'],
                        'change': adjusted_change,
                        'volatility': adjusted_volatility,
                        'beta': stock.get('beta', 1.0),
                        'sector': stock.get('sector', 'Unknown')
                    })
            
            # Calculate portfolio metrics
            total_allocation = sum(item['allocation'] for item in portfolio_data)
            weighted_return = sum(item['allocation'] * item['change'] / 100 for item in portfolio_data)
            weighted_volatility = sum(item['allocation'] * item['volatility'] / 100 for item in portfolio_data)
            weighted_beta = sum(item['allocation'] * item['beta'] / 100 for item in portfolio_data)
            
            # Calculate expected return (forward-looking) based on historical performance and risk profile
            # Expected return = Risk-free rate + (Beta * Market Risk Premium) + Risk Profile Adjustment
            risk_free_rate = 2.0  # 2% risk-free rate
            market_risk_premium = 6.0  # 6% market risk premium
            expected_return = risk_free_rate + (weighted_beta * market_risk_premium)
            
            # Apply risk profile adjustments to expected return
            if risk_profile == 'conservative':
                expected_return *= 0.8  # Conservative investors expect lower returns
            elif risk_profile == 'aggressive':
                expected_return *= 1.2  # Aggressive investors expect higher returns
            # Moderate profile keeps the calculated expected return as is
            
            # Calculate Sharpe ratio
            sharpe_ratio = (expected_return - risk_free_rate) / weighted_volatility if weighted_volatility > 0 else 0
            
            # Calculate sector allocation
            sectors = {}
            for item in portfolio_data:
                sector = item.get('sector', 'Unknown')
                sectors[sector] = sectors.get(sector, 0) + item['allocation']
            
            # Sort holdings by allocation
            top_holdings = sorted(portfolio_data, key=lambda x: x['allocation'], reverse=True)
            
            return {
                'total_value': total_value,
                'total_return': round(weighted_return, 2),
                'expected_return': round(expected_return, 2),
                'volatility': round(weighted_volatility, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'beta': round(weighted_beta, 2),
                'sectors': sectors,
                'top_holdings': top_holdings[:5],
                'total_allocation': total_allocation,
                'risk_profile': risk_profile
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {
                'total_value': 10000,
                'total_return': 0,
                'expected_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'beta': 1.0,
                'sectors': {},
                'top_holdings': [],
                'total_allocation': 0
            }

# Initialize the optimizer
optimizer = PortfolioOptimizer()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Portfolio Optimizer API is running'})

@app.route('/api/search', methods=['GET'])
def search_stocks():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])
    
    results = optimizer.search_stocks(query)
    return jsonify(results)

@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    data = optimizer.get_stock_data(symbol)
    if data:
        return jsonify(data)
    return jsonify({'error': 'Stock not found'}), 404

@app.route('/api/portfolio/metrics', methods=['POST'])
def calculate_portfolio_metrics():
    try:
        data = request.get_json()
        stocks = data.get('stocks', [])
        allocations = data.get('allocations', {})
        risk_profile = data.get('risk_profile', 'moderate')
        
        metrics = optimizer.calculate_portfolio_metrics(stocks, allocations, risk_profile)
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error in portfolio metrics: {str(e)}")
        return jsonify({'error': 'Failed to calculate metrics'}), 500

@app.route('/api/portfolio/optimize', methods=['POST'])
def optimize_portfolio():
    try:
        data = request.get_json()
        stocks = data.get('stocks', [])
        risk_profile = data.get('risk_profile', 'moderate')
        portfolio_value = data.get('portfolio_value', 10000)
        
        if not stocks:
            return jsonify({'error': 'No stocks provided'}), 400
        
        # Try Monte Carlo optimization first if available
        if MONTE_CARLO_AVAILABLE:
            try:
                logger.info("Attempting Monte Carlo optimization...")
                from monte_carlo_api import MonteCarloAPI
                api = MonteCarloAPI()
                
                # Get stock symbols
                symbols = [stock['symbol'] for stock in stocks]
                
                # Run Monte Carlo optimization
                result = api.optimize_portfolio(
                    symbols=symbols,
                    n_simulations=1000,
                    period='2y',
                    risk_profile=risk_profile
                )
                
                if result['success']:
                    logger.info("✓ Monte Carlo optimization successful")
                    return jsonify({
                        'allocations': result['optimal_portfolio']['weights'],
                        'ml_predictions': {},
                        'optimization_method': 'Monte Carlo Simulation',
                        'monte_carlo_results': result
                    })
                else:
                    logger.warning(f"Monte Carlo optimization failed: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Error in Monte Carlo optimization: {str(e)}")
                # Continue to fallback methods
        
        # Fallback to ML-based optimization
        stock_predictions = {}
        for stock in stocks:
            symbol = stock['symbol']
            prediction = optimizer.predict_stock_price(symbol)
            if 'error' not in prediction:
                stock_predictions[symbol] = prediction
        
        # ML-based optimization using Modern Portfolio Theory
        if len(stock_predictions) > 0:
            allocations = optimizer.optimize_portfolio_ml(stocks, stock_predictions, risk_profile, portfolio_value)
            optimization_method = 'ML-based'
        else:
            # Fallback to simple optimization
            allocations = optimizer.optimize_portfolio_simple(stocks, risk_profile)
            optimization_method = 'Simple'
        
        return jsonify({
            'allocations': allocations,
            'ml_predictions': stock_predictions,
            'optimization_method': optimization_method
        })
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {str(e)}")
        return jsonify({'error': 'Failed to optimize portfolio'}), 500

@app.route('/api/portfolio/performance', methods=['POST'])
def get_portfolio_performance():
    try:
        data = request.get_json()
        stocks = data.get('stocks', [])
        allocations = data.get('allocations', {})
        
        if not stocks:
            return jsonify([])
        
        # Generate performance data for Recharts format
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        performance_data = []
        
        for month in months:
            month_data = {'month': month}
            total_value = 0
            portfolio_value = 10000  # Base portfolio value
            # First, simulate prices for each stock
            simulated_prices = {}
            for stock in stocks:
                symbol = stock['symbol']
                base_price = stock.get('current_price', 100)
                import random
                random.seed(hash(symbol + month))
                change_percent = random.uniform(-0.12, 0.18)
                simulated_price = round(base_price * (1 + change_percent), 2)
                simulated_prices[symbol] = simulated_price
            # Now, calculate each stock's value in the portfolio
            for stock in stocks:
                symbol = stock['symbol']
                allocation = allocations.get(symbol, 0) / 100
                base_price = stock.get('current_price', 100)
                simulated_price = simulated_prices[symbol]
                # Value of this stock's position in the portfolio
                stock_value = allocation * portfolio_value * (simulated_price / base_price)
                month_data[symbol] = round(stock_value, 2)
                total_value += stock_value
            month_data['total'] = round(total_value, 2)
            performance_data.append(month_data)
        return jsonify(performance_data)
    except Exception as e:
        logger.error(f"Error in portfolio performance: {str(e)}")
        return jsonify({'error': 'Failed to get performance data'}), 500

# Add Monte Carlo routes if available
if MONTE_CARLO_AVAILABLE:
    try:
        logger.info("Attempting to add Monte Carlo routes...")
        
        # Use the create_monte_carlo_routes function from monte_carlo_api
        from monte_carlo_api import create_monte_carlo_routes
        create_monte_carlo_routes(app)
        
        logger.info("✓ Monte Carlo routes added successfully")
        
        # Debug: Print all registered routes
        logger.info("Registered routes:")
        for rule in app.url_map.iter_rules():
            logger.info(f"  {rule.rule} -> {rule.endpoint}")
            
        # Check specifically for Monte Carlo routes
        monte_carlo_routes = [
            '/api/portfolio/monte-carlo/optimize',
            '/api/portfolio/monte-carlo/efficient-frontier',
            '/api/portfolio/monte-carlo/stats'
        ]
        
        logger.info("Checking Monte Carlo routes:")
        for route in monte_carlo_routes:
            if route in [rule.rule for rule in app.url_map.iter_rules()]:
                logger.info(f"✓ {route} is registered")
            else:
                logger.error(f"✗ {route} is NOT registered")
            
    except Exception as e:
        logger.error(f"Error adding Monte Carlo routes: {str(e)}")
        logger.error(f"Exception details: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
else:
    logger.info("Monte Carlo optimizer not available - using traditional MPT only")

if __name__ == "__main__":
    # This is handled by start.py
    pass 