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
        """ML-based portfolio optimization using Modern Portfolio Theory"""
        try:
            # Calculate expected returns from ML predictions
            expected_returns = {}
            for symbol, prediction in predictions.items():
                if 'prediction_change' in prediction:
                    expected_returns[symbol] = prediction['prediction_change'] / 100  # Convert to decimal
            
            # Risk profile adjustments
            risk_multipliers = {
                'conservative': 0.7,  # Reduce risk
                'moderate': 1.0,      # Standard risk
                'aggressive': 1.3     # Increase risk
            }
            risk_mult = risk_multipliers.get(risk_profile, 1.0)
            
            # Calculate optimal weights using ML predictions
            total_weight = 0
            allocations = {}
            
            for stock in stocks:
                symbol = stock['symbol']
                if symbol in expected_returns:
                    # Weight by expected return and confidence
                    prediction = predictions[symbol]
                    expected_return = expected_returns[symbol]
                    confidence = prediction.get('confidence', 70) / 100
                    
                    # Calculate weight based on expected return and risk profile
                    weight = expected_return * confidence * risk_mult
                    allocations[symbol] = max(0, weight)  # No negative allocations
                    total_weight += allocations[symbol]
            
            # Normalize allocations to sum to 100%
            if total_weight > 0:
                for symbol in allocations:
                    allocations[symbol] = (allocations[symbol] / total_weight) * 100
            else:
                # Equal weight if no positive weights
                equal_weight = 100 / len(stocks)
                for stock in stocks:
                    allocations[stock['symbol']] = equal_weight
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error in ML portfolio optimization: {str(e)}")
            return self.optimize_portfolio_simple(stocks, risk_profile)
    
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
                'volatility': info.get('regularMarketPrice', 0) * 0.02,  # Estimate volatility
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
        
        # Get ML predictions for all stocks
        stock_predictions = {}
        for stock in stocks:
            symbol = stock['symbol']
            prediction = optimizer.predict_stock_price(symbol)
            if 'error' not in prediction:
                stock_predictions[symbol] = prediction
        
        # ML-based optimization using Modern Portfolio Theory
        if len(stock_predictions) > 0:
            allocations = optimizer.optimize_portfolio_ml(stocks, stock_predictions, risk_profile, portfolio_value)
        else:
            # Fallback to simple optimization
            allocations = optimizer.optimize_portfolio_simple(stocks, risk_profile)
        
        return jsonify({
            'allocations': allocations,
            'ml_predictions': stock_predictions,
            'optimization_method': 'ML-based' if len(stock_predictions) > 0 else 'Simple'
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
            
            # Generate individual stock performance based on real current prices
            for stock in stocks:
                symbol = stock['symbol']
                base_price = stock.get('current_price', 100)
                # Simulate realistic price movement with some randomness
                import random
                random.seed(hash(symbol + month))  # Consistent randomness for same stock/month
                change_percent = random.uniform(-0.12, 0.18)  # -12% to +18% for more realistic range
                month_data[symbol] = round(base_price * (1 + change_percent), 2)
            
            # Calculate portfolio total based on allocations
            total_value = 0
            for stock in stocks:
                symbol = stock['symbol']
                allocation = allocations.get(symbol, 0) / 100
                stock_value = month_data[symbol] * allocation * 100  # Assuming 100 shares per stock
                total_value += stock_value
            
            month_data['total'] = round(total_value, 2)
            performance_data.append(month_data)
        
        return jsonify(performance_data)
    except Exception as e:
        logger.error(f"Error in portfolio performance: {str(e)}")
        return jsonify({'error': 'Failed to get performance data'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 