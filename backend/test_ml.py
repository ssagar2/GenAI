#!/usr/bin/env python3
"""
Test script for ML-based portfolio optimization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import PortfolioOptimizer

def test_ml_functionality():
    """Test the ML functionality"""
    print("🧪 Testing ML-based Portfolio Optimizer...")
    
    optimizer = PortfolioOptimizer()
    
    # Test stock data retrieval
    print("\n📊 Testing stock data retrieval...")
    stock_data = optimizer.get_stock_data('AAPL')
    if stock_data:
        print(f"✅ AAPL data retrieved: ${stock_data.get('current_price', 0):.2f}")
        if 'ml_prediction' in stock_data:
            prediction = stock_data['ml_prediction']
            if 'error' not in prediction:
                print(f"🤖 ML Prediction: ${prediction.get('predicted_price', 0):.2f}")
                print(f"📈 Expected Change: {prediction.get('prediction_change', 0):.2f}%")
                print(f"🎯 Confidence: {prediction.get('confidence', 0):.1f}%")
            else:
                print(f"⚠️ ML Prediction failed: {prediction['error']}")
    else:
        print("❌ Failed to retrieve AAPL data")
    
    # Test portfolio optimization
    print("\n🎯 Testing portfolio optimization...")
    stocks = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.', 'sector': 'Technology'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corp.', 'sector': 'Technology'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'sector': 'Technology'}
    ]
    
    # Test different risk profiles
    for risk_profile in ['conservative', 'moderate', 'aggressive']:
        print(f"\n🔍 Testing {risk_profile} risk profile...")
        
        # Get predictions
        predictions = {}
        for stock in stocks:
            symbol = stock['symbol']
            prediction = optimizer.predict_stock_price(symbol)
            if 'error' not in prediction:
                predictions[symbol] = prediction
        
        # Optimize portfolio
        if predictions:
            allocations = optimizer.optimize_portfolio_ml(stocks, predictions, risk_profile, 10000)
            print(f"📊 Allocations for {risk_profile}:")
            for symbol, allocation in allocations.items():
                print(f"   {symbol}: {allocation:.1f}%")
        else:
            print("⚠️ No ML predictions available, using simple optimization")
            allocations = optimizer.optimize_portfolio_simple(stocks, risk_profile)
            print(f"📊 Simple allocations for {risk_profile}:")
            for symbol, allocation in allocations.items():
                print(f"   {symbol}: {allocation:.1f}%")
    
    print("\n✅ ML functionality test completed!")

if __name__ == "__main__":
    test_ml_functionality() 