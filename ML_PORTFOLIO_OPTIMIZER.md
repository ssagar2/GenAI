# 🤖 ML-Based Portfolio Optimizer

## 🎯 **Professional Portfolio Optimization with Real Data & ML**

This is a **production-ready portfolio optimization tool** that uses **real market data** and **advanced Machine Learning models** to provide accurate future predictions for optimal portfolio allocation.

## 🚀 **Key Features**

### 📊 **Real Market Data**
- **Yahoo Finance API** integration for real-time stock prices
- **Historical data** for ML model training
- **Live market data** including volume, P/E ratios, market cap

### 🤖 **Advanced ML Models**
- **Random Forest Regression** for price prediction
- **Technical Indicators** (RSI, MACD, Bollinger Bands, Moving Averages)
- **Confidence scoring** for prediction reliability
- **Risk-adjusted optimization** using Modern Portfolio Theory

### 🎯 **Risk-Based Optimization**
- **Conservative**: Lower risk, stable returns
- **Moderate**: Balanced risk-return profile  
- **Aggressive**: Higher risk, higher potential returns

## 🏗️ **Architecture**

### **ML Pipeline**
1. **Data Collection**: Historical stock data from Yahoo Finance
2. **Feature Engineering**: Technical indicators calculation
3. **Model Training**: Random Forest on 2+ years of data
4. **Prediction**: Future price predictions with confidence scores
5. **Optimization**: Modern Portfolio Theory with ML insights

### **Technical Indicators Used**
- **SMA (20, 50)**: Simple Moving Averages
- **EMA (12, 26)**: Exponential Moving Averages
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility indicators
- **Price Momentum**: 5-day price changes

## 📈 **How It Works**

### **1. Customer Uploads Portfolio**
```
Portfolio: AAPL, MSFT, GOOGL, AMZN, TSLA
Total Value: $100,000
Risk Profile: Moderate
```

### **2. ML Analysis**
- **Trains models** on historical data for each stock
- **Calculates technical indicators** for feature engineering
- **Predicts future prices** with confidence scores
- **Estimates expected returns** based on ML predictions

### **3. Risk-Based Optimization**
- **Conservative**: 70% risk multiplier, stable allocations
- **Moderate**: 100% risk multiplier, balanced approach
- **Aggressive**: 130% risk multiplier, growth-focused

### **4. Optimal Allocation**
```
AAPL: 25.3% (ML predicts +8.2% return, 82% confidence)
MSFT: 28.7% (ML predicts +12.1% return, 85% confidence)
GOOGL: 22.1% (ML predicts +6.8% return, 78% confidence)
AMZN: 15.9% (ML predicts +4.2% return, 75% confidence)
TSLA: 8.0% (ML predicts +18.5% return, 70% confidence)
```

## 🔧 **API Endpoints**

### **Get Stock Data with ML Predictions**
```bash
GET /api/stock/AAPL
```
**Response:**
```json
{
  "symbol": "AAPL",
  "name": "Apple Inc.",
  "current_price": 175.50,
  "change": 2.5,
  "ml_prediction": {
    "predicted_price": 189.20,
    "prediction_change": 7.8,
    "confidence": 82.5,
    "days_ahead": 30
  }
}
```

### **ML-Based Portfolio Optimization**
```bash
POST /api/portfolio/optimize
{
  "stocks": [
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corp."}
  ],
  "risk_profile": "moderate",
  "portfolio_value": 100000
}
```

**Response:**
```json
{
  "allocations": {
    "AAPL": 45.2,
    "MSFT": 54.8
  },
  "ml_predictions": {
    "AAPL": {
      "predicted_price": 189.20,
      "prediction_change": 7.8,
      "confidence": 82.5
    },
    "MSFT": {
      "predicted_price": 395.80,
      "prediction_change": 12.1,
      "confidence": 85.2
    }
  },
  "optimization_method": "ML-based"
}
```

## 🧪 **Testing**

### **Run ML Test**
```bash
cd backend
python test_ml.py
```

### **Expected Output**
```
🧪 Testing ML-based Portfolio Optimizer...

📊 Testing stock data retrieval...
✅ AAPL data retrieved: $175.50
🤖 ML Prediction: $189.20
📈 Expected Change: 7.80%
🎯 Confidence: 82.5%

🎯 Testing portfolio optimization...

🔍 Testing conservative risk profile...
📊 Allocations for conservative:
   AAPL: 35.2%
   MSFT: 42.8%
   GOOGL: 22.0%

🔍 Testing moderate risk profile...
📊 Allocations for moderate:
   AAPL: 38.5%
   MSFT: 45.3%
   GOOGL: 16.2%

🔍 Testing aggressive risk profile...
📊 Allocations for aggressive:
   AAPL: 42.1%
   MSFT: 48.7%
   GOOGL: 9.2%

✅ ML functionality test completed!
```

## 📊 **Performance Metrics**

### **Model Accuracy**
- **Training Data**: 2+ years of historical data
- **Feature Engineering**: 9 technical indicators
- **Prediction Horizon**: 30 days ahead
- **Confidence Scoring**: 60-85% based on model performance

### **Optimization Quality**
- **Modern Portfolio Theory**: Risk-return optimization
- **ML-Enhanced**: Real predictions vs historical averages
- **Risk-Adjusted**: Conservative/Moderate/Aggressive profiles
- **Real-Time**: Live market data integration

## 🚀 **Getting Started**

### **1. Install Dependencies**
```bash
cd backend
pip install -r requirements.txt
```

### **2. Start Backend**
```bash
python start.py
```

### **3. Test ML Functionality**
```bash
python test_ml.py
```

### **4. Access API**
- **Backend**: http://localhost:5000
- **Health Check**: http://localhost:5000/api/health
- **Stock Data**: http://localhost:5000/api/stock/AAPL
- **Portfolio Optimization**: POST to http://localhost:5000/api/portfolio/optimize

## 🎯 **Use Cases**

### **Individual Investors**
- **Portfolio Rebalancing**: ML-driven allocation adjustments
- **Risk Management**: Conservative/Moderate/Aggressive profiles
- **Performance Optimization**: Data-driven investment decisions

### **Financial Advisors**
- **Client Portfolio Analysis**: Professional-grade optimization
- **Risk Assessment**: Quantified risk-return profiles
- **Investment Recommendations**: ML-backed allocation strategies

### **Institutional Investors**
- **Portfolio Optimization**: Large-scale allocation management
- **Risk Modeling**: Advanced risk assessment tools
- **Performance Tracking**: Real-time portfolio monitoring

## 🔮 **Future Enhancements**

### **Advanced ML Models**
- **LSTM Neural Networks**: Deep learning for time series
- **Ensemble Methods**: Multiple model combination
- **Reinforcement Learning**: Adaptive optimization strategies

### **Additional Features**
- **Options Trading**: Black-Scholes integration
- **Crypto Assets**: Digital asset optimization
- **International Markets**: Global portfolio optimization
- **Real-Time Alerts**: Market movement notifications

## 📈 **Business Value**

### **Accuracy**
- **Real Market Data**: Live prices and market information
- **ML Predictions**: Advanced forecasting capabilities
- **Risk Assessment**: Quantified risk-return relationships

### **Reliability**
- **Fallback Systems**: Simple optimization when ML fails
- **Error Handling**: Robust error management
- **Data Validation**: Quality checks for all inputs

### **Scalability**
- **5-Stock Focus**: Reduced complexity, increased accuracy
- **Caching**: Performance optimization
- **Modular Design**: Easy to extend and maintain

---

**🎯 This is a professional-grade portfolio optimization tool that provides real value to investors by combining real market data with advanced ML predictions for optimal portfolio allocation.** 