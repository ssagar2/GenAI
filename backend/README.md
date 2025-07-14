# Portfolio Optimizer Backend

A Python Flask backend that handles all heavy computational work for portfolio optimization, including real-time data fetching, risk calculations, and optimization algorithms.

## Features

- **Real-time Stock Data**: Fetches live data from Yahoo Finance
- **Portfolio Optimization**: Uses advanced algorithms for optimal allocation
- **Risk Assessment**: Calculates volatility, beta, and Sharpe ratios
- **Performance Analysis**: Generates historical performance data
- **Caching**: Intelligent caching to reduce API calls
- **Error Handling**: Robust error handling and logging

## Setup

### 1. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the Backend Server

```bash
python start.py
```

The backend will be available at `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /api/health
```

### Search Stocks
```
GET /api/search?q=<query>
```

### Get Stock Data
```
GET /api/stock/<symbol>
```

### Calculate Portfolio Metrics
```
POST /api/portfolio/metrics
Body: { "stocks": [...], "allocations": {...} }
```

### Optimize Portfolio
```
POST /api/portfolio/optimize
Body: { "stocks": [...], "risk_profile": "moderate" }
```

### Get Performance Data
```
POST /api/portfolio/performance
Body: { "stocks": [...], "allocations": {...} }
```

## Architecture Benefits

### Server-Side Processing
- **Heavy computations** moved from client to server
- **Real-time data fetching** with intelligent caching
- **Advanced algorithms** for portfolio optimization
- **Risk calculations** using financial libraries

### Performance Optimizations
- **Caching layer** to reduce API calls
- **Batch processing** for multiple stock requests
- **Error handling** with graceful fallbacks
- **Memory management** with proper cleanup

### Financial Libraries Used
- `yfinance`: Real-time stock data
- `pandas` & `numpy`: Data processing
- `scipy`: Statistical calculations
- `scikit-learn`: Machine learning algorithms
- `cvxpy`: Portfolio optimization
- `pyportfolioopt`: Advanced portfolio theory

## Development

### Running in Development Mode
```bash
python app.py
```

### Running with Gunicorn (Production)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Environment Variables

Create a `.env` file in the backend directory:

```env
FLASK_ENV=development
FLASK_DEBUG=1
CACHE_TIMEOUT=300
```

## Troubleshooting

### Common Issues

1. **Port 5000 in use**: Change the port in `app.py`
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **API rate limits**: The backend includes caching to minimize API calls

### Logs

The backend includes comprehensive logging. Check the console output for detailed error messages and API call information. 