# ML-Powered Portfolio Optimizer

A professional-grade portfolio optimization tool with React frontend and Flask backend, featuring machine learning predictions and real-time stock data.

## Features

- **ML-Powered Predictions**: Random Forest models for stock price forecasting
- **Real-Time Data**: Yahoo Finance integration with fallback to mock data
- **Risk-Based Optimization**: Dynamic portfolio allocation based on risk profiles
- **Modern UI**: React frontend with Tailwind CSS
- **Professional Backend**: Flask API with comprehensive error handling

## Quick Start

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend server:**
   ```bash
   python start.py
   ```
   
   The server will start at `http://localhost:5000` with relative path logging enabled.

### Frontend Setup

1. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

2. **Start the React development server:**
   ```bash
   npm start
   ```
   
   The frontend will be available at `http://localhost:3000`

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/search?q=<query>` - Search stocks
- `GET /api/stock/<symbol>` - Get stock data
- `POST /api/portfolio/metrics` - Calculate portfolio metrics
- `POST /api/portfolio/optimize` - Optimize portfolio with ML
- `POST /api/portfolio/performance` - Get performance data

## ML Features

- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands
- **Price Prediction**: 30-day forward price predictions
- **Risk Assessment**: Dynamic risk profile adjustments
- **Portfolio Optimization**: Modern Portfolio Theory with ML enhancements

## Project Structure

```
portfolio-optimizer/
├── backend/
│   ├── app.py              # Main Flask application
│   ├── start.py            # Startup script with relative paths
│   └── requirements.txt    # Python dependencies
├── src/
│   ├── components/         # React components
│   └── services/          # API services
└── README.md
```

## Configuration

The backend is configured to use relative paths in logs, making it suitable for deployment and sharing. The Flask development server will show relative file paths instead of absolute paths when files are modified.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

If you encounter any issues or have questions, please open an issue on the project repository.

---

**Happy Portfolio Optimizing! 🚀** 
