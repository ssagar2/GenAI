import axios from 'axios';

class StockAPI {
  constructor() {
    this.cache = new Map();
    this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
    this.baseURL = 'https://query1.finance.yahoo.com/v8/finance';
  }

  // Get real-time quote for a single stock
  async getQuote(symbol) {
    try {
      const cacheKey = `quote_${symbol}`;
      const cached = this.getCachedData(cacheKey);
      if (cached) return cached;

      const response = await axios.get(`${this.baseURL}/chart/${symbol}`, {
        params: {
          interval: '1d',
          range: '1d'
        }
      });

      const result = response.data.chart.result[0];
      const quote = result.indicators.quote[0];
      const meta = result.meta;

      const stockData = {
        symbol: symbol.toUpperCase(),
        name: meta.symbol,
        price: quote.close[quote.close.length - 1],
        change: ((quote.close[quote.close.length - 1] - quote.open[0]) / quote.open[0]) * 100,
        previousClose: quote.open[0],
        volume: quote.volume[quote.volume.length - 1],
        sector: this.getSectorForSymbol(symbol),
        industry: this.getIndustryForSymbol(symbol),
        currency: 'USD',
        exchange: 'NASDAQ',
        timestamp: new Date().toISOString()
      };

      this.setCachedData(cacheKey, stockData);
      return stockData;
    } catch (error) {
      console.error(`Error fetching quote for ${symbol}:`, error);
      // Return mock data as fallback
      return this.getMockQuote(symbol);
    }
  }

  // Get quotes for multiple stocks
  async getQuotes(symbols) {
    try {
      const promises = symbols.map(symbol => this.getQuote(symbol));
      const results = await Promise.allSettled(promises);
      
      return results
        .map((result, index) => {
          if (result.status === 'fulfilled' && result.value) {
            return result.value;
          }
          return this.getMockQuote(symbols[index]);
        })
        .filter(Boolean);
    } catch (error) {
      console.error('Error fetching multiple quotes:', error);
      return symbols.map(symbol => this.getMockQuote(symbol));
    }
  }

  // Search for stocks
  async searchStocks(query) {
    try {
      const cacheKey = `search_${query}`;
      const cached = this.getCachedData(cacheKey);
      if (cached) return cached;

      // For now, return popular stocks that match the query
      const popularStocks = [
        { symbol: 'AAPL', name: 'Apple Inc.', exchange: 'NASDAQ' },
        { symbol: 'MSFT', name: 'Microsoft Corporation', exchange: 'NASDAQ' },
        { symbol: 'GOOGL', name: 'Alphabet Inc.', exchange: 'NASDAQ' },
        { symbol: 'AMZN', name: 'Amazon.com Inc.', exchange: 'NASDAQ' },
        { symbol: 'TSLA', name: 'Tesla Inc.', exchange: 'NASDAQ' },
        { symbol: 'META', name: 'Meta Platforms Inc.', exchange: 'NASDAQ' },
        { symbol: 'NVDA', name: 'NVIDIA Corporation', exchange: 'NASDAQ' },
        { symbol: 'NFLX', name: 'Netflix Inc.', exchange: 'NASDAQ' },
        { symbol: 'JPM', name: 'JPMorgan Chase & Co.', exchange: 'NYSE' },
        { symbol: 'JNJ', name: 'Johnson & Johnson', exchange: 'NYSE' },
        { symbol: 'V', name: 'Visa Inc.', exchange: 'NYSE' },
        { symbol: 'PG', name: 'Procter & Gamble Co.', exchange: 'NYSE' },
        { symbol: 'UNH', name: 'UnitedHealth Group Inc.', exchange: 'NYSE' },
        { symbol: 'HD', name: 'Home Depot Inc.', exchange: 'NYSE' },
        { symbol: 'DIS', name: 'Walt Disney Co.', exchange: 'NYSE' },
        { symbol: 'PYPL', name: 'PayPal Holdings Inc.', exchange: 'NASDAQ' }
      ];

      const results = popularStocks.filter(stock => 
        stock.symbol.toLowerCase().includes(query.toLowerCase()) ||
        stock.name.toLowerCase().includes(query.toLowerCase())
      );

      this.setCachedData(cacheKey, results);
      return results;
    } catch (error) {
      console.error('Error searching stocks:', error);
      return [];
    }
  }

  // Get historical data for performance charts
  async getHistoricalData(symbol, period = '6mo') {
    try {
      const cacheKey = `historical_${symbol}_${period}`;
      const cached = this.getCachedData(cacheKey);
      if (cached) return cached;

      const response = await axios.get(`${this.baseURL}/chart/${symbol}`, {
        params: {
          interval: '1d',
          range: period
        }
      });

      const result = response.data.chart.result[0];
      const timestamps = result.timestamp;
      const quotes = result.indicators.quote[0];

      const historicalData = timestamps.map((timestamp, index) => ({
        date: new Date(timestamp * 1000),
        close: quotes.close[index],
        volume: quotes.volume[index],
        high: quotes.high[index],
        low: quotes.low[index],
        open: quotes.open[index]
      })).filter(item => item.close !== null);

      this.setCachedData(cacheKey, historicalData);
      return historicalData;
    } catch (error) {
      console.error(`Error fetching historical data for ${symbol}:`, error);
      return this.getMockHistoricalData(symbol, period);
    }
  }

  // Calculate portfolio metrics
  calculatePortfolioMetrics(stocks, allocations) {
    if (!stocks || stocks.length === 0) {
      return {
        totalValue: 0,
        totalReturn: 0,
        weightedReturn: 0,
        sectors: {},
        topHoldings: []
      };
    }

    let totalValue = 0;
    let totalReturn = 0;
    const sectors = {};
    const holdings = [];

    stocks.forEach(stock => {
      const allocation = allocations[stock.symbol] || 0;
      const stockValue = (allocation / 100) * 100000; // Assuming $100k portfolio
      const stockReturn = stock.change || 0;
      
      totalValue += stockValue;
      totalReturn += (stockReturn * allocation / 100);

      // Track sectors
      const sector = stock.sector || 'Unknown';
      sectors[sector] = (sectors[sector] || 0) + allocation;

      holdings.push({
        symbol: stock.symbol,
        name: stock.name,
        allocation,
        price: stock.price,
        change: stockReturn,
        value: stockValue
      });
    });

    // Sort holdings by allocation
    holdings.sort((a, b) => b.allocation - a.allocation);

    return {
      totalValue,
      totalReturn,
      weightedReturn: totalReturn,
      sectors,
      topHoldings: holdings.slice(0, 5)
    };
  }

  // Helper methods
  getSectorForSymbol(symbol) {
    const sectorMap = {
      'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
      'AMZN': 'Consumer Discretionary', 'TSLA': 'Automotive',
      'META': 'Technology', 'NVDA': 'Technology', 'NFLX': 'Technology',
      'JPM': 'Financial', 'JNJ': 'Healthcare', 'V': 'Financial',
      'PG': 'Consumer Staples', 'UNH': 'Healthcare', 'HD': 'Consumer Discretionary',
      'DIS': 'Consumer Discretionary', 'PYPL': 'Financial'
    };
    return sectorMap[symbol.toUpperCase()] || 'Technology';
  }

  getIndustryForSymbol(symbol) {
    const industryMap = {
      'AAPL': 'Consumer Electronics', 'MSFT': 'Software',
      'GOOGL': 'Internet Services', 'AMZN': 'E-commerce',
      'TSLA': 'Automotive', 'META': 'Social Media',
      'NVDA': 'Semiconductors', 'NFLX': 'Streaming Services',
      'JPM': 'Banking', 'JNJ': 'Pharmaceuticals',
      'V': 'Payment Processing', 'PG': 'Consumer Goods',
      'UNH': 'Healthcare Insurance', 'HD': 'Home Improvement',
      'DIS': 'Entertainment', 'PYPL': 'Payment Processing'
    };
    return industryMap[symbol.toUpperCase()] || 'Technology';
  }

  // Mock data fallbacks
  getMockQuote(symbol) {
    const mockPrices = {
      'AAPL': { price: 150.25, change: 2.5 },
      'MSFT': { price: 320.10, change: -1.2 },
      'GOOGL': { price: 2750.50, change: 5.8 },
      'AMZN': { price: 3200.75, change: -0.8 },
      'TSLA': { price: 850.30, change: 12.5 },
      'META': { price: 280.45, change: 3.2 },
      'NVDA': { price: 450.80, change: 8.7 },
      'NFLX': { price: 380.20, change: -2.1 },
      'JPM': { price: 145.60, change: 1.3 },
      'JNJ': { price: 165.40, change: -0.5 },
      'V': { price: 245.80, change: 3.2 },
      'PG': { price: 135.90, change: 0.8 },
      'UNH': { price: 485.20, change: 4.1 },
      'HD': { price: 320.75, change: 1.9 },
      'DIS': { price: 85.30, change: -1.5 },
      'PYPL': { price: 58.90, change: 2.8 }
    };

    const mock = mockPrices[symbol.toUpperCase()] || { price: 100, change: 0 };
    
    return {
      symbol: symbol.toUpperCase(),
      name: `${symbol.toUpperCase()} Corporation`,
      price: mock.price,
      change: mock.change,
      previousClose: mock.price * (1 - mock.change / 100),
      volume: 1000000,
      sector: this.getSectorForSymbol(symbol),
      industry: this.getIndustryForSymbol(symbol),
      currency: 'USD',
      exchange: 'NASDAQ',
      timestamp: new Date().toISOString()
    };
  }

  getMockHistoricalData(symbol, period) {
    const months = 6;
    const data = [];
    const basePrice = this.getMockQuote(symbol).price;
    
    for (let i = 0; i < months; i++) {
      const date = new Date();
      date.setMonth(date.getMonth() - (months - i - 1));
      
      data.push({
        date,
        close: basePrice * (1 + (Math.random() - 0.5) * 0.2),
        volume: 1000000 + Math.random() * 5000000,
        high: basePrice * 1.1,
        low: basePrice * 0.9,
        open: basePrice
      });
    }
    
    return data;
  }

  // Cache management
  getCachedData(key) {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }
    return null;
  }

  setCachedData(key, data) {
    this.cache.set(key, {
      data,
      timestamp: Date.now()
    });
  }

  // Clear cache
  clearCache() {
    this.cache.clear();
  }
}

export default new StockAPI(); 