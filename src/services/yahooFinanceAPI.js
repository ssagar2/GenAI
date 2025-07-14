import yahooFinance from 'yahoo-finance2';

class YahooFinanceAPI {
  constructor() {
    this.cache = new Map();
    this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
  }

  // Get real-time quote for a single stock
  async getQuote(symbol) {
    try {
      const cacheKey = `quote_${symbol}`;
      const cached = this.getCachedData(cacheKey);
      if (cached) return cached;

      const quote = await yahooFinance.quote(symbol);
      const result = {
        symbol: quote.symbol,
        name: quote.longName || quote.shortName,
        price: quote.regularMarketPrice,
        change: quote.regularMarketChangePercent,
        previousClose: quote.regularMarketPreviousClose,
        marketCap: quote.marketCap,
        volume: quote.volume,
        sector: quote.sector || 'Unknown',
        industry: quote.industry || 'Unknown',
        currency: quote.currency,
        exchange: quote.exchange,
        timestamp: new Date().toISOString()
      };

      this.setCachedData(cacheKey, result);
      return result;
    } catch (error) {
      console.error(`Error fetching quote for ${symbol}:`, error);
      return null;
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
          return null;
        })
        .filter(Boolean);
    } catch (error) {
      console.error('Error fetching multiple quotes:', error);
      return [];
    }
  }

  // Get historical data for performance charts
  async getHistoricalData(symbol, period = '6mo') {
    try {
      const cacheKey = `historical_${symbol}_${period}`;
      const cached = this.getCachedData(cacheKey);
      if (cached) return cached;

      const historical = await yahooFinance.historical(symbol, {
        period1: this.getDateFromPeriod(period),
        period2: new Date(),
        interval: '1d'
      });

      const result = historical.map(item => ({
        date: item.date,
        close: item.close,
        volume: item.volume,
        high: item.high,
        low: item.low,
        open: item.open
      }));

      this.setCachedData(cacheKey, result);
      return result;
    } catch (error) {
      console.error(`Error fetching historical data for ${symbol}:`, error);
      return [];
    }
  }

  // Get sector and industry information
  async getSectorInfo(symbol) {
    try {
      const quote = await this.getQuote(symbol);
      return {
        sector: quote?.sector || 'Unknown',
        industry: quote?.industry || 'Unknown'
      };
    } catch (error) {
      console.error(`Error fetching sector info for ${symbol}:`, error);
      return { sector: 'Unknown', industry: 'Unknown' };
    }
  }

  // Search for stocks
  async searchStocks(query) {
    try {
      const cacheKey = `search_${query}`;
      const cached = this.getCachedData(cacheKey);
      if (cached) return cached;

      // Using Yahoo Finance search functionality
      const searchResults = await yahooFinance.search(query, {
        quotesCount: 10,
        newsCount: 0
      });

      const results = searchResults.quotes.map(quote => ({
        symbol: quote.symbol,
        name: quote.longname || quote.shortname,
        exchange: quote.exchange,
        type: quote.quoteType
      }));

      this.setCachedData(cacheKey, results);
      return results;
    } catch (error) {
      console.error('Error searching stocks:', error);
      return [];
    }
  }

  // Get market summary data
  async getMarketSummary() {
    try {
      const cacheKey = 'market_summary';
      const cached = this.getCachedData(cacheKey);
      if (cached) return cached;

      // Get major indices
      const indices = ['^GSPC', '^DJI', '^IXIC', '^RUT'];
      const indexQuotes = await this.getQuotes(indices);

      const result = {
        sp500: indexQuotes.find(q => q.symbol === '^GSPC'),
        dowJones: indexQuotes.find(q => q.symbol === '^DJI'),
        nasdaq: indexQuotes.find(q => q.symbol === '^IXIC'),
        russell2000: indexQuotes.find(q => q.symbol === '^RUT'),
        timestamp: new Date().toISOString()
      };

      this.setCachedData(cacheKey, result);
      return result;
    } catch (error) {
      console.error('Error fetching market summary:', error);
      return null;
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

  // Helper method to get date from period string
  getDateFromPeriod(period) {
    const now = new Date();
    switch (period) {
      case '1mo':
        return new Date(now.getFullYear(), now.getMonth() - 1, now.getDate());
      case '3mo':
        return new Date(now.getFullYear(), now.getMonth() - 3, now.getDate());
      case '6mo':
        return new Date(now.getFullYear(), now.getMonth() - 6, now.getDate());
      case '1y':
        return new Date(now.getFullYear() - 1, now.getMonth(), now.getDate());
      case '2y':
        return new Date(now.getFullYear() - 2, now.getMonth(), now.getDate());
      default:
        return new Date(now.getFullYear(), now.getMonth() - 6, now.getDate());
    }
  }

  // Clear cache
  clearCache() {
    this.cache.clear();
  }
}

export default new YahooFinanceAPI(); 