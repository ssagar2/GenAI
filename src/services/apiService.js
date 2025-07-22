const API_BASE_URL = 'http://localhost:5000/api';

class ApiService {
  constructor() {
    this.cache = new Map();
    this.cacheTimeout = 10 * 60 * 1000; // 10 minutes
  }

  // Cache management
  getCached(key) {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }
    return null;
  }

  setCached(key, data) {
    this.cache.set(key, {
      data,
      timestamp: Date.now()
    });
  }

  // Generic API call with error handling
  async apiCall(endpoint, options = {}) {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
      });

      if (!response.ok) {
        throw new Error(`API call failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API call error:', error);
      throw error;
    }
  }

  // Health check
  async healthCheck() {
    return this.apiCall('/health');
  }

  // Search stocks
  async searchStocks(query) {
    if (!query || query.length < 2) return [];
    
    const cacheKey = `search_${query}`;
    const cached = this.getCached(cacheKey);
    if (cached) return cached;

    const results = await this.apiCall(`/search?q=${encodeURIComponent(query)}`);
    this.setCached(cacheKey, results);
    return results;
  }

  // Get stock data
  async getStockData(symbol) {
    const cacheKey = `stock_${symbol}`;
    const cached = this.getCached(cacheKey);
    if (cached) return cached;

    const data = await this.apiCall(`/stock/${symbol}`);
    this.setCached(cacheKey, data);
    return data;
  }

  // Calculate portfolio metrics
  async calculatePortfolioMetrics(stocks, allocations, riskProfile = 'moderate') {
    if (!stocks || stocks.length === 0) {
      return {
        total_value: 10000,
        total_return: 0,
        expected_return: 0,
        volatility: 0,
        sharpe_ratio: 0,
        beta: 1.0,
        sectors: {},
        top_holdings: [],
        total_allocation: 0
      };
    }

    const cacheKey = `metrics_${JSON.stringify({ stocks: stocks.map(s => s.symbol), allocations, riskProfile })}`;
    const cached = this.getCached(cacheKey);
    if (cached) return cached;

    const metrics = await this.apiCall('/portfolio/metrics', {
      method: 'POST',
      body: JSON.stringify({ stocks, allocations, risk_profile: riskProfile })
    });
    
    this.setCached(cacheKey, metrics);
    return metrics;
  }

  // Optimize portfolio
  async optimizePortfolio(stocks, riskProfile, portfolioValue = 10000) {
    const result = await this.apiCall('/portfolio/optimize', {
      method: 'POST',
      body: JSON.stringify({ stocks, risk_profile: riskProfile, portfolio_value: portfolioValue })
    });
    // Return allocations, ml_predictions, and optimization_method
    return {
      allocations: result.allocations,
      mlPredictions: result.ml_predictions,
      optimizationMethod: result.optimization_method
    };
  }

  // Monte Carlo portfolio optimization
  async monteCarloOptimize(symbols, riskProfile = 'moderate', nSimulations = 1000) {
    try {
      const result = await this.apiCall('/portfolio/monte-carlo/optimize', {
        method: 'POST',
        body: JSON.stringify({ 
          symbols, 
          risk_profile: riskProfile, 
          n_simulations: nSimulations 
        })
      });
      
      return {
        success: result.success,
        optimizationMethod: 'Monte Carlo Simulation',
        riskProfile: result.risk_profile,
        simulationStats: result.simulation_stats,
        optimalPortfolio: result.optimal_portfolio,
        alternativePortfolios: result.alternative_portfolios,
        efficientFrontierData: result.efficient_frontier_data
      };
    } catch (error) {
      console.error('Monte Carlo optimization failed:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  // Get Monte Carlo statistics
  async getMonteCarloStats(symbols, nSimulations = 1000) {
    try {
      const result = await this.apiCall('/portfolio/monte-carlo/stats', {
        method: 'POST',
        body: JSON.stringify({ 
          symbols, 
          n_simulations: nSimulations 
        })
      });
      
      return result;
    } catch (error) {
      console.error('Monte Carlo stats failed:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  // Get portfolio performance data
  async getPortfolioPerformance(stocks, allocations) {
    const cacheKey = `performance_${JSON.stringify({ stocks: stocks.map(s => s.symbol), allocations })}`;
    const cached = this.getCached(cacheKey);
    if (cached) return cached;

    const performance = await this.apiCall('/portfolio/performance', {
      method: 'POST',
      body: JSON.stringify({ stocks, allocations })
    });
    
    this.setCached(cacheKey, performance);
    return performance;
  }

  // Clear cache
  clearCache() {
    this.cache.clear();
  }
}

export default new ApiService(); 