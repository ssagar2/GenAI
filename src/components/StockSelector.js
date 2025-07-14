import React, { useState, useEffect, useCallback } from 'react';
import { Search, Plus, X, TrendingUp, TrendingDown, Loader2 } from 'lucide-react';
import apiService from '../services/apiService';

const StockSelector = ({ selectedStocks, onStockSelection }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Popular stocks for quick access
  const popularStocks = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'DIS', 'PYPL'
  ];

  // Search stocks with debouncing
  useEffect(() => {
    const searchStocks = async () => {
      if (!searchTerm || searchTerm.length < 2) {
        setSearchResults([]);
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        const results = await apiService.searchStocks(searchTerm);
        setSearchResults(results);
      } catch (err) {
        setError('Failed to search stocks. Please try again.');
        console.error('Search error:', err);
      } finally {
        setIsLoading(false);
      }
    };

    const timeoutId = setTimeout(searchStocks, 300);
    return () => clearTimeout(timeoutId);
  }, [searchTerm]);

  const handleStockSelect = async (stock) => {
    if (!selectedStocks.find(s => s.symbol === stock.symbol)) {
      setIsLoading(true);
      try {
        // Get real-time quote for the selected stock
        const quote = await apiService.getStockData(stock.symbol);
        if (quote) {
          onStockSelection([...selectedStocks, quote]);
        }
      } catch (err) {
        console.error('Error fetching stock quote:', err);
        setError('Failed to add stock. Please try again.');
      } finally {
        setIsLoading(false);
      }
    }
    setSearchTerm('');
    setShowDropdown(false);
  };

  const handleStockRemove = (symbol) => {
    const updatedStocks = selectedStocks.filter(stock => stock.symbol !== symbol);
    onStockSelection(updatedStocks);
  };

  const handlePopularStockSelect = async (symbol) => {
    if (!selectedStocks.find(s => s.symbol === symbol)) {
      setIsLoading(true);
      try {
        const quote = await apiService.getStockData(symbol);
        if (quote) {
          onStockSelection([...selectedStocks, quote]);
        }
      } catch (err) {
        console.error('Error fetching popular stock:', err);
        setError('Failed to add stock. Please try again.');
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="space-y-4">
      {/* Search Input */}
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <Search className="h-5 w-5 text-gray-400" />
        </div>
        <input
          type="text"
          className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
          placeholder="Search stocks by symbol or company name..."
          value={searchTerm}
          onChange={(e) => {
            setSearchTerm(e.target.value);
            setShowDropdown(true);
          }}
          onFocus={() => setShowDropdown(true)}
        />
        
        {isLoading && (
          <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
            <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />
          </div>
        )}
        
        {/* Dropdown */}
        {showDropdown && (searchTerm || searchResults.length > 0) && (
          <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-auto">
            {error && (
              <div className="px-4 py-2 text-sm text-red-600 bg-red-50 border-b border-red-200">
                {error}
              </div>
            )}
            
            {searchResults.map((stock) => (
              <div
                key={stock.symbol}
                className="px-4 py-2 hover:bg-gray-50 cursor-pointer border-b border-gray-100 last:border-b-0"
                onClick={() => handleStockSelect(stock)}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-gray-900">{stock.symbol}</div>
                    <div className="text-sm text-gray-500">{stock.name}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium text-gray-900">
                      ${stock.current_price ? stock.current_price.toFixed(2) : 'N/A'}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Popular Stocks */}
      <div className="space-y-2">
        <h3 className="text-sm font-medium text-gray-700">Popular Stocks</h3>
        <div className="grid grid-cols-2 gap-2">
          {popularStocks.map((symbol) => (
            <button
              key={symbol}
              onClick={() => handlePopularStockSelect(symbol)}
              disabled={selectedStocks.find(s => s.symbol === symbol) || isLoading}
              className="px-3 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {symbol}
            </button>
          ))}
        </div>
      </div>

      {/* Selected Stocks */}
      <div className="space-y-2">
        <h3 className="text-sm font-medium text-gray-700">Selected Stocks ({selectedStocks.length})</h3>
        <div className="space-y-2">
          {selectedStocks.map((stock) => (
            <div
              key={stock.symbol}
              className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200"
            >
              <div className="flex items-center space-x-3">
                <div className="bg-blue-100 p-2 rounded-lg">
                  <span className="text-sm font-bold text-blue-600">{stock.symbol}</span>
                </div>
                <div>
                  <div className="text-sm font-medium text-gray-900">{stock.name}</div>
                  <div className="text-xs text-gray-500">{stock.sector || 'Unknown Sector'}</div>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <div className="text-right">
                  <div className="text-sm font-medium text-gray-900">
                    ${stock.current_price ? stock.current_price.toFixed(2) : 'N/A'}
                  </div>
                  <div className={`text-xs flex items-center ${
                    stock.change >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {stock.change !== undefined ? (
                      <>
                        {stock.change >= 0 ? (
                          <TrendingUp className="h-3 w-3 mr-1" />
                        ) : (
                          <TrendingDown className="h-3 w-3 mr-1" />
                        )}
                        {Math.abs(stock.change).toFixed(2)}%
                      </>
                    ) : (
                      'N/A'
                    )}
                  </div>
                </div>
                <button
                  onClick={() => handleStockRemove(stock.symbol)}
                  className="p-1 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-full transition-colors"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
        
        {selectedStocks.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <Plus className="h-8 w-8 mx-auto mb-2 text-gray-300" />
            <p className="text-sm">No stocks selected</p>
            <p className="text-xs">Search and select stocks to build your portfolio</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default StockSelector; 