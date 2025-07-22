import React, { useState, useEffect, useMemo } from 'react';
import { TrendingUp, TrendingDown, DollarSign, Target, AlertCircle, CheckCircle, Info, Loader2 } from 'lucide-react';
import apiService from '../services/apiService';

const PortfolioSummary = ({ selectedStocks, allocations, riskProfile }) => {
  const [metrics, setMetrics] = useState({
    total_value: 10000,
    total_return: 0,
    expected_return: 0,
    volatility: 0,
    sharpe_ratio: 0,
    sectors: {},
    top_holdings: []
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Calculate portfolio metrics
  const calculateMetrics = async () => {
    if (selectedStocks.length === 0) {
      setMetrics({
        total_value: 10000,
        total_return: 0,
        expected_return: 0,
        volatility: 0,
        sharpe_ratio: 0,
        sectors: {},
        top_holdings: []
      });
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const portfolioMetrics = await apiService.calculatePortfolioMetrics(selectedStocks, allocations, riskProfile);
      setMetrics(portfolioMetrics);
    } catch (err) {
      console.error('Error calculating portfolio metrics:', err);
      setError('Failed to update portfolio metrics');
    } finally {
      setIsLoading(false);
    }
  };

  // Update metrics when stocks, allocations, or risk profile changes
  useEffect(() => {
      calculateMetrics();
  }, [selectedStocks.length, JSON.stringify(allocations), riskProfile]);

  const getRiskProfileColor = (profile) => {
    switch (profile) {
      case 'conservative': return 'text-green-600 bg-green-100';
      case 'moderate': return 'text-blue-600 bg-blue-100';
      case 'aggressive': return 'text-orange-600 bg-orange-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getReturnColor = (returnValue) => {
    return returnValue >= 0 ? 'text-green-600' : 'text-red-600';
  };

  const getSharpeRatioColor = (ratio) => {
    if (ratio >= 1) return 'text-green-600';
    if (ratio >= 0.5) return 'text-yellow-600';
    return 'text-red-600';
  };

  if (selectedStocks.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <Target className="h-8 w-8 mx-auto mb-2 text-gray-300" />
        <p className="text-sm">No portfolio data</p>
        <p className="text-xs">Select stocks to view portfolio summary</p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="text-center py-8 min-h-48 flex items-center justify-center">
        <div>
          <Loader2 className="h-8 w-8 mx-auto mb-2 text-blue-500 animate-spin" />
          <p className="text-sm text-gray-600">Calculating portfolio metrics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8 text-red-500">
        <AlertCircle className="h-8 w-8 mx-auto mb-2 text-red-300" />
        <p className="text-sm">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Portfolio Overview */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Portfolio Value</p>
              <p className="text-2xl font-bold text-gray-900">
                ${(metrics.total_value || 0).toLocaleString()}
              </p>
            </div>
            <DollarSign className="h-8 w-8 text-green-600" />
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Total Return</p>
              <p className={`text-2xl font-bold ${getReturnColor(metrics.total_return || 0)}`}>
                {(metrics.total_return || 0) >= 0 ? '+' : ''}{(metrics.total_return || 0).toFixed(2)}%
              </p>
              <p className="text-xs text-gray-500">Current performance</p>
            </div>
            {(metrics.total_return || 0) >= 0 ? (
              <TrendingUp className="h-8 w-8 text-green-600" />
            ) : (
              <TrendingDown className="h-8 w-8 text-red-600" />
            )}
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Expected Return</p>
              <p className="text-2xl font-bold text-gray-900">
                {(metrics.expected_return || 0).toFixed(2)}%
              </p>
              <p className="text-xs text-gray-500">Annual projection</p>
            </div>
            <Target className="h-8 w-8 text-blue-600" />
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Sharpe Ratio</p>
              <p className={`text-2xl font-bold ${getSharpeRatioColor(metrics.sharpe_ratio || 0)}`}>
                {(metrics.sharpe_ratio || 0).toFixed(2)}
              </p>
              <p className="text-xs text-gray-500">Risk-adjusted return</p>
            </div>
            <TrendingUp className="h-8 w-8 text-purple-600" />
          </div>
        </div>
      </div>

      {/* Risk Profile Impact */}
      <div className="bg-white p-4 rounded-lg border border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium text-gray-700">Risk Profile Impact</h3>
            <p className="text-xs text-gray-500">
              {riskProfile === 'conservative' && 'Conservative profile reduces volatility and returns for stability'}
              {riskProfile === 'moderate' && 'Moderate profile balances risk and return'}
              {riskProfile === 'aggressive' && 'Aggressive profile increases potential returns and volatility'}
            </p>
          </div>
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${getRiskProfileColor(riskProfile)}`}>
            {riskProfile.charAt(0).toUpperCase() + riskProfile.slice(1)}
          </div>
        </div>
      </div>

      {/* Portfolio Details */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Top Holdings */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Holdings</h3>
          <div className="space-y-3">
            {(metrics.top_holdings || []).map((holding, index) => (
              <div key={holding.symbol} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm ${
                    index === 0 ? 'bg-yellow-500' : 
                    index === 1 ? 'bg-gray-400' : 'bg-orange-500'
                  }`}>
                    {index + 1}
                  </div>
                  <div>
                    <div className="font-medium text-gray-900">{holding.symbol}</div>
                    <div className="text-xs text-gray-500">{holding.name}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-medium text-gray-900">{holding.allocation.toFixed(1)}%</div>
                  <div className={`text-xs ${getReturnColor(holding.change)}`}>
                    {holding.change >= 0 ? '+' : ''}{holding.change.toFixed(2)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Sector Allocation */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Sector Allocation</h3>
          <div className="space-y-3">
            {Object.entries(metrics.sectors || {}).map(([sector, allocation], index) => (
              <div key={sector} className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full bg-blue-${(index + 1) * 100}`}></div>
                  <span className="text-sm font-medium text-gray-900">{sector}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-24 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full" 
                      style={{ width: `${allocation}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900 w-12 text-right">
                    {allocation.toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Portfolio Recommendations */}
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Portfolio Recommendations</h3>
        <div className="space-y-3">
          {(metrics.sharpe_ratio || 0) < 0.5 && (
            <div className="flex items-start space-x-3 p-3 bg-red-50 rounded-lg border border-red-200">
              <AlertCircle className="h-5 w-5 text-red-600 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-sm font-medium text-red-800">Low Risk-Adjusted Returns</p>
                <p className="text-xs text-red-600">Consider diversifying your portfolio to improve risk-adjusted returns.</p>
              </div>
            </div>
          )}
          
          {(metrics.volatility || 0) > 20 && (
            <div className="flex items-start space-x-3 p-3 bg-yellow-50 rounded-lg border border-yellow-200">
              <AlertCircle className="h-5 w-5 text-yellow-600 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-sm font-medium text-yellow-800">High Portfolio Volatility</p>
                <p className="text-xs text-yellow-600">Your portfolio shows high volatility. Consider adding defensive stocks.</p>
              </div>
            </div>
          )}
          
          {(metrics.sharpe_ratio || 0) >= 1.0 && (
            <div className="flex items-start space-x-3 p-3 bg-green-50 rounded-lg border border-green-200">
              <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-sm font-medium text-green-800">Excellent Risk-Adjusted Returns</p>
                <p className="text-xs text-green-600">Your portfolio shows strong risk-adjusted performance.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PortfolioSummary; 