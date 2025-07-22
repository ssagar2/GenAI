import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import StockSelector from './components/StockSelector';
import AllocationSliders from './components/AllocationSliders';
import RiskAssessment from './components/RiskAssessment';
import PerformanceCharts from './components/PerformanceCharts';
import PortfolioSummary from './components/PortfolioSummary';
import { TrendingUp, PieChart, Shield, BarChart3, Target, Loader2 } from 'lucide-react';
import apiService from './services/apiService';

function App() {
  const [selectedStocks, setSelectedStocks] = useState([]);
  const [allocations, setAllocations] = useState({});
  const [riskProfile, setRiskProfile] = useState('moderate');
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optimizationMethod, setOptimizationMethod] = useState('');
  const [error, setError] = useState(null);

  // Run Monte Carlo optimization when stocks or risk profile changes
  useEffect(() => {
    if (selectedStocks.length > 0) {
      runMonteCarloOptimization();
    }
  }, [selectedStocks, riskProfile]);

  const runMonteCarloOptimization = async () => {
    if (selectedStocks.length === 0) return;

    setIsOptimizing(true);
    setError(null);

    try {
      // Use the existing portfolio optimization endpoint with Monte Carlo integration
      const result = await apiService.optimizePortfolio(
        selectedStocks,
        riskProfile,
        10000
      );

      if (result.allocations) {
        setAllocations(result.allocations);
        setOptimizationMethod(result.optimizationMethod || 'Monte Carlo');
      } else {
        // Fallback to equal weight allocation
        const equalAllocation = 100 / selectedStocks.length;
        const fallbackAllocations = {};
        selectedStocks.forEach(stock => {
          fallbackAllocations[stock.symbol] = equalAllocation;
        });
        setAllocations(fallbackAllocations);
        setOptimizationMethod('Equal Weight (Fallback)');
      }
    } catch (error) {
      console.error('Monte Carlo optimization failed:', error);
      setError('Failed to optimize portfolio. Using equal weight allocation.');
      
      // Fallback to equal weight allocation
      const equalAllocation = 100 / selectedStocks.length;
      const fallbackAllocations = {};
      selectedStocks.forEach(stock => {
        fallbackAllocations[stock.symbol] = equalAllocation;
      });
      setAllocations(fallbackAllocations);
      setOptimizationMethod('Equal Weight (Fallback)');
    } finally {
      setIsOptimizing(false);
    }
  };

  const handleStockSelection = (stocks) => {
    setSelectedStocks(stocks);
  };

  const handleAllocationChange = (symbol, value) => {
    setAllocations(prev => ({
      ...prev,
      [symbol]: value
    }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-r from-blue-600 to-indigo-600 p-2 rounded-lg">
                <TrendingUp className="h-6 w-6 text-white" />
              </div>
              <h1 className="text-2xl font-bold text-gray-900">Portfolio Optimizer</h1>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-500">Monte Carlo Portfolio Management</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 lg:gap-8">
          {/* Left Column - Stock Selection and Allocation */}
          <div className="xl:col-span-1 space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="bg-white rounded-xl shadow-lg p-6"
            >
              <div className="flex items-center space-x-2 mb-4">
                <Target className="h-5 w-5 text-blue-600" />
                <h2 className="text-lg font-semibold text-gray-900">Stock Selection</h2>
              </div>
              <StockSelector 
                selectedStocks={selectedStocks}
                onStockSelection={handleStockSelection}
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="bg-white rounded-xl shadow-lg p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                <PieChart className="h-5 w-5 text-green-600" />
                <h2 className="text-lg font-semibold text-gray-900">Allocation</h2>
                </div>
                {isOptimizing && (
                  <div className="flex items-center space-x-2 text-sm text-blue-600">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>Optimizing...</span>
                  </div>
                )}
              </div>
              
              {optimizationMethod && (
                <div className="mb-4 p-2 bg-blue-50 rounded-lg border border-blue-200">
                  <p className="text-xs text-blue-700">
                    <strong>Method:</strong> {optimizationMethod}
                  </p>
                </div>
              )}
              
              <AllocationSliders
                selectedStocks={selectedStocks}
                allocations={allocations}
                onAllocationChange={handleAllocationChange}
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="bg-white rounded-xl shadow-lg p-6"
            >
              <div className="flex items-center space-x-2 mb-4">
                <Shield className="h-5 w-5 text-orange-600" />
                <h2 className="text-lg font-semibold text-gray-900">Risk Assessment</h2>
              </div>
              <RiskAssessment
                riskProfile={riskProfile}
                onRiskChange={setRiskProfile}
                selectedStocks={selectedStocks}
                allocations={allocations}
              />
            </motion.div>
          </div>

          {/* Right Column - Charts and Summary */}
          <div className="xl:col-span-2 space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="bg-white rounded-xl shadow-lg p-6"
            >
              <div className="flex items-center space-x-2 mb-4">
                <BarChart3 className="h-5 w-5 text-purple-600" />
                <h2 className="text-lg font-semibold text-gray-900">Performance Analysis</h2>
              </div>
              <PerformanceCharts
                selectedStocks={selectedStocks}
                allocations={allocations}
                riskProfile={riskProfile}
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="bg-white rounded-xl shadow-lg p-6"
            >
              <div className="flex items-center space-x-2 mb-4">
                <TrendingUp className="h-5 w-5 text-indigo-600" />
                <h2 className="text-lg font-semibold text-gray-900">Portfolio Summary</h2>
              </div>
              <PortfolioSummary
                selectedStocks={selectedStocks}
                allocations={allocations}
                riskProfile={riskProfile}
              />
            </motion.div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App; 