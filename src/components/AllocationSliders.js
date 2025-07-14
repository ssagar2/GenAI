import React from 'react';
import { Percent, Target } from 'lucide-react';

const AllocationSliders = ({ selectedStocks, allocations, onAllocationChange }) => {
  const totalAllocation = Object.values(allocations).reduce((sum, value) => sum + value, 0);
  const isValidAllocation = Math.abs(totalAllocation - 100) < 0.1;

  const handleSliderChange = (symbol, value) => {
    onAllocationChange(symbol, parseFloat(value));
  };

  const getColorForStock = (index) => {
    const colors = [
      'bg-blue-500',
      'bg-green-500',
      'bg-purple-500',
      'bg-orange-500',
      'bg-pink-500',
      'bg-indigo-500',
      'bg-red-500',
      'bg-yellow-500',
      'bg-teal-500',
      'bg-gray-500'
    ];
    return colors[index % colors.length];
  };

  if (selectedStocks.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <Target className="h-8 w-8 mx-auto mb-2 text-gray-300" />
        <p className="text-sm">No stocks selected</p>
        <p className="text-xs">Select stocks to configure allocations</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Total Allocation Display */}
      <div className={`p-3 rounded-lg border-2 ${
        isValidAllocation 
          ? 'border-green-200 bg-green-50' 
          : 'border-red-200 bg-red-50'
      }`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Percent className="h-4 w-4 text-gray-600" />
            <span className="text-sm font-medium text-gray-700">Total Allocation</span>
          </div>
          <div className={`text-sm font-bold ${
            isValidAllocation ? 'text-green-700' : 'text-red-700'
          }`}>
            {totalAllocation.toFixed(1)}%
          </div>
        </div>
        {!isValidAllocation && (
          <p className="text-xs text-red-600 mt-1">
            Total allocation must equal 100%
          </p>
        )}
      </div>

      {/* Individual Stock Allocations */}
      <div className="space-y-4">
        {selectedStocks.map((stock, index) => (
          <div key={stock.symbol} className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${getColorForStock(index)}`}></div>
                <span className="text-sm font-medium text-gray-900">{stock.symbol}</span>
              </div>
              <div className="flex items-center space-x-2">
                <input
                  type="number"
                  min="0"
                  max="100"
                  step="0.1"
                  value={allocations[stock.symbol] || 0}
                  onChange={(e) => handleSliderChange(stock.symbol, e.target.value)}
                  className="w-16 text-right text-sm border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-500">%</span>
              </div>
            </div>
            
            <div className="relative">
              <input
                type="range"
                min="0"
                max="100"
                step="0.1"
                value={allocations[stock.symbol] || 0}
                onChange={(e) => handleSliderChange(stock.symbol, e.target.value)}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                style={{
                  background: `linear-gradient(to right, ${getColorForStock(index).replace('bg-', '')} 0%, ${getColorForStock(index).replace('bg-', '')} ${allocations[stock.symbol] || 0}%, #e5e7eb ${allocations[stock.symbol] || 0}%, #e5e7eb 100%)`
                }}
              />
            </div>
            
            <div className="flex justify-between text-xs text-gray-500">
              <span>0%</span>
              <span>50%</span>
              <span>100%</span>
            </div>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="flex space-x-2 pt-2">
        <button
          onClick={() => {
            const equalAllocation = 100 / selectedStocks.length;
            selectedStocks.forEach(stock => {
              onAllocationChange(stock.symbol, equalAllocation);
            });
          }}
          className="flex-1 px-3 py-2 text-xs font-medium text-blue-700 bg-blue-100 rounded-md hover:bg-blue-200 transition-colors"
        >
          Equal Weight
        </button>
        <button
          onClick={() => {
            selectedStocks.forEach(stock => {
              onAllocationChange(stock.symbol, 0);
            });
          }}
          className="flex-1 px-3 py-2 text-xs font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
        >
          Reset All
        </button>
      </div>

      {/* Portfolio Preview */}
      {selectedStocks.length > 0 && (
        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Portfolio Preview</h4>
          <div className="space-y-1">
            {selectedStocks.map((stock, index) => (
              <div key={stock.symbol} className="flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${getColorForStock(index)}`}></div>
                  <span className="text-gray-600">{stock.symbol}</span>
                </div>
                <span className="font-medium text-gray-900">
                  {allocations[stock.symbol] || 0}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default AllocationSliders; 