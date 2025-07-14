import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, PieChart as PieChartIcon, BarChart3, LineChart as LineChartIcon, Loader2 } from 'lucide-react';
import apiService from '../services/apiService';

const PerformanceCharts = ({ selectedStocks, allocations, riskProfile }) => {
  const [activeChart, setActiveChart] = useState('performance');
  const [performanceData, setPerformanceData] = useState([]);
  const [sectorData, setSectorData] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const updateTimeoutRef = useRef(null);

  // Memoize chart data to prevent unnecessary re-renders
  const memoizedPerformanceData = useMemo(() => performanceData, [performanceData]);
  const memoizedSectorData = useMemo(() => sectorData, [sectorData]);

  // Generate performance data from backend
  const generatePerformanceData = useCallback(async () => {
    if (selectedStocks.length === 0) {
      setPerformanceData([]);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const performanceData = await apiService.getPortfolioPerformance(selectedStocks, allocations);
      setPerformanceData(performanceData);
      setLastUpdate(new Date());
    } catch (err) {
      console.error('Error generating performance data:', err);
      setError('Failed to load performance data');
    } finally {
      setIsLoading(false);
    }
  }, [selectedStocks, allocations]);

  // Generate sector data from real-time stock data
  const generateSectorData = useCallback(() => {
    if (selectedStocks.length === 0) {
      setSectorData([]);
      return;
    }

    const sectorMap = {};
    selectedStocks.forEach(stock => {
      const allocation = allocations[stock.symbol] || 0;
      const sector = stock.sector || 'Unknown';
      sectorMap[sector] = (sectorMap[sector] || 0) + allocation;
    });

    const sectorData = Object.entries(sectorMap).map(([sector, allocation]) => ({
      name: sector,
      value: allocation,
      fill: getSectorColor(sector)
    }));

    setSectorData(sectorData);
  }, [selectedStocks, allocations]);

  // Update data less frequently - only when stocks or allocations change
  useEffect(() => {
    if (selectedStocks.length > 0) {
      // Clear any existing timeout
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
      
      // Debounce the update to prevent rapid re-renders
      updateTimeoutRef.current = setTimeout(() => {
        generatePerformanceData();
        generateSectorData();
      }, 500); // 500ms debounce
    }
    
    return () => {
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
    };
  }, [selectedStocks.length, JSON.stringify(allocations)]);

  // Update data every 5 minutes instead of 2 minutes
  useEffect(() => {
    if (selectedStocks.length === 0) return;
    
    const interval = setInterval(() => {
      generatePerformanceData();
      generateSectorData();
    }, 300000); // 5 minutes

    return () => clearInterval(interval);
  }, [selectedStocks.length]);

  const getSectorColor = (sector) => {
    const colors = {
      'Technology': '#3B82F6',
      'Healthcare': '#10B981',
      'Financial': '#F59E0B',
      'Consumer Discretionary': '#EF4444',
      'Consumer Staples': '#8B5CF6',
      'Automotive': '#06B6D4',
      'Energy': '#F97316',
      'Industrial': '#6B7280',
      'Unknown': '#9CA3AF'
    };
    return colors[sector] || '#9CA3AF';
  };

  const getStockColor = (index) => {
    const colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4', '#F97316', '#6B7280'];
    return colors[index % colors.length];
  };

  const chartTypes = [
    { id: 'performance', name: 'Performance', icon: LineChartIcon },
    { id: 'allocation', name: 'Allocation', icon: PieChartIcon },
    { id: 'sectors', name: 'Sectors', icon: BarChart3 }
  ];

  if (selectedStocks.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        <TrendingUp className="h-12 w-12 mx-auto mb-4 text-gray-300" />
        <p className="text-lg font-medium mb-2">No Data Available</p>
        <p className="text-sm">Select stocks to view performance charts</p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="text-center py-12 min-h-64 lg:min-h-80 flex items-center justify-center">
        <div>
          <Loader2 className="h-12 w-12 mx-auto mb-4 text-blue-500 animate-spin" />
          <p className="text-lg font-medium text-gray-700 mb-2">Loading Real-time Data</p>
          <p className="text-sm text-gray-500">Fetching latest market data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12 text-red-500">
        <TrendingUp className="h-12 w-12 mx-auto mb-4 text-red-300" />
        <p className="text-lg font-medium mb-2">Data Error</p>
        <p className="text-sm">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Chart Type Selector */}
      <div className="flex flex-wrap gap-2">
        {chartTypes.map((chart) => {
          const Icon = chart.icon;
          return (
            <button
              key={chart.id}
              onClick={() => setActiveChart(chart.id)}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeChart === chart.id
                  ? 'bg-blue-100 text-blue-700 border border-blue-200'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              <Icon className="h-4 w-4" />
              <span className="hidden sm:inline">{chart.name}</span>
            </button>
          );
        })}
      </div>

      {/* Chart Container */}
      <div className="bg-gray-50 rounded-lg p-4">
        <div className="h-64 lg:h-80">
          {activeChart === 'performance' && memoizedPerformanceData.length > 0 && (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={memoizedPerformanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                <XAxis dataKey="month" stroke="#6B7280" />
                <YAxis stroke="#6B7280" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #E5E7EB',
                    borderRadius: '8px'
                  }}
                  formatter={(value, name) => [
                    `$${value.toFixed(2)}`, 
                    name === 'total' ? 'Portfolio Total' : name
                  ]}
                />
                <Legend />
                {selectedStocks.map((stock, index) => (
                  <Line
                    key={stock.symbol}
                    type="monotone"
                    dataKey={stock.symbol}
                    stroke={getStockColor(index)}
                    strokeWidth={2}
                    dot={{ fill: getStockColor(index), strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6, stroke: getStockColor(index), strokeWidth: 2 }}
                  />
                ))}
                <Line
                  type="monotone"
                  dataKey="total"
                  stroke="#1F2937"
                  strokeWidth={3}
                  strokeDasharray="5 5"
                  dot={{ fill: '#1F2937', strokeWidth: 2, r: 5 }}
                  activeDot={{ r: 7, stroke: '#1F2937', strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          )}

          {activeChart === 'allocation' && selectedStocks.length > 0 && (
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={selectedStocks.map((stock, index) => ({
                    name: stock.symbol,
                    value: allocations[stock.symbol] || 0,
                    fill: getStockColor(index)
                  }))}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  innerRadius={40}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {selectedStocks.map((stock, index) => (
                    <Cell key={`cell-${index}`} fill={getStockColor(index)} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #E5E7EB',
                    borderRadius: '8px'
                  }}
                  formatter={(value) => [`${value}%`, 'Allocation']}
                />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          )}

          {activeChart === 'sectors' && memoizedSectorData.length > 0 && (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={memoizedSectorData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                <XAxis dataKey="name" stroke="#6B7280" />
                <YAxis stroke="#6B7280" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #E5E7EB',
                    borderRadius: '8px'
                  }}
                  formatter={(value) => [`${value}%`, 'Allocation']}
                />
                <Bar dataKey="value" fill="#3B82F6" radius={[4, 4, 0, 0]}>
                  {memoizedSectorData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Chart Info */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
        <div className="bg-white p-3 rounded-lg border border-gray-200">
          <div className="text-gray-500 mb-1">Total Stocks</div>
          <div className="text-2xl font-bold text-gray-900">{selectedStocks.length}</div>
        </div>
        <div className="bg-white p-3 rounded-lg border border-gray-200">
          <div className="text-gray-500 mb-1">Sectors</div>
          <div className="text-2xl font-bold text-gray-900">
            {[...new Set(selectedStocks.map(stock => stock.sector))].length}
          </div>
        </div>
        <div className="bg-white p-3 rounded-lg border border-gray-200">
          <div className="text-gray-500 mb-1">Risk Profile</div>
          <div className="text-2xl font-bold text-gray-900 capitalize">{riskProfile}</div>
        </div>
      </div>

      {/* Real-time Data Status */}
      <div className="text-xs text-gray-500 text-center">
        📊 Charts update every 2 minutes | Last update: {lastUpdate ? lastUpdate.toLocaleTimeString() : 'Never'}
      </div>
    </div>
  );
};

export default PerformanceCharts; 