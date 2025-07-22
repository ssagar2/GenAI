import React, { useState, useEffect } from 'react';
import { Shield, AlertTriangle, CheckCircle, Info, Zap } from 'lucide-react';
import apiService from '../services/apiService';

const RiskAssessment = ({ riskProfile, onRiskChange, selectedStocks, allocations }) => {
  const riskProfiles = [
    {
      id: 'conservative',
      name: 'Conservative',
      description: 'Low risk, stable returns',
      color: 'bg-green-100 text-green-800 border-green-200',
      icon: CheckCircle,
      volatility: '5-10%',
      expectedReturn: '4-6%'
    },
    {
      id: 'moderate',
      name: 'Moderate',
      description: 'Balanced risk and return',
      color: 'bg-blue-100 text-blue-800 border-blue-200',
      icon: Shield,
      volatility: '10-15%',
      expectedReturn: '6-8%'
    },
    {
      id: 'aggressive',
      name: 'Aggressive',
      description: 'Higher risk, higher potential returns',
      color: 'bg-orange-100 text-orange-800 border-orange-200',
      icon: AlertTriangle,
      volatility: '15-25%',
      expectedReturn: '8-12%'
    }
  ];

  const calculatePortfolioRisk = () => {
    if (selectedStocks.length === 0) return { risk: 'Low', score: 0, diversification: 'Poor' };
    
    // Simple risk calculation based on number of stocks and sectors
    const sectors = [...new Set(selectedStocks.map(stock => stock.sector))];
    const diversificationScore = sectors.length / selectedStocks.length;
    
    let riskScore = 0;
    selectedStocks.forEach(stock => {
      const allocation = allocations[stock.symbol] || 0;
      // Higher allocation to single stock = higher risk
      if (allocation > 30) riskScore += 3;
      else if (allocation > 20) riskScore += 2;
      else if (allocation > 10) riskScore += 1;
    });
    
    // Add diversification factor
    riskScore += (1 - diversificationScore) * 2;
    
    if (riskScore <= 2) return { risk: 'Low', score: riskScore, diversification: 'Good' };
    else if (riskScore <= 4) return { risk: 'Medium', score: riskScore, diversification: 'Fair' };
    else return { risk: 'High', score: riskScore, diversification: 'Poor' };
  };

  const portfolioRisk = calculatePortfolioRisk();

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'Low': return 'text-green-600 bg-green-100';
      case 'Medium': return 'text-yellow-600 bg-yellow-100';
      case 'High': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getDiversificationColor = (diversification) => {
    switch (diversification) {
      case 'Good': return 'text-green-600 bg-green-100';
      case 'Fair': return 'text-yellow-600 bg-yellow-100';
      case 'Poor': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="space-y-4">
      {/* Risk Profile Selection */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium text-gray-700">Risk Profile</h3>
        <div className="space-y-2">
          {riskProfiles.map((profile) => {
            const Icon = profile.icon;
            return (
              <label
                key={profile.id}
                className={`flex items-center p-3 rounded-lg border-2 cursor-pointer transition-colors ${
                  riskProfile === profile.id
                    ? profile.color
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <input
                  type="radio"
                  name="riskProfile"
                  value={profile.id}
                  checked={riskProfile === profile.id}
                  onChange={(e) => onRiskChange(e.target.value)}
                  className="sr-only"
                />
                <Icon className="h-5 w-5 mr-3" />
                <div className="flex-1">
                  <div className="font-medium">{profile.name}</div>
                  <div className="text-xs opacity-75">{profile.description}</div>
                </div>
                <div className="text-xs text-right">
                  <div>Volatility: {profile.volatility}</div>
                  <div>Expected Return: {profile.expectedReturn}</div>
                </div>
              </label>
            );
          })}
        </div>
      </div>

      {/* Portfolio Risk Analysis */}
      {selectedStocks.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-gray-700">Portfolio Risk Analysis</h3>
          
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-2 mb-1">
                <Shield className="h-4 w-4 text-gray-600" />
                <span className="text-xs font-medium text-gray-700">Risk Level</span>
              </div>
              <div className={`inline-block px-2 py-1 rounded text-xs font-medium ${getRiskColor(portfolioRisk.risk)}`}>
                {portfolioRisk.risk}
              </div>
            </div>
            
            <div className="p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-2 mb-1">
                <Info className="h-4 w-4 text-gray-600" />
                <span className="text-xs font-medium text-gray-700">Diversification</span>
              </div>
              <div className={`inline-block px-2 py-1 rounded text-xs font-medium ${getDiversificationColor(portfolioRisk.diversification)}`}>
                {portfolioRisk.diversification}
              </div>
            </div>
          </div>

          {/* Risk Metrics */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-gray-600">Number of Stocks</span>
              <span className="font-medium">{selectedStocks.length}</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-gray-600">Sectors</span>
              <span className="font-medium">{[...new Set(selectedStocks.map(stock => stock.sector))].length}</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-gray-600">Largest Position</span>
              <span className="font-medium">
                {Math.max(...Object.values(allocations).map(v => v || 0)).toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Risk Recommendations */}
          <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
            <div className="flex items-start space-x-2">
              <Info className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
              <div className="text-xs text-blue-800">
                <div className="font-medium mb-1">Risk Recommendations:</div>
                {portfolioRisk.risk === 'High' && (
                  <ul className="space-y-1">
                    <li>• Consider reducing position sizes</li>
                    <li>• Add more stocks for diversification</li>
                    <li>• Include defensive sectors</li>
                  </ul>
                )}
                {portfolioRisk.risk === 'Medium' && (
                  <ul className="space-y-1">
                    <li>• Portfolio is moderately balanced</li>
                    <li>• Consider adding 1-2 more stocks</li>
                    <li>• Monitor sector concentration</li>
                  </ul>
                )}
                {portfolioRisk.risk === 'Low' && (
                  <ul className="space-y-1">
                    <li>• Well-diversified portfolio</li>
                    <li>• Good risk management</li>
                    <li>• Consider growth opportunities</li>
                  </ul>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {selectedStocks.length === 0 && (
        <div className="text-center py-6 text-gray-500">
          <Shield className="h-8 w-8 mx-auto mb-2 text-gray-300" />
          <p className="text-sm">No stocks selected</p>
          <p className="text-xs">Select stocks to analyze portfolio risk</p>
        </div>
      )}
    </div>
  );
};

export default RiskAssessment; 