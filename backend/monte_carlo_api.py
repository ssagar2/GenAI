"""
Monte Carlo API Integration
==========================

Integration module to connect the Monte Carlo optimizer with the Flask API.
Provides clean API endpoints for portfolio optimization.
"""

from monte_carlo_optimizer import MonteCarloOptimizer, EfficientFrontier
from flask import jsonify, request
import logging
from typing import List, Dict, Any, Optional
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)

class MonteCarloAPI:
    """
    API wrapper for Monte Carlo portfolio optimization
    """
    
    def __init__(self):
        self.optimizer = MonteCarloOptimizer(risk_free_rate=0.02)
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def optimize_portfolio(self, symbols: List[str], n_simulations: int = 10000,
                         period: str = '2y', risk_profile: str = 'moderate') -> Dict[str, Any]:
        """
        Optimize portfolio using Monte Carlo simulation
        
        Args:
            symbols: List of stock symbols
            n_simulations: Number of Monte Carlo simulations
            period: Historical data period
            risk_profile: Risk profile (conservative, moderate, aggressive)
            
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info(f"Starting Monte Carlo optimization for {len(symbols)} symbols")
            
            # Run Monte Carlo simulation
            efficient_frontier = self.optimizer.run_monte_carlo_simulation(
                symbols=symbols,
                n_simulations=n_simulations,
                period=period,
                use_parallel=True
            )
            
            # Get summary
            summary = self.optimizer.get_optimal_portfolio_summary(efficient_frontier)
            
            # Apply risk profile adjustments
            adjusted_weights = self._apply_risk_profile_adjustments(
                summary, risk_profile, efficient_frontier
            )
            
            # Prepare response
            response = {
                'success': True,
                'optimization_method': 'Monte Carlo Simulation',
                'risk_profile': risk_profile,
                'simulation_stats': summary['simulation_stats'],
                'optimal_portfolio': {
                    'weights': adjusted_weights,
                    'expected_return': summary['max_sharpe_portfolio']['expected_return'],
                    'volatility': summary['max_sharpe_portfolio']['volatility'],
                    'sharpe_ratio': summary['max_sharpe_portfolio']['sharpe_ratio']
                },
                'alternative_portfolios': {
                    'min_volatility': summary['min_volatility_portfolio'],
                    'max_sharpe': summary['max_sharpe_portfolio']
                },
                'efficient_frontier_data': {
                    'returns': efficient_frontier.returns.tolist(),
                    'volatilities': efficient_frontier.volatilities.tolist(),
                    'sharpe_ratios': efficient_frontier.sharpe_ratios.tolist()
                }
            }
            
            logger.info(f"✓ Monte Carlo optimization completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo optimization: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f'Monte Carlo optimization failed: {str(e)}'
            }
    
    def _apply_risk_profile_adjustments(self, summary: Dict, risk_profile: str, 
                                       efficient_frontier: EfficientFrontier) -> Dict[str, float]:
        """
        Apply risk profile adjustments to portfolio weights
        
        Args:
            summary: Portfolio summary
            risk_profile: Risk profile
            efficient_frontier: Efficient frontier results
            
        Returns:
            Adjusted weights dictionary
        """
        try:
            base_weights = summary['max_sharpe_portfolio']['weights']
            
            if risk_profile == 'conservative':
                # Conservative: Reduce allocation to high-volatility stocks
                # Increase allocation to low-volatility stocks
                adjusted_weights = self._adjust_for_conservative(base_weights, efficient_frontier)
                
            elif risk_profile == 'aggressive':
                # Aggressive: Increase allocation to high-volatility stocks
                # Reduce allocation to low-volatility stocks
                adjusted_weights = self._adjust_for_aggressive(base_weights, efficient_frontier)
                
            else:  # moderate
                # Use original Monte Carlo weights
                adjusted_weights = base_weights
            
            # Ensure weights sum to 100%
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {k: (v / total_weight) * 100 for k, v in adjusted_weights.items()}
            
            return adjusted_weights
            
        except Exception as e:
            logger.error(f"Error applying risk profile adjustments: {str(e)}")
            return summary['max_sharpe_portfolio']['weights']
    
    def _adjust_for_conservative(self, weights: Dict[str, float], 
                                efficient_frontier: EfficientFrontier) -> Dict[str, float]:
        """Adjust weights for conservative risk profile - favor low volatility stocks"""
        # Get volatility rankings
        volatilities = {}
        if self.optimizer.symbols is None:
            return weights
            
        for i, symbol in enumerate(self.optimizer.symbols):
            # Calculate individual stock volatility
            if self.optimizer.returns_data is not None:
                stock_returns = self.optimizer.returns_data.get(symbol)
                if stock_returns is not None:
                    volatilities[symbol] = stock_returns.std() * 100
                else:
                    volatilities[symbol] = 0.0
            else:
                volatilities[symbol] = 0.0
        
        # Sort by volatility (ascending - lowest volatility first)
        sorted_stocks = sorted(volatilities.items(), key=lambda x: x[1])
        
        # Conservative adjustment: dramatically favor low volatility stocks
        adjusted_weights = {}
        total_adjustment = 0
        
        for i, (symbol, volatility) in enumerate(sorted_stocks):
            original_weight = weights.get(symbol, 0)
            
            # Create dramatic shifts based on volatility ranking
            if i == 0:  # Lowest volatility stock
                adjusted_weights[symbol] = original_weight * 3.0  # Triple the weight
            elif i == 1:  # Second lowest volatility
                adjusted_weights[symbol] = original_weight * 2.0  # Double the weight
            elif i == len(sorted_stocks) - 1:  # Highest volatility stock
                adjusted_weights[symbol] = original_weight * 0.1  # Reduce to 10%
            elif i == len(sorted_stocks) - 2:  # Second highest volatility
                adjusted_weights[symbol] = original_weight * 0.3  # Reduce to 30%
            else:  # Middle volatility stocks
                adjusted_weights[symbol] = original_weight * 0.8  # Slight reduction
            
            total_adjustment += adjusted_weights[symbol]
        
        # Normalize to ensure weights sum to 100%
        if total_adjustment > 0:
            adjusted_weights = {k: (v / total_adjustment) * 100 for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _adjust_for_aggressive(self, weights: Dict[str, float], 
                              efficient_frontier: EfficientFrontier) -> Dict[str, float]:
        """Adjust weights for aggressive risk profile - favor high volatility stocks"""
        # Get volatility rankings
        volatilities = {}
        if self.optimizer.symbols is None:
            return weights
            
        for i, symbol in enumerate(self.optimizer.symbols):
            # Calculate individual stock volatility
            if self.optimizer.returns_data is not None:
                stock_returns = self.optimizer.returns_data.get(symbol)
                if stock_returns is not None:
                    volatilities[symbol] = stock_returns.std() * 100
                else:
                    volatilities[symbol] = 0.0
            else:
                volatilities[symbol] = 0.0
        
        # Sort by volatility (descending - highest volatility first)
        sorted_stocks = sorted(volatilities.items(), key=lambda x: x[1], reverse=True)
        
        # Aggressive adjustment: dramatically favor high volatility stocks
        adjusted_weights = {}
        total_adjustment = 0
        
        for i, (symbol, volatility) in enumerate(sorted_stocks):
            original_weight = weights.get(symbol, 0)
            
            # Create dramatic shifts based on volatility ranking
            if i == 0:  # Highest volatility stock
                adjusted_weights[symbol] = original_weight * 3.0  # Triple the weight
            elif i == 1:  # Second highest volatility
                adjusted_weights[symbol] = original_weight * 2.0  # Double the weight
            elif i == len(sorted_stocks) - 1:  # Lowest volatility stock
                adjusted_weights[symbol] = original_weight * 0.1  # Reduce to 10%
            elif i == len(sorted_stocks) - 2:  # Second lowest volatility
                adjusted_weights[symbol] = original_weight * 0.3  # Reduce to 30%
            else:  # Middle volatility stocks
                adjusted_weights[symbol] = original_weight * 0.8  # Slight reduction
            
            total_adjustment += adjusted_weights[symbol]
        
        # Normalize to ensure weights sum to 100%
        if total_adjustment > 0:
            adjusted_weights = {k: (v / total_adjustment) * 100 for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def get_efficient_frontier_plot(self, symbols: List[str], 
                                   n_simulations: int = 5000) -> Optional[str]:
        """
        Generate and save efficient frontier plot
        
        Args:
            symbols: List of stock symbols
            n_simulations: Number of simulations for plot
            
        Returns:
            Path to saved plot file or None if failed
        """
        try:
            # Run simulation
            efficient_frontier = self.optimizer.run_monte_carlo_simulation(
                symbols=symbols,
                n_simulations=n_simulations,
                period='2y',
                use_parallel=True
            )
            
            # Generate plot
            plot_path = f"efficient_frontier_{'_'.join(symbols)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.optimizer.visualize_efficient_frontier(efficient_frontier, save_path=plot_path)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error generating efficient frontier plot: {str(e)}")
            return None

# Global instance for API use
monte_carlo_api = MonteCarloAPI()

def create_monte_carlo_routes(app):
    """
    Create Flask routes for Monte Carlo optimization
    
    Args:
        app: Flask app instance
    """
    
    @app.route('/api/portfolio/monte-carlo/optimize', methods=['POST'])
    def monte_carlo_optimize():
        """Monte Carlo portfolio optimization endpoint"""
        try:
            print("[DEBUG] Raw request data:", request.data)
            print("[DEBUG] Request content type:", request.content_type)
            try:
                data = request.get_json(force=True, silent=True)
            except Exception as e:
                print("[DEBUG] Exception in get_json:", str(e))
                data = None
            print("[DEBUG] Parsed JSON:", data)
            if not data:
                return jsonify({'error': 'No JSON body received'}), 400
            symbols = data.get('symbols', [])
            n_simulations = data.get('n_simulations', 10000)
            period = data.get('period', '2y')
            risk_profile = data.get('risk_profile', 'moderate')
            print(f"[DEBUG] symbols={symbols}, n_simulations={n_simulations}, period={period}, risk_profile={risk_profile}")
            if not symbols:
                return jsonify({'error': 'No symbols provided'}), 400
            # Run optimization
            result = monte_carlo_api.optimize_portfolio(
                symbols=symbols,
                n_simulations=n_simulations,
                period=period,
                risk_profile=risk_profile
            )
            print("[DEBUG] Optimization result:", result)
            if result['success']:
                return jsonify(result)
            else:
                return jsonify({'error': result['error']}), 500
        except Exception as e:
            logger.error(f"Error in Monte Carlo optimization endpoint: {str(e)}")
            return jsonify({'error': f'Internal server error: {str(e)}'}), 500
    
    @app.route('/api/portfolio/monte-carlo/efficient-frontier', methods=['POST'])
    def generate_efficient_frontier_plot():
        """Generate efficient frontier plot endpoint"""
        try:
            data = request.get_json()
            symbols = data.get('symbols', [])
            n_simulations = data.get('n_simulations', 5000)
            
            if not symbols:
                return jsonify({'error': 'No symbols provided'}), 400
            
            # Generate plot
            plot_path = monte_carlo_api.get_efficient_frontier_plot(
                symbols=symbols,
                n_simulations=n_simulations
            )
            
            if plot_path:
                return jsonify({
                    'success': True,
                    'plot_path': plot_path,
                    'message': 'Efficient frontier plot generated successfully'
                })
            else:
                return jsonify({'error': 'Failed to generate plot'}), 500
                
        except Exception as e:
            logger.error(f"Error generating efficient frontier plot: {str(e)}")
            return jsonify({'error': f'Internal server error: {str(e)}'}), 500
    
    @app.route('/api/portfolio/monte-carlo/stats', methods=['POST'])
    def get_monte_carlo_stats():
        """Get Monte Carlo simulation statistics"""
        try:
            data = request.get_json()
            symbols = data.get('symbols', [])
            n_simulations = data.get('n_simulations', 1000)
            
            if not symbols:
                return jsonify({'error': 'No symbols provided'}), 400
            
            # Run quick simulation for stats
            result = monte_carlo_api.optimize_portfolio(
                symbols=symbols,
                n_simulations=n_simulations,
                period='1y',
                risk_profile='moderate'
            )
            
            if result['success']:
                return jsonify({
                    'success': True,
                    'simulation_stats': result['simulation_stats'],
                    'optimal_portfolio': result['optimal_portfolio']
                })
            else:
                return jsonify({'error': result['error']}), 500
                
        except Exception as e:
            logger.error(f"Error getting Monte Carlo stats: {str(e)}")
            return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# Example usage function
def test_monte_carlo_api():
    """Test the Monte Carlo API functionality"""
    try:
        # Test symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        print("Testing Monte Carlo API...")
        
        # Test optimization
        result = monte_carlo_api.optimize_portfolio(
            symbols=symbols,
            n_simulations=5000,
            risk_profile='moderate'
        )
        
        if result['success']:
            print("✓ Monte Carlo optimization successful")
            print(f"   Optimal Sharpe Ratio: {result['optimal_portfolio']['sharpe_ratio']:.3f}")
            print(f"   Expected Return: {result['optimal_portfolio']['expected_return']:.2f}%")
            print(f"   Volatility: {result['optimal_portfolio']['volatility']:.2f}%")
            
            print("\n   Optimal Weights:")
            for symbol, weight in result['optimal_portfolio']['weights'].items():
                print(f"     {symbol}: {weight:.1f}%")
        else:
            print(f"✗ Monte Carlo optimization failed: {result['error']}")
        
        return result
        
    except Exception as e:
        print(f"Error testing Monte Carlo API: {str(e)}")
        return None

if __name__ == "__main__":
    # Test the API
    test_monte_carlo_api() 