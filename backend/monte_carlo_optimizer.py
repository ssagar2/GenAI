"""
Monte Carlo Portfolio Optimization
================================

A sophisticated Monte Carlo simulation for portfolio optimization using historical asset data.
Features:
- Clean data handling from Yahoo Finance
- Random portfolio weight generation
- Efficient frontier calculation
- Sharpe ratio optimization
- Performance optimization for large-scale simulations
- Modular design for API integration
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PortfolioResult:
    """Data class to store portfolio optimization results"""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    risk_free_rate: float = 0.02

@dataclass
class EfficientFrontier:
    """Data class to store efficient frontier results"""
    returns: np.ndarray
    volatilities: np.ndarray
    sharpe_ratios: np.ndarray
    weights_matrix: np.ndarray
    optimal_portfolio: PortfolioResult
    min_volatility_portfolio: PortfolioResult
    max_sharpe_portfolio: PortfolioResult

class MonteCarloOptimizer:
    """
    Advanced Monte Carlo simulation for portfolio optimization
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.returns_data = None
        self.covariance_matrix = None
        self.mean_returns = None
        self.symbols = None
        
    def fetch_historical_data(self, symbols: List[str], period: str = '2y') -> pd.DataFrame:
        """
        Fetch clean historical data from Yahoo Finance
        
        Args:
            symbols: List of stock symbols
            period: Data period (e.g., '2y', '1y', '6mo')
            
        Returns:
            DataFrame with daily returns for all symbols
        """
        try:
            logger.info(f"Fetching historical data for {len(symbols)} symbols...")
            
            # Fetch data for all symbols
            data_dict = {}
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    
                    if len(hist) > 60:  # At least 60 days of data
                        # Calculate daily returns
                        returns = hist['Close'].pct_change().dropna()
                        data_dict[symbol] = returns
                        logger.info(f"✓ {symbol}: {len(returns)} days of data")
                    else:
                        logger.warning(f"✗ {symbol}: Insufficient data ({len(hist)} days)")
                        
                except Exception as e:
                    logger.error(f"✗ {symbol}: Error fetching data - {str(e)}")
            
            # Create DataFrame with aligned dates
            if data_dict:
                returns_df = pd.DataFrame(data_dict)
                returns_df = returns_df.dropna()  # Remove any NaN values
                
                if len(returns_df) > 30:
                    logger.info(f"✓ Successfully fetched data: {len(returns_df)} days, {len(returns_df.columns)} symbols")
                    return returns_df
                else:
                    logger.error("✗ Insufficient data after alignment")
                    return pd.DataFrame()
            else:
                logger.error("✗ No valid data fetched")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_portfolio_metrics(self, weights: np.ndarray) -> PortfolioResult:
        """
        Calculate portfolio metrics for given weights
        
        Args:
            weights: Portfolio weights (must sum to 1)
            
        Returns:
            PortfolioResult with calculated metrics
        """
        try:
            # Ensure weights sum to 1
            weights = weights / np.sum(weights)
            
            # Calculate expected return
            expected_return = np.sum(weights * self.mean_returns)
            
            # Calculate portfolio volatility
            portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
            volatility = np.sqrt(portfolio_variance)
            
            # Calculate Sharpe ratio
            sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            return PortfolioResult(
                weights=weights,
                expected_return=expected_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                risk_free_rate=self.risk_free_rate
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return None
    
    def generate_random_weights(self, n_assets: int, n_simulations: int) -> np.ndarray:
        """
        Generate random portfolio weights using Dirichlet distribution
        for better distribution than uniform random
        
        Args:
            n_assets: Number of assets
            n_simulations: Number of simulations
            
        Returns:
            Array of shape (n_simulations, n_assets) with weights summing to 1
        """
        try:
            # Use Dirichlet distribution for better weight distribution
            # Alpha = 1 gives uniform distribution over the simplex
            alpha = np.ones(n_assets)
            weights = np.random.dirichlet(alpha, n_simulations)
            
            # Ensure all weights sum to 1 (should already be true with Dirichlet)
            weights = weights / np.sum(weights, axis=1, keepdims=True)
            
            return weights
            
        except Exception as e:
            logger.error(f"Error generating random weights: {str(e)}")
            # Fallback to uniform random
            weights = np.random.random((n_simulations, n_assets))
            weights = weights / np.sum(weights, axis=1, keepdims=True)
            return weights
    
    def run_monte_carlo_simulation(self, symbols: List[str], n_simulations: int = 10000,
                                  period: str = '2y', use_parallel: bool = True) -> EfficientFrontier:
        """
        Run Monte Carlo simulation for portfolio optimization
        
        Args:
            symbols: List of stock symbols
            n_simulations: Number of simulations (default: 10,000)
            period: Historical data period
            use_parallel: Whether to use parallel processing
            
        Returns:
            EfficientFrontier with all simulation results
        """
        try:
            logger.info(f"Starting Monte Carlo simulation with {n_simulations:,} portfolios...")
            
            # Fetch and prepare data
            self.returns_data = self.fetch_historical_data(symbols, period)
            if self.returns_data.empty:
                raise ValueError("No valid historical data available")
            
            self.symbols = symbols
            n_assets = len(symbols)
            
            # Calculate mean returns and covariance matrix
            self.mean_returns = self.returns_data.mean().values
            self.covariance_matrix = self.returns_data.cov().values
            
            logger.info(f"Data prepared: {n_assets} assets, {len(self.returns_data)} days")
            
            # Generate random weights
            weights_matrix = self.generate_random_weights(n_assets, n_simulations)
            
            # Calculate portfolio metrics
            if use_parallel and n_simulations > 1000:
                results = self._calculate_metrics_parallel(weights_matrix)
            else:
                results = self._calculate_metrics_sequential(weights_matrix)
            
            # Extract results
            returns = np.array([r.expected_return for r in results if r is not None])
            volatilities = np.array([r.volatility for r in results if r is not None])
            sharpe_ratios = np.array([r.sharpe_ratio for r in results if r is not None])
            valid_weights = np.array([r.weights for r in results if r is not None])
            
            # Find optimal portfolios
            max_sharpe_idx = np.argmax(sharpe_ratios)
            min_vol_idx = np.argmin(volatilities)
            
            max_sharpe_portfolio = results[max_sharpe_idx]
            min_volatility_portfolio = results[min_vol_idx]
            
            # Create efficient frontier result
            efficient_frontier = EfficientFrontier(
                returns=returns,
                volatilities=volatilities,
                sharpe_ratios=sharpe_ratios,
                weights_matrix=valid_weights,
                optimal_portfolio=max_sharpe_portfolio,
                min_volatility_portfolio=min_volatility_portfolio,
                max_sharpe_portfolio=max_sharpe_portfolio
            )
            
            logger.info(f"✓ Simulation completed: {len(returns)} valid portfolios")
            logger.info(f"✓ Max Sharpe Ratio: {max_sharpe_portfolio.sharpe_ratio:.4f}")
            logger.info(f"✓ Min Volatility: {min_volatility_portfolio.volatility:.4f}")
            
            return efficient_frontier
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            raise
    
    def _calculate_metrics_sequential(self, weights_matrix: np.ndarray) -> List[PortfolioResult]:
        """Calculate metrics sequentially"""
        results = []
        for i, weights in enumerate(weights_matrix):
            if i % 1000 == 0:
                logger.info(f"Processing portfolio {i:,}/{len(weights_matrix):,}")
            result = self.calculate_portfolio_metrics(weights)
            results.append(result)
        return results
    
    def _calculate_metrics_parallel(self, weights_matrix: np.ndarray) -> List[PortfolioResult]:
        """Calculate metrics using parallel processing"""
        results = []
        
        def calculate_single_portfolio(weights):
            return self.calculate_portfolio_metrics(weights)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            future_to_weights = {
                executor.submit(calculate_single_portfolio, weights): i 
                for i, weights in enumerate(weights_matrix)
            }
            
            # Collect results
            for i, future in enumerate(as_completed(future_to_weights)):
                if i % 1000 == 0:
                    logger.info(f"Processing portfolio {i:,}/{len(weights_matrix):,}")
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel calculation: {str(e)}")
                    results.append(None)
        
        return results
    
    def visualize_efficient_frontier(self, efficient_frontier: EfficientFrontier, 
                                   save_path: Optional[str] = None) -> None:
        """
        Visualize the efficient frontier and optimal portfolios
        
        Args:
            efficient_frontier: Results from Monte Carlo simulation
            save_path: Optional path to save the plot
        """
        try:
            plt.style.use('seaborn-v0_8')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Efficient Frontier
            scatter = ax1.scatter(efficient_frontier.volatilities * 100, 
                                 efficient_frontier.returns * 100,
                                 c=efficient_frontier.sharpe_ratios, 
                                 cmap='viridis', alpha=0.6, s=20)
            
            # Highlight optimal portfolios
            ax1.scatter(efficient_frontier.max_sharpe_portfolio.volatility * 100,
                       efficient_frontier.max_sharpe_portfolio.expected_return * 100,
                       color='red', s=200, marker='*', label='Max Sharpe Ratio')
            
            ax1.scatter(efficient_frontier.min_volatility_portfolio.volatility * 100,
                       efficient_frontier.min_volatility_portfolio.expected_return * 100,
                       color='green', s=200, marker='s', label='Min Volatility')
            
            ax1.set_xlabel('Portfolio Volatility (%)')
            ax1.set_ylabel('Expected Return (%)')
            ax1.set_title('Efficient Frontier')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Sharpe Ratio')
            
            # Plot 2: Weight Distribution for Optimal Portfolio
            optimal_weights = efficient_frontier.max_sharpe_portfolio.weights
            ax2.bar(range(len(self.symbols)), optimal_weights * 100, 
                   color='skyblue', alpha=0.7)
            ax2.set_xlabel('Assets')
            ax2.set_ylabel('Weight (%)')
            ax2.set_title('Optimal Portfolio Weights (Max Sharpe Ratio)')
            ax2.set_xticks(range(len(self.symbols)))
            ax2.set_xticklabels(self.symbols, rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add weight percentages on bars
            for i, weight in enumerate(optimal_weights):
                ax2.text(i, weight * 100 + 1, f'{weight*100:.1f}%', 
                        ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✓ Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error visualizing efficient frontier: {str(e)}")
    
    def get_optimal_portfolio_summary(self, efficient_frontier: EfficientFrontier) -> Dict:
        """
        Get a summary of the optimal portfolio
        
        Args:
            efficient_frontier: Results from Monte Carlo simulation
            
        Returns:
            Dictionary with optimal portfolio details
        """
        try:
            max_sharpe = efficient_frontier.max_sharpe_portfolio
            min_vol = efficient_frontier.min_volatility_portfolio
            
            # Create weight dictionary
            max_sharpe_weights = {
                symbol: weight * 100 
                for symbol, weight in zip(self.symbols, max_sharpe.weights)
            }
            
            min_vol_weights = {
                symbol: weight * 100 
                for symbol, weight in zip(self.symbols, min_vol.weights)
            }
            
            summary = {
                'max_sharpe_portfolio': {
                    'weights': max_sharpe_weights,
                    'expected_return': max_sharpe.expected_return * 100,
                    'volatility': max_sharpe.volatility * 100,
                    'sharpe_ratio': max_sharpe.sharpe_ratio
                },
                'min_volatility_portfolio': {
                    'weights': min_vol_weights,
                    'expected_return': min_vol.expected_return * 100,
                    'volatility': min_vol.volatility * 100,
                    'sharpe_ratio': min_vol.sharpe_ratio
                },
                'simulation_stats': {
                    'total_simulations': len(efficient_frontier.returns),
                    'max_return': np.max(efficient_frontier.returns) * 100,
                    'min_return': np.min(efficient_frontier.returns) * 100,
                    'max_volatility': np.max(efficient_frontier.volatilities) * 100,
                    'min_volatility': np.min(efficient_frontier.volatilities) * 100,
                    'max_sharpe': np.max(efficient_frontier.sharpe_ratios),
                    'min_sharpe': np.min(efficient_frontier.sharpe_ratios)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating portfolio summary: {str(e)}")
            return {}

def run_optimization_example():
    """
    Example usage of the Monte Carlo optimizer
    """
    # Initialize optimizer
    optimizer = MonteCarloOptimizer(risk_free_rate=0.02)
    
    # Define symbols (example with tech stocks)
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    try:
        # Run Monte Carlo simulation
        efficient_frontier = optimizer.run_monte_carlo_simulation(
            symbols=symbols,
            n_simulations=10000,
            period='2y',
            use_parallel=True
        )
        
        # Get summary
        summary = optimizer.get_optimal_portfolio_summary(efficient_frontier)
        
        # Print results
        print("\n" + "="*60)
        print("MONTE CARLO PORTFOLIO OPTIMIZATION RESULTS")
        print("="*60)
        
        print(f"\n📊 Simulation Statistics:")
        print(f"   Total Portfolios: {summary['simulation_stats']['total_simulations']:,}")
        print(f"   Return Range: {summary['simulation_stats']['min_return']:.2f}% - {summary['simulation_stats']['max_return']:.2f}%")
        print(f"   Volatility Range: {summary['simulation_stats']['min_volatility']:.2f}% - {summary['simulation_stats']['max_volatility']:.2f}%")
        print(f"   Sharpe Ratio Range: {summary['simulation_stats']['min_sharpe']:.3f} - {summary['simulation_stats']['max_sharpe']:.3f}")
        
        print(f"\n🏆 Maximum Sharpe Ratio Portfolio:")
        max_sharpe = summary['max_sharpe_portfolio']
        print(f"   Expected Return: {max_sharpe['expected_return']:.2f}%")
        print(f"   Volatility: {max_sharpe['volatility']:.2f}%")
        print(f"   Sharpe Ratio: {max_sharpe['sharpe_ratio']:.3f}")
        print(f"   Weights:")
        for symbol, weight in max_sharpe['weights'].items():
            print(f"     {symbol}: {weight:.1f}%")
        
        print(f"\n🛡️  Minimum Volatility Portfolio:")
        min_vol = summary['min_volatility_portfolio']
        print(f"   Expected Return: {min_vol['expected_return']:.2f}%")
        print(f"   Volatility: {min_vol['volatility']:.2f}%")
        print(f"   Sharpe Ratio: {min_vol['sharpe_ratio']:.3f}")
        print(f"   Weights:")
        for symbol, weight in min_vol['weights'].items():
            print(f"     {symbol}: {weight:.1f}%")
        
        # Visualize results
        optimizer.visualize_efficient_frontier(efficient_frontier)
        
        return efficient_frontier, summary
        
    except Exception as e:
        logger.error(f"Error in optimization example: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Run example
    run_optimization_example() 