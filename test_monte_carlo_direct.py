#!/usr/bin/env python3
"""
Direct test of Monte Carlo optimizer
"""

import sys
import os
sys.path.append('backend')

try:
    from monte_carlo_optimizer import MonteCarloOptimizer
    from monte_carlo_api import MonteCarloAPI
    print("✓ Monte Carlo modules imported successfully")
    
    # Test the Monte Carlo optimizer directly
    print("\n🔬 Testing Monte Carlo Portfolio Optimization...")
    print("=" * 50)
    
    # Initialize the optimizer
    optimizer = MonteCarloOptimizer(risk_free_rate=0.02)
    api = MonteCarloAPI()
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    print(f"📊 Testing with symbols: {', '.join(symbols)}")
    
    # Run Monte Carlo simulation
    print("\n🔄 Running Monte Carlo simulation (1000 iterations)...")
    efficient_frontier = optimizer.run_monte_carlo_simulation(
        symbols=symbols,
        n_simulations=1000,
        period='1y',
        use_parallel=True
    )
    
    # Get summary
    summary = optimizer.get_optimal_portfolio_summary(efficient_frontier)
    
    print("\n📈 Monte Carlo Optimization Results:")
    print("=" * 50)
    
    # Simulation stats
    stats = summary.get('simulation_stats', {})
    print(f"📊 Simulation Statistics:")
    print(f"   • Total simulations: {stats.get('total_simulations', 'N/A')}")
    print(f"   • Execution time: {stats.get('execution_time', 'N/A')} seconds")
    
    # Optimal portfolio (Max Sharpe)
    max_sharpe = summary.get('max_sharpe_portfolio', {})
    print(f"\n🏆 Optimal Portfolio (Max Sharpe Ratio):")
    print(f"   • Expected Return: {max_sharpe.get('expected_return', 'N/A'):.2f}%")
    print(f"   • Volatility: {max_sharpe.get('volatility', 'N/A'):.2f}%")
    print(f"   • Sharpe Ratio: {max_sharpe.get('sharpe_ratio', 'N/A'):.3f}")
    print(f"   • Portfolio Weights:")
    weights = max_sharpe.get('weights', {})
    for symbol, weight in weights.items():
        print(f"     - {symbol}: {weight:.1f}%")
    
    # Min volatility portfolio
    min_vol = summary.get('min_volatility_portfolio', {})
    print(f"\n🛡️  Minimum Volatility Portfolio:")
    print(f"   • Expected Return: {min_vol.get('expected_return', 'N/A'):.2f}%")
    print(f"   • Volatility: {min_vol.get('volatility', 'N/A'):.2f}%")
    print(f"   • Sharpe Ratio: {min_vol.get('sharpe_ratio', 'N/A'):.3f}")
    print(f"   • Portfolio Weights:")
    min_vol_weights = min_vol.get('weights', {})
    for symbol, weight in min_vol_weights.items():
        print(f"     - {symbol}: {weight:.1f}%")
    
    # Test different risk profiles
    print(f"\n🎯 Risk Profile Adjustments:")
    print("=" * 50)
    
    for risk_profile in ['conservative', 'moderate', 'aggressive']:
        print(f"\n📊 {risk_profile.title()} Risk Profile:")
        
        # Apply risk profile adjustments
        adjusted_weights = api._apply_risk_profile_adjustments(
            summary, risk_profile, efficient_frontier
        )
        
        print(f"   • Adjusted Weights:")
        for symbol, weight in adjusted_weights.items():
            print(f"     - {symbol}: {weight:.1f}%")
    
    print(f"\n✅ Monte Carlo optimization test completed successfully!")
    print(f"💡 The optimizer successfully processed {len(efficient_frontier.returns)} portfolios")
    print(f"⚡ Best Sharpe ratio: {max_sharpe.get('sharpe_ratio', 'N/A'):.3f}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
except Exception as e:
    print(f"❌ Error during Monte Carlo test: {e}")
    import traceback
    traceback.print_exc() 