#!/usr/bin/env python3
"""
Test Monte Carlo API import
"""

import sys
import os

# Add backend to path
sys.path.append('backend')

try:
    print("Testing Monte Carlo API import...")
    
    # Test basic import
    from monte_carlo_optimizer import MonteCarloOptimizer
    print("✓ MonteCarloOptimizer imported successfully")
    
    # Test API import
    from monte_carlo_api import MonteCarloAPI, create_monte_carlo_routes
    print("✓ MonteCarloAPI imported successfully")
    print("✓ create_monte_carlo_routes imported successfully")
    
    # Test API instance creation
    api = MonteCarloAPI()
    print("✓ MonteCarloAPI instance created successfully")
    
    # Test route creation function
    from flask import Flask
    app = Flask(__name__)
    create_monte_carlo_routes(app)
    print("✓ Monte Carlo routes created successfully")
    
    print("\n🎉 All Monte Carlo imports working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1) 