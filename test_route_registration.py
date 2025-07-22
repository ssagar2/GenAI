#!/usr/bin/env python3
"""
Test Monte Carlo route registration
"""

import sys
import os

# Add backend to path
sys.path.append('backend')

try:
    from flask import Flask
    from monte_carlo_api import create_monte_carlo_routes
    
    print("Testing Monte Carlo route registration...")
    
    # Create Flask app
    app = Flask(__name__)
    
    # Register Monte Carlo routes
    create_monte_carlo_routes(app)
    
    # Print all registered routes
    print("\nRegistered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.rule} -> {rule.endpoint}")
    
    # Check if Monte Carlo routes are registered
    monte_carlo_routes = [
        '/api/portfolio/monte-carlo/optimize',
        '/api/portfolio/monte-carlo/efficient-frontier',
        '/api/portfolio/monte-carlo/stats'
    ]
    
    print("\nChecking Monte Carlo routes:")
    for route in monte_carlo_routes:
        if route in [rule.rule for rule in app.url_map.iter_rules()]:
            print(f"✓ {route} is registered")
        else:
            print(f"✗ {route} is NOT registered")
    
    print("\n🎉 Route registration test completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 