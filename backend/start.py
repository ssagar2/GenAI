#!/usr/bin/env python3
"""
Startup script for the Portfolio Optimizer Backend
"""

import os
import sys
from app import app

if __name__ == '__main__':
    print("🚀 Starting Portfolio Optimizer Backend...")
    print("📍 Backend will be available at: http://localhost:5000")
    print("🔗 API endpoints:")
    print("   - GET  /api/health")
    print("   - GET  /api/search?q=<query>")
    print("   - GET  /api/stock/<symbol>")
    print("   - POST /api/portfolio/metrics")
    print("   - POST /api/portfolio/optimize")
    print("   - POST /api/portfolio/performance")
    print("\n📊 Backend is ready to handle heavy computations!")
    
    # Disable all reloader and debug features
    os.environ['FLASK_ENV'] = 'production'
    os.environ['FLASK_DEBUG'] = '0'
    
    app.run(
        debug=False,
        host='0.0.0.0',
        port=5000,
        use_reloader=False,
        threaded=True,
        processes=1
    ) 