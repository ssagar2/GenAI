import requests
import json

def test_monte_carlo_endpoint():
    """Test the Monte Carlo optimization endpoint"""
    
    # Test data
    test_data = {
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "risk_profile": "moderate",
        "n_simulations": 1000
    }
    
    try:
        # Test the Monte Carlo endpoint
        print("Testing Monte Carlo optimization endpoint...")
        response = requests.post(
            "http://localhost:5000/api/portfolio/monte-carlo/optimize",
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Monte Carlo optimization successful!")
            print(f"Optimization Method: {result.get('optimization_method', 'N/A')}")
            print(f"Risk Profile: {result.get('risk_profile', 'N/A')}")
            
            if 'optimal_portfolio' in result:
                portfolio = result['optimal_portfolio']
                print(f"Expected Return: {portfolio.get('expected_return', 'N/A')}%")
                print(f"Volatility: {portfolio.get('volatility', 'N/A')}%")
                print(f"Sharpe Ratio: {portfolio.get('sharpe_ratio', 'N/A')}")
                
                weights = portfolio.get('weights', {})
                print("\nOptimal Weights:")
                for symbol, weight in weights.items():
                    print(f"  {symbol}: {weight:.2f}%")
        
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_monte_carlo_endpoint() 