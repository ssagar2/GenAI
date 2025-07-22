// Test Monte Carlo integration
const testMonteCarloIntegration = async () => {
  console.log('🧪 Testing Monte Carlo Integration...');
  
  try {
    // Test the Monte Carlo endpoint
    const response = await fetch('http://localhost:5000/api/portfolio/monte-carlo/optimize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        symbols: ['AAPL', 'MSFT', 'GOOGL'],
        risk_profile: 'moderate',
        n_simulations: 100
      })
    });
    
    if (response.ok) {
      const result = await response.json();
      console.log('✅ Monte Carlo endpoint working:', result);
      return true;
    } else {
      console.log('❌ Monte Carlo endpoint failed:', response.status, response.statusText);
      return false;
    }
  } catch (error) {
    console.log('❌ Monte Carlo test failed:', error);
    return false;
  }
};

// Run the test
testMonteCarloIntegration(); 