# Portfolio Optimizer

A modern, user-friendly portfolio optimization tool built with React. This application provides comprehensive features for stock selection, portfolio allocation, risk assessment, and performance visualization.

## Features

### 🎯 Stock Selection
- Search and select from a curated list of popular stocks
- Real-time stock price and performance data
- Easy add/remove functionality with visual feedback

### 📊 Allocation Management
- Interactive sliders for precise portfolio weight allocation
- Real-time validation ensuring 100% allocation
- Equal weight and reset functionality
- Visual portfolio preview

### 🛡️ Risk Assessment
- Three risk profiles: Conservative, Moderate, Aggressive
- Real-time portfolio risk analysis
- Diversification metrics
- Personalized risk recommendations

### 📈 Performance Charts
- Interactive line charts for performance tracking
- Pie charts for allocation visualization
- Bar charts for sector analysis
- Multiple chart types with smooth transitions

### 📋 Portfolio Summary
- Comprehensive portfolio metrics
- Top holdings analysis
- Sector allocation breakdown
- Smart recommendations based on portfolio health

## Technology Stack

- **React 18** - Modern React with hooks
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Smooth animations and transitions
- **Recharts** - Beautiful, responsive charts
- **Lucide React** - Modern icon library

## Getting Started

### Prerequisites

Make sure you have Node.js installed on your system. You can download it from [nodejs.org](https://nodejs.org/).

### Installation

1. Navigate to the project directory:
   ```bash
   cd Users/swapn/Documents/portfolio-optimizer
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```
4. Open your browser and visit `http://localhost:3000`

## Usage

### 1. Stock Selection
- Use the search bar to find stocks by symbol or company name
- Click on a stock to add it to your portfolio
- Remove stocks by clicking the X button next to each stock

### 2. Portfolio Allocation
- Adjust allocation percentages using the sliders
- Use the number input for precise control
- Ensure total allocation equals 100%
- Use "Equal Weight" to distribute evenly
- Use "Reset All" to clear allocations

### 3. Risk Assessment
- Select your risk profile: Conservative, Moderate, or Aggressive
- View real-time portfolio risk analysis
- Check diversification metrics
- Review personalized recommendations

### 4. Performance Analysis
- Switch between different chart types
- View performance trends over time
- Analyze sector allocations
- Monitor portfolio metrics

### 5. Portfolio Summary
- Review comprehensive portfolio metrics
- Check top holdings and their performance
- Analyze sector diversification
- Read smart recommendations

## Project Structure

```
src/
├── components/
│   ├── StockSelector.js      # Stock search and selection
│   ├── AllocationSliders.js  # Portfolio allocation controls
│   ├── RiskAssessment.js     # Risk analysis and profiles
│   ├── PerformanceCharts.js  # Interactive charts
│   └── PortfolioSummary.js   # Portfolio metrics and summary
├── App.js                    # Main application component
├── index.js                  # Application entry point
└── index.css                 # Global styles and Tailwind imports
```

## Customization

### Adding New Stocks
Edit the `availableStocks` array in `StockSelector.js` to add more stocks:

```javascript
const availableStocks = [
  { symbol: 'NEW', name: 'New Company Inc.', price: 100.00, change: 1.5, sector: 'Technology' },
  // Add more stocks here
];
```

### Modifying Risk Profiles
Update the `riskProfiles` array in `RiskAssessment.js` to customize risk profiles:

```javascript
const riskProfiles = [
  {
    id: 'custom',
    name: 'Custom Profile',
    description: 'Your custom risk profile',
    volatility: '5-15%',
    expectedReturn: '6-9%'
  }
];
```

### Styling
The application uses Tailwind CSS for styling. You can customize colors, spacing, and other design elements by modifying the Tailwind classes or extending the configuration in `tailwind.config.js`.

## Performance Features

- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Smooth Animations**: Framer Motion provides fluid transitions and interactions
- **Real-time Updates**: All components update instantly as you make changes
- **Interactive Charts**: Recharts provides beautiful, responsive data visualization

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

If you encounter any issues or have questions, please open an issue on the project repository.

---

**Happy Portfolio Optimizing! 🚀** 
