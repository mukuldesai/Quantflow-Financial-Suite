# quantflow/fetchers/market.py
"""
QuantFlow Financial Suite - Market Data Fetcher
Fetches current market data, stock prices, and market metrics for DCF valuation
"""

import pandas as pd
import yfinance as yf
import requests
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import numpy as np

from ..config import get_config

logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

class MarketDataFetcher:
    """
    Market data fetcher for current prices, market cap, and valuation metrics
    Essential for DCF model final valuation and comparison
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
    
    def fetch_current_market_data(self, ticker: str) -> Dict:
        """
        Fetch current market data for a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with current market metrics
        """
        logger.info(f"Fetching market data for {ticker}")
        
        try:
            yf_ticker = yf.Ticker(ticker)
            
            # Get basic info
            info = yf_ticker.info
            
            # Get recent price history
            hist = yf_ticker.history(period="1y")
            
            # Get shares outstanding data
            shares_data = self._get_shares_outstanding(yf_ticker)
            
            market_data = {
                # Current pricing
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'day_change': 0,  # Will calculate below
                'day_change_percent': 0,
                
                # Market cap and valuation
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'shares_outstanding': shares_data.get('shares_outstanding', 0),
                'float_shares': info.get('floatShares', 0),
                
                # Trading metrics
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'avg_volume_10d': info.get('averageVolume10days', 0),
                
                # Price ranges
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'week_52_high': info.get('fiftyTwoWeekHigh', 0),
                'week_52_low': info.get('fiftyTwoWeekLow', 0),
                
                # Valuation multiples
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'ev_to_revenue': info.get('enterpriseToRevenue', 0),
                'ev_to_ebitda': info.get('enterpriseToEbitda', 0),
                
                # Financial strength
                'beta': info.get('beta', 1.0),
                'dividend_yield': info.get('dividendYield', 0),
                'payout_ratio': info.get('payoutRatio', 0),
                
                # Company info
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market': info.get('market', ''),
                'currency': info.get('currency', 'USD'),
                
                # Data quality
                'last_updated': datetime.now().isoformat(),
                'data_source': 'yfinance'
            }
            
            # Calculate day change
            if market_data['current_price'] > 0 and market_data['previous_close'] > 0:
                day_change = market_data['current_price'] - market_data['previous_close']
                market_data['day_change'] = day_change
                market_data['day_change_percent'] = day_change / market_data['previous_close']
            
            # Add price performance metrics
            if not hist.empty:
                performance_metrics = self._calculate_price_performance(hist)
                market_data.update(performance_metrics)
            
            logger.info(f"Successfully fetched market data for {ticker}")
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {e}")
            return self._get_default_market_data(ticker)
    
    def _get_shares_outstanding(self, yf_ticker) -> Dict:
        """Get shares outstanding data"""
        try:
            # Try to get from info first
            info = yf_ticker.info
            shares_outstanding = info.get('sharesOutstanding', 0)
            
            if shares_outstanding == 0:
                shares_outstanding = info.get('floatShares', 0)
 
            return {
                'shares_outstanding': shares_outstanding,
                'shares_float': info.get('floatShares', shares_outstanding)
            }
            
        except Exception as e:
            logger.warning(f"Could not fetch shares data: {e}")
            return {'shares_outstanding': 0, 'shares_float': 0}
    
    def _calculate_price_performance(self, hist: pd.DataFrame) -> Dict:
        """Calculate price performance metrics"""
        
        if hist.empty:
            return {}
        
        current_price = hist['Close'].iloc[-1]
        
        performance = {}
        
        # Calculate returns for different periods
        periods = {
            '1_week': 7,
            '1_month': 30,
            '3_month': 90,
            '6_month': 180,
            '1_year': 252
        }
        
        for period_name, days in periods.items():
            if len(hist) > days:
                past_price = hist['Close'].iloc[-days-1]
                if past_price > 0:
                    return_pct = (current_price - past_price) / past_price
                    performance[f'return_{period_name}'] = return_pct
            else:
                performance[f'return_{period_name}'] = 0
        
        # Calculate volatility (annualized)
        if len(hist) > 20:
            daily_returns = hist['Close'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252)  # Annualized
            performance['volatility_1yr'] = volatility
        
        # Calculate price statistics
        performance.update({
            'price_avg_50d': hist['Close'].tail(50).mean() if len(hist) >= 50 else current_price,
            'price_avg_200d': hist['Close'].tail(200).mean() if len(hist) >= 200 else current_price,
            'volume_avg_50d': hist['Volume'].tail(50).mean() if len(hist) >= 50 else 0
        })
        
        return performance
    
    def _get_default_market_data(self, ticker: str) -> Dict:
        """Return default market data structure if fetch fails"""
        return {
            'ticker': ticker.upper(),
            'current_price': 0,
            'market_cap': 0,
            'shares_outstanding': 0,
            'beta': 1.0,
            'error': 'Failed to fetch market data',
            'last_updated': datetime.now().isoformat(),
            'data_source': 'default'
        }
    
    def get_dcf_market_inputs(self, ticker: str) -> Dict:
        """
        Get market data specifically needed for DCF valuation
        
        Returns:
            Dictionary with DCF-specific market inputs
        """
        market_data = self.fetch_current_market_data(ticker)
        
        return {
            'current_stock_price': market_data.get('current_price', 0),
            'shares_outstanding': market_data.get('shares_outstanding', 0),
            'market_cap': market_data.get('market_cap', 0),
            'enterprise_value': market_data.get('enterprise_value', 0),
            'beta': market_data.get('beta', 1.0),
            
            # For comparison
            'current_valuation_multiples': {
                'pe_ratio': market_data.get('pe_ratio', 0),
                'ev_to_revenue': market_data.get('ev_to_revenue', 0),
                'ev_to_ebitda': market_data.get('ev_to_ebitda', 0),
                'price_to_book': market_data.get('price_to_book', 0)
            },
            
            # For risk assessment
            'risk_metrics': {
                'beta': market_data.get('beta', 1.0),
                'volatility': market_data.get('volatility_1yr', 0),
                'dividend_yield': market_data.get('dividend_yield', 0)
            }
        }
    
    def fetch_peer_comparison_data(self, ticker: str, peer_tickers: List[str]) -> pd.DataFrame:
        """
        Fetch comparison data for peer companies
        
        Args:
            ticker: Primary ticker
            peer_tickers: List of peer company tickers
            
        Returns:
            DataFrame with comparison metrics
        """
        all_tickers = [ticker] + peer_tickers
        comparison_data = []
        
        for t in all_tickers:
            try:
                market_data = self.fetch_current_market_data(t)
                
                comparison_data.append({
                    'ticker': t.upper(),
                    'market_cap': market_data.get('market_cap', 0),
                    'current_price': market_data.get('current_price', 0),
                    'pe_ratio': market_data.get('pe_ratio', 0),
                    'ev_to_revenue': market_data.get('ev_to_revenue', 0),
                    'ev_to_ebitda': market_data.get('ev_to_ebitda', 0),
                    'beta': market_data.get('beta', 0),
                    'return_1_year': market_data.get('return_1_year', 0),
                    'sector': market_data.get('sector', ''),
                    'industry': market_data.get('industry', '')
                })
                
            except Exception as e:
                logger.warning(f"Could not fetch data for peer {t}: {e}")
        
        columns = ['ticker', 'market_cap', 'current_price', 'pe_ratio', 'ev_to_revenue', 
           'ev_to_ebitda', 'beta', 'return_1_year', 'sector', 'industry'] 
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.set_index('ticker', inplace=True)
        
        return comparison_df
    
    def save_to_csv(self, market_data: Dict, ticker: str) -> str:
        """Save market data to CSV"""
        
        filename = self.config.get_filename(ticker, 'market_data')
        filepath = self.config.processed_data_path / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame for saving
        market_df = pd.DataFrame([market_data])
        market_df.to_csv(filepath, index=False)
        
        logger.info(f"Market data saved to: {filepath}")
        return str(filepath)
    
    def get_comprehensive_market_data(self, ticker: str, 
                                    peer_tickers: Optional[List[str]] = None) -> Dict:
        """Get comprehensive market analysis including peer comparison"""
        
        # Get primary market data
        market_data = self.fetch_current_market_data(ticker)
        
        # Get DCF-specific inputs
        dcf_inputs = self.get_dcf_market_inputs(ticker)
        
        # Get peer comparison if peers provided
        peer_comparison = None
        if peer_tickers:
            peer_comparison = self.fetch_peer_comparison_data(ticker, peer_tickers)
        
        # Save to CSV
        csv_path = self.save_to_csv(market_data, ticker)
        
        return {
            'ticker': ticker.upper(),
            'market_data': market_data,
            'dcf_inputs': dcf_inputs,
            'peer_comparison': peer_comparison,
            'csv_path': csv_path,
            'last_updated': datetime.now().isoformat()
        }

# Convenience function
def fetch_market_data(ticker: str, peer_tickers: Optional[List[str]] = None) -> Dict:
    """
    Quick function to fetch market data
    
    Usage:
        data = fetch_market_data("ADBE", ["MSFT", "GOOGL", "AAPL"])
        print(f"Current Price: ${data['market_data']['current_price']}")
    """
    fetcher = MarketDataFetcher()
    return fetcher.get_comprehensive_market_data(ticker, peer_tickers)

if __name__ == "__main__":
    # Example usage
    fetcher = MarketDataFetcher()
    
    # Test with Adobe
    adobe_data = fetcher.get_comprehensive_market_data("ADBE", ["MSFT", "GOOGL"])
    
    print(" Adobe Market Data:")
    market_info = adobe_data['market_data']
    print(f"Current Price: ${market_info['current_price']:.2f}")
    print(f"Market Cap: ${market_info['market_cap']:,.0f}")
    print(f"P/E Ratio: {market_info['pe_ratio']:.1f}")
    print(f"Beta: {market_info['beta']:.2f}")
    
    if adobe_data['peer_comparison'] is not None:
        print(f"\n Peer Comparison:")
        print(adobe_data['peer_comparison'][['market_cap', 'pe_ratio', 'beta']].head())
    
    print(f"\n Saved to: {adobe_data['csv_path']}")