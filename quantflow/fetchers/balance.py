# quantflow/fetchers/balance.py
"""
QuantFlow Financial Suite - Balance Sheet Fetcher
Fetches and standardizes balance sheet data in Adobe DCF format
"""

import pandas as pd
import yfinance as yf
import requests
from typing import Dict, List, Optional
import logging
from datetime import datetime
import numpy as np

from ..config import get_config

logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

class BalanceFetcher:
    """
    Balance Sheet data fetcher with Adobe DCF standardization
    Focuses on key balance sheet items needed for DCF modeling
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.adobe_columns = self._get_adobe_column_mapping()
    
    def _get_adobe_column_mapping(self) -> Dict[str, str]:
        """
        Standard column mapping to match Adobe DCF balance sheet format
        """
        return {
            # Current Assets
            'Cash And Cash Equivalents': 'Cash & Cash Equivalents',
            'Cash Cash Equivalents And Short Term Investments': 'Cash & Cash Equivalents',
            'cashAndCashEquivalents': 'Cash & Cash Equivalents',
            'Cash': 'Cash & Cash Equivalents',
            
            'Short Term Investments': 'Short-term Investments',
            'Short-Term Investments': 'Short-term Investments',
            'shortTermInvestments': 'Short-term Investments',
            
            'Current Assets': 'Total Current Assets',
            'Total Current Assets': 'Total Current Assets',
            'totalCurrentAssets': 'Total Current Assets',
            
            'Accounts Receivable': 'Accounts Receivable',
            'Net Receivables': 'Accounts Receivable',
            'accountsReceivable': 'Accounts Receivable',
            
            'Inventory': 'Inventory',
            'inventory': 'Inventory',
            
            # Non-Current Assets
            'Property Plant Equipment Net': 'PP&E, Net',
            'Property Plant And Equipment Net': 'PP&E, Net',
            'Net PPE': 'PP&E, Net',
            'propertyPlantEquipmentNet': 'PP&E, Net',
            'PP&E': 'PP&E, Net',
            
            'Goodwill': 'Goodwill',
            'goodwill': 'Goodwill',
            
            'Intangible Assets': 'Intangible Assets',
            'Intangible Assets Net': 'Intangible Assets',
            'intangibleAssets': 'Intangible Assets',
            
            'Total Assets': 'Total Assets',
            'Total Non Current Assets': 'Total Non-Current Assets',
            'totalAssets': 'Total Assets',
            
            # Current Liabilities
            'Current Liabilities': 'Current Liabilities',
            'Total Current Liabilities': 'Current Liabilities',
            'totalCurrentLiabilities': 'Current Liabilities',
            
            'Accounts Payable': 'Accounts Payable',
            'accountsPayable': 'Accounts Payable',
            
            'Short Term Debt': 'Short-term Debt',
            'Short-Term Debt': 'Short-term Debt',
            'shortTermDebt': 'Short-term Debt',
            
            'Accrued Liabilities': 'Accrued Liabilities',
            'Accrued Expenses': 'Accrued Liabilities',
            
            # Non-Current Liabilities
            'Long Term Debt': 'Long-term Debt',
            'Long-Term Debt': 'Long-term Debt',
            'longTermDebt': 'Long-term Debt',
            
            'Total Debt': 'Total Debt',
            'totalDebt': 'Total Debt',
            
            'Deferred Revenue': 'Deferred Revenue',
            'Deferred Revenue Non Current': 'Deferred Revenue',
            'deferredRevenue': 'Deferred Revenue',
            
            # Equity
            'Stockholders Equity': 'Total Stockholders\' Equity',
            'Total Stockholders Equity': 'Total Stockholders\' Equity',
            'Total Shareholders Equity': 'Total Stockholders\' Equity',
            'totalStockholdersEquity': 'Total Stockholders\' Equity',
            'Shareholders Equity': 'Total Stockholders\' Equity',
            
            'Common Stock': 'Common Stock',
            'commonStock': 'Common Stock',
            
            'Retained Earnings': 'Retained Earnings',
            'retainedEarnings': 'Retained Earnings',
            
            'Additional Paid In Capital': 'Additional Paid-in Capital',
            'additionalPaidInCapital': 'Additional Paid-in Capital',
        }
    
    def fetch_balance_sheet(self, ticker: str, 
                           source: str = "yfinance",allow_fallback=True) -> pd.DataFrame:
        """
        Fetch balance sheet data for a given ticker
        
        Args:
            ticker: Stock ticker symbol
            source: Data source ('yfinance', 'alpha_vantage')
            
        Returns:
            DataFrame with standardized balance sheet in Adobe format
        """
        logger.info(f"Fetching balance sheet for {ticker} from {source}")
        
        try:
            if source == "yfinance":
                return self._fetch_yfinance_balance(ticker)
            elif source == "alpha_vantage":
                return self._fetch_alpha_vantage_balance(ticker)
            elif source == "iex":
                raise ValueError("IEX Cloud support has been removed or disabled")
            else:
                raise ValueError(f"Unsupported data source: {source}")
                
        except Exception as e:
            logger.error(f"Error fetching balance sheet for {ticker}: {e}")
            if allow_fallback:
                return self._fetch_with_fallback(ticker, exclude_sources=[source])
            else:
                raise

    
    def _fetch_yfinance_balance(self, ticker: str) -> pd.DataFrame:
        """Fetch balance sheet from Yahoo Finance"""
        
        yf_ticker = yf.Ticker(ticker)
        balance_sheet = yf_ticker.balance_sheet.T  # Transpose for years as rows

        if balance_sheet.empty:
            raise ValueError(f"No balance sheet data found for {ticker}")

        #  Drop duplicate years BEFORE further processing
        # Ensure DatetimeIndex
        balance_sheet.index = pd.to_datetime(balance_sheet.index, errors='coerce')

        # Drop duplicates by fiscal year (keep latest)
        year_series = balance_sheet.index.to_series().dt.year
        balance_sheet = balance_sheet.loc[~year_series.duplicated(keep='first')]

        balance_sheet = balance_sheet.sort_index(ascending=False)


        # Standardize and calculate metrics
        standardized_df = self._standardize_columns(balance_sheet)
        standardized_df = standardized_df.loc[:, ~standardized_df.columns.duplicated()]
        standardized_df = self._calculate_derived_metrics(standardized_df)

        logger.info(f"Successfully fetched {len(standardized_df)} years of balance sheet data for {ticker}")
        return standardized_df

    
    def _fetch_alpha_vantage_balance(self, ticker: str) -> pd.DataFrame:
        """Fetch balance sheet from Alpha Vantage API"""
        
        api_key = self.config.alpha_vantage_api_key
        if not api_key:
            raise ValueError("Alpha Vantage API key not found")
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'BALANCE_SHEET',
            'symbol': ticker,
            'apikey': api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        
        if 'annualReports' not in data:
            raise ValueError("No annual reports found")
        
        # Convert to DataFrame
        annual_reports = data['annualReports']
        df = pd.DataFrame(annual_reports)
        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
        df.set_index('fiscalDateEnding', inplace=True)
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return self._standardize_columns(df)
    
    def _fetch_with_fallback(self, ticker: str, exclude_sources=None):
        exclude_sources = exclude_sources or []
        sources = [s for s in ['yfinance', 'alpha_vantage'] if s not in exclude_sources]

        for source in sources:
            try:
                logger.info(f"Trying fallback source: {source}")
                return self.fetch_balance_sheet(ticker, source, allow_fallback=False)
            except Exception as e:
                logger.warning(f"Fallback source {source} failed: {e}")
                continue

        raise ValueError(f"All data sources failed for ticker {ticker}")


    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to Adobe format"""
        
        standardized_df = df.copy()
        
        # Rename columns
        rename_dict = {}
        for col in df.columns:
            if col in self.adobe_columns:
                rename_dict[col] = self.adobe_columns[col]
        
        if rename_dict:
            standardized_df = standardized_df.rename(columns=rename_dict)
        
        # Ensure key Adobe columns exist
        adobe_required_columns = [
            'Cash & Cash Equivalents', 'Short-term Investments', 
            'Total Current Assets', 'PP&E, Net', 'Goodwill',
            'Total Assets', 'Current Liabilities', 'Long-term Debt',
            'Total Stockholders\' Equity'
        ]
        
        for col in adobe_required_columns:
            if col not in standardized_df.columns:
                standardized_df[col] = np.nan
        
        return standardized_df
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived balance sheet metrics for DCF analysis"""
        
        result_df = df.copy()
        
        # Calculate Total Debt if missing
        if 'Total Debt' not in df.columns or df['Total Debt'].isna().all():
            short_term = df.get('Short-term Debt', 0)
            long_term = df.get('Long-term Debt', 0)
            result_df['Total Debt'] = short_term + long_term
        
        # Calculate Net Cash (Cash - Total Debt)
        if ('Cash & Cash Equivalents' in df.columns and 
            'Total Debt' in result_df.columns):
            cash = df['Cash & Cash Equivalents'].fillna(0).infer_objects(copy=False)
            debt = result_df['Total Debt'].fillna(0).infer_objects(copy=False)
            result_df['Net Cash (Debt)'] = cash - debt
        
        # Calculate Working Capital
        if ('Total Current Assets' in df.columns and 
            'Current Liabilities' in df.columns):
            current_assets = df['Total Current Assets'].fillna(0).infer_objects(copy=False)
            current_liab = df['Current Liabilities'].fillna(0).infer_objects(copy=False)
            result_df['Working Capital'] = current_assets - current_liab
        
        # Calculate Total Tangible Assets (Total Assets - Goodwill - Intangibles)
        if 'Total Assets' in df.columns:
            total_assets = df['Total Assets'].fillna(0).infer_objects(copy=False)
            goodwill = df.get('Goodwill', 0)
            intangibles = df.get('Intangible Assets', 0)
            result_df['Tangible Assets'] = total_assets - goodwill - intangibles
        
        # Calculate Book Value per Share (if we have shares data)
        # This would need to be enhanced with shares outstanding data
        
        return result_df
    
    def calculate_balance_sheet_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate balance sheet ratios useful for DCF analysis"""
        
        ratios_df = pd.DataFrame(index=df.index)
        
        # Asset Turnover ratios would need revenue data
        # Debt ratios
        if ('Total Debt' in df.columns and 
            'Total Stockholders\' Equity' in df.columns):
            debt = df['Total Debt'].fillna(0)
            equity = df['Total Stockholders\' Equity'].fillna(0)
            total_capital = debt + equity
            
            # Debt-to-Equity
            ratios_df['Debt-to-Equity'] = debt / equity.replace(0, np.nan)
            
            # Debt-to-Total Capital
            ratios_df['Debt-to-Total Capital'] = debt / total_capital.replace(0, np.nan)
        
        # Current Ratio
        if ('Total Current Assets' in df.columns and 
            'Current Liabilities' in df.columns):
            current_assets = df['Total Current Assets']
            current_liab = df['Current Liabilities']
            ratios_df['Current Ratio'] = current_assets / current_liab.replace(0, np.nan)
        
        # Quick Ratio (more conservative liquidity measure)
        if ('Cash & Cash Equivalents' in df.columns and 
            'Short-term Investments' in df.columns and
            'Accounts Receivable' in df.columns and
            'Current Liabilities' in df.columns):
            
            quick_assets = (df['Cash & Cash Equivalents'].fillna(0) + 
                          df['Short-term Investments'].fillna(0) + 
                          df['Accounts Receivable'].fillna(0))
            current_liab = df['Current Liabilities']
            ratios_df['Quick Ratio'] = quick_assets / current_liab.replace(0, np.nan)
        
        # Asset Quality ratios
        if 'Total Assets' in df.columns:
            total_assets = df['Total Assets']
            
            # Goodwill as % of Total Assets
            if 'Goodwill' in df.columns:
                ratios_df['Goodwill % of Assets'] = df['Goodwill'] / total_assets
            
            # Tangible Assets Ratio
            if 'Tangible Assets' in df.columns:
                ratios_df['Tangible Asset Ratio'] = df['Tangible Assets'] / total_assets
        
        return ratios_df
    
    def save_to_csv(self, df: pd.DataFrame, ticker: str, 
                   include_ratios: bool = True) -> str:
        """Save balance sheet to CSV in Adobe format"""
        
        filename = self.config.get_filename(ticker, 'balance_sheet')
        filepath = self.config.processed_data_path / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if include_ratios:
            # Calculate and append ratios
            ratios_df = self.calculate_balance_sheet_ratios(df)
            
            # Add separator row
            separator = pd.DataFrame(index=['--- RATIOS ---'], columns=df.columns)
            separator.fillna('', inplace=True)
            
            # Combine data
            combined_df = pd.concat([df, separator, ratios_df])
            combined_df.to_csv(filepath)
        else:
            df.to_csv(filepath)
        
        logger.info(f"Balance sheet saved to: {filepath}")
        return str(filepath)
    
    def get_comprehensive_balance_data(self, ticker: str) -> Dict:
        """Get comprehensive balance sheet data including ratios and metadata"""
        
        # Fetch base balance sheet
        balance_df = self.fetch_balance_sheet(ticker)
        
        balance_df = balance_df[~balance_df.index.duplicated(keep='first')]
        balance_df = balance_df.sort_index(ascending=False)

        # Calculate ratios
        ratios_df = self.calculate_balance_sheet_ratios(balance_df)
        
        # Get company-specific configuration
        company_config = self.config.get_company_config(ticker)
        
        # Save to CSV
        csv_path = self.save_to_csv(balance_df, ticker)
        
        # Extract key metrics for DCF
        latest_data = balance_df.iloc[0] if not balance_df.empty else pd.Series()
        
        dcf_metrics = {
            'cash_and_equivalents': latest_data.get('Cash & Cash Equivalents', 0),
            'total_debt': latest_data.get('Total Debt', 0),
            'net_cash': latest_data.get('Net Cash (Debt)', 0),
            'total_assets': latest_data.get('Total Assets', 0),
            'stockholders_equity': latest_data.get('Total Stockholders\' Equity', 0),
            'working_capital': latest_data.get('Working Capital', 0)
        }
        
        return {
            'balance_sheet': balance_df,
            'key_ratios': ratios_df,
            'dcf_metrics': dcf_metrics,
            'company_config': company_config,
            'csv_path': csv_path,
            'ticker': ticker.upper(),
            'last_updated': datetime.now().isoformat(),
            'data_years': len(balance_df),
            'latest_year': balance_df.index[0] if not balance_df.empty else None
        }

# Convenience function
def fetch_balance_data(ticker: str, source: str = "yfinance") -> Dict:
    """
    Quick function to fetch balance sheet data
    
    Usage:
        data = fetch_balance_data("ADBE")
        print(data['balance_sheet'])
    """
    fetcher = BalanceFetcher()
    return fetcher.get_comprehensive_balance_data(ticker)

if __name__ == "__main__":
    # Example usage
    fetcher = BalanceFetcher()
    
    # Test with Adobe
    adobe_data = fetcher.get_comprehensive_balance_data("ADBE")
    print(" Adobe Balance Sheet Data:")
    print(adobe_data['balance_sheet'].head())
    print(f"\n Key DCF Metrics:")
    for metric, value in adobe_data['dcf_metrics'].items():
        if value != 0:
            print(f"   {metric}: ${value:,.0f}")
    print(f"\n Saved to: {adobe_data['csv_path']}")