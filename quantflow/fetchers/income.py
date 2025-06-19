# quantflow/fetchers/income.py
"""
QuantFlow Financial Suite - Income Statement Fetcher
Fetches and standardizes income statement data in Adobe DCF format
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

class IncomeFetcher:
    """
    Income Statement data fetcher with Adobe DCF standardization
    Supports multiple data sources with fallback mechanisms
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.adobe_columns = self._get_adobe_column_mapping()
    
    def _get_adobe_column_mapping(self) -> Dict[str, str]:
        """
        Standard column mapping to match Adobe DCF format
        Maps various data provider columns to our standardized format
        """
        return {
            # Revenue components
            'Total Revenue': 'Total Revenue',
            'TotalRevenue': 'Total Revenue',
            'totalRevenue': 'Total Revenue',
            'Revenue': 'Total Revenue',
            'Net Sales': 'Total Revenue',
            'Subscription Revenue': 'Subscription Revenue',
            'Product Revenue': 'Product Revenue',
            'Services Revenue': 'Services & Other Revenue',
            'Other Revenue': 'Services & Other Revenue',
            
            # Cost components
            'Cost Of Revenue': 'Cost of Revenue',
            'Cost of Goods Sold': 'Cost of Revenue',
            'Cost of Sales': 'Cost of Revenue',
            'Total Cost Of Revenue': 'Cost of Revenue',
            'costOfRevenue': 'Cost of Revenue',
            
            # Gross profit
            'Gross Profit': 'Gross Profit',
            'grossProfit': 'Gross Profit',
            
            # Operating expenses
            'Research And Development': 'Research & Development',
            'Research Development': 'Research & Development',
            'researchAndDevelopment': 'Research & Development',
            'R&D': 'Research & Development',
            
            'Selling General Administrative': 'Sales & Marketing',
            'Sales And Marketing': 'Sales & Marketing',
            'Sales Marketing': 'Sales & Marketing',
            'sellingGeneralAdministrative': 'Sales & Marketing',
            'SG&A': 'Sales & Marketing',
            
            'General Administrative': 'General & Administrative',
            'General And Administrative': 'General & Administrative',
            'generalAdministrative': 'General & Administrative',
            'G&A': 'General & Administrative',
            
            'Amortization Of Intangibles': 'Amortization of Intangibles',
            'Amortization': 'Amortization of Intangibles',
            'amortizationOfIntangibleAssets': 'Amortization of Intangibles',
            
            # Operating income
            'Operating Income': 'Operating Income (EBIT)',
            'Operating Revenue': 'Operating Income (EBIT)',
            'EBIT': 'Operating Income (EBIT)',
            'operatingIncome': 'Operating Income (EBIT)',
            
            # Non-operating items
            'Interest Expense': 'Interest Expense',
            'Interest Income': 'Interest Income',
            'Other Income Expense': 'Other Income (Expense), Net',
            'Other Income': 'Other Income (Expense), Net',
            'Non Operating Income Net Other': 'Other Income (Expense), Net',
            
            # Pre-tax and taxes
            'Pretax Income': 'Pre-Tax Income',
            'Income Before Tax': 'Pre-Tax Income',
            'incomeBeforeTax': 'Pre-Tax Income',
            'Earnings Before Tax': 'Pre-Tax Income',
            
            'Tax Provision': 'Tax Expense',
            'Income Tax Expense': 'Tax Expense',
            'incomeTaxExpense': 'Tax Expense',
            'Provision For Income Taxes': 'Tax Expense',
            
            # Net income
            'Net Income': 'Net Income',
            'Net Income Common Stockholders': 'Net Income',
            'netIncome': 'Net Income',
            
            # Shares
            'Weighted Average Shares Outstanding': 'Diluted Shares Outstanding',
            'Weighted Average Shs Out Dil': 'Diluted Shares Outstanding',
            'weightedAverageShsOutDil': 'Diluted Shares Outstanding',
            'Diluted Average Shares': 'Diluted Shares Outstanding',
        }
    
    def fetch_income_statement(self, ticker: str, 
                             source: str = "yfinance",allow_fallback=True) -> pd.DataFrame:
        """
        Fetch income statement data for a given ticker
        
        Args:
            ticker: Stock ticker symbol (e.g., 'ADBE', 'AAPL')
            source: Data source ('yfinance', 'alpha_vantage')
            
        Returns:
            DataFrame with standardized income statement in Adobe format
        """
        logger.info(f"Fetching income statement for {ticker} from {source}")
        
        try:
            if source == "yfinance":
                return self._fetch_yfinance_income(ticker)
            elif source == "alpha_vantage":
                return self._fetch_alpha_vantage_income(ticker)
            elif source == "iex":
                raise ValueError("IEX Cloud support has been removed or disabled")
            else:
                raise ValueError(f"Unsupported data source: {source}")
                
        except Exception as e:
            logger.error(f"Error fetching income statement for {ticker}: {e}")
            if allow_fallback:
                return self._fetch_with_fallback(ticker, exclude_sources=[source])
            else:
                raise

    def _fetch_yfinance_income(self, ticker: str) -> pd.DataFrame:
        """Fetch income statement from Yahoo Finance using yfinance"""

        yf_ticker = yf.Ticker(ticker)
        income_stmt = yf_ticker.financials.T  # Transpose to have years as rows

        if income_stmt.empty:
            raise ValueError(f"No income statement data found for {ticker}")

        # Drop duplicate years BEFORE further processing
        income_stmt = income_stmt[~income_stmt.index.duplicated(keep='first')]
        income_stmt = income_stmt.sort_index(ascending=False)

        # Standardize and calculate metrics
        standardized_df = self._standardize_columns(income_stmt)
        standardized_df = self._calculate_derived_metrics(standardized_df)
        standardized_df = self._calculate_eps(standardized_df, yf_ticker)

        logger.info(f"Successfully fetched {len(standardized_df)} years of income data for {ticker}")
        return standardized_df


    
    def _fetch_alpha_vantage_income(self, ticker: str) -> pd.DataFrame:
        """Fetch income statement from Alpha Vantage API"""
        
        api_key = self.config.alpha_vantage_api_key
        if not api_key:
            raise ValueError("Alpha Vantage API key not found in configuration")
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'INCOME_STATEMENT',
            'symbol': ticker,
            'apikey': api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        
        if 'annualReports' not in data:
            raise ValueError("No annual reports found in Alpha Vantage response")
        
        if not isinstance(data, dict) or 'annualReports' not in data:
            raise ValueError("No annual reports found in Alpha Vantage response")
        
        # Convert to DataFrame
        annual_reports = data['annualReports']
        df = pd.DataFrame(annual_reports)
        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
        df.set_index('fiscalDateEnding', inplace=True)
        
        # Convert to numeric
        for col in df.columns:
            if col != 'fiscalDateEnding':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Standardize columns
        standardized_df = self._standardize_columns(df)
        
        return standardized_df
    
    
    def _fetch_with_fallback(self, ticker: str, exclude_sources=None):
        exclude_sources = exclude_sources or []
        sources = [s for s in ['yfinance', 'alpha_vantage'] if s not in exclude_sources]

        for source in sources:
            try:
                logger.info(f"Trying fallback source: {source}")
                return self.fetch_income_statement(ticker, source=source, allow_fallback=False)
            except Exception as e:
                logger.warning(f"Fallback source {source} failed: {e}")
                continue

        raise ValueError(f"All data sources failed for ticker {ticker}")

    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to Adobe DCF format"""
        
        # Create a copy to avoid modifying original
        standardized_df = df.copy()
        
        # Rename columns based on mapping
        rename_dict = {}
        for col in df.columns:
            if col in self.adobe_columns:
                rename_dict[col] = self.adobe_columns[col]
        
        if rename_dict:
            standardized_df = standardized_df.rename(columns=rename_dict)
        
        # Ensure we have the key Adobe columns (create empty if missing)
        adobe_required_columns = [
            'Total Revenue', 'Cost of Revenue', 'Gross Profit',
            'Research & Development', 'Sales & Marketing', 
            'General & Administrative', 'Operating Income (EBIT)',
            'Pre-Tax Income', 'Tax Expense', 'Net Income'
        ]
        
        for col in adobe_required_columns:
            if col not in standardized_df.columns:
                standardized_df[col] = np.nan
        
        return standardized_df
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics like those in Adobe DCF model"""
        
        result_df = df.copy()
        
        # Calculate Gross Profit if missing
        if 'Gross Profit' not in result_df.columns or result_df['Gross Profit'].isna().all():
            if 'Total Revenue' in result_df.columns and 'Cost of Revenue' in result_df.columns:
                result_df['Gross Profit'] = result_df['Total Revenue'] - result_df['Cost of Revenue']

        
        # Calculate Total Operating Expenses
        opex_columns = [
            'Research & Development', 'Sales & Marketing', 
            'General & Administrative', 'Amortization of Intangibles'
        ]
        
        available_opex = [col for col in opex_columns if col in df.columns]
        if available_opex:
            result_df['Total Operating Expenses'] = df[available_opex].sum(axis=1, skipna=True)
        
        # Calculate Operating Income (EBIT) if missing
        if 'Operating Income (EBIT)' not in result_df.columns or result_df['Operating Income (EBIT)'].isna().all():
            if 'Gross Profit' in result_df.columns and 'Total Operating Expenses' in result_df.columns:
                result_df['Operating Income (EBIT)'] = (
                    result_df['Gross Profit'] - result_df['Total Operating Expenses']
                )
        
        # Calculate Pre-Tax Income if missing
        if ('Pre-Tax Income' not in result_df.columns or 
            result_df['Pre-Tax Income'].isna().all()):
            ebit = result_df.get('Operating Income (EBIT)', pd.Series([0] * len(result_df), index=result_df.index))
            interest_exp = result_df.get('Interest Expense', pd.Series([0] * len(result_df), index=result_df.index))
            other_income = result_df.get('Other Income (Expense), Net', pd.Series([0] * len(result_df), index=result_df.index))
            
            result_df['Pre-Tax Income'] = ebit - interest_exp + other_income
        
        return result_df
    
    def _calculate_eps(self, df: pd.DataFrame, yf_ticker) -> pd.DataFrame:
        """Calculate Diluted EPS like Adobe model"""
        
        result_df = df.copy()
        
        try:
            # Try to get shares outstanding from yfinance
            shares_data = getattr(yf_ticker, "get_shares_full", lambda: pd.DataFrame())()
            
            if shares_data is not None and not shares_data.empty:
                # Get annual shares data
                annual_shares = shares_data.resample('Y').last()
                
                # Align with our dataframe dates
                for idx in result_df.index:
                    year = idx.year
                    matching_shares = annual_shares[annual_shares.index.year == year]
                    
                    if not matching_shares.empty:
                        result_df.loc[idx, 'Diluted Shares Outstanding'] = matching_shares.squeeze()

            # Calculate EPS where we have both net income and shares
            if ('Net Income' in result_df.columns and 
                'Diluted Shares Outstanding' in result_df.columns):
                
                mask = (result_df['Net Income'].notna() & 
                       result_df['Diluted Shares Outstanding'].notna() &
                       (result_df['Diluted Shares Outstanding'] > 0))
                
                result_df.loc[mask, 'Diluted EPS'] = (
                    result_df.loc[mask, 'Net Income'] / 
                    result_df.loc[mask, 'Diluted Shares Outstanding']
                )
        
        except Exception as e:
            logger.warning(f"Could not calculate EPS: {e}")
        
        return result_df
    
    def calculate_key_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate key financial ratios like Adobe model"""
        
        ratios_df = pd.DataFrame(index=df.index)
        
        # Revenue Growth
        if 'Total Revenue' in df.columns:
            revenue = df['Total Revenue']
            ratios_df['Revenue Growth'] = revenue.pct_change(periods=-1)  # Year-over-year
        
        # Gross Margin
        if 'Gross Profit' in df.columns and 'Total Revenue' in df.columns:
            ratios_df['Gross Margin'] = df['Gross Profit'] / df['Total Revenue']
        
        # Operating Margin
        if 'Operating Income (EBIT)' in df.columns and 'Total Revenue' in df.columns:
            ratios_df['Operating Margin'] = df['Operating Income (EBIT)'] / df['Total Revenue']
        
        # Net Margin
        if 'Net Income' in df.columns and 'Total Revenue' in df.columns:
            ratios_df['Net Margin'] = df['Net Income'] / df['Total Revenue']
        
        # Tax Rate
        if 'Tax Expense' in df.columns and 'Pre-Tax Income' in df.columns:
            ratios_df['Tax Rate'] = df['Tax Expense'] / df['Pre-Tax Income']
        
        # R&D as % of Revenue
        if 'Research & Development' in df.columns and 'Total Revenue' in df.columns:
            ratios_df['R&D as % of Revenue'] = df['Research & Development'] / df['Total Revenue']
        
        return ratios_df
    
    def save_to_csv(self, df: pd.DataFrame, ticker: str, 
                   include_ratios: bool = True) -> str:
        """Save income statement to CSV in Adobe format"""
        
        # Create filename using config
        filename = self.config.get_filename(ticker, 'income_statement')
        filepath = self.config.processed_data_path / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if include_ratios:
            # Calculate and append ratios
            ratios_df = self.calculate_key_ratios(df)
            
            # Add separator row
            separator = pd.DataFrame(index=['--- RATIOS ---'], columns=df.columns)
            separator.fillna('', inplace=True)
            
            # Combine data
            combined_df = pd.concat([df, separator, ratios_df])
            combined_df.to_csv(filepath)
        else:
            df.to_csv(filepath)
        
        logger.info(f"Income statement saved to: {filepath}")
        return str(filepath)
    
    def get_comprehensive_income_data(self, ticker: str) -> Dict:
        """Get comprehensive income statement data including ratios and metadata"""
        
        # Fetch base income statement
        income_df = self.fetch_income_statement(ticker)

        income_df = income_df[~income_df.index.duplicated(keep='first')]
        income_df = income_df.sort_index(ascending=False)
        
        # Calculate ratios
        ratios_df = self.calculate_key_ratios(income_df)
        
        # Get company-specific configuration
        company_config = self.config.get_company_config(ticker)
        
        # Save to CSV
        csv_path = self.save_to_csv(income_df, ticker)
        
        return {
            'income_statement': income_df,
            'key_ratios': ratios_df,
            'company_config': company_config,
            'csv_path': csv_path,
            'ticker': ticker.upper(),
            'last_updated': datetime.now().isoformat(),
            'data_years': len(income_df),
            'latest_year': income_df.index[0] if not income_df.empty else None
        }

# Convenience function for quick usage
def fetch_income_data(ticker: str, source: str = "yfinance") -> Dict:
    """
    Quick function to fetch income statement data
    
    Usage:
        data = fetch_income_data("ADBE")
        print(data['income_statement'])
    """
    fetcher = IncomeFetcher()
    return fetcher.get_comprehensive_income_data(ticker)

if __name__ == "__main__":
    # Example usage
    fetcher = IncomeFetcher()
    
    # Test with Adobe
    adobe_data = fetcher.get_comprehensive_income_data("ADBE")
    print(" Adobe Income Statement Data:")
    print(adobe_data['income_statement'].head())
    print(f"\n Available years: {adobe_data['data_years']}")
    print(f" Saved to: {adobe_data['csv_path']}")