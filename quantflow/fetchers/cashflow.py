# quantflow/fetchers/cashflow.py
"""
QuantFlow Financial Suite - Cash Flow Statement Fetcher
Fetches and standardizes cash flow data with focus on Free Cash Flow to Firm (FCFF)
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

class CashFlowFetcher:
    """
    Cash Flow Statement fetcher optimized for DCF modeling
    Calculates Free Cash Flow to Firm (FCFF) like Adobe DCF model
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.adobe_columns = self._get_adobe_column_mapping()
    
    def _get_adobe_column_mapping(self) -> Dict[str, str]:
        """
        Standard column mapping for cash flow statement
        Focus on items needed for FCFF calculation
        """
        return {
            # Operating Cash Flow components
            'Net Income': 'Net Income',
            'Net Income From Continuing Operations': 'Net Income',
            'netIncome': 'Net Income',
            
            'Depreciation': 'Depreciation & Amortization',
            'Depreciation And Amortization': 'Depreciation & Amortization',
            'Depreciation Depletion And Amortization': 'Depreciation & Amortization',
            'depreciationAndAmortization': 'Depreciation & Amortization',
            
            'Stock Based Compensation': 'Stock-Based Compensation',
            'Share Based Compensation': 'Stock-Based Compensation',
            'Stock-Based Compensation': 'Stock-Based Compensation',
            'stockBasedCompensation': 'Stock-Based Compensation',
            
            'Change In Working Capital': 'Changes in Working Capital',
            'Changes In Working Capital': 'Changes in Working Capital',
            'Working Capital Changes': 'Changes in Working Capital',
            'changeInWorkingCapital': 'Changes in Working Capital',
            
            'Operating Cash Flow': 'Operating Cash Flow',
            'Cash Flow From Operations': 'Operating Cash Flow',
            'Total Cash From Operating Activities': 'Operating Cash Flow',
            'operatingCashflow': 'Operating Cash Flow',
            
            # Investing Cash Flow
            'Capital Expenditure': 'Capital Expenditures',
            'Capital Expenditures': 'Capital Expenditures',
            'Cash Flow From Capex': 'Capital Expenditures',
            'capitalExpenditures': 'Capital Expenditures',
            'CapEx': 'Capital Expenditures',
            
            'Acquisitions': 'Acquisitions',
            'Acquisitions Net': 'Acquisitions',
            'Business Acquisitions Disposals': 'Acquisitions',
            'acquisitions': 'Acquisitions',
            
            'Investing Cash Flow': 'Investing Cash Flow',
            'Cash Flow From Investing': 'Investing Cash Flow',
            'Total Cash From Investing Activities': 'Investing Cash Flow',
            'investingCashflow': 'Investing Cash Flow',
            
            # Financing Cash Flow
            'Stock Repurchase': 'Stock Repurchases',
            'Repurchase Of Capital Stock': 'Stock Repurchases',
            'Common Stock Repurchase': 'Stock Repurchases',
            'stockRepurchase': 'Stock Repurchases',
            
            'Dividends Paid': 'Dividends Paid',
            'Cash Dividends Paid': 'Dividends Paid',
            'dividendsPaid': 'Dividends Paid',
            
            'Debt Proceeds': 'Debt Proceeds',
            'Long Term Debt Issuance': 'Debt Proceeds',
            'debtProceeds': 'Debt Proceeds',
            
            'Debt Repayment': 'Debt Repayment',
            'Long Term Debt Payments': 'Debt Repayment',
            'debtRepayment': 'Debt Repayment',
            
            'Financing Cash Flow': 'Financing Cash Flow',
            'Cash Flow From Financing': 'Financing Cash Flow',
            'Total Cash Flow From Financing Activities': 'Financing Cash Flow',
            'financingCashflow': 'Financing Cash Flow',
            
            # Free Cash Flow
            'Free Cash Flow': 'Free Cash Flow',
            'freeCashFlow': 'Free Cash Flow',
        }
    
    def fetch_cashflow_statement(self, ticker: str, 
                              source: str = "yfinance", 
                              allow_fallback=True) -> pd.DataFrame:
        """
        Fetch cash flow statement data for a given ticker

        Args:
            ticker: Stock ticker symbol
            source: Data source ('yfinance', 'alpha_vantage')
            allow_fallback: Whether to try fallback sources if primary fails

        Returns:
            DataFrame with standardized cash flow statement
        """
        logger.info(f"Fetching cash flow statement for {ticker} from {source}")

        try:
            if source == "yfinance":
                return self._fetch_yfinance_cashflow(ticker)
            elif source == "alpha_vantage":
                return self._fetch_alpha_vantage_cashflow(ticker)
            elif source == "iex":
                raise ValueError("IEX Cloud support has been removed or disabled")
            else:
                raise ValueError(f"Unsupported data source: {source}")
        except Exception as e:
            logger.error(f"Error fetching cash flow for {ticker}: {e}")
            if allow_fallback:
                return self._fetch_with_fallback(ticker)
            else:
                raise

    
    def _fetch_yfinance_cashflow(self, ticker: str) -> pd.DataFrame:
        ...
        yf_ticker = yf.Ticker(ticker)
        cashflow = yf_ticker.cashflow.T

        if cashflow.empty:
            raise ValueError(f"No cash flow data found for {ticker}")

        # Ensure DatetimeIndex
        cashflow.index = pd.to_datetime(cashflow.index, errors='coerce')

        # Keep one record per fiscal year (drop duplicate years)
        cashflow = cashflow[~cashflow.index.year.duplicated(keep='first')]
        cashflow = cashflow.sort_index(ascending=False)

        standardized_df = self._standardize_columns(cashflow)
        standardized_df = standardized_df.loc[:, ~standardized_df.columns.duplicated()]
        standardized_df = self._calculate_fcff_metrics(standardized_df)

        logger.info(f"Successfully fetched {len(standardized_df)} years of cash flow data for {ticker}")
        return standardized_df

    
    def _fetch_alpha_vantage_cashflow(self, ticker: str) -> pd.DataFrame:
        """Fetch cash flow from Alpha Vantage API"""
        
        api_key = self.config.alpha_vantage_api_key
        if not api_key:
            raise ValueError("Alpha Vantage API key not found")
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'CASH_FLOW',
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
    
    def _fetch_with_fallback(self, ticker: str) -> pd.DataFrame:
        fallback_sources = ["alpha_vantage", "yfinance"]
        for fallback_source in fallback_sources:
            try:
                logger.info(f"Trying fallback source: {fallback_source}")
                return self.fetch_cashflow_statement(ticker, source=fallback_source, allow_fallback=False)
            except Exception as e:
                logger.warning(f"Fallback source {fallback_source} failed: {e}")
        raise RuntimeError(f"All data sources failed for ticker {ticker}")


    
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
            'Net Income', 'Depreciation & Amortization', 'Stock-Based Compensation',
            'Changes in Working Capital', 'Operating Cash Flow', 'Capital Expenditures',
            'Free Cash Flow'
        ]
        
        for col in adobe_required_columns:
            if col not in standardized_df.columns:
                standardized_df[col] = np.nan
        
        return standardized_df
    
    def _calculate_fcff_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Free Cash Flow to Firm (FCFF) using Adobe DCF methodology
        FCFF = NOPAT + D&A - CapEx - Δ NWC
        """
        result_df = df.copy()
        
        # Calculate Free Cash Flow if missing (simple method)
        if 'Free Cash Flow' not in df.columns or df['Free Cash Flow'].isna().all():
            ocf = df.get('Operating Cash Flow', 0)
            capex = df.get('Capital Expenditures', 0)
            # CapEx is typically negative in cash flow statements
            result_df['Free Cash Flow'] = ocf + capex  # Adding because capex is negative
        
        # For FCFF calculation, we need NOPAT which requires EBIT and tax rate
        # This will be calculated when we combine with income statement data
        
        # Calculate change in working capital if missing
        if 'Changes in Working Capital' not in df.columns or df['Changes in Working Capital'].isna().all():
            # This is complex to calculate from cash flow alone
            # Would need balance sheet data for proper calculation
            pass
        
        # Ensure CapEx is negative (it's a cash outflow)
        if 'Capital Expenditures' in result_df.columns:
            capex = result_df['Capital Expenditures']
            # If CapEx values are positive, make them negative
            if (capex > 0).any():
                result_df['Capital Expenditures'] = -abs(capex)
        
        return result_df
    
    def calculate_fcff_with_income_data(self, cashflow_df: pd.DataFrame, 
                                       income_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate FCFF using the NOPAT method (like Adobe DCF model)
        FCFF = NOPAT + D&A - CapEx - Δ NWC
        
        This requires both cash flow and income statement data
        """
        result_df = cashflow_df.copy()
        
        # Align dataframes by index (year)
        common_years = cashflow_df.index.intersection(income_df.index)
        
        for year in common_years:
            # Get EBIT from income statement
            ebit = income_df.loc[year].get('Operating Income (EBIT)', 0)
            
            # Calculate tax rate from income statement
            pre_tax = income_df.loc[year].get('Pre-Tax Income', 0)
            tax_expense = income_df.loc[year].get('Tax Expense', 0)
            tax_rate = tax_expense / pre_tax if pre_tax != 0 else self.config.tax_rate
            
            # Calculate NOPAT = EBIT * (1 - Tax Rate)
            nopat = ebit * (1 - tax_rate)
            result_df.loc[year, 'NOPAT'] = nopat
            
            # Get cash flow components
            da = cashflow_df.loc[year].get('Depreciation & Amortization', 0)
            capex = cashflow_df.loc[year].get('Capital Expenditures', 0)
            delta_nwc = cashflow_df.loc[year].get('Changes in Working Capital', 0)
            
            # Calculate FCFF = NOPAT + D&A - CapEx - Δ NWC
            fcff = nopat + da - abs(capex) - delta_nwc
            result_df.loc[year, 'FCFF (NOPAT Method)'] = fcff
        
        return result_df
    
    def calculate_cash_flow_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cash flow ratios for analysis"""
        
        ratios_df = pd.DataFrame(index=df.index)
        
        # Cash Flow Growth
        if 'Operating Cash Flow' in df.columns:
            ocf = df['Operating Cash Flow']
            ratios_df['OCF Growth'] = ocf.pct_change(periods=-1)
        
        if 'Free Cash Flow' in df.columns:
            fcf = df['Free Cash Flow']
            ratios_df['FCF Growth'] = fcf.pct_change(periods=-1)
        
        # Cash Flow Margins (would need revenue data)
        # This would be calculated when combined with income statement
        
        # Capital Efficiency
        if ('Free Cash Flow' in df.columns and 
            'Capital Expenditures' in df.columns):
            fcf = df['Free Cash Flow']
            capex = abs(df['Capital Expenditures'])  # Make positive for ratio
            ratios_df['FCF to CapEx'] = fcf / capex.replace(0, np.nan)
        
        # Cash Conversion
        if ('Operating Cash Flow' in df.columns and 
            'Net Income' in df.columns):
            ocf = df['Operating Cash Flow']
            net_income = df['Net Income']
            ratios_df['Cash Conversion Ratio'] = ocf / net_income.replace(0, np.nan)
        
        return ratios_df
    
    def save_to_csv(self, df: pd.DataFrame, ticker: str, 
                   include_ratios: bool = True) -> str:
        """Save cash flow statement to CSV"""
        
        filename = self.config.get_filename(ticker, 'cashflow')
        filepath = self.config.processed_data_path / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if include_ratios:
            # Calculate and append ratios
            ratios_df = self.calculate_cash_flow_ratios(df)
            
            # Add separator row
            separator = pd.DataFrame(index=['--- RATIOS ---'], columns=df.columns)
            separator.fillna('', inplace=True)
            
            # Combine data
            combined_df = pd.concat([df, separator, ratios_df])
            combined_df.to_csv(filepath)
        else:
            df.to_csv(filepath)
        
        logger.info(f"Cash flow statement saved to: {filepath}")
        return str(filepath)
    
    def get_comprehensive_cashflow_data(self, ticker: str) -> Dict:
        """Get comprehensive cash flow data including FCFF metrics"""
        
        # Fetch base cash flow statement
        cashflow_df = self.fetch_cashflow_statement(ticker)

        cashflow_df = cashflow_df[~cashflow_df.index.duplicated(keep='first')]
        cashflow_df = cashflow_df.sort_index(ascending=False)
        
        # Calculate ratios
        ratios_df = self.calculate_cash_flow_ratios(cashflow_df)
        
        # Get company-specific configuration
        company_config = self.config.get_company_config(ticker)
        
        # Save to CSV
        csv_path = self.save_to_csv(cashflow_df, ticker)
        
        # Extract key metrics for DCF
        latest_data = cashflow_df.iloc[0] if not cashflow_df.empty else pd.Series()
        
        dcf_metrics = {
            'operating_cash_flow': latest_data.get('Operating Cash Flow', 0),
            'free_cash_flow': latest_data.get('Free Cash Flow', 0),
            'capital_expenditures': latest_data.get('Capital Expenditures', 0),
            'depreciation_amortization': latest_data.get('Depreciation & Amortization', 0),
            'stock_based_compensation': latest_data.get('Stock-Based Compensation', 0),
            'working_capital_change': latest_data.get('Changes in Working Capital', 0)
        }
        
        return {
            'cashflow_statement': cashflow_df,
            'key_ratios': ratios_df,
            'dcf_metrics': dcf_metrics,
            'company_config': company_config,
            'csv_path': csv_path,
            'ticker': ticker.upper(),
            'last_updated': datetime.now().isoformat(),
            'data_years': len(cashflow_df),
            'latest_year': cashflow_df.index[0] if not cashflow_df.empty else None
        }

# Convenience function
def fetch_cashflow_data(ticker: str, source: str = "yfinance") -> Dict:
    """
    Quick function to fetch cash flow data
    
    Usage:
        data = fetch_cashflow_data("ADBE")
        print(data['cashflow_statement'])
    """
    fetcher = CashFlowFetcher()
    return fetcher.get_comprehensive_cashflow_data(ticker)

if __name__ == "__main__":
    # Example usage
    fetcher = CashFlowFetcher()
    
    # Test with Adobe
    adobe_data = fetcher.get_comprehensive_cashflow_data("ADBE")
    print(" Adobe Cash Flow Data:")
    print(adobe_data['cashflow_statement'].head())
    print(f"\n Key FCFF Metrics:")
    for metric, value in adobe_data['dcf_metrics'].items():
        if value != 0:
            print(f"   {metric}: ${value:,.0f}")
    print(f"\n Saved to: {adobe_data['csv_path']}")