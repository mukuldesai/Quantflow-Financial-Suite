# quantflow/fetchers/__init__.py
"""
QuantFlow Financial Suite - Unified Financial Data Fetcher
Combines income, balance, cash flow, and market fetchers for comprehensive Adobe DCF-style analysis
"""

from .income import IncomeFetcher, fetch_income_data
from .balance import BalanceFetcher, fetch_balance_data  
from .cashflow import CashFlowFetcher, fetch_cashflow_data
from .market import MarketDataFetcher, fetch_market_data


import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from datetime import datetime, timedelta
import warnings


from ..config import get_config

logger = logging.getLogger(__name__)

class FinancialDataFetcher:
    """
    Unified fetcher for all financial statements with Adobe DCF optimization
    Provides comprehensive financial analysis with enhanced error handling and data quality checks
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.income_fetcher = IncomeFetcher(config)
        self.balance_fetcher = BalanceFetcher(config)
        self.cashflow_fetcher = CashFlowFetcher(config)
        self.market_fetcher = MarketDataFetcher(config)
        
        # Data quality tracking
        self.data_quality_issues = []
        self.fetch_errors = {}
    
    def fetch_all_financials(self, ticker: str, 
                           source: str = "yfinance",
                           include_peer_analysis: bool = False,
                           peer_tickers: Optional[List[str]] = None) -> Dict:
        """
        Fetch comprehensive financial data for Adobe DCF-style analysis
        
        Args:
            ticker: Stock ticker symbol
            source: Data source preference ('yfinance', 'alpha_vantage')
            include_peer_analysis: Whether to include peer comparison
            peer_tickers: List of peer tickers for comparison
            
        Returns:
            Dictionary with all financial statements, market data, and derived metrics
        """
        logger.info(f" Fetching comprehensive financial data for {ticker}")
        
        # Reset tracking variables
        self.data_quality_issues = []
        self.fetch_errors = {}
        
        try:
            # Fetch all financial statements with error tracking
            income_data = self._safe_fetch(
                self.income_fetcher.get_comprehensive_income_data, 
                ticker, 'income_statement'
            )
            
            balance_data = self._safe_fetch(
                self.balance_fetcher.get_comprehensive_balance_data,
                ticker, 'balance_sheet'
            )
            
            cashflow_data = self._safe_fetch(
                self.cashflow_fetcher.get_comprehensive_cashflow_data,
                ticker, 'cashflow_statement'
            )
            
            market_data = self._safe_fetch(
                lambda t: self.market_fetcher.get_comprehensive_market_data(t, peer_tickers=peer_tickers),
                ticker, 'market_data'
            )

            
            # Check data quality before proceeding
            self._validate_data_quality(income_data, balance_data, cashflow_data, market_data)
            
            # Combine and enhance the data
            enhanced_data = self._combine_and_enhance_data(
                ticker, income_data, balance_data, cashflow_data, market_data
            )
            
            # Add peer analysis if requested
            if include_peer_analysis and peer_tickers:
                enhanced_data['peer_analysis'] = self._perform_peer_analysis(
                    ticker, peer_tickers, enhanced_data
                )
            
            # Add data quality report
            enhanced_data['data_quality_report'] = self._generate_data_quality_report()
            
            logger.info(f" Successfully fetched all financial data for {ticker}")
            return enhanced_data
            
        except Exception as e:
            logger.error(f" Error fetching comprehensive data for {ticker}: {e}")
            # Return partial data if available
            return self._create_partial_data_response(ticker, e)
    
    def _safe_fetch(self, fetch_method, ticker: str, data_type: str, *args) -> Dict:
        """Safely fetch data with error handling and fallback"""
        try:
            return fetch_method(ticker, *args)
        except Exception as e:
            self.fetch_errors[data_type] = str(e)
            logger.warning(f"Failed to fetch {data_type} for {ticker}: {e}")
            return self._get_empty_data_structure(data_type)
    
    def _get_empty_data_structure(self, data_type: str) -> Dict:
        """Return empty data structure for failed fetches"""
        base_structure = {
            'ticker': '',
            'last_updated': datetime.now().isoformat(),
            'csv_path': '',
            'data_years': 0,
            'latest_year': None,
            'error': f'Failed to fetch {data_type}'
        }
        
        if data_type == 'income_statement':
            base_structure.update({
                'income_statement': pd.DataFrame(),
                'key_ratios': pd.DataFrame(),
                'dcf_metrics': {}
            })
        elif data_type == 'balance_sheet':
            base_structure.update({
                'balance_sheet': pd.DataFrame(), 
                'key_ratios': pd.DataFrame(),
                'dcf_metrics': {}
            })
        elif data_type == 'cashflow_statement':
            base_structure.update({
                'cashflow_statement': pd.DataFrame(),
                'key_ratios': pd.DataFrame(), 
                'dcf_metrics': {}
            })
        elif data_type == 'market_data':
            base_structure.update({
                'market_data': {},
                'dcf_inputs': {},
                'peer_comparison': None
            })
        
        return base_structure
    
    def _validate_data_quality(self, income_data: Dict, balance_data: Dict, 
                              cashflow_data: Dict, market_data: Dict) -> None:
        """Validate data quality and track issues"""
        
        # Check if we have any data at all
        has_income = not income_data.get('income_statement', pd.DataFrame()).empty
        has_balance = not balance_data.get('balance_sheet', pd.DataFrame()).empty  
        has_cashflow = not cashflow_data.get('cashflow_statement', pd.DataFrame()).empty
        has_market = bool(market_data.get('market_data', {}))
        
        if not any([has_income, has_balance, has_cashflow]):
            self.data_quality_issues.append("CRITICAL: No financial statement data available")
        
        if not has_market:
            self.data_quality_issues.append("WARNING: No market data available")
        
        # Check data completeness
        if has_income:
            income_df = income_data['income_statement']
            if len(income_df) < 3:
                self.data_quality_issues.append("WARNING: Less than 3 years of income data")
            
            # Check for key missing fields
            key_income_fields = ['Total Revenue', 'Operating Income (EBIT)', 'Net Income']
            missing_fields = [f for f in key_income_fields if f not in income_df.columns or income_df[f].isna().all()]
            if missing_fields:
                self.data_quality_issues.append(f"WARNING: Missing key income fields: {missing_fields}")
        
        # Check temporal alignment
        if has_income and has_balance and has_cashflow:
            income_years = set(income_data['income_statement'].index.year)
            balance_years = set(balance_data['balance_sheet'].index.year)
            cashflow_years = set(cashflow_data['cashflow_statement'].index.year)
            
            common_years = income_years.intersection(balance_years).intersection(cashflow_years)
            if len(common_years) < 2:
                self.data_quality_issues.append("WARNING: Less than 2 years of aligned financial data")
    
    def _combine_and_enhance_data(self, ticker: str, 
                                 income_data: Dict, 
                                 balance_data: Dict,
                                 cashflow_data: Dict,
                                 market_data: Dict) -> Dict:
        """Combine all financial data with enhanced Adobe DCF calculations"""
        
        # Get the dataframes
        income_df = income_data.get('income_statement', pd.DataFrame())
        balance_df = balance_data.get('balance_sheet', pd.DataFrame())
        cashflow_df = cashflow_data.get('cashflow_statement', pd.DataFrame())
        market_info = market_data.get('market_data', {})
        
        # Enhanced FCFF calculation using NOPAT method (Adobe style)
        enhanced_cashflow_df = self._calculate_enhanced_fcff(income_df, cashflow_df)
        
        # Calculate comprehensive ratios using all statements
        comprehensive_ratios = self._calculate_comprehensive_ratios(
            income_df, balance_df, enhanced_cashflow_df
        )
        
        # Calculate WACC with market data integration
        wacc_components = self._calculate_enhanced_wacc(ticker, balance_df, market_info)
        
        # Create Adobe-style DCF summary
        dcf_summary = self._create_enhanced_dcf_summary(
            income_df, balance_df, enhanced_cashflow_df, market_info, ticker
        )
        
        # Calculate working capital changes for FCFF
        working_capital_analysis = self._analyze_working_capital(balance_df)
        
        # Performance benchmarking
        performance_metrics = self._calculate_performance_metrics(
            income_df, balance_df, enhanced_cashflow_df, market_info
        )
        
        return {
            'ticker': ticker.upper(),
            'last_updated': datetime.now().isoformat(),
            'fetch_timestamp': datetime.now().isoformat(),
            
            # Core Financial Statements (Enhanced)
            'income_statement': income_df,
            'balance_sheet': balance_df,
            'cashflow_statement': enhanced_cashflow_df,
            
            # Individual fetcher metrics
            'income_metrics': income_data.get('dcf_metrics', {}),
            'balance_metrics': balance_data.get('dcf_metrics', {}),
            'cashflow_metrics': cashflow_data.get('dcf_metrics', {}),

            # Market Data Integration
            'market_data': market_info,
            'dcf_market_inputs': market_data.get('dcf_inputs', {}),
            'peer_comparison': market_data.get('peer_comparison'),
            
            # Enhanced Combined Analysis
            'comprehensive_ratios': comprehensive_ratios,
            'wacc_components': wacc_components,
            'dcf_summary': dcf_summary,
            'working_capital_analysis': working_capital_analysis,
            'performance_metrics': performance_metrics,
            
            # Adobe DCF Specific Calculations
            'fcff_calculations': self._get_fcff_breakdown(enhanced_cashflow_df),
            'growth_analysis': self._analyze_growth_trends(income_df),
            'margin_analysis': self._analyze_margin_trends(income_df),
            
            # Company Configuration
            'company_config': self.config.get_company_config(ticker),
            
            # File Management
            'csv_paths': {
                'income': income_data.get('csv_path', ''),
                'balance': balance_data.get('csv_path', ''), 
                'cashflow': cashflow_data.get('csv_path', ''),
                'market': market_data.get('csv_path', '')
            },
            
            # Data Quality Information
            'data_quality': {
                'years_available': {
                    'income': len(income_df),
                    'balance': len(balance_df),
                    'cashflow': len(enhanced_cashflow_df)
                },
                'latest_year': self._get_latest_common_year(income_df, balance_df, enhanced_cashflow_df),
                'data_completeness_score': self._calculate_completeness_score(income_df, balance_df, enhanced_cashflow_df),
                'issues': self.data_quality_issues,
                'fetch_errors': self.fetch_errors
            }
        }
    
    def _calculate_enhanced_fcff(self, income_df: pd.DataFrame, 
                               cashflow_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced FCFF using Adobe DCF NOPAT methodology"""
        
        if income_df.empty or cashflow_df.empty:
            return cashflow_df
        
        enhanced_df = cashflow_df.copy()
        
        # Get common years
        common_years = income_df.index.intersection(cashflow_df.index)
        
        for year in common_years:
            try:
                # Get EBIT from income statement
                ebit = income_df.loc[year].get('Operating Income (EBIT)', 0)
                
                # Calculate tax rate
                pre_tax = income_df.loc[year].get('Pre-Tax Income', 0)
                tax_expense = income_df.loc[year].get('Tax Expense', 0)
                tax_rate = tax_expense / pre_tax if pre_tax != 0 else self.config.tax_rate
                
                # Calculate NOPAT = EBIT * (1 - Tax Rate)
                nopat = ebit * (1 - tax_rate)
                enhanced_df.loc[year, 'NOPAT'] = nopat
                enhanced_df.loc[year, 'Tax Rate Used'] = tax_rate
                
                # Get cash flow components
                da = cashflow_df.loc[year].get('Depreciation & Amortization', 0)
                capex = cashflow_df.loc[year].get('Capital Expenditures', 0)
                delta_nwc = cashflow_df.loc[year].get('Changes in Working Capital', 0)
                
                # Calculate FCFF = NOPAT + D&A - CapEx - Δ NWC (Adobe method)
                fcff = nopat + da - abs(capex) - delta_nwc
                enhanced_df.loc[year, 'FCFF (NOPAT Method)'] = fcff
                
                # Calculate FCFF components breakdown
                enhanced_df.loc[year, 'FCFF - NOPAT Component'] = nopat
                enhanced_df.loc[year, 'FCFF - D&A Component'] = da
                enhanced_df.loc[year, 'FCFF - CapEx Component'] = -abs(capex)
                enhanced_df.loc[year, 'FCFF - NWC Component'] = -delta_nwc
                
            except Exception as e:
                logger.warning(f"Could not calculate FCFF for {year}: {e}")
        
        return enhanced_df
    
    def _calculate_comprehensive_ratios(self, income_df: pd.DataFrame,
                                      balance_df: pd.DataFrame,
                                      cashflow_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive financial ratios using all three statements"""
        
        # Get common years across all statements
        common_years = income_df.index.intersection(balance_df.index).intersection(cashflow_df.index)
        
        if len(common_years) == 0:
            logger.warning("No common years found across financial statements")
            return pd.DataFrame()
        
        ratios_df = pd.DataFrame(index=common_years)
        
        for year in common_years:
            try:
                # Core financial metrics
                revenue = income_df.loc[year].get('Total Revenue', 0)
                gross_profit = income_df.loc[year].get('Gross Profit', 0)
                ebit = income_df.loc[year].get('Operating Income (EBIT)', 0)
                net_income = income_df.loc[year].get('Net Income', 0)
                
                total_assets = balance_df.loc[year].get('Total Assets', 0)
                total_equity = balance_df.loc[year].get('Total Stockholders\' Equity', 0)
                total_debt = balance_df.loc[year].get('Total Debt', 0)
                
                ocf = cashflow_df.loc[year].get('Operating Cash Flow', 0)
                fcf = cashflow_df.loc[year].get('Free Cash Flow', 0)
                fcff_nopat = cashflow_df.loc[year].get('FCFF (NOPAT Method)', 0)
                capex = abs(cashflow_df.loc[year].get('Capital Expenditures', 0))
                
                # Profitability Ratios (Adobe DCF focus)
                if revenue > 0:
                    ratios_df.loc[year, 'Gross Margin %'] = (gross_profit / revenue) * 100
                    ratios_df.loc[year, 'Operating Margin %'] = (ebit / revenue) * 100
                    ratios_df.loc[year, 'Net Margin %'] = (net_income / revenue) * 100
                    ratios_df.loc[year, 'OCF Margin %'] = (ocf / revenue) * 100
                    ratios_df.loc[year, 'FCF Margin %'] = (fcf / revenue) * 100
                    ratios_df.loc[year, 'FCFF Margin %'] = (fcff_nopat / revenue) * 100
                
                # Efficiency Ratios
                if total_assets > 0:
                    ratios_df.loc[year, 'Asset Turnover'] = revenue / total_assets
                    ratios_df.loc[year, 'ROA %'] = (net_income / total_assets) * 100
                
                if total_equity > 0:
                    ratios_df.loc[year, 'ROE %'] = (net_income / total_equity) * 100
                
                # Leverage Ratios (important for WACC)
                if total_equity > 0 and total_debt > 0:
                    ratios_df.loc[year, 'Debt-to-Equity'] = total_debt / total_equity
                    ratios_df.loc[year, 'Equity Ratio %'] = (total_equity / (total_debt + total_equity)) * 100
                
                # Cash Flow Quality Ratios
                if net_income > 0:
                    ratios_df.loc[year, 'Cash Conversion Ratio'] = ocf / net_income
                
                if capex > 0:
                    ratios_df.loc[year, 'FCF to CapEx Ratio'] = fcf / capex
                    ratios_df.loc[year, 'FCFF to CapEx Ratio'] = fcff_nopat / capex
                
                # Growth Rates (year-over-year)
                if year != common_years[-1]:  # Not the oldest year
                    prev_year_idx = common_years[common_years.get_loc(year) + 1]
                    
                    prev_revenue = income_df.loc[prev_year_idx].get('Total Revenue', 0)
                    if prev_revenue > 0:
                        ratios_df.loc[year, 'Revenue Growth %'] = ((revenue - prev_revenue) / prev_revenue) * 100
                    
                    prev_fcf = cashflow_df.loc[prev_year_idx].get('Free Cash Flow', 0)
                    if prev_fcf > 0:
                        ratios_df.loc[year, 'FCF Growth %'] = ((fcf - prev_fcf) / prev_fcf) * 100
                
            except Exception as e:
                logger.warning(f"Could not calculate ratios for {year}: {e}")
        
        return ratios_df
    
    def _calculate_enhanced_wacc(self, ticker: str, balance_df: pd.DataFrame, 
                               market_info: Dict) -> Dict:
        """Calculate WACC with market data integration"""
        
        # Get market data if available
        current_price = market_info.get('current_price', 0)
        market_cap = market_info.get('market_cap', 0)
        beta = market_info.get('beta', self.config.default_beta)
        
        # Get latest balance sheet data
        if not balance_df.empty:
            latest_data = balance_df.iloc[0]
            book_debt = latest_data.get('Total Debt', 0)
            book_equity = latest_data.get('Total Stockholders\' Equity', 0)
        else:
            book_debt = 0
            book_equity = 0
        
        # Market values (preferred for WACC)
        market_value_equity = market_cap if market_cap > 0 else book_equity
        market_value_debt = book_debt  # Approximate with book value
        
        total_capital = market_value_equity + market_value_debt
        
        # Weights
        if total_capital > 0:
            weight_equity = market_value_equity / total_capital
            weight_debt = market_value_debt / total_capital
        else:
            # Default for tech companies
            weight_equity = 0.95
            weight_debt = 0.05
        
        # Cost components from config
        rf_rate = self.config.risk_free_rate
        market_premium = self.config.market_risk_premium
        tax_rate = self.config.tax_rate
        
        # Cost of Equity = Risk-free rate + Beta * Market risk premium
        cost_of_equity = rf_rate + (beta * market_premium)
        
        # Cost of Debt (would need more sophisticated calculation in practice)
        cost_of_debt = getattr(self.config, 'default_cost_of_debt', 0.035)
        after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
        
        # WACC calculation
        wacc = (weight_equity * cost_of_equity) + (weight_debt * after_tax_cost_of_debt)
        
        return {
            'inputs': {
                'risk_free_rate': rf_rate,
                'market_risk_premium': market_premium,
                'beta': beta,
                'tax_rate': tax_rate
            },
            'market_values': {
                'equity': market_value_equity,
                'debt': market_value_debt,
                'total_capital': total_capital
            },
            'weights': {
                'equity_weight': weight_equity,
                'debt_weight': weight_debt
            },
            'costs': {
                'cost_of_equity': cost_of_equity,
                'cost_of_debt': cost_of_debt,
                'after_tax_cost_of_debt': after_tax_cost_of_debt
            },
            'wacc': wacc,
            'wacc_percentage': wacc * 100
        }
    
    def _create_enhanced_dcf_summary(self, income_df: pd.DataFrame,
                                   balance_df: pd.DataFrame,
                                   cashflow_df: pd.DataFrame,
                                   market_info: Dict,
                                   ticker: str) -> Dict:
        """Create Adobe DCF-style summary with enhanced analysis"""
        
        if all(df.empty for df in [income_df, balance_df, cashflow_df]):
            logger.warning("All financial statements are empty — skipping DCF summary.")
            return {'error': 'All financial statements are empty — cannot generate DCF summary'}



        
        # Get latest year data
        latest_income = income_df.iloc[0]
        latest_balance = balance_df.iloc[0]
        if cashflow_df.empty:
            logger.warning(f"No cashflow data available for {ticker}. Skipping DCF summary.")
            return {'error': 'Cashflow data missing — cannot build DCF summary'}

        latest_cashflow = cashflow_df.iloc[0]

        
        
        # Revenue analysis
        revenue_series = income_df['Total Revenue'].dropna()
        fcf_series = cashflow_df['Free Cash Flow'].dropna()
        
        # Growth calculations
        revenue_growth_rates = []
        fcf_growth_rates = []
        
        for i in range(min(3, len(revenue_series) - 1)):
            current_rev = revenue_series.iloc[i]
            prior_rev = revenue_series.iloc[i + 1]
            if prior_rev > 0:
                growth = (current_rev - prior_rev) / prior_rev
                revenue_growth_rates.append(growth)
        
        for i in range(min(3, len(fcf_series) - 1)):
            current_fcf = fcf_series.iloc[i]
            prior_fcf = fcf_series.iloc[i + 1]
            if prior_fcf > 0:
                growth = (current_fcf - prior_fcf) / prior_fcf
                fcf_growth_rates.append(growth)
        
        avg_revenue_growth = np.mean(revenue_growth_rates) if revenue_growth_rates else 0
        avg_fcf_growth = np.mean(fcf_growth_rates) if fcf_growth_rates else 0
        
        # Get current market valuation
        current_price = market_info.get('current_price', 0)
        market_cap = market_info.get('market_cap', 0)
        
        return {
            'current_financials': {
                'revenue': latest_income.get('Total Revenue', 0),
                'gross_profit': latest_income.get('Gross Profit', 0),
                'ebit': latest_income.get('Operating Income (EBIT)', 0),
                'net_income': latest_income.get('Net Income', 0),
                'operating_cash_flow': latest_cashflow.get('Operating Cash Flow', 0),
                'free_cash_flow': latest_cashflow.get('Free Cash Flow', 0),
                'fcff_nopat_method': latest_cashflow.get('FCFF (NOPAT Method)', 0),
                'total_assets': latest_balance.get('Total Assets', 0),
                'total_debt': latest_balance.get('Total Debt', 0),
                'cash': latest_balance.get('Cash & Cash Equivalents', 0),
                'stockholders_equity': latest_balance.get('Total Stockholders\' Equity', 0)
            },
            
            'market_valuation': {
                'current_stock_price': current_price,
                'market_capitalization': market_cap,
                'enterprise_value': market_info.get('enterprise_value', 0),
                'shares_outstanding': market_info.get('shares_outstanding', 0)
            },
            
            'historical_growth': {
                'revenue_growth_3yr_avg': avg_revenue_growth,
                'fcf_growth_3yr_avg': avg_fcf_growth,
                'revenue_cagr': self._calculate_cagr(revenue_series),
                'fcf_cagr': self._calculate_cagr(fcf_series)
            },
            
            'current_margins': {
                'gross_margin': self._safe_ratio(latest_income.get('Gross Profit', 0), latest_income.get('Total Revenue', 1)),
                'operating_margin': self._safe_ratio(latest_income.get('Operating Income (EBIT)', 0), latest_income.get('Total Revenue', 1)),
                'net_margin': self._safe_ratio(latest_income.get('Net Income', 0), latest_income.get('Total Revenue', 1)),
                'fcf_margin': self._safe_ratio(latest_cashflow.get('Free Cash Flow', 0), latest_income.get('Total Revenue', 1)),
                'fcff_margin': self._safe_ratio(latest_cashflow.get('FCFF (NOPAT Method)', 0), latest_income.get('Total Revenue', 1))
            },
            
            'dcf_assumptions': {
                'projection_years': self.config.projection_years,
                'terminal_growth_rate': self.config.terminal_growth_rate,
                'tax_rate': self.config.tax_rate,
                'projected_growth_rates': self.config.get_company_growth_rates(ticker),
                'margin_assumptions': self.config.get_company_margins(ticker)
            },
            
            'valuation_multiples': {
                'pe_ratio': market_info.get('pe_ratio', 0),
                'ev_to_revenue': market_info.get('ev_to_revenue', 0),
                'ev_to_ebitda': market_info.get('ev_to_ebitda', 0),
                'price_to_book': market_info.get('price_to_book', 0),
                'price_to_fcf': self._safe_ratio(current_price, latest_cashflow.get('Free Cash Flow', 1) / market_info.get('shares_outstanding', 1)) if market_info.get('shares_outstanding', 0) > 0 else 0
            }
        }
    
    def _analyze_working_capital(self, balance_df: pd.DataFrame) -> Dict:
        """Analyze working capital changes for FCFF calculation"""
        
        if balance_df.empty:
            return {}
        
        wc_analysis = {}
        
        # Calculate working capital for each year
        for idx, row in balance_df.iterrows():
            current_assets = row.get('Total Current Assets', 0)
            current_liabilities = row.get('Current Liabilities', 0)
            working_capital = current_assets - current_liabilities
            wc_analysis[idx] = working_capital
        
        # Calculate year-over-year changes
        wc_changes = {}
        years = sorted(wc_analysis.keys(), reverse=True)  # Most recent first
        
        for i in range(len(years) - 1):
            current_year = years[i]
            prior_year = years[i + 1]
            change = wc_analysis[current_year] - wc_analysis[prior_year]
            wc_changes[current_year] = change
        
        return {
            'working_capital_by_year': wc_analysis,
            'working_capital_changes': wc_changes,
            'avg_wc_change': np.mean(list(wc_changes.values())) if wc_changes else 0,
            'wc_volatility': np.std(list(wc_changes.values())) if len(wc_changes) > 1 else 0
        }
    
    def _calculate_performance_metrics(self, income_df: pd.DataFrame,
                                     balance_df: pd.DataFrame,
                                     cashflow_df: pd.DataFrame,
                                     market_info: Dict) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        performance = {}
        
        if not income_df.empty:
            # Revenue consistency
            revenue_series = income_df['Total Revenue'].dropna()
            revenue_growth_rates = revenue_series.pct_change(periods=-1).dropna()
            
            performance['revenue_metrics'] = {
                'revenue_growth_volatility': revenue_growth_rates.std() if len(revenue_growth_rates) > 1 else 0,
                'revenue_growth_consistency': 1 - (revenue_growth_rates.std() / abs(revenue_growth_rates.mean())) if revenue_growth_rates.mean() != 0 else 0,
                'years_of_positive_growth': (revenue_growth_rates > 0).sum(),
                'total_years': len(revenue_growth_rates)
            }
        
        if not cashflow_df.empty:
            # Cash flow quality
            fcf_series = cashflow_df['Free Cash Flow'].dropna()
            ocf_series = cashflow_df['Operating Cash Flow'].dropna()
            
            performance['cashflow_metrics'] = {
                'years_positive_fcf': (fcf_series > 0).sum(),
                'fcf_consistency_score': (fcf_series > 0).mean() if len(fcf_series) > 0 else 0,
                'avg_fcf_conversion': (fcf_series / ocf_series).mean() if len(ocf_series) > 0 and (ocf_series != 0).all() else 0
            }
        
        # Market performance (if available)
        if market_info:
            performance['market_metrics'] = {
                'current_valuation': {
                    'market_cap': market_info.get('market_cap', 0),
                    'enterprise_value': market_info.get('enterprise_value', 0),
                    'price_performance_1yr': market_info.get('return_1_year', 0)
                }
            }
        
        return performance
    
    def _get_fcff_breakdown(self, cashflow_df: pd.DataFrame) -> Dict:
        """Get detailed FCFF calculation breakdown"""
        
        if cashflow_df.empty:
            return {}
        
        fcff_breakdown = {}
        
        for idx, row in cashflow_df.iterrows():
            if 'FCFF (NOPAT Method)' in row and not pd.isna(row['FCFF (NOPAT Method)']):
                fcff_breakdown[idx] = {
                    'nopat': row.get('FCFF - NOPAT Component', 0),
                    'depreciation_amortization': row.get('FCFF - D&A Component', 0),
                    'capex': row.get('FCFF - CapEx Component', 0),
                    'working_capital_change': row.get('FCFF - NWC Component', 0),
                    'total_fcff': row.get('FCFF (NOPAT Method)', 0),
                    'tax_rate_used': row.get('Tax Rate Used', 0)
                }
        
        return fcff_breakdown
    
    def _analyze_growth_trends(self, income_df: pd.DataFrame) -> Dict:
        """Analyze revenue and profit growth trends"""
        
        if income_df.empty:
            return {}
        
        trends = {}
        
        # Revenue trend analysis
        revenue_series = income_df['Total Revenue'].dropna()
        if len(revenue_series) > 1:
            revenue_growth = revenue_series.pct_change(periods=-1).dropna()
            trends['revenue'] = {
                'cagr': self._calculate_cagr(revenue_series),
                'average_growth': revenue_growth.mean(),
                'growth_volatility': revenue_growth.std(),
                'growth_trend': 'accelerating' if revenue_growth.iloc[0] > revenue_growth.iloc[-1] else 'decelerating'
            }
        
        # Profit trend analysis
        ebit_series = income_df['Operating Income (EBIT)'].dropna()
        if len(ebit_series) > 1:
            ebit_growth = ebit_series.pct_change(periods=-1).dropna()
            trends['operating_profit'] = {
                'cagr': self._calculate_cagr(ebit_series),
                'average_growth': ebit_growth.mean(),
                'growth_volatility': ebit_growth.std()
            }
        
        return trends
    
    def _analyze_margin_trends(self, income_df: pd.DataFrame) -> Dict:
        """Analyze margin expansion/contraction trends"""
        
        if income_df.empty:
            return {}
        
        margin_trends = {}
        
        # Calculate margins over time
        revenue = income_df['Total Revenue']
        gross_profit = income_df.get('Gross Profit', pd.Series())
        ebit = income_df.get('Operating Income (EBIT)', pd.Series())
        net_income = income_df.get('Net Income', pd.Series())
        
        if not revenue.empty:
            # Gross margin trend
            if not gross_profit.empty:
                gross_margins = (gross_profit / revenue).dropna()
                margin_trends['gross_margin'] = {
                    'current': gross_margins.iloc[0] if len(gross_margins) > 0 else 0,
                    'average': gross_margins.mean(),
                    'trend': 'expanding' if len(gross_margins) > 1 and gross_margins.iloc[0] > gross_margins.iloc[-1] else 'contracting',
                    'volatility': gross_margins.std()
                }
            
            # Operating margin trend
            if not ebit.empty:
                operating_margins = (ebit / revenue).dropna()
                margin_trends['operating_margin'] = {
                    'current': operating_margins.iloc[0] if len(operating_margins) > 0 else 0,
                    'average': operating_margins.mean(),
                    'trend': 'expanding' if len(operating_margins) > 1 and operating_margins.iloc[0] > operating_margins.iloc[-1] else 'contracting',
                    'volatility': operating_margins.std()
                }
        
        return margin_trends
    
    def _perform_peer_analysis(self, ticker: str, peer_tickers: List[str], 
                             company_data: Dict) -> Dict:
        """Perform peer comparison analysis"""
        
        peer_analysis = {'primary_ticker': ticker, 'peers': {}}
        
        # Get peer data
        for peer in peer_tickers:
            try:
                peer_data = self.fetch_all_financials(peer, include_peer_analysis=False)
                
                peer_summary = {
                    'ticker': peer,
                    'current_metrics': peer_data['dcf_summary']['current_financials'],
                    'margins': peer_data['dcf_summary']['current_margins'],
                    'growth': peer_data['dcf_summary']['historical_growth'],
                    'wacc': peer_data['wacc_components']['wacc'],
                    'market_data': peer_data.get('market_data', {})
                }
                
                peer_analysis['peers'][peer] = peer_summary
                
            except Exception as e:
                logger.warning(f"Could not fetch peer data for {peer}: {e}")
                peer_analysis['peers'][peer] = {'error': str(e)}
        
        # Calculate peer averages and comparisons
        if peer_analysis['peers']:
            peer_analysis['comparison'] = self._calculate_peer_comparisons(
                company_data, peer_analysis['peers']
            )
        
        return peer_analysis
    
    def _calculate_peer_comparisons(self, company_data: Dict, peers_data: Dict) -> Dict:
        """Calculate peer comparison metrics"""
        
        # Extract metrics for comparison
        company_metrics = company_data['dcf_summary']
        
        peer_metrics = []
        for peer, data in peers_data.items():
            if 'error' not in data:
                peer_metrics.append(data)
        
        if not peer_metrics:
            return {}
        
        # Calculate peer averages
        peer_averages = {}
        
        # Revenue growth comparison
        company_revenue_growth = company_metrics['historical_growth']['revenue_growth_3yr_avg']
        peer_revenue_growths = [p['growth']['revenue_growth_3yr_avg'] for p in peer_metrics if 'growth' in p]
        
        if peer_revenue_growths:
            peer_averages['revenue_growth'] = {
                'company': company_revenue_growth,
                'peer_average': np.mean(peer_revenue_growths),
                'peer_median': np.median(peer_revenue_growths),
                'company_vs_peers': 'above' if company_revenue_growth > np.mean(peer_revenue_growths) else 'below'
            }
        
        # Margin comparison
        company_operating_margin = company_metrics['current_margins']['operating_margin']
        peer_operating_margins = [p['margins']['operating_margin'] for p in peer_metrics if 'margins' in p]
        
        if peer_operating_margins:
            peer_averages['operating_margin'] = {
                'company': company_operating_margin,
                'peer_average': np.mean(peer_operating_margins),
                'peer_median': np.median(peer_operating_margins),
                'company_vs_peers': 'above' if company_operating_margin > np.mean(peer_operating_margins) else 'below'
            }
        
        return peer_averages
    
    def _generate_data_quality_report(self) -> Dict:
        """Generate comprehensive data quality report"""
        
        return {
            'overall_quality': 'Good' if len(self.data_quality_issues) == 0 else 'Fair' if len(self.data_quality_issues) < 3 else 'Poor',
            'issues_found': self.data_quality_issues,
            'fetch_errors': self.fetch_errors,
            'recommendations': self._get_data_quality_recommendations()
        }
    
    def _get_data_quality_recommendations(self) -> List[str]:
        """Get recommendations based on data quality issues"""
        
        recommendations = []
        
        if 'income_statement' in self.fetch_errors:
            recommendations.append("Consider using alternative data source for income statement")
        
        if any('Less than' in issue for issue in self.data_quality_issues):
            recommendations.append("Consider extending historical data range for better trend analysis")
        
        if any('Missing' in issue for issue in self.data_quality_issues):
            recommendations.append("Verify data completeness with SEC filings")
        
        if not recommendations:
            recommendations.append("Data quality is good for DCF analysis")
        
        return recommendations
    
    def _create_partial_data_response(self, ticker: str, error: Exception) -> Dict:
        """Create response when data fetch fails"""
        
        return {
            'ticker': ticker.upper(),
            'error': str(error),
            'partial_data': True,
            'data_quality_report': {
                'overall_quality': 'Failed',
                'issues_found': [f"Critical error: {error}"],
                'fetch_errors': self.fetch_errors
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_latest_common_year(self, *dataframes) -> Optional[datetime]:
        """Get the latest year common across all dataframes"""
        
        non_empty_dfs = [df for df in dataframes if not df.empty]
        if not non_empty_dfs:
            return None
        
        # Get intersection of all indices
        common_dates = non_empty_dfs[0].index
        for df in non_empty_dfs[1:]:
            common_dates = common_dates.intersection(df.index)
        
        return common_dates[0] if len(common_dates) > 0 else None
    
    def _calculate_completeness_score(self, *dataframes) -> float:
        """Calculate data completeness score (0-100)"""
        
        total_score = 0
        max_score = 0
        
        for df in dataframes:
            if not df.empty:
                # Score based on years of data and key field completion
                years_score = min(len(df) * 10, 50)  # Max 50 points for 5+ years
                
                # Key fields completion
                key_fields_score = 0
                if 'Total Revenue' in df.columns:
                    completion_rate = (df['Total Revenue'].notna().sum() / len(df))
                    key_fields_score = completion_rate * 50  # Max 50 points
                
                total_score += years_score + key_fields_score
            
            max_score += 100  # 100 points per statement
        
        return (total_score / max_score * 100) if max_score > 0 else 0
    
    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        """Safely calculate ratio avoiding division by zero"""
        return numerator / denominator if denominator != 0 else 0
    
    def _calculate_cagr(self, series: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate"""
        if len(series) < 2:
            return 0
        
        # Sort by index (oldest first for CAGR calculation)
        sorted_series = series.sort_index()
        start_value = sorted_series.iloc[0]
        end_value = sorted_series.iloc[-1]
        years = len(sorted_series) - 1
        
        if start_value <= 0 or end_value <= 0 or years <= 0:
            return 0
        
        cagr = (end_value / start_value) ** (1 / years) - 1
        return cagr

# Enhanced convenience functions for easy usage
def fetch_comprehensive_data(ticker: str, source: str = "yfinance", 
                           include_peers: bool = False,
                           peer_tickers: Optional[List[str]] = None) -> Dict:
    """
    Quick function to fetch all financial data with enhanced features
    
    Args:
        ticker: Stock ticker symbol
        source: Data source ('yfinance', 'alpha_vantage')
        include_peers: Whether to include peer analysis
        peer_tickers: List of peer tickers for comparison
    
    Returns:
        Comprehensive financial data dictionary
    
    Usage:
        # Basic usage
        data = fetch_comprehensive_data("ADBE")
        
        # With peer analysis
        data = fetch_comprehensive_data("ADBE", include_peers=True, 
                                      peer_tickers=["MSFT", "GOOGL", "AAPL"])
        
        print(f"Revenue: ${data['dcf_summary']['current_financials']['revenue']:,.0f}")
        print(f"FCFF: ${data['dcf_summary']['current_financials']['fcff_nopat_method']:,.0f}")
        print(f"WACC: {data['wacc_components']['wacc']:.1%}")
    """
    fetcher = FinancialDataFetcher()
    return fetcher.fetch_all_financials(
        ticker, source, include_peers, peer_tickers
    )

def quick_dcf_analysis(ticker: str) -> Dict:
    """
    Quick DCF analysis with key metrics for Adobe-style valuation
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Key DCF metrics and assumptions
    
    Usage:
        analysis = quick_dcf_analysis("ADBE")
        print(f"Current FCFF: ${analysis['current_fcff']:,.0f}")
        print(f"WACC: {analysis['wacc']:.1%}")
        print(f"Revenue Growth: {analysis['revenue_growth']:.1%}")
    """
    data = fetch_comprehensive_data(ticker)
    
    # Extract key DCF metrics
    return {
        'ticker': ticker.upper(),
        'current_fcff': data['dcf_summary']['current_financials']['fcff_nopat_method'],
        'latest_revenue': data['dcf_summary']['current_financials']['revenue'],
        'wacc': data['wacc_components']['wacc'],
        'wacc_percentage': data['wacc_components']['wacc_percentage'],
        'revenue_growth': data['dcf_summary']['historical_growth']['revenue_growth_3yr_avg'],
        'fcf_growth': data['dcf_summary']['historical_growth']['fcf_growth_3yr_avg'],
        'current_margins': data['dcf_summary']['current_margins'],
        'market_cap': data['dcf_summary']['market_valuation']['market_capitalization'],
        'current_price': data['dcf_summary']['market_valuation']['current_stock_price'],
        'data_quality': data['data_quality_report']['overall_quality']
    }

def get_adobe_dcf_inputs(ticker: str) -> Dict:
    """
    Get inputs specifically formatted for Adobe DCF model
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Adobe DCF formatted inputs
    
    Usage:
        inputs = get_adobe_dcf_inputs("ADBE")
        print("Ready for Excel DCF model:")
        print(f"Latest FCFF: ${inputs['fcff_latest']:,.0f}")
        print(f"Growth Rate: {inputs['terminal_growth_rate']:.1%}")
    """
    data = fetch_comprehensive_data(ticker)
    
    # Format for Adobe DCF model
    return {
        'company_name': ticker.upper(),
        'valuation_date': data['last_updated'],
        
        # Historical financials (last 5 years)
        'historical_revenue': data['income_statement']['Total Revenue'].head(5).to_dict(),
        'historical_fcff': data['cashflow_statement']['FCFF (NOPAT Method)'].head(5).to_dict(),
        
        # Current year metrics
        'revenue_latest': data['dcf_summary']['current_financials']['revenue'],
        'fcff_latest': data['dcf_summary']['current_financials']['fcff_nopat_method'],
        'ebit_latest': data['dcf_summary']['current_financials']['ebit'],
        
        # Growth assumptions
        'revenue_growth_historical': data['dcf_summary']['historical_growth']['revenue_growth_3yr_avg'],
        'terminal_growth_rate': data['dcf_summary']['dcf_assumptions']['terminal_growth_rate'],
        
        # WACC components
        'wacc': data['wacc_components']['wacc'],
        'cost_of_equity': data['wacc_components']['costs']['cost_of_equity'],
        'cost_of_debt': data['wacc_components']['costs']['cost_of_debt'],
        'tax_rate': data['wacc_components']['inputs']['tax_rate'],
        
        # Market data
        'current_stock_price': data['dcf_summary']['market_valuation']['current_stock_price'],
        'shares_outstanding': data['dcf_summary']['market_valuation']['shares_outstanding'],
        'market_cap': data['dcf_summary']['market_valuation']['market_capitalization'],
        
        # Balance sheet items
        'cash': data['dcf_summary']['current_financials']['cash'],
        'total_debt': data['dcf_summary']['current_financials']['total_debt'],
        
        # Quality metrics
        'data_quality_score': data['data_quality']['data_completeness_score'],
        'years_of_data': data['data_quality']['years_available']
    }

# Make all fetchers and functions available
__all__ = [
    # Main classes
    'FinancialDataFetcher',
    'IncomeFetcher', 
    'BalanceFetcher',
    'CashFlowFetcher',
    'MarketDataFetcher',
    
    # Enhanced convenience functions
    'fetch_comprehensive_data',
    'quick_dcf_analysis', 
    'get_adobe_dcf_inputs',
    
    # Individual fetcher functions
    'fetch_income_data',
    'fetch_balance_data', 
    'fetch_cashflow_data',
    'fetch_market_data'
]

if __name__ == "__main__":
    # Example usage and testing
    print(" Testing Enhanced QuantFlow Financial Data Fetcher")
    
    # Test comprehensive data fetch
    print("\n Fetching comprehensive data for Adobe (ADBE)...")
    data = fetch_comprehensive_data("ADBE")
    
    print(f"\n {data['ticker']} Financial Summary:")
    print(f"Revenue: ${data['dcf_summary']['current_financials']['revenue']:,.0f}")
    print(f"FCFF (NOPAT): ${data['dcf_summary']['current_financials']['fcff_nopat_method']:,.0f}")
    print(f"WACC: {data['wacc_components']['wacc_percentage']:.1f}%")
    print(f"Revenue Growth (3yr avg): {data['dcf_summary']['historical_growth']['revenue_growth_3yr_avg']:.1%}")
    print(f"Data Quality: {data['data_quality_report']['overall_quality']}")
    
    # Test quick analysis
    print(f"\n Quick DCF Analysis:")
    quick_analysis = quick_dcf_analysis("ADBE")
    print(f"Operating Margin: {quick_analysis['current_margins']['operating_margin']:.1%}")
    print(f"FCF Margin: {quick_analysis['current_margins']['fcf_margin']:.1%}")
    print(f"Current Stock Price: ${quick_analysis['current_price']:.2f}")
    
    # Test Adobe DCF inputs
    print(f"\n Adobe DCF Model Inputs:")
    adobe_inputs = get_adobe_dcf_inputs("ADBE")
    print(f"Ready for Excel model with {adobe_inputs['years_of_data']['income']} years of data")
    print(f"Data Quality Score: {adobe_inputs['data_quality_score']:.0f}/100")
    
    print(f"\n CSV files saved to:")
    for statement, path in data['csv_paths'].items():
        print(f"  {statement}: {path}")
    
    print(f"\n QuantFlow Enhanced Fetcher Ready for Adobe DCF Analysis!")