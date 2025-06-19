# quantflow/analyzers/comps.py
"""
QuantFlow Financial Suite - Comparable Companies Analyzer
Comprehensive peer analysis and valuation multiples comparison
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import warnings
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from ..config import get_config

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

@dataclass
class CompanyProfile:
    """Data class for company profile information"""
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: float
    revenue: float
    ebitda: float
    net_income: float
    total_assets: float
    enterprise_value: float
    
    # Valuation multiples
    pe_ratio: float = 0
    ev_revenue: float = 0
    ev_ebitda: float = 0
    price_to_book: float = 0
    price_to_sales: float = 0
    
    # Growth metrics
    revenue_growth: float = 0
    earnings_growth: float = 0
    
    # Profitability metrics
    gross_margin: float = 0
    operating_margin: float = 0
    net_margin: float = 0
    roe: float = 0
    roa: float = 0
    
    # Financial health
    debt_to_equity: float = 0
    current_ratio: float = 0
    interest_coverage: float = 0

class ComparableAnalyzer:
    """
    Comprehensive comparable companies analyzer
    Performs peer analysis, valuation multiples comparison, and benchmarking
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.industry_multiples = {}
        self.peer_data = {}
    
    def analyze_comparables(self, 
                          primary_ticker: str,
                          peer_tickers: List[str],
                          analysis_type: str = "comprehensive") -> Dict:
        """
        Comprehensive comparable companies analysis
        
        Args:
            primary_ticker: Primary company ticker
            peer_tickers: List of peer company tickers
            analysis_type: Type of analysis ('comprehensive', 'valuation', 'financial')
            
        Returns:
            Dictionary with complete comparable analysis
        """
        logger.info(f"Analyzing comparables for {primary_ticker} vs {len(peer_tickers)} peers")
        
        try:
            # Fetch financial data for all companies
            all_companies_data = self._fetch_all_company_data(primary_ticker, peer_tickers)
            
            # Create company profiles
            company_profiles = self._create_company_profiles(all_companies_data)
            
            # Perform various analyses based on type
            analysis_results = {}
            
            if analysis_type in ["comprehensive", "valuation"]:
                analysis_results['valuation_analysis'] = self._perform_valuation_analysis(
                    primary_ticker, company_profiles
                )
                analysis_results['multiples_comparison'] = self._compare_valuation_multiples(
                    primary_ticker, company_profiles
                )
            
            if analysis_type in ["comprehensive", "financial"]:
                analysis_results['financial_comparison'] = self._compare_financial_metrics(
                    primary_ticker, company_profiles
                )
                analysis_results['profitability_analysis'] = self._analyze_profitability(
                    primary_ticker, company_profiles
                )
            
            if analysis_type == "comprehensive":
                analysis_results['growth_comparison'] = self._compare_growth_metrics(
                    primary_ticker, company_profiles
                )
                analysis_results['risk_analysis'] = self._analyze_risk_metrics(
                    primary_ticker, company_profiles
                )
                analysis_results['industry_benchmarks'] = self._calculate_industry_benchmarks(
                    company_profiles
                )
                analysis_results['relative_valuation'] = self._perform_relative_valuation(
                    primary_ticker, company_profiles
                )
            
            # Generate summary and recommendations
            analysis_results['summary'] = self._generate_analysis_summary(
                primary_ticker, company_profiles, analysis_results
            )
            
            # Store peer data for future reference
            self.peer_data[primary_ticker] = {
                'profiles': company_profiles,
                'analysis_date': datetime.now(),
                'peer_tickers': peer_tickers
            }
            
            logger.info(f"Comparable analysis completed for {primary_ticker}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in comparable analysis: {e}")
            raise
    
    def _fetch_all_company_data(self, 
                               primary_ticker: str, 
                               peer_tickers: List[str]) -> Dict:
        """Fetch financial data for all companies"""
        from ..fetchers import fetch_comprehensive_data
        
        all_data = {}
        all_tickers = [primary_ticker] + peer_tickers
        
        for ticker in all_tickers:
            try:
                logger.info(f"Fetching data for {ticker}")
                company_data = fetch_comprehensive_data(ticker)
                all_data[ticker] = company_data
            except Exception as e:
                logger.warning(f"Failed to fetch data for {ticker}: {e}")
                all_data[ticker] = None
        
        return all_data
    
    def _create_company_profiles(self, all_companies_data: Dict) -> Dict[str, CompanyProfile]:
        """Create CompanyProfile objects for all companies"""
        
        profiles = {}
        
        for ticker, data in all_companies_data.items():
            if data is None:
                continue
            
            try:
                profiles[ticker] = self._extract_company_profile(ticker, data)
            except Exception as e:
                logger.warning(f"Could not create profile for {ticker}: {e}")
        
        return profiles
    
    def _extract_company_profile(self, ticker: str, data: Dict) -> CompanyProfile:
        """Extract company profile from financial data"""
        
        # Extract key data sections
        dcf_summary = data.get('dcf_summary', {})
        current_financials = dcf_summary.get('current_financials', {})
        current_margins = dcf_summary.get('current_margins', {})
        historical_growth = dcf_summary.get('historical_growth', {})
        market_valuation = dcf_summary.get('market_valuation', {})
        market_data = data.get('market_data', {})
        
        # Get comprehensive ratios
        comp_ratios = data.get('comprehensive_ratios', pd.DataFrame())
        latest_ratios = comp_ratios.iloc[0] if not comp_ratios.empty else pd.Series()
        
        return CompanyProfile(
            ticker=ticker,
            name=market_data.get('longName', ticker),
            sector=market_data.get('sector', ''),
            industry=market_data.get('industry', ''),
            
            # Financial data
            market_cap=market_valuation.get('market_capitalization', 0),
            revenue=current_financials.get('revenue', 0),
            ebitda=current_financials.get('ebit', 0),  # Approximation
            net_income=current_financials.get('net_income', 0),
            total_assets=current_financials.get('total_assets', 0),
            enterprise_value=market_valuation.get('enterprise_value', 0),
            
            # Valuation multiples
            pe_ratio=market_data.get('pe_ratio', 0),
            ev_revenue=market_data.get('ev_to_revenue', 0),
            ev_ebitda=market_data.get('ev_to_ebitda', 0),
            price_to_book=market_data.get('price_to_book', 0),
            price_to_sales=market_data.get('price_to_sales', 0),
            
            # Growth metrics
            revenue_growth=historical_growth.get('revenue_growth_3yr_avg', 0),
            earnings_growth=historical_growth.get('fcf_growth_3yr_avg', 0),
            
            # Profitability metrics
            gross_margin=current_margins.get('gross_margin', 0),
            operating_margin=current_margins.get('operating_margin', 0),
            net_margin=current_margins.get('net_margin', 0),
            roe=latest_ratios.get('ROE %', 0) / 100 if 'ROE %' in latest_ratios else 0,
            roa=latest_ratios.get('ROA %', 0) / 100 if 'ROA %' in latest_ratios else 0,
            
            # Financial health
            debt_to_equity=latest_ratios.get('Debt-to-Equity', 0),
            current_ratio=latest_ratios.get('Current Ratio', 0),
            interest_coverage=0  # Would need more detailed calculation
        )
    
    def _perform_valuation_analysis(self, 
                                  primary_ticker: str,
                                  company_profiles: Dict[str, CompanyProfile]) -> Dict:
        """Perform comprehensive valuation analysis"""
        
        primary_profile = company_profiles[primary_ticker]
        peer_profiles = {k: v for k, v in company_profiles.items() if k != primary_ticker}
        
        # Calculate peer statistics for each multiple
        valuation_metrics = ['pe_ratio', 'ev_revenue', 'ev_ebitda', 'price_to_book', 'price_to_sales']
        
        valuation_analysis = {
            'primary_company': {
                'ticker': primary_ticker,
                'multiples': {metric: getattr(primary_profile, metric) for metric in valuation_metrics}
            },
            'peer_statistics': {},
            'relative_positioning': {},
            'implied_valuations': {}
        }
        
        # Calculate peer statistics
        for metric in valuation_metrics:
            peer_values = [getattr(profile, metric) for profile in peer_profiles.values() 
                          if getattr(profile, metric) > 0]
            
            if peer_values:
                valuation_analysis['peer_statistics'][metric] = {
                    'mean': np.mean(peer_values),
                    'median': np.median(peer_values),
                    'min': np.min(peer_values),
                    'max': np.max(peer_values),
                    'std': np.std(peer_values),
                    'count': len(peer_values)
                }
                
                # Relative positioning
                primary_value = getattr(primary_profile, metric)
                if primary_value > 0:
                    peer_mean = np.mean(peer_values)
                    valuation_analysis['relative_positioning'][metric] = {
                        'vs_mean': (primary_value - peer_mean) / peer_mean,
                        'percentile': np.percentile(peer_values + [primary_value], 
                                                  [25, 50, 75, 100]).tolist(),
                        'z_score': (primary_value - peer_mean) / np.std(peer_values) if np.std(peer_values) > 0 else 0
                    }
        
        # Calculate implied valuations using peer multiples
        if primary_profile.revenue > 0:
            ev_rev_mean = valuation_analysis['peer_statistics'].get('ev_revenue', {}).get('mean', 0)
            if ev_rev_mean > 0:
                implied_ev = primary_profile.revenue * ev_rev_mean
                valuation_analysis['implied_valuations']['ev_revenue_method'] = implied_ev
        
        if primary_profile.net_income > 0:
            pe_mean = valuation_analysis['peer_statistics'].get('pe_ratio', {}).get('mean', 0)
            if pe_mean > 0:
                implied_market_cap = primary_profile.net_income * pe_mean
                valuation_analysis['implied_valuations']['pe_method'] = implied_market_cap
        
        return valuation_analysis
    
    def _compare_valuation_multiples(self, 
                                   primary_ticker: str,
                                   company_profiles: Dict[str, CompanyProfile]) -> pd.DataFrame:
        """Create comprehensive valuation multiples comparison table"""
        
        multiples_data = []
        
        for ticker, profile in company_profiles.items():
            multiples_data.append({
                'Ticker': ticker,
                'Company': profile.name[:30],  # Truncate long names
                'Market Cap ($M)': profile.market_cap / 1_000_000 if profile.market_cap > 0 else 0,
                'P/E Ratio': profile.pe_ratio,
                'EV/Revenue': profile.ev_revenue,
                'EV/EBITDA': profile.ev_ebitda,
                'P/B Ratio': profile.price_to_book,
                'P/S Ratio': profile.price_to_sales,
                'Is Primary': ticker == primary_ticker
            })
        
        multiples_df = pd.DataFrame(multiples_data)
        
        # Sort by market cap
        multiples_df = multiples_df.sort_values('Market Cap ($M)', ascending=False)
        
        # Calculate peer averages and medians
        peer_multiples = multiples_df[multiples_df['Is Primary'] == False]
        
        if not peer_multiples.empty:
            peer_avg = peer_multiples[['P/E Ratio', 'EV/Revenue', 'EV/EBITDA', 'P/B Ratio', 'P/S Ratio']].mean()
            peer_median = peer_multiples[['P/E Ratio', 'EV/Revenue', 'EV/EBITDA', 'P/B Ratio', 'P/S Ratio']].median()
            
            # Add summary rows
            avg_row = {
                'Ticker': 'PEER_AVG',
                'Company': 'Peer Average',
                'Market Cap ($M)': peer_multiples['Market Cap ($M)'].mean(),
                'P/E Ratio': peer_avg['P/E Ratio'],
                'EV/Revenue': peer_avg['EV/Revenue'],
                'EV/EBITDA': peer_avg['EV/EBITDA'],
                'P/B Ratio': peer_avg['P/B Ratio'],
                'P/S Ratio': peer_avg['P/S Ratio'],
                'Is Primary': False
            }
            
            median_row = {
                'Ticker': 'PEER_MEDIAN',
                'Company': 'Peer Median',
                'Market Cap ($M)': peer_multiples['Market Cap ($M)'].median(),
                'P/E Ratio': peer_median['P/E Ratio'],
                'EV/Revenue': peer_median['EV/Revenue'],
                'EV/EBITDA': peer_median['EV/EBITDA'],
                'P/B Ratio': peer_median['P/B Ratio'],
                'P/S Ratio': peer_median['P/S Ratio'],
                'Is Primary': False
            }
            
            multiples_df = pd.concat([multiples_df, pd.DataFrame([avg_row, median_row])], ignore_index=True)
        
        return multiples_df
    
    def _compare_financial_metrics(self, 
                                 primary_ticker: str,
                                 company_profiles: Dict[str, CompanyProfile]) -> pd.DataFrame:
        """Compare key financial metrics across companies"""
        
        financial_data = []
        
        for ticker, profile in company_profiles.items():
            financial_data.append({
                'Ticker': ticker,
                'Revenue ($M)': profile.revenue / 1_000_000 if profile.revenue > 0 else 0,
                'Net Income ($M)': profile.net_income / 1_000_000 if profile.net_income > 0 else 0,
                'Total Assets ($M)': profile.total_assets / 1_000_000 if profile.total_assets > 0 else 0,
                'Revenue Growth %': profile.revenue_growth * 100,
                'Gross Margin %': profile.gross_margin * 100,
                'Operating Margin %': profile.operating_margin * 100,
                'Net Margin %': profile.net_margin * 100,
                'ROE %': profile.roe * 100,
                'ROA %': profile.roa * 100,
                'Debt/Equity': profile.debt_to_equity,
                'Current Ratio': profile.current_ratio,
                'Is Primary': ticker == primary_ticker
            })
        
        financial_df = pd.DataFrame(financial_data)
        
        # Sort by revenue
        financial_df = financial_df.sort_values('Revenue ($M)', ascending=False)
        
        return financial_df
    
    def _analyze_profitability(self, 
                             primary_ticker: str,
                             company_profiles: Dict[str, CompanyProfile]) -> Dict:
        """Analyze profitability metrics relative to peers"""
        
        primary_profile = company_profiles[primary_ticker]
        peer_profiles = [v for k, v in company_profiles.items() if k != primary_ticker]
        
        profitability_metrics = ['gross_margin', 'operating_margin', 'net_margin', 'roe', 'roa']
        
        profitability_analysis = {
            'primary_metrics': {},
            'peer_benchmarks': {},
            'relative_performance': {}
        }
        
        # Extract primary company metrics
        for metric in profitability_metrics:
            profitability_analysis['primary_metrics'][metric] = getattr(primary_profile, metric)
        
        # Calculate peer benchmarks
        for metric in profitability_metrics:
            peer_values = [getattr(profile, metric) for profile in peer_profiles 
                          if getattr(profile, metric) > 0]
            
            if peer_values:
                profitability_analysis['peer_benchmarks'][metric] = {
                    'mean': np.mean(peer_values),
                    'median': np.median(peer_values),
                    'top_quartile': np.percentile(peer_values, 75),
                    'bottom_quartile': np.percentile(peer_values, 25)
                }
                
                # Relative performance
                primary_value = getattr(primary_profile, metric)
                peer_mean = np.mean(peer_values)
                
                profitability_analysis['relative_performance'][metric] = {
                    'vs_mean': primary_value - peer_mean,
                    'vs_mean_pct': (primary_value - peer_mean) / peer_mean if peer_mean > 0 else 0,
                    'percentile_rank': (np.sum(np.array(peer_values) <= primary_value) / len(peer_values)) * 100
                }
        
        return profitability_analysis
    
    def _compare_growth_metrics(self, 
                              primary_ticker: str,
                              company_profiles: Dict[str, CompanyProfile]) -> Dict:
        """Compare growth metrics across companies"""
        
        primary_profile = company_profiles[primary_ticker]
        peer_profiles = [v for k, v in company_profiles.items() if k != primary_ticker]
        
        growth_metrics = ['revenue_growth', 'earnings_growth']
        
        growth_analysis = {
            'primary_growth': {},
            'peer_growth_stats': {},
            'growth_ranking': {}
        }
        
        # Primary company growth
        for metric in growth_metrics:
            growth_analysis['primary_growth'][metric] = getattr(primary_profile, metric)
        
        # Peer growth statistics
        for metric in growth_metrics:
            peer_values = [getattr(profile, metric) for profile in peer_profiles]
            valid_values = [v for v in peer_values if not np.isnan(v) and v != 0]
            
            if valid_values:
                growth_analysis['peer_growth_stats'][metric] = {
                    'mean': np.mean(valid_values),
                    'median': np.median(valid_values),
                    'std': np.std(valid_values),
                    'min': np.min(valid_values),
                    'max': np.max(valid_values)
                }
                
                # Growth ranking
                primary_value = getattr(primary_profile, metric)
                all_values = valid_values + [primary_value]
                rank = sorted(all_values, reverse=True).index(primary_value) + 1
                
                growth_analysis['growth_ranking'][metric] = {
                    'rank': rank,
                    'total_companies': len(all_values),
                    'percentile': ((len(all_values) - rank) / len(all_values)) * 100
                }
        
        return growth_analysis
    
    def _analyze_risk_metrics(self, 
                            primary_ticker: str,
                            company_profiles: Dict[str, CompanyProfile]) -> Dict:
        """Analyze financial risk metrics"""
        
        primary_profile = company_profiles[primary_ticker]
        peer_profiles = [v for k, v in company_profiles.items() if k != primary_ticker]
        
        risk_metrics = ['debt_to_equity', 'current_ratio']
        
        risk_analysis = {
            'primary_risk_metrics': {},
            'peer_risk_benchmarks': {},
            'risk_assessment': {}
        }
        
        # Primary company risk metrics
        for metric in risk_metrics:
            risk_analysis['primary_risk_metrics'][metric] = getattr(primary_profile, metric)
        
        # Peer risk benchmarks
        for metric in risk_metrics:
            peer_values = [getattr(profile, metric) for profile in peer_profiles 
                          if getattr(profile, metric) > 0]
            
            if peer_values:
                risk_analysis['peer_risk_benchmarks'][metric] = {
                    'mean': np.mean(peer_values),
                    'median': np.median(peer_values),
                    'safe_range': [np.percentile(peer_values, 25), np.percentile(peer_values, 75)]
                }
        
        # Risk assessment
        debt_equity = primary_profile.debt_to_equity
        current_ratio = primary_profile.current_ratio
        
        risk_assessment = "Low Risk"
        if debt_equity > 1.0 or current_ratio < 1.0:
            risk_assessment = "Moderate Risk"
        if debt_equity > 2.0 or current_ratio < 0.8:
            risk_assessment = "High Risk"
        
        risk_analysis['risk_assessment']['overall'] = risk_assessment
        
        return risk_analysis
    
    def _calculate_industry_benchmarks(self, 
                                     company_profiles: Dict[str, CompanyProfile]) -> Dict:
        """Calculate industry-wide benchmarks"""
        
        # Group companies by industry
        industry_groups = {}
        for ticker, profile in company_profiles.items():
            industry = profile.industry or "Unknown"
            if industry not in industry_groups:
                industry_groups[industry] = []
            industry_groups[industry].append(profile)
        
        industry_benchmarks = {}
        
        for industry, profiles in industry_groups.items():
            if len(profiles) < 2:  # Need at least 2 companies for meaningful benchmarks
                continue
            
            # Calculate industry averages
            metrics = ['pe_ratio', 'ev_revenue', 'gross_margin', 'operating_margin', 
                      'net_margin', 'roe', 'revenue_growth']
            
            industry_stats = {}
            for metric in metrics:
                values = [getattr(profile, metric) for profile in profiles 
                         if getattr(profile, metric) > 0]
                
                if values:
                    industry_stats[metric] = {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
            
            industry_benchmarks[industry] = industry_stats
        
        return industry_benchmarks
    
    def _perform_relative_valuation(self, 
                                   primary_ticker: str,
                                   company_profiles: Dict[str, CompanyProfile]) -> Dict:
        """Perform relative valuation using peer multiples"""
        
        primary_profile = company_profiles[primary_ticker]
        peer_profiles = [v for k, v in company_profiles.items() if k != primary_ticker]
        
        relative_valuation = {
            'current_valuation': primary_profile.market_cap,
            'implied_valuations': {},
            'valuation_range': {},
            'recommendation': {}
        }
        
        # EV/Revenue method
        ev_revenue_multiples = [p.ev_revenue for p in peer_profiles if p.ev_revenue > 0]
        if ev_revenue_multiples and primary_profile.revenue > 0:
            mean_multiple = np.mean(ev_revenue_multiples)
            median_multiple = np.median(ev_revenue_multiples)
            
            implied_ev_mean = primary_profile.revenue * mean_multiple
            implied_ev_median = primary_profile.revenue * median_multiple
            
            relative_valuation['implied_valuations']['ev_revenue'] = {
                'mean_method': implied_ev_mean,
                'median_method': implied_ev_median,
                'peer_multiple_mean': mean_multiple,
                'peer_multiple_median': median_multiple
            }
        
        # P/E method
        pe_ratios = [p.pe_ratio for p in peer_profiles if p.pe_ratio > 0]
        if pe_ratios and primary_profile.net_income > 0:
            mean_pe = np.mean(pe_ratios)
            median_pe = np.median(pe_ratios)
            
            implied_mc_mean = primary_profile.net_income * mean_pe
            implied_mc_median = primary_profile.net_income * median_pe
            
            relative_valuation['implied_valuations']['pe_ratio'] = {
                'mean_method': implied_mc_mean,
                'median_method': implied_mc_median,
                'peer_pe_mean': mean_pe,
                'peer_pe_median': median_pe
            }
        
        # Calculate valuation range
        all_implied_values = []
        for method_data in relative_valuation['implied_valuations'].values():
            if isinstance(method_data, dict):
                all_implied_values.extend([
                    method_data.get('mean_method', 0),
                    method_data.get('median_method', 0)
                ])
        
        if all_implied_values:
            valid_values = [v for v in all_implied_values if v > 0]
            if valid_values:
                relative_valuation['valuation_range'] = {
                    'low': np.min(valid_values),
                    'high': np.max(valid_values),
                    'average': np.mean(valid_values),
                    'current_vs_range': self._assess_valuation_vs_range(
                        primary_profile.market_cap, valid_values
                    )
                }
        
        return relative_valuation
    
    def _assess_valuation_vs_range(self, current_value: float, implied_values: List[float]) -> str:
        """Assess current valuation relative to implied range"""
        
        if not implied_values:
            return "Insufficient data"
        
        range_low = np.min(implied_values)
        range_high = np.max(implied_values)
        range_avg = np.mean(implied_values)
        
        if current_value < range_low * 0.9:
            return "Significantly Undervalued"
        elif current_value < range_avg * 0.95:
            return "Potentially Undervalued"
        elif current_value > range_high * 1.1:
            return "Significantly Overvalued"
        elif current_value > range_avg * 1.05:
            return "Potentially Overvalued"
        else:
            return "Fairly Valued"
    
    def _generate_analysis_summary(self, 
                                 primary_ticker: str,
                                 company_profiles: Dict[str, CompanyProfile],
                                 analysis_results: Dict) -> Dict:
        """Generate comprehensive analysis summary"""
        
        primary_profile = company_profiles[primary_ticker]
        
        summary = {
            'company_overview': {
                'ticker': primary_ticker,
                'name': primary_profile.name,
                'sector': primary_profile.sector,
                'industry': primary_profile.industry,
                'market_cap': primary_profile.market_cap
            },
            'peer_group_size': len(company_profiles) - 1,
            'key_findings': [],
            'strengths': [],
            'concerns': [],
            'overall_assessment': ""
        }
        
        # Valuation assessment
        if 'valuation_analysis' in analysis_results:
            val_analysis = analysis_results['valuation_analysis']
            relative_pos = val_analysis.get('relative_positioning', {})
            
            # Check if overvalued/undervalued
            pe_vs_mean = relative_pos.get('pe_ratio', {}).get('vs_mean', 0)
            if pe_vs_mean > 0.2:
                summary['key_findings'].append(f"Trading at {pe_vs_mean:.1%} premium to peer average P/E")
                summary['concerns'].append("Higher valuation multiple than peers")
            elif pe_vs_mean < -0.2:
                summary['key_findings'].append(f"Trading at {abs(pe_vs_mean):.1%} discount to peer average P/E")
                summary['strengths'].append("Attractive valuation relative to peers")
        
        # Profitability assessment
        if 'profitability_analysis' in analysis_results:
            prof_analysis = analysis_results['profitability_analysis']
            relative_perf = prof_analysis.get('relative_performance', {})
            
            # Operating margin comparison
            op_margin_vs_mean = relative_perf.get('operating_margin', {}).get('vs_mean_pct', 0)
            if op_margin_vs_mean > 0.1:
                summary['strengths'].append("Superior operating margins vs peers")
            elif op_margin_vs_mean < -0.1:
                summary['concerns'].append("Below-average operating margins")
        
        # Growth assessment
        if 'growth_comparison' in analysis_results:
            growth_analysis = analysis_results['growth_comparison']
            growth_ranking = growth_analysis.get('growth_ranking', {})
            
            rev_growth_percentile = growth_ranking.get('revenue_growth', {}).get('percentile', 50)
            if rev_growth_percentile > 75:
                summary['strengths'].append("Top-quartile revenue growth")
            elif rev_growth_percentile < 25:
                summary['concerns'].append("Below-average revenue growth")
        
        # Risk assessment
        if 'risk_analysis' in analysis_results:
            risk_analysis = analysis_results['risk_analysis']
            overall_risk = risk_analysis.get('risk_assessment', {}).get('overall', 'Unknown')
            
            if overall_risk == "Low Risk":
                summary['strengths'].append("Strong financial health")
            elif overall_risk == "High Risk":
                summary['concerns'].append("Elevated financial risk")
        
        # Overall assessment
        strength_count = len(summary['strengths'])
        concern_count = len(summary['concerns'])
        
        if strength_count > concern_count:
            summary['overall_assessment'] = "Positive - Outperforming peers in key metrics"
        elif concern_count > strength_count:
            summary['overall_assessment'] = "Cautious - Underperforming peers in several areas"
        else:
            summary['overall_assessment'] = "Neutral - Mixed performance vs peers"
        
        return summary
    
    def create_comparison_dashboard(self, 
                                  primary_ticker: str,
                                  analysis_results: Dict) -> Dict[str, pd.DataFrame]:
        """Create dashboard-ready comparison tables"""
        
        dashboard_tables = {}
        
        # Valuation multiples table
        if 'multiples_comparison' in analysis_results:
            multiples_df = analysis_results['multiples_comparison']
            dashboard_tables['valuation_multiples'] = multiples_df
        
        # Financial metrics table
        if 'financial_comparison' in analysis_results:
            financial_df = analysis_results['financial_comparison']
            dashboard_tables['financial_metrics'] = financial_df
        
        # Summary table
        if 'summary' in analysis_results:
            summary = analysis_results['summary']
            summary_data = {
                'Metric': ['Company', 'Sector', 'Peer Group Size', 'Overall Assessment'],
                'Value': [
                    summary['company_overview']['name'],
                    summary['company_overview']['sector'],
                    summary['peer_group_size'],
                    summary['overall_assessment']
                ]
            }
            dashboard_tables['summary'] = pd.DataFrame(summary_data)
        
        return dashboard_tables
    
    def save_analysis_report(self, 
                           primary_ticker: str,
                           analysis_results: Dict,
                           output_path: Optional[str] = None) -> str:
        """Save comprehensive analysis report to Excel"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{primary_ticker}_Comparable_Analysis_{timestamp}.xlsx"
        
        dashboard_tables = self.create_comparison_dashboard(primary_ticker, analysis_results)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write each table to a separate sheet
            for sheet_name, table in dashboard_tables.items():
                table.to_excel(writer, sheet_name=sheet_name.replace('_', ' ').title(), index=False)
            
            # Write detailed analysis results
            if 'valuation_analysis' in analysis_results:
                val_data = []
                val_analysis = analysis_results['valuation_analysis']
                
                # Peer statistics
                for metric, stats in val_analysis.get('peer_statistics', {}).items():
                    val_data.append({
                        'Metric': metric,
                        'Type': 'Peer Mean',
                        'Value': stats.get('mean', 0)
                    })
                    val_data.append({
                        'Metric': metric,
                        'Type': 'Peer Median', 
                        'Value': stats.get('median', 0)
                    })
                
                if val_data:
                    val_df = pd.DataFrame(val_data)
                    val_df.to_excel(writer, sheet_name='Valuation Analysis', index=False)
        
        logger.info(f"Comparable analysis report saved to: {output_path}")
        return output_path

# Convenience functions
def quick_peer_analysis(primary_ticker: str, peer_tickers: List[str]) -> Dict:
    """
    Quick peer analysis with key metrics
    
    Usage:
        analysis = quick_peer_analysis("ADBE", ["MSFT", "GOOGL", "AAPL"])
        print(analysis['summary']['overall_assessment'])
    """
    analyzer = ComparableAnalyzer()
    return analyzer.analyze_comparables(primary_ticker, peer_tickers, "comprehensive")

def valuation_multiples_comparison(primary_ticker: str, peer_tickers: List[str]) -> pd.DataFrame:
    """
    Get valuation multiples comparison table
    
    Usage:
        multiples = valuation_multiples_comparison("ADBE", ["MSFT", "GOOGL"])
        print(multiples)
    """
    analyzer = ComparableAnalyzer()
    analysis = analyzer.analyze_comparables(primary_ticker, peer_tickers, "valuation")
    return analysis.get('multiples_comparison', pd.DataFrame())

def financial_metrics_comparison(primary_ticker: str, peer_tickers: List[str]) -> pd.DataFrame:
    """
    Get financial metrics comparison table
    
    Usage:
        metrics = financial_metrics_comparison("ADBE", ["MSFT", "GOOGL"])
        print(metrics)
    """
    analyzer = ComparableAnalyzer()
    analysis = analyzer.analyze_comparables(primary_ticker, peer_tickers, "financial")
    return analysis.get('financial_comparison', pd.DataFrame())

def get_industry_positioning(primary_ticker: str, peer_tickers: List[str]) -> Dict:
    """
    Get industry positioning analysis
    
    Usage:
        positioning = get_industry_positioning("ADBE", ["MSFT", "GOOGL"])
        print(f"Overall assessment: {positioning['overall_assessment']}")
    """
    analyzer = ComparableAnalyzer()
    analysis = analyzer.analyze_comparables(primary_ticker, peer_tickers, "comprehensive")
    return analysis.get('summary', {})

if __name__ == "__main__":
    # Example usage
    print("📊 Testing Comparable Companies Analyzer")
    
    # Test with Adobe vs tech peers
    try:
        primary = "ADBE"
        peers = ["MSFT", "GOOGL", "AAPL", "ORCL"]
        
        print(f"\n🔍 Analyzing {primary} vs peers: {', '.join(peers)}")
        
        # Quick analysis
        analysis = quick_peer_analysis(primary, peers)
        
        print(f"\n✅ Analysis Complete!")
        print(f"Peer Group Size: {analysis['summary']['peer_group_size']}")
        print(f"Overall Assessment: {analysis['summary']['overall_assessment']}")
        
        # Show key strengths and concerns
        if analysis['summary']['strengths']:
            print(f"\n💪 Key Strengths:")
            for strength in analysis['summary']['strengths']:
                print(f"  • {strength}")
        
        if analysis['summary']['concerns']:
            print(f"\n⚠️  Key Concerns:")
            for concern in analysis['summary']['concerns']:
                print(f"  • {concern}")
        
        # Show valuation multiples
        print(f"\n📈 Valuation Multiples Comparison:")
        multiples = valuation_multiples_comparison(primary, peers)
        if not multiples.empty:
            # Show just the key columns for primary company and averages
            key_cols = ['Ticker', 'P/E Ratio', 'EV/Revenue', 'EV/EBITDA']
            primary_row = multiples[multiples['Ticker'] == primary][key_cols]
            avg_row = multiples[multiples['Ticker'] == 'PEER_AVG'][key_cols]
            
            if not primary_row.empty and not avg_row.empty:
                print(f"  {primary}: P/E={primary_row['P/E Ratio'].iloc[0]:.1f}, EV/Rev={primary_row['EV/Revenue'].iloc[0]:.1f}")
                print(f"  Peer Avg: P/E={avg_row['P/E Ratio'].iloc[0]:.1f}, EV/Rev={avg_row['EV/Revenue'].iloc[0]:.1f}")
        
        # Show relative valuation
        if 'relative_valuation' in analysis:
            rel_val = analysis['relative_valuation']
            val_range = rel_val.get('valuation_range', {})
            if val_range:
                assessment = val_range.get('current_vs_range', 'Unknown')
                print(f"\n🎯 Relative Valuation: {assessment}")
        
        # Create analyzer for saving report
        analyzer = ComparableAnalyzer()
        report_path = analyzer.save_analysis_report(primary, analysis)
        print(f"\n💾 Full report saved to: {report_path}")
        
    except Exception as e:
        print(f"❌ Error in comparable analysis: {e}")
        import traceback
        traceback.print_exc()