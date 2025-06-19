# quantflow/__init__.py
"""
QuantFlow Financial Suite
Professional DCF valuation and financial analysis toolkit

A comprehensive financial analysis suite implementing Adobe-style DCF methodology
with advanced peer comparison, scenario analysis, and professional reporting.
"""

__version__ = "1.0.0"
__author__ = "QuantFlow Development Team"

# Core modules
from . import fetchers
from . import models  
from . import analyzers
from . import utils
from .config import get_config

# Main convenience functions for quick usage
from .fetchers import (
    fetch_comprehensive_data,
    quick_dcf_analysis,
    get_adobe_dcf_inputs
)

from .models import (
    quick_dcf_valuation,
    create_scenario_analysis
)

from .analyzers import (
    quick_peer_analysis,
    valuation_multiples_comparison
)

from .utils import (
    export_financial_data,
    create_dcf_report,
    batch_export_companies
)

# Main classes for advanced usage
from .fetchers import FinancialDataFetcher
from .models import DCFCalculator, DCFAssumptions, DCFResults
from .analyzers import ComparableAnalyzer, CompanyProfile
from .utils import DataExporter, DataLoader, ReportGenerator

def analyze_company(ticker: str, 
                   peer_tickers: list = None,
                   include_dcf: bool = True,
                   include_peers: bool = True,
                   export_format: str = 'excel') -> dict:
    """
    One-stop function for comprehensive company analysis
    
    Args:
        ticker: Stock ticker symbol
        peer_tickers: List of peer company tickers
        include_dcf: Whether to perform DCF valuation
        include_peers: Whether to perform peer analysis
        export_format: Export format ('excel', 'html', 'csv')
    
    Returns:
        Dictionary with all analysis results and file paths
    
    Example:
        >>> import quantflow as qf
        >>> results = qf.analyze_company("ADBE", ["MSFT", "ORCL"], True, True, "excel")
        >>> print(f"DCF Value: ${results['dcf_results'].value_per_share:.2f}")
        >>> print(f"Report saved to: {results['report_path']}")
    """
    
    print(f" Starting comprehensive analysis for {ticker}")
    
    results = {
        'ticker': ticker,
        'analysis_timestamp': None,
        'financial_data': None,
        'dcf_results': None,
        'peer_analysis': None,
        'export_paths': [],
        'summary': {}
    }
    
    try:
        # 1. Fetch comprehensive financial data
        print(f" Fetching financial data...")
        financial_data = fetch_comprehensive_data(ticker)
        results['financial_data'] = financial_data
        results['analysis_timestamp'] = financial_data.get('last_updated')
        
        # 2. DCF Analysis
        if include_dcf:
            print(f" Performing DCF valuation...")
            dcf_results = quick_dcf_valuation(ticker)
            results['dcf_results'] = dcf_results
            
            # Create DCF report
            dcf_report_path = create_dcf_report(ticker, dcf_results, financial_data, export_format)
            results['export_paths'].append(dcf_report_path)
            
            results['summary']['dcf'] = {
                'fair_value': dcf_results.value_per_share,
                'current_price': dcf_results.current_price,
                'implied_return': dcf_results.implied_return,
                'recommendation': 'BUY' if dcf_results.implied_return > 0.1 else 'HOLD' if dcf_results.implied_return > -0.05 else 'SELL'
            }
        
        # 3. Peer Analysis
        if include_peers and peer_tickers:
            print(f"🔍 Performing peer analysis...")
            peer_analysis = quick_peer_analysis(ticker, peer_tickers)
            results['peer_analysis'] = peer_analysis
            
            results['summary']['peers'] = {
                'peer_count': len(peer_tickers),
                'overall_assessment': peer_analysis['summary']['overall_assessment'],
                'key_strengths': peer_analysis['summary']['strengths'],
                'key_concerns': peer_analysis['summary']['concerns']
            }
        
        # 4. Export financial data
        print(f" Exporting financial data...")
        financial_export_path = export_financial_data(financial_data, ticker, export_format)
        results['export_paths'].append(financial_export_path)
        
        # 5. Create summary
        results['summary']['company'] = {
            'name': financial_data.get('dcf_summary', {}).get('market_valuation', {}).get('company_name', ticker),
            'sector': financial_data.get('market_data', {}).get('sector', 'Unknown'),
            'market_cap': financial_data.get('dcf_summary', {}).get('market_valuation', {}).get('market_capitalization', 0),
            'data_quality': financial_data.get('data_quality_report', {}).get('overall_quality', 'Unknown')
        }
        
        print(f" Analysis complete! Reports saved:")
        for path in results['export_paths']:
            print(f"   {path}")
        
        return results
        
    except Exception as e:
        print(f" Error in comprehensive analysis: {e}")
        results['error'] = str(e)
        return results

def quick_valuation(ticker: str) -> dict:
    """
    Quick valuation summary for a stock
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with key valuation metrics
    
    Example:
        >>> import quantflow as qf
        >>> valuation = qf.quick_valuation("ADBE")
        >>> print(f"Fair Value: ${valuation['fair_value']:.2f}")
    """
    
    try:
        # Get DCF valuation
        dcf_results = quick_dcf_valuation(ticker)
        
        # Get basic financial data
        financial_data = fetch_comprehensive_data(ticker)
        dcf_summary = financial_data.get('dcf_summary', {})
        
        return {
            'ticker': ticker,
            'fair_value': dcf_results.value_per_share,
            'current_price': dcf_results.current_price,
            'implied_return': dcf_results.implied_return,
            'enterprise_value': dcf_results.enterprise_value,
            'market_cap': dcf_summary.get('market_valuation', {}).get('market_capitalization', 0),
            'revenue': dcf_summary.get('current_financials', {}).get('revenue', 0),
            'fcf': dcf_summary.get('current_financials', {}).get('free_cash_flow', 0),
            'recommendation': 'BUY' if dcf_results.implied_return > 0.1 else 'HOLD' if dcf_results.implied_return > -0.05 else 'SELL'
        }
        
    except Exception as e:
        return {
            'ticker': ticker,
            'error': str(e),
            'fair_value': 0,
            'current_price': 0,
            'implied_return': 0
        }

def screen_stocks(tickers: list, 
                 min_market_cap: float = 1e9,
                 min_implied_return: float = 0.05) -> list:
    """
    Screen stocks based on DCF valuation criteria
    
    Args:
        tickers: List of stock tickers to screen
        min_market_cap: Minimum market cap threshold
        min_implied_return: Minimum implied return threshold
    
    Returns:
        List of stocks meeting criteria, sorted by implied return
    
    Example:
        >>> import quantflow as qf
        >>> candidates = qf.screen_stocks(["ADBE", "MSFT", "ORCL"], 1e9, 0.05)
        >>> for stock in candidates:
        ...     print(f"{stock['ticker']}: {stock['implied_return']:.1%}")
    """
    
    print(f" Screening {len(tickers)} stocks...")
    
    results = []
    
    for ticker in tickers:
        try:
            print(f"  🔍 Analyzing {ticker}...")
            valuation = quick_valuation(ticker)
            
            # Apply screening criteria
            if (valuation.get('market_cap', 0) >= min_market_cap and 
                valuation.get('implied_return', -1) >= min_implied_return and
                'error' not in valuation):
                
                results.append(valuation)
                
        except Exception as e:
            print(f"    ⚠️ Error analyzing {ticker}: {e}")
    
    # Sort by implied return (descending)
    results.sort(key=lambda x: x.get('implied_return', 0), reverse=True)
    
    print(f" Screening complete. {len(results)} stocks meet criteria.")
    return results

# Expose main functionality
__all__ = [
    # Main analysis functions
    'analyze_company',
    'quick_valuation', 
    'screen_stocks',
    
    # Individual module functions
    'fetch_comprehensive_data',
    'quick_dcf_analysis',
    'get_adobe_dcf_inputs',
    'quick_dcf_valuation',
    'create_scenario_analysis',
    'quick_peer_analysis',
    'valuation_multiples_comparison',
    'export_financial_data',
    'create_dcf_report',
    'batch_export_companies',
    
    # Main classes
    'FinancialDataFetcher',
    'DCFCalculator',
    'DCFAssumptions', 
    'DCFResults',
    'ComparableAnalyzer',
    'CompanyProfile',
    'DataExporter',
    'DataLoader',
    'ReportGenerator',
    
    # Configuration
    'get_config',
    
    # Modules
    'fetchers',
    'models',
    'analyzers', 
    'utils'
]

# Package metadata
__package_info__ = {
    'name': 'QuantFlow Financial Suite',
    'version': __version__,
    'description': 'Professional DCF valuation and financial analysis toolkit',
    'author': __author__,
    'features': [
        'Adobe-style DCF valuation with NOPAT methodology',
        'Comprehensive peer analysis and benchmarking',
        'Professional Excel and HTML report generation',
        'Multi-source financial data fetching with fallbacks',
        'Sensitivity analysis and scenario modeling',
        'Industry-specific assumptions and configurations',
        'Batch processing and screening capabilities'
    ],
    'supported_data_sources': ['Yahoo Finance', 'Alpha Vantage'],
    'export_formats': ['Excel', 'CSV', 'JSON', 'HTML'],
    'requirements': [
        'pandas>=1.5.0',
        'numpy>=1.21.0', 
        'yfinance>=0.2.0',
        'openpyxl>=3.0.0',
        'requests>=2.28.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'pyyaml>=6.0'
    ]
}

if __name__ == "__main__":
    print(f" {__package_info__['name']} v{__version__}")
    print(f" {__package_info__['description']}")
    print(f"\n✨ Key Features:")
    for feature in __package_info__['features']:
        print(f"  • {feature}")
    
    print(f"\n🧪 Quick Test:")
    try:
        # Test with Adobe
        result = quick_valuation("ADBE")
        if 'error' not in result:
            print(f"   ADBE Fair Value: ${result['fair_value']:.2f}")
            print(f"   Current Price: ${result['current_price']:.2f}")
            print(f"   Implied Return: {result['implied_return']:.1%}")
            print(f"   Recommendation: {result['recommendation']}")
        else:
            print(f"   Test failed: {result['error']}")
    except Exception as e:
        print(f"   Test error: {e}")
    
    print(f"\n QuantFlow is ready for analysis!")