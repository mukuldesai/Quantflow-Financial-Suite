# quantflow/models/dcf_calculator.py
"""
QuantFlow Financial Suite - DCF Calculator
Implements Adobe-style DCF valuation with NOPAT methodology and comprehensive sensitivity analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from ..config import get_config

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

@dataclass
class DCFAssumptions:
    """Data class for DCF model assumptions"""
    projection_years: int = 5
    terminal_growth_rate: float = 0.04
    tax_rate: float = 0.21
    wacc: float = 0.117
    
    # Growth assumptions by year
    revenue_growth_rates: List[float] = None
    
    # Margin assumptions
    gross_margin: float = 0.875
    rd_margin: float = 0.185
    sm_margin: float = 0.275
    ga_margin: float = 0.075
    
    # Cash flow assumptions
    da_margin: float = 0.044
    capex_margin: float = 0.023
    nwc_growth_factor: float = 0.10
    
    def __post_init__(self):
        if self.revenue_growth_rates is None:
            # Default Adobe-style growth rates: declining over time
            self.revenue_growth_rates = [0.12, 0.105, 0.09, 0.075, 0.06]

@dataclass
class DCFResults:
    """Data class for DCF valuation results"""
    enterprise_value: float
    equity_value: float
    value_per_share: float
    current_price: float
    implied_return: float
    
    # Detailed breakdown
    pv_of_fcff: float
    terminal_value: float
    pv_of_terminal_value: float
    net_cash: float
    shares_outstanding: float
    
    # Sensitivity analysis
    sensitivity_table: pd.DataFrame = None
    
    # Component analysis
    fcff_projections: pd.DataFrame = None
    assumptions_used: DCFAssumptions = None

class DCFCalculator:
    """
    Adobe-style DCF Calculator with NOPAT methodology
    Implements comprehensive valuation with sensitivity analysis and scenario modeling
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.results_history = []
    
    def calculate_dcf_valuation(self, 
                              financial_data: Dict,
                              assumptions: Optional[DCFAssumptions] = None,
                              include_sensitivity: bool = True) -> DCFResults:
        """
        Calculate DCF valuation using Adobe methodology
        
        Args:
            financial_data: Comprehensive financial data from fetchers
            assumptions: DCF assumptions (uses defaults if None)
            include_sensitivity: Whether to include sensitivity analysis
            
        Returns:
            DCFResults with complete valuation analysis
        """
        logger.info(f"Calculating DCF valuation for {financial_data.get('ticker', 'Unknown')}")
        
        # Use provided assumptions or create defaults
        if assumptions is None:
            assumptions = self._create_default_assumptions(financial_data)
        
        try:
            # Step 1: Project financial statements
            projections = self._project_financial_statements(financial_data, assumptions)
            
            # Step 2: Calculate FCFF projections
            fcff_projections = self._calculate_fcff_projections(projections, assumptions)
            
            # Step 3: Calculate terminal value
            terminal_value = self._calculate_terminal_value(fcff_projections, assumptions)
            
            # Step 4: Discount to present value
            valuation_components = self._calculate_present_values(
                fcff_projections, terminal_value, assumptions
            )
            
            # Step 5: Calculate equity value
            equity_value = self._calculate_equity_value(
                valuation_components, financial_data
            )
            
            # Step 6: Create results object
            results = self._create_dcf_results(
                valuation_components, equity_value, financial_data, 
                assumptions, fcff_projections
            )
            
            # Step 7: Sensitivity analysis
            if include_sensitivity:
                results.sensitivity_table = self._perform_sensitivity_analysis(
                    financial_data, assumptions
                )
            
            # Store results for comparison
            self.results_history.append(results)
            
            logger.info(f"DCF calculation completed. Value per share: ${results.value_per_share:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in DCF calculation: {e}")
            raise
    
    def _create_default_assumptions(self, financial_data: Dict) -> DCFAssumptions:
        """Create default assumptions based on historical data"""
        
        # Extract historical growth rates
        dcf_summary = financial_data.get('dcf_summary', {})
        historical_growth = dcf_summary.get('historical_growth', {})
        
        # Get WACC from financial data
        wacc_components = financial_data.get('wacc_components', {})
        wacc = wacc_components.get('wacc', 0.117)
        
        # Get margins from current financials
        current_margins = dcf_summary.get('current_margins', {})
        
        # Create revenue growth projection (declining from historical)
        hist_revenue_growth = historical_growth.get('revenue_growth_3yr_avg', 0.10)
        revenue_growth_rates = [
            max(hist_revenue_growth, 0.12),  # Year 1: historical or 12%, whichever is higher
            hist_revenue_growth * 0.9,       # Year 2: 90% of historical
            hist_revenue_growth * 0.8,       # Year 3: 80% of historical
            hist_revenue_growth * 0.7,       # Year 4: 70% of historical
            hist_revenue_growth * 0.6        # Year 5: 60% of historical
        ]
        
        # Ensure no negative growth rates
        revenue_growth_rates = [max(rate, 0.03) for rate in revenue_growth_rates]
        
        return DCFAssumptions(
            wacc=wacc,
            revenue_growth_rates=revenue_growth_rates,
            gross_margin=current_margins.get('gross_margin', 0.875),
            tax_rate=wacc_components.get('inputs', {}).get('tax_rate', 0.21)
        )
    
    def _project_financial_statements(self, 
                                    financial_data: Dict, 
                                    assumptions: DCFAssumptions) -> pd.DataFrame:
        """Project financial statements for DCF calculation"""
        
        # Get base year data
        dcf_summary = financial_data.get('dcf_summary', {})
        current_financials = dcf_summary.get('current_financials', {})
        
        base_revenue = current_financials.get('revenue', 0)
        base_ebit = current_financials.get('ebit', 0)
        
        if base_revenue == 0:
            raise ValueError("No base revenue data available for projections")
        
        # Create projection DataFrame
        years = [f"Year {i+1}" for i in range(assumptions.projection_years)]
        projections = pd.DataFrame(index=years)
        
        # Project revenues
        projections['Revenue'] = 0
        current_revenue = base_revenue
        
        for i, growth_rate in enumerate(assumptions.revenue_growth_rates):
            projected_revenue = current_revenue * (1 + growth_rate)
            projections.loc[f"Year {i+1}", 'Revenue'] = projected_revenue
            current_revenue = projected_revenue
        
        # Project cost structure
        projections['Cost of Revenue'] = projections['Revenue'] * (1 - assumptions.gross_margin)
        projections['Gross Profit'] = projections['Revenue'] - projections['Cost of Revenue']
        
        # Operating expenses
        projections['R&D'] = projections['Revenue'] * assumptions.rd_margin
        projections['S&M'] = projections['Revenue'] * assumptions.sm_margin
        projections['G&A'] = projections['Revenue'] * assumptions.ga_margin
        
        # Calculate EBIT
        total_opex = projections['R&D'] + projections['S&M'] + projections['G&A']
        projections['EBIT'] = projections['Gross Profit'] - total_opex
        
        # Tax calculation
        projections['Tax on EBIT'] = projections['EBIT'] * assumptions.tax_rate
        projections['NOPAT'] = projections['EBIT'] - projections['Tax on EBIT']
        
        # Cash flow components
        projections['D&A'] = projections['Revenue'] * assumptions.da_margin
        projections['CapEx'] = projections['Revenue'] * assumptions.capex_margin
        
        # Working capital changes (based on revenue growth)
        revenue_growth = projections['Revenue'].pct_change().fillna(assumptions.revenue_growth_rates[0])
        projections['Delta NWC'] = revenue_growth * assumptions.nwc_growth_factor * projections['Revenue']
        
        return projections
    
    def _calculate_fcff_projections(self, 
                                  projections: pd.DataFrame, 
                                  assumptions: DCFAssumptions) -> pd.DataFrame:
        """Calculate Free Cash Flow to Firm projections using NOPAT method"""
        
        fcff_df = projections.copy()
        
        # FCFF = NOPAT + D&A - CapEx - Δ NWC
        fcff_df['FCFF'] = (
            fcff_df['NOPAT'] + 
            fcff_df['D&A'] - 
            fcff_df['CapEx'] - 
            fcff_df['Delta NWC']
        )
        
        # Calculate discount factors
        fcff_df['Discount Factor'] = 0
        for i in range(len(fcff_df)):
            year = i + 1
            discount_factor = 1 / ((1 + assumptions.wacc) ** year)
            fcff_df.iloc[i, fcff_df.columns.get_loc('Discount Factor')] = discount_factor
        
        # Present value of FCFF
        fcff_df['PV of FCFF'] = fcff_df['FCFF'] * fcff_df['Discount Factor']
        
        return fcff_df
    
    def _calculate_terminal_value(self, 
                                fcff_projections: pd.DataFrame, 
                                assumptions: DCFAssumptions) -> Dict[str, float]:
        """Calculate terminal value using Gordon Growth Model"""
        
        # Terminal year FCFF (Year 5 FCFF grown by terminal growth rate)
        final_year_fcff = fcff_projections['FCFF'].iloc[-1]
        terminal_fcff = final_year_fcff * (1 + assumptions.terminal_growth_rate)
        
        # Terminal value = Terminal FCFF / (WACC - g)
        terminal_value = terminal_fcff / (assumptions.wacc - assumptions.terminal_growth_rate)
        
        # Present value of terminal value
        terminal_discount_factor = 1 / ((1 + assumptions.wacc) ** assumptions.projection_years)
        pv_terminal_value = terminal_value * terminal_discount_factor
        
        return {
            'terminal_fcff': terminal_fcff,
            'terminal_value': terminal_value,
            'terminal_discount_factor': terminal_discount_factor,
            'pv_terminal_value': pv_terminal_value
        }
    
    def _calculate_present_values(self, 
                                fcff_projections: pd.DataFrame,
                                terminal_value: Dict[str, float],
                                assumptions: DCFAssumptions) -> Dict[str, float]:
        """Calculate present values and enterprise value"""
        
        # Sum of PV of projected FCFFs
        pv_of_fcff = fcff_projections['PV of FCFF'].sum()
        
        # PV of terminal value
        pv_terminal_value = terminal_value['pv_terminal_value']
        
        # Enterprise Value = PV of FCFFs + PV of Terminal Value
        enterprise_value = pv_of_fcff + pv_terminal_value
        
        return {
            'pv_of_fcff': pv_of_fcff,
            'pv_terminal_value': pv_terminal_value,
            'enterprise_value': enterprise_value,
            'terminal_value': terminal_value['terminal_value']
        }
    
    def _calculate_equity_value(self, 
                              valuation_components: Dict[str, float],
                              financial_data: Dict) -> Dict[str, float]:
        """Calculate equity value and value per share"""
        
        enterprise_value = valuation_components['enterprise_value']
        
        # Get net cash (cash - debt)
        dcf_summary = financial_data.get('dcf_summary', {})
        current_financials = dcf_summary.get('current_financials', {})
        
        cash = current_financials.get('cash', 0)
        total_debt = current_financials.get('total_debt', 0)
        net_cash = cash - total_debt
        
        # Equity Value = Enterprise Value + Net Cash
        equity_value = enterprise_value + net_cash
        
        # Get shares outstanding
        market_valuation = dcf_summary.get('market_valuation', {})
        shares_outstanding = market_valuation.get('shares_outstanding', 0)
        
        if shares_outstanding == 0:
            # Try to get from market data
            market_data = financial_data.get('market_data', {})
            shares_outstanding = market_data.get('shares_outstanding', 0)
        
        # Value per share
        value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0
        
        # Current price for comparison
        current_price = market_valuation.get('current_stock_price', 0)
        if current_price == 0:
            market_data = financial_data.get('market_data', {})
            current_price = market_data.get('current_price', 0)
        
        # Implied return
        implied_return = (value_per_share - current_price) / current_price if current_price > 0 else 0
        
        return {
            'equity_value': equity_value,
            'value_per_share': value_per_share,
            'current_price': current_price,
            'implied_return': implied_return,
            'net_cash': net_cash,
            'shares_outstanding': shares_outstanding
        }
    
    def _create_dcf_results(self, 
                          valuation_components: Dict[str, float],
                          equity_value: Dict[str, float],
                          financial_data: Dict,
                          assumptions: DCFAssumptions,
                          fcff_projections: pd.DataFrame) -> DCFResults:
        """Create DCFResults object with all components"""
        
        return DCFResults(
            enterprise_value=valuation_components['enterprise_value'],
            equity_value=equity_value['equity_value'],
            value_per_share=equity_value['value_per_share'],
            current_price=equity_value['current_price'],
            implied_return=equity_value['implied_return'],
            
            pv_of_fcff=valuation_components['pv_of_fcff'],
            terminal_value=valuation_components['terminal_value'],
            pv_of_terminal_value=valuation_components['pv_terminal_value'],
            net_cash=equity_value['net_cash'],
            shares_outstanding=equity_value['shares_outstanding'],
            
            fcff_projections=fcff_projections,
            assumptions_used=assumptions
        )
    
    def _perform_sensitivity_analysis(self, 
                                    financial_data: Dict,
                                    base_assumptions: DCFAssumptions) -> pd.DataFrame:
        """Perform sensitivity analysis on WACC and terminal growth rate"""
        
        # Sensitivity ranges
        wacc_range = np.arange(base_assumptions.wacc - 0.02, 
                              base_assumptions.wacc + 0.025, 0.005)
        terminal_growth_range = np.arange(base_assumptions.terminal_growth_rate - 0.015,
                                        base_assumptions.terminal_growth_rate + 0.015, 0.005)
        
        # Create sensitivity table
        sensitivity_results = []
        
        for wacc in wacc_range:
            row_results = []
            for terminal_growth in terminal_growth_range:
                # Create modified assumptions
                modified_assumptions = DCFAssumptions(
                    projection_years=base_assumptions.projection_years,
                    terminal_growth_rate=terminal_growth,
                    tax_rate=base_assumptions.tax_rate,
                    wacc=wacc,
                    revenue_growth_rates=base_assumptions.revenue_growth_rates,
                    gross_margin=base_assumptions.gross_margin,
                    rd_margin=base_assumptions.rd_margin,
                    sm_margin=base_assumptions.sm_margin,
                    ga_margin=base_assumptions.ga_margin,
                    da_margin=base_assumptions.da_margin,
                    capex_margin=base_assumptions.capex_margin,
                    nwc_growth_factor=base_assumptions.nwc_growth_factor
                )
                
                # Calculate DCF with modified assumptions
                try:
                    temp_results = self.calculate_dcf_valuation(
                        financial_data, modified_assumptions, include_sensitivity=False
                    )
                    row_results.append(temp_results.value_per_share)
                except:
                    row_results.append(0)
            
            sensitivity_results.append(row_results)
        
        # Create DataFrame
        sensitivity_df = pd.DataFrame(
            sensitivity_results,
            index=[f"{wacc:.1%}" for wacc in wacc_range],
            columns=[f"{tg:.1%}" for tg in terminal_growth_range]
        )
        
        return sensitivity_df
    
    def create_dcf_summary_table(self, results: DCFResults) -> pd.DataFrame:
        """Create Adobe-style DCF summary table"""
        
        summary_data = {
            'Metric': [
                'Sum of PV of FCFFs (Years 1-5)',
                'PV of Terminal Value',
                'Enterprise Value',
                'Add: Net Cash',
                'Equity Value',
                'Shares Outstanding (M)',
                'Value per Share',
                'Current Stock Price',
                'Implied Return'
            ],
            'Value': [
                f"${results.pv_of_fcff:,.0f}",
                f"${results.pv_of_terminal_value:,.0f}",
                f"${results.enterprise_value:,.0f}",
                f"${results.net_cash:,.0f}",
                f"${results.equity_value:,.0f}",
                f"{results.shares_outstanding / 1_000_000:.0f}",
                f"${results.value_per_share:.2f}",
                f"${results.current_price:.2f}",
                f"{results.implied_return:.1%}"
            ]
        }
        
        return pd.DataFrame(summary_data)
    
    def create_fcff_breakdown_table(self, results: DCFResults) -> pd.DataFrame:
        """Create detailed FCFF breakdown table"""
        
        if results.fcff_projections is None:
            return pd.DataFrame()
        
        # Format for display
        fcff_display = results.fcff_projections.copy()
        
        # Convert to millions and format
        currency_columns = ['Revenue', 'NOPAT', 'D&A', 'CapEx', 'Delta NWC', 'FCFF', 'PV of FCFF']
        for col in currency_columns:
            if col in fcff_display.columns:
                fcff_display[col] = fcff_display[col].apply(lambda x: f"${x/1_000_000:.0f}M")
        
        # Format percentages
        fcff_display['Discount Factor'] = fcff_display['Discount Factor'].apply(lambda x: f"{x:.3f}")
        
        return fcff_display
    
    def compare_scenarios(self, 
                         financial_data: Dict,
                         scenarios: Dict[str, DCFAssumptions]) -> pd.DataFrame:
        """Compare multiple DCF scenarios"""
        
        scenario_results = {}
        
        for scenario_name, assumptions in scenarios.items():
            try:
                results = self.calculate_dcf_valuation(
                    financial_data, assumptions, include_sensitivity=False
                )
                scenario_results[scenario_name] = {
                    'Value per Share': results.value_per_share,
                    'Enterprise Value': results.enterprise_value,
                    'Implied Return': results.implied_return,
                    'Terminal Growth': assumptions.terminal_growth_rate,
                    'WACC': assumptions.wacc,
                    'Avg Revenue Growth': np.mean(assumptions.revenue_growth_rates)
                }
            except Exception as e:
                logger.warning(f"Failed to calculate scenario {scenario_name}: {e}")
                scenario_results[scenario_name] = {
                    'Value per Share': 0,
                    'Enterprise Value': 0,
                    'Implied Return': 0,
                    'Terminal Growth': 0,
                    'WACC': 0,
                    'Avg Revenue Growth': 0
                }
        
        return pd.DataFrame(scenario_results).T
    
    def save_dcf_analysis(self, 
                         results: DCFResults,
                         ticker: str,
                         output_path: Optional[str] = None) -> str:
        """Save complete DCF analysis to Excel file"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{ticker}_DCF_Analysis_{timestamp}.xlsx"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_table = self.create_dcf_summary_table(results)
            summary_table.to_excel(writer, sheet_name='DCF Summary', index=False)
            
            # FCFF projections
            if results.fcff_projections is not None:
                results.fcff_projections.to_excel(writer, sheet_name='FCFF Projections')
            
            # Sensitivity analysis
            if results.sensitivity_table is not None:
                results.sensitivity_table.to_excel(writer, sheet_name='Sensitivity Analysis')
            
            # Assumptions
            assumptions_data = {
                'Parameter': [
                    'Projection Years',
                    'Terminal Growth Rate',
                    'Tax Rate', 
                    'WACC',
                    'Gross Margin',
                    'R&D Margin',
                    'S&M Margin',
                    'G&A Margin',
                    'D&A Margin',
                    'CapEx Margin',
                    'NWC Growth Factor'
                ],
                'Value': [
                    results.assumptions_used.projection_years,
                    f"{results.assumptions_used.terminal_growth_rate:.1%}",
                    f"{results.assumptions_used.tax_rate:.1%}",
                    f"{results.assumptions_used.wacc:.1%}",
                    f"{results.assumptions_used.gross_margin:.1%}",
                    f"{results.assumptions_used.rd_margin:.1%}",
                    f"{results.assumptions_used.sm_margin:.1%}",
                    f"{results.assumptions_used.ga_margin:.1%}",
                    f"{results.assumptions_used.da_margin:.1%}",
                    f"{results.assumptions_used.capex_margin:.1%}",
                    f"{results.assumptions_used.nwc_growth_factor:.1%}"
                ]
            }
            assumptions_df = pd.DataFrame(assumptions_data)
            assumptions_df.to_excel(writer, sheet_name='Assumptions', index=False)
        
        logger.info(f"DCF analysis saved to: {output_path}")
        return output_path

# Convenience functions
def quick_dcf_valuation(ticker: str, **kwargs) -> DCFResults:
    """
    Quick DCF valuation using fetched data
    
    Usage:
        results = quick_dcf_valuation("ADBE")
        print(f"Value per share: ${results.value_per_share:.2f}")
    """
    from ..fetchers import fetch_comprehensive_data
    
    financial_data = fetch_comprehensive_data(ticker)
    calculator = DCFCalculator()
    return calculator.calculate_dcf_valuation(financial_data, **kwargs)

def create_scenario_analysis(ticker: str, scenarios: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create scenario analysis for DCF valuation
    
    Usage:
        scenarios = {
            'Bull Case': {'terminal_growth_rate': 0.05, 'wacc': 0.10},
            'Base Case': {'terminal_growth_rate': 0.04, 'wacc': 0.117},
            'Bear Case': {'terminal_growth_rate': 0.03, 'wacc': 0.13}
        }
        analysis = create_scenario_analysis("ADBE", scenarios)
    """
    from ..fetchers import fetch_comprehensive_data
    
    financial_data = fetch_comprehensive_data(ticker)
    calculator = DCFCalculator()
    
    # Convert scenario dicts to DCFAssumptions
    dcf_scenarios = {}
    for name, params in scenarios.items():
        base_assumptions = calculator._create_default_assumptions(financial_data)
        
        # Update with scenario parameters
        for param, value in params.items():
            setattr(base_assumptions, param, value)
        
        dcf_scenarios[name] = base_assumptions
    
    return calculator.compare_scenarios(financial_data, dcf_scenarios)

if __name__ == "__main__":
    # Example usage
    print(" Testing DCF Calculator")
    
    # Test with Adobe
    try:
        results = quick_dcf_valuation("ADBE")
        
        print(f"\n Adobe DCF Valuation Results:")
        print(f"Value per Share: ${results.value_per_share:.2f}")
        print(f"Current Price: ${results.current_price:.2f}")
        print(f"Implied Return: {results.implied_return:.1%}")
        print(f"Enterprise Value: ${results.enterprise_value:,.0f}")
        
        # Create calculator for additional analysis
        calculator = DCFCalculator()
        summary_table = calculator.create_dcf_summary_table(results)
        print(f"\n DCF Summary:")
        print(summary_table.to_string(index=False))
        
        # Scenario analysis
        scenarios = {
            'Bull Case': {'terminal_growth_rate': 0.05, 'wacc': 0.10},
            'Base Case': {'terminal_growth_rate': 0.04, 'wacc': 0.117},
            'Bear Case': {'terminal_growth_rate': 0.03, 'wacc': 0.13}
        }
        scenario_analysis = create_scenario_analysis("ADBE", scenarios)
        print(f"\n Scenario Analysis:")
        print(scenario_analysis[['Value per Share', 'Implied Return']].to_string())
        
    except Exception as e:
        print(f" Error in DCF calculation: {e}")