# test_dcf.py - Quick DCF Valuation Test
"""
Test script for QuantFlow DCF Calculator
Run this to test your DCF valuation on Adobe
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantflow.fetchers import fetch_comprehensive_data
from quantflow.models.dcf_calculator import quick_dcf_valuation, DCFCalculator
from quantflow.utils.io_utils import create_dcf_report

def test_quick_dcf():
    """Test quick DCF valuation"""
    print(" Testing Quick DCF Valuation for Adobe (ADBE)")
    print("=" * 50)
    
    try:
        # Quick DCF valuation
        results = quick_dcf_valuation("ADBE")
        
        print(f" DCF Analysis Complete!")
        print(f" Results Summary:")
        print(f"  Fair Value per Share: ${results.value_per_share:.2f}")
        print(f"  Current Price: ${results.current_price:.2f}")
        print(f"  Implied Return: {results.implied_return:.1%}")
        print(f"  Enterprise Value: ${results.enterprise_value:,.0f}")
        print(f"  Equity Value: ${results.equity_value:,.0f}")
        
        # Investment recommendation
        if results.implied_return > 0.15:
            recommendation = " STRONG BUY"
        elif results.implied_return > 0.05:
            recommendation = " BUY"
        elif results.implied_return > -0.05:
            recommendation = " HOLD"
        else:
            recommendation = " SELL"
        
        print(f"  Recommendation: {recommendation}")
        print()
        
        return results
        
    except Exception as e:
        print(f" Error in DCF calculation: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_detailed_dcf():
    """Test detailed DCF with custom assumptions"""
    print(" Testing Detailed DCF Analysis")
    print("=" * 50)
    
    try:
        # Fetch data first
        print(" Fetching financial data...")
        financial_data = fetch_comprehensive_data("ADBE")
        
        # Create DCF calculator
        calculator = DCFCalculator()
        
        # Run DCF analysis
        print(" Running DCF calculation...")
        results = calculator.calculate_dcf_valuation(
            financial_data, 
            include_sensitivity=True
        )
        
        # Show detailed results
        print(f" Detailed DCF Complete!")
        print(f" DCF Breakdown:")
        print(f"  PV of FCFFs (5 years): ${results.pv_of_fcff:,.0f}")
        print(f"  PV of Terminal Value: ${results.pv_of_terminal_value:,.0f}")
        print(f"  Enterprise Value: ${results.enterprise_value:,.0f}")
        print(f"  Net Cash: ${results.net_cash:,.0f}")
        print(f"  Equity Value: ${results.equity_value:,.0f}")
        print(f"  Shares Outstanding: {results.shares_outstanding:,.0f}")
        print()
        
        # Show FCFF projections
        if results.fcff_projections is not None:
            print(" FCFF Projections (in millions):")
            fcff_display = results.fcff_projections[['Revenue', 'NOPAT', 'FCFF']].copy()
            for col in ['Revenue', 'NOPAT', 'FCFF']:
                fcff_display[col] = fcff_display[col] / 1_000_000
            print(fcff_display.round(0))
            print()
        
        # Show sensitivity analysis
        if results.sensitivity_table is not None and not results.sensitivity_table.empty:
            print(" Sensitivity Analysis (Value per Share):")
            print("WACC vs Terminal Growth Rate")
            print(results.sensitivity_table.round(2))
            print()
        
        return results
        
    except Exception as e:
        print(f" Error in detailed DCF: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_dcf_report():
    """Test DCF report generation"""
    print(" Testing DCF Report Generation")
    print("=" * 50)
    
    try:
        # Get DCF results
        results = quick_dcf_valuation("ADBE")
        financial_data = fetch_comprehensive_data("ADBE")
        
        # Create report
        print(" Generating DCF report...")
        report_path = create_dcf_report("ADBE", results, financial_data, "excel")
        
        print(f" DCF Report created: {report_path}")
        return report_path
        
    except Exception as e:
        print(f" Error creating DCF report: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print(" QuantFlow DCF Calculator Test Suite")
    print("=" * 60)
    
    # Test 1: Quick DCF
    results1 = test_quick_dcf()
    
    if results1:
        print("\n" + "="*60)
        
        # Test 2: Detailed DCF
        results2 = test_detailed_dcf()
        
        if results2:
            print("\n" + "="*60)
            
            # Test 3: Report Generation
            report_path = test_dcf_report()
            
            print("\n" + "="*60)
            print(" All DCF tests completed successfully!")
            print(f" Adobe Fair Value: ${results1.value_per_share:.2f}")
            print(f" Implied Return: {results1.implied_return:.1%}")
            if report_path:
                print(f" Report saved to: {report_path}")
        else:
            print("⚠️ Detailed DCF test failed")
    else:
        print(" Quick DCF test failed")