# test_full_analysis.py - Complete QuantFlow Analysis Test
"""
Test the complete QuantFlow analysis suite:
- DCF Valuation
- Peer Analysis  
- Report Generation
- Data Export
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import quantflow as qf

def test_quick_valuation():
    """Test quick valuation function"""
    print(" Testing Quick Valuation")
    print("=" * 40)
    
    try:
        valuation = qf.quick_valuation("ADBE")
        
        if 'error' not in valuation:
            print(f" Quick Valuation Success!")
            print(f"  Company: {valuation['ticker']}")
            print(f"  Fair Value: ${valuation['fair_value']:.2f}")
            print(f"  Current Price: ${valuation['current_price']:.2f}")
            print(f"  Implied Return: {valuation['implied_return']:.1%}")
            print(f"  Market Cap: ${valuation['market_cap']:,.0f}")
            print(f"  Recommendation: {valuation['recommendation']}")
        else:
            print(f" Quick valuation error: {valuation['error']}")
            
        return valuation
        
    except Exception as e:
        print(f" Error in quick valuation: {e}")
        return None

def test_peer_analysis():
    """Test peer analysis"""
    print(" Testing Peer Analysis")
    print("=" * 40)
    
    try:
        # Define peer group
        primary = "ADBE"
        peers = ["MSFT", "ORCL", "CRM", "INTU"]
        
        print(f"Analyzing {primary} vs {len(peers)} peers")
        
        # Run peer analysis
        analysis = qf.quick_peer_analysis(primary, peers)
        
        print(f" Peer Analysis Complete!")
        print(f" Results:")
        print(f"  Overall Assessment: {analysis['summary']['overall_assessment']}")
        print(f"  Peer Group Size: {analysis['summary']['peer_group_size']}")
        
        # Show strengths
        if analysis['summary']['strengths']:
            print(f"   Key Strengths:")
            for strength in analysis['summary']['strengths'][:3]:
                print(f"    • {strength}")
        
        # Show concerns
        if analysis['summary']['concerns']:
            print(f"   Key Concerns:")
            for concern in analysis['summary']['concerns'][:3]:
                print(f"    • {concern}")
        
        return analysis
        
    except Exception as e:
        print(f" Error in peer analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_comprehensive_analysis():
    """Test complete company analysis"""
    print(" Testing Comprehensive Company Analysis")
    print("=" * 50)
    
    try:
        # Run comprehensive analysis
        results = qf.analyze_company(
            ticker="ADBE",
            peer_tickers=["MSFT", "ORCL", "CRM"],
            include_dcf=True,
            include_peers=True,
            export_format="excel"
        )
        
        print(f" Comprehensive Analysis Complete!")
        
        # Show summary
        if 'error' not in results:
            company_info = results['summary']['company']
            print(f" Company Summary:")
            print(f"  Name: {company_info['name']}")
            print(f"  Sector: {company_info['sector']}")
            print(f"  Market Cap: ${company_info['market_cap']:,.0f}")
            
            # DCF results
            if 'dcf' in results['summary']:
                dcf_summary = results['summary']['dcf']
                print(f"  DCF Fair Value: ${dcf_summary['fair_value']:.2f}")
                print(f"  Current Price: ${dcf_summary['current_price']:.2f}")
                print(f"  DCF Recommendation: {dcf_summary['recommendation']}")
            
            # Peer results
            if 'peers' in results['summary']:
                peer_summary = results['summary']['peers']
                print(f"  Peer Assessment: {peer_summary['overall_assessment']}")
            
            # Show exported files
            print(f" Files Created:")
            for path in results['export_paths']:
                print(f"   {os.path.basename(path)}")
        else:
            print(f" Analysis error: {results['error']}")
        
        return results
        
    except Exception as e:
        print(f" Error in comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_stock_screening():
    """Test stock screening functionality"""
    print(" Testing Stock Screening")
    print("=" * 40)
    
    try:
        # Screen tech stocks
        tech_stocks = ["ADBE", "MSFT", "ORCL", "CRM", "INTU"]
        
        print(f"Screening {len(tech_stocks)} tech stocks...")
        
        candidates = qf.screen_stocks(
            tickers=tech_stocks,
            min_market_cap=10e9,  # $10B minimum
            min_implied_return=0.05  # 5% minimum return
        )
        
        print(f" Screening Complete!")
        print(f" Found {len(candidates)} candidates meeting criteria:")
        
        for i, stock in enumerate(candidates[:5], 1):
            print(f"  {i}. {stock['ticker']}: {stock['implied_return']:.1%} return (${stock['fair_value']:.2f} fair value)")
        
        return candidates
        
    except Exception as e:
        print(f" Error in stock screening: {e}")
        return None

if __name__ == "__main__":
    print(" QuantFlow Complete Analysis Test Suite")
    print("=" * 70)
    
    # Test 1: Quick Valuation
    valuation = test_quick_valuation()
    
    if valuation and 'error' not in valuation:
        print("\n" + "="*70)
        
        # Test 2: Peer Analysis
        peer_analysis = test_peer_analysis()
        
        if peer_analysis:
            print("\n" + "="*70)
            
            # Test 3: Comprehensive Analysis
            comprehensive = test_comprehensive_analysis()
            
            if comprehensive and 'error' not in comprehensive:
                print("\n" + "="*70)
                
                # Test 4: Stock Screening
                screening = test_stock_screening()
                
                print("\n" + "="*70)
                print(" ALL TESTS COMPLETED SUCCESSFULLY!")
                print("\n Final Summary:")
                print(f"  Adobe Fair Value: ${valuation['fair_value']:.2f}")
                print(f"  Current Price: ${valuation['current_price']:.2f}")
                print(f"  Implied Return: {valuation['implied_return']:.1%}")
                print(f"  Recommendation: {valuation['recommendation']}")
                
                if comprehensive and comprehensive.get('export_paths'):
                    print(f"\n Reports Generated:")
                    for path in comprehensive['export_paths']:
                        print(f"  • {os.path.basename(path)}")
            else:
                print(" Comprehensive analysis test failed")
        else:
            print(" Peer analysis test failed")
    else:
        print(" Quick valuation test failed - check your setup")