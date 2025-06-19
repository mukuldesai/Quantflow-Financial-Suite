#!/usr/bin/env python3
"""
QuantFlow Financial Suite - Quick Start Script
Run this after setup to test your installation
"""

import sys
from pathlib import Path

# Add the quantflow package to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_installation():
    """Test that QuantFlow is properly installed and configured"""
    
    print("🧪 Testing QuantFlow Financial Suite Installation...")
    print("=" * 50)
    
    try:
        # Test 1: Import quantflow
        print("1️⃣  Testing package import...")
        import quantflow
        print(f"   ✅ QuantFlow v{quantflow.get_version()} imported successfully")
        
        # Test 2: Configuration
        print("\n2️⃣  Testing configuration...")
        from quantflow.config import get_config
        config = get_config()
        print(f"   ✅ Configuration loaded")
        print(f"   📁 Root path: {config.root_path}")
        print(f"   📊 Risk-free rate: {config.risk_free_rate:.2%}")
        
        # Test 3: Directory structure
        print("\n3️⃣  Testing directory structure...")
        required_dirs = ['data/raw', 'data/processed', 'quantflow', 'config']
        for dir_path in required_dirs:
            full_path = config.root_path / dir_path
            if full_path.exists():
                print(f"   ✅ {dir_path}")
            else:
                print(f"   ❌ {dir_path} - Missing!")
        
        # Test 4: Dependencies
        print("\n4️⃣  Testing key dependencies...")
        dependencies = [
            ('pandas', 'Data manipulation'),
            ('yfinance', 'Financial data'),
            ('yaml', 'Configuration'),
            ('pathlib', 'File handling')
        ]
        
        for package, description in dependencies:
            try:
                __import__(package)
                print(f"   ✅ {package} - {description}")
            except ImportError:
                print(f"   ❌ {package} - Not installed!")
        
        print("\n🎉 Installation test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Installation test failed: {e}")
        return False

def demo_basic_usage():
    """Demonstrate basic QuantFlow usage"""
    
    print("\n🚀 QuantFlow Demo - Basic Usage")
    print("=" * 40)
    
    try:
        import quantflow
        from quantflow.config import get_config
        
        # Show welcome message
        quantflow.welcome()
        
        print("\n📊 Configuration Info:")
        quantflow.info()
        
        # Show company-specific config
        config = get_config()
        print("\n🏢 Adobe (ADBE) Configuration:")
        adobe_config = config.get_company_config('ADBE')
        if adobe_config:
            print(f"   Beta: {config.get_company_beta('ADBE')}")
            growth_rates = config.get_company_growth_rates('ADBE')
            print(f"   Year 1 Growth: {growth_rates.get('year_1', 0):.1%}")
        else:
            print("   Using default configuration")
        
        print("\n✨ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def show_next_steps():
    """Show user what to do next"""
    
    print("\n📋 Next Steps to Get Started:")
    print("=" * 40)
    print("1. 🔑 Set up API keys:")
    print("   - Copy .env.template to .env")
    print("   - Add your Alpha Vantage, IEX, or other API keys")
    print("")
    print("2. ⚙️  Review configuration:")
    print("   - Edit config/assumptions.yaml")
    print("   - Adjust growth rates, margins, WACC assumptions")
    print("")
    print("3. 📊 Start analyzing:")
    print("   - Open notebooks/00_getting_started.ipynb")
    print("   - Try fetching data for a company (e.g., ADBE)")
    print("")
    print("4. 🏗️  Build models:")
    print("   - Use the DCF calculator")
    print("   - Create sensitivity analysis")
    print("   - Generate valuation reports")
    print("")
    print("5. 📈 Launch dashboard:")
    print("   - Run: streamlit run dashboard/streamlit_app.py")
    print("   - Explore interactive valuation tools")

def main():
    """Main function to run all tests and demos"""
    
    # Test installation
    success = test_installation()
    
    if success:
        # Run demo
        demo_basic_usage()
        
        # Show next steps
        show_next_steps()
        
        print(f"\n🎯 QuantFlow is ready to use!")
        print("Happy modeling! 📊💰")
    else:
        print("\n🔧 Please fix the installation issues and try again.")
        print("Refer to README.md for detailed setup instructions.")

if __name__ == "__main__":
    main()