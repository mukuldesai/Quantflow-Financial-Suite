import sys
import os
from quantflow.fetchers import fetch_comprehensive_data

if os.name == 'nt':
    sys.stdout.reconfigure(encoding='utf-8')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure terminal prints Unicode correctly (optional for Windows)
sys.stdout.reconfigure(encoding='utf-8')

print("🚀 Running QuantFlow Fetcher Test")

ticker = "ADBE"
data = fetch_comprehensive_data(ticker)

# Handle fetch failure or missing sections gracefully
dcf = data.get('dcf_summary', {})
current = dcf.get('current_financials')

if current and 'revenue' in current:
    print(f"✅ Revenue: ${current['revenue']:,.0f}")
else:
    print("⚠️  DCF summary is incomplete or missing.")
    if 'error' in dcf:
        print(f"❌ DCF error: {dcf['error']}")

# Optionally show fetch errors if available
fetch_errors = data.get("data_quality", {}).get("fetch_errors", {})
if fetch_errors:
    print("\n⚠️  Data Fetch Errors:")
    for key, err in fetch_errors.items():
        print(f" - {key}: {err}")
