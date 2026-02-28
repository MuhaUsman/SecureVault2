import os
import sys
import json
import logging
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the MettisGlobalClient
from api.mettis_client import MettisGlobalClient

def test_authentication():
    """Test authentication with Mettis Global API"""
    try:
        # Initialize client
        client = MettisGlobalClient()
        
        # Mask credentials for logging
        username = os.getenv('METTIS_USERNAME')
        masked_username = username[:3] + "*" * (len(username) - 6) + username[-3:] if len(username) > 6 else "***"
        
        print("\n🔑 Testing Authentication with Mettis Global API")
        print(f"API Endpoint: {client.auth_url}")
        print(f"Username: {masked_username}")
        
        # Attempt authentication
        success = client.authenticate()
        
        if success:
            print("\n✅ Authentication successful!")
            print(f"Token expires in: {client.token_expires_in} seconds")
            print(f"Authentication method used: {client.successful_content_type}")
            
            # Mask token for display
            if client.access_token:
                masked_token = client.access_token[:10] + "..." + client.access_token[-10:] if len(client.access_token) > 20 else "***"
                print(f"Access token: {masked_token}")
            
            return True
        else:
            print("\n❌ Authentication failed!")
            print("\nAuthentication Details:")
            print(f"API Endpoint: {client.auth_url}")
            print(f"Username: {masked_username}")
            
            print("\n🔍 Troubleshooting tips:")
            print("1. Verify your username and password in the .env file")
            print("   - Check for typos or extra spaces")
            print("   - Ensure credentials are correctly formatted")
            print("2. Check if the API endpoint is correct")
            print(f"   - Current endpoint: {client.auth_url}")
            print("   - Verify with Mettis Global documentation")
            print("3. Ensure you have internet connectivity")
            print("   - Try accessing https://drapi.mg-link.net in your browser")
            print("4. Check if the API service is available")
            print("   - The service might be experiencing downtime")
            print("5. Check your firewall or network settings")
            print("   - Your network might be blocking the API requests")
            
            return False
    except Exception as e:
        print(f"\n❌ Error during authentication test: {str(e)}")
        import traceback
        print(f"\nDetailed error information:\n{traceback.format_exc()}")
        return False

def test_adjusted_data_api(symbol="HBL", days=30):
    """Test fetching adjusted data for a specific symbol and date range"""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    print(f"\n📊 Testing AdjustedData API for symbol: {symbol} from {start_date} to {end_date}...")
    client = MettisGlobalClient()
    
    if not client.ensure_authenticated():
        print("❌ Authentication failed!")
        return False
    
    data = client.get_adjusted_data(symbol, start_date, end_date)
    
    if data is None:
        print("❌ Failed to fetch adjusted data!")
        return False
    else:
        print("✅ Successfully fetched adjusted data!")
        # Extract time series and print a sample
        time_series = client.extract_time_series(data, "AdjustedPrice")
        if time_series is None or len(time_series) == 0:
            print("❌ No time series data found!")
            print("Raw response:")
            print(json.dumps(data, indent=2)[:500] + "..." if len(json.dumps(data)) > 500 else json.dumps(data, indent=2))
            return False
        else:
            sample_size = min(5, len(time_series))
            print(f"\nSample of {sample_size} adjusted data points:")
            for i in range(sample_size):
                print(f"  {time_series[i]['Date']}: {time_series[i]['AdjustedPrice']}")
            return True

def test_daily_stock_prices_api(symbol="HBL", days=7):
    """Test fetching daily stock prices for a specific symbol"""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    print(f"\n📈 Testing PSXStockPrices API for symbol: {symbol} from {start_date} to {end_date}...")
    client = MettisGlobalClient()
    
    if not client.ensure_authenticated():
        print("❌ Authentication failed!")
        return False
    
    data = client.get_daily_stock_prices_range(symbol, start_date, end_date)
    
    if data is None:
        print("❌ Failed to fetch daily stock prices!")
        return False
    else:
        print("✅ Successfully fetched daily stock prices!")
        if len(data) > 0:
            sample_size = min(5, len(data))
            print(f"\nSample of {sample_size} daily price data points:")
            for i in range(sample_size):
                print(f"  {data[i]['Date']}: {data[i]['LastPrice']}")
            return True
        else:
            print("❌ No daily stock price data found!")
            return False

def main():
    """Main function to run all tests"""
    print("\n=== Mettis Global API Connection Test Tool ===")
    
    # Display environment information
    print("\n📊 Environment Information:")
    import platform
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Requests Version: {requests.__version__}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Check for .env file
    env_file_path = os.path.join(os.getcwd(), '.env')
    env_file_exists = os.path.isfile(env_file_path)
    print(f".env File Exists: {'✅ Yes' if env_file_exists else '❌ No'}")
    
    if not env_file_exists:
        print("\n❌ Error: .env file not found. Please create a .env file with your credentials.")
        print("Example .env file content:")
        print("METTIS_USERNAME=your_username")
        print("METTIS_PASSWORD=your_password")
        print("METTIS_BASE_URL=https://drapi.mg-link.net")
        return 1
    
    # Load environment variables
    print("\n🔄 Loading environment variables...")
    load_dotenv()
    
    # Check if environment variables are loaded
    username_loaded = bool(os.getenv('METTIS_USERNAME'))
    password_loaded = bool(os.getenv('METTIS_PASSWORD'))
    base_url_loaded = bool(os.getenv('METTIS_BASE_URL'))
    
    print(f"METTIS_USERNAME loaded: {'✅ Yes' if username_loaded else '❌ No'}")
    print(f"METTIS_PASSWORD loaded: {'✅ Yes' if password_loaded else '❌ No'}")
    print(f"METTIS_BASE_URL loaded: {'✅ Yes' if base_url_loaded else '❌ No'}")
    
    if not username_loaded or not password_loaded:
        print("\n❌ Error: Required environment variables not loaded. Please check your .env file.")
        print("Make sure your .env file contains:")
        if not username_loaded:
            print("- METTIS_USERNAME=your_username")
        if not password_loaded:
            print("- METTIS_PASSWORD=your_password")
        if not base_url_loaded:
            print("- METTIS_BASE_URL=https://drapi.mg-link.net")
        return 1
    
    # Check internet connectivity
    print("\n🌐 Checking internet connectivity...")
    try:
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code == 200:
            print("Internet connectivity: ✅ Connected")
        else:
            print(f"Internet connectivity: ⚠️ Limited (Status code: {response.status_code})")
    except requests.exceptions.RequestException:
        print("Internet connectivity: ❌ Not connected")
        print("Please check your internet connection before proceeding.")
    
    # Test authentication
    auth_success = test_authentication()
    if not auth_success:
        print("\nAuthentication failed. Skipping API tests.")
        print("\n🔍 Additional Troubleshooting Steps:")
        print("1. Verify API documentation for any recent changes")
        print("2. Check if your account has the necessary permissions")
        print("3. Try with a different network connection")
        print("4. Contact Mettis Global support if the issue persists")
        return 1
    
    # Test adjusted data API
    adjusted_data_success = test_adjusted_data_api()
    
    # Test daily stock prices API
    daily_prices_success = test_daily_stock_prices_api()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Authentication: {'✅ Passed' if auth_success else '❌ Failed'}")
    print(f"Adjusted Data API: {'✅ Passed' if adjusted_data_success else '❌ Failed'}")
    print(f"Daily Stock Prices API: {'✅ Passed' if daily_prices_success else '❌ Failed'}")
    
    if auth_success and adjusted_data_success and daily_prices_success:
        print("\n✅ All tests passed! You're ready to use the PSX Stock Forecasting App with both data sources.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting tips:")
        print("1. Verify your API credentials in the .env file")
        print("2. Check your internet connection")
        print("3. Ensure the API endpoints are accessible from your network")
        print("4. Check if the API service is currently available")
        return 1

if __name__ == "__main__":
    sys.exit(main())
