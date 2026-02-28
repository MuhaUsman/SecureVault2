import os
import sys
import subprocess
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_env_variables():
    """Check if required environment variables are set"""
    load_dotenv()
    required_vars = ['METTIS_USERNAME', 'METTIS_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        print(f"\nError: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in a .env file or in your environment.")
        print("You can use the .env file as a template.")
        return False
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import requests
        logger.info("All core dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        print(f"\nError: Missing dependency: {str(e)}")
        print("Please install the required dependencies.")
        return False

def run_app():
    """Run the Streamlit app"""
    try:
        print("\nStarting PSX Stock Forecasting App...")
        print("The app will be available at: http://localhost:5000")
        print("Press Ctrl+C to stop the application")
        
        # Run streamlit with specific configuration
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "5000",
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Streamlit app: {str(e)}")
        print(f"\nError: Failed to start Streamlit app: {str(e)}")
        return False
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
        return True
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\nUnexpected error: {str(e)}")
        return False

def main():
    """Main entry point"""
    print("\n=== PSX Stock Forecasting App ===\n")
    
    # Check environment variables
    if not check_env_variables():
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Run the app
    if not run_app():
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
