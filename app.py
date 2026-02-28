import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta
import dotenv

# Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="PSX Stock Forecasting App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom modules
from api.mettis_client import MettisGlobalClient
from models.model_factory import ModelFactory
from utils.data_utils import validate_symbol, validate_date_range, prepare_time_series, get_forecast_horizon_days
from utils.visualization import create_stock_plot, create_model_comparison_plot, create_performance_metrics_plot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize API client
api_client = MettisGlobalClient()

# App title and description
st.title("📈 PSX Stock Forecasting App")
st.markdown("""
This application predicts stock prices for companies listed on the Pakistan Stock Exchange (PSX) 
using the Mettis Global API. Select a data source, stock symbol, date range, forecasting model, and horizon to get started.
""")

# Sidebar for inputs
st.sidebar.header("📊 Parameters")

# Data source selection
st.sidebar.subheader("Data Source")
data_source_options = {
    "Adjusted Data": {
        "value": "adjusted",
        "description": "Historical adjusted prices with splits and dividends",
        "price_field": "AdjustedPrice"
    },
    "Daily Stock Prices": {
        "value": "daily",
        "description": "Daily last trading prices from PSX",
        "price_field": "LastPrice"
    }
}

selected_data_source_name = st.sidebar.selectbox(
    "Select Data Source", 
    list(data_source_options.keys()),
    help="Choose between adjusted historical data or daily stock prices"
)
selected_data_source = data_source_options[selected_data_source_name]

# Display data source description
st.sidebar.markdown(f"*{selected_data_source['description']}*")

# Stock symbol input
symbol = st.sidebar.text_input("Stock Symbol (e.g., HBL)", value="HBL").upper()

# Date range selection (only for adjusted data)
if selected_data_source["value"] == "adjusted":
    st.sidebar.subheader("Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        today = datetime.now()
        default_end_date = today.strftime("%Y-%m-%d")
        default_start_date = (today - timedelta(days=365)).strftime("%Y-%m-%d")
        start_date = st.date_input("Start Date", value=datetime.strptime(default_start_date, "%Y-%m-%d"))

    with col2:
        end_date = st.date_input("End Date", value=datetime.strptime(default_end_date, "%Y-%m-%d"))

    # Convert dates to string format
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
else:
    # For daily data, use date range selection like adjusted data
    st.sidebar.subheader("Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        today = datetime.now()
        default_end_date = today.strftime("%Y-%m-%d")
        default_start_date = (today - timedelta(days=90)).strftime("%Y-%m-%d")
        start_date = st.date_input("Start Date", value=datetime.strptime(default_start_date, "%Y-%m-%d"), key="daily_start_date")

    with col2:
        end_date = st.date_input("End Date", value=datetime.strptime(default_end_date, "%Y-%m-%d"), key="daily_end_date")

    # Convert dates to string format
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

# Model selection
st.sidebar.subheader("Forecasting Model")
model_info = ModelFactory.get_model_info()
model_options = {f"{model['name']}: {model['description']}": model['name'] for model in model_info}
selected_model_option = st.sidebar.selectbox("Select Model", list(model_options.keys()))
selected_model = model_options[selected_model_option]

# Forecast horizon selection
st.sidebar.subheader("Forecast Horizon")
horizon_options = {
    "1-day: Next day forecast": "1-day",
    "7-day: Weekly forecast": "7-day",
    "30-day: Monthly forecast": "30-day"
}
selected_horizon_option = st.sidebar.selectbox("Select Horizon", list(horizon_options.keys()))
selected_horizon = horizon_options[selected_horizon_option]

# Fetch data button
fetch_button = st.sidebar.button("Fetch Data & Forecast")

# Main content area
if fetch_button:
    # Validate inputs
    if not validate_symbol(symbol):
        st.error(f"Invalid stock symbol: {symbol}. Symbol should be 1-5 uppercase letters.")
    else:
        # Show loading spinner while fetching data
        with st.spinner(f"Fetching {selected_data_source_name.lower()} from Mettis Global API..."):
            # Validate date range (only for adjusted data)
            if selected_data_source["value"] == "adjusted":
                is_valid_range, error_message = validate_date_range(start_date_str, end_date_str)
                if not is_valid_range:
                    st.error(error_message)
                    st.stop()
            
            # Fetch data from API
            try:
                # Authenticate with API
                if not api_client.ensure_authenticated():
                    st.error("Failed to authenticate with Mettis Global API. Please check your credentials.")
                    st.stop()
                
                # Fetch data based on selected source
                if selected_data_source["value"] == "adjusted":
                    data = api_client.get_adjusted_data(symbol, start_date_str, end_date_str)
                    if data is None:
                        st.error("Failed to fetch adjusted data from Mettis Global API. Please try again later.")
                        st.stop()
                    
                    # Extract time series data
                    time_series = api_client.extract_time_series(data, price_field="AdjustedPrice")
                else:
                    # Fetch daily stock prices
                    data = api_client.get_daily_stock_prices_range(symbol, start_date_str, end_date_str)
                    if data is None:
                        st.error("Failed to fetch daily stock prices from Mettis Global API. Please try again later.")
                        st.stop()
                    
                    # Extract time series data
                    time_series = data  # Already in the right format
                
                if time_series is None or len(time_series) == 0:
                    st.error("No time series data found for the specified parameters.")
                    st.stop()
                
                # Prepare data for modeling
                is_valid_data, result = prepare_time_series(time_series, price_field=selected_data_source["price_field"])
                
                if not is_valid_data:
                    st.error(result)  # Display error message
                    st.stop()
                
                # Display success message
                st.success(f"Successfully fetched {selected_data_source_name.lower()} for {symbol}")
                
                # Display data summary
                st.subheader("Data Summary")
                df = result  # This is the prepared DataFrame
                
                # Display basic statistics
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                with stats_col1:
                    st.metric("Number of Data Points", len(df))
                with stats_col2:
                    latest_price = df[selected_data_source["price_field"]].iloc[-1]
                    st.metric("Latest Price", f"{latest_price:.2f}")
                with stats_col3:
                    price_change = df[selected_data_source["price_field"]].iloc[-1] - df[selected_data_source["price_field"]].iloc[0]
                    st.metric("Price Change", f"{price_change:.2f}", delta=f"{price_change:.2f}")
                
                # Display data table
                with st.expander("View Raw Data"):
                    st.dataframe(df)
                
                # Train the selected model
                st.subheader(f"Training {selected_model} Model")
                with st.spinner(f"Training {selected_model} model..."):
                    # Create model instance
                    model = ModelFactory.create_model(selected_model)
                    
                    if model is None:
                        st.error(f"Failed to create {selected_model} model.")
                        st.stop()
                    
                    # Fit model to data
                    success = model.fit(time_series, symbol, price_field=selected_data_source["price_field"])
                    
                    if not success:
                        st.error(f"Failed to train {selected_model} model.")
                        st.stop()
                    
                    st.success(f"Successfully trained {selected_model} model")
                    
                    # Generate forecast
                    st.subheader(f"{selected_horizon} Forecast")
                    with st.spinner("Generating forecast..."):
                        # Get number of days to forecast
                        horizon_days = get_forecast_horizon_days(selected_horizon)
                        
                        # Generate forecast
                        forecast = model.forecast(horizon_days)
                        
                        if forecast is None:
                            st.error("Failed to generate forecast.")
                            st.stop()
                        
                        # Display forecast
                        st.json(forecast)
                        
                        # Create and display plot
                        st.subheader("Forecast Visualization")
                        fig = create_stock_plot(df, forecast, selected_model, price_field=selected_data_source["price_field"], data_source=selected_data_source_name)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            # Convert numpy types to native Python types for JSON serialization
                            def convert_numpy_types(obj):
                                if isinstance(obj, np.integer):
                                    return int(obj)
                                elif isinstance(obj, np.floating):
                                    return float(obj)
                                elif isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                elif isinstance(obj, dict):
                                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                                elif isinstance(obj, list):
                                    return [convert_numpy_types(item) for item in obj]
                                else:
                                    return obj
                            
                            json_data = convert_numpy_types(forecast)
                            st.download_button(
                                label="Download Forecast as JSON",
                                data=json.dumps(json_data, indent=2),
                                file_name=f"{symbol}_{selected_model}_{selected_horizon}_forecast.json",
                                mime="application/json"
                            )
                        
                        with col2:
                            # Convert forecast to DataFrame for CSV download
                            forecast_df = pd.DataFrame(forecast['forecast'])
                            st.download_button(
                                label="Download Forecast as CSV",
                                data=forecast_df.to_csv(index=False),
                                file_name=f"{symbol}_{selected_model}_{selected_horizon}_forecast.csv",
                                mime="text/csv"
                            )
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.error(f"Error in main app: {e}", exc_info=True)

# Display information about the app when no data is loaded
if not fetch_button:
    # Display app information
    st.info("👈 Please select parameters and click 'Fetch Data & Forecast' to get started.")
    
    # Display information about data sources
    st.subheader("Available Data Sources")
    
    for source_name, source_info in data_source_options.items():
        st.markdown(f"**{source_name}**: {source_info['description']}")
    
    # Display information about the models
    st.subheader("Available Forecasting Models")
    
    for model in model_info:
        st.markdown(f"**{model['name']}**: {model['description']}")
    
    # Display example usage
    st.subheader("How to Use")
    st.markdown("""
    1. **Select Data Source**: Choose between adjusted historical data or daily stock prices
    2. **Enter Stock Symbol**: Type the PSX stock symbol (e.g., HBL, UBL, MCB)
    3. **Set Date Range**: For adjusted data, select start and end dates. For daily data, choose number of days
    4. **Choose Model**: Select from ARIMA, SARIMA, Prophet, or XGBoost models
    5. **Select Horizon**: Choose forecast period (1 day, 7 days, or 30 days)
    6. **Fetch & Forecast**: Click the button to fetch data and generate predictions
    """)
