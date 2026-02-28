import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from api.mettis_client import MettisGlobalClient
from models.model_factory import ModelFactory
from utils.data_utils import prepare_time_series
from utils.visualization import create_model_comparison_plot, create_performance_metrics_plot

def evaluate_model(model_name, train_data, test_data, symbol, price_field="AdjustedPrice"):
    """Evaluate a single model on test data
    
    Args:
        model_name (str): Name of the model to evaluate
        train_data (list): Training data as a list of dicts with Date and price field
        test_data (list): Test data as a list of dicts with Date and price field
        symbol (str): Stock symbol
        price_field (str): Name of the price field to use
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    # Create model instance
    model = ModelFactory.create_model(model_name)
    
    if model is None:
        logger.error(f"Failed to create {model_name} model")
        return None
    
    # Fit model to training data
    logger.info(f"Training {model_name} model...")
    success = model.fit(train_data, symbol, price_field=price_field)
    
    if not success:
        logger.error(f"Failed to train {model_name} model")
        return None
    
    # Generate forecast for test period
    logger.info(f"Generating forecast with {model_name} model...")
    forecast = model.forecast(len(test_data))
    
    if forecast is None:
        logger.error(f"Failed to generate forecast with {model_name} model")
        return None
    
    # Extract forecasted values
    forecast_values = [item['predicted_price'] for item in forecast['forecast']]
    
    # Extract actual values from test data
    actual_values = [item[price_field] for item in test_data]
    
    # Ensure we have the same number of values
    min_length = min(len(actual_values), len(forecast_values))
    actual_values = actual_values[:min_length]
    forecast_values = forecast_values[:min_length]
    
    if len(actual_values) == 0:
        logger.error("No data points available for evaluation")
        return None
    
    # Calculate metrics
    mse = mean_squared_error(actual_values, forecast_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_values, forecast_values)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Handle division by zero
    actual_array = np.array(actual_values)
    forecast_array = np.array(forecast_values)
    non_zero_mask = actual_array != 0
    
    if np.sum(non_zero_mask) > 0:
        mape = np.mean(np.abs((actual_array[non_zero_mask] - forecast_array[non_zero_mask]) / actual_array[non_zero_mask])) * 100
    else:
        mape = float('inf')
    
    # Return evaluation metrics
    return {
        'model': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'forecast': forecast,
        'actual': test_data[:min_length]
    }

def evaluate_all_models(symbol, data_source="adjusted", train_period=180, test_period=30):
    """Evaluate all models on historical data
    
    Args:
        symbol (str): Stock symbol to evaluate
        data_source (str): Either "adjusted" or "daily"
        train_period (int): Number of days for training data
        test_period (int): Number of days for test data
        
    Returns:
        list: List of evaluation results for each model
    """
    # Calculate date ranges
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=train_period + test_period)).strftime("%Y-%m-%d")
    
    # Initialize API client
    client = MettisGlobalClient()
    
    # Authenticate with API
    if not client.ensure_authenticated():
        logger.error("Failed to authenticate with Mettis Global API")
        return None
    
    # Fetch data from API based on source
    logger.info(f"Fetching {data_source} data for {symbol} from {start_date} to {end_date}...")
    
    if data_source == "adjusted":
        data = client.get_adjusted_data(symbol, start_date, end_date)
        price_field = "AdjustedPrice"
        
        if data is None:
            logger.error("Failed to fetch adjusted data from Mettis Global API")
            return None
        
        # Extract time series data
        time_series = client.extract_time_series(data, price_field)
    else:  # daily
        data = client.get_daily_stock_prices_range(symbol, start_date, end_date)
        price_field = "LastPrice"
        time_series = data
    
    if time_series is None or len(time_series) == 0:
        logger.error("No time series data found")
        return None
    
    # Prepare data for modeling
    is_valid_data, result = prepare_time_series(time_series, price_field=price_field)
    
    if not is_valid_data:
        logger.error(f"Invalid time series data: {result}")
        return None
    
    # Convert DataFrame back to list of dicts for model input
    df = result
    all_data = df.to_dict('records')
    
    # Split data into training and test sets
    split_idx = len(all_data) - test_period
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    logger.info(f"Training data: {len(train_data)} records, Test data: {len(test_data)} records")
    
    # Get available models
    model_names = [model['name'] for model in ModelFactory.get_model_info()]
    
    # Evaluate each model
    results = []
    for model_name in model_names:
        logger.info(f"Evaluating {model_name} model...")
        result = evaluate_model(model_name, train_data, test_data, symbol, price_field)
        if result is not None:
            results.append(result)
            logger.info(f"{model_name} - RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}, MAPE: {result['mape']:.2f}%")
    
    return results

def plot_evaluation_results(results, symbol, data_source):
    """Plot evaluation results
    
    Args:
        results (list): List of evaluation results for each model
        symbol (str): Stock symbol
        data_source (str): Data source type
    """
    if not results:
        logger.error("No results to plot")
        return
    
    # Create DataFrame for metrics
    metrics_df = pd.DataFrame([
        {
            'Model': r['model'],
            'RMSE': r['rmse'],
            'MAE': r['mae'],
            'MAPE': r['mape']
        } for r in results
    ])
    
    # Sort by RMSE (lower is better)
    metrics_df = metrics_df.sort_values('RMSE')
    
    # Create bar chart for metrics
    fig = px.bar(metrics_df, x='Model', y=['RMSE', 'MAE'], barmode='group',
                title=f'Model Performance Comparison for {symbol} ({data_source.title()} Data)')
    fig.update_layout(yaxis_title='Error (lower is better)', xaxis_title='Model')
    fig.write_html(f"{symbol}_{data_source}_model_metrics.html")
    
    # Create MAPE chart
    fig2 = px.bar(metrics_df, x='Model', y='MAPE',
                 title=f'Mean Absolute Percentage Error (MAPE) for {symbol} ({data_source.title()} Data)')
    fig2.update_layout(yaxis_title='MAPE % (lower is better)', xaxis_title='Model')
    fig2.write_html(f"{symbol}_{data_source}_model_mape.html")
    
    # Create comparison plot of actual vs predicted for all models
    # Get test data from first result (same for all models)
    test_data = pd.DataFrame(results[0]['actual'])
    price_field = "AdjustedPrice" if data_source == "adjusted" else "LastPrice"
    
    # Create figure
    fig3 = go.Figure()
    
    # Add actual prices
    fig3.add_trace(go.Scatter(
        x=test_data['Date'],
        y=test_data[price_field],
        mode='lines+markers',
        name='Actual',
        line=dict(color='black', width=2)
    ))
    
    # Add forecasted prices for each model
    colors = px.colors.qualitative.Plotly
    for i, result in enumerate(results):
        forecast_df = pd.DataFrame(result['forecast']['forecast'])
        fig3.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['predicted_price'],
            mode='lines',
            name=f"{result['model']} (RMSE: {result['rmse']:.2f})",
            line=dict(color=colors[i % len(colors)])
        ))
    
    # Update layout
    fig3.update_layout(
        title=f"Model Forecast Comparison for {symbol} ({data_source.title()} Data)",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Models",
        hovermode="x unified"
    )
    
    # Save figure
    fig3.write_html(f"{symbol}_{data_source}_forecast_comparison.html")
    
    logger.info(f"Saved evaluation plots to {symbol}_{data_source}_*.html files")

def main():
    """Main function"""
    print("=== PSX Stock Forecasting Model Evaluation ===\n")
    
    # Load environment variables
    load_dotenv()
    
    # Check if required environment variables are set
    if not os.getenv("METTIS_USERNAME") or not os.getenv("METTIS_PASSWORD"):
        print("❌ Missing required environment variables: METTIS_USERNAME and/or METTIS_PASSWORD")
        print("Please set these variables in a .env file or in your environment.")
        return 1
    
    # Get parameters from command line arguments or use defaults
    symbol = sys.argv[1] if len(sys.argv) > 1 else "HBL"
    data_source = sys.argv[2] if len(sys.argv) > 2 else "adjusted"  # "adjusted" or "daily"
    train_period = int(sys.argv[3]) if len(sys.argv) > 3 else 180
    test_period = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    
    # Validate data source
    if data_source not in ["adjusted", "daily"]:
        print("❌ Invalid data source. Use 'adjusted' or 'daily'")
        return 1
    
    print(f"Evaluating models for {symbol} using {data_source} data")
    print(f"Training period: {train_period} days, Test period: {test_period} days\n")
    
    # Evaluate all models
    results = evaluate_all_models(symbol, data_source, train_period, test_period)
    
    if results is None or len(results) == 0:
        print("❌ Failed to evaluate models. See log for details.")
        return 1
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Data Source: {data_source.title()}")
    print("-" * 50)
    for result in sorted(results, key=lambda x: x['rmse']):
        print(f"{result['model']}:")
        print(f"  RMSE: {result['rmse']:.4f}")
        print(f"  MAE: {result['mae']:.4f}")
        print(f"  MAPE: {result['mape']:.2f}%")
        print()
    
    # Find best model
    best_model = min(results, key=lambda x: x['rmse'])
    print(f"🏆 Best Model: {best_model['model']} (RMSE: {best_model['rmse']:.4f})")
    
    # Plot results
    plot_evaluation_results(results, symbol, data_source)
    
    print(f"\n✅ Evaluation complete! Results saved to {symbol}_{data_source}_*.html files.")
    
    # Print usage instructions
    print(f"\n📖 Usage Instructions:")
    print(f"python evaluate_models.py [SYMBOL] [DATA_SOURCE] [TRAIN_DAYS] [TEST_DAYS]")
    print(f"  SYMBOL: Stock symbol (default: HBL)")
    print(f"  DATA_SOURCE: 'adjusted' or 'daily' (default: adjusted)")
    print(f"  TRAIN_DAYS: Training period in days (default: 180)")
    print(f"  TEST_DAYS: Test period in days (default: 30)")
    print(f"\nExample:")
    print(f"python evaluate_models.py UBL daily 90 15")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
