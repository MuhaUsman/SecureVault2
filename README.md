# PSX Stock Forecasting App

A comprehensive stock forecasting application for Pakistan Stock Exchange (PSX) listed companies. This app fetches real-time and historical stock data from the Mettis Global API and uses various forecasting models to predict future stock prices with support for dual data sources.

## Features

- **Dual Data Sources**: Choose between adjusted historical data or daily stock prices
  - **Adjusted Data**: Historical prices adjusted for splits and dividends
  - **Daily Stock Prices**: Real-time daily last trading prices from PSX
- **Real-time Data**: Fetch the latest stock prices from PSX using the Mettis Global API
- **Historical Analysis**: Analyze historical stock data with comprehensive adjustments
- **Multiple Forecasting Models**:
  - **ARIMA**: Best for linear, stationary price patterns
  - **SARIMA**: Best for seasonal trends in time series
  - **Prophet**: Good for capturing trend + seasonality with minimal tuning
  - **XGBoost**: Ensemble tree model using lag features (e.g., last 5 days, volatility, momentum)
- **Flexible Forecast Horizons**: Predict stock prices for next day, week, or month
- **Interactive Visualizations**: View historical data and forecasts with interactive plots
- **Model Evaluation**: Compare performance of different forecasting models across data sources
- **Secure API Handling**: Proper authentication and token management
- **Export Capabilities**: Download forecasts as JSON or CSV files

## Prerequisites

- Python 3.8 or higher
- Mettis Global API credentials (username and password)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd psx-stock-forecasting-app
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install streamlit pandas numpy plotly requests python-dotenv
   pip install statsmodels prophet xgboost scikit-learn matplotlib
   ```

4. Create a `.env` file with your Mettis Global API credentials:
   ```
   METTIS_USERNAME=your_username
   METTIS_PASSWORD=your_password
   METTIS_BASE_URL=https://drapi.mg-link.net
   ```

## Usage

### Running the App

Use the provided run script to start the application:

```bash
python run.py
