# PSX Stock Forecasting App

## Overview

This is a comprehensive stock forecasting application for Pakistan Stock Exchange (PSX) listed companies. The application fetches real-time and historical stock data from the Mettis Global API and provides multiple forecasting models including ARIMA, SARIMA, Prophet, and XGBoost. Built with Streamlit for the web interface, it offers interactive visualizations and model performance comparisons.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **Port**: 5000 (configured for Replit deployment)
- **Theme**: Custom dark theme with orange highlights
- **Layout**: Wide layout with expandable sidebar for parameters
- **Visualization**: Plotly for interactive charts and graphs

### Backend Architecture
- **Language**: Python 3.11
- **Structure**: Modular design with separate packages for API, models, and utilities
- **API Client**: Custom MettisGlobalClient for external data integration
- **Model Factory**: Factory pattern for creating different forecasting models
- **Data Processing**: Pandas and NumPy for time series manipulation

## Key Components

### API Integration (`api/`)
- **MettisGlobalClient**: Handles authentication and data fetching from Mettis Global API
- **Authentication**: OAuth2 password grant flow with token management
- **Data Sources**: 
  - Adjusted historical data (splits/dividends adjusted)
  - Daily stock prices (real-time PSX data)
- **Error Handling**: Comprehensive error handling with fallback mechanisms

### Forecasting Models (`models/`)
- **ARIMAModel**: Auto-regressive integrated moving average for linear patterns
- **SARIMAModel**: Seasonal ARIMA for data with seasonal components
- **ProphetModel**: Facebook Prophet for trend and seasonality capture
- **XGBoostModel**: Gradient boosting with engineered features (lag, volatility, momentum)
- **ModelFactory**: Centralized model creation and management

### Utilities (`utils/`)
- **data_utils.py**: Data validation, time series preparation, and preprocessing
- **visualization.py**: Interactive plotting functions using Plotly

## Data Flow

1. **User Input**: Symbol selection, date range, model choice, forecast horizon
2. **Authentication**: Token-based authentication with Mettis Global API
3. **Data Fetching**: Retrieve historical or real-time stock data
4. **Data Preprocessing**: Clean, validate, and prepare time series data
5. **Model Training**: Fit selected forecasting model to historical data
6. **Prediction**: Generate forecasts for specified horizon
7. **Visualization**: Display results with interactive charts
8. **Export**: Optional download of forecasts as JSON/CSV

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualizations
- **requests**: HTTP client for API calls
- **python-dotenv**: Environment variable management

### Forecasting Libraries
- **statsmodels**: ARIMA and SARIMA models
- **prophet**: Facebook Prophet forecasting
- **xgboost**: Gradient boosting framework
- **scikit-learn**: Machine learning utilities and metrics

### API Integration
- **Mettis Global API**: Stock data provider for PSX
- **Authentication**: OAuth2 password grant flow
- **Rate Limiting**: Built-in request throttling

## Deployment Strategy

### Replit Configuration
- **Runtime**: Python 3.11 with Nix package manager
- **Deployment Target**: Autoscale
- **Run Command**: `streamlit run app.py --server.port 5000`
- **Dependencies**: Managed via pyproject.toml with uv package manager

### Environment Variables
- **METTIS_USERNAME**: API username for authentication
- **METTIS_PASSWORD**: API password for authentication  
- **METTIS_BASE_URL**: Base URL for Mettis Global API (defaults to https://drapi.mg-link.net)

### Security Considerations
- Credentials stored in environment variables
- Token-based authentication with automatic refresh
- Sensitive data masking in logs
- Secure HTTPS communication with external APIs

## Changelog

- June 26, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.