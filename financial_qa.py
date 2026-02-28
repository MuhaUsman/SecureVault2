import os
import json
import pandas as pd
import difflib
import requests
from typing import Dict, Any, List, Tuple
from ctransformers import AutoModelForCausalLM
from dotenv import load_dotenv  # Add this import

# Load the Mistral 7B model
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
    
    # Initialize the model with appropriate parameters
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type="mistral",
        context_length=2048,  # Context window size
        gpu_layers=0          # Set higher if you have GPU support
    )
    
    return model

# Extract query elements using the Mistral 7B model
# In the extract_query_elements function, improve date handling
def extract_query_elements(question: str, model, company_mapping: Dict[str, str], metric_mapping: Dict[str, str]) -> Dict[str, Any]:
    # Prepare the prompt for the model
    prompt = f"""<s>[INST] You are a financial data analyst. Extract the key elements from the following financial question and format the response as a JSON object with the following fields:
- symbols: List of company symbols mentioned (e.g., [\"OGDC\", \"UBL\"])
- metrics: List of financial metrics requested (e.g., [\"EPS\", \"Revenue\"])
- date: The date or period mentioned (in YYYY-MM-DD format, or YYYY for annual data)
- operation: The type of operation requested (\"value\" for direct lookup, \"cumulative\" for summing over periods, \"growth\" for percentage change, \"yes_no\" for boolean questions)

Question: {question}

Provide only the JSON object in your response, nothing else.
[/INST]</s>
"""
    
    # Generate response from the model
    response_text = model(prompt, max_new_tokens=512, temperature=0.1, top_p=0.9)
    
    # Extract and parse the JSON response
    try:
        # Find JSON content (it might be surrounded by markdown code blocks or other text)
        json_start = response_text.find('{')
        json_end = response_text.rfind('}')
        if json_start >= 0 and json_end >= 0:
            json_str = response_text[json_start:json_end+1]
            query_elements = json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")
            
        # Normalize company symbols and metrics
        if 'symbols' in query_elements:
            query_elements['symbols'] = [normalize_name(symbol, company_mapping) for symbol in query_elements['symbols']]
        if 'metrics' in query_elements:
            query_elements['metrics'] = [normalize_name(metric, metric_mapping) for metric in query_elements['metrics']]
            
        # Ensure date is in the correct format
        if 'date' in query_elements and query_elements['date']:
            query_elements['date'] = convert_date_format(query_elements['date'])
            
        return query_elements
    except Exception as e:
        print(f"Error parsing model response: {e}")
        print(f"Raw response: {response_text}")
        return {
            "symbols": [],
            "metrics": [],
            "date": "",
            "operation": "value"
        }

# Load and normalize company names and financial metrics
def load_mappings():
    # Load company name mappings
    company_df = pd.read_excel("CompanyName.xlsx")
    company_mapping = {}
    for _, row in company_df.iterrows():
        if 'Symbol' in row and 'Name' in row and not pd.isna(row['Symbol']) and not pd.isna(row['Name']):
            company_mapping[row['Name'].lower()] = row['Symbol']
            company_mapping[row['Symbol'].lower()] = row['Symbol']
    
    # Load financial metric mappings
    metric_df = pd.read_excel("SubHeadName.xlsx")
    metric_mapping = {}
    for _, row in metric_df.iterrows():
        if 'SubHeadName' in row and 'StandardName' in row and not pd.isna(row['SubHeadName']) and not pd.isna(row['StandardName']):
            metric_mapping[row['SubHeadName'].lower()] = row['StandardName']
            metric_mapping[row['StandardName'].lower()] = row['StandardName']
    
    return company_mapping, metric_mapping

# Normalize company and metric names using exact and fuzzy matching
def normalize_name(name: str, mapping: Dict[str, str]) -> str:
    name_lower = name.lower()
    
    # Try exact matching first
    if name_lower in mapping:
        return mapping[name_lower]
    
    # Try fuzzy matching if exact match fails
    matches = difflib.get_close_matches(name_lower, mapping.keys(), n=1, cutoff=0.7)
    if matches:
        return mapping[matches[0]]
    
    # Return original if no match found
    return name

# Fetch authentication token for Mettis Global API
# Fetch authentication token for Mettis Global API
# Improve the get_auth_token function with better debugging
def get_auth_token() -> str:
    # Load environment variables from .env file
    load_dotenv()
    
    # Get credentials from environment variables
    username = os.getenv("MG_API_USERNAME")
    password = os.getenv("MG_API_PASSWORD")
    
    if not username or not password:
        raise Exception("API credentials not found in .env file")
    
    print(f"Using API credentials for: {username}")
    
    # Prepare the authentication request
    auth_url = "https://drapi.mg-link.net/api/auth/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    
    # Form data for authentication - add grant_type parameter
    auth_data = {
        "grant_type": "password",  # Add this line - required by OAuth2 token endpoints
        "username": username,
        "password": password
    }
    
    print(f"Auth request URL: {auth_url}")
    print(f"Auth request headers: {headers}")
    print(f"Auth request data: {auth_data}")
    
    # Make the authentication request
    response = requests.post(auth_url, headers=headers, data=auth_data)
    
    print(f"Auth response status: {response.status_code}")
    
    if response.status_code == 200:
        token = response.json().get("access_token", "")
        print(f"Received token: {token[:10]}..." if token else "No token received")
        return token
    else:
        print(f"Auth response text: {response.text}")
        raise Exception(f"Authentication failed: {response.status_code} - {response.text}")

# Fetch financial data from Mettis Global API
# In the fetch_financial_data function, convert the date format
def fetch_financial_data(symbol: str, start_date: str, end_date: str = None, consolidated: int = 2, term: str = "INTERIM") -> Dict[str, Any]:
    token = get_auth_token()
    
    # If end_date is not provided, use start_date
    if not end_date:
        end_date = start_date
    
    # Convert date format from DD-MM-YYYY to YYYY-MM-DD if needed
    start_date_formatted = convert_date_format(start_date)
    end_date_formatted = convert_date_format(end_date)
    
    print(f"Using dates: {start_date_formatted} to {end_date_formatted}")
    
    api_url = "https://drapi.mg-link.net/api/Data/FinancialData"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-www-form-urlencoded"  # Updated content type
    }
    
    params = {
        "CompanySymbol": symbol,
        "StartDate": start_date_formatted,
        "EndDate": end_date_formatted,
        "Consolidated": consolidated,
        "Term": term
    }
    
    # Print the full request details for debugging
    print(f"API Request URL: {api_url}")
    print(f"API Request Headers: {headers}")
    print(f"API Request Params: {params}")
    
    response = requests.get(api_url, headers=headers, params=params)
    if response.status_code == 200:
        # Add debugging to see the structure of the response
        data = response.json()
        print(f"API Response structure for {symbol}: {type(data)}")
        print(f"API Response sample: {str(data)[:200]}...")
        return data
    else:
        print(f"API request failed: {response.status_code} - {response.text}")
        return {}

# Add this helper function to convert date formats
def convert_date_format(date_str: str) -> str:
    """Convert date from DD-MM-YYYY to YYYY-MM-DD format if needed."""
    if not date_str:
        return date_str
        
    # Check if the date is already in YYYY-MM-DD format
    if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
        return date_str
        
    # Try to parse DD-MM-YYYY format
    try:
        parts = date_str.split('-')
        if len(parts) == 3:
            day, month, year = parts
            # Ensure each part has the right length
            if len(day) == 2 and len(month) == 2 and len(year) == 4:
                return f"{year}-{month}-{day}"
            elif len(day) == 2 and len(month) == 2 and len(year) == 2:
                return f"20{year}-{month}-{day}"  # Assuming 20xx for 2-digit years
    except Exception as e:
        print(f"Error parsing date {date_str}: {e}")
        
    # Return original if we couldn't parse it
    return date_str

# Perform the requested operation on the financial data
# Update the perform_operation function to handle None values
def perform_operation(data: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    operation = query.get("operation", "value")
    metrics = query.get("metrics", [])
    symbols = query.get("symbols", [])
    
    # Add debugging
    print(f"Data structure: {type(data)}")
    if data is None:
        print("Warning: Received None data from API")
        return {symbol: {metric: None for metric in metrics} for symbol in symbols}
        
    if isinstance(data, dict):
        print(f"Data keys: {list(data.keys())}")
    elif isinstance(data, list) and len(data) > 0:
        print(f"First item type: {type(data[0])}")
        if isinstance(data[0], dict):
            print(f"First item keys: {list(data[0].keys())}")
    
    for symbol in symbols:
        result[symbol] = {}
        symbol_data = data.get(symbol, [])
        
        for metric in metrics:
            # Extract the relevant values from the data
            values = []
            
            # Handle different possible data structures
            if isinstance(symbol_data, dict) and "periods" in symbol_data:
                # Handle dictionary with periods key
                periods = symbol_data.get("periods", [])
                for period in periods:
                    if isinstance(period, dict) and "metrics" in period:
                        metrics_data = period["metrics"]
                        if isinstance(metrics_data, dict) and metric in metrics_data:
                            values.append(metrics_data[metric])
            elif isinstance(symbol_data, list):
                # Handle list response - this is the actual format from the API
                for item in symbol_data:
                    if isinstance(item, dict):
                        # Look for the metric in SubHeadName field
                        if item.get('SubHeadName') == metric:
                            values.append(item.get('Value'))
                        # Also check for common variations of metric names
                        elif metric == "EPS" and item.get('SubHeadName') == "Earnings Per Share":
                            values.append(item.get('Value'))
                        elif metric == "Revenue" and item.get('SubHeadName') in ["Sales", "Net Sales", "Revenue", "Total Revenue"]:
                            values.append(item.get('Value'))
            
            if not values:
                result[symbol][metric] = None
                continue
                
            # Perform the requested operation
            if operation == "value":
                # Return the most recent value
                result[symbol][metric] = values[0]
            elif operation == "cumulative":
                # Sum the values over the periods
                result[symbol][metric] = sum(values)
            elif operation == "growth":
                # Calculate percentage change
                if len(values) >= 2 and values[-1] != 0:
                    result[symbol][metric] = ((values[0] - values[-1]) / values[-1]) * 100
                else:
                    result[symbol][metric] = None
            elif operation == "yes_no":
                # Evaluate as boolean (e.g., for "positive" questions)
                result[symbol][metric] = values[0] > 0
    
    return result

# Main function to answer financial questions
def answer_question(question: str) -> Dict[str, Any]:
    # Load model and mappings
    model = load_model()
    company_mapping, metric_mapping = load_mappings()
    
    # Extract query elements
    query = extract_query_elements(question, model, company_mapping, metric_mapping)
    print(f"Extracted query: {query}")
    
    # Determine date range based on query
    start_date = query.get("date", "")
    end_date = None
    
    # For cumulative operations, we might need to adjust the date range
    if query.get("operation") == "cumulative" and "months" in question.lower():
        # This is a simplification - in a real app, you'd parse the number of months
        # and calculate the appropriate start date
        months = 3  # Default to quarterly
        if "9 months" in question.lower():
            months = 9
        elif "6 months" in question.lower():
            months = 6
        
        # Adjust start_date based on months
        # This is a placeholder - you'd need proper date handling
        year = int(start_date.split("-")[0])
        end_date = f"{year}-12-31"
    
    # Fetch data for each symbol
    all_data = {}
    for symbol in query.get("symbols", []):
        data = fetch_financial_data(symbol, start_date, end_date)
        if data:
            all_data[symbol] = data
    
    # Perform the requested operation
    results = {}
    for symbol, data in all_data.items():
        symbol_result = perform_operation({symbol: data}, query)
        if symbol in symbol_result:
            results[symbol] = symbol_result[symbol]
    
    return results

# CLI interface
def main():
    print("Financial Question Answering System")
    print("Type 'exit' to quit")
    
    while True:
        question = input("\nEnter your financial question: ")
        if question.lower() == 'exit':
            break
            
        try:
            result = answer_question(question)
            print("\nAnswer:")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()