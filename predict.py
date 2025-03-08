import sys
import json
import joblib
import pandas as pd
import numpy as np
import os

def log(message):
    """Helper function to print logs that will be captured by Node.js"""
    print(f"LOG: {message}", file=sys.stderr)

def validate_input(input_data):
    """Validate input ranges"""
    validations = [
        (0 <= input_data['hour'] <= 23, "Hour must be between 0 and 23"),
        (0 <= input_data['humidity'] <= 100, "Humidity must be between 0 and 100"),
        (-50 <= input_data['temperature'] <= 50, "Temperature must be between -50 and 50 Celsius"),
        (0 <= input_data['visibility'] <= 10000, "Visibility must be between 0 and 10000"),
        (0 <= input_data['windSpeed'] <= 100, "Wind speed must be between 0 and 100 m/s"),
        (0 <= input_data['rainfall'], "Rainfall cannot be negative"),
        (0 <= input_data['snowfall'], "Snowfall cannot be negative"),
        (0 <= input_data['solarRadiation'] <= 5, "Solar radiation must be between 0 and 5 MJ/m2"),
        (input_data['season'].lower() in ['spring', 'summer', 'fall', 'winter'], "Invalid season")
    ]
    
    for condition, message in validations:
        if not condition:
            raise ValueError(message)

def prepare_input_data(input_data):
    """Prepare input data with correct column names"""
    column_mapping = {
        'hour': 'Hour',
        'temperature': 'Temperature(°C)',
        'humidity': 'Humidity(%)',
        'windSpeed': 'Wind speed (m/s)',
        'visibility': 'Visibility (10m)',
        'dewPoint': 'Dew point temperature(°C)',
        'solarRadiation': 'Solar Radiation (MJ/m2)',
        'rainfall': 'Rainfall(mm)',
        'snowfall': 'Snowfall (cm)'
    }
    
    # Create a new dict with mapped column names
    mapped_data = {}
    for old_key, new_key in column_mapping.items():
        mapped_data[new_key] = input_data[old_key]
    
    return mapped_data

def predict(input_json):
    try:
        log(f"Current working directory: {os.getcwd()}")
        log(f"Looking for model files...")
        
        # Check if files exist
        model_path = 'xgboost_bike_rental_model.pkl'
        columns_path = 'X_columns.pkl'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(columns_path):
            raise FileNotFoundError(f"Columns file not found: {columns_path}")
            
        log("Loading model and columns...")
        # Load the model and columns
        model = joblib.load(model_path)
        X_columns = joblib.load(columns_path)
        log("Model and columns loaded successfully")
        
        # Parse and validate input data
        log("Parsing input data...")
        input_data = json.loads(input_json)
        log(f"Received input: {json.dumps(input_data, indent=2)}")
        
        # Validate input ranges
        validate_input(input_data)
        
        # Prepare input data with correct column names
        mapped_data = prepare_input_data(input_data)
        
        # Convert to DataFrame
        log("Converting to DataFrame...")
        input_df = pd.DataFrame([mapped_data])
        
        # Handle categorical variables
        log("Handling categorical variables...")
        season = input_data['season'].lower()
        seasons_cols = {
            'Seasons_Spring': 1 if season == 'spring' else 0,
            'Seasons_Summer': 1 if season == 'summer' else 0,
            'Seasons_Winter': 1 if season == 'winter' else 0
        }
        holiday_cols = {
            'Holiday_No Holiday': 1 if input_data['holiday'] == 'No Holiday' else 0
        }
        
        # Add categorical columns
        for col, value in {**seasons_cols, **holiday_cols}.items():
            input_df[col] = value
            
        log(f"Available columns: {input_df.columns.tolist()}")
        
        # Ensure DataFrame has all necessary columns in correct order
        log("Ensuring all necessary columns are present and ordered...")
        for col in X_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[X_columns]
        log(f"Final input shape: {input_df.shape}")
        log(f"Final columns: {input_df.columns.tolist()}")
        
        # Make prediction
        log("Making prediction...")
        prediction = model.predict(input_df)
        log(f"Raw prediction value: {prediction[0]}")
        
        # Ensure prediction is non-negative and rounded
        final_prediction = max(0, int(round(float(prediction[0]))))
        
        # Return result
        result = {
            'status': 'success',
            'prediction': final_prediction,
            'input_features': {
                'numerical': mapped_data,
                'categorical': {
                    'season': input_data['season'],
                    'holiday': input_data['holiday']
                }
            }
        }
        log(f"Prediction successful: {result}")
        print(json.dumps(result))
        
    except Exception as e:
        log(f"Error occurred: {str(e)}")
        error_result = {
            'status': 'error',
            'error': str(e)
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    # Read input from command line argument
    if len(sys.argv) > 1:
        log("Starting prediction process...")
        predict(sys.argv[1])
    else:
        log("No input data provided")
        print(json.dumps({'status': 'error', 'error': 'No input data provided'}))
        sys.exit(1) 