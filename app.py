from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and columns
MODEL_PATH = "xgboost_bike_rental_model.pkl"
COLUMNS_PATH = "X_columns.pkl"

model = None
X_columns = None

def load_model_and_columns():
    global model, X_columns
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
            model = joblib.load(MODEL_PATH)
            X_columns = joblib.load(COLUMNS_PATH)
            print("Model and columns loaded successfully")
        else:
            print("Model or columns file not found. Please train the model first.")
    except Exception as e:
        print(f"Error loading model or columns: {e}")

load_model_and_columns()

# API Route for Health Check
@app.route('/api/health', methods=['GET'])
def health_check():
    if model is None or X_columns is None:
        return jsonify({
            'status': 'error',
            'message': 'Model or columns not loaded. Please ensure model files exist.'
        }), 503
    return jsonify({'status': 'running', 'message': 'Backend is working'}), 200

# API Route for Prediction
@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None or X_columns is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded. Please ensure model files exist.'
        }), 503

    try:
        # Get input data from request
        input_data = request.json

        # Convert input dictionary to DataFrame
        input_df = pd.DataFrame([input_data])

        # One-hot encode categorical variables
        categorical_features = ["season", "holiday"]  # Updated to match frontend field names
        input_df = pd.get_dummies(input_df, columns=categorical_features)

        # Ensure DataFrame has all necessary columns
        for col in X_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match training data
        input_df = input_df[X_columns]

        # Make prediction
        prediction = model.predict(input_df)

        # Return prediction as JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
