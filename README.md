# Bike Rental Prediction Backend

This is the Node.js backend for the Bike Rental Prediction application that uses a Python XGBoost model for predictions.

## Prerequisites

1. Node.js (v14 or higher)
2. Python (v3.7 or higher)
3. Python packages:
   - scikit-learn
   - xgboost
   - pandas
   - joblib

## Setup

1. Install Node.js dependencies:
```bash
npm install
```

2. Install Python dependencies:
```bash
pip install scikit-learn xgboost pandas joblib
```

3. Ensure you have the model files in the backend directory:
   - xgboost_bike_rental_model.pkl
   - X_columns.pkl

4. Start the server:
```bash
# Development mode with auto-reload
npm run dev

# Production mode
npm start
```

The server will run on port 5000 by default.

## API Endpoints

### Health Check
```
GET /api/health
```

### Predict
```
POST /api/predict
```

Request body example:
```json
{
  "hour": 12,
  "season": "Summer",
  "holiday": "No Holiday",
  "temperature": 20.0,
  "humidity": 60,
  "dewPoint": 12.0,
  "windSpeed": 2.5,
  "visibility": 2000,
  "solarRadiation": 1.0,
  "rainfall": 0.0,
  "snowfall": 0.0
}
```

## Architecture

The backend uses:
- Express.js for the web server
- Python subprocess for model predictions
- Original XGBoost model for accurate predictions
- CORS support for frontend communication 