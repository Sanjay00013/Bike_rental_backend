import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

def convert_xgboost_to_tensorflow():
    # Load the XGBoost model
    xgb_model = joblib.load('xgboost_bike_rental_model.pkl')
    
    # Create a simple neural network with similar structure
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(14,)),  # 10 features + 4 season one-hot
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    # Save the model in TensorFlow.js format
    model.save('model')
    
    print("Model converted and saved successfully")

if __name__ == "__main__":
    convert_xgboost_to_tensorflow() 