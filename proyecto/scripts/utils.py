"""
Utility Functions Module
Helper functions for forecasting and data analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def train_test_split(df, test_size=0.2):
    """
    Split time series data into train and test sets
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    return train_df, test_df

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error
    """
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Square Error
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error
    """
    return np.mean(np.abs(y_true - y_pred))

def simple_moving_average(series, window):
    """
    Calculate simple moving average
    """
    return series.rolling(window=window).mean()

def exponential_smoothing(series, alpha=0.3):
    """
    Apply exponential smoothing forecast
    """
    result = [series.iloc[0]]
    for i in range(1, len(series)):
        result.append(alpha * series.iloc[i] + (1 - alpha) * result[i-1])
    return np.array(result)

def random_forest_forecast(X_train, y_train, X_test, n_estimators=100):
    """
    Simple random forest-like forecaster using random projections
    Creates ensemble of random linear models
    """
    n_features = X_train.shape[1]
    n_models = n_estimators
    predictions_list = []
    
    np.random.seed(42)
    
    for _ in range(n_models):
        # Random feature selection
        random_features = np.random.choice(n_features, size=max(1, n_features // 2), replace=False)
        X_train_subset = X_train[:, random_features]
        X_test_subset = X_test[:, random_features]
        
        # Simple linear regression on random subset
        try:
            # Add bias term
            X_train_bias = np.column_stack([X_train_subset, np.ones(X_train_subset.shape[0])])
            X_test_bias = np.column_stack([X_test_subset, np.ones(X_test_subset.shape[0])])
            
            # Solve least squares
            coeffs = np.linalg.lstsq(X_train_bias, y_train, rcond=None)[0]
            predictions = X_test_bias @ coeffs
            predictions_list.append(predictions)
        except:
            pass
    
    # Average ensemble predictions
    if predictions_list:
        final_predictions = np.mean(predictions_list, axis=0)
    else:
        final_predictions = np.mean(y_train) * np.ones(X_test.shape[0])
    
    return final_predictions

def naive_forecast(series, steps=1):
    """
    Naive forecast: repeat the last value
    """
    last_value = series.iloc[-1]
    return np.array([last_value] * steps)

def linear_regression_forecast(X_train, y_train, X_test):
    """
    Simple linear regression forecaster
    """
    # Add bias term
    X_train_bias = np.column_stack([X_train, np.ones(X_train.shape[0])])
    X_test_bias = np.column_stack([X_test, np.ones(X_test.shape[0])])
    
    # Solve least squares
    coeffs = np.linalg.lstsq(X_train_bias, y_train, rcond=None)[0]
    predictions = X_test_bias @ coeffs
    
    return predictions
