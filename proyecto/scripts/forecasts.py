"""
Time Series Forecasting Script
Uses various forecasting methods to predict asset prices
Saves results and metrics to results/
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Import utility functions
from utils import (
    train_test_split, calculate_mape, calculate_rmse, calculate_mae,
    simple_moving_average, exponential_smoothing, random_forest_forecast,
    naive_forecast, linear_regression_forecast
)

# ============================================================================
# 1. LOAD PROCESSED DATA
# ============================================================================

def load_data():
    """Load processed combined data"""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    df_combined = pd.read_csv(os.path.join(data_dir, 'combined_data.csv'))
    df_combined['date'] = pd.to_datetime(df_combined['date'])
    
    print("✓ Loaded combined processed data")
    
    return df_combined

# ============================================================================
# 2. PREPARE FORECASTING DATA
# ============================================================================

def prepare_forecast_data(df_combined, target_asset='Asset_A', lookback=5):
    """Prepare data for forecasting"""
    # Extract features and target
    feature_cols = [col for col in df_combined.columns if 'lag' in col or 'rolling' in col]
    
    if len(feature_cols) == 0:
        # Use available price columns as features
        feature_cols = [col for col in df_combined.columns 
                       if col.startswith('Asset_') and not col.endswith('_Return')]
    
    X = df_combined[feature_cols].values if feature_cols else df_combined[[col for col in df_combined.columns if col.startswith('Asset_')]].values
    y = df_combined[target_asset].values if target_asset in df_combined.columns else df_combined['Asset_A'].values
    
    return X, y, target_asset

# ============================================================================
# 3. FORECASTING METHODS
# ============================================================================

def forecast_naive(y_train, y_test):
    """Naive forecasting method"""
    predictions = naive_forecast(pd.Series(y_train), steps=len(y_test))
    return predictions

def forecast_moving_average(y_train, y_test, window=5):
    """Moving average forecasting"""
    ma_series = simple_moving_average(pd.Series(y_train), window=window)
    # Forecast: use the last MA value
    last_ma = ma_series.iloc[-1]
    predictions = np.array([last_ma] * len(y_test))
    return predictions

def forecast_exponential_smoothing(y_train, y_test, alpha=0.3):
    """Exponential smoothing forecasting"""
    smoothed = exponential_smoothing(pd.Series(y_train), alpha=alpha)
    # Forecast: use the last smoothed value
    predictions = np.array([smoothed[-1]] * len(y_test))
    return predictions

def forecast_linear_regression(X_train, y_train, X_test, y_test):
    """Linear regression forecasting"""
    predictions = linear_regression_forecast(X_train, y_train, X_test)
    return predictions

def forecast_random_forest(X_train, y_train, X_test, y_test, n_estimators=100):
    """Random forest-like ensemble forecasting"""
    predictions = random_forest_forecast(X_train, y_train, X_test, n_estimators=n_estimators)
    return predictions

# ============================================================================
# 4. RUN ALL FORECASTS
# ============================================================================

def run_all_forecasts(X, y, test_size=0.2):
    """Run all forecasting methods and compare"""
    X_train, X_test = train_test_split(pd.DataFrame(X), test_size=test_size)
    y_train_df, y_test_df = train_test_split(pd.DataFrame({'y': y}), test_size=test_size)
    
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train_df['y'].values
    y_test = y_test_df['y'].values
    
    results = {}
    
    print("\n" + "="*70)
    print("RUNNING FORECASTS")
    print("="*70)
    
    # 1. Naive Forecast
    print("\n1. Naive Forecast...")
    try:
        pred_naive = forecast_naive(y_train, y_test)
        results['Naive'] = {
            'predictions': pred_naive,
            'rmse': calculate_rmse(y_test, pred_naive),
            'mae': calculate_mae(y_test, pred_naive),
            'mape': calculate_mape(y_test, pred_naive)
        }
        print(f"   ✓ RMSE: {results['Naive']['rmse']:.6f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 2. Moving Average
    print("2. Moving Average Forecast...")
    try:
        pred_ma = forecast_moving_average(y_train, y_test, window=5)
        results['Moving Average'] = {
            'predictions': pred_ma,
            'rmse': calculate_rmse(y_test, pred_ma),
            'mae': calculate_mae(y_test, pred_ma),
            'mape': calculate_mape(y_test, pred_ma)
        }
        print(f"   ✓ RMSE: {results['Moving Average']['rmse']:.6f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 3. Exponential Smoothing
    print("3. Exponential Smoothing Forecast...")
    try:
        pred_exp = forecast_exponential_smoothing(y_train, y_test, alpha=0.3)
        results['Exponential Smoothing'] = {
            'predictions': pred_exp,
            'rmse': calculate_rmse(y_test, pred_exp),
            'mae': calculate_mae(y_test, pred_exp),
            'mape': calculate_mape(y_test, pred_exp)
        }
        print(f"   ✓ RMSE: {results['Exponential Smoothing']['rmse']:.6f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 4. Linear Regression
    print("4. Linear Regression Forecast...")
    try:
        pred_lr = forecast_linear_regression(X_train, y_train, X_test, y_test)
        results['Linear Regression'] = {
            'predictions': pred_lr,
            'rmse': calculate_rmse(y_test, pred_lr),
            'mae': calculate_mae(y_test, pred_lr),
            'mape': calculate_mape(y_test, pred_lr)
        }
        print(f"   ✓ RMSE: {results['Linear Regression']['rmse']:.6f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 5. Random Forest Ensemble
    print("5. Random Forest Ensemble Forecast...")
    try:
        pred_rf = forecast_random_forest(X_train, y_train, X_test, y_test, n_estimators=100)
        results['Random Forest Ensemble'] = {
            'predictions': pred_rf,
            'rmse': calculate_rmse(y_test, pred_rf),
            'mae': calculate_mae(y_test, pred_rf),
            'mape': calculate_mape(y_test, pred_rf)
        }
        print(f"   ✓ RMSE: {results['Random Forest Ensemble']['rmse']:.6f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    return results, y_test, X_train, X_test, y_train

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================

def save_forecast_results(results, y_test, target_asset='Asset_A'):
    """Save forecast results and metrics"""
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create metrics summary
    metrics_data = []
    for method, data in results.items():
        metrics_data.append({
            'Method': method,
            'RMSE': data['rmse'],
            'MAE': data['mae'],
            'MAPE (%)': data['mape']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_filepath = os.path.join(results_dir, f'{target_asset}_forecast_metrics.csv')
    metrics_df.to_csv(metrics_filepath, index=False)
    print(f"\n✓ Saved metrics: {target_asset}_forecast_metrics.csv")
    
    # Save detailed predictions
    predictions_data = {'Actual': y_test}
    for method, data in results.items():
        predictions_data[f'{method}_Predictions'] = data['predictions']
    
    predictions_df = pd.DataFrame(predictions_data)
    predictions_filepath = os.path.join(results_dir, f'{target_asset}_forecast_predictions.csv')
    predictions_df.to_csv(predictions_filepath, index=False)
    print(f"✓ Saved predictions: {target_asset}_forecast_predictions.csv")
    
    # Save summary report
    report_filepath = os.path.join(results_dir, f'{target_asset}_forecast_report.txt')
    with open(report_filepath, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"FORECAST RESULTS FOR {target_asset}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL PERFORMANCE METRICS\n")
        f.write("-"*70 + "\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")
        
        # Find best model
        best_model_idx = metrics_df['RMSE'].idxmin()
        best_model = metrics_df.loc[best_model_idx, 'Method']
        f.write(f"BEST MODEL: {best_model}\n")
        f.write(f"RMSE: {metrics_df.loc[best_model_idx, 'RMSE']:.6f}\n\n")
        
        f.write("PREDICTIONS vs ACTUALS (First 10 samples)\n")
        f.write("-"*70 + "\n")
        f.write(predictions_df.head(10).to_string())
        f.write("\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"✓ Saved report: {target_asset}_forecast_report.txt")
    
    return metrics_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("TIME SERIES FORECASTING PIPELINE")
    print("="*70)
    print()
    
    # Load data
    print("STEP 1: Loading Data")
    print("-"*70)
    df_combined = load_data()
    print()
    
    # Prepare forecast data
    print("STEP 2: Preparing Forecast Data")
    print("-"*70)
    X, y, target_asset = prepare_forecast_data(df_combined, target_asset='Asset_A')
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print()
    
    # Run forecasts
    print("STEP 3: Training and Testing Forecasting Models")
    print("-"*70)
    results, y_test, X_train, X_test, y_train = run_all_forecasts(X, y, test_size=0.2)
    print()
    
    # Save results
    print("STEP 4: Saving Results")
    print("-"*70)
    metrics_df = save_forecast_results(results, y_test, target_asset=target_asset)
    print()
    
    # Display summary
    print("STEP 5: Results Summary")
    print("-"*70)
    print(metrics_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("✓ FORECASTING COMPLETE")
    print(f"✓ All results saved to: results/")
    print("="*70)

if __name__ == "__main__":
    main()
