"""
Financial Data Processing Script
Processes raw financial data for forecasting and analysis
Applies normalization, feature engineering, and data quality transformations
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ============================================================================
# 1. LOAD RAW DATA
# ============================================================================

def load_raw_data():
    """Load all raw CSV files from data/raw directory"""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    
    # Load datasets
    df_assets = pd.read_csv(os.path.join(data_dir, 'financial_assets_main.csv'))
    df_returns = pd.read_csv(os.path.join(data_dir, 'financial_returns_volume.csv'))
    df_indicators = pd.read_csv(os.path.join(data_dir, 'market_indicators.csv'))
    
    # Convert date columns to datetime
    df_assets['date'] = pd.to_datetime(df_assets['date'])
    df_returns['date'] = pd.to_datetime(df_returns['date'])
    df_indicators['date'] = pd.to_datetime(df_indicators['date'])
    
    print("✓ Loaded raw data files:")
    print(f"  - Assets: {len(df_assets)} rows")
    print(f"  - Returns: {len(df_returns)} rows")
    print(f"  - Indicators: {len(df_indicators)} rows")
    
    return df_assets, df_returns, df_indicators

# ============================================================================
# 2. DATA QUALITY & PREPROCESSING
# ============================================================================

def handle_missing_values(df, method='forward_fill'):
    """Handle missing values in the dataset"""
    if df.isnull().sum().sum() > 0:
        if method == 'forward_fill':
            df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')  # Backward fill any remaining
        print(f"✓ Handled missing values using {method}")
    return df

def remove_duplicates(df):
    """Remove duplicate rows based on date"""
    n_before = len(df)
    df = df.drop_duplicates(subset=['date'], keep='first')
    if len(df) < n_before:
        print(f"✓ Removed {n_before - len(df)} duplicate rows")
    return df

# ============================================================================
# 3. NORMALIZATION & SCALING
# ============================================================================

def normalize_prices(df_assets):
    """Apply Z-score normalization to asset prices"""
    asset_cols = [col for col in df_assets.columns if col.startswith('Asset_')]
    
    df_normalized = df_assets.copy()
    
    # Z-score normalization: (x - mean) / std
    for col in asset_cols:
        mean = df_assets[col].mean()
        std = df_assets[col].std()
        df_normalized[col] = (df_assets[col] - mean) / std
    
    print(f"✓ Applied Z-score normalization to {len(asset_cols)} assets")
    
    return df_normalized

def normalize_minmax(df, columns, range=(0, 1)):
    """Apply Min-Max scaling to specified columns"""
    df_scaled = df.copy()
    
    # Min-Max scaling: (x - min) / (max - min) * (range_max - range_min) + range_min
    range_min, range_max = range
    
    for col in columns:
        x_min = df[col].min()
        x_max = df[col].max()
        if x_max != x_min:  # Avoid division by zero
            df_scaled[col] = (df[col] - x_min) / (x_max - x_min) * (range_max - range_min) + range_min
        else:
            df_scaled[col] = range_min
    
    return df_scaled

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

def create_lagged_features(df, columns, lags=[1, 5, 20]):
    """Create lagged features for time series forecasting"""
    df_lag = df.copy()
    
    for col in columns:
        for lag in lags:
            df_lag[f'{col}_lag_{lag}'] = df_lag[col].shift(lag)
    
    print(f"✓ Created lagged features (lags: {lags}) for {len(columns)} columns")
    
    return df_lag

def create_rolling_statistics(df, columns, windows=[5, 20]):
    """Create rolling mean and std deviation"""
    df_roll = df.copy()
    
    for col in columns:
        for window in windows:
            df_roll[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
            df_roll[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
    
    print(f"✓ Created rolling statistics (windows: {windows}) for {len(columns)} columns")
    
    return df_roll

def create_momentum_indicators(df_returns):
    """Create momentum and trend indicators"""
    df_mom = df_returns.copy()
    
    return_cols = [col for col in df_returns.columns if col.endswith('_Return')]
    
    # Cumulative returns
    for col in return_cols:
        df_mom[f'{col.replace("_Return", "")}_cumulative_return'] = (1 + df_returns[col]).cumprod() - 1
    
    print(f"✓ Created momentum indicators for {len(return_cols)} assets")
    
    return df_mom

# ============================================================================
# 5. MERGE AND COMBINE DATASETS
# ============================================================================

def merge_all_data(df_assets_norm, df_returns_proc, df_indicators):
    """Merge all processed datasets"""
    # Merge on date
    df_merged = df_assets_norm.merge(df_returns_proc, on='date', how='inner')
    df_merged = df_merged.merge(df_indicators, on='date', how='inner')
    
    # Sort by date
    df_merged = df_merged.sort_values('date').reset_index(drop=True)
    
    print(f"✓ Merged datasets: {len(df_merged)} rows × {len(df_merged.columns)} columns")
    
    return df_merged

# ============================================================================
# 6. CORRELATION ANALYSIS
# ============================================================================

def compute_correlations(df_assets):
    """Compute correlation matrix between assets"""
    asset_cols = [col for col in df_assets.columns if col.startswith('Asset_')]
    corr_matrix = df_assets[asset_cols].corr()
    
    print(f"✓ Computed correlation matrix:")
    print(corr_matrix.round(3))
    
    return corr_matrix

# ============================================================================
# 7. SAVE PROCESSED DATA
# ============================================================================

def save_processed_data(dfs_to_save):
    """Save all processed dataframes to CSV"""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, df in dfs_to_save.items():
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"✓ Saved: {filename} ({len(df)} rows)")
    
    print(f"\n✓ All processed data saved to data/processed/")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("FINANCIAL DATA PROCESSING PIPELINE")
    print("="*70)
    print()
    
    # Load raw data
    print("STEP 1: Loading Raw Data")
    print("-"*70)
    df_assets, df_returns, df_indicators = load_raw_data()
    print()
    
    # Data quality
    print("STEP 2: Data Quality & Preprocessing")
    print("-"*70)
    df_assets = handle_missing_values(df_assets)
    df_returns = handle_missing_values(df_returns)
    df_indicators = handle_missing_values(df_indicators)
    df_assets = remove_duplicates(df_assets)
    df_returns = remove_duplicates(df_returns)
    df_indicators = remove_duplicates(df_indicators)
    print()
    
    # Normalization
    print("STEP 3: Normalization & Scaling")
    print("-"*70)
    df_assets_normalized = normalize_prices(df_assets)
    
    # Normalize indicators (0-1 scale)
    indicator_cols = ['Market_Index', 'Risk_Free_Rate', 'Volatility_Index', 
                      'Sector_A', 'Sector_B', 'Sector_C']
    df_indicators_scaled = normalize_minmax(df_indicators, indicator_cols, range=(0, 1))
    print(f"✓ Applied Min-Max scaling (0-1) to {len(indicator_cols)} indicator columns")
    print()
    
    # Feature engineering
    print("STEP 4: Feature Engineering")
    print("-"*70)
    asset_cols = [col for col in df_assets_normalized.columns if col.startswith('Asset_')]
    df_assets_features = create_lagged_features(df_assets_normalized, asset_cols, lags=[1, 5, 20])
    df_assets_features = create_rolling_statistics(df_assets_features, asset_cols, windows=[5, 20])
    
    df_returns_features = create_momentum_indicators(df_returns)
    print()
    
    # Correlation analysis
    print("STEP 5: Correlation Analysis")
    print("-"*70)
    corr_matrix = compute_correlations(df_assets)
    print()
    
    # Merge datasets
    print("STEP 6: Merging Datasets")
    print("-"*70)
    df_combined = merge_all_data(df_assets_features, df_returns_features, df_indicators_scaled)
    
    # Remove rows with NaN from lagged features
    df_combined = df_combined.dropna()
    print(f"✓ After removing NaN from lagged features: {len(df_combined)} rows")
    print()
    
    # Save processed data
    print("STEP 7: Saving Processed Data")
    print("-"*70)
    
    dfs_to_save = {
        'assets_normalized.csv': df_assets_normalized,
        'assets_features.csv': df_assets_features,
        'returns_features.csv': df_returns_features,
        'indicators_scaled.csv': df_indicators_scaled,
        'combined_data.csv': df_combined,
        'correlation_matrix.csv': corr_matrix
    }
    
    save_processed_data(dfs_to_save)
    print()
    
    # Summary statistics
    print("STEP 8: Summary Statistics")
    print("-"*70)
    print("\nCombined Dataset Shape:", df_combined.shape)
    print("\nFirst few rows of combined data:")
    print(df_combined.head(3))
    print("\nData types:")
    print(df_combined.dtypes)
    print("\nBasic statistics:")
    print(df_combined.describe().round(3))
    
    print("\n" + "="*70)
    print("✓ PROCESSING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
