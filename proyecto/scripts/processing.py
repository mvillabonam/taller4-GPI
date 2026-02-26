"""
Financial Data Processing Script
Processes raw financial data for forecasting and analysis
Applies normalization, feature engineering, and data quality transformations
"""

import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime

# ============================================================================
# 0. UTILITIES 
# ============================================================================

def validate_columns(df, required, df_name="dataframe"):
    """Validate required columns exist."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{df_name}] Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path

# ============================================================================
# 1. LOAD RAW DATA (MODIFIED: configurable path + validations)
# ============================================================================

def load_raw_data(data_dir=None):
    """Load all raw CSV files from data/raw directory."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

    assets_path = os.path.join(data_dir, 'financial_assets_main.csv')
    returns_path = os.path.join(data_dir, 'financial_returns_volume.csv')
    indicators_path = os.path.join(data_dir, 'market_indicators.csv')

    # Existence checks (NEW)
    for p in [assets_path, returns_path, indicators_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Raw file not found: {p}")

    df_assets = pd.read_csv(assets_path)
    df_returns = pd.read_csv(returns_path)
    df_indicators = pd.read_csv(indicators_path)

    # Validate date column (NEW)
    validate_columns(df_assets, ['date'], "df_assets")
    validate_columns(df_returns, ['date'], "df_returns")
    validate_columns(df_indicators, ['date'], "df_indicators")

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
# 2. DATA QUALITY & PREPROCESSING (MODIFIED: modern fill + generic dup removal)
# ============================================================================

def handle_missing_values(df, method='forward_fill'):
    """Handle missing values in the dataset."""
    total_missing = int(df.isnull().sum().sum())
    if total_missing > 0:
        if method == 'forward_fill':
            df = df.ffill()
        df = df.bfill()
        print(f"✓ Handled {total_missing} missing values using {method} + backward_fill")
    return df

def remove_duplicates(df, subset_cols=None):
    """Remove duplicate rows (default: based on date if available)."""
    if subset_cols is None:
        subset_cols = ['date'] if 'date' in df.columns else None

    n_before = len(df)
    df = df.drop_duplicates(subset=subset_cols, keep='first')
    removed = n_before - len(df)
    if removed > 0:
        key = subset_cols if subset_cols is not None else "all columns"
        print(f"✓ Removed {removed} duplicate rows (subset={key})")
    return df

# ============================================================================
# 3. NORMALIZATION & SCALING
# ============================================================================

def normalize_prices(df_assets):
    """Apply Z-score normalization to asset prices."""
    asset_cols = [col for col in df_assets.columns if col.startswith('Asset_')]
    if len(asset_cols) == 0:
        raise ValueError("No asset columns found starting with 'Asset_' in df_assets.")

    df_normalized = df_assets.copy()

    for col in asset_cols:
        mean = df_assets[col].mean()
        std = df_assets[col].std()
        if std == 0 or np.isnan(std):
            # NEW: handle constant series
            df_normalized[col] = 0.0
        else:
            df_normalized[col] = (df_assets[col] - mean) / std

    print(f"✓ Applied Z-score normalization to {len(asset_cols)} assets")
    return df_normalized

def normalize_minmax(df, columns, range=(0, 1)):
    """Apply Min-Max scaling to specified columns."""
    df_scaled = df.copy()
    range_min, range_max = range

    for col in columns:
        if col not in df_scaled.columns:
            raise ValueError(f"Column '{col}' not found for Min-Max scaling.")
        x_min = df_scaled[col].min()
        x_max = df_scaled[col].max()
        if x_max != x_min:
            df_scaled[col] = (df_scaled[col] - x_min) / (x_max - x_min) * (range_max - range_min) + range_min
        else:
            df_scaled[col] = range_min

    return df_scaled

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

def create_lagged_features(df, columns, lags=[1, 5, 20]):
    """Create lagged features for time series forecasting."""
    df_lag = df.copy()
    for col in columns:
        for lag in lags:
            df_lag[f'{col}_lag_{lag}'] = df_lag[col].shift(lag)

    print(f"✓ Created lagged features (lags: {lags}) for {len(columns)} columns")
    return df_lag

def create_rolling_statistics(df, columns, windows=[5, 20]):
    """Create rolling mean and std deviation."""
    df_roll = df.copy()
    for col in columns:
        for window in windows:
            df_roll[f'{col}_rolling_mean_{window}'] = df_roll[col].rolling(window=window).mean()
            df_roll[f'{col}_rolling_std_{window}'] = df_roll[col].rolling(window=window).std()

    print(f"✓ Created rolling statistics (windows: {windows}) for {len(columns)} columns")
    return df_roll

def create_momentum_indicators(df_returns):
    """Create momentum and trend indicators."""
    df_mom = df_returns.copy()
    return_cols = [col for col in df_returns.columns if col.endswith('_Return')]

    if len(return_cols) == 0:
        print("⚠ No columns ending with '_Return' found. Skipping momentum indicators.")
        return df_mom

    for col in return_cols:
        base = col.replace("_Return", "")
        df_mom[f'{base}_cumulative_return'] = (1 + df_mom[col]).cumprod() - 1

    print(f"✓ Created momentum indicators for {len(return_cols)} assets")
    return df_mom

# ============================================================================
# 5. MERGE AND COMBINE DATASETS
# ============================================================================

def merge_all_data(df_assets_norm, df_returns_proc, df_indicators):
    """Merge all processed datasets."""
    validate_columns(df_assets_norm, ['date'], "df_assets_norm")
    validate_columns(df_returns_proc, ['date'], "df_returns_proc")
    validate_columns(df_indicators, ['date'], "df_indicators")

    df_merged = df_assets_norm.merge(df_returns_proc, on='date', how='inner')
    df_merged = df_merged.merge(df_indicators, on='date', how='inner')
    df_merged = df_merged.sort_values('date').reset_index(drop=True)

    print(f"✓ Merged datasets: {len(df_merged)} rows × {len(df_merged.columns)} columns")
    return df_merged

# ============================================================================
# 6. CORRELATION ANALYSIS (MODIFIED: option to compute on normalized assets)
# ============================================================================

def compute_correlations(df, prefix='Asset_'):
    """Compute correlation matrix between columns with a given prefix."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    if len(cols) < 2:
        print(f"⚠ Not enough columns with prefix '{prefix}' to compute correlations.")
        return pd.DataFrame()

    corr_matrix = df[cols].corr()
    print(f"✓ Computed correlation matrix for prefix '{prefix}'")
    return corr_matrix

# ============================================================================
# 7. SAVE PROCESSED DATA (MODIFIED: handle DataFrame/Series; keep index for corr)
# ============================================================================

def save_processed_data(dfs_to_save, output_dir=None):
    """Save all processed dataframes to CSV."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

    safe_mkdir(output_dir)

    for filename, obj in dfs_to_save.items():
        filepath = os.path.join(output_dir, filename)

        if isinstance(obj, pd.DataFrame):
            # Keep index only for correlation matrix by default
            keep_index = filename.lower().startswith("correlation")
            obj.to_csv(filepath, index=keep_index)
            n_rows = len(obj)
        elif isinstance(obj, pd.Series):
            obj.to_csv(filepath, index=True)
            n_rows = len(obj)
        else:
            raise TypeError(f"Object for {filename} must be pandas DataFrame/Series. Got {type(obj)}")

        print(f"✓ Saved: {filename} ({n_rows} rows)")

    print(f"\n✓ All processed data saved to {output_dir}")

# ============================================================================
# MAIN EXECUTION (MODIFIED: CLI args)
# ============================================================================

def main(data_dir=None, output_dir=None):
    print("="*70)
    print("FINANCIAL DATA PROCESSING PIPELINE")
    print("="*70)
    print()

    # Load raw data
    print("STEP 1: Loading Raw Data")
    print("-"*70)
    df_assets, df_returns, df_indicators = load_raw_data(data_dir=data_dir)
    print()

    # Data quality
    print("STEP 2: Data Quality & Preprocessing")
    print("-"*70)
    df_assets = remove_duplicates(handle_missing_values(df_assets))
    df_returns = remove_duplicates(handle_missing_values(df_returns))
    df_indicators = remove_duplicates(handle_missing_values(df_indicators))
    print()

    # Normalization
    print("STEP 3: Normalization & Scaling")
    print("-"*70)
    df_assets_normalized = normalize_prices(df_assets)

    indicator_cols = ['Market_Index', 'Risk_Free_Rate', 'Volatility_Index',
                      'Sector_A', 'Sector_B', 'Sector_C']
    validate_columns(df_indicators, indicator_cols, "df_indicators")
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
    corr_matrix = compute_correlations(df_assets_normalized, prefix='Asset_')
    print()

    # Merge datasets
    print("STEP 6: Merging Datasets")
    print("-"*70)
    df_combined = merge_all_data(df_assets_features, df_returns_features, df_indicators_scaled)

    # Remove rows with NaN from lagged features
    before = len(df_combined)
    df_combined = df_combined.dropna()
    print(f"✓ After removing NaN from lagged features: {len(df_combined)} rows (dropped {before - len(df_combined)})")
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

    save_processed_data(dfs_to_save, output_dir=output_dir)
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
    parser = argparse.ArgumentParser(description="Financial data processing pipeline")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to raw data directory (default: proyecto/data/raw)")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to processed data directory (default: proyecto/data/processed)")
    args = parser.parse_args()

    main(data_dir=args.data_dir, output_dir=args.output_dir)