"""
Financial Asset Price Simulation Script
Generates correlated financial time series data for forecasting analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
start_date = '2024-01-01'
num_days = 44
num_assets = 4

# Create date range
dates = pd.date_range(start=start_date, periods=num_days, freq='D')

# ============================================================================
# 1. GENERATE CORRELATED ASSET PRICES
# ============================================================================

def generate_correlated_prices(dates, initial_prices, correlations, volatilities):
    """
    Generate correlated asset prices using geometric Brownian motion
    """
    n_days = len(dates)
    n_assets = len(initial_prices)
    
    # Generate correlated returns
    mean_return = 0.001
    dt = 1/252  # daily returns
    
    # Cholesky decomposition for correlation matrix
    L = np.linalg.cholesky(correlations)
    
    # Generate independent normal returns
    independent_returns = np.random.normal(0, 1, (n_days, n_assets))
    
    # Apply correlation structure
    correlated_returns = independent_returns @ L.T
    
    # Scale by volatilities
    scaled_returns = correlated_returns * volatilities * np.sqrt(dt)
    scaled_returns += mean_return * dt
    
    # Generate price paths
    prices = np.zeros((n_days, n_assets))
    prices[0] = initial_prices
    
    for i in range(1, n_days):
        prices[i] = prices[i-1] * (1 + scaled_returns[i])
    
    return prices

# Initial conditions
initial_prices = np.array([100.0, 50.0, 150.0, 80.0])
asset_names = ['Asset_A', 'Asset_B', 'Asset_C', 'Asset_D']

# Correlation matrix (positive correlations reflecting market co-movement)
correlation_matrix = np.array([
    [1.00, 0.75, 0.68, 0.72],
    [0.75, 1.00, 0.62, 0.68],
    [0.68, 0.62, 1.00, 0.65],
    [0.72, 0.68, 0.65, 1.00]
])

# Volatilities (annualized)
volatilities = np.array([0.18, 0.16, 0.15, 0.17])

# Generate prices
prices = generate_correlated_prices(dates, initial_prices, correlation_matrix, volatilities)

# Create DataFrame
df_assets = pd.DataFrame(prices, columns=asset_names)
df_assets.insert(0, 'date', dates)
df_assets['date'] = df_assets['date'].dt.strftime('%Y-%m-%d')

# Round to 2 decimal places
for col in asset_names:
    df_assets[col] = df_assets[col].round(2)

# ============================================================================
# 2. GENERATE RETURNS AND VOLUME DATA
# ============================================================================

# Calculate returns
returns = df_assets[asset_names].pct_change().fillna(0.0)
returns = returns.round(4)

# Generate trading volumes (in millions)
base_volumes = np.array([5000000, 3000000, 2000000, 4000000])
volume_volatility = 0.05
volumes = np.random.normal(base_volumes, base_volumes * volume_volatility, (num_days, num_assets))
volumes = np.abs(volumes).astype(int)

# Create returns DataFrame
df_returns = pd.DataFrame({
    'date': dates.strftime('%Y-%m-%d'),
    'Asset_A_Return': returns['Asset_A'],
    'Asset_B_Return': returns['Asset_B'],
    'Asset_C_Return': returns['Asset_C'],
    'Asset_D_Return': returns['Asset_D'],
    'Volume_A': volumes[:, 0],
    'Volume_B': volumes[:, 1],
    'Volume_C': volumes[:, 2],
    'Volume_D': volumes[:, 3]
})

# ============================================================================
# 3. GENERATE MARKET INDICATORS
# ============================================================================

# Construct market index (weighted average of assets)
market_weights = np.array([0.3, 0.2, 0.3, 0.2])
market_index = (df_assets[asset_names].values @ market_weights)
market_index_normalized = (market_index / market_index[0]) * 1000

# Risk-free rate (slightly increasing)
risk_free_rate = 0.045 + np.linspace(0, 0.002, num_days)

# Volatility index (inverse to market performance)
market_returns = np.diff(market_index_normalized) / market_index_normalized[:-1]
market_returns = np.insert(market_returns, 0, 0)
volatility_index = 15 - (market_returns * 100)
volatility_index = np.clip(volatility_index, 8, 16)

# Sector indices
sector_a = df_assets['Asset_A'].values / df_assets['Asset_A'].iloc[0] * 25
sector_b = df_assets['Asset_B'].values / df_assets['Asset_B'].iloc[0] * 15.5
sector_c = df_assets['Asset_C'].values / df_assets['Asset_C'].iloc[0] * 20.3

# Dynamic correlations (increasing over time)
base_corr_ab = 0.85
base_corr_ac = 0.78
base_corr_bc = 0.72

corr_ab = base_corr_ab + (np.arange(num_days) / num_days) * 0.28
corr_ac = base_corr_ac + (np.arange(num_days) / num_days) * 0.23
corr_bc = base_corr_bc + (np.arange(num_days) / num_days) * 0.28

# Create market indicators DataFrame
df_indicators = pd.DataFrame({
    'date': dates.strftime('%Y-%m-%d'),
    'Market_Index': market_index_normalized.round(2),
    'Risk_Free_Rate': risk_free_rate.round(4),
    'Volatility_Index': volatility_index.round(1),
    'Sector_A': sector_a.round(1),
    'Sector_B': sector_b.round(1),
    'Sector_C': sector_c.round(1),
    'Correlation_AB': corr_ab.round(2),
    'Correlation_AC': corr_ac.round(2),
    'Correlation_BC': corr_bc.round(2)
})

# ============================================================================
# 4. SAVE TO CSV FILES
# ============================================================================

# Create data/processed directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(output_dir, exist_ok=True)

# Save files
file_paths = {
    'financial_assets_main.csv': df_assets,
    'financial_returns_volume.csv': df_returns,
    'market_indicators.csv': df_indicators
}

for filename, df in file_paths.items():
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"✓ Saved: {filepath}")

print(f"\n✓ Successfully generated {len(file_paths)} simulation files")
print(f"  - {num_days} days of data")
print(f"  - {num_assets} correlated assets")
print(f"  - Correlation matrix shape: {correlation_matrix.shape}")
