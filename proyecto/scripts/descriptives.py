"""
Descriptive Analysis Script
Generates statistical summaries, visualizations, and tables from processed financial data
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ============================================================================
# 1. LOAD PROCESSED DATA
# ============================================================================
X = "HOLA" # Este es el cambio que voy a generar para tener el conflicto
def load_processed_data():
    """Load processed data files"""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    
    df_combined = pd.read_csv(os.path.join(data_dir, 'combined_data.csv'))
    df_corr = pd.read_csv(os.path.join(data_dir, 'correlation_matrix.csv'))
    df_assets = pd.read_csv(os.path.join(data_dir, 'assets_normalized.csv'))
    
    df_combined['date'] = pd.to_datetime(df_combined['date'])
    df_assets['date'] = pd.to_datetime(df_assets['date'])
    
    print("✓ Loaded processed data")
    
    return df_combined, df_corr, df_assets

# ============================================================================
# 2. GENERATE DESCRIPTIVE STATISTICS
# ============================================================================

def generate_descriptive_stats(df):
    """Generate comprehensive descriptive statistics"""
    asset_cols = [col for col in df.columns if col.startswith('Asset_') and not col.endswith('_Return')]
    
    # Remove normalized suffix if exists
    asset_cols = [col for col in asset_cols if col in df.columns and 'lag' not in col and 'rolling' not in col]
    
    stats = {}
    for col in asset_cols:
        if col in df.columns:
            stats[col] = {
                'Mean': df[col].mean(),
                'Std Dev': df[col].std(),
                'Min': df[col].min(),
                'Max': df[col].max(),
                'Median': df[col].median(),
                'Skewness': df[col].skew(),
                'Kurtosis': df[col].kurtosis()
            }
    
    stats_df = pd.DataFrame(stats).T
    
    print("✓ Generated descriptive statistics")
    
    return stats_df

def generate_return_stats(df_combined):
    """Generate return statistics"""
    return_cols = [col for col in df_combined.columns if col.endswith('_Return')]
    
    return_stats = {}
    for col in return_cols:
        if col in df_combined.columns:
            return_stats[col] = {
                'Mean Return': df_combined[col].mean(),
                'Std Dev': df_combined[col].std(),
                'Min Return': df_combined[col].min(),
                'Max Return': df_combined[col].max(),
                'Cumulative Return': (1 + df_combined[col]).prod() - 1
            }
    
    return_stats_df = pd.DataFrame(return_stats).T
    
    print("✓ Generated return statistics")
    
    return return_stats_df

def generate_correlation_analysis(df_corr):
    """Generate correlation analysis"""
    print("✓ Loaded correlation analysis")
    return df_corr

# ============================================================================
# 3. GENERATE TIME SERIES SUMMARIES
# ============================================================================

def generate_timeseries_summary(df_assets, df_combined):
    """Generate time series summaries"""
    asset_cols = [col for col in df_assets.columns if col.startswith('Asset_')]
    
    summary_data = []
    
    for col in asset_cols:
        if col in df_assets.columns:
            summary_data.append({
                'Asset': col,
                'Start Price': df_assets[col].iloc[0],
                'End Price': df_assets[col].iloc[-1],
                'Price Change': df_assets[col].iloc[-1] - df_assets[col].iloc[0],
                'Price Change %': ((df_assets[col].iloc[-1] - df_assets[col].iloc[0]) / df_assets[col].iloc[0] * 100),
                'Days of Data': len(df_assets)
            })
    
    ts_summary_df = pd.DataFrame(summary_data)
    
    print("✓ Generated time series summary")
    
    return ts_summary_df

# ============================================================================
# 4. SAVE TABLES
# ============================================================================

def save_tables(tables_dict):
    """Save tables to results/tables"""
    tables_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    for name, df in tables_dict.items():
        filepath = os.path.join(tables_dir, f'{name}.csv')
        df.to_csv(filepath)
        print(f"✓ Saved table: {name}.csv")
    
    # Also save as formatted text versions
    for name, df in tables_dict.items():
        filepath = os.path.join(tables_dir, f'{name}_formatted.txt')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"{name.replace('_', ' ').upper()}\n")
            f.write(f"{'='*80}\n\n")
            f.write(df.to_string())
            f.write(f"\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"✓ Saved formatted table: {name}_formatted.txt")

# ============================================================================
# 5. GENERATE VISUALIZATIONS (TEXT-BASED)
# ============================================================================

def create_ascii_chart(data, height=10, width=40, title="Chart"):
    """Create text-based simple ASCII chart"""
    if len(data) == 0:
        return ""
    
    data_np = np.array(data)
    min_val = data_np.min()
    max_val = data_np.max()
    
    # Normalize data
    if max_val == min_val:
        normalized = [height // 2] * len(data)
    else:
        normalized = ((data_np - min_val) / (max_val - min_val) * (height - 1)).astype(int)
    
    # Create chart
    chart = []
    chart.append(f"\n{'='*width}")
    chart.append(f"{title.center(width)}")
    chart.append(f"{'='*width}\n")
    
    for row in range(height - 1, -1, -1):
        line = ""
        for val in normalized:
            if val >= row:
                line += "█ "
            else:
                line += "  "
        chart.append(line)
    
    chart.append("-" * width)
    chart.append(f"Min: {min_val:.2f} | Max: {max_val:.2f}\n")
    
    return "\n".join(chart)

def generate_visualization_summaries(df_assets):
    """Generate text-based visualization summaries"""
    asset_cols = [col for col in df_assets.columns if col.startswith('Asset_')]
    
    viz_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    os.makedirs(viz_dir, exist_ok=True)
    
    for col in asset_cols:
        if col in df_assets.columns:
            chart_text = create_ascii_chart(
                df_assets[col].values,
                height=15,
                width=60,
                title=f"{col} Price Over Time"
            )
            
            # Save to file
            filepath = os.path.join(viz_dir, f'{col}_chart.txt')
            with open(filepath, 'w') as f:
                f.write(chart_text)
            
            print(f"✓ Saved visualization: {col}_chart.txt")
    
    # Correlation heatmap (text-based)
    heatmap_filepath = os.path.join(viz_dir, 'correlation_heatmap.txt')
    with open(heatmap_filepath, 'w') as f:
        f.write("CORRELATION MATRIX (Text-based)\n")
        f.write("="*60 + "\n\n")
        
        asset_cols_simple = [col for col in df_assets.columns if col.startswith('Asset_')]
        corr_matrix = df_assets[asset_cols_simple].corr()
        
        # Header
        f.write("         ")
        for col in asset_cols_simple:
            f.write(f"{col:>12}")
        f.write("\n")
        
        # Rows
        for idx_label, row in corr_matrix.iterrows():
            f.write(f"{idx_label:>8} ")
            for val in row:
                # Color-coded representation
                if abs(val) >= 0.8:
                    marker = "████"
                elif abs(val) >= 0.6:
                    marker = "███"
                elif abs(val) >= 0.4:
                    marker = "██"
                else:
                    marker = "█"
                f.write(f"{val:>8.3f}{marker:>3} ")
            f.write("\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Legend: ████ = Strong correlation (>0.8)\n")
        f.write("        ███  = Moderate correlation (0.6-0.8)\n")
        f.write("        ██   = Weak correlation (0.4-0.6)\n")
        f.write("        █    = Weak correlation (<0.4)\n")
    
    print("✓ Saved visualization: correlation_heatmap.txt")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("DESCRIPTIVE ANALYSIS PIPELINE")
    print("="*70)
    print()
    
    # Load data
    print("STEP 1: Loading Data")
    print("-"*70)
    df_combined, df_corr, df_assets = load_processed_data()
    print()
    
    # Generate statistics
    print("STEP 2: Generating Descriptive Statistics")
    print("-"*70)
    desc_stats = generate_descriptive_stats(df_assets)
    return_stats = generate_return_stats(df_combined)
    ts_summary = generate_timeseries_summary(df_assets, df_combined)
    corr_analysis = generate_correlation_analysis(df_corr)
    print()
    
    # Save tables
    print("STEP 3: Saving Tables")
    print("-"*70)
    tables_to_save = {
        'descriptive_statistics': desc_stats,
        'return_statistics': return_stats,
        'timeseries_summary': ts_summary,
        'correlation_matrix': corr_analysis
    }
    save_tables(tables_to_save)
    print()
    
    # Generate visualizations
    print("STEP 4: Generating Visualizations")
    print("-"*70)
    generate_visualization_summaries(df_assets)
    print()
    
    # Print summaries
    print("STEP 5: Summary Output")
    print("-"*70)
    print("\nDescriptive Statistics of Assets:")
    print(desc_stats.round(4))
    print("\n\nReturn Statistics:")
    print(return_stats.round(6))
    print("\n\nTime Series Summary:")
    print(ts_summary.round(4))
    
    print("\n" + "="*70)
    print("✓ DESCRIPTIVE ANALYSIS COMPLETE")
    print("✓ All tables saved to: results/tables/")
    print("✓ All figures saved to: results/figures/")
    print("="*70)

if __name__ == "__main__":
    main()

# CONFLICT_BLOCK: VERSION_A

# CONFLICT_BLOCK: VERSION_A

# CONFLICT_BLOCK: VERSION_A

# CONFLICT_BLOCK: VERSION_A
