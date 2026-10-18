"""
Main script for Middle-Out hierarchical forecasting with 3-level hierarchy
"""

import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization import plot_forecasts, plot_all_oblasts, visualize_data
from middle_out.data_loader import MiddleOutDataLoader
from middle_out.forecaster import MiddleOutForecaster


def plot_hromadas_sample(hierarchy_df, fitted_df, oblast, n_hromadas=5):
    """Plot sample hromadas for a given oblast"""
    import matplotlib.pyplot as plt
    
    hromada_cols = [col for col in hierarchy_df.columns 
                   if col.startswith(oblast + '_')]
    
    if not hromada_cols:
        print(f"No hromadas found for {oblast}")
        return
    
    # Take first n_hromadas
    sample_hromadas = hromada_cols[:n_hromadas]
    
    n_cols = min(3, len(sample_hromadas))
    n_rows = int(np.ceil(len(sample_hromadas) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()
    
    for idx, hromada in enumerate(sample_hromadas):
        ax = axes[idx]
        
        hierarchy_df[hromada].plot(ax=ax, label='Observed', 
                                  color='green', linewidth=1, alpha=0.7)
        fitted_df[hromada].plot(ax=ax, label='Predicted', 
                                color='blue', linewidth=1, alpha=0.7)
        
        ax.set_title(f'{hromada}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Alerts', fontsize=9)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    
    # Hide empty subplots
    for idx in range(len(sample_hromadas), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Sample Hromadas for {oblast} (Middle-Out)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


def main():
    """Main function for Middle-Out forecasting with 3-level hierarchy"""
    print("=" * 70)
    print("MIDDLE-OUT HIERARCHICAL FORECASTING (3-Level)")
    print("=" * 70)
    
    # Load data with 3-level hierarchy
    data_loader = MiddleOutDataLoader(data_path='../data/official_data_uk.csv')
    hierarchy_df = data_loader.load_data()
    hierarchy = data_loader.create_hierarchy()
    
    # Visualize
    print("\nVisualizing data...")
    visualize_data(hierarchy_df, level='total')
    
    # Forecast with Middle-Out
    forecaster = MiddleOutForecaster(hierarchy_df, hierarchy)
    fitted_df, forecast_df, _ = forecaster.forecast(
        model='holt_winters',
        horizon=30
    )
    
    # Plot results
    print("\nPlotting hierarchical forecasts...")
    
    # 1. Show total
    print("\n" + "=" * 70)
    print("STEP 1: PLOTTING TOTAL UKRAINE")
    print("=" * 70)
    plot_forecasts(
        hierarchy_df, fitted_df, 
        'total', 'Middle-Out (Bottom-Up: Oblasts → Total)'
    )
    
    # 2. Show all oblasts
    print("\n" + "=" * 70)
    print("STEP 2: PLOTTING ALL OBLASTS (Middle Level)")
    print("=" * 70)
    oblast_cols = [col for col in hierarchy_df.columns 
                  if col != 'total' and '_' not in col]
    
    print(f"Oblast columns to plot: {len(oblast_cols)}")
    print(f"Sample oblast: {oblast_cols[0] if oblast_cols else 'None'}")
    
    # Use fitted_df which has same length as hierarchy_df
    # Pass full DataFrames - plot_all_oblasts will filter oblasts internally
    plot_all_oblasts(
        hierarchy_df, fitted_df, 
        'Middle-Out (Forecasted at Middle Level)'
    )
    
    # 3. Show sample hromadas
    if len(oblast_cols) > 0:
        sample_oblast = oblast_cols[0]  # First oblast
        print("\n" + "=" * 70)
        print(f"STEP 3: PLOTTING SAMPLE HROMADAS (Top-Down: {sample_oblast} → Hromadas)")
        print("=" * 70)
        
        # Create temporary fitted_df with hromada forecasts
        hromada_cols = [col for col in hierarchy_df.columns if '_' in col]
        hromada_fitted = hierarchy_df[hromada_cols].copy()
        hromada_fitted = pd.concat([hromada_fitted, forecast_df[hromada_cols]], axis=0)
        hromada_fitted = hromada_fitted.sort_index()
        
        plot_hromadas_sample(
            hierarchy_df, fitted_df,
            sample_oblast, n_hromadas=5
        )
    
    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()

