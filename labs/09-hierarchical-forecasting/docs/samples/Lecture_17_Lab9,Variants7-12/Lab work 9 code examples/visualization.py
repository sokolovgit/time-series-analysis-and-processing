"""
Visualization functions for hierarchical forecasting results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_forecasts(hierarchy_df, fitted_df, series_name='total', model_name=''):
    """Plot historical data with fitted values (hierarchical reconciliation)"""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    if series_name not in hierarchy_df.columns:
        print(f"Series {series_name} not found")
        return
    
    # Plot observed historical data
    hierarchy_df[series_name].plot(
        ax=ax, label='Observed', 
        color='green', linewidth=1.5, alpha=0.8
    )
    
    # Plot fitted values (in-sample predictions after reconciliation)
    fitted_df[series_name].plot(
        ax=ax, label='Predicted', 
        color='blue', linewidth=1.5, alpha=0.8
    )
    
    ax.set_title(f'{series_name} - {model_name}', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Alerts', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_all_oblasts(hierarchy_df, fitted_df, model_name=''):
    """Plot forecasts for all oblasts in a grid layout"""
    # Filter only oblasts: exclude 'total' and columns with '_' (which are hromadas)
    oblasts = [col for col in hierarchy_df.columns 
              if col != 'total' and '_' not in col]
    
    if len(oblasts) == 0:
        print("No oblasts found")
        return
    
    print(f"Found {len(oblasts)} oblasts to plot")
    
    # Calculate grid size (approximately square)
    n_oblasts = len(oblasts)
    n_cols = int(np.ceil(np.sqrt(n_oblasts)))
    n_rows = int(np.ceil(n_oblasts / n_cols))
    
    print(f"\nPlotting {n_oblasts} oblasts in {n_rows}x{n_cols} grid...")
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    
    # Flatten axes array to 1D
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()
    
    for idx, oblast in enumerate(oblasts):
        ax = axes[idx]
        
        # Check if data exists
        if oblast not in hierarchy_df.columns:
            print(f"Warning: {oblast} not found in hierarchy_df")
            continue
        if oblast not in fitted_df.columns:
            print(f"Warning: {oblast} not found in fitted_df")
            continue
        
        # Get data
        observed_series = hierarchy_df[oblast]
        predicted_series = fitted_df[oblast]
        
        # Ensure indices match
        if not observed_series.index.equals(predicted_series.index):
            print(f"Warning: Index mismatch for {oblast}")
            # Align indices
            common_index = observed_series.index.intersection(predicted_series.index)
            observed_series = observed_series.loc[common_index]
            predicted_series = predicted_series.loc[common_index]
        
        # Plot observed using ax.plot for better control
        ax.plot(observed_series.index, observed_series.values, 
                label='Observed', color='green', linewidth=1, alpha=0.7)
        
        # Plot predicted using ax.plot
        ax.plot(predicted_series.index, predicted_series.values, 
                label='Predicted', color='blue', linewidth=1, alpha=0.7)
        
        # Calculate MAPE for this oblast
        actual = observed_series.values
        predicted = predicted_series.values
        mape = np.mean(np.abs((actual - predicted) / (actual + 1))) * 100
        
        ax.set_title(f'{oblast}\nMAPE: {mape:.1f}%', 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Alerts', fontsize=9)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    
    # Hide empty subplots
    for idx in range(n_oblasts, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'All Oblasts - {model_name}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


def visualize_data(hierarchy_df, level='total', oblast=None):
    """Visualize time series at different levels"""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    if level == 'total':
        hierarchy_df['total'].plot(ax=ax, title='Total Ukraine - Daily Alert Count', linewidth=1.5)
        ax.set_ylabel('Number of Alerts')
        ax.set_xlabel('Date')
    
    elif level == 'oblast' and oblast:
        if oblast in hierarchy_df.columns:
            hierarchy_df[oblast].plot(ax=ax, title=f'{oblast} - Daily Alert Count', linewidth=1.5)
            ax.set_ylabel('Number of Alerts')
            ax.set_xlabel('Date')
        else:
            print(f"Oblast {oblast} not found")
            return
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

