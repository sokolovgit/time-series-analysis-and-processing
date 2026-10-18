"""
Main script for hierarchical time series forecasting of Ukrainian air alerts

Allows selection of different forecasting methods:
- Optimal Combination (OLS)
- Bottom-Up
- Top-Down
- Middle-Out
"""

import numpy as np
import sys
from data_loader import AlertDataLoader
from visualization import plot_forecasts, plot_all_oblasts, visualize_data

# Import forecasters from respective modules
from optimal_combination.forecaster import OptimalCombinationForecaster
from bottom_up.forecaster import BottomUpForecaster
from top_down.forecaster import TopDownForecaster
from middle_out.forecaster import MiddleOutForecaster


def calculate_fit_quality(hierarchy_df, fitted_df):
    """Calculate and display fit quality metrics"""
    print("\n" + "=" * 70)
    print("FIT QUALITY BY LEVEL (After OLS Reconciliation)")
    print("=" * 70)
    
    # Total level
    actual = hierarchy_df['total'].values
    predicted = fitted_df['total'].values
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1))) * 100  # +1 to avoid division by zero
    
    print(f"\nTotal Ukraine:")
    print(f"  MAE: {mae:.1f} alerts")
    print(f"  MAPE: {mape:.2f}%")
    
    # Top oblasts
    print(f"\nTop 5 Oblasts by Alert Count:")
    print(f"{'Oblast':<30s} {'MAPE':>8s} {'Quality':>10s}")
    print("-" * 52)
    
    # Get all oblast columns (exclude 'total')
    oblasts = [col for col in hierarchy_df.columns if col != 'total']
    
    oblast_mape = []
    for oblast in oblasts[:10]:  # Check top 10
        actual_oblast = hierarchy_df[oblast].values
        predicted_oblast = fitted_df[oblast].values
        mape_oblast = np.mean(np.abs((actual_oblast - predicted_oblast) / (actual_oblast + 1))) * 100
        
        if hierarchy_df[oblast].sum() > 0:  # Only if has alerts
            oblast_mape.append((oblast, mape_oblast, hierarchy_df[oblast].sum()))
    
    # Sort by total alerts
    oblast_mape.sort(key=lambda x: x[2], reverse=True)
    
    for oblast, mape_val, total_alerts in oblast_mape[:5]:
        if mape_val < 30:
            quality = "Good ✓"
        elif mape_val < 50:
            quality = "Fair"
        else:
            quality = "Poor ✗"
        
        print(f"{oblast:<30s} {mape_val:7.2f}% {quality:>10s}")


def check_hierarchy_consistency(hierarchy_df, fitted_df):
    """Check if hierarchy consistency is maintained"""
    print("\n" + "=" * 70)
    print("HIERARCHY CONSISTENCY CHECK")
    print("=" * 70)
    print("OLS ensures oblasts sum to total...")
    
    oblasts = [col for col in hierarchy_df.columns if col != 'total']
    
    last_idx = -1
    total_pred = fitted_df['total'].iloc[last_idx]
    oblasts_sum = sum([fitted_df[oblast].iloc[last_idx] 
                      for oblast in oblasts])
    
    print(f"\nLast day:")
    print(f"  Predicted Total: {total_pred:,.1f}")
    print(f"  Sum of Oblasts: {oblasts_sum:,.1f}")
    print(f"  Difference: {abs(total_pred - oblasts_sum):.6f}")
    
    if abs(total_pred - oblasts_sum) < 0.01:
        print("\n✓ Perfect hierarchy consistency!")
    else:
        print("\n✗ Hierarchy not perfectly consistent")


def show_forecast_values(forecast_df, n_days=7):
    """Display future forecast values"""
    print("\n" + "=" * 70)
    print(f"FUTURE FORECAST VALUES (Next {n_days} days)")
    print("=" * 70)
    
    print(f"\nTotal Ukraine:")
    for i, (date, val) in enumerate(forecast_df['total'].head(n_days).items(), 1):
        print(f"  {date.strftime('%Y-%m-%d')}: {val:,.1f} alerts")


def main(method='optimal'):
    """
    Main function for Ukrainian alert forecasting
    
    Args:
        method: 'optimal', 'bottom_up', 'top_down', 'middle_out'
    """
    print("=" * 70)
    print("HIERARCHICAL TIME SERIES FORECASTING FOR UKRAINIAN AIR ALERTS")
    print("=" * 70)
    
    # Load data
    data_loader = AlertDataLoader(data_path='../data/official_data_uk.csv')
    hierarchy_df = data_loader.load_data()
    hierarchy = data_loader.create_hierarchy()
    
    # Visualize
    print("\nVisualizing data...")
    visualize_data(hierarchy_df, level='total')
    
    # Select forecaster based on method
    method_map = {
        'optimal': (OptimalCombinationForecaster, 'Optimal Combination (OLS)'),
        'bottom_up': (BottomUpForecaster, 'Bottom-Up'),
        'top_down': (TopDownForecaster, 'Top-Down (AHP)'),
        'middle_out': (MiddleOutForecaster, 'Middle-Out')
    }
    
    if method not in method_map:
        print(f"Unknown method: {method}")
        print(f"Available methods: {', '.join(method_map.keys())}")
        return
    
    ForecasterClass, method_name = method_map[method]
    
    print("\n" + "=" * 70)
    print(f"HIERARCHICAL RECONCILIATION METHOD: {method_name.upper()}")
    print("=" * 70)
    
    # Create forecaster
    if method == 'top_down':
        forecaster = ForecasterClass(hierarchy_df, hierarchy, method='AHP')
    else:
        forecaster = ForecasterClass(hierarchy_df, hierarchy)
    
    # Forecast
    fitted_df, forecast_df, _ = forecaster.forecast(
        model='holt_winters',
        horizon=30  # 30 days ahead
    )
    
    # Plot results - FIRST show all oblasts, THEN total
    print("\nPlotting hierarchical forecasts...")
    
    # 1. Show all oblasts first
    print("\n" + "=" * 70)
    print("STEP 1: PLOTTING ALL OBLASTS")
    print("=" * 70)
    plot_all_oblasts(
        hierarchy_df, fitted_df, 
        method_name
    )
    
    # 2. Then show total
    print("\n" + "=" * 70)
    print("STEP 2: PLOTTING TOTAL UKRAINE")
    print("=" * 70)
    plot_forecasts(
        hierarchy_df, fitted_df, 
        'total', method_name
    )
    
    # Calculate and display fit quality
    calculate_fit_quality(hierarchy_df, fitted_df)
    
    # Check hierarchy consistency
    check_hierarchy_consistency(hierarchy_df, fitted_df)
    
    # Show forecast values
    show_forecast_values(forecast_df, n_days=7)
    
    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    # Allow method selection via command line argument
    if len(sys.argv) > 1:
        method = sys.argv[1].lower()
    else:
        # Default to optimal
        method = 'optimal'
        print("\nUsage: python main.py [method]")
        print("Available methods: optimal, bottom_up, top_down, middle_out")
        print(f"Using default: {method}\n")
    
    main(method=method)

