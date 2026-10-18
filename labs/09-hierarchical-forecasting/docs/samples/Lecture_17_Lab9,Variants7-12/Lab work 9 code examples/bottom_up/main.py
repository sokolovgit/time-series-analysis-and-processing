"""
Main script for Bottom-Up hierarchical forecasting
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import AlertDataLoader
from visualization import plot_forecasts, plot_all_oblasts, visualize_data
from bottom_up.forecaster import BottomUpForecaster


def main():
    """Main function for Bottom-Up forecasting"""
    print("=" * 70)
    print("BOTTOM-UP HIERARCHICAL FORECASTING")
    print("=" * 70)
    
    # Load data
    data_loader = AlertDataLoader(data_path='../data/official_data_uk.csv')
    hierarchy_df = data_loader.load_data()
    hierarchy = data_loader.create_hierarchy()
    
    # Visualize
    print("\nVisualizing data...")
    visualize_data(hierarchy_df, level='total')
    
    # Forecast with Bottom-Up
    forecaster = BottomUpForecaster(hierarchy_df, hierarchy)
    fitted_df, forecast_df, _ = forecaster.forecast(
        model='holt_winters',
        horizon=30
    )
    
    # Plot results
    print("\nPlotting hierarchical forecasts...")
    
    # Show all oblasts first
    print("\n" + "=" * 70)
    print("STEP 1: PLOTTING ALL OBLASTS")
    print("=" * 70)
    plot_all_oblasts(
        hierarchy_df, fitted_df, 
        'Bottom-Up'
    )
    
    # Then show total
    print("\n" + "=" * 70)
    print("STEP 2: PLOTTING TOTAL UKRAINE")
    print("=" * 70)
    plot_forecasts(
        hierarchy_df, fitted_df, 
        'total', 'Bottom-Up'
    )
    
    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()

