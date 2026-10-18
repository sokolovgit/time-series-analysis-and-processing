"""
Bottom-Up Hierarchical Forecasting

Forecasts only at the lowest level (oblasts), then aggregates upwards to total.
"""

import pandas as pd
import numpy as np
import warnings
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

try:
    from hts import HTSRegressor
    HTS_AVAILABLE = True
except ImportError:
    print("Error: scikit-hts not installed")
    print("Install with: pip install scikit-hts")
    HTS_AVAILABLE = False


class BottomUpForecaster:
    """
    Bottom-Up hierarchical forecasting for Ukrainian air alerts
    
    Method: Forecast only at oblast level, then sum to get total
    Hierarchy: Oblasts → Total Ukraine
    """
    
    def __init__(self, hierarchy_df, hierarchy):
        self.hierarchy_df = hierarchy_df
        self.hierarchy = hierarchy
        
    def forecast(self, model='holt_winters', horizon=30):
        """
        Perform bottom-up forecasting
        
        Args:
            model: 'holt_winters', 'auto_arima', 'prophet', 'sarimax'
            horizon: number of days to forecast ahead
        """
        if not HTS_AVAILABLE:
            raise ImportError("scikit-hts required for forecasting")
        
        print("\n" + "=" * 70)
        print("BOTTOM-UP HIERARCHICAL FORECASTING")
        print("=" * 70)
        print("Method: Forecast only at lowest level (oblasts), then aggregate upwards")
        print("Advantages: Preserves detail, automatic coherence")
        print("Disadvantages: Error accumulation from lower levels")
        print(f"\nModel: {model}")
        print(f"Horizon: {horizon} days ahead")
        print(f"Training on: {len(self.hierarchy_df)} historical days")
        
        # Create model with Bottom-Up reconciliation
        hts_model = HTSRegressor(
            model=model,
            revision_method='BU',  # Bottom-Up
            n_jobs=0
        )
        
        # Fit model on ALL data
        print("\nFitting model on full dataset...")
        hts_model = hts_model.fit(
            df=self.hierarchy_df,
            nodes=self.hierarchy
        )
        
        # Predict
        print(f"Generating {horizon}-step ahead forecasts...")
        predictions = hts_model.predict(steps_ahead=horizon)
        
        # Split into fitted and forecast
        n_hist = len(self.hierarchy_df)
        fitted_df = predictions.iloc[:n_hist].copy()
        fitted_df.index = self.hierarchy_df.index
        
        forecast_df = predictions.iloc[n_hist:].copy()
        # Create future dates
        last_date = self.hierarchy_df.index[-1]
        forecast_dates = pd.date_range(
            start=last_date,
            periods=horizon+1,
            freq='D'
        )[1:]
        forecast_df.index = forecast_dates
        
        print(f"\nFitted values: {len(fitted_df)} days")
        print(f"Future forecasts: {len(forecast_df)} days")
        
        return fitted_df, forecast_df, hts_model

