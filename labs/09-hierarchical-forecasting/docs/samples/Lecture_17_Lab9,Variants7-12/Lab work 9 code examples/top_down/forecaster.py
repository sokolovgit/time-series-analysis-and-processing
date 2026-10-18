"""
Top-Down Hierarchical Forecasting

Forecasts at the top level (total), then distributes downwards to oblasts using proportions.
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


class TopDownForecaster:
    """
    Top-Down hierarchical forecasting for Ukrainian air alerts
    
    Method: Forecast at top level (total), then distribute to oblasts using proportions
    Hierarchy: Total Ukraine → Oblasts
    """
    
    def __init__(self, hierarchy_df, hierarchy, method='AHP'):
        """
        Args:
            hierarchy_df: DataFrame with time series
            hierarchy: Hierarchy dictionary
            method: 'AHP' (Average Historical Proportions), 
                   'PHA' (Proportions of Historical Averages),
                   'FP' (Forecasted Proportions)
        """
        self.hierarchy_df = hierarchy_df
        self.hierarchy = hierarchy
        self.method = method
        
    def forecast(self, model='holt_winters', horizon=30):
        """
        Perform top-down forecasting
        
        Args:
            model: 'holt_winters', 'auto_arima', 'prophet', 'sarimax'
            horizon: number of days to forecast ahead
        """
        if not HTS_AVAILABLE:
            raise ImportError("scikit-hts required for forecasting")
        
        method_names = {
            'AHP': 'Average Historical Proportions',
            'PHA': 'Proportions of Historical Averages',
            'FP': 'Forecasted Proportions'
        }
        
        print("\n" + "=" * 70)
        print(f"TOP-DOWN HIERARCHICAL FORECASTING ({self.method})")
        print("=" * 70)
        print(f"Method: Forecast at top level, then distribute using {method_names.get(self.method, self.method)}")
        print("Advantages: Stable top-level forecasts")
        print("Disadvantages: Loss of lower-level detail")
        print(f"\nModel: {model}")
        print(f"Horizon: {horizon} days ahead")
        print(f"Training on: {len(self.hierarchy_df)} historical days")
        
        # Create model with Top-Down reconciliation
        hts_model = HTSRegressor(
            model=model,
            revision_method=self.method,  # AHP, PHA, or FP
            n_jobs=0
        )
        
        # Fit model on ALL data
        print("\nFitting model on full dataset...")
        try:
            hts_model = hts_model.fit(
                df=self.hierarchy_df,
                nodes=self.hierarchy
            )
        except Exception as e:
            print(f"Error fitting model: {e}")
            print(f"Trying alternative method...")
            # Fallback: try AHP if other methods fail
            if self.method != 'AHP':
                hts_model = HTSRegressor(
                    model=model,
                    revision_method='AHP',
                    n_jobs=0
                )
                hts_model = hts_model.fit(
                    df=self.hierarchy_df,
                    nodes=self.hierarchy
                )
                self.method = 'AHP'
            else:
                raise
        
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

