"""
Middle-Out Hierarchical Forecasting

Forecasts at an intermediate level (oblasts), then:
- Bottom-up: Sum oblasts to get total (higher level)
- Top-down: Distribute oblasts to hromadas (lower level)
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
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HTS_AVAILABLE = True
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Error: scikit-hts or statsmodels not installed")
    HTS_AVAILABLE = False
    STATSMODELS_AVAILABLE = False


class MiddleOutForecaster:
    """
    Middle-Out hierarchical forecasting for Ukrainian air alerts
    
    4-Level Hierarchy: Total → Oblast → Raion → Hromada
    
    Method:
    1. Forecast at middle level (raions)
    2. Bottom-up: Sum raions → oblasts → total
    3. Top-down: Distribute raions → hromadas using historical proportions
    """
    
    def __init__(self, hierarchy_df, hierarchy):
        self.hierarchy_df = hierarchy_df
        self.hierarchy = hierarchy
        
    def _simple_forecast(self, series, horizon, model='holt_winters'):
        """Simple forecasting for a single time series"""
        if not STATSMODELS_AVAILABLE:
            # Fallback: use mean
            mean_val = series.mean()
            return np.full(horizon, mean_val)
        
        try:
            if model == 'holt_winters':
                # Try full Holt-Winters with seasonality
                try:
                    model_fit = ExponentialSmoothing(
                        series.values,
                        seasonal='add',
                        trend='add',
                        seasonal_periods=7  # Weekly seasonality for daily data
                    ).fit()
                    forecast = model_fit.forecast(steps=horizon)
                except:
                    # Fallback to Holt's method
                    try:
                        model_fit = ExponentialSmoothing(
                            series.values,
                            trend='add'
                        ).fit()
                        forecast = model_fit.forecast(steps=horizon)
                    except:
                        # Fallback to simple exponential smoothing
                        model_fit = ExponentialSmoothing(series.values).fit()
                        forecast = model_fit.forecast(steps=horizon)
                
                return np.maximum(forecast, 0)  # Ensure non-negative
            else:
                # Simple mean fallback
                mean_val = series.mean()
                return np.full(horizon, mean_val)
        except Exception as e:
            # Final fallback
            mean_val = series.mean()
            return np.full(horizon, mean_val)
    
    def forecast(self, model='holt_winters', horizon=30):
        """
        Perform middle-out forecasting with 3-level hierarchy
        
        Args:
            model: 'holt_winters', 'auto_arima', 'prophet', 'sarimax'
            horizon: number of days to forecast ahead
        """
        print("\n" + "=" * 70)
        print("MIDDLE-OUT HIERARCHICAL FORECASTING (4-Level)")
        print("=" * 70)
        print("Method:")
        print("  1. Forecast at middle level (raions)")
        print("  2. Bottom-up: Sum raions → oblasts → total")
        print("  3. Top-down: Distribute raions → hromadas")
        print("Advantages: Balance between detail and stability")
        print("Disadvantages: Requires choosing appropriate middle level")
        print(f"\nModel: {model}")
        print(f"Horizon: {horizon} days ahead")
        print(f"Training on: {len(self.hierarchy_df)} historical days")
        
        # Get levels
        oblast_cols = [col for col in self.hierarchy_df.columns 
                      if col != 'total' and '_' not in col]
        raion_cols = [col for col in self.hierarchy_df.columns 
                     if '_' in col and col.count('_') == 1]  # oblast_raion format
        hromada_cols = [col for col in self.hierarchy_df.columns 
                       if '_' in col and col.count('_') == 2]  # oblast_raion_hromada format
        
        print(f"\nLevels:")
        print(f"  - Top (Total): 1 series")
        print(f"  - Level 2 (Oblasts): {len(oblast_cols)} series")
        print(f"  - Middle (Raions): {len(raion_cols)} series")
        print(f"  - Bottom (Hromadas): {len(hromada_cols)} series")
        
        # STEP 1: Forecast at middle level (raions) using HTSRegressor
        print("\n" + "-" * 70)
        print("STEP 1: Forecasting at middle level (raions) using HTSRegressor...")
        print("-" * 70)
        
        if not HTS_AVAILABLE:
            raise ImportError("scikit-hts required for forecasting")
        
        # Create a 3-level hierarchy for raions: raions → oblasts → total
        # This allows us to use HTSRegressor for raion forecasting
        raion_hierarchy = {
            'total': oblast_cols
        }
        
        # For each oblast, find its raions
        for oblast in oblast_cols:
            oblast_raions = [r for r in raion_cols if r.startswith(oblast + '_')]
            if oblast_raions:
                raion_hierarchy[oblast] = oblast_raions
        
        # Create DataFrame with raions, oblasts, and total
        raion_df = self.hierarchy_df[['total'] + oblast_cols + raion_cols].copy()
        
        # Use HTSRegressor with Bottom-Up to forecast raions
        # BU will forecast raions, aggregate to oblasts, then to total
        hts_model = HTSRegressor(
            model=model,
            revision_method='BU',  # Bottom-up for raions
            n_jobs=0
        )
        
        print("Fitting HTSRegressor model for raions...")
        hts_model = hts_model.fit(
            df=raion_df,
            nodes=raion_hierarchy
        )
        
        print(f"Generating {horizon}-step ahead forecasts for raions...")
        raion_predictions = hts_model.predict(steps_ahead=horizon)
        
        # Extract raion forecasts
        n_hist = len(raion_df)
        raion_forecasts_dict = {}
        for raion in raion_cols:
            # Get forecast part (last horizon rows)
            raion_forecasts_dict[raion] = raion_predictions[raion].iloc[-horizon:].values
        
        # Extract oblast forecasts (from HTSRegressor aggregation)
        oblast_forecasts_dict = {}
        for oblast in oblast_cols:
            oblast_forecasts_dict[oblast] = raion_predictions[oblast].iloc[-horizon:].values
        
        # Create forecast dates
        last_date = self.hierarchy_df.index[-1]
        forecast_dates = pd.date_range(
            start=last_date,
            periods=horizon+1,
            freq='D'
        )[1:]
        
        # STEP 2: Bottom-up - Raions → Oblasts → Total
        print("\n" + "-" * 70)
        print("STEP 2: Bottom-up aggregation (raions → oblasts → total)...")
        print("-" * 70)
        
        # Total forecast is already calculated by HTSRegressor (BU method)
        total_forecast = raion_predictions['total'].iloc[-horizon:].values
        
        # Verify: sum of oblasts should equal total
        oblast_sum = sum([oblast_forecasts_dict[oblast] for oblast in oblast_cols])
        if not np.allclose(oblast_sum, total_forecast, atol=0.01):
            print("Warning: Oblasts sum doesn't match total, using sum instead")
            total_forecast = oblast_sum
        
        # STEP 3: Top-down - Distribute raions to hromadas
        print("\n" + "-" * 70)
        print("STEP 3: Top-down distribution (raions → hromadas)...")
        print("-" * 70)
        print("Note: Middle-out applies only to raions that have hromadas")
        print("      Raions without hromadas use direct forecasts (no top-down)")
        
        hromada_forecasts = {}
        raions_with_hromadas = []
        raions_without_hromadas = []
        
        # Identify raions that have hromadas
        print("\nIdentifying raions with hromadas...")
        for raion in raion_cols:
            # Find hromadas belonging to this raion
            raion_hromadas = [h for h in hromada_cols if h.startswith(raion + '_')]
            
            if not raion_hromadas:
                raions_without_hromadas.append(raion)
                continue
            
            # Check if hromadas have real data (non-zero alerts)
            hromadas_with_data = 0
            for hromada in raion_hromadas:
                if hromada in self.hierarchy_df.columns:
                    hromada_sum = self.hierarchy_df[hromada].sum()
                    if hromada_sum > 0:
                        hromadas_with_data += 1
            
            # A raion is considered to have hromadas if:
            # - It has more than 1 hromada with real data
            if len(raion_hromadas) > 1 and hromadas_with_data > 1:
                raions_with_hromadas.append(raion)
            else:
                raions_without_hromadas.append(raion)
        
        # Print summary
        print(f"\nRaions classification:")
        print(f"  - With hromadas (top-down will be applied): {len(raions_with_hromadas)}")
        print(f"  - Without hromadas (using raion's own data): {len(raions_without_hromadas)}")
        
        # Process raions with hromadas (apply top-down)
        print("\nProcessing raions with hromadas (applying top-down distribution)...")
        for raion in raions_with_hromadas:
            # Find hromadas belonging to this raion
            raion_hromadas = [h for h in hromada_cols if h.startswith(raion + '_')]
            print(f"  {raion}: {len(raion_hromadas)} hromadas - applying top-down distribution")
            
            # Calculate historical proportions
            raion_data = self.hierarchy_df[raion].values
            proportions = {}
            
            for hromada in raion_hromadas:
                hromada_series = self.hierarchy_df[hromada].values
                
                # Calculate average proportion (avoid division by zero)
                raion_sum = raion_data.sum()
                if raion_sum > 0:
                    proportions[hromada] = hromada_series.sum() / raion_sum
                else:
                    proportions[hromada] = 1.0 / len(raion_hromadas)  # Equal distribution
            
            # Normalize proportions to sum to 1
            total_prop = sum(proportions.values())
            if total_prop > 0:
                proportions = {k: v / total_prop for k, v in proportions.items()}
            else:
                # Equal distribution if no data
                proportions = {k: 1.0 / len(raion_hromadas) for k in raion_hromadas}
            
            # Distribute raion forecast to hromadas
            for hromada in raion_hromadas:
                hromada_forecasts[hromada] = raion_forecasts_dict[raion] * proportions[hromada]
        
        # Process raions without hromadas (use raion's own forecast directly)
        if raions_without_hromadas:
            print("\nProcessing raions without hromadas (using raion's own forecast data)...")
            for raion in raions_without_hromadas:
                print(f"  {raion}: No hromadas - using raion's own forecast data (no top-down distribution)")
                # For raions without hromadas, the raion forecast is already in raion_forecasts_dict
                # No hromada forecasts will be created for this raion
        
        # Generate fitted values for historical data
        # For middle-out: get fitted values from model for oblasts, then apply BU/TD
        print("\n" + "-" * 70)
        print("Generating fitted values for historical data...")
        print("-" * 70)
        
        # Get fitted values from HTSRegressor for raions, oblasts, and total
        # Use the same model that was fitted above
        raion_fitted = {}
        for raion in raion_cols:
            raion_fitted[raion] = raion_predictions[raion].iloc[:n_hist].values
        
        oblast_fitted = {}
        for oblast in oblast_cols:
            oblast_fitted[oblast] = raion_predictions[oblast].iloc[:n_hist].values
        
        # Bottom-up: Sum raions → oblasts → total (fitted)
        # Total fitted is already calculated by HTSRegressor
        total_fitted = raion_predictions['total'].iloc[:n_hist].values
        
        # Verify consistency
        oblast_sum_fitted = sum([oblast_fitted[oblast] for oblast in oblast_cols])
        if not np.allclose(oblast_sum_fitted, total_fitted, atol=0.01):
            print("Warning: Using sum of oblasts for total fitted")
            total_fitted = oblast_sum_fitted
        
        # Top-down: Distribute raions to hromadas (fitted)
        # Only for raions that have hromadas
        hromada_fitted = {}
        
        print("Applying top-down distribution for fitted values (only for raions with hromadas)...")
        # Calculate historical proportions (same as for forecast)
        # Only process raions that were identified as having hromadas
        for raion in raions_with_hromadas:
            raion_hromadas = [h for h in hromada_cols if h.startswith(raion + '_')]
            
            if not raion_hromadas:
                continue
            
            # Calculate historical proportions
            raion_data = self.hierarchy_df[raion].values
            proportions = {}
            
            for hromada in raion_hromadas:
                hromada_series = self.hierarchy_df[hromada].values
                raion_sum = raion_data.sum()
                if raion_sum > 0:
                    proportions[hromada] = hromada_series.sum() / raion_sum
                else:
                    proportions[hromada] = 1.0 / len(raion_hromadas)
            
            # Normalize proportions
            total_prop = sum(proportions.values())
            if total_prop > 0:
                proportions = {k: v / total_prop for k, v in proportions.items()}
            else:
                proportions = {k: 1.0 / len(raion_hromadas) for k in raion_hromadas}
            
            # Distribute raion fitted values to hromadas
            for hromada in raion_hromadas:
                hromada_fitted[hromada] = raion_fitted[raion] * proportions[hromada]
        
        # Create fitted DataFrame (same index as historical data)
        fitted_df = pd.DataFrame(index=self.hierarchy_df.index)
        fitted_df['total'] = total_fitted
        
        # Add oblast fitted values
        for oblast in oblast_cols:
            fitted_df[oblast] = oblast_fitted[oblast]
        
        # Add raion fitted values
        for raion in raion_cols:
            fitted_df[raion] = raion_fitted[raion]
        
        # Add hromada fitted values (only for raions that have hromadas)
        for hromada in hromada_cols:
            if hromada in hromada_fitted:
                fitted_df[hromada] = hromada_fitted[hromada]
            else:
                fitted_df[hromada] = 0
        
        # Create forecast DataFrame (future dates)
        forecast_df = pd.DataFrame(index=forecast_dates)
        forecast_df['total'] = total_forecast
        
        # Add oblast forecasts
        for oblast in oblast_cols:
            forecast_df[oblast] = oblast_forecasts_dict[oblast]
        
        # Add raion forecasts
        for raion in raion_cols:
            forecast_df[raion] = raion_forecasts_dict[raion]
        
        # Add hromada forecasts (only for raions that have hromadas)
        for hromada in hromada_cols:
            if hromada in hromada_forecasts:
                forecast_df[hromada] = hromada_forecasts[hromada]
            else:
                forecast_df[hromada] = 0
        
        print(f"\nForecasts generated:")
        print(f"  - Total: {len(forecast_df)} days")
        print(f"  - Oblasts: {len(oblast_cols)} series")
        print(f"  - Raions: {len(raion_cols)} series")
        print(f"    * With hromadas (top-down applied): {len(raions_with_hromadas)}")
        print(f"    * Without hromadas (direct forecast): {len(raions_without_hromadas)}")
        print(f"  - Hromadas: {len(hromada_forecasts)} series")
        
        # Verify consistency
        print("\n" + "-" * 70)
        print("Verifying hierarchy consistency...")
        print("-" * 70)
        
        # Check: sum of oblasts = total
        oblast_sum = forecast_df[oblast_cols].sum(axis=1).values
        total_diff = np.abs(oblast_sum - forecast_df['total'].values)
        print(f"Oblasts → Total consistency:")
        print(f"  Max difference: {total_diff.max():.6f}")
        print(f"  Mean difference: {total_diff.mean():.6f}")
        
        # Check: sum of raions = oblast (for each oblast)
        raion_oblast_consistency = []
        for oblast in oblast_cols:
            oblast_raions = [r for r in raion_cols if r.startswith(oblast + '_')]
            if oblast_raions and all(r in forecast_df.columns for r in oblast_raions):
                raion_sum = forecast_df[oblast_raions].sum(axis=1).values
                oblast_forecast = forecast_df[oblast].values
                diff = np.abs(raion_sum - oblast_forecast)
                raion_oblast_consistency.append(diff.max())
        
        if raion_oblast_consistency:
            print(f"Raions → Oblasts consistency:")
            print(f"  Max difference: {max(raion_oblast_consistency):.6f}")
            print(f"  Mean difference: {np.mean(raion_oblast_consistency):.6f}")
        
        # Check: sum of hromadas = raion (for each raion WITH hromadas)
        hromada_raion_consistency = []
        for raion in raions_with_hromadas:
            raion_hromadas = [h for h in hromada_cols if h.startswith(raion + '_')]
            if raion_hromadas and all(h in forecast_df.columns for h in raion_hromadas):
                hromada_sum = forecast_df[raion_hromadas].sum(axis=1).values
                raion_forecast = forecast_df[raion].values
                diff = np.abs(hromada_sum - raion_forecast)
                hromada_raion_consistency.append(diff.max())
        
        if hromada_raion_consistency:
            print(f"Hromadas → Raions consistency (for raions with hromadas):")
            print(f"  Max difference: {max(hromada_raion_consistency):.6f}")
            print(f"  Mean difference: {np.mean(hromada_raion_consistency):.6f}")
        
        if raions_without_hromadas:
            print(f"\nRaions without hromadas (using raion's own forecast data, no top-down):")
            for raion in raions_without_hromadas[:5]:  # Show first 5
                print(f"  - {raion}: Forecast = Raion forecast (no top-down distribution)")
            if len(raions_without_hromadas) > 5:
                print(f"  ... and {len(raions_without_hromadas) - 5} more")
        
        return fitted_df, forecast_df, None

