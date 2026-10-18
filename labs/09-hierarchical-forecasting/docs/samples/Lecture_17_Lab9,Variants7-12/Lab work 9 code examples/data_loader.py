"""
Data loading and preprocessing for Ukrainian air alerts
"""

import pandas as pd
import numpy as np


class AlertDataLoader:
    """Load and preprocess Ukrainian alert data for hierarchical forecasting"""
    
    def __init__(self, data_path='../data/official_data_uk.csv'):
        self.data_path = data_path
        self.hierarchy_df = None
        
    def load_data(self):
        """Load and preprocess Ukrainian alert data"""
        print("Loading Ukrainian alert data...")
        
        # Load data
        df = pd.read_csv(self.data_path, parse_dates=['started_at', 'finished_at'])
        
        # Filter ONLY oblast-level alerts (exclude raion and hromada)
        df = df[df['level'] == 'oblast'].copy()
        
        # Also filter out rows with missing oblast name
        df = df[df['oblast'].notna()].copy()
        
        print(f"Filtered to oblast-level only: {len(df)} records")
        
        # Count alerts per oblast per day
        df['date'] = df['started_at'].dt.date
        
        # Count alerts per oblast per day
        daily_alerts = df.groupby(['oblast', 'date']).size().reset_index(name='alert_count')
        
        # Create full date range
        all_oblasts = daily_alerts['oblast'].unique()
        date_range = pd.date_range(
            start=daily_alerts['date'].min(),
            end=daily_alerts['date'].max(),
            freq='D'
        )
        
        # Create complete time series (fill missing days with 0)
        index = pd.MultiIndex.from_product(
            [all_oblasts, date_range.date],
            names=['oblast', 'date']
        )
        full_df = pd.DataFrame(index=index).reset_index()
        
        # Merge with actual data
        full_df = full_df.merge(
            daily_alerts[['oblast', 'date', 'alert_count']],
            on=['oblast', 'date'],
            how='left'
        )
        full_df['alert_count'] = full_df['alert_count'].fillna(0).astype(int)
        
        # Pivot to wide format: each oblast as column, dates as rows
        self.hierarchy_df = full_df.pivot(
            index='date',
            columns='oblast',
            values='alert_count'
        )
        
        # Convert date index to datetime
        self.hierarchy_df.index = pd.to_datetime(self.hierarchy_df.index)
        self.hierarchy_df = self.hierarchy_df.sort_index()
        
        # Calculate total (sum of all oblasts)
        self.hierarchy_df['total'] = self.hierarchy_df.sum(axis=1)
        
        # Reorder columns: total first, then oblasts
        cols = ['total'] + [c for c in self.hierarchy_df.columns if c != 'total']
        self.hierarchy_df = self.hierarchy_df[cols]
        
        print(f"Data loaded: {len(self.hierarchy_df)} days")
        print(f"Period: {self.hierarchy_df.index.min()} to {self.hierarchy_df.index.max()}")
        print(f"Oblasts: {len(self.hierarchy_df.columns) - 1}")
        print(f"Total alerts: {self.hierarchy_df['total'].sum():,.0f}")
        
        return self.hierarchy_df
    
    def create_hierarchy(self):
        """
        Create hierarchy dictionary for scikit-hts
        Total has all oblasts as children
        """
        print("\nCreating hierarchy structure...")
        
        # Get all oblast columns (excluding 'total')
        oblasts = [col for col in self.hierarchy_df.columns if col != 'total']
        
        hierarchy = {
            'total': oblasts
        }
        
        print(f"Hierarchy created:")
        print(f"  - Total: 1 series")
        print(f"  - Oblasts: {len(oblasts)} series")
        print(f"  - Total time series: {len(oblasts) + 1}")
        
        return hierarchy

