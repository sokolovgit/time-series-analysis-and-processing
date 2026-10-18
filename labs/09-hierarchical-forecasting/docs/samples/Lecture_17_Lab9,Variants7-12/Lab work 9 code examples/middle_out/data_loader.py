"""
Data loading for 4-level hierarchical structure: Total → Oblast → Raion → Hromada
For Middle-Out: Forecast at Raion level (middle)
"""

import pandas as pd
import numpy as np


class MiddleOutDataLoader:
    """Load and preprocess Ukrainian alert data for 4-level hierarchy"""
    
    def __init__(self, data_path='../data/official_data_uk.csv'):
        self.data_path = data_path
        self.hierarchy_df = None
        
    def load_data(self):
        """Load and preprocess Ukrainian alert data with 4-level hierarchy"""
        print("Loading Ukrainian alert data for 4-level hierarchy (Total → Oblast → Raion → Hromada)...")
        
        # Load data
        df = pd.read_csv(self.data_path, parse_dates=['started_at', 'finished_at'])
        
        # Filter out rows with missing oblast
        df = df[df['oblast'].notna()].copy()
        
        # Exclude non-oblast entries (like "Лубенський район")
        df = df[~df['oblast'].str.contains('район', case=False, na=False)].copy()
        
        print(f"Total records after filtering: {len(df)}")
        
        # Separate data by level
        # 1. Oblast-level data (level == 'oblast') - oblasts without raions/hromadas
        oblast_level_df = df[df['level'] == 'oblast'].copy()
        
        # 2. Raion-level data (level == 'raion') - raions within oblasts
        raion_level_df = df[df['level'] == 'raion'].copy()
        raion_level_df = raion_level_df[raion_level_df['raion'].notna()].copy()
        
        # 3. Hromada-level data (level == 'hromada') - hromadas within raions
        hromada_level_df = df[df['level'] == 'hromada'].copy()
        hromada_level_df = hromada_level_df[hromada_level_df['hromada'].notna()].copy()
        
        print(f"  - Oblast-level records: {len(oblast_level_df)}")
        print(f"  - Raion-level records: {len(raion_level_df)}")
        print(f"  - Hromada-level records: {len(hromada_level_df)}")
        
        # Process oblast-level data
        oblast_level_df['date'] = oblast_level_df['started_at'].dt.date
        daily_oblast_only = oblast_level_df.groupby(['oblast', 'date']).size().reset_index(name='alert_count')
        
        # Process raion-level data
        raion_level_df['date'] = raion_level_df['started_at'].dt.date
        raion_level_df['oblast_raion'] = raion_level_df['oblast'] + '_' + raion_level_df['raion']
        daily_raion = raion_level_df.groupby(['oblast_raion', 'oblast', 'raion', 'date']).size().reset_index(name='alert_count')
        
        # Process hromada-level data
        hromada_level_df['date'] = hromada_level_df['started_at'].dt.date
        hromada_level_df['oblast_raion_hromada'] = (
            hromada_level_df['oblast'] + '_' + 
            hromada_level_df['raion'].fillna('') + '_' + 
            hromada_level_df['hromada']
        )
        daily_hromada = hromada_level_df.groupby(['oblast_raion_hromada', 'oblast', 'raion', 'hromada', 'date']).size().reset_index(name='alert_count')
        
        # Aggregate: raions → oblasts
        daily_oblast_from_raions = raion_level_df.groupby(['oblast', 'date']).size().reset_index(name='alert_count')
        
        # Aggregate: hromadas → raions
        daily_raion_from_hromadas = hromada_level_df.groupby(['oblast', 'raion', 'date']).size().reset_index(name='alert_count')
        daily_raion_from_hromadas['oblast_raion'] = daily_raion_from_hromadas['oblast'] + '_' + daily_raion_from_hromadas['raion']
        
        # Combine oblast data from all sources
        daily_oblast = pd.concat([
            daily_oblast_only,
            daily_oblast_from_raions,
            hromada_level_df.groupby(['oblast', 'date']).size().reset_index(name='alert_count')
        ]).groupby(['oblast', 'date']).sum().reset_index()
        
        # Combine raion data from both sources
        daily_raion_combined = pd.concat([
            daily_raion[['oblast_raion', 'date', 'alert_count']],
            daily_raion_from_hromadas[['oblast_raion', 'date', 'alert_count']]
        ]).groupby(['oblast_raion', 'date']).sum().reset_index()
        
        # Create full date range (from all data)
        all_dates = pd.concat([
            daily_oblast['date'],
            daily_raion_combined['date'] if len(daily_raion_combined) > 0 else pd.Series(),
            daily_hromada['date'] if len(daily_hromada) > 0 else pd.Series()
        ]).unique()
        date_range = pd.date_range(
            start=pd.to_datetime(all_dates).min(),
            end=pd.to_datetime(all_dates).max(),
            freq='D'
        )
        
        # Get unique values
        all_oblasts = daily_oblast['oblast'].unique()
        all_raions = daily_raion_combined['oblast_raion'].unique() if len(daily_raion_combined) > 0 else []
        all_hromadas = daily_hromada['oblast_raion_hromada'].unique() if len(daily_hromada) > 0 else []
        
        print(f"  Oblasts: {len(all_oblasts)}")
        print(f"  Raions: {len(all_raions)}")
        print(f"  Hromadas: {len(all_hromadas)}")
        
        # Create oblast level DataFrame
        oblast_index = pd.MultiIndex.from_product(
            [all_oblasts, date_range.date],
            names=['oblast', 'date']
        )
        oblast_df = pd.DataFrame(index=oblast_index).reset_index()
        oblast_df = oblast_df.merge(
            daily_oblast[['oblast', 'date', 'alert_count']],
            on=['oblast', 'date'],
            how='left'
        )
        oblast_df['alert_count'] = oblast_df['alert_count'].fillna(0).astype(int)
        oblast_pivot = oblast_df.pivot(index='date', columns='oblast', values='alert_count')
        
        # Create raion level DataFrame
        if len(all_raions) > 0:
            raion_index = pd.MultiIndex.from_product(
                [all_raions, date_range.date],
                names=['raion', 'date']
            )
            raion_df = pd.DataFrame(index=raion_index).reset_index()
            raion_df = raion_df.merge(
                daily_raion_combined[['oblast_raion', 'date', 'alert_count']],
                left_on=['raion', 'date'],
                right_on=['oblast_raion', 'date'],
                how='left'
            )
            raion_df['alert_count'] = raion_df['alert_count'].fillna(0).astype(int)
            raion_pivot = raion_df.pivot(index='date', columns='raion', values='alert_count')
        else:
            raion_pivot = pd.DataFrame(index=date_range.date)
        
        # Create hromada level DataFrame
        if len(all_hromadas) > 0:
            hromada_index = pd.MultiIndex.from_product(
                [all_hromadas, date_range.date],
                names=['hromada', 'date']
            )
            hromada_df = pd.DataFrame(index=hromada_index).reset_index()
            hromada_df = hromada_df.merge(
                daily_hromada[['oblast_raion_hromada', 'date', 'alert_count']],
                left_on=['hromada', 'date'],
                right_on=['oblast_raion_hromada', 'date'],
                how='left'
            )
            hromada_df['alert_count'] = hromada_df['alert_count'].fillna(0).astype(int)
            hromada_pivot = hromada_df.pivot(index='date', columns='hromada', values='alert_count')
        else:
            hromada_pivot = pd.DataFrame(index=date_range.date)
        
        # Calculate total (sum of all oblasts)
        total_series = oblast_pivot.sum(axis=1).to_frame('total')
        
        # Combine all levels: total, oblasts, raions, hromadas
        dfs_to_concat = [total_series, oblast_pivot]
        if len(raion_pivot.columns) > 0:
            dfs_to_concat.append(raion_pivot)
        if len(hromada_pivot.columns) > 0:
            dfs_to_concat.append(hromada_pivot)
        
        self.hierarchy_df = pd.concat(dfs_to_concat, axis=1)
        
        # Convert date index to datetime
        self.hierarchy_df.index = pd.to_datetime(self.hierarchy_df.index)
        self.hierarchy_df = self.hierarchy_df.sort_index()
        
        print(f"\nData loaded: {len(self.hierarchy_df)} days")
        print(f"Period: {self.hierarchy_df.index.min()} to {self.hierarchy_df.index.max()}")
        print(f"  - Total: 1 series")
        print(f"  - Oblasts: {len(all_oblasts)} series")
        print(f"  - Raions: {len(all_raions)} series")
        print(f"  - Hromadas: {len(all_hromadas)} series")
        print(f"  - Total time series: {1 + len(all_oblasts) + len(all_raions) + len(all_hromadas)}")
        print(f"Total alerts: {self.hierarchy_df['total'].sum():,.0f}")
        
        return self.hierarchy_df
    
    def create_hierarchy(self):
        """
        Create 4-level hierarchy dictionary for scikit-hts
        Total → Oblast → Raion → Hromada
        """
        print("\nCreating 4-level hierarchy structure...")
        
        # Get columns
        total_col = 'total'
        oblast_cols = [col for col in self.hierarchy_df.columns 
                      if col != 'total' and '_' not in col]
        raion_cols = [col for col in self.hierarchy_df.columns 
                     if '_' in col and col.count('_') == 1]  # oblast_raion format
        hromada_cols = [col for col in self.hierarchy_df.columns 
                       if '_' in col and col.count('_') == 2]  # oblast_raion_hromada format
        
        # Build hierarchy: Total → Oblast → Raion → Hromada
        hierarchy = {
            'total': oblast_cols
        }
        
        # For each oblast, find its raions
        for oblast in oblast_cols:
            oblast_raions = [r for r in raion_cols if r.startswith(oblast + '_')]
            if oblast_raions:
                hierarchy[oblast] = oblast_raions
        
        # For each raion, find its hromadas
        for raion in raion_cols:
            raion_hromadas = [h for h in hromada_cols if h.startswith(raion + '_')]
            if raion_hromadas:
                hierarchy[raion] = raion_hromadas
        
        print(f"Hierarchy created:")
        print(f"  - Level 1 (Top): Total - 1 series")
        print(f"  - Level 2: Oblasts - {len(oblast_cols)} series")
        print(f"  - Level 3 (Middle for Middle-Out): Raions - {len(raion_cols)} series")
        print(f"  - Level 4 (Bottom): Hromadas - {len(hromada_cols)} series")
        print(f"  - Total time series: {1 + len(oblast_cols) + len(raion_cols) + len(hromada_cols)}")
        
        return hierarchy

