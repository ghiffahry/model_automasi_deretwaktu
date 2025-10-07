import pandas as pd
import numpy as np
from typing import Tuple, Optional
from config import Config

class TimeSeriesDataLoader:
    def __init__(self, filepath: str, date_column: str, value_column: str, freq: str = 'D'):
        self.filepath = filepath
        self.date_column = date_column
        self.value_column = value_column
        self.freq = freq
        self.data = None
        self.train = None
        self.test = None
        
    def load_data(self) -> pd.Series:
        """Load data dari CSV dan convert ke time series"""
        df = pd.read_csv(self.filepath)
        
        # Parse tanggal
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df = df.sort_values(self.date_column)
        
        # Set index sebagai datetime
        df.set_index(self.date_column, inplace=True)
        
        # Extract series
        self.data = df[self.value_column]
        
        # Set frequency
        self.data = self.data.asfreq(self.freq)
        
        return self.data
    
    def handle_missing_values(self, method: str = 'interpolate') -> pd.Series:
        """
        Handle missing values
        Args:
            method: 'interpolate', 'ffill', 'bfill', 'drop'
        """
        if self.data is None:
            raise ValueError("Data belum di-load. Jalankan load_data() terlebih dahulu.")
        
        if method == 'interpolate':
            self.data = self.data.interpolate(method='linear')
        elif method == 'ffill':
            self.data = self.data.fillna(method='ffill')
        elif method == 'bfill':
            self.data = self.data.fillna(method='bfill')
        elif method == 'drop':
            self.data = self.data.dropna()
        
        return self.data
    
    def split_data(self, train_ratio: float = Config.TRAIN_TEST_SPLIT) -> Tuple[pd.Series, pd.Series]:
        """Split data menjadi train dan test"""
        if self.data is None:
            raise ValueError("Data belum di-load.")
        
        split_idx = int(len(self.data) * train_ratio)
        self.train = self.data[:split_idx]
        self.test = self.data[split_idx:]
        
        return self.train, self.test
    
    def get_exogenous_variables(self, exog_columns: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load variabel eksogen untuk ARIMAX/SARIMAX
        Args:
            exog_columns: List nama kolom eksogen
        Returns:
            X_train, X_test
        """
        df = pd.read_csv(self.filepath)
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df = df.sort_values(self.date_column)
        df.set_index(self.date_column, inplace=True)
        
        X = df[exog_columns]
        
        split_idx = int(len(X) * Config.TRAIN_TEST_SPLIT)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        
        return X_train, X_test
    
    def get_summary(self) -> dict:
        """Ringkasan statistik data"""
        if self.data is None:
            raise ValueError("Data belum di-load.")
        
        return {
            'length': len(self.data),
            'start_date': self.data.index[0],
            'end_date': self.data.index[-1],
            'mean': self.data.mean(),
            'std': self.data.std(),
            'min': self.data.min(),
            'max': self.data.max(),
            'missing_values': self.data.isna().sum()
        }