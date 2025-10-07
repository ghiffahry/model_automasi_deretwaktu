import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from config import Config

class StationarityTester:
    def __init__(self, data: pd.Series):
        self.data = data
        self.original_data = data.copy()
        
    def adf_test(self, alpha: float = Config.ADF_ALPHA) -> dict:
        """
        Augmented Dickey-Fuller Test
        H0: Series has unit root (non-stationary)
        H1: Series is stationary
        """
        result = adfuller(self.data.dropna(), autolag='AIC')
        
        output = {
            'test_statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'n_observations': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < alpha,
            'conclusion': 'Stationary' if result[1] < alpha else 'Non-Stationary'
        }
        
        return output
    
    def kpss_test(self, regression: str = 'c', alpha: float = Config.ADF_ALPHA) -> dict:
        """
        KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin)
        H0: Series is stationary
        H1: Series has unit root
        Args:
            regression: 'c' (constant), 'ct' (constant+trend)
        """
        result = kpss(self.data.dropna(), regression=regression, nlags='auto')
        
        output = {
            'test_statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'critical_values': result[3],
            'is_stationary': result[1] > alpha,
            'conclusion': 'Stationary' if result[1] > alpha else 'Non-Stationary'
        }
        
        return output
    
    def difference(self, order: int = 1) -> pd.Series:
        """
        Differencing untuk stasioneritas
        Args:
            order: Order differencing (1 = first difference, 2 = second difference)
        """
        self.data = self.data.diff(order).dropna()
        return self.data
    
    def seasonal_difference(self, period: int = 12) -> pd.Series:
        """Seasonal differencing"""
        self.data = self.data.diff(period).dropna()
        return self.data
    
    def log_transform(self) -> pd.Series:
        """Log transformation untuk stabilkan variance"""
        if (self.data <= 0).any():
            raise ValueError("Data mengandung nilai <= 0. Log transform tidak bisa diaplikasikan.")
        self.data = np.log(self.data)
        return self.data
    
    def sqrt_transform(self) -> pd.Series:
        """Square root transformation"""
        if (self.data < 0).any():
            raise ValueError("Data mengandung nilai negatif. Sqrt transform tidak bisa diaplikasikan.")
        self.data = np.sqrt(self.data)
        return self.data
    
    def box_cox_transform(self):
        """Box-Cox transformation"""
        from scipy.stats import boxcox
        if (self.data <= 0).any():
            raise ValueError("Data mengandung nilai <= 0. Box-Cox transform tidak bisa diaplikasikan.")
        self.data, lambda_param = boxcox(self.data)
        self.data = pd.Series(self.data, index=self.original_data.index[:len(self.data)])
        return self.data, lambda_param
    
    def auto_stationarize(self, max_diff: int = 2) -> tuple:
        """
        Otomatis cari kombinasi differencing untuk stasioneritas
        Returns:
            (transformed_data, d_order, is_stationary)
        """
        d = 0
        current_data = self.data.copy()
        
        while d <= max_diff:
            adf_result = adfuller(current_data.dropna(), autolag='AIC')
            if adf_result[1] < Config.ADF_ALPHA:
                self.data = current_data
                return self.data, d, True
            
            d += 1
            if d <= max_diff:
                current_data = current_data.diff().dropna()
        
        # Jika tidak stasioner sampai max_diff
        self.data = current_data
        return self.data, max_diff, False
    
    def reset(self):
        """Reset ke data original"""
        self.data = self.original_data.copy()
        return self.data