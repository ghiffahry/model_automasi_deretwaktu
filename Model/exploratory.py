import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesExplorer:
    def __init__(self, data: pd.Series):
        self.data = data
    
    def plot_acf_pacf(self, lags=40, figsize=(14, 6)):
        """
        Plot ACF and PACF - METHOD BARU
        Returns: (figure, recommendations_dict)
        """
        try:
            data_clean = self.data.dropna()
            
            if len(data_clean) < 10:
                return None, {
                    'model_type': 'Unknown',
                    'recommended_p': 1,
                    'recommended_q': 1
                }
            
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Calculate ACF and PACF
            max_lags = min(lags, len(data_clean) // 2 - 1)
            acf_values = acf(data_clean, nlags=max_lags, alpha=0.05)
            pacf_values = pacf(data_clean, nlags=max_lags, alpha=0.05)
            
            # Plot ACF
            axes[0].stem(range(len(acf_values[0])), acf_values[0])
            axes[0].axhline(y=0, color='black', linewidth=0.5)
            axes[0].set_title('Autocorrelation Function (ACF)')
            axes[0].set_xlabel('Lag')
            axes[0].set_ylabel('ACF')
            axes[0].grid(True, alpha=0.3)
            
            # Plot PACF
            axes[1].stem(range(len(pacf_values[0])), pacf_values[0])
            axes[1].axhline(y=0, color='black', linewidth=0.5)
            axes[1].set_title('Partial Autocorrelation Function (PACF)')
            axes[1].set_xlabel('Lag')
            axes[1].set_ylabel('PACF')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Recommendations
            n = len(data_clean)
            ci = 1.96 / np.sqrt(n)
            
            # Detect q from ACF
            q = 0
            for i in range(1, len(acf_values[0])):
                if abs(acf_values[0][i]) < ci:
                    q = max(0, i - 1)
                    break
            
            # Detect p from PACF
            p = 0
            for i in range(1, len(pacf_values[0])):
                if abs(pacf_values[0][i]) < ci:
                    p = max(0, i - 1)
                    break
            
            recommendations = {
                'model_type': 'ARIMA',
                'recommended_p': min(p, 5),
                'recommended_q': min(q, 5)
            }
            
            return fig, recommendations
            
        except Exception as e:
            print(f"ACF/PACF plot error: {e}")
            return None, {
                'model_type': 'ARIMA',
                'recommended_p': 1,
                'recommended_q': 1
            }
        
    def detect_outliers(self, method='iqr', threshold=1.5):
        """
        Detect outliers using IQR or Z-score method
        FIXED: All array comparison issues
        """
        data_clean = self.data.dropna()
        
        if len(data_clean) == 0:
            return {
                'indices': [], 
                'count': 0, 
                'outlier_count': 0, 
                'outlier_percentage': 0.0
            }
        
        try:
            if method == 'iqr':
                Q1 = data_clean.quantile(0.25)
                Q3 = data_clean.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # CRITICAL FIX: Use bitwise OR (|) not logical or
                outliers_mask = (data_clean < lower_bound) | (data_clean > upper_bound)
                outlier_indices = data_clean[outliers_mask].index.tolist()
                
                return {
                    'indices': outlier_indices,
                    'count': len(outlier_indices),
                    'outlier_count': len(outlier_indices),
                    'outlier_percentage': (len(outlier_indices) / len(data_clean)) * 100,
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data_clean))
                # CRITICAL FIX: Direct comparison, no ambiguous truth value
                outliers_mask = z_scores > threshold
                outlier_indices = data_clean[outliers_mask].index.tolist()
                
                return {
                    'indices': outlier_indices,
                    'count': len(outlier_indices),
                    'outlier_count': len(outlier_indices),
                    'outlier_percentage': (len(outlier_indices) / len(data_clean)) * 100,
                    'threshold': float(threshold)
                }
            else:
                return {
                    'indices': [], 
                    'count': 0, 
                    'outlier_count': 0, 
                    'outlier_percentage': 0.0
                }
                
        except Exception as e:
            print(f"Error in detect_outliers: {e}")
            return {
                'indices': [], 
                'count': 0, 
                'outlier_count': 0, 
                'outlier_percentage': 0.0, 
                'error': str(e)
            }
    
    def decompose(self, model='additive', period=None):
        """
        Seasonal decomposition
        FIXED: Better error handling and validation
        """
        try:
            data_clean = self.data.dropna()
            
            if len(data_clean) < 2:
                return None, None, None, None
            
            if period is None:
                period = min(12, max(2, len(data_clean) // 2))
            
            if len(data_clean) < period * 2:
                print(f"Data too short for decomposition with period {period}")
                return None, None, None, None
            
            decomposition = seasonal_decompose(
                data_clean, 
                model=model, 
                period=period,
                extrapolate_trend='freq'
            )
            
            # Calculate seasonal strength safely
            seasonal_var = np.var(decomposition.seasonal.dropna())
            residual_var = np.var(decomposition.resid.dropna())
            
            total_var = seasonal_var + residual_var
            if total_var > 0:
                seasonal_strength = seasonal_var / total_var
            else:
                seasonal_strength = 0.0
            
            return decomposition, decomposition, period, seasonal_strength
            
        except Exception as e:
            print(f"Decomposition error: {e}")
            return None, None, None, None
    
    def plot_acf_pacf_after_differencing(self, order=1, lags=40):
        """
        Calculate ACF and PACF after differencing
        FIXED: Proper error handling and return structure
        Returns: (plot_data_dict, statistical_info_dict)
        """
        try:
            # Original series
            original_clean = self.data.dropna()
            
            if len(original_clean) < 10:
                return None, {
                    'adf_original': 1.0,
                    'adf_differenced': 1.0,
                    'is_stationary_after_diff': False,
                    'error': 'Data too short'
                }
            
            # Differenced series
            differenced = original_clean.diff(order).dropna()
            
            if len(differenced) < 10:
                return None, {
                    'adf_original': 1.0,
                    'adf_differenced': 1.0,
                    'is_stationary_after_diff': False,
                    'error': 'Differenced data too short'
                }
            
            # ADF tests
            try:
                adf_original = adfuller(original_clean)
                adf_diff = adfuller(differenced)
                
                # Calculate ACF and PACF
                max_lags = min(lags, len(differenced) // 2 - 1)
                
                acf_original = acf(original_clean, nlags=max_lags, alpha=0.05)
                pacf_original = pacf(original_clean, nlags=max_lags, alpha=0.05)
                acf_diff = acf(differenced, nlags=max_lags, alpha=0.05)
                pacf_diff = pacf(differenced, nlags=max_lags, alpha=0.05)
                
                # Prepare plot data
                plot_data = {
                    'original': {
                        'dates': original_clean.index.strftime('%Y-%m-%d').tolist(),
                        'values': original_clean.tolist(),
                        'is_stationary': bool(adf_original[1] < 0.05),
                        'adf_statistic': float(adf_original[0]),
                        'adf_pvalue': float(adf_original[1])
                    },
                    'differenced': {
                        'dates': differenced.index.strftime('%Y-%m-%d').tolist(),
                        'values': differenced.tolist(),
                        'is_stationary': bool(adf_diff[1] < 0.05),
                        'adf_statistic': float(adf_diff[0]),
                        'adf_pvalue': float(adf_diff[1])
                    },
                    'acf_original': {
                        'lags': list(range(len(acf_original[0]))),
                        'values': acf_original[0].tolist()
                    },
                    'pacf_original': {
                        'lags': list(range(len(pacf_original[0]))),
                        'values': pacf_original[0].tolist()
                    },
                    'acf_differenced': {
                        'lags': list(range(len(acf_diff[0]))),
                        'values': acf_diff[0].tolist()
                    },
                    'pacf_differenced': {
                        'lags': list(range(len(pacf_diff[0]))),
                        'values': pacf_diff[0].tolist()
                    }
                }
                
                statistical_info = {
                    'adf_original': float(adf_original[1]),
                    'adf_differenced': float(adf_diff[1]),
                    'is_stationary_after_diff': bool(adf_diff[1] < 0.05),
                    'differencing_order': order
                }
                
                return plot_data, statistical_info
                
            except Exception as e:
                print(f"ADF test error: {e}")
                return None, {
                    'adf_original': 1.0,
                    'adf_differenced': 1.0,
                    'is_stationary_after_diff': False,
                    'error': str(e)
                }
                
        except Exception as e:
            print(f"Differencing analysis error: {e}")
            return None, {
                'adf_original': 1.0,
                'adf_differenced': 1.0,
                'is_stationary_after_diff': False,
                'error': str(e)
            }
    
    def comprehensive_eda_report(self):
        """
        Generate comprehensive EDA report
        FIXED: All array operations
        """
        try:
            data_clean = self.data.dropna()
            
            if len(data_clean) == 0:
                return {'error': 'No data available'}
            
            # Basic statistics
            basic_stats = {
                'mean': float(data_clean.mean()),
                'median': float(data_clean.median()),
                'std': float(data_clean.std()),
                'min': float(data_clean.min()),
                'max': float(data_clean.max()),
                'range': float(data_clean.max() - data_clean.min()),
                'cv': float((data_clean.std() / data_clean.mean()) * 100) if data_clean.mean() != 0 else 0.0,
                'skewness': float(data_clean.skew()),
                'kurtosis': float(data_clean.kurtosis())
            }
            
            # Trend analysis
            try:
                x = np.arange(len(data_clean))
                y = data_clean.values
                
                # Linear regression for trend
                slope, intercept = np.polyfit(x, y, 1)
                r_squared = np.corrcoef(x, y)[0, 1] ** 2
                
                # Determine trend direction
                if abs(slope) < 0.01:
                    trend_direction = 'Flat'
                elif slope > 0:
                    trend_direction = 'Increasing'
                else:
                    trend_direction = 'Decreasing'
                
                trend_info = {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_squared),
                    'trend_direction': trend_direction,
                    'p_value': 0.0  # Placeholder
                }
            except Exception as e:
                print(f"Trend analysis error: {e}")
                trend_info = {
                    'slope': 0.0,
                    'intercept': 0.0,
                    'r_squared': 0.0,
                    'trend_direction': 'Unknown',
                    'p_value': 1.0
                }
            
            return {
                'basic_stats': basic_stats,
                'trend_info': trend_info
            }
            
        except Exception as e:
            print(f"Comprehensive EDA error: {e}")
            return {
                'error': str(e),
                'basic_stats': {},
                'trend_info': {}
            }
    
    def plot_time_series(self, figsize=(14, 6)):
        """Plot time series"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.data.index, self.data.values, linewidth=2)
        ax.set_title('Time Series Plot')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_distribution(self, figsize=(12, 5)):
        """Plot distribution"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        axes[0].hist(self.data.dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_title('Distribution')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(self.data.dropna())
        axes[1].set_title('Box Plot')
        axes[1].set_ylabel('Value')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig