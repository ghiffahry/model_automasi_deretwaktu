import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy import stats
from itertools import product
from config import Config
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ARIMAModel:
    def __init__(self, train_data: pd.Series, test_data: pd.Series = None, 
                 exog_train=None, exog_test=None):
        self.train = train_data
        self.test = test_data
        self.exog_train = exog_train
        self.exog_test = exog_test
        self.model = None
        self.fitted_model = None
        self.best_params = None
        self.tentative_models = {}
        self.training_history = []
        
    def auto_detect_params(self, max_lags: int = 40) -> dict:
        """Auto-detect ARIMA parameters from ACF/PACF"""
        try:
            clean_train = self.train.dropna()
            if len(clean_train) < 10:
                return {'p': 1, 'd': 1, 'q': 1}
            
            acf_values = acf(clean_train, nlags=min(max_lags, len(clean_train)//2 - 1), alpha=0.05)
            pacf_values = pacf(clean_train, nlags=min(max_lags, len(clean_train)//2 - 1), alpha=0.05)
            
            n = len(clean_train)
            ci = 1.96 / np.sqrt(n)
            
            # Find cutoff for q (MA order)
            q = 0
            for i in range(1, len(acf_values[0])):
                if abs(acf_values[0][i]) < ci:
                    q = max(0, i - 1)
                    break
            
            # Find cutoff for p (AR order)
            p = 0
            for i in range(1, len(pacf_values[0])):
                if abs(pacf_values[0][i]) < ci:
                    p = max(0, i - 1)
                    break
            
            # Detect d from stationarity test
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(clean_train)
            d = 0 if adf_result[1] < 0.05 else 1
            
            # Limit maximum
            p = min(p, Config.ARIMA_MAX_P)
            q = min(q, Config.ARIMA_MAX_Q)
            d = min(d, Config.ARIMA_MAX_D)
            
            return {'p': p, 'd': d, 'q': q}
        except Exception as e:
            print(f"Error in auto_detect_params: {e}")
            return {'p': 1, 'd': 1, 'q': 1}
    
    def grid_search(self, p_range: list = None, d_range: list = None, 
                    q_range: list = None, criterion: str = 'aic') -> dict:
        """Grid search for best parameters"""
        if p_range is None:
            auto_params = self.auto_detect_params()
            p_range = range(max(0, auto_params['p'] - 2), 
                          min(auto_params['p'] + 3, Config.ARIMA_MAX_P + 1))
        if d_range is None:
            d_range = range(0, Config.ARIMA_MAX_D + 1)
        if q_range is None:
            auto_params = self.auto_detect_params()
            q_range = range(max(0, auto_params['q'] - 2), 
                          min(auto_params['q'] + 3, Config.ARIMA_MAX_Q + 1))
        
        best_score = np.inf
        best_params = None
        results = []
        
        print(f"Testing {len(list(product(p_range, d_range, q_range)))} parameter combinations...")
        
        for p, d, q in product(p_range, d_range, q_range):
            try:
                model = ARIMA(self.train, order=(p, d, q), exog=self.exog_train)
                fitted = model.fit()
                
                score = getattr(fitted, criterion)
                
                model_key = f"ARIMA({p},{d},{q})"
                self.tentative_models[model_key] = {
                    'order': (p, d, q),
                    'fitted_model': fitted,
                    'aic': fitted.aic,
                    'bic': fitted.bic,
                    'hqic': fitted.hqic,
                    'log_likelihood': fitted.llf
                }
                
                results.append({
                    'p': p, 'd': d, 'q': q, 
                    criterion: score,
                    'aic': fitted.aic,
                    'bic': fitted.bic
                })
                
                if score < best_score:
                    best_score = score
                    best_params = (p, d, q)
                    
            except Exception:
                continue
        
        self.best_params = best_params
        
        results_sorted = sorted(results, key=lambda x: x[criterion])[:10]
        
        print(f"\nTop 5 models by {criterion.upper()}:")
        for i, res in enumerate(results_sorted[:5], 1):
            print(f"  {i}. ARIMA({res['p']},{res['d']},{res['q']}) - "
                  f"{criterion.upper()}: {res[criterion]:.2f}")
        
        return {
            'best_order': best_params,
            f'best_{criterion}': best_score,
            'all_results': results_sorted,
            'tentative_count': len(self.tentative_models)
        }
    
    def fit(self, order: tuple = None):
        """Fit ARIMA model"""
        if order is None:
            if self.best_params is None:
                auto_params = self.auto_detect_params()
                order = (auto_params['p'], auto_params['d'], auto_params['q'])
            else:
                order = self.best_params
        
        print(f"\nFitting ARIMA{order}...")
        self.model = ARIMA(self.train, order=order, exog=self.exog_train)
        self.fitted_model = self.model.fit()
        
        self.training_history.append({
            'order': order,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'log_likelihood': self.fitted_model.llf
        })
        
        return self.fitted_model
    
    def fit_custom_order(self, p: int, d: int, q: int):
        """Fit specific ARIMA order chosen by user"""
        return self.fit(order=(p, d, q))
    
    def test_residual_assumptions(self):
        """
        Test residual assumptions
        CRITICAL: Residual = Actual - Predicted (statsmodels does this automatically)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        # RESIDUAL DEFINITION: statsmodels.resid already = actual - fitted
        residuals = self.fitted_model.resid
        if residuals is None:
            raise ValueError("No residuals available from fitted model")
        
        residuals = residuals.dropna()
        
        if len(residuals) == 0:
            raise ValueError("No residuals available after dropping NA")
        
        results = {}
        
        # 1. Heteroscedasticity test (constant variance)
        try:
            arch_test = het_arch(residuals)
            results['heteroscedasticity'] = {
                'test': 'ARCH LM Test',
                'statistic': float(arch_test[0]),
                'p_value': float(arch_test[1]),
                'is_homoscedastic': bool(arch_test[1] > 0.05),
                'conclusion': 'Constant variance' if bool(arch_test[1] > 0.05) else 'Heteroscedastic'
            }
        except Exception as e:
            print(f"Heteroscedasticity test error: {e}")
            results['heteroscedasticity'] = {'error': str(e)}
        
        # 2. Normality test
        try:
            jb_result = stats.jarque_bera(residuals)
            if len(jb_result) == 4:
                jb_stat, jb_pval, skew, kurtosis = jb_result
            else:
                jb_stat, jb_pval = jb_result
                skew = float(residuals.skew())
                kurtosis = float(residuals.kurtosis())
            
            results['normality'] = {
                'test': 'Jarque-Bera Test',
                'statistic': float(jb_stat),
                'p_value': float(jb_pval),
                'skewness': float(skew),
                'kurtosis': float(kurtosis),
                'is_normal': bool(jb_pval > 0.05),
                'conclusion': 'Normal distribution' if bool(jb_pval > 0.05) else 'Non-normal distribution'
            }
        except Exception as e:
            print(f"Normality test error: {e}")
            results['normality'] = {'error': str(e)}
        
        # 3. White noise test (no autocorrelation)
        try:
            max_lags = min(10, len(residuals)//5)
            if max_lags < 1:
                max_lags = 1
            
            lb_test = acorr_ljungbox(residuals, lags=max_lags)
            # CRITICAL FIX: Use .all() to check all p-values
            all_pvalues_above_threshold = (lb_test['lb_pvalue'] > 0.05).all()
            
            results['white_noise'] = {
                'test': 'Ljung-Box Test',
                'lags_tested': len(lb_test),
                'p_values': lb_test['lb_pvalue'].tolist(),
                'is_white_noise': bool(all_pvalues_above_threshold),
                'min_p_value': float(lb_test['lb_pvalue'].min()),
                'conclusion': 'White noise (no autocorrelation)' if bool(all_pvalues_above_threshold) else 'Autocorrelation present'
            }
        except Exception as e:
            print(f"White noise test error: {e}")
            results['white_noise'] = {'error': str(e)}
        
        return results
    
    def get_fitted_values_data(self):
        """
        Get fitted vs actual data for plotting
        Returns actual values and model's fitted values
        FIXED: Handle index mismatch safely
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        try:
            # Get fitted values (in-sample predictions)
            fitted_values = self.fitted_model.fittedvalues
            
            # CRITICAL FIX: Use minimum length to avoid index issues
            min_len = min(len(self.train), len(fitted_values))
            
            # Ensure we're using aligned indices
            train_subset = self.train.iloc[:min_len]
            fitted_subset = fitted_values.iloc[:min_len]
            
            return {
                'dates': train_subset.index.strftime('%Y-%m-%d').tolist(),
                'actual': train_subset.values.tolist(),
                'fitted': fitted_subset.values.tolist()
            }
        except Exception as e:
            print(f"Error getting fitted data: {e}")
            import traceback
            print(traceback.format_exc())
            return {
                'dates': [],
                'actual': [],
                'fitted': []
            }
    
    def get_residual_plot_data(self):
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
    
        try:
            residuals = self.fitted_model.resid.dropna()
        
            # Time series data
            residual_plot = {
                'dates': residuals.index.strftime('%Y-%m-%d').tolist(),
                'residuals': residuals.tolist()
            }
        
            # Histogram
            counts, bins = np.histogram(residuals, bins=30)
            histogram = {
                'bins': bins[:-1].tolist(),
                'counts': counts.tolist()
            }
        
            # Normal distribution overlay
            mu, std = residuals.mean(), residuals.std()
            x = np.linspace(residuals.min(), residuals.max(), 100)
            from scipy.stats import norm
            normal_y = norm.pdf(x, mu, std) * len(residuals) * (bins[1] - bins[0])
        
            histogram['normal_x'] = x.tolist()
            histogram['normal_y'] = normal_y.tolist()
        
            # ACF
            max_lags = min(40, len(residuals) // 2 - 1)
            acf_vals = acf(residuals, nlags=max_lags, alpha=0.05)
        
            acf_data = {
                'lags': list(range(len(acf_vals[0]))),
                'acf': acf_vals[0].tolist(),
                'confidence_interval': [[float(ci[0]), float(ci[1])] for ci in acf_vals[1]] if len(acf_vals) > 1 else None
            }
        
            return {
                'residual_plot': residual_plot,
                'histogram': histogram,
                'acf': acf_data
            }  
        
        except Exception as e:
            print(f"Error in get_residual_plot_data: {e}")
            return {
                'residual_plot': {'dates': [], 'residuals': []},
                'histogram': {'bins': [], 'counts': []},
                'acf': {'lags': [], 'acf': []}
            }
                
    def plot_residual_diagnostics(self, figsize=(14, 10)):
        """Comprehensive residual diagnostic plots"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        residuals = self.fitted_model.resid
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Residuals over time
        axes[0, 0].plot(residuals)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Histogram
        axes[0, 1].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
        mu, std = residuals.mean(), residuals.std()
        xmin, xmax = axes[0, 1].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        axes[0, 1].plot(x, p, 'r-', linewidth=2, label='Normal Distribution')
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. ACF
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals, lags=40, ax=axes[1, 1], alpha=0.05)
        axes[1, 1].set_title('ACF of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def predict(self, steps: int = None, exog=None) -> pd.Series:
        """Generate forecast"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        if steps is None:
            steps = len(self.test) if self.test is not None else 10
        
        if exog is None:
            exog = self.exog_test
        
        forecast = self.fitted_model.forecast(steps=steps, exog=exog)
        
        return forecast
    
    def get_forecast_with_ci(self, steps: int = None, alpha: float = 0.05, exog=None) -> dict:
        """Forecast with confidence intervals"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        if steps is None:
            steps = len(self.test) if self.test is not None else 10
        
        if exog is None:
            exog = self.exog_test
        
        forecast_obj = self.fitted_model.get_forecast(steps=steps, exog=exog)
        forecast = forecast_obj.predicted_mean
        ci = forecast_obj.conf_int(alpha=alpha)
        
        return {
            'forecast': forecast,
            'lower_ci': ci.iloc[:, 0],
            'upper_ci': ci.iloc[:, 1]
        }
    
    def summary(self):
        """Print model summary"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        return self.fitted_model.summary()
    
    def residuals(self) -> pd.Series:
        """Return residuals (actual - fitted)"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        return self.fitted_model.resid