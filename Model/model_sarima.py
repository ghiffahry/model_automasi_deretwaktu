import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
from itertools import product
from config import Config
import warnings
warnings.filterwarnings('ignore')


class SARIMAModel:
    def __init__(self, train_data: pd.Series, test_data: pd.Series = None, 
                 exog_train=None, exog_test=None, seasonal_period: int = None):
        self.train = train_data
        self.test = test_data
        self.exog_train = exog_train
        self.exog_test = exog_test
        self.seasonal_period = seasonal_period if seasonal_period else Config.SARIMA_SEASONAL_PERIOD
        self.model = None
        self.fitted_model = None
        self.best_params = None
        
    def detect_seasonal_period(self) -> int:
        """Auto-detect seasonal period dari ACF"""
        try:
            clean_train = self.train.dropna()
            if len(clean_train) < 4:
                return Config.SARIMA_SEASONAL_PERIOD
            
            acf_values = acf(clean_train, nlags=min(len(clean_train)//2, 50))
            peaks = []
            for i in range(1, len(acf_values) - 1):
                if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
                    if acf_values[i] > 0.3:
                        peaks.append(i)
            
            if len(peaks) > 0:
                self.seasonal_period = peaks[0]
            else:
                self.seasonal_period = Config.SARIMA_SEASONAL_PERIOD
            
            return self.seasonal_period
        except Exception as e:
            print(f"Error detecting seasonal period: {e}")
            return Config.SARIMA_SEASONAL_PERIOD
    
    def auto_detect_params(self, max_lags: int = 40) -> dict:
        """Deteksi parameter SARIMA otomatis"""
        try:
            clean_train = self.train.dropna()
            if len(clean_train) < 10:
                return {'p': 1, 'd': 1, 'q': 1, 'P': 1, 'D': 1, 'Q': 1, 's': 12}
            
            acf_values = acf(clean_train, nlags=min(max_lags, len(clean_train)//2 - 1), alpha=0.05)
            pacf_values = pacf(clean_train, nlags=min(max_lags, len(clean_train)//2 - 1), alpha=0.05)
            
            n = len(clean_train)
            ci = 1.96 / np.sqrt(n)
            
            # q dari ACF
            q = 0
            for i in range(1, len(acf_values[0])):
                if abs(acf_values[0][i]) < ci:
                    q = max(0, i - 1)
                    break
            
            # p dari PACF
            p = 0
            for i in range(1, len(pacf_values[0])):
                if abs(pacf_values[0][i]) < ci:
                    p = max(0, i - 1)
                    break
            
            adf_result = adfuller(clean_train)
            d = 0 if adf_result[1] < 0.05 else 1
            
            p = min(p, Config.SARIMA_MAX_P)
            q = min(q, Config.SARIMA_MAX_Q)
            d = min(d, Config.SARIMA_MAX_D)
            
            s = self.seasonal_period if self.seasonal_period else self.detect_seasonal_period()
            
            if len(clean_train) > s:
                seasonal_diff_data = clean_train.diff(s).dropna()
                if len(seasonal_diff_data) > 0:
                    adf_seasonal = adfuller(seasonal_diff_data)
                    D = 0 if adf_seasonal[1] < 0.05 else 1
                else:
                    D = 1
            else:
                D = 1
            
            if s < len(pacf_values[0]):
                P = 1 if abs(pacf_values[0][s]) > ci else 0
            else:
                P = 1
            
            if s < len(acf_values[0]):
                Q = 1 if abs(acf_values[0][s]) > ci else 0
            else:
                Q = 1
            
            return {
                'p': p, 'd': d, 'q': q,
                'P': P, 'D': D, 'Q': Q, 's': s
            }
        except Exception as e:
            print(f"Error in auto_detect_params: {e}")
            return {'p': 1, 'd': 1, 'q': 1, 'P': 1, 'D': 1, 'Q': 1, 's': 12}
    
    def grid_search(self, p_range: list = None, d_range: list = None, q_range: list = None,
                    P_range: list = None, D_range: list = None, Q_range: list = None,
                    criterion: str = 'aic') -> dict:
        """Grid search untuk parameter terbaik SARIMA"""
        if p_range is None or q_range is None:
            auto = self.auto_detect_params()
            if p_range is None:
                p_range = range(max(0, auto['p'] - 1), min(auto['p'] + 2, Config.SARIMA_MAX_P + 1))
            if q_range is None:
                q_range = range(max(0, auto['q'] - 1), min(auto['q'] + 2, Config.SARIMA_MAX_Q + 1))
        if d_range is None:
            d_range = range(0, Config.SARIMA_MAX_D + 1)
        if P_range is None:
            P_range = range(0, Config.SARIMA_MAX_SEASONAL_P + 1)
        if D_range is None:
            D_range = range(0, Config.SARIMA_MAX_SEASONAL_D + 1)
        if Q_range is None:
            Q_range = range(0, Config.SARIMA_MAX_SEASONAL_Q + 1)
        
        s = self.seasonal_period
        best_score = np.inf
        best_params = None
        results = []
        
        print(f"Running SARIMA grid search with seasonal period s={s}...")
        
        for p, d, q in product(p_range, d_range, q_range):
            for P, D, Q in product(P_range, D_range, Q_range):
                try:
                    model = SARIMAX(self.train, 
                                   order=(p, d, q),
                                   seasonal_order=(P, D, Q, s),
                                   exog=self.exog_train,
                                   enforce_stationarity=False,
                                   enforce_invertibility=False)
                    fitted = model.fit(disp=False, maxiter=200)
                    score = getattr(fitted, criterion)
                    results.append({
                        'p': p, 'd': d, 'q': q,
                        'P': P, 'D': D, 'Q': Q, 's': s,
                        criterion: score
                    })
                    if score < best_score:
                        best_score = score
                        best_params = ((p, d, q), (P, D, Q, s))
                except Exception:
                    continue
        
        self.best_params = best_params
        return {
            'best_order': best_params[0] if best_params else None,
            'best_seasonal_order': best_params[1] if best_params else None,
            f'best_{criterion}': best_score,
            'all_results': sorted(results, key=lambda x: x[criterion])[:10]
        }
    
    def fit(self, order: tuple = None, seasonal_order: tuple = None):
        """Fit SARIMA model"""
        if order is None or seasonal_order is None:
            if self.best_params is None:
                auto = self.auto_detect_params()
                order = (auto['p'], auto['d'], auto['q'])
                seasonal_order = (auto['P'], auto['D'], auto['Q'], auto['s'])
            else:
                order = self.best_params[0]
                seasonal_order = self.best_params[1]
        
        self.model = SARIMAX(self.train, 
                            order=order,
                            seasonal_order=seasonal_order,
                            exog=self.exog_train,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
        self.fitted_model = self.model.fit(disp=False, maxiter=200)
        return self.fitted_model
    
    def predict(self, steps: int = None, exog=None) -> pd.Series:
        """Forecast"""
        if self.fitted_model is None:
            raise ValueError("Model belum di-fit.")
        if steps is None:
            steps = len(self.test) if self.test is not None else 10
        if exog is None:
            exog = self.exog_test
        forecast = self.fitted_model.forecast(steps=steps, exog=exog)
        return forecast
    
    def get_forecast_with_ci(self, steps: int = None, alpha: float = 0.05, exog=None) -> dict:
        """Forecast dengan confidence interval"""
        if self.fitted_model is None:
            raise ValueError("Model belum di-fit.")
        if steps is None:
            steps = len(self.test) if self.test is not None else 10
        if exog is None:
            exog = self.exog_test
        forecast_obj = self.fitted_model.get_forecast(steps=steps, exog=exog)
        forecast = forecast_obj.predicted_mean
        ci = forecast_obj.conf_int(alpha=alpha)
        return {'forecast': forecast,'lower_ci': ci.iloc[:, 0],'upper_ci': ci.iloc[:, 1]}
    
    def summary(self):
        if self.fitted_model is None:
            raise ValueError("Model belum di-fit.")
        return self.fitted_model.summary()
    
    def residuals(self) -> pd.Series:
        """Return residuals (actual - fitted)"""
        if self.fitted_model is None:
            raise ValueError("Model belum di-fit.")
        return self.fitted_model.resid
    
    def diagnostic_plots(self):
        if self.fitted_model is None:
            raise ValueError("Model belum di-fit.")
        return self.fitted_model.plot_diagnostics(figsize=(14, 8))
    
    def test_residual_assumptions(self):
        """
        Test residual assumptions for SARIMA
        CRITICAL: Residual = Actual - Predicted (statsmodels does this)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        # statsmodels.resid = actual - fitted values
        residuals = self.fitted_model.resid
        if residuals is None:
            raise ValueError("No residuals available from fitted model")
        
        residuals = residuals.dropna()
        
        if len(residuals) == 0:
            raise ValueError("No residuals available after dropping NA")
        
        results = {}
        
        # 1. Heteroscedasticity test
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
                'conclusion': 'Normal' if bool(jb_pval > 0.05) else 'Non-normal'
            }
        except Exception as e:
            print(f"Normality test error: {e}")
            results['normality'] = {'error': str(e)}
        
        # 3. White noise test
        try:
            max_lags = min(10, len(residuals)//5)
            if max_lags < 1:
                max_lags = 1
            
            lb_test = acorr_ljungbox(residuals, lags=max_lags)
            # FIXED: Use .all() for proper boolean evaluation
            all_pvalues_above = (lb_test['lb_pvalue'] > 0.05).all()
            
            results['white_noise'] = {
                'test': 'Ljung-Box Test',
                'is_white_noise': bool(all_pvalues_above),
                'min_p_value': float(lb_test['lb_pvalue'].min()),
                'conclusion': 'White noise' if bool(all_pvalues_above) else 'Autocorrelation present'
            }
        except Exception as e:
            print(f"White noise test error: {e}")
            results['white_noise'] = {'error': str(e)}
        
        return results
    
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
                
    def plot_residual_diagnostics(self, figsize=(14, 10)):
        """Comprehensive residual diagnostic plots"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        residuals = self.fitted_model.resid
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes[0, 0].plot(residuals)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
        mu, std = residuals.mean(), residuals.std()
        xmin, xmax = axes[0, 1].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        axes[0, 1].plot(x, p, 'r-', linewidth=2)
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        plot_acf(residuals, lags=40, ax=axes[1, 1], alpha=0.05)
        axes[1, 1].set_title('ACF of Residuals')
        plt.tight_layout()
        return fig