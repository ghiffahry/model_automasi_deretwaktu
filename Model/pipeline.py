import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import joblib
import os

from data_loader import TimeSeriesDataLoader
from exploratory import TimeSeriesExplorer
from stationarity import StationarityTester
from model_arima import ARIMAModel
from model_sarima import SARIMAModel
from model_transformer import TransformerModel
from evaluator import ModelEvaluator, MultiModelComparator
from visualizer import ForecastVisualizer
from config import Config

class ForecastingPipeline:
    def __init__(self, filepath: str, date_column: str, value_column: str, 
                 freq: str = 'D', exog_columns: List[str] = None):
        """
        Initialize forecasting pipeline
        
        Args:
            filepath: Path to CSV file
            date_column: Name of date column
            value_column: Name of value column
            freq: Frequency ('D', 'M', 'Y', etc.)
            exog_columns: List of exogenous variable column names (for ARIMAX/SARIMAX)
        """
        self.filepath = filepath
        self.date_column = date_column
        self.value_column = value_column
        self.freq = freq
        self.exog_columns = exog_columns
        
        # Components
        self.data_loader = None
        self.explorer = None
        self.stationarity_tester = None
        
        # Data
        self.data = None
        self.train = None
        self.test = None
        self.exog_train = None
        self.exog_test = None
        
        # Models
        self.models = {}
        self.fitted_models = {}
        self.forecasts = {}
        
        # Results
        self.metrics = {}
        self.comparator = MultiModelComparator()
        self.visualizer = ForecastVisualizer()
        
        # Insights
        self.eda_results = {}
        self.stationarity_results = {}
        
    def load_and_explore(self, handle_missing: str = 'interpolate') -> dict:
        """
        Step 1: Load data dan exploratory analysis
        """
        print("=" * 60)
        print("STEP 1: DATA LOADING & EXPLORATION")
        print("=" * 60)
        
        # Load data
        self.data_loader = TimeSeriesDataLoader(
            self.filepath, self.date_column, self.value_column, self.freq
        )
        self.data = self.data_loader.load_data()
        self.data = self.data_loader.handle_missing_values(method=handle_missing)
        
        summary = self.data_loader.get_summary()
        print(f"\nData Summary:")
        for key, val in summary.items():
            print(f"  {key}: {val}")
        
        # Split data
        self.train, self.test = self.data_loader.split_data()
        print(f"\nTrain size: {len(self.train)}, Test size: {len(self.test)}")
        
        # Load exogenous variables if specified
        if self.exog_columns and len(self.exog_columns) > 0:
            try:
                self.exog_train, self.exog_test = self.data_loader.get_exogenous_variables(
                    self.exog_columns
                )
                print(f"\nLoaded {len(self.exog_columns)} exogenous variables:")
                for col in self.exog_columns:
                    print(f"  - {col}")
            except Exception as e:
                print(f"\nWarning: Could not load exogenous variables: {e}")
                self.exog_train = None
                self.exog_test = None
        
        # Exploratory analysis
        self.explorer = TimeSeriesExplorer(self.train)
        
        # ACF/PACF analysis
        _, acf_pacf_rec = self.explorer.plot_acf_pacf()
        self.eda_results['acf_pacf_recommendations'] = acf_pacf_rec
        
        print(f"\nACF/PACF Analysis:")
        print(f"  Model Type: {acf_pacf_rec['model_type']}")
        print(f"  Recommended p: {acf_pacf_rec['recommended_p']}")
        print(f"  Recommended q: {acf_pacf_rec['recommended_q']}")
        
        # Decomposition
        try:
            _, decomp, period = self.explorer.decompose()
            self.eda_results['seasonal_period'] = period
            print(f"  Detected Seasonal Period: {period}")
        except Exception as e:
            print(f"  Warning: Could not perform decomposition: {e}")
            self.eda_results['seasonal_period'] = 12
        
        return self.eda_results
    
    def test_stationarity(self) -> dict:
        """
        Step 2: Stationarity testing
        """
        print("\n" + "=" * 60)
        print("STEP 2: STATIONARITY TESTING")
        print("=" * 60)
        
        self.stationarity_tester = StationarityTester(self.train)
        
        # ADF test
        adf_result = self.stationarity_tester.adf_test()
        print(f"\nADF Test:")
        print(f"  Test Statistic: {adf_result['test_statistic']:.4f}")
        print(f"  p-value: {adf_result['p_value']:.4f}")
        print(f"  Conclusion: {adf_result['conclusion']}")
        
        # KPSS test
        kpss_result = self.stationarity_tester.kpss_test()
        print(f"\nKPSS Test:")
        print(f"  Test Statistic: {kpss_result['test_statistic']:.4f}")
        print(f"  p-value: {kpss_result['p_value']:.4f}")
        print(f"  Conclusion: {kpss_result['conclusion']}")
        
        self.stationarity_results = {
            'adf': adf_result,
            'kpss': kpss_result
        }
        
        # Auto-stationarize jika perlu
        if not adf_result['is_stationary']:
            print("\n  Data is non-stationary. Applying auto-differencing...")
            _, d_order, is_stat = self.stationarity_tester.auto_stationarize()
            print(f"  Differencing order applied: {d_order}")
            print(f"  Is stationary now: {is_stat}")
            self.stationarity_results['differencing_order'] = d_order
        
        return self.stationarity_results
    
    def train_arima(self, order: tuple = None, use_grid_search: bool = True) -> Dict:
        """
        Step 3a: Train ARIMA model
        """
        print("\n" + "=" * 60)
        print("STEP 3A: TRAINING ARIMA MODEL")
        print("=" * 60)
        
        arima = ARIMAModel(self.train, self.test)
        
        if use_grid_search:
            print("\nRunning grid search for best parameters...")
            grid_results = arima.grid_search()
            print(f"  Best order: {grid_results['best_order']}")
            print(f"  Best AIC: {grid_results['best_aic']:.2f}")
        
        # Fit model
        arima.fit(order=order)
        print(f"\nARIMA model fitted!")
        print(f"  Order: {arima.fitted_model.specification['order']}")
        print(f"  AIC: {arima.fitted_model.aic:.2f}")
        print(f"  BIC: {arima.fitted_model.bic:.2f}")
        
        # Forecast
        forecast = arima.predict()
        forecast_ci = arima.get_forecast_with_ci()
        
        # Store
        self.fitted_models['ARIMA'] = arima
        self.forecasts['ARIMA'] = {
            'forecast': forecast,
            'lower_ci': forecast_ci['lower_ci'],
            'upper_ci': forecast_ci['upper_ci']
        }
        
        # Evaluate
        evaluator = ModelEvaluator(self.test, forecast, 'ARIMA')
        metrics = evaluator.calculate_all_metrics()
        self.comparator.add_model('ARIMA', self.test, forecast)
        
        print(f"\nARIMA Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def train_arimax(self, order: tuple = None, use_grid_search: bool = True) -> Dict:
        """
        Step 3a-X: Train ARIMAX model (ARIMA with exogenous variables)
        """
        print("\n" + "=" * 60)
        print("STEP 3A-X: TRAINING ARIMAX MODEL")
        print("=" * 60)
        
        if self.exog_train is None or len(self.exog_train.columns) == 0:
            raise ValueError("ARIMAX requires exogenous variables. Please specify exog_columns.")
        
        arimax = ARIMAModel(self.train, self.test, 
                           exog_train=self.exog_train, 
                           exog_test=self.exog_test)
        
        if use_grid_search:
            print("\nRunning grid search for best parameters...")
            grid_results = arimax.grid_search()
            print(f"  Best order: {grid_results['best_order']}")
            print(f"  Best AIC: {grid_results['best_aic']:.2f}")
        
        # Fit model
        arimax.fit(order=order)
        print(f"\nARIMAX model fitted!")
        print(f"  Order: {arimax.fitted_model.specification['order']}")
        print(f"  AIC: {arimax.fitted_model.aic:.2f}")
        print(f"  BIC: {arimax.fitted_model.bic:.2f}")
        print(f"  Exogenous variables: {', '.join(self.exog_columns)}")
        
        # Forecast
        forecast = arimax.predict()
        forecast_ci = arimax.get_forecast_with_ci()
        
        # Store
        self.fitted_models['ARIMAX'] = arimax
        self.forecasts['ARIMAX'] = {
            'forecast': forecast,
            'lower_ci': forecast_ci['lower_ci'],
            'upper_ci': forecast_ci['upper_ci']
        }
        
        # Evaluate
        evaluator = ModelEvaluator(self.test, forecast, 'ARIMAX')
        metrics = evaluator.calculate_all_metrics()
        self.comparator.add_model('ARIMAX', self.test, forecast)
        
        print(f"\nARIMAX Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def train_sarima(self, order: tuple = None, seasonal_order: tuple = None,
                     use_grid_search: bool = True) -> Dict:
        """
        Step 3b: Train SARIMA model
        """
        print("\n" + "=" * 60)
        print("STEP 3B: TRAINING SARIMA MODEL")
        print("=" * 60)
        
        seasonal_period = self.eda_results.get('seasonal_period', Config.SARIMA_SEASONAL_PERIOD)
        sarima = SARIMAModel(self.train, self.test, seasonal_period=seasonal_period)
        
        if use_grid_search:
            print("\nRunning grid search for best parameters...")
            grid_results = sarima.grid_search()
            print(f"  Best order: {grid_results['best_order']}")
            print(f"  Best seasonal order: {grid_results['best_seasonal_order']}")
            print(f"  Best AIC: {grid_results['best_aic']:.2f}")
        
        # Fit model
        sarima.fit(order=order, seasonal_order=seasonal_order)
        print(f"\nSARIMA model fitted!")
        print(f"  Order: {sarima.fitted_model.specification['order']}")
        print(f"  Seasonal Order: {sarima.fitted_model.specification['seasonal_order']}")
        print(f"  AIC: {sarima.fitted_model.aic:.2f}")
        print(f"  BIC: {sarima.fitted_model.bic:.2f}")
        
        # Forecast
        forecast = sarima.predict()
        forecast_ci = sarima.get_forecast_with_ci()
        
        # Store
        self.fitted_models['SARIMA'] = sarima
        self.forecasts['SARIMA'] = {
            'forecast': forecast,
            'lower_ci': forecast_ci['lower_ci'],
            'upper_ci': forecast_ci['upper_ci']
        }
        
        # Evaluate
        evaluator = ModelEvaluator(self.test, forecast, 'SARIMA')
        metrics = evaluator.calculate_all_metrics()
        self.comparator.add_model('SARIMA', self.test, forecast)
        
        print(f"\nSARIMA Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def train_sarimax(self, order: tuple = None, seasonal_order: tuple = None,
                      use_grid_search: bool = True) -> Dict:
        """
        Step 3b-X: Train SARIMAX model (SARIMA with exogenous variables)
        """
        print("\n" + "=" * 60)
        print("STEP 3B-X: TRAINING SARIMAX MODEL")
        print("=" * 60)
        
        if self.exog_train is None or len(self.exog_train.columns) == 0:
            raise ValueError("SARIMAX requires exogenous variables. Please specify exog_columns.")
        
        seasonal_period = self.eda_results.get('seasonal_period', Config.SARIMA_SEASONAL_PERIOD)
        sarimax = SARIMAModel(self.train, self.test, 
                             exog_train=self.exog_train,
                             exog_test=self.exog_test,
                             seasonal_period=seasonal_period)
        
        if use_grid_search:
            print("\nRunning grid search for best parameters...")
            grid_results = sarimax.grid_search()
            print(f"  Best order: {grid_results['best_order']}")
            print(f"  Best seasonal order: {grid_results['best_seasonal_order']}")
            print(f"  Best AIC: {grid_results['best_aic']:.2f}")
        
        # Fit model
        sarimax.fit(order=order, seasonal_order=seasonal_order)
        print(f"\nSARIMAX model fitted!")
        print(f"  Order: {sarimax.fitted_model.specification['order']}")
        print(f"  Seasonal Order: {sarimax.fitted_model.specification['seasonal_order']}")
        print(f"  AIC: {sarimax.fitted_model.aic:.2f}")
        print(f"  BIC: {sarimax.fitted_model.bic:.2f}")
        print(f"  Exogenous variables: {', '.join(self.exog_columns)}")
        
        # Forecast
        forecast = sarimax.predict()
        forecast_ci = sarimax.get_forecast_with_ci()
        
        # Store
        self.fitted_models['SARIMAX'] = sarimax
        self.forecasts['SARIMAX'] = {
            'forecast': forecast,
            'lower_ci': forecast_ci['lower_ci'],
            'upper_ci': forecast_ci['upper_ci']
        }
        
        # Evaluate
        evaluator = ModelEvaluator(self.test, forecast, 'SARIMAX')
        metrics = evaluator.calculate_all_metrics()
        self.comparator.add_model('SARIMAX', self.test, forecast)
        
        print(f"\nSARIMAX Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def train_transformer(self, window_size: int = None, epochs: int = None) -> Dict:
        """
        Step 3c: Train Transformer model
        """
        print("\n" + "=" * 60)
        print("STEP 3C: TRAINING TRANSFORMER MODEL")
        print("=" * 60)
        
        if window_size is None:
            window_size = Config.TRANSFORMER_WINDOW_SIZE
        if epochs is None:
            epochs = Config.TRANSFORMER_EPOCHS
        
        transformer = TransformerModel(
            self.train, self.test,
            window_size=window_size,
            epochs=epochs
        )
        
        # Train
        print(f"\nTraining Transformer with window_size={window_size}, epochs={epochs}...")
        transformer.fit()
        
        # Forecast
        forecast = transformer.predict()
        
        # Store
        self.fitted_models['Transformer'] = transformer
        self.forecasts['Transformer'] = {
            'forecast': forecast,
            'lower_ci': None,
            'upper_ci': None
        }
        
        # Evaluate
        evaluator = ModelEvaluator(self.test, forecast, 'Transformer')
        metrics = evaluator.calculate_all_metrics()
        self.comparator.add_model('Transformer', self.test, forecast)
        
        print(f"\nTransformer Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def train_all_models(self, arima_grid: bool = True, sarima_grid: bool = True,
                        include_exog: bool = False) -> pd.DataFrame:
        """
        Train semua model
        
        Args:
            arima_grid: Enable grid search for ARIMA
            sarima_grid: Enable grid search for SARIMA
            include_exog: Include ARIMAX/SARIMAX if exogenous variables available
        """
        print("\n" + "=" * 60)
        print("TRAINING ALL MODELS")
        print("=" * 60)
        
        # Base models
        self.train_arima(use_grid_search=arima_grid)
        self.train_sarima(use_grid_search=sarima_grid)
        self.train_transformer()
        
        # Exogenous models if available
        if include_exog and self.exog_train is not None:
            try:
                self.train_arimax(use_grid_search=arima_grid)
                self.train_sarimax(use_grid_search=sarima_grid)
            except Exception as e:
                print(f"\nWarning: Could not train exogenous models: {e}")
        
        comparison_table = self.comparator.get_comparison_table()
        
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(comparison_table.to_string(index=False))
        
        best_model = self.comparator.get_best_model('RMSE')
        print(f"\n🏆 Best Model (by RMSE): {best_model}")
        
        return comparison_table
    
    def generate_future_forecast(self, model_name: str, steps: int = 10) -> pd.Series:
        """
        Generate future forecast (beyond test set)
        
        Args:
            model_name: Name of fitted model to use
            steps: Number of steps to forecast
            
        Returns:
            pd.Series with forecast values
        """
        print(f"\nGenerating {steps}-step future forecast using {model_name}...")
        
        if model_name not in self.fitted_models:
            raise ValueError(f"Model {model_name} not trained yet.")
        
        model = self.fitted_models[model_name]
        
        # Retrain on full data for future forecast
        if model_name in ['ARIMA', 'ARIMAX']:
            # Prepare exogenous data if needed
            if model_name == 'ARIMAX':
                if self.exog_train is None:
                    raise ValueError("ARIMAX requires exogenous variables")
                # Combine exog train and test
                exog_full = pd.concat([self.exog_train, self.exog_test])
                # For future forecast, use last known exog values or extrapolate
                exog_future = self.exog_test.iloc[-steps:] if len(self.exog_test) >= steps else None
            else:
                exog_full = None
                exog_future = None
            
            full_model = ARIMAModel(self.data, None, 
                                   exog_train=exog_full,
                                   exog_test=exog_future)
            full_model.best_params = model.best_params
            full_model.fit()
            future_forecast = full_model.predict(steps=steps, exog=exog_future)
            
        elif model_name in ['SARIMA', 'SARIMAX']:
            # Prepare exogenous data if needed
            if model_name == 'SARIMAX':
                if self.exog_train is None:
                    raise ValueError("SARIMAX requires exogenous variables")
                exog_full = pd.concat([self.exog_train, self.exog_test])
                exog_future = self.exog_test.iloc[-steps:] if len(self.exog_test) >= steps else None
            else:
                exog_full = None
                exog_future = None
            
            full_model = SARIMAModel(self.data, None,
                                    exog_train=exog_full,
                                    exog_test=exog_future,
                                    seasonal_period=model.seasonal_period)
            full_model.best_params = model.best_params
            full_model.fit()
            future_forecast = full_model.predict(steps=steps, exog=exog_future)
            
        else:  # Transformer
            full_model = TransformerModel(
                self.data, None,
                window_size=model.window_size,
                hidden_dim=model.hidden_dim,
                num_layers=model.num_layers,
                num_heads=model.num_heads,
                dropout=model.dropout,
                learning_rate=model.learning_rate,
                batch_size=model.batch_size,
                epochs=model.epochs
            )
            full_model.fit()
            future_forecast = full_model.predict(steps=steps)
        
        return future_forecast
    
    def save_models(self, save_dir: str = Config.MODEL_SAVE_PATH):
        """Save all trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.fitted_models.items():
            filepath = os.path.join(save_dir, f"{model_name.lower()}_model.pkl")
            
            if model_name == 'Transformer':
                model.save_model(filepath.replace('.pkl', '.pth'))
            else:
                joblib.dump(model, filepath)
            
            print(f"  {model_name} saved to {filepath}")
    
    def get_visualizations(self) -> dict:
        """Generate all visualizations"""
        figs = {}
        
        # 1. Time series plot
        figs['time_series'] = self.explorer.plot_series()
        
        # 2. ACF/PACF
        figs['acf_pacf'], _ = self.explorer.plot_acf_pacf()
        
        # 3. Decomposition
        try:
            figs['decomposition'], _, _ = self.explorer.decompose()
        except:
            pass
        
        # 4. Forecasts for each model
        for model_name, forecast_data in self.forecasts.items():
            figs[f'forecast_{model_name}'] = self.visualizer.plot_train_test_forecast(
                self.train, self.test, forecast_data['forecast'],
                model_name=model_name,
                lower_ci=forecast_data.get('lower_ci'),
                upper_ci=forecast_data.get('upper_ci')
            )
        
        # 5. Comparison plot
        forecast_series = {k: v['forecast'] for k, v in self.forecasts.items()}
        figs['comparison'] = self.visualizer.plot_multiple_forecasts(
            self.train, self.test, forecast_series
        )
        
        # 6. Dashboard
        comparison_table = self.comparator.get_comparison_table()
        figs['dashboard'] = self.visualizer.create_dashboard(
            self.train, self.test, forecast_series, comparison_table
        )
        
        # 7. Model comparison
        figs['rmse_comparison'] = self.comparator.plot_comparison('RMSE')
        figs['mae_comparison'] = self.comparator.plot_comparison('MAE')
        
        return figs
    
    def run_complete_pipeline(self, models: List[str] = ['ARIMA', 'SARIMA', 'Transformer'],
                             save_models: bool = True, save_plots: bool = True,
                             plot_dir: str = './plots/') -> dict:
        """
        Run complete end-to-end pipeline
        
        Args:
            models: List of models to train ['ARIMA', 'SARIMA', 'Transformer', 'ARIMAX', 'SARIMAX']
            save_models: Whether to save trained models
            save_plots: Whether to save plots
            plot_dir: Directory untuk save plots
        """
        print("\n" + "=" * 60)
        print("RUNNING COMPLETE FORECASTING PIPELINE")
        print("=" * 60)
        
        # Step 1: Load & Explore
        self.load_and_explore()
        
        # Step 2: Stationarity
        self.test_stationarity()
        
        # Step 3: Train Models
        results = {}
        for model_name in models:
            try:
                if model_name == 'ARIMA':
                    results['ARIMA'] = self.train_arima()
                elif model_name == 'ARIMAX':
                    results['ARIMAX'] = self.train_arimax()
                elif model_name == 'SARIMA':
                    results['SARIMA'] = self.train_sarima()
                elif model_name == 'SARIMAX':
                    results['SARIMAX'] = self.train_sarimax()
                elif model_name == 'Transformer':
                    results['Transformer'] = self.train_transformer()
            except Exception as e:
                print(f"\nWarning: Could not train {model_name}: {e}")
        
        # Step 4: Comparison
        comparison = self.comparator.get_comparison_table()
        best_model = self.comparator.get_best_model('RMSE')
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED!")
        print("=" * 60)
        print(f"\n🏆 Best Model: {best_model}")
        print("\nFinal Comparison:")
        print(comparison.to_string(index=False))
        
        # Step 5: Save models
        if save_models:
            print("\nSaving models...")
            self.save_models()
        
        # Step 6: Save plots
        if save_plots:
            print("\nGenerating and saving plots...")
            os.makedirs(plot_dir, exist_ok=True)
            
            figs = self.get_visualizations()
            for fig_name, fig in figs.items():
                try:
                    filepath = os.path.join(plot_dir, f"{fig_name}.png")
                    fig.savefig(filepath, dpi=150, bbox_inches='tight')
                    print(f"  Saved: {filepath}")
                except Exception as e:
                    print(f"  Warning: Could not save {fig_name}: {e}")
        
        return {
            'comparison': comparison,
            'best_model': best_model,
            'eda_results': self.eda_results,
            'stationarity_results': self.stationarity_results,
            'metrics': results
        }