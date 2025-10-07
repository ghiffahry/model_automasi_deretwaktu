import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import Config

class ForecastVisualizer:
    def __init__(self):
        plt.style.use(Config.PLOT_STYLE)
        
    def plot_train_test_forecast(self, train: pd.Series, test: pd.Series, 
                                  forecast: pd.Series, model_name: str = "Model",
                                  lower_ci: pd.Series = None, upper_ci: pd.Series = None,
                                  figsize: tuple = Config.FIGSIZE):
        """
        Plot train, test, dan forecast dengan confidence interval
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=Config.DPI)
        
        # Plot training data
        ax.plot(train.index, train.values, label='Training Data', 
                linewidth=2, color='blue', alpha=0.7)
        
        # Plot test data
        if test is not None and len(test) > 0:
            ax.plot(test.index, test.values, label='Test Data', 
                    linewidth=2, color='green', alpha=0.7)
        
        # Plot forecast
        ax.plot(forecast.index, forecast.values, label='Forecast', 
                linewidth=2.5, color='red', marker='o', markersize=4)
        
        # Plot confidence interval
        if lower_ci is not None and upper_ci is not None:
            ax.fill_between(forecast.index, lower_ci, upper_ci, 
                           color='red', alpha=0.2, label='95% Confidence Interval')
        
        ax.set_title(f'{model_name} - Train/Test/Forecast', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_future_forecast(self, historical: pd.Series, forecast: pd.Series,
                            model_name: str = "Model", 
                            lower_ci: pd.Series = None, upper_ci: pd.Series = None,
                            figsize: tuple = Config.FIGSIZE):
        """
        Plot historical data + future forecast
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=Config.DPI)
        
        # Plot historical
        ax.plot(historical.index, historical.values, 
                label='Historical Data', linewidth=2, color='blue')
        
        # Plot forecast
        ax.plot(forecast.index, forecast.values, 
                label='Future Forecast', linewidth=2.5, color='red', 
                marker='o', markersize=5, linestyle='--')
        
        # Confidence interval
        if lower_ci is not None and upper_ci is not None:
            ax.fill_between(forecast.index, lower_ci, upper_ci,
                           color='red', alpha=0.2, label='95% CI')
        
        # Vertical line pemisah
        ax.axvline(x=historical.index[-1], color='gray', 
                   linestyle='--', linewidth=2, alpha=0.5)
        
        ax.set_title(f'{model_name} - Future Forecast', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_multiple_forecasts(self, train: pd.Series, test: pd.Series,
                               forecasts: dict, figsize: tuple = (14, 8)):
        """
        Plot multiple model forecasts
        Args:
            forecasts: Dict of {model_name: forecast_series}
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=Config.DPI)
        
        # Training data
        ax.plot(train.index, train.values, label='Training', 
                linewidth=2, color='blue', alpha=0.7)
        
        # Test data
        if test is not None and len(test) > 0:
            ax.plot(test.index, test.values, label='Test (Actual)', 
                    linewidth=2.5, color='black', marker='o', markersize=4)
        
        # Multiple forecasts
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
        for idx, (model_name, forecast) in enumerate(forecasts.items()):
            ax.plot(forecast.index, forecast.values, 
                   label=f'{model_name} Forecast',
                   linewidth=2, color=colors[idx % len(colors)], 
                   marker='s', markersize=3, alpha=0.7)
        
        ax.set_title('Multiple Models Forecast Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_forecast_intervals(self, forecast: pd.Series, intervals: list,
                               model_name: str = "Model", figsize: tuple = Config.FIGSIZE):
        """
        Plot forecast dengan multiple confidence intervals
        Args:
            intervals: List of tuples [(lower_80, upper_80), (lower_95, upper_95)]
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=Config.DPI)
        
        # Forecast line
        ax.plot(forecast.index, forecast.values, 
                label='Forecast', linewidth=2.5, color='red', marker='o')
        
        # Plot intervals
        colors = ['blue', 'green', 'orange']
        alphas = [0.3, 0.2, 0.1]
        labels = ['80% CI', '95% CI', '99% CI']
        
        for idx, (lower, upper) in enumerate(intervals):
            ax.fill_between(forecast.index, lower, upper,
                           color=colors[idx % len(colors)],
                           alpha=alphas[idx % len(alphas)],
                           label=labels[idx % len(labels)])
        
        ax.set_title(f'{model_name} - Forecast with Confidence Intervals', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_training_history(self, history: dict, figsize: tuple = (10, 6)):
        """
        Plot training history (untuk neural network models)
        Args:
            history: Dict dengan 'train_losses' dan optional 'val_losses'
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=Config.DPI)
        
        epochs = range(1, len(history['train_losses']) + 1)
        ax.plot(epochs, history['train_losses'], 
               label='Training Loss', linewidth=2, color='blue')
        
        if 'val_losses' in history and history['val_losses']:
            ax.plot(epochs, history['val_losses'],
                   label='Validation Loss', linewidth=2, color='red')
        
        ax.set_title('Model Training History', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_dashboard(self, train: pd.Series, test: pd.Series,
                        forecasts: dict, metrics: pd.DataFrame,
                        figsize: tuple = (16, 12)):
        """
        Create comprehensive dashboard
        """
        fig = plt.figure(figsize=figsize, dpi=Config.DPI)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Main forecast plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(train.index, train.values, label='Train', linewidth=2, color='blue', alpha=0.7)
        if test is not None and len(test) > 0:
            ax1.plot(test.index, test.values, label='Test', linewidth=2, color='green', alpha=0.7)
        
        colors = ['red', 'orange', 'purple']
        for idx, (model_name, forecast) in enumerate(forecasts.items()):
            ax1.plot(forecast.index, forecast.values, 
                    label=f'{model_name}', linewidth=2, 
                    color=colors[idx % len(colors)], marker='o', markersize=3)
        
        ax1.set_title('Forecasting Results', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Metrics table
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.axis('tight')
        ax2.axis('off')
        table = ax2.table(cellText=metrics.values, colLabels=metrics.columns,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax2.set_title('Model Metrics', fontsize=12, fontweight='bold', pad=20)
        
        # 3. Error distribution
        ax3 = fig.add_subplot(gs[1, 1])
        if test is not None and len(forecasts) > 0:
            first_model = list(forecasts.keys())[0]
            errors = test - forecasts[first_model]
            ax3.hist(errors.dropna(), bins=20, edgecolor='black', alpha=0.7)
            ax3.set_title(f'{first_model} Error Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Error')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # 4. Model comparison
        ax4 = fig.add_subplot(gs[2, :])
        if len(metrics) > 0:
            x = np.arange(len(metrics))
            width = 0.25
            
            ax4.bar(x - width, metrics['RMSE'], width, label='RMSE', alpha=0.8)
            ax4.bar(x, metrics['MAE'], width, label='MAE', alpha=0.8)
            ax4.bar(x + width, metrics['MAPE'], width, label='MAPE', alpha=0.8)
            
            ax4.set_xlabel('Model')
            ax4.set_ylabel('Score')
            ax4.set_title('Model Comparison', fontsize=12, fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics['Model'])
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Time Series Forecasting Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig