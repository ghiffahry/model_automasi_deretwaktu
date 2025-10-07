import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from config import Config

class ModelEvaluator:
    def __init__(self, actual: pd.Series, predicted: pd.Series, model_name: str = "Model"):
        """
        Args:
            actual: Actual values
            predicted: Predicted values
            model_name: Name of the model
        """
        self.actual = actual
        self.predicted = predicted
        self.model_name = model_name
        self.metrics = {}
        
    def calculate_rmse(self) -> float:
        """Root Mean Squared Error"""
        mse = np.mean((self.actual - self.predicted) ** 2)
        rmse = np.sqrt(mse)
        self.metrics['RMSE'] = rmse
        return rmse
    
    def calculate_mae(self) -> float:
        """Mean Absolute Error"""
        mae = np.mean(np.abs(self.actual - self.predicted))
        self.metrics['MAE'] = mae
        return mae
    
    def calculate_mape(self) -> float:
        """Mean Absolute Percentage Error"""
        # Hindari division by zero
        mask = self.actual != 0
        mape = np.mean(np.abs((self.actual[mask] - self.predicted[mask]) / self.actual[mask])) * 100
        self.metrics['MAPE'] = mape
        return mape
    
    def calculate_mse(self) -> float:
        """Mean Squared Error"""
        mse = np.mean((self.actual - self.predicted) ** 2)
        self.metrics['MSE'] = mse
        return mse
    
    def calculate_r2(self) -> float:
        """R-squared score"""
        ss_res = np.sum((self.actual - self.predicted) ** 2)
        ss_tot = np.sum((self.actual - self.actual.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        self.metrics['R2'] = r2
        return r2
    
    def calculate_all_metrics(self) -> Dict[str, float]:
        """Calculate all metrics"""
        self.calculate_rmse()
        self.calculate_mae()
        self.calculate_mape()
        self.calculate_mse()
        self.calculate_r2()
        return self.metrics
    
    def get_metrics_df(self) -> pd.DataFrame:
        """Return metrics as DataFrame"""
        if not self.metrics:
            self.calculate_all_metrics()
        
        return pd.DataFrame({
            'Metric': list(self.metrics.keys()),
            'Value': list(self.metrics.values())
        })
    
    def plot_predictions(self, figsize: tuple = Config.FIGSIZE):
        """Plot actual vs predicted"""
        plt.style.use(Config.PLOT_STYLE)
        fig, ax = plt.subplots(figsize=figsize, dpi=Config.DPI)
        
        ax.plot(self.actual.index, self.actual.values, 
                label='Actual', linewidth=2, marker='o', markersize=4)
        ax.plot(self.predicted.index, self.predicted.values, 
                label='Predicted', linewidth=2, marker='s', markersize=4, alpha=0.7)
        
        ax.set_title(f'{self.model_name} - Actual vs Predicted', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(self, figsize: tuple = Config.FIGSIZE):
        """Plot residuals"""
        residuals = self.actual - self.predicted
        
        plt.style.use(Config.PLOT_STYLE)
        fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=Config.DPI)
        
        # Residual plot
        axes[0].plot(residuals.index, residuals.values, linewidth=1.5)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_title(f'{self.model_name} - Residuals', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Residual')
        axes[0].grid(True, alpha=0.3)
        
        # Residual histogram
        axes[1].hist(residuals.values, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_error_distribution(self, figsize: tuple = (10, 6)):
        """Plot error distribution"""
        errors = self.actual - self.predicted
        percentage_errors = (errors / self.actual) * 100
        
        plt.style.use(Config.PLOT_STYLE)
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=Config.DPI)
        
        # Absolute errors
        axes[0].boxplot(errors.values)
        axes[0].set_title('Absolute Errors', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Error')
        axes[0].grid(True, alpha=0.3)
        
        # Percentage errors
        axes[1].boxplot(percentage_errors.values)
        axes[1].set_title('Percentage Errors', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Error (%)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

class MultiModelComparator:
    """Compare multiple models"""
    def __init__(self):
        self.models = {}
        self.results = []
        
    def add_model(self, model_name: str, actual: pd.Series, predicted: pd.Series):
        """Add model for comparison"""
        evaluator = ModelEvaluator(actual, predicted, model_name)
        metrics = evaluator.calculate_all_metrics()
        
        self.models[model_name] = evaluator
        self.results.append({
            'Model': model_name,
            **metrics
        })
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Return comparison table"""
        return pd.DataFrame(self.results)
    
    def plot_comparison(self, metric: str = 'RMSE', figsize: tuple = (10, 6)):
        """Plot metric comparison across models"""
        df = self.get_comparison_table()
        
        plt.style.use(Config.PLOT_STYLE)
        fig, ax = plt.subplots(figsize=figsize, dpi=Config.DPI)
        
        ax.bar(df['Model'], df[metric], alpha=0.7, edgecolor='black')
        ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate labels if many models
        if len(df) > 3:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def plot_all_predictions(self, actual: pd.Series, figsize: tuple = (14, 8)):
        """Plot all model predictions together"""
        plt.style.use(Config.PLOT_STYLE)
        fig, ax = plt.subplots(figsize=figsize, dpi=Config.DPI)
        
        # Plot actual
        ax.plot(actual.index, actual.values, 
                label='Actual', linewidth=2.5, color='black', marker='o', markersize=4)
        
        # Plot each model's predictions
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        for idx, (model_name, evaluator) in enumerate(self.models.items()):
            ax.plot(evaluator.predicted.index, evaluator.predicted.values,
                   label=model_name, linewidth=2, alpha=0.7, 
                   color=colors[idx % len(colors)], marker='s', markersize=3)
        
        ax.set_title('All Models Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_best_model(self, metric: str = 'RMSE') -> str:
        """Get best model based on metric (lower is better for RMSE, MAE, MAPE)"""
        df = self.get_comparison_table()
        
        if metric in ['RMSE', 'MAE', 'MAPE', 'MSE']:
            best_idx = df[metric].idxmin()
        elif metric == 'R2':
            best_idx = df[metric].idxmax()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return df.loc[best_idx, 'Model']