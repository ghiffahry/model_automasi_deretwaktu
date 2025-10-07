import argparse
import sys
from pipeline import ForecastingPipeline
from config import Config

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Time Series Forecasting Pipeline - ARIMA, SARIMA, Transformer'
    )
    
    # Required arguments
    parser.add_argument(
        '--filepath',
        type=str,
        required=True,
        help='Path to CSV file'
    )
    parser.add_argument(
        '--date_col',
        type=str,
        required=True,
        help='Name of date column'
    )
    parser.add_argument(
        '--value_col',
        type=str,
        required=True,
        help='Name of value column'
    )
    
    # Optional arguments
    parser.add_argument(
        '--freq',
        type=str,
        default='D',
        help='Frequency of time series (D=daily, M=monthly, Y=yearly)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['ARIMA', 'SARIMA', 'Transformer'],
        help='Models to train (ARIMA, SARIMA, Transformer)'
    )
    parser.add_argument(
        '--no-arima-grid',
        action='store_true',
        help='Disable grid search for ARIMA'
    )
    parser.add_argument(
        '--no-sarima-grid',
        action='store_true',
        help='Disable grid search for SARIMA'
    )
    parser.add_argument(
        '--transformer-epochs',
        type=int,
        default=Config.TRANSFORMER_EPOCHS,
        help='Number of epochs for Transformer training'
    )
    parser.add_argument(
        '--save-models',
        action='store_true',
        default=True,
        help='Save trained models'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        default=True,
        help='Save visualization plots'
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        default='./plots/',
        help='Directory to save plots'
    )
    parser.add_argument(
        '--forecast-steps',
        type=int,
        default=10,
        help='Number of steps to forecast into future'
    )
    
    return parser.parse_args()

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         TIME SERIES FORECASTING PIPELINE                     ║
    ║         ARIMA | SARIMA | Transformer                         ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Parse arguments
    args = parse_arguments()
    
    print(f"\n📊 Configuration:")
    print(f"  File: {args.filepath}")
    print(f"  Date Column: {args.date_col}")
    print(f"  Value Column: {args.value_col}")
    print(f"  Frequency: {args.freq}")
    print(f"  Models: {', '.join(args.models)}")
    
    try:
        # Initialize pipeline
        pipeline = ForecastingPipeline(
            filepath=args.filepath,
            date_column=args.date_col,
            value_column=args.value_col,
            freq=args.freq
        )
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            models=args.models,
            save_models=args.save_models,
            save_plots=args.save_plots,
            plot_dir=args.plot_dir
        )
        
        # Generate future forecast with best model
        print("\n" + "=" * 60)
        print("GENERATING FUTURE FORECAST")
        print("=" * 60)
        
        best_model = results['best_model']
        future_forecast = pipeline.generate_future_forecast(
            model_name=best_model,
            steps=args.forecast_steps
        )
        
        print(f"\nFuture Forecast ({args.forecast_steps} steps) using {best_model}:")
        print(future_forecast.to_string())
        
        # Save future forecast
        if args.save_plots:
            fig = pipeline.visualizer.plot_future_forecast(
                pipeline.data,
                future_forecast,
                model_name=best_model
            )
            import os
            filepath = os.path.join(args.plot_dir, f'future_forecast_{best_model}.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"\n💾 Future forecast plot saved to: {filepath}")
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"\n📈 Summary:")
        print(f"  Best Model: {results['best_model']}")
        print(f"  Models Trained: {', '.join(args.models)}")
        if args.save_models:
            print(f"  Models Saved: {Config.MODEL_SAVE_PATH}")
        if args.save_plots:
            print(f"  Plots Saved: {args.plot_dir}")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())