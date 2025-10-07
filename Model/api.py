import fastapi as fastapi_module
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import io
import tempfile
import json
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
try:
    from pipeline import ForecastingPipeline
    from config import Config
    from exploratory import TimeSeriesExplorer
    logger.info("✓ All modules imported successfully")
except ImportError as e:
    logger.error(f"✗ Import error: {e}")
    raise

# Validate Config
if not hasattr(Config, 'VALIDATION'):
    logger.error("Config.VALIDATION not found!")
    raise AttributeError("Config.VALIDATION is required")

logger.info(f"✓ Config validated: MIN_DATA_POINTS={Config.VALIDATION['MIN_DATA_POINTS']}")

# ==================== FastAPI Initialization ====================

app = FastAPI(
    title="SURADATA Time Series Forecasting API",
    description="API for ARIMA, SARIMA, and Transformer forecasting models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Global Storage ====================

pipelines = {}
current_pipeline_id = None

# ==================== Request Models ====================

class ExploreRequest(BaseModel):
    pipeline_id: str

class TrainRequest(BaseModel):
    pipeline_id: str
    models: List[str] = ['ARIMA', 'SARIMA']
    arima_grid_search: bool = True
    sarima_grid_search: bool = True
    transformer_epochs: Optional[int] = None

class PredictRequest(BaseModel):
    pipeline_id: str
    model_name: str
    steps: int = 30

class ResidualRequest(BaseModel):
    pipeline_id: str
    model_name: str

# ==================== Helper Functions ====================

def handle_error(message: str, status_code: int = 400):
    """Consistent error response"""
    logger.error(f"Error {status_code}: {message}")
    raise HTTPException(
        status_code=status_code,
        detail={"error": message, "success": False}
    )

def safe_convert(value, default=0):
    """Safely convert numpy types to Python native types"""
    try:
        if pd.isna(value) or value is None:
            return default
        if isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        if isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value
    except:
        return default

def convert_numpy_recursive(obj):
    """Recursively convert numpy types"""
    if obj is None:
        return None
    elif pd.isna(obj):
        return None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_recursive(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_recursive(item) for item in obj]
    return obj

# ==================== Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SURADATA API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check called")
    return {
        "status": "healthy",
        "message": "SURADATA API is running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "pipelines_count": len(pipelines)
    }

@app.post("/upload")
async def upload_data(file: UploadFile = File(...), config: str = None):
    """Upload CSV file and create pipeline"""
    logger.info(f"Upload request: {file.filename}")
    
    try:
        # Validate file
        if not file.filename.endswith('.csv'):
            handle_error("Only CSV files are allowed", 400)

        contents = await file.read()
        if len(contents) == 0:
            handle_error("Uploaded file is empty", 400)

        try:
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            logger.info(f"CSV parsed: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            handle_error(f"Failed to parse CSV: {str(e)}", 400)

        if df.empty or len(df) < Config.VALIDATION['MIN_DATA_POINTS']:
            handle_error(f"Dataset must have at least {Config.VALIDATION['MIN_DATA_POINTS']} rows", 400)

        # Parse config
        if config:
            try:
                config_dict = json.loads(config)
            except json.JSONDecodeError:
                handle_error("Invalid JSON configuration", 400)
        else:
            config_dict = {
                'date_column': df.columns[0],
                'value_column': df.columns[1] if len(df.columns) > 1 else df.columns[0],
                'frequency': 'D',
                'pipeline_name': f"Pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'exogenous_columns': []
            }

        date_col = config_dict.get('date_column')
        value_col = config_dict.get('value_column')

        if date_col not in df.columns:
            handle_error(f"Date column '{date_col}' not found", 400)
        if value_col not in df.columns:
            handle_error(f"Value column '{value_col}' not found", 400)

        exog_cols = config_dict.get('exogenous_columns', [])
        if exog_cols:
            missing_cols = [col for col in exog_cols if col not in df.columns]
            if missing_cols:
                handle_error(f"Exogenous columns not found: {', '.join(missing_cols)}", 400)

        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}"
        temp_dir = tempfile.gettempdir()
        temp_filepath = os.path.join(temp_dir, f"{pipeline_id}.csv")
        df.to_csv(temp_filepath, index=False)
        
        logger.info(f"Creating pipeline: {pipeline_id}")

        pipeline = ForecastingPipeline(
            filepath=temp_filepath,
            date_column=date_col,
            value_column=value_col,
            freq=config_dict.get('frequency', 'D'),
            exog_columns=exog_cols if len(exog_cols) > 0 else None
        )

        pipelines[pipeline_id] = {
            'pipeline': pipeline,
            'config': config_dict,
            'created_at': datetime.now().isoformat(),
            'filepath': temp_filepath
        }

        global current_pipeline_id
        current_pipeline_id = pipeline_id
        
        logger.info(f"✓ Pipeline created successfully: {pipeline_id}")

        return {
            "pipeline_id": pipeline_id,
            "pipeline_name": config_dict.get('pipeline_name', pipeline_id),
            "date_column": date_col,
            "value_column": value_col,
            "frequency": config_dict.get('frequency', 'D'),
            "exogenous_columns": exog_cols,
            "data_shape": {
                "rows": len(df),
                "columns": len(df.columns)
            },
            "columns": df.columns.tolist(),
            "message": "Data uploaded successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Upload error: {traceback.format_exc()}")
        handle_error(f"Upload failed: {str(e)}", 500)

@app.post("/explore")
async def explore_data(request: ExploreRequest):
    """Run exploratory data analysis"""
    logger.info(f"Explore request: {request.pipeline_id}")
    
    try:
        if request.pipeline_id not in pipelines:
            handle_error("Pipeline not found", 404)

        pipeline_obj = pipelines[request.pipeline_id]
        pipeline = pipeline_obj['pipeline']

        try:
            pipeline.load_and_explore()
            stationarity_results = pipeline.test_stationarity()
        except Exception as e:
            logger.error(f"Exploration error: {e}")
            handle_error(f"Failed to load/explore data: {str(e)}", 500)

        if pipeline.train is None or len(pipeline.train) == 0:
            handle_error("No training data available", 400)

        train_values = [safe_convert(v) for v in pipeline.train.values.tolist()]
        train_dates = pipeline.train.index.strftime('%Y-%m-%d').tolist()

        basic_stats = {
            'mean': safe_convert(pipeline.train.mean()),
            'std': safe_convert(pipeline.train.std()),
            'min': safe_convert(pipeline.train.min()),
            'max': safe_convert(pipeline.train.max()),
            'count': int(len(pipeline.train))
        }

        time_series_data = {
            'dates': train_dates,
            'values': train_values
        }

        # Decomposition
        trend = {'dates': [], 'values': []}
        seasonal = {'dates': [], 'values': []}

        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            data_length = len(pipeline.train)
            period = min(12, max(2, data_length // 2))
            if data_length >= period * 2:
                clean_data = pipeline.train.dropna()
                if len(clean_data) >= period * 2:
                    decomp = seasonal_decompose(
                        clean_data,
                        model='additive',
                        period=period,
                        extrapolate_trend='freq'
                    )
                    trend_component = decomp.trend.dropna()
                    if len(trend_component) > 0:
                        trend = {
                            'dates': trend_component.index.strftime('%Y-%m-%d').tolist(),
                            'values': [safe_convert(v) for v in trend_component.values]
                        }
                    seasonal_component = decomp.seasonal.dropna()
                    if len(seasonal_component) > 0:
                        seasonal = {
                            'dates': seasonal_component.index.strftime('%Y-%m-%d').tolist(),
                            'values': [safe_convert(v) for v in seasonal_component.values]
                        }
        except Exception as e:
            logger.warning(f"Decomposition error: {e}")
            trend = {'dates': train_dates, 'values': train_values}
            seasonal = {'dates': train_dates, 'values': [0] * len(train_values)}

        # Distribution
        distribution = {'bins': [], 'counts': []}
        try:
            if len(train_values) > 0:
                hist, bins = np.histogram(train_values, bins=min(30, len(train_values) // 2))
                distribution = {
                    'bins': [safe_convert(b) for b in bins[:-1]],
                    'counts': [safe_convert(c) for c in hist]
                }
        except Exception as e:
            logger.warning(f"Distribution error: {e}")

        # ACF
        acf_data = {'lags': [], 'values': [], 'labels': []}
        try:
            from statsmodels.tsa.stattools import acf
            clean_data = pipeline.train.dropna()
            if len(clean_data) > 2:
                max_lags = min(40, len(clean_data) // 2 - 1)
                if max_lags > 0:
                    acf_values = acf(clean_data, nlags=max_lags)
                    acf_data = {
                        'lags': list(range(len(acf_values))),
                        'values': [safe_convert(v) for v in acf_values],
                        'labels': [str(i) for i in range(len(acf_values))]
                    }
        except Exception as e:
            logger.warning(f"ACF error: {e}")

        # Statistical tests
        statistical_tests = {
            'adf_statistic': safe_convert(stationarity_results.get('adf', {}).get('test_statistic', 0)),
            'adf_pvalue': safe_convert(stationarity_results.get('adf', {}).get('p_value', 1)),
            'is_stationary': bool(stationarity_results.get('adf', {}).get('is_stationary', False)),
            'has_seasonality': bool(pipeline.eda_results.get('seasonal_period', 0) > 1)
        }

        response = {
            "basic_stats": basic_stats,
            "time_series_data": time_series_data,
            "trend": trend,
            "seasonal": seasonal,
            "distribution": distribution,
            "acf": acf_data,
            "statistical_tests": statistical_tests
        }
        
        logger.info(f"✓ Exploration completed: {request.pipeline_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Explore error: {traceback.format_exc()}")
        handle_error(f"Exploration failed: {str(e)}", 500)

@app.post("/explore/detailed")
async def explore_detailed(request: ExploreRequest):
    """Enhanced exploration with outlier detection"""
    logger.info(f"Detailed explore request: {request.pipeline_id}")
    
    try:
        if request.pipeline_id not in pipelines:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "Pipeline not found"}
            )

        pipeline_obj = pipelines[request.pipeline_id]
        pipeline = pipeline_obj['pipeline']

        if pipeline.train is None:
            pipeline.load_and_explore()

        if len(pipeline.train) == 0:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Training data is empty"}
            )

        explorer = TimeSeriesExplorer(pipeline.train)

        # EDA Report
        eda_report = {}
        try:
            eda_report = explorer.comprehensive_eda_report()
            eda_report = convert_numpy_recursive(eda_report)
            logger.info("✓ EDA report generated")
        except Exception as e:
            logger.warning(f"EDA report error: {e}")
            eda_report = {"error": str(e), "basic_stats": {}, "trend_info": {}}

        # Outlier detection - IQR
        outliers_iqr = {'indices': [], 'count': 0, 'outlier_count': 0, 'outlier_percentage': 0.0}
        try:
            iqr_result = explorer.detect_outliers(method='iqr')
            outliers_iqr = convert_numpy_recursive(iqr_result)
            logger.info(f"✓ IQR outliers: {outliers_iqr.get('count', 0)}")
        except Exception as e:
            logger.warning(f"IQR outlier detection error: {e}")
            outliers_iqr['error'] = str(e)

        # Outlier detection - Z-score
        outliers_zscore = {'indices': [], 'count': 0, 'outlier_count': 0, 'outlier_percentage': 0.0}
        try:
            zscore_result = explorer.detect_outliers(method='zscore', threshold=3)
            outliers_zscore = convert_numpy_recursive(zscore_result)
            logger.info(f"✓ Z-score outliers: {outliers_zscore.get('count', 0)}")
        except Exception as e:
            logger.warning(f"Z-score outlier detection error: {e}")
            outliers_zscore['error'] = str(e)

        # Differencing analysis
        differencing_results = {
            'adf_original': 1.0,
            'adf_differenced': 1.0,
            'is_stationary_after_diff': False
        }
        try:
            _, diff_results = explorer.plot_acf_pacf_after_differencing(order=1)
            if diff_results:
                differencing_results = convert_numpy_recursive(diff_results)
                logger.info("✓ Differencing analysis completed")
        except Exception as e:
            logger.warning(f"Differencing analysis error: {e}")
            differencing_results['error'] = str(e)

        # Seasonality analysis
        seasonality_info = {'period': None, 'strength': None}
        try:
            if len(pipeline.train) >= 4:
                _, decomp_obj, period, seasonal_strength = explorer.decompose()
                if period is not None and seasonal_strength is not None:
                    seasonality_info = {
                        'period': int(period),
                        'strength': float(seasonal_strength)
                    }
                    logger.info(f"✓ Seasonality: period={period}, strength={seasonal_strength:.4f}")
        except Exception as e:
            logger.warning(f"Seasonality analysis error: {e}")
            seasonality_info['error'] = str(e)

        response_data = {
            "eda_report": eda_report,
            "outliers": {
                "iqr": outliers_iqr,
                "zscore": outliers_zscore
            },
            "differencing": differencing_results,
            "seasonality": seasonality_info
        }
        
        logger.info(f"✓ Detailed exploration completed: {request.pipeline_id}")
        return JSONResponse(status_code=200, content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Detailed exploration error: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Detailed exploration failed: {str(e)}"}
        )

@app.post("/explore/differencing-plots")
async def get_differencing_plots(request: ExploreRequest):
    logger.info(f"[Differencing] Request for pipeline: {request.pipeline_id}")
    
    try:
        # Validate pipeline exists
        if request.pipeline_id not in pipelines:
            logger.error(f"[Differencing] Pipeline not found: {request.pipeline_id}")
            return JSONResponse(
                status_code=404,
                content={"error": "Pipeline not found"}
            )

        pipeline_obj = pipelines[request.pipeline_id]
        pipeline = pipeline_obj['pipeline']

        # Ensure data is loaded
        if pipeline.train is None:
            logger.info("[Differencing] Loading pipeline data...")
            pipeline.load_and_explore()

        if len(pipeline.train) == 0:
            logger.error("[Differencing] Training data is empty")
            return JSONResponse(
                status_code=400,
                content={"error": "Training data is empty"}
            )

        logger.info(f"[Differencing] Processing {len(pipeline.train)} data points")

        from statsmodels.tsa.stattools import acf, pacf, adfuller
        
        def safe_convert(val):
            """Safely convert numpy/pandas types to native Python types"""
            if val is None:
                return None
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, pd.Series):
                return val.tolist()
            try:
                if pd.isna(val):
                    return None
            except (ValueError, TypeError):
                pass
            if isinstance(val, (np.integer, np.int64, np.int32)):
                return int(val)
            if isinstance(val, (np.floating, np.float64, np.float32)):
                return float(val)
            if isinstance(val, np.bool_):
                return bool(val)
            return val

        # Original series
        original_series = pipeline.train.dropna()
        
        if len(original_series) < 10:
            logger.error("[Differencing] Not enough data points")
            return JSONResponse(
                status_code=400,
                content={"error": "Not enough data points (minimum 10 required)"}
            )
        
        # Differenced series
        differenced_series = original_series.diff().dropna()
        
        logger.info(f"[Differencing] Original: {len(original_series)}, Differenced: {len(differenced_series)}")
        
        # ADF tests for stationarity
        try:
            adf_orig = adfuller(original_series)
            adf_orig_stat = float(adf_orig[0])
            adf_orig_pval = float(adf_orig[1])
            is_stat_orig = bool(adf_orig_pval < 0.05)
            logger.info(f"[Differencing] Original ADF: stat={adf_orig_stat:.4f}, p={adf_orig_pval:.4f}, stationary={is_stat_orig}")
        except Exception as e:
            logger.error(f"[Differencing] ADF test failed for original: {e}")
            adf_orig_stat, adf_orig_pval, is_stat_orig = 0.0, 1.0, False
        
        try:
            adf_diff = adfuller(differenced_series)
            adf_diff_stat = float(adf_diff[0])
            adf_diff_pval = float(adf_diff[1])
            is_stat_diff = bool(adf_diff_pval < 0.05)
            logger.info(f"[Differencing] Differenced ADF: stat={adf_diff_stat:.4f}, p={adf_diff_pval:.4f}, stationary={is_stat_diff}")
        except Exception as e:
            logger.error(f"[Differencing] ADF test failed for differenced: {e}")
            adf_diff_stat, adf_diff_pval, is_stat_diff = 0.0, 1.0, False
        
        # ACF/PACF calculations
        max_lags = min(40, len(original_series) // 2)
        max_lags_diff = min(40, len(differenced_series) // 2)
        
        logger.info(f"[Differencing] Computing ACF/PACF with lags: orig={max_lags}, diff={max_lags_diff}")
        
        # ACF/PACF for original series
        try:
            acf_orig = acf(original_series, nlags=max_lags, alpha=0.05)
            acf_orig_vals = safe_convert(acf_orig[0]) if isinstance(acf_orig, tuple) else safe_convert(acf_orig)
            acf_orig_ci = [[safe_convert(ci[0]), safe_convert(ci[1])] for ci in acf_orig[1]] if isinstance(acf_orig, tuple) and len(acf_orig) > 1 else None
            logger.info(f"[Differencing] ACF original computed: {len(acf_orig_vals)} values")
        except Exception as e:
            logger.error(f"[Differencing] ACF original failed: {e}")
            acf_orig_vals = [1.0] + [0.0] * 20
            acf_orig_ci = None
        
        try:
            pacf_orig = pacf(original_series, nlags=max_lags, alpha=0.05)
            pacf_orig_vals = safe_convert(pacf_orig[0]) if isinstance(pacf_orig, tuple) else safe_convert(pacf_orig)
            pacf_orig_ci = [[safe_convert(ci[0]), safe_convert(ci[1])] for ci in pacf_orig[1]] if isinstance(pacf_orig, tuple) and len(pacf_orig) > 1 else None
            logger.info(f"[Differencing] PACF original computed: {len(pacf_orig_vals)} values")
        except Exception as e:
            logger.error(f"[Differencing] PACF original failed: {e}")
            pacf_orig_vals = [1.0] + [0.0] * 20
            pacf_orig_ci = None
        
        # ACF/PACF for differenced series
        try:
            acf_diff = acf(differenced_series, nlags=max_lags_diff, alpha=0.05)
            acf_diff_vals = safe_convert(acf_diff[0]) if isinstance(acf_diff, tuple) else safe_convert(acf_diff)
            acf_diff_ci = [[safe_convert(ci[0]), safe_convert(ci[1])] for ci in acf_diff[1]] if isinstance(acf_diff, tuple) and len(acf_diff) > 1 else None
            logger.info(f"[Differencing] ACF differenced computed: {len(acf_diff_vals)} values")
        except Exception as e:
            logger.error(f"[Differencing] ACF differenced failed: {e}")
            acf_diff_vals = [1.0] + [0.0] * 20
            acf_diff_ci = None
        
        try:
            pacf_diff = pacf(differenced_series, nlags=max_lags_diff, alpha=0.05)
            pacf_diff_vals = safe_convert(pacf_diff[0]) if isinstance(pacf_diff, tuple) else safe_convert(pacf_diff)
            pacf_diff_ci = [[safe_convert(ci[0]), safe_convert(ci[1])] for ci in pacf_diff[1]] if isinstance(pacf_diff, tuple) and len(pacf_diff) > 1 else None
            logger.info(f"[Differencing] PACF differenced computed: {len(pacf_diff_vals)} values")
        except Exception as e:
            logger.error(f"[Differencing] PACF differenced failed: {e}")
            pacf_diff_vals = [1.0] + [0.0] * 20
            pacf_diff_ci = None
        
        # Build response
        response_data = {
            "original": {
                "dates": original_series.index.strftime('%Y-%m-%d').tolist(),
                "values": safe_convert(original_series.values),
                "adf_statistic": adf_orig_stat,
                "adf_pvalue": adf_orig_pval,
                "is_stationary": is_stat_orig
            },
            "differenced": {
                "dates": differenced_series.index.strftime('%Y-%m-%d').tolist(),
                "values": safe_convert(differenced_series.values),
                "adf_statistic": adf_diff_stat,
                "adf_pvalue": adf_diff_pval,
                "is_stationary": is_stat_diff
            },
            "acf_original": {
                "lags": list(range(len(acf_orig_vals))),
                "acf": acf_orig_vals,
                "confidence_interval": acf_orig_ci
            },
            "pacf_original": {
                "lags": list(range(len(pacf_orig_vals))),
                "values": pacf_orig_vals,
                "confidence_interval": pacf_orig_ci
            },
            "acf_differenced": {
                "lags": list(range(len(acf_diff_vals))),
                "acf": acf_diff_vals,
                "confidence_interval": acf_diff_ci
            },
            "pacf_differenced": {
                "lags": list(range(len(pacf_diff_vals))),
                "values": pacf_diff_vals,
                "confidence_interval": pacf_diff_ci
            }
        }
        
        logger.info(f"[Differencing] ✓ Analysis completed successfully for {request.pipeline_id}")
        return JSONResponse(status_code=200, content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"[Differencing] Exception: {error_trace}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Differencing analysis failed: {str(e)}"}
        )

@app.post("/explore/differencing")
async def get_differencing_alias(request: ExploreRequest):
    """Alias endpoint for differencing-plots"""
    return await get_differencing_plots(request)

@app.post("/train")
async def train_models(request: TrainRequest):
    """Train selected models"""
    logger.info(f"Train request: {request.pipeline_id}, models: {request.models}")
    
    try:
        if request.pipeline_id not in pipelines:
            handle_error("Pipeline not found", 404)

        pipeline_obj = pipelines[request.pipeline_id]
        pipeline = pipeline_obj['pipeline']

        if pipeline.train is None:
            pipeline.load_and_explore()

        available_models = ['ARIMA', 'ARIMAX', 'SARIMA', 'SARIMAX', 'Transformer']
        invalid_models = [m for m in request.models if m not in available_models]
        if invalid_models:
            handle_error(f"Invalid models: {', '.join(invalid_models)}", 400)

        exog_models = ['ARIMAX', 'SARIMAX']
        requires_exog = any(m in request.models for m in exog_models)
        if requires_exog and (not pipeline.exog_train or len(pipeline.exog_train.columns) == 0):
            handle_error(f"Models requiring exogenous variables selected but none available", 400)

        trained_models = {}

        for model_name in request.models:
            start_time = time.time()
            try:
                logger.info(f"Training {model_name}...")
                
                if model_name == 'ARIMA':
                    metrics = pipeline.train_arima(use_grid_search=request.arima_grid_search)
                elif model_name == 'SARIMA':
                    metrics = pipeline.train_sarima(use_grid_search=request.sarima_grid_search)
                elif model_name == 'Transformer':
                    epochs = request.transformer_epochs if request.transformer_epochs else Config.TRANSFORMER_EPOCHS
                    metrics = pipeline.train_transformer(epochs=epochs)
                elif model_name == 'ARIMAX':
                    metrics = pipeline.train_arimax(use_grid_search=request.arima_grid_search)
                elif model_name == 'SARIMAX':
                    metrics = pipeline.train_sarimax(use_grid_search=request.sarima_grid_search)
                else:
                    continue

                training_time = time.time() - start_time
                trained_models[model_name] = {
                    'rmse': float(metrics.get('RMSE', 0)),
                    'mae': float(metrics.get('MAE', 0)),
                    'mape': float(metrics.get('MAPE', 0)),
                    'r2': float(metrics.get('R2', 0)),
                    'mse': float(metrics.get('MSE', 0)),
                    'training_time': round(training_time, 2),
                    'sample_size': len(pipeline.test)
                }
                
                logger.info(f"✓ {model_name} trained successfully")
                
            except Exception as model_error:
                logger.error(f"Error training {model_name}: {str(model_error)}")
                continue

        if len(trained_models) == 0:
            handle_error("All model training failed", 500)

        best_model = min(trained_models.items(), key=lambda x: x[1]['rmse'])[0]
        
        logger.info(f"✓ Training completed. Best model: {best_model}")

        return {
            "models": trained_models,
            "best_model": best_model
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Train error: {traceback.format_exc()}")
        handle_error(f"Training failed: {str(e)}", 500)

@app.post("/predict")
async def predict(request: PredictRequest):
    """Generate forecast"""
    logger.info(f"Predict request: {request.pipeline_id}, model: {request.model_name}, steps: {request.steps}")
    
    try:
        if request.pipeline_id not in pipelines:
            handle_error("Pipeline not found", 404)

        pipeline_obj = pipelines[request.pipeline_id]
        pipeline = pipeline_obj['pipeline']

        if request.model_name not in pipeline.fitted_models:
            handle_error(f"Model {request.model_name} not trained yet", 400)

        if request.steps < 1 or request.steps > 365:
            handle_error("Steps must be between 1 and 365", 400)

        forecast = pipeline.generate_future_forecast(
            model_name=request.model_name,
            steps=request.steps
        )

        historical_dates = pipeline.data.index.strftime('%Y-%m-%d').tolist()
        historical_values = pipeline.data.values.flatten().tolist()
        forecast_dates = forecast.index.strftime('%Y-%m-%d').tolist()
        forecast_values = forecast.values.flatten().tolist()

        if pipeline.test is not None and len(pipeline.test) > 0:
            residuals = pipeline.test - pipeline.forecasts.get(request.model_name, {}).get('forecast', pipeline.test)
            std_error = float(residuals.std())
        else:
            std_error = float(forecast.std())

        lower_bound = [float(v - 1.96 * std_error) for v in forecast_values]
        upper_bound = [float(v + 1.96 * std_error) for v in forecast_values]
        
        logger.info(f"✓ Forecast generated successfully")

        return {
            "historical_dates": historical_dates,
            "historical_values": historical_values,
            "forecast_dates": forecast_dates,
            "forecast_values": forecast_values,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "model_name": request.model_name,
            "steps": request.steps
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Predict error: {traceback.format_exc()}")
        handle_error(f"Forecast generation failed: {str(e)}", 500)

@app.post("/models/residual-analysis")
async def residual_analysis(request: ResidualRequest):
    """Get residual analysis for a model - FULLY FIXED"""
    logger.info(f"Residual analysis request: {request.pipeline_id}, model: {request.model_name}")
    
    try:
        # Validate pipeline exists
        if request.pipeline_id not in pipelines:
            logger.error(f"Pipeline not found: {request.pipeline_id}")
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": "Pipeline not found",
                    "data": None
                }
            )

        pipeline_obj = pipelines[request.pipeline_id]
        pipeline = pipeline_obj['pipeline']

        # Validate model is trained
        if request.model_name not in pipeline.fitted_models:
            logger.error(f"Model not trained: {request.model_name}")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Model {request.model_name} not trained yet",
                    "data": None
                }
            )

        model = pipeline.fitted_models[request.model_name]

        # Check if model supports residual analysis
        # FIXED: Normalize model name
        supported_models = ['ARIMA', 'ARIMAX', 'SARIMA', 'SARIMAX']
        model_name_upper = request.model_name.upper()
        
        if model_name_upper not in supported_models:
            logger.info(f"Model {request.model_name} does not support residual analysis")
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error": f"Residual analysis not available for {request.model_name} model type",
                    "data": None
                }
            )

        logger.info(f"Generating residual analysis for {request.model_name}...")
        
        # Get assumption tests
        try:
            assumptions = model.test_residual_assumptions()
            assumptions = convert_numpy_recursive(assumptions)
            logger.info(f"Assumption tests completed: {list(assumptions.keys())}")
        except Exception as e:
            logger.error(f"Error in assumption tests: {e}")
            assumptions = {
                "error": str(e),
                "normality": {"error": "Failed to compute"},
                "heteroscedasticity": {"error": "Failed to compute"},
                "white_noise": {"error": "Failed to compute"}
            }
        
        # Get plot data
        plot_data = None
        try:
            plot_data = model.get_residual_plot_data()
            plot_data = convert_numpy_recursive(plot_data)
            logger.info(f"Plot data generated: {list(plot_data.keys())}")
        except Exception as e:
            logger.error(f"Error generating plot data: {e}")
            plot_data = {
                "error": str(e),
                "residual_plot": {"dates": [], "residuals": []},
                "histogram": {"counts": [], "bins": []},
                "acf": {"lags": [], "acf": []}
            }
        
        # Get fitted vs actual - CRITICAL FIX
        fitted_data = None
        try:
            if hasattr(model, 'get_fitted_values_data'):
                # Use the dedicated method if available
                fitted_data = model.get_fitted_values_data()
                fitted_data = convert_numpy_recursive(fitted_data)
                logger.info(f"Fitted data from method: {len(fitted_data.get('dates', []))} points")
            elif hasattr(model, 'fitted_model') and model.fitted_model is not None:
                # Fallback to manual extraction
                fitted_values = model.fitted_model.fittedvalues
                actual_values = pipeline.train
                
                min_len = min(len(fitted_values), len(actual_values))
                
                fitted_data = {
                    "dates": actual_values.index[:min_len].strftime('%Y-%m-%d').tolist(),
                    "actual": convert_numpy_recursive(actual_values.values[:min_len].tolist()),
                    "fitted": convert_numpy_recursive(fitted_values.values[:min_len].tolist())
                }
                logger.info(f"Fitted data prepared: {len(fitted_data['dates'])} points")
            else:
                fitted_data = {"dates": [], "actual": [], "fitted": []}
        except Exception as e:
            logger.error(f"Error preparing fitted data: {e}")
            fitted_data = {"dates": [], "actual": [], "fitted": [], "error": str(e)}
        
        # Construct response
        response_data = {
            "model_name": request.model_name,
            "assumption_tests": assumptions,
            "fitted_data": fitted_data
        }
        
        # Add plot data if available
        if plot_data and "error" not in plot_data:
            response_data.update({
                "residual_plot": plot_data.get("residual_plot", {"dates": [], "residuals": []}),
                "histogram": plot_data.get("histogram", {"counts": [], "bins": []}),
                "acf": plot_data.get("acf", {"lags": [], "acf": []})
            })
        else:
            response_data.update({
                "residual_plot": {"dates": [], "residuals": []},
                "histogram": {"counts": [], "bins": []},
                "acf": {"lags": [], "acf": []}
            })
        
        # Check if we have any valid data
        has_valid_data = (
            (assumptions and not all("error" in v for v in assumptions.values() if isinstance(v, dict))) or
            (fitted_data and len(fitted_data.get("dates", [])) > 0) or
            (plot_data and not plot_data.get("error"))
        )
        
        if not has_valid_data:
            logger.warning(f"No valid residual data for {request.model_name}")
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error": "Could not generate residual analysis",
                    "data": response_data
                }
            )
        
        logger.info(f"✓ Residual analysis completed successfully for {request.model_name}")
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": response_data
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Residual analysis error: {error_trace}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Residual analysis failed: {str(e)}",
                "data": None
            }
        )

@app.get("/pipelines")
async def list_pipelines():
    """List all active pipelines"""
    logger.info(f"List pipelines: {len(pipelines)} active")
    
    pipeline_list = []
    for pid, pdata in pipelines.items():
        pipeline_list.append({
            "id": pid,
            "name": pdata['config'].get('pipeline_name', pid),
            "created_at": pdata.get('created_at'),
            "date_column": pdata['config'].get('date_column'),
            "value_column": pdata['config'].get('value_column')
        })

    return {
        "pipelines": pipeline_list,
        "count": len(pipelines),
        "current_pipeline": current_pipeline_id
    }

@app.delete("/pipeline/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Delete pipeline"""
    logger.info(f"Delete pipeline request: {pipeline_id}")
    
    try:
        if pipeline_id not in pipelines:
            handle_error("Pipeline not found", 404)

        filepath = pipelines[pipeline_id].get('filepath')
        del pipelines[pipeline_id]

        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                logger.warning(f"Could not delete temp file {filepath}: {e}")

        logger.info(f"Pipeline deleted: {pipeline_id}")
        
        return {
            "message": f"Pipeline {pipeline_id} deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        handle_error(f"Delete failed: {str(e)}", 500)

# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Run on startup"""
    import fastapi
    logger.info("=" * 60)
    logger.info("SURADATA API STARTING")
    logger.info("=" * 60)
    logger.info(f"FastAPI version: {fastapi.__version__}")
    logger.info(f"Config validated: {hasattr(Config, 'VALIDATION')}")
    logger.info(f"Min data points: {Config.VALIDATION['MIN_DATA_POINTS']}")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("=" * 60)
    logger.info("SURADATA API SHUTTING DOWN")
    logger.info(f"Active pipelines: {len(pipelines)}")
    
    # Cleanup temp files
    for pid, pdata in pipelines.items():
        filepath = pdata.get('filepath')
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up: {filepath}")
            except Exception as e:
                logger.warning(f"Could not cleanup {filepath}: {e}")
    
    logger.info("=" * 60)

# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("\n" + "=" * 60)
    logger.info("STARTING SURADATA API SERVER")
    logger.info("=" * 60)
    logger.info(f"Host: {Config.API_HOST}")
    logger.info(f"Port: {Config.API_PORT}")
    logger.info(f"Docs: http://localhost:{Config.API_PORT}/docs")
    logger.info("=" * 60 + "\n")
    
    uvicorn.run(
        "api:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True,
        log_level="info"
    )