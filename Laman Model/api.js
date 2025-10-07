// ==================== COMPLETE API HANDLER WITH VALIDATION (FULLY FIXED) ====================

const API = {
    baseURL: CONFIG.API.BASE_URL,
    timeout: CONFIG.API.TIMEOUT,
    
    /**
     * Validate response structure
     */
    validateResponse(data, requiredFields = []) {
        if (!data || typeof data !== 'object') {
            return { valid: false, error: 'Invalid response format' };
        }
        
        const missing = requiredFields.filter(field => {
            const parts = field.split('.');
            let current = data;
            for (const part of parts) {
                if (!current || !(part in current)) return true;
                current = current[part];
            }
            return false;
        });
        
        if (missing.length > 0) {
            return { valid: false, error: `Missing fields: ${missing.join(', ')}` };
        }
        
        return { valid: true };
    },
    
    /**
     * Safe get nested property
     */
    getNestedValue(obj, path, defaultValue = null) {
        const parts = path.split('.');
        let current = obj;
        for (const part of parts) {
            if (!current || typeof current !== 'object' || !(part in current)) {
                return defaultValue;
            }
            current = current[part];
        }
        return current !== undefined ? current : defaultValue;
    },
    
    /**
     * Generic request with comprehensive error handling
     */
    async request(endpoint, options = {}) {
        const { method = 'GET', data = null, headers = {}, timeout = this.timeout } = options;
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);
        
        const config = {
            method,
            headers: { ...headers },
            signal: controller.signal
        };
        
        if (data) {
            if (data instanceof FormData) {
                config.body = data;
            } else {
                config.headers['Content-Type'] = 'application/json';
                config.body = JSON.stringify(data);
            }
        }
        
        try {
            const response = await fetch(`${this.baseURL}${endpoint}`, config);
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
                
                try {
                    const errorData = await response.json();
                    errorMessage = this.extractErrorMessage(errorData) || errorMessage;
                } catch (e) {
                    // Response not JSON, use status text
                }
                
                throw new Error(errorMessage);
            }
            
            const result = await response.json();
            return { success: true, data: result };
            
        } catch (error) {
            clearTimeout(timeoutId);
            
            if (error.name === 'AbortError') {
                return { success: false, error: 'Request timeout' };
            }
            
            console.error(`API Error [${method} ${endpoint}]:`, error);
            return { 
                success: false, 
                error: error.message || 'Network error occurred'
            };
        }
    },
    
    /**
     * Extract error message from various response formats
     */
    extractErrorMessage(errorData) {
        if (typeof errorData === 'string') return errorData;
        
        const errorFields = [
            'detail.error',
            'detail',
            'error',
            'message',
            'errors',
            'error_message'
        ];
        
        for (const field of errorFields) {
            const value = this.getNestedValue(errorData, field);
            if (value) {
                if (typeof value === 'string') return value;
                if (Array.isArray(value)) return value.join(', ');
                if (typeof value === 'object') return JSON.stringify(value);
            }
        }
        
        return null;
    },
    
    /**
     * Upload file with complete validation
     */
    async uploadFile(file, config) {
        if (!file) {
            return { success: false, error: 'No file provided' };
        }
        
        if (!file.name.endsWith('.csv')) {
            return { success: false, error: 'Only CSV files allowed' };
        }
        
        if (file.size === 0) {
            return { success: false, error: 'File is empty' };
        }
        
        if (file.size > CONFIG.UPLOAD.MAX_FILE_SIZE) {
            return { 
                success: false, 
                error: `File too large. Max: ${Utils.formatFileSize(CONFIG.UPLOAD.MAX_FILE_SIZE)}` 
            };
        }
        
        if (!config.date_column || !config.value_column) {
            return { success: false, error: 'Date and value columns required' };
        }
        
        if (!config.pipeline_name || config.pipeline_name.trim() === '') {
            return { success: false, error: 'Pipeline name required' };
        }
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('config', JSON.stringify(config));
        
        const result = await this.request('/upload', {
            method: 'POST',
            data: formData,
            timeout: 60000
        });
        
        if (result.success && result.data) {
            const validation = this.validateResponse(result.data, ['pipeline_id', 'date_column', 'value_column']);
            if (!validation.valid) {
                console.error('Invalid upload response:', result.data);
                return { success: false, error: 'Invalid server response: ' + validation.error };
            }
            
            const normalized = {
                pipeline_id: result.data.pipeline_id,
                pipeline_name: config.pipeline_name,
                date_column: result.data.date_column,
                value_column: result.data.value_column,
                frequency: config.frequency || 'D',
                exogenous_columns: config.exogenous_columns || [],
                data_shape: {
                    rows: this.getNestedValue(result.data, 'data_shape.rows', 0),
                    columns: this.getNestedValue(result.data, 'data_shape.columns', 0)
                },
                columns: result.data.columns || [],
                created_at: new Date().toISOString()
            };
            
            return { success: true, data: normalized };
        }
        
        return result;
    },
    
    /**
     * Run basic exploration
     */
    async runExploration(pipelineId) {
        if (!pipelineId) {
            return { success: false, error: 'Pipeline ID required' };
        }
        
        const result = await this.request('/explore', {
            method: 'POST',
            data: { pipeline_id: pipelineId },
            timeout: 60000
        });
        
        if (result.success && result.data) {
            const normalized = {
                basic_stats: {
                    mean: this.getNestedValue(result.data, 'basic_stats.mean', 0),
                    std: this.getNestedValue(result.data, 'basic_stats.std', 0),
                    min: this.getNestedValue(result.data, 'basic_stats.min', 0),
                    max: this.getNestedValue(result.data, 'basic_stats.max', 0),
                    median: this.getNestedValue(result.data, 'basic_stats.median', 0)
                },
                time_series_data: {
                    dates: this.getNestedValue(result.data, 'time_series_data.dates', []),
                    values: this.getNestedValue(result.data, 'time_series_data.values', [])
                },
                trend: {
                    dates: this.getNestedValue(result.data, 'trend.dates', []),
                    values: this.getNestedValue(result.data, 'trend.values', [])
                },
                seasonal: {
                    dates: this.getNestedValue(result.data, 'seasonal.dates', []),
                    values: this.getNestedValue(result.data, 'seasonal.values', [])
                },
                statistical_tests: {
                    adf_statistic: this.getNestedValue(result.data, 'statistical_tests.adf_statistic', 0),
                    adf_pvalue: this.getNestedValue(result.data, 'statistical_tests.adf_pvalue', 1),
                    is_stationary: this.getNestedValue(result.data, 'statistical_tests.is_stationary', false),
                    has_seasonality: this.getNestedValue(result.data, 'statistical_tests.has_seasonality', false)
                },
                distribution: {
                    bins: this.getNestedValue(result.data, 'distribution.bins', []),
                    counts: this.getNestedValue(result.data, 'distribution.counts', [])
                },
                acf: {
                    lags: this.getNestedValue(result.data, 'acf.lags', []),
                    values: this.getNestedValue(result.data, 'acf.values', []),
                    labels: this.getNestedValue(result.data, 'acf.labels', null)
                }
            };
            
            return { success: true, data: normalized };
        }
        
        return result;
    },
    
    /**
     * Run detailed exploration
     */
    async getDetailedExploration(pipelineId) {
        if (!pipelineId) {
            return { success: false, error: 'Pipeline ID required' };
        }
        
        const result = await this.request('/explore/detailed', {
            method: 'POST',
            data: { pipeline_id: pipelineId },
            timeout: 60000
        });
        
        if (result.success && result.data) {
            const normalized = {
                outliers: {
                    iqr: {
                        outlier_count: this.getNestedValue(result.data, 'outliers.iqr.outlier_count', 0),
                        outlier_percentage: this.getNestedValue(result.data, 'outliers.iqr.outlier_percentage', 0)
                    },
                    zscore: {
                        outlier_count: this.getNestedValue(result.data, 'outliers.zscore.outlier_count', 0),
                        outlier_percentage: this.getNestedValue(result.data, 'outliers.zscore.outlier_percentage', 0)
                    }
                },
                differencing: {
                    adf_original: this.getNestedValue(result.data, 'differencing.adf_original', 1),
                    adf_differenced: this.getNestedValue(result.data, 'differencing.adf_differenced', 1),
                    is_stationary_after_diff: this.getNestedValue(result.data, 'differencing.is_stationary_after_diff', false)
                },
                seasonality: {
                    period: this.getNestedValue(result.data, 'seasonality.period', 'N/A'),
                    strength: this.getNestedValue(result.data, 'seasonality.strength', 0)
                },
                eda_report: {
                    basic_stats: this.getNestedValue(result.data, 'eda_report.basic_stats', {}),
                    trend_info: this.getNestedValue(result.data, 'eda_report.trend_info', {})
                }
            };
            
            return { success: true, data: normalized };
        }
        
        return result;
    },
    
    /**
     * FIXED: Get differencing plots data
     * This uses the detailed exploration endpoint which already contains differencing data
     */
    async getDifferencingPlots(pipelineId) {
        if (!pipelineId) {
            return { success: false, error: 'Pipeline ID required' };
        }
        
        const result = await this.request('/explore/differencing', {
            method: 'POST',
            data: { pipeline_id: pipelineId },
            timeout: 60000
        });
        
        if (result.success && result.data) {
            // Normalize the differencing response
            const normalized = {
                original: {
                    dates: this.getNestedValue(result.data, 'original.dates', []),
                    values: this.getNestedValue(result.data, 'original.values', []),
                    is_stationary: this.getNestedValue(result.data, 'original.is_stationary', false),
                    adf_statistic: this.getNestedValue(result.data, 'original.adf_statistic', 0),
                    adf_pvalue: this.getNestedValue(result.data, 'original.adf_pvalue', 1)
                },
                differenced: {
                    dates: this.getNestedValue(result.data, 'differenced.dates', []),
                    values: this.getNestedValue(result.data, 'differenced.values', []),
                    is_stationary: this.getNestedValue(result.data, 'differenced.is_stationary', false),
                    adf_statistic: this.getNestedValue(result.data, 'differenced.adf_statistic', 0),
                    adf_pvalue: this.getNestedValue(result.data, 'differenced.adf_pvalue', 1)
                },
                acf_original: {
                    lags: this.getNestedValue(result.data, 'acf_original.lags', []),
                    values: this.getNestedValue(result.data, 'acf_original.values', [])
                },
                pacf_original: {
                    lags: this.getNestedValue(result.data, 'pacf_original.lags', []),
                    values: this.getNestedValue(result.data, 'pacf_original.values', [])
                },
                acf_differenced: {
                    lags: this.getNestedValue(result.data, 'acf_differenced.lags', []),
                    values: this.getNestedValue(result.data, 'acf_differenced.values', [])
                },
                pacf_differenced: {
                    lags: this.getNestedValue(result.data, 'pacf_differenced.lags', []),
                    values: this.getNestedValue(result.data, 'pacf_differenced.values', [])
                }
            };
            
            return { success: true, data: normalized };
        }
        
        return result;
    },
    
    /**
     * Train models
     */
    async trainModels(pipelineId, models, options = {}) {
        if (!pipelineId) {
            return { success: false, error: 'Pipeline ID required' };
        }
        
        if (!models || !Array.isArray(models) || models.length === 0) {
            return { success: false, error: 'At least one model required' };
        }
        
        const validModels = CONFIG.MODELS.AVAILABLE.map(m => m.id);
        const invalidModels = models.filter(m => !validModels.includes(m));
        if (invalidModels.length > 0) {
            return { success: false, error: `Invalid models: ${invalidModels.join(', ')}` };
        }
        
        const result = await this.request('/train', {
            method: 'POST',
            data: {
                pipeline_id: pipelineId,
                models: models,
                arima_grid_search: options.arimaGrid !== false,
                sarima_grid_search: options.sarimaGrid !== false,
                transformer_epochs: options.transformerEpochs || null
            },
            timeout: 300000
        });
        
        if (result.success && result.data) {
            const normalized = { 
                models: {}, 
                residual_analysis: {},
                best_model: this.getNestedValue(result.data, 'best_model', null)
            };
            
            const rawModels = this.getNestedValue(result.data, 'models', {});
            Object.entries(rawModels).forEach(([name, metrics]) => {
                normalized.models[name] = {
                    rmse: this.getNestedValue(metrics, 'rmse', 0),
                    mae: this.getNestedValue(metrics, 'mae', 0),
                    mape: this.getNestedValue(metrics, 'mape', 0),
                    r2: this.getNestedValue(metrics, 'r2', 0),
                    training_time: this.getNestedValue(metrics, 'training_time', 0),
                    sample_size: this.getNestedValue(metrics, 'sample_size', null),
                    parameters: this.getNestedValue(metrics, 'parameters', null)
                };
            });
            
            const residuals = this.getNestedValue(result.data, 'residual_analysis', {});
            if (Object.keys(residuals).length > 0) {
                normalized.residual_analysis = residuals;
            }
            
            return { success: true, data: normalized };
        }
        
        return result;
    },
    
    /**
     * Generate forecast
     */
    async generateForecast(pipelineId, modelName, steps) {
        if (!pipelineId || !modelName) {
            return { success: false, error: 'Pipeline ID and model name required' };
        }
        
        if (!steps || steps < CONFIG.FORECAST.MIN_STEPS || steps > CONFIG.FORECAST.MAX_STEPS) {
            return { 
                success: false, 
                error: `Steps must be ${CONFIG.FORECAST.MIN_STEPS}-${CONFIG.FORECAST.MAX_STEPS}` 
            };
        }
        
        const result = await this.request('/predict', {
            method: 'POST',
            data: {
                pipeline_id: pipelineId,
                model_name: modelName,
                steps: parseInt(steps)
            },
            timeout: 60000
        });
        
        if (result.success && result.data) {
            const normalized = {
                historical_dates: this.getNestedValue(result.data, 'historical_dates', []),
                historical_values: this.getNestedValue(result.data, 'historical_values', []),
                forecast_dates: this.getNestedValue(result.data, 'forecast_dates', []),
                forecast_values: this.getNestedValue(result.data, 'forecast_values', []),
                lower_bound: this.getNestedValue(result.data, 'lower_bound', []),
                upper_bound: this.getNestedValue(result.data, 'upper_bound', [])
            };
            
            if (normalized.forecast_dates.length === 0 || normalized.forecast_values.length === 0) {
                return { success: false, error: 'Empty forecast returned from server' };
            }
            
            return { success: true, data: normalized };
        }
        
        return result;
    },
    
    /**
     * FIXED: Get residual analysis with proper fitted data extraction
     */
    async getResidualAnalysis(pipelineId, modelName) {
        if (!pipelineId || !modelName) {
            return { success: false, error: 'Pipeline ID and model name required' };
        }
        
        const result = await this.request('/models/residual-analysis', {
            method: 'POST',
            data: { pipeline_id: pipelineId, model_name: modelName },
            timeout: 60000
        });
        
        if (result.success && result.data) {
            const normalized = {
                assumption_tests: this.getNestedValue(result.data, 'assumption_tests', {}),
                plots: this.getNestedValue(result.data, 'plots', {}),
                residual_plot: this.getNestedValue(result.data, 'residual_plot', {}),
                histogram: this.getNestedValue(result.data, 'histogram', {}),
                acf: this.getNestedValue(result.data, 'acf', {}),
                fitted_data: {
                    dates: this.getNestedValue(result.data, 'fitted_data.dates', []),
                    actual: this.getNestedValue(result.data, 'fitted_data.actual', []),
                    fitted: this.getNestedValue(result.data, 'fitted_data.fitted', [])
                }
            };
            
            return { success: true, data: normalized };
        }
        
        return result;
    },
    
    /**
     * Get pipelines
     */
    async getPipelines() {
        const result = await this.request('/pipelines');
        
        if (result.success && result.data) {
            return { 
                success: true, 
                data: { 
                    pipelines: this.getNestedValue(result.data, 'pipelines', []) 
                } 
            };
        }
        
        return result;
    },
    
    /**
     * Delete pipeline
     */
    async deletePipeline(pipelineId) {
        if (!pipelineId) {
            return { success: false, error: 'Pipeline ID required' };
        }
        
        return await this.request(`/pipeline/${pipelineId}`, {
            method: 'DELETE'
        });
    },
    
    /**
     * Health check
     */
    async healthCheck() {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);
            
            const response = await fetch(`${this.baseURL}/health`, {
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (response.ok) {
                const data = await response.json();
                return { success: true, data };
            }
            
            return { success: false, error: `HTTP ${response.status}` };
            
        } catch (error) {
            if (error.name === 'AbortError') {
                return { success: false, error: 'Timeout' };
            }
            return { success: false, error: error.message || 'Connection failed' };
        }
    }
};

window.API = API;