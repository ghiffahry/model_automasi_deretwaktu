// ==================== CENTRALIZED STATE MANAGEMENT ====================

const StateManager = {
    /**
     * State schema definitions for validation
     */
    schemas: {
        pipeline: {
            required: ['pipeline_id', 'pipeline_name', 'date_column', 'value_column'],
            optional: ['frequency', 'exogenous_columns', 'data_shape', 'columns', 'created_at']
        },
        exploration: {
            required: ['basic_stats', 'time_series_data', 'statistical_tests'],
            optional: ['trend', 'seasonal', 'distribution', 'acf']
        },
        detailed_exploration: {
            required: ['outliers', 'differencing', 'seasonality'],
            optional: ['eda_report']
        },
        trained_models: {
            required: [],
            optional: []
        },
        forecast: {
            required: ['forecast_dates', 'forecast_values'],
            optional: ['historical_dates', 'historical_values', 'lower_bound', 'upper_bound']
        }
    },

    /**
     * Validate state object against schema
     */
    validate(stateKey, data) {
        const schema = this.schemas[stateKey];
        if (!schema) {
            console.warn(`No schema defined for: ${stateKey}`);
            return { valid: true, missing: [] };
        }

        if (!data || typeof data !== 'object') {
            return { valid: false, missing: ['data is null or not an object'] };
        }

        const missing = schema.required.filter(field => {
            return !Utils.isValidValue(Utils.getNestedValue(data, field));
        });

        return {
            valid: missing.length === 0,
            missing: missing
        };
    },

    /**
     * Save pipeline data with validation
     */
    savePipeline(data) {
        const validation = this.validate('pipeline', data);
        
        if (!validation.valid) {
            console.error('Invalid pipeline data:', validation.missing);
            Utils.showToast('Invalid pipeline data: missing ' + validation.missing.join(', '), 'error');
            return false;
        }

        const normalized = {
            pipeline_id: data.pipeline_id,
            pipeline_name: data.pipeline_name,
            date_column: data.date_column,
            value_column: data.value_column,
            frequency: data.frequency || 'D',
            exogenous_columns: data.exogenous_columns || [],
            data_shape: {
                rows: Utils.getNestedValue(data, 'data_shape.rows', 0),
                columns: Utils.getNestedValue(data, 'data_shape.columns', 0)
            },
            columns: data.columns || [],
            created_at: data.created_at || new Date().toISOString()
        };

        return Utils.saveToMemory('current_pipeline', normalized);
    },

    /**
     * Get pipeline data with validation
     */
    getPipeline() {
        const data = Utils.getFromMemory('current_pipeline');
        
        if (!data) {
            console.warn('No pipeline data found in memory');
            return null;
        }

        const validation = this.validate('pipeline', data);
        if (!validation.valid) {
            console.error('Corrupted pipeline data:', validation.missing);
            return null;
        }

        return data;
    },

    /**
     * Save exploration results with validation
     */
    saveExploration(data) {
        const validation = this.validate('exploration', data);
        
        if (!validation.valid) {
            console.error('Invalid exploration data:', validation.missing);
            // Don't show toast, just log - exploration might be partial
        }

        const normalized = {
            basic_stats: {
                mean: Utils.getNestedValue(data, 'basic_stats.mean', 0),
                std: Utils.getNestedValue(data, 'basic_stats.std', 0),
                min: Utils.getNestedValue(data, 'basic_stats.min', 0),
                max: Utils.getNestedValue(data, 'basic_stats.max', 0),
                count: Utils.getNestedValue(data, 'basic_stats.count', 0)
            },
            time_series_data: {
                dates: Utils.getNestedValue(data, 'time_series_data.dates', []),
                values: Utils.getNestedValue(data, 'time_series_data.values', [])
            },
            trend: {
                dates: Utils.getNestedValue(data, 'trend.dates', []),
                values: Utils.getNestedValue(data, 'trend.values', [])
            },
            seasonal: {
                dates: Utils.getNestedValue(data, 'seasonal.dates', []),
                values: Utils.getNestedValue(data, 'seasonal.values', [])
            },
            statistical_tests: {
                adf_statistic: Utils.getNestedValue(data, 'statistical_tests.adf_statistic', 0),
                adf_pvalue: Utils.getNestedValue(data, 'statistical_tests.adf_pvalue', 1),
                is_stationary: Utils.getNestedValue(data, 'statistical_tests.is_stationary', false),
                has_seasonality: Utils.getNestedValue(data, 'statistical_tests.has_seasonality', false)
            },
            distribution: {
                bins: Utils.getNestedValue(data, 'distribution.bins', []),
                counts: Utils.getNestedValue(data, 'distribution.counts', [])
            },
            acf: {
                lags: Utils.getNestedValue(data, 'acf.lags', []),
                values: Utils.getNestedValue(data, 'acf.values', []),
                labels: Utils.getNestedValue(data, 'acf.labels', null)
            }
        };

        return Utils.saveToMemory('exploration_results', normalized);
    },

    /**
     * Get exploration results
     */
    getExploration() {
        return Utils.getFromMemory('exploration_results');
    },

    /**
     * Save detailed exploration with validation
     */
    saveDetailedExploration(data) {
        const normalized = {
            outliers: {
                iqr: {
                    outlier_count: Utils.getNestedValue(data, 'outliers.iqr.outlier_count', 0),
                    outlier_percentage: Utils.getNestedValue(data, 'outliers.iqr.outlier_percentage', 0)
                },
                zscore: {
                    outlier_count: Utils.getNestedValue(data, 'outliers.zscore.outlier_count', 0),
                    outlier_percentage: Utils.getNestedValue(data, 'outliers.zscore.outlier_percentage', 0)
                }
            },
            differencing: {
                adf_original: Utils.getNestedValue(data, 'differencing.adf_original', 1.0),
                adf_differenced: Utils.getNestedValue(data, 'differencing.adf_differenced', 1.0),
                is_stationary_after_diff: Utils.getNestedValue(data, 'differencing.is_stationary_after_diff', false)
            },
            seasonality: {
                period: Utils.getNestedValue(data, 'seasonality.period', null),
                strength: Utils.getNestedValue(data, 'seasonality.strength', 0)
            },
            eda_report: Utils.getNestedValue(data, 'eda_report', {})
        };

        return Utils.saveToMemory('detailed_exploration', normalized);
    },

    /**
     * Get detailed exploration
     */
    getDetailedExploration() {
        return Utils.getFromMemory('detailed_exploration');
    },

    /**
     * Save trained models with validation
     */
    saveTrainedModels(data) {
        if (!data || typeof data !== 'object') {
            console.error('Invalid trained models data');
            return false;
        }

        const normalized = {};

        Object.entries(data).forEach(([modelName, metrics]) => {
            normalized[modelName] = {
                rmse: Utils.getNestedValue(metrics, 'rmse', 0),
                mae: Utils.getNestedValue(metrics, 'mae', 0),
                mape: Utils.getNestedValue(metrics, 'mape', 0),
                r2: Utils.getNestedValue(metrics, 'r2', 0),
                training_time: Utils.getNestedValue(metrics, 'training_time', 0),
                sample_size: Utils.getNestedValue(metrics, 'sample_size', null)
            };
        });

        return Utils.saveToMemory('trained_models', normalized);
    },

    /**
     * Get trained models
     */
    getTrainedModels() {
        return Utils.getFromMemory('trained_models') || {};
    },

    /**
     * Save forecast results with validation
     */
    saveForecast(data) {
        const validation = this.validate('forecast', data);
        
        if (!validation.valid) {
            console.error('Invalid forecast data:', validation.missing);
            return false;
        }

        const normalized = {
            historical_dates: Utils.getNestedValue(data, 'historical_dates', []),
            historical_values: Utils.getNestedValue(data, 'historical_values', []),
            forecast_dates: data.forecast_dates || [],
            forecast_values: data.forecast_values || [],
            lower_bound: Utils.getNestedValue(data, 'lower_bound', []),
            upper_bound: Utils.getNestedValue(data, 'upper_bound', []),
            model_name: Utils.getNestedValue(data, 'model_name', 'Unknown'),
            steps: Utils.getNestedValue(data, 'steps', 0)
        };

        return Utils.saveToMemory('forecast_results', normalized);
    },

    /**
     * Get forecast results
     */
    getForecast() {
        return Utils.getFromMemory('forecast_results');
    },

    /**
     * Clear all application state
     */
    clearAll() {
        const confirmed = confirm('This will clear all data. Are you sure?');
        if (!confirmed) return false;

        Utils.removeFromMemory('current_pipeline');
        Utils.removeFromMemory('exploration_results');
        Utils.removeFromMemory('detailed_exploration');
        Utils.removeFromMemory('trained_models');
        Utils.removeFromMemory('forecast_results');
        Utils.removeFromMemory('residual_analysis');

        console.log('✓ All state cleared');
        Utils.showToast('All data cleared', 'success');
        return true;
    },

    /**
     * Export all state as JSON
     */
    exportState() {
        const state = {
            pipeline: this.getPipeline(),
            exploration: this.getExploration(),
            detailed_exploration: this.getDetailedExploration(),
            trained_models: this.getTrainedModels(),
            forecast: this.getForecast(),
            exported_at: new Date().toISOString(),
            version: '1.0.0'
        };

        const filename = `suradata_state_${new Date().toISOString().split('T')[0]}.json`;
        Utils.downloadJSON(state, filename);
        
        console.log('✓ State exported');
        return state;
    },

    /**
     * Import state from JSON
     */
    async importState(file) {
        try {
            const text = await file.text();
            const state = JSON.parse(text);

            if (!state.version) {
                throw new Error('Invalid state file: missing version');
            }

            let imported = 0;

            if (state.pipeline) {
                if (this.savePipeline(state.pipeline)) imported++;
            }

            if (state.exploration) {
                if (this.saveExploration(state.exploration)) imported++;
            }

            if (state.detailed_exploration) {
                if (this.saveDetailedExploration(state.detailed_exploration)) imported++;
            }

            if (state.trained_models) {
                if (this.saveTrainedModels(state.trained_models)) imported++;
            }

            if (state.forecast) {
                if (this.saveForecast(state.forecast)) imported++;
            }

            Utils.showToast(`Imported ${imported} state objects`, 'success');
            console.log(`✓ Imported ${imported} objects from state file`);

            return true;
        } catch (error) {
            console.error('Import error:', error);
            Utils.showToast('Failed to import state: ' + error.message, 'error');
            return false;
        }
    },

    /**
     * Get state summary
     */
    getSummary() {
        const pipeline = this.getPipeline();
        const exploration = this.getExploration();
        const trainedModels = this.getTrainedModels();
        const forecast = this.getForecast();

        return {
            has_pipeline: !!pipeline,
            has_exploration: !!exploration,
            has_trained_models: Object.keys(trainedModels).length > 0,
            has_forecast: !!forecast,
            pipeline_id: pipeline?.pipeline_id || null,
            model_count: Object.keys(trainedModels).length,
            memory_stats: Utils.getMemoryStats()
        };
    },

    /**
     * Check if ready for next step
     */
    canExplore() {
        const pipeline = this.getPipeline();
        return !!pipeline && !!pipeline.pipeline_id;
    },

    canTrain() {
        const pipeline = this.getPipeline();
        const exploration = this.getExploration();
        return this.canExplore() && !!exploration;
    },

    canForecast() {
        const models = this.getTrainedModels();
        return this.canTrain() && Object.keys(models).length > 0;
    }
};

// Make StateManager available globally
window.StateManager = StateManager;