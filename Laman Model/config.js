// ==================== CONFIGURATION (ENHANCED) ====================

const CONFIG = {
    // API Configuration
    API: {
        BASE_URL: 'http://localhost:8000',
        TIMEOUT: 30000,
        RETRY_ATTEMPTS: 3
    },

    // Chart Configuration
    CHARTS: {
        PLOTLY_CONFIG: {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        },
        PLOTLY_LAYOUT_DARK: {
            plot_bgcolor: '#0f2744',
            paper_bgcolor: 'rgba(15, 39, 68, 0.8)',
            font: { 
                color: '#f5f7fa',
                family: 'Segoe UI, sans-serif'
            },
            xaxis: { 
                gridcolor: 'rgba(33, 150, 243, 0.2)',
                color: '#cbd5e1'
            },
            yaxis: { 
                gridcolor: 'rgba(33, 150, 243, 0.2)',
                color: '#cbd5e1'
            },
            showlegend: true,
            legend: {
                bgcolor: 'rgba(10, 22, 40, 0.8)',
                bordercolor: 'rgba(33, 150, 243, 0.2)',
                borderwidth: 1
            }
        },
        CHART_JS_DEFAULTS: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { 
                        color: '#f5f7fa',
                        font: {
                            size: 12,
                            family: 'Segoe UI, sans-serif'
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { color: '#cbd5e1' },
                    grid: { color: 'rgba(33, 150, 243, 0.2)' }
                },
                x: {
                    ticks: { color: '#cbd5e1' },
                    grid: { color: 'rgba(33, 150, 243, 0.2)' }
                }
            }
        }
    },

    // Model Configuration
    MODELS: {
        AVAILABLE: [
            {
                id: 'ARIMA',
                name: 'ARIMA',
                icon: '📈',
                description: 'AutoRegressive Integrated Moving Average',
                requiresExogenous: false,
                category: 'classical',
                complexity: 'medium'
            },
            {
                id: 'ARIMAX',
                name: 'ARIMAX',
                icon: '📊',
                description: 'ARIMA with Exogenous Variables',
                requiresExogenous: true,
                category: 'classical',
                complexity: 'medium'
            },
            {
                id: 'SARIMA',
                name: 'SARIMA',
                icon: '🌊',
                description: 'Seasonal ARIMA for periodic patterns',
                requiresExogenous: false,
                category: 'classical',
                complexity: 'high'
            },
            {
                id: 'SARIMAX',
                name: 'SARIMAX',
                icon: '🎯',
                description: 'Seasonal ARIMA with Exogenous Variables',
                requiresExogenous: true,
                category: 'classical',
                complexity: 'high'
            },
            {
                id: 'Transformer',
                name: 'Transformer',
                icon: '🤖',
                description: 'Deep Learning with Attention Mechanism',
                requiresExogenous: false,
                category: 'deep_learning',
                complexity: 'very_high'
            }
        ],
        DEFAULT_SELECTED: ['ARIMA', 'SARIMA']
    },

    // Upload Configuration
    UPLOAD: {
        MAX_FILE_SIZE: 50 * 1024 * 1024, // 50MB
        ALLOWED_EXTENSIONS: ['.csv'],
        DEFAULT_FREQUENCY: 'D',
        FREQUENCIES: [
            { value: 'D', label: 'Daily', description: 'Harian (setiap hari)' },
            { value: 'W', label: 'Weekly', description: 'Mingguan (setiap minggu)' },
            { value: 'M', label: 'Monthly', description: 'Bulanan (setiap bulan)' },
            { value: 'Q', label: 'Quarterly', description: 'Kuartalan (setiap 3 bulan)' },
            { value: 'Y', label: 'Yearly', description: 'Tahunan (setiap tahun)' }
        ]
    },

    // Forecast Configuration
    FORECAST: {
        DEFAULT_STEPS: 30,
        MIN_STEPS: 1,
        MAX_STEPS: 365,
        CONFIDENCE_LEVEL: 0.95
    },

    // UI Configuration
    UI: {
        TOAST_DURATION: 3000,
        ANIMATION_DURATION: 500,
        DEBOUNCE_DELAY: 300,
        MAX_PREVIEW_ROWS: 10
    },

    // Validation Rules
    VALIDATION: {
        MIN_DATA_POINTS: 30,
        MIN_TRAIN_RATIO: 0.6,
        MAX_TRAIN_RATIO: 0.9,
        MAX_MISSING_VALUES_RATIO: 0.1
    },

    // Error Messages
    ERRORS: {
        NO_FILE: 'No file selected',
        INVALID_FILE_TYPE: 'Invalid file type. Only CSV files are allowed',
        FILE_TOO_LARGE: 'File is too large',
        EMPTY_FILE: 'File is empty',
        INSUFFICIENT_DATA: 'Insufficient data points',
        NO_PIPELINE: 'No active pipeline found',
        NO_MODEL_SELECTED: 'Please select at least one model',
        TRAINING_FAILED: 'Model training failed',
        FORECAST_FAILED: 'Forecast generation failed',
        API_ERROR: 'API connection error',
        TIMEOUT: 'Request timeout'
    },

    // Success Messages
    SUCCESS: {
        FILE_UPLOADED: 'File uploaded successfully',
        PIPELINE_CREATED: 'Pipeline created successfully',
        EXPLORATION_COMPLETE: 'Data exploration completed',
        MODELS_TRAINED: 'Models trained successfully',
        FORECAST_GENERATED: 'Forecast generated successfully',
        DATA_EXPORTED: 'Data exported successfully'
    },

    // Feature Flags
    FEATURES: {
        ENABLE_GRID_SEARCH: true,
        ENABLE_RESIDUAL_ANALYSIS: true,
        ENABLE_DETAILED_EXPLORATION: true,
        ENABLE_MODEL_COMPARISON: true,
        ENABLE_EXPORT: true
    }
};

// Freeze config to prevent modifications
Object.freeze(CONFIG);
Object.freeze(CONFIG.API);
Object.freeze(CONFIG.CHARTS);
Object.freeze(CONFIG.MODELS);
Object.freeze(CONFIG.UPLOAD);
Object.freeze(CONFIG.FORECAST);
Object.freeze(CONFIG.UI);
Object.freeze(CONFIG.VALIDATION);
Object.freeze(CONFIG.ERRORS);
Object.freeze(CONFIG.SUCCESS);
Object.freeze(CONFIG.FEATURES);

// Make config available globally
window.CONFIG = CONFIG;

console.log('✓ Configuration loaded');