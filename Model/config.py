class Config:
    # Data Settings
    TRAIN_TEST_SPLIT = 0.8
    RANDOM_STATE = 86
    
    # Stationarity Test
    ADF_ALPHA = 0.05  
    
    # ARIMA Settings
    ARIMA_MAX_P = 5
    ARIMA_MAX_D = 2
    ARIMA_MAX_Q = 5
    
    # SARIMA Settings
    SARIMA_MAX_P = 3
    SARIMA_MAX_D = 2
    SARIMA_MAX_Q = 3
    SARIMA_MAX_SEASONAL_P = 2
    SARIMA_MAX_SEASONAL_D = 2
    SARIMA_MAX_SEASONAL_Q = 2
    SARIMA_SEASONAL_PERIOD = 12 
    
    # Transformer Settings
    TRANSFORMER_WINDOW_SIZE = 30
    TRANSFORMER_HIDDEN_DIM = 64
    TRANSFORMER_NUM_LAYERS = 2
    TRANSFORMER_NUM_HEADS = 4
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_LEARNING_RATE = 0.001
    TRANSFORMER_BATCH_SIZE = 32
    TRANSFORMER_EPOCHS = 100
    TRANSFORMER_PATIENCE = 10  
    
    # Evaluation
    METRICS = ['RMSE', 'MAE', 'MAPE']
    
    # API Settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    MODEL_SAVE_PATH = "./saved_models/"
    
    # Visualization
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'
    FIGSIZE = (14, 6)
    DPI = 100
    
    # VALIDATION SETTINGS (DITAMBAHKAN - CRITICAL)
    VALIDATION = {
        'MIN_DATA_POINTS': 30,
        'MIN_TRAIN_RATIO': 0.6,
        'MAX_TRAIN_RATIO': 0.9,
        'MAX_MISSING_VALUES_RATIO': 0.1
    }