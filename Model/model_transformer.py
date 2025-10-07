import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config import Config
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size, forecast_horizon=1, stride=1):
        self.data = data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        
    def __len__(self):
        return max(0, (len(self.data) - self.window_size - self.forecast_horizon) // self.stride + 1)
    
    def __getitem__(self, idx):
        actual_idx = idx * self.stride
        x = self.data[actual_idx:actual_idx + self.window_size]
        y = self.data[actual_idx + self.window_size:actual_idx + self.window_size + self.forecast_horizon]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    """Enhanced Transformer model for time series forecasting"""
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=256, dropout=0.1, output_dim=1, activation='gelu'):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, output_dim)
        )
        
        self.d_model = d_model
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, src, src_mask=None):
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        output = self.output_projection(output[:, -1, :])
        return output

class TransformerModel:
    """Enhanced wrapper for training and inference - FULLY FIXED"""
    def __init__(self, train_data: pd.Series, test_data: pd.Series = None,
                 window_size: int = None,
                 hidden_dim: int = None,
                 num_layers: int = None,
                 num_heads: int = None,
                 dropout: float = None,
                 learning_rate: float = None,
                 batch_size: int = None,
                 epochs: int = None,
                 use_cuda: bool = True,
                 scaler_type: str = 'standard'):
        
        self.train = train_data
        self.test = test_data
        
        # Use Config defaults if not provided
        self.window_size = window_size if window_size is not None else Config.TRANSFORMER_WINDOW_SIZE
        self.hidden_dim = hidden_dim if hidden_dim is not None else Config.TRANSFORMER_HIDDEN_DIM
        self.num_layers = num_layers if num_layers is not None else Config.TRANSFORMER_NUM_LAYERS
        self.num_heads = num_heads if num_heads is not None else Config.TRANSFORMER_NUM_HEADS
        self.dropout = dropout if dropout is not None else Config.TRANSFORMER_DROPOUT
        self.learning_rate = learning_rate if learning_rate is not None else Config.TRANSFORMER_LEARNING_RATE
        self.batch_size = batch_size if batch_size is not None else Config.TRANSFORMER_BATCH_SIZE
        self.epochs = epochs if epochs is not None else Config.TRANSFORMER_EPOCHS
        
        # CRITICAL: Validate data size
        min_required = self.window_size * 3
        if len(train_data) < min_required:
            raise ValueError(
                f"Training data too small for Transformer model.\n"
                f"Required: at least {min_required} data points (window_size * 3)\n"
                f"Got: {len(train_data)} data points\n"
                f"Suggestions:\n"
                f"  1. Use ARIMA or SARIMA instead (work with smaller datasets)\n"
                f"  2. Reduce window_size parameter (current: {self.window_size})\n"
                f"  3. Collect more data"
            )
        
        # Scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        self.model = None
        
        # CUDA with fallback
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        if self.use_cuda:
            try:
                print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
                print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            except:
                print("CUDA available but error getting device info")
                self.use_cuda = False
                self.device = torch.device('cpu')
        
        if not self.use_cuda:
            print("Using CPU")
        
        self.train_losses = []
        self.val_losses = []
        self.best_model_state = None
        
    def prepare_data(self, validation_split=0.2):
        """Prepare and scale data with validation split - FULLY FIXED"""
        try:
            # Scale data
            train_scaled = self.scaler.fit_transform(self.train.values.reshape(-1, 1))
            
            # Validate scaled data
            if np.isnan(train_scaled).any() or np.isinf(train_scaled).any():
                raise ValueError("Data contains NaN or Inf after scaling")
            
            # Calculate validation size
            val_size = max(self.window_size * 2, int(len(train_scaled) * validation_split))
            train_size = len(train_scaled) - val_size
            
            # Ensure enough training data
            if train_size < self.window_size * 2:
                raise ValueError(
                    f"Not enough training data after validation split.\n"
                    f"Need: at least {self.window_size * 2} points\n"
                    f"Got: {train_size} points after split\n"
                    f"Try: Reduce validation_split or window_size"
                )
            
            train_data = train_scaled[:train_size]
            val_data = train_scaled[train_size:]
            
            # Create datasets
            train_dataset = TimeSeriesDataset(train_data, self.window_size, forecast_horizon=1)
            val_dataset = TimeSeriesDataset(val_data, self.window_size, forecast_horizon=1)
            
            if len(train_dataset) == 0 or len(val_dataset) == 0:
                raise ValueError(f"Dataset too small: train={len(train_dataset)}, val={len(val_dataset)}")
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=min(self.batch_size, len(train_dataset)), 
                shuffle=True,
                num_workers=0,
                pin_memory=self.use_cuda
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=min(self.batch_size, len(val_dataset)),
                shuffle=False,
                num_workers=0,
                pin_memory=self.use_cuda
            )
            
            return train_loader, val_loader, train_scaled
            
        except Exception as e:
            print(f"Error in prepare_data: {e}")
            raise
    
    def build_model(self):
        """Build enhanced Transformer model"""
        try:
            self.model = TimeSeriesTransformer(
                input_dim=1,
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                num_layers=self.num_layers,
                dim_feedforward=self.hidden_dim * 4,
                dropout=self.dropout,
                output_dim=1,
                activation='gelu'
            ).to(self.device)
            
            if self.use_cuda:
                torch.backends.cudnn.benchmark = True
            
            return self.model
        except Exception as e:
            print(f"Error building model: {e}")
            raise
    
    def fit(self, patience: int = None, validation_split=0.2):
        """Train model - FULLY FIXED"""
        if patience is None:
            patience = Config.TRANSFORMER_PATIENCE
            
        try:
            train_loader, val_loader, _ = self.prepare_data(validation_split)
            
            if self.model is None:
                self.build_model()
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=1e-5
            )
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate * 10,
                epochs=self.epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3
            )
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            print(f"\nTraining Transformer on {self.device}...")
            print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Epochs: {self.epochs}, Batch Size: {self.batch_size}")
            print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
            
            for epoch in range(self.epochs):
                # Training
                self.model.train()
                epoch_train_loss = 0
                train_batches = 0
                
                for batch_x, batch_y in train_loader:
                    try:
                        batch_x = batch_x.to(self.device, non_blocking=True)
                        batch_y = batch_y.to(self.device, non_blocking=True)
                        
                        optimizer.zero_grad()
                        output = self.model(batch_x)
                        loss = criterion(output, batch_y)
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"Warning: Invalid loss at epoch {epoch+1}")
                            continue
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        
                        epoch_train_loss += loss.item()
                        train_batches += 1
                    except Exception as e:
                        print(f"Error in training batch: {e}")
                        continue
                
                if train_batches == 0:
                    raise ValueError("No successful training batches")
                
                avg_train_loss = epoch_train_loss / train_batches
                self.train_losses.append(avg_train_loss)
                
                # Validation
                self.model.eval()
                epoch_val_loss = 0
                val_batches = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        try:
                            batch_x = batch_x.to(self.device, non_blocking=True)
                            batch_y = batch_y.to(self.device, non_blocking=True)
                            
                            output = self.model(batch_x)
                            loss = criterion(output, batch_y)
                            
                            if not (torch.isnan(loss) or torch.isinf(loss)):
                                epoch_val_loss += loss.item()
                                val_batches += 1
                        except Exception as e:
                            print(f"Error in validation batch: {e}")
                            continue
                
                if val_batches == 0:
                    avg_val_loss = avg_train_loss
                else:
                    avg_val_loss = epoch_val_loss / val_batches
                
                self.val_losses.append(avg_val_loss)
                
                # Logging
                if (epoch + 1) % max(1, self.epochs // 10) == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] - "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {avg_val_loss:.6f}, "
                          f"LR: {scheduler.get_last_lr()[0]:.2e}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        if self.best_model_state:
                            self.model.load_state_dict(self.best_model_state)
                        break
            
            print("Training completed!")
            print(f"Best validation loss: {best_val_loss:.6f}")
            
            return self.model
            
        except Exception as e:
            print(f"Training error: {e}")
            raise
    
    def predict(self, steps: int = None) -> pd.Series:
        """Multi-step forecast - FULLY FIXED"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        if steps is None:
            steps = len(self.test) if self.test is not None else 10
        
        try:
            self.model.eval()
            
            train_scaled = self.scaler.transform(self.train.values.reshape(-1, 1))
            current_window = train_scaled[-self.window_size:].copy()
            
            predictions = []
            
            with torch.no_grad():
                for step in range(steps):
                    x = torch.FloatTensor(current_window).unsqueeze(0).to(self.device)
                    pred = self.model(x)
                    pred_value = pred.cpu().numpy()[0, 0]
                    
                    if np.isnan(pred_value) or np.isinf(pred_value):
                        print(f"Warning: Invalid prediction at step {step+1}")
                        pred_value = predictions[-1] if predictions else 0.0
                    
                    predictions.append(pred_value)
                    current_window = np.append(current_window[1:], [[pred_value]], axis=0)
            
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            if self.test is not None:
                pred_index = self.test.index[:steps]
            else:
                last_date = self.train.index[-1]
                pred_index = pd.date_range(start=last_date, periods=steps+1, freq=self.train.index.freq)[1:]
            
            return pd.Series(predictions.flatten(), index=pred_index)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            raise
    
    def plot_training_history(self, figsize=(12, 5)):
        """Plot training history"""
        if not self.train_losses:
            return None
            
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        epochs = range(1, len(self.train_losses) + 1)
        
        axes[0].plot(epochs, self.train_losses, label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.val_losses, label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].semilogy(epochs, self.train_losses, label='Train Loss', linewidth=2)
        axes[1].semilogy(epochs, self.val_losses, label='Val Loss', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (log scale)')
        axes[1].set_title('Training History (Log Scale)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_train_vs_fitted(self, figsize=(14, 6)):
        """Plot training data vs fitted values"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
            
        try:
            self.model.eval()
            train_scaled = self.scaler.transform(self.train.values.reshape(-1, 1))
            
            fitted_values = []
            with torch.no_grad():
                for i in range(self.window_size, len(train_scaled)):
                    window = train_scaled[i-self.window_size:i]
                    x = torch.FloatTensor(window).unsqueeze(0).to(self.device)
                    pred = self.model(x)
                    fitted_values.append(pred.cpu().numpy()[0, 0])
            
            fitted_values = self.scaler.inverse_transform(np.array(fitted_values).reshape(-1, 1))
            
            fig, ax = plt.subplots(figsize=figsize)
            
            train_subset = self.train.iloc[self.window_size:]
            ax.plot(train_subset.index, train_subset.values, 
                   label='Actual Train', linewidth=2, color='blue')
            ax.plot(train_subset.index, fitted_values.flatten(), 
                   label='Fitted', linewidth=2, color='red', alpha=0.7)
            
            ax.set_title('Training Data vs Fitted Values')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error in plot_train_vs_fitted: {e}")
            return None
    
    def plot_test_vs_forecast(self, figsize=(14, 6)):
        """Plot test data vs forecast"""
        if self.test is None:
            raise ValueError("Test data must exist")
        
        try:
            forecast = self.predict()
            
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.plot(self.test.index, self.test.values, 
                   label='Actual Test', linewidth=2, color='blue', marker='o')
            ax.plot(forecast.index, forecast.values, 
                   label='Forecast', linewidth=2, color='red', marker='s', alpha=0.7)
            
            ax.set_title('Test Data vs Forecast')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error in plot_test_vs_forecast: {e}")
            return None
    
    def get_training_history(self):
        """Return training history"""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs': len(self.train_losses),
            'best_val_loss': min(self.val_losses) if self.val_losses else None
        }
    
    def save_model(self, filepath: str):
        """Save model weights"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        try:
            torch.save({
                'model_state_dict': self.best_model_state or self.model.state_dict(),
                'scaler': self.scaler,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'config': {
                    'window_size': self.window_size,
                    'hidden_dim': self.hidden_dim,
                    'num_layers': self.num_layers,
                    'num_heads': self.num_heads,
                    'dropout': self.dropout
                }
            }, filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str):
        """Load model weights"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            config = checkpoint['config']
            self.window_size = config['window_size']
            self.hidden_dim = config['hidden_dim']
            self.num_layers = config['num_layers']
            self.num_heads = config['num_heads']
            self.dropout = config['dropout']
            
            self.build_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler']
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise