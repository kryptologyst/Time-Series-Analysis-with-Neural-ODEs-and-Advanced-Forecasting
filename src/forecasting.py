"""
Advanced Forecasting Methods for Time Series Analysis.

This module implements state-of-the-art forecasting methods including
ARIMA, Prophet, LSTM, GRU, and Transformer models for comprehensive
time series prediction and comparison.
"""

import logging
from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

# Time series libraries
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. ARIMA methods will be disabled.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("prophet not available. Prophet methods will be disabled.")

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    warnings.warn("pmdarima not available. Auto-ARIMA will be disabled.")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class ForecastConfig:
    """Configuration for forecasting models."""
    # General settings
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # ARIMA settings
    arima_order: Tuple[int, int, int] = (2, 1, 2)
    arima_seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)
    
    # Prophet settings
    prophet_yearly_seasonality: bool = True
    prophet_weekly_seasonality: bool = True
    prophet_daily_seasonality: bool = False
    
    # Deep learning settings
    sequence_length: int = 60
    hidden_size: int = 50
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Fit the forecasting model."""
        pass
    
    @abstractmethod
    def predict(self, n_periods: int) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass


class ARIMAForecaster(BaseForecaster):
    """ARIMA forecasting model."""
    
    def __init__(self, config: ForecastConfig):
        super().__init__(config)
        self.model = None
        self.data = None
    
    def fit(self, data: np.ndarray) -> None:
        """Fit ARIMA model."""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA forecasting")
        
        self.data = data
        try:
            self.model = ARIMA(
                data,
                order=self.config.arima_order,
                seasonal_order=self.config.arima_seasonal_order
            ).fit()
            self.is_fitted = True
            logger.info("ARIMA model fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            raise
    
    def predict(self, n_periods: int) -> np.ndarray:
        """Make ARIMA predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        forecast = self.model.forecast(steps=n_periods)
        return forecast.values if hasattr(forecast, 'values') else forecast
    
    def get_model_name(self) -> str:
        """Get model name."""
        return "ARIMA"


class AutoARIMAForecaster(BaseForecaster):
    """Auto-ARIMA forecasting model."""
    
    def __init__(self, config: ForecastConfig):
        super().__init__(config)
        self.model = None
        self.data = None
    
    def fit(self, data: np.ndarray) -> None:
        """Fit Auto-ARIMA model."""
        if not PMDARIMA_AVAILABLE:
            raise ImportError("pmdarima is required for Auto-ARIMA forecasting")
        
        self.data = data
        try:
            self.model = auto_arima(
                data,
                seasonal=True,
                m=12,
                suppress_warnings=True,
                error_action='ignore'
            )
            self.is_fitted = True
            logger.info("Auto-ARIMA model fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting Auto-ARIMA model: {e}")
            raise
    
    def predict(self, n_periods: int) -> np.ndarray:
        """Make Auto-ARIMA predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        forecast = self.model.predict(n_periods=n_periods)
        return forecast.values if hasattr(forecast, 'values') else forecast
    
    def get_model_name(self) -> str:
        """Get model name."""
        return "Auto-ARIMA"


class ProphetForecaster(BaseForecaster):
    """Prophet forecasting model."""
    
    def __init__(self, config: ForecastConfig):
        super().__init__(config)
        self.model = None
        self.data = None
    
    def fit(self, data: np.ndarray) -> None:
        """Fit Prophet model."""
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet is required for Prophet forecasting")
        
        self.data = data
        
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=len(data), freq='D'),
            'y': data
        })
        
        try:
            self.model = Prophet(
                yearly_seasonality=self.config.prophet_yearly_seasonality,
                weekly_seasonality=self.config.prophet_weekly_seasonality,
                daily_seasonality=self.config.prophet_daily_seasonality
            )
            self.model.fit(df)
            self.is_fitted = True
            logger.info("Prophet model fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {e}")
            raise
    
    def predict(self, n_periods: int) -> np.ndarray:
        """Make Prophet predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=n_periods)
        forecast = self.model.predict(future)
        
        # Return only the forecasted values
        return forecast['yhat'].values[-n_periods:]
    
    def get_model_name(self) -> str:
        """Get model name."""
        return "Prophet"


class LSTMForecaster(BaseForecaster):
    """LSTM forecasting model."""
    
    def __init__(self, config: ForecastConfig):
        super().__init__(config)
        self.model = None
        self.scaler = MinMaxScaler()
        self.data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _prepare_data(self, data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for LSTM training."""
        # Normalize data
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = [], []
        for i in range(self.config.sequence_length, len(data_scaled)):
            X.append(data_scaled[i-self.config.sequence_length:i])
            y.append(data_scaled[i])
        
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def fit(self, data: np.ndarray) -> None:
        """Fit LSTM model."""
        self.data = data
        
        # Prepare data
        X, y = self._prepare_data(data)
        
        # Create model
        self.model = LSTMModel(
            input_size=1,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X.unsqueeze(-1), y.unsqueeze(-1))
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.config.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"LSTM Epoch {epoch}, Loss: {loss.item():.6f}")
        
        self.is_fitted = True
        logger.info("LSTM model fitted successfully")
    
    def predict(self, n_periods: int) -> np.ndarray:
        """Make LSTM predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        predictions = []
        
        # Use last sequence_length points for prediction
        last_sequence = self.data[-self.config.sequence_length:]
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        
        current_sequence = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            for _ in range(n_periods):
                pred = self.model(current_sequence)
                predictions.append(pred.cpu().numpy()[0, 0])
                
                # Update sequence
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    pred.unsqueeze(0)
                ], dim=1)
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions
    
    def get_model_name(self) -> str:
        """Get model name."""
        return "LSTM"


class GRUForecaster(BaseForecaster):
    """GRU forecasting model."""
    
    def __init__(self, config: ForecastConfig):
        super().__init__(config)
        self.model = None
        self.scaler = MinMaxScaler()
        self.data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _prepare_data(self, data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for GRU training."""
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        X, y = [], []
        for i in range(self.config.sequence_length, len(data_scaled)):
            X.append(data_scaled[i-self.config.sequence_length:i])
            y.append(data_scaled[i])
        
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def fit(self, data: np.ndarray) -> None:
        """Fit GRU model."""
        self.data = data
        
        X, y = self._prepare_data(data)
        
        self.model = GRUModel(
            input_size=1,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        dataset = TensorDataset(X.unsqueeze(-1), y.unsqueeze(-1))
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.config.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"GRU Epoch {epoch}, Loss: {loss.item():.6f}")
        
        self.is_fitted = True
        logger.info("GRU model fitted successfully")
    
    def predict(self, n_periods: int) -> np.ndarray:
        """Make GRU predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        predictions = []
        
        last_sequence = self.data[-self.config.sequence_length:]
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        
        current_sequence = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            for _ in range(n_periods):
                pred = self.model(current_sequence)
                predictions.append(pred.cpu().numpy()[0, 0])
                
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    pred.unsqueeze(0)
                ], dim=1)
        
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions
    
    def get_model_name(self) -> str:
        """Get model name."""
        return "GRU"


class TransformerForecaster(BaseForecaster):
    """Transformer forecasting model."""
    
    def __init__(self, config: ForecastConfig):
        super().__init__(config)
        self.model = None
        self.scaler = MinMaxScaler()
        self.data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _prepare_data(self, data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for Transformer training."""
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        X, y = [], []
        for i in range(self.config.sequence_length, len(data_scaled)):
            X.append(data_scaled[i-self.config.sequence_length:i])
            y.append(data_scaled[i])
        
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def fit(self, data: np.ndarray) -> None:
        """Fit Transformer model."""
        self.data = data
        
        X, y = self._prepare_data(data)
        
        self.model = TransformerModel(
            input_size=1,
            d_model=64,
            nhead=4,
            num_layers=2,
            dropout=self.config.dropout
        ).to(self.device)
        
        dataset = TensorDataset(X.unsqueeze(-1), y.unsqueeze(-1))
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.config.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Transformer Epoch {epoch}, Loss: {loss.item():.6f}")
        
        self.is_fitted = True
        logger.info("Transformer model fitted successfully")
    
    def predict(self, n_periods: int) -> np.ndarray:
        """Make Transformer predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        predictions = []
        
        last_sequence = self.data[-self.config.sequence_length:]
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        
        current_sequence = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            for _ in range(n_periods):
                pred = self.model(current_sequence)
                predictions.append(pred.cpu().numpy()[0, 0])
                
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    pred.unsqueeze(0)
                ], dim=1)
        
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions
    
    def get_model_name(self) -> str:
        """Get model name."""
        return "Transformer"


# Neural Network Models
class LSTMModel(nn.Module):
    """LSTM model for time series forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output


class GRUModel(nn.Module):
    """GRU model for time series forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out[:, -1, :])
        return output


class TransformerModel(nn.Module):
    """Transformer model for time series forecasting."""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout),
            num_layers
        )
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # Transformer expects (seq_len, batch, d_model)
        transformer_out = self.transformer(x)
        output = self.fc(transformer_out[-1])  # Use last output
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(1000, d_model)  # Max sequence length
        position = torch.arange(0, 1000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ForecastingPipeline:
    """Pipeline for comparing multiple forecasting methods."""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.forecasters = {}
        self.results = {}
    
    def add_forecaster(self, name: str, forecaster: BaseForecaster) -> None:
        """Add a forecaster to the pipeline."""
        self.forecasters[name] = forecaster
    
    def fit_all(self, data: np.ndarray) -> None:
        """Fit all forecasters."""
        logger.info("Fitting all forecasting models...")
        
        for name, forecaster in self.forecasters.items():
            try:
                logger.info(f"Fitting {name}...")
                forecaster.fit(data)
            except Exception as e:
                logger.error(f"Error fitting {name}: {e}")
    
    def predict_all(self, n_periods: int) -> Dict[str, np.ndarray]:
        """Make predictions with all models."""
        predictions = {}
        
        for name, forecaster in self.forecasters.items():
            try:
                predictions[name] = forecaster.predict(n_periods)
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")
        
        return predictions
    
    def evaluate_models(self, test_data: np.ndarray, predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Evaluate all models."""
        results = {}
        
        for name, pred in predictions.items():
            if len(pred) == len(test_data):
                mse = mean_squared_error(test_data, pred)
                mae = mean_absolute_error(test_data, pred)
                r2 = r2_score(test_data, pred)
                
                results[name] = {
                    'MSE': mse,
                    'MAE': mae,
                    'R2': r2,
                    'RMSE': np.sqrt(mse)
                }
        
        return results
    
    def plot_comparison(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        predictions: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """Plot comparison of all models."""
        n_models = len(predictions)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (name, pred) in enumerate(predictions.items()):
            ax = axes[i]
            
            # Plot training data
            ax.plot(range(len(train_data)), train_data, label='Training Data', alpha=0.7)
            
            # Plot test data
            test_start = len(train_data)
            test_end = test_start + len(test_data)
            ax.plot(range(test_start, test_end), test_data, label='Actual', color='green')
            
            # Plot predictions
            ax.plot(range(test_start, test_end), pred, label=f'{name} Prediction', color='red')
            
            ax.set_title(f'{name} Forecasting Results')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """Demonstrate forecasting pipeline."""
    # Configuration
    config = ForecastConfig()
    
    # Generate synthetic data
    from data_generator import TimeSeriesGenerator, DataConfig
    
    data_config = DataConfig(n_points=1000, noise_level=0.1)
    generator = TimeSeriesGenerator(data_config)
    t, data = generator.generate_complex_time_series(generator.generate_time_points())
    
    # Split data
    train_size = int(len(data) * config.train_ratio)
    test_size = int(len(data) * config.test_ratio)
    
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    
    # Initialize pipeline
    pipeline = ForecastingPipeline(config)
    
    # Add forecasters
    if STATSMODELS_AVAILABLE:
        pipeline.add_forecaster("ARIMA", ARIMAForecaster(config))
    
    if PMDARIMA_AVAILABLE:
        pipeline.add_forecaster("Auto-ARIMA", AutoARIMAForecaster(config))
    
    if PROPHET_AVAILABLE:
        pipeline.add_forecaster("Prophet", ProphetForecaster(config))
    
    pipeline.add_forecaster("LSTM", LSTMForecaster(config))
    pipeline.add_forecaster("GRU", GRUForecaster(config))
    pipeline.add_forecaster("Transformer", TransformerForecaster(config))
    
    # Fit all models
    pipeline.fit_all(train_data)
    
    # Make predictions
    n_periods = len(test_data)
    predictions = pipeline.predict_all(n_periods)
    
    # Evaluate models
    results = pipeline.evaluate_models(test_data, predictions)
    
    # Print results
    logger.info("Forecasting Results:")
    for name, metrics in results.items():
        logger.info(f"{name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # Plot comparison
    pipeline.plot_comparison(train_data, test_data, predictions)
    
    return results


if __name__ == "__main__":
    main()
