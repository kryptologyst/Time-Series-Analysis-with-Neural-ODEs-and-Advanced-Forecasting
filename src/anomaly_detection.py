"""
Anomaly Detection Module for Time Series Analysis.

This module implements various anomaly detection methods including
Isolation Forest, Autoencoders, LSTM-based anomaly detection,
statistical methods, and ensemble approaches for comprehensive
anomaly detection in time series data.
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

# Anomaly detection libraries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN

# Statistical methods
from scipy import stats
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection methods."""
    # General settings
    contamination: float = 0.1
    random_state: int = 42
    
    # Isolation Forest settings
    isolation_n_estimators: int = 100
    isolation_max_samples: Union[str, int] = "auto"
    
    # Autoencoder settings
    encoding_dim: int = 16
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    
    # LSTM Autoencoder settings
    sequence_length: int = 60
    hidden_size: int = 50
    num_layers: int = 2
    dropout: float = 0.2
    
    # Statistical methods
    z_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    
    # DBSCAN settings
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5


class BaseAnomalyDetector(ABC):
    """Abstract base class for all anomaly detection methods."""
    
    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Fit the anomaly detection model."""
        pass
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies in the data."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass


class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest anomaly detector."""
    
    def __init__(self, config: AnomalyConfig):
        super().__init__(config)
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, data: np.ndarray) -> None:
        """Fit Isolation Forest model."""
        # Prepare data
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        data_scaled = self.scaler.fit_transform(data)
        
        # Initialize and fit model
        self.model = IsolationForest(
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_estimators=self.config.isolation_n_estimators,
            max_samples=self.config.isolation_max_samples
        )
        
        self.model.fit(data_scaled)
        self.is_fitted = True
        logger.info("Isolation Forest model fitted successfully")
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        data_scaled = self.scaler.transform(data)
        predictions = self.model.predict(data_scaled)
        
        # Convert -1/1 to 0/1 (0 = normal, 1 = anomaly)
        return (predictions == -1).astype(int)
    
    def get_model_name(self) -> str:
        """Get model name."""
        return "Isolation Forest"


class OneClassSVMDetector(BaseAnomalyDetector):
    """One-Class SVM anomaly detector."""
    
    def __init__(self, config: AnomalyConfig):
        super().__init__(config)
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, data: np.ndarray) -> None:
        """Fit One-Class SVM model."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        data_scaled = self.scaler.fit_transform(data)
        
        self.model = OneClassSVM(
            nu=self.config.contamination,
            kernel='rbf',
            gamma='scale'
        )
        
        self.model.fit(data_scaled)
        self.is_fitted = True
        logger.info("One-Class SVM model fitted successfully")
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        data_scaled = self.scaler.transform(data)
        predictions = self.model.predict(data_scaled)
        
        # Convert -1/1 to 0/1 (0 = normal, 1 = anomaly)
        return (predictions == -1).astype(int)
    
    def get_model_name(self) -> str:
        """Get model name."""
        return "One-Class SVM"


class AutoencoderDetector(BaseAnomalyDetector):
    """Autoencoder-based anomaly detector."""
    
    def __init__(self, config: AnomalyConfig):
        super().__init__(config)
        self.model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reconstruction_errors = None
    
    def fit(self, data: np.ndarray) -> None:
        """Fit Autoencoder model."""
        # Prepare data
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        data_scaled = self.scaler.fit_transform(data)
        
        # Create model
        input_dim = data_scaled.shape[1]
        self.model = AutoencoderModel(
            input_dim=input_dim,
            encoding_dim=self.config.encoding_dim
        ).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(torch.FloatTensor(data_scaled))
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
            total_loss = 0
            for batch in dataloader:
                batch_data = batch[0].to(self.device)
                
                optimizer.zero_grad()
                reconstructed = self.model(batch_data)
                loss = criterion(reconstructed, batch_data)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f"Autoencoder Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")
        
        # Calculate reconstruction errors for threshold
        self.model.eval()
        with torch.no_grad():
            all_reconstructions = []
            for batch in dataloader:
                batch_data = batch[0].to(self.device)
                reconstructed = self.model(batch_data)
                all_reconstructions.append(reconstructed.cpu().numpy())
            
            all_reconstructions = np.concatenate(all_reconstructions)
            reconstruction_errors = np.mean((data_scaled - all_reconstructions) ** 2, axis=1)
            self.reconstruction_errors = reconstruction_errors
        
        self.is_fitted = True
        logger.info("Autoencoder model fitted successfully")
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        data_scaled = self.scaler.transform(data)
        
        # Calculate reconstruction errors
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data_scaled).to(self.device)
            reconstructed = self.model(data_tensor)
            reconstruction_errors = np.mean((data_scaled - reconstructed.cpu().numpy()) ** 2, axis=1)
        
        # Use threshold based on training data
        threshold = np.percentile(self.reconstruction_errors, (1 - self.config.contamination) * 100)
        predictions = (reconstruction_errors > threshold).astype(int)
        
        return predictions
    
    def get_model_name(self) -> str:
        """Get model name."""
        return "Autoencoder"


class LSTMAutoencoderDetector(BaseAnomalyDetector):
    """LSTM Autoencoder anomaly detector."""
    
    def __init__(self, config: AnomalyConfig):
        super().__init__(config)
        self.model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reconstruction_errors = None
    
    def _prepare_sequences(self, data: np.ndarray) -> np.ndarray:
        """Prepare sequences for LSTM."""
        sequences = []
        for i in range(self.config.sequence_length, len(data)):
            sequences.append(data[i-self.config.sequence_length:i])
        return np.array(sequences)
    
    def fit(self, data: np.ndarray) -> None:
        """Fit LSTM Autoencoder model."""
        # Prepare data
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        sequences = self._prepare_sequences(data_scaled)
        
        # Create model
        self.model = LSTMAutoencoderModel(
            input_size=1,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(torch.FloatTensor(sequences).unsqueeze(-1))
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
            total_loss = 0
            for batch in dataloader:
                batch_data = batch[0].to(self.device)
                
                optimizer.zero_grad()
                reconstructed = self.model(batch_data)
                loss = criterion(reconstructed, batch_data)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f"LSTM Autoencoder Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")
        
        # Calculate reconstruction errors
        self.model.eval()
        with torch.no_grad():
            all_reconstructions = []
            for batch in dataloader:
                batch_data = batch[0].to(self.device)
                reconstructed = self.model(batch_data)
                all_reconstructions.append(reconstructed.cpu().numpy())
            
            all_reconstructions = np.concatenate(all_reconstructions)
            reconstruction_errors = np.mean((sequences - all_reconstructions.squeeze()) ** 2, axis=1)
            self.reconstruction_errors = reconstruction_errors
        
        self.is_fitted = True
        logger.info("LSTM Autoencoder model fitted successfully")
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        data_scaled = self.scaler.transform(data.reshape(-1, 1)).flatten()
        sequences = self._prepare_sequences(data_scaled)
        
        # Calculate reconstruction errors
        self.model.eval()
        with torch.no_grad():
            sequences_tensor = torch.FloatTensor(sequences).unsqueeze(-1).to(self.device)
            reconstructed = self.model(sequences_tensor)
            reconstruction_errors = np.mean((sequences - reconstructed.cpu().numpy().squeeze()) ** 2, axis=1)
        
        # Use threshold based on training data
        threshold = np.percentile(self.reconstruction_errors, (1 - self.config.contamination) * 100)
        predictions = (reconstruction_errors > threshold).astype(int)
        
        # Pad predictions to match original data length
        padded_predictions = np.zeros(len(data))
        padded_predictions[self.config.sequence_length:] = predictions
        
        return padded_predictions
    
    def get_model_name(self) -> str:
        """Get model name."""
        return "LSTM Autoencoder"


class StatisticalDetector(BaseAnomalyDetector):
    """Statistical anomaly detector using Z-score and IQR methods."""
    
    def __init__(self, config: AnomalyConfig):
        super().__init__(config)
        self.z_threshold = None
        self.iqr_bounds = None
    
    def fit(self, data: np.ndarray) -> None:
        """Fit statistical model."""
        # Calculate Z-score threshold
        mean = np.mean(data)
        std = np.std(data)
        self.z_threshold = self.config.z_threshold
        
        # Calculate IQR bounds
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - self.config.iqr_multiplier * iqr
        upper_bound = q3 + self.config.iqr_multiplier * iqr
        self.iqr_bounds = (lower_bound, upper_bound)
        
        self.is_fitted = True
        logger.info("Statistical model fitted successfully")
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies using statistical methods."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Z-score method
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / std)
        z_anomalies = (z_scores > self.z_threshold).astype(int)
        
        # IQR method
        iqr_anomalies = ((data < self.iqr_bounds[0]) | (data > self.iqr_bounds[1])).astype(int)
        
        # Combine both methods
        predictions = np.logical_or(z_anomalies, iqr_anomalies).astype(int)
        
        return predictions
    
    def get_model_name(self) -> str:
        """Get model name."""
        return "Statistical"


class DBSCANDetector(BaseAnomalyDetector):
    """DBSCAN-based anomaly detector."""
    
    def __init__(self, config: AnomalyConfig):
        super().__init__(config)
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, data: np.ndarray) -> None:
        """Fit DBSCAN model."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        data_scaled = self.scaler.fit_transform(data)
        
        self.model = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples
        )
        
        self.model.fit(data_scaled)
        self.is_fitted = True
        logger.info("DBSCAN model fitted successfully")
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        data_scaled = self.scaler.transform(data)
        predictions = self.model.fit_predict(data_scaled)
        
        # Convert -1 (noise) to 1 (anomaly), others to 0 (normal)
        return (predictions == -1).astype(int)
    
    def get_model_name(self) -> str:
        """Get model name."""
        return "DBSCAN"


# Neural Network Models
class AutoencoderModel(nn.Module):
    """Autoencoder model for anomaly detection."""
    
    def __init__(self, input_dim: int, encoding_dim: int):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class LSTMAutoencoderModel(nn.Module):
    """LSTM Autoencoder model for anomaly detection."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            hidden_size, input_size, num_layers,
            dropout=dropout, batch_first=True
        )
    
    def forward(self, x):
        # Encode
        encoded, _ = self.encoder(x)
        
        # Decode
        decoded, _ = self.decoder(encoded)
        
        return decoded


class AnomalyDetectionPipeline:
    """Pipeline for comparing multiple anomaly detection methods."""
    
    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.detectors = {}
        self.results = {}
    
    def add_detector(self, name: str, detector: BaseAnomalyDetector) -> None:
        """Add a detector to the pipeline."""
        self.detectors[name] = detector
    
    def fit_all(self, data: np.ndarray) -> None:
        """Fit all detectors."""
        logger.info("Fitting all anomaly detection models...")
        
        for name, detector in self.detectors.items():
            try:
                logger.info(f"Fitting {name}...")
                detector.fit(data)
            except Exception as e:
                logger.error(f"Error fitting {name}: {e}")
    
    def predict_all(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Detect anomalies with all models."""
        predictions = {}
        
        for name, detector in self.detectors.items():
            try:
                predictions[name] = detector.predict(data)
            except Exception as e:
                logger.error(f"Error detecting anomalies with {name}: {e}")
        
        return predictions
    
    def plot_results(
        self,
        data: np.ndarray,
        predictions: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """Plot anomaly detection results."""
        n_models = len(predictions)
        fig, axes = plt.subplots(n_models + 1, 1, figsize=(12, 4 * (n_models + 1)))
        
        # Plot original data
        axes[0].plot(data, label='Original Data', alpha=0.7)
        axes[0].set_title('Original Time Series')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot each detector's results
        for i, (name, pred) in enumerate(predictions.items()):
            ax = axes[i + 1]
            
            # Plot data
            ax.plot(data, label='Data', alpha=0.7, color='blue')
            
            # Highlight anomalies
            anomaly_indices = np.where(pred == 1)[0]
            if len(anomaly_indices) > 0:
                ax.scatter(
                    anomaly_indices,
                    data[anomaly_indices],
                    color='red',
                    s=50,
                    label='Anomalies',
                    zorder=5
                )
            
            ax.set_title(f'{name} Anomaly Detection')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def evaluate_ensemble(
        self,
        predictions: Dict[str, np.ndarray],
        threshold: float = 0.5
    ) -> np.ndarray:
        """Create ensemble prediction based on voting."""
        if not predictions:
            raise ValueError("No predictions available for ensemble")
        
        # Convert predictions to array
        pred_array = np.array(list(predictions.values()))
        
        # Majority voting
        ensemble_pred = (np.mean(pred_array, axis=0) > threshold).astype(int)
        
        return ensemble_pred


def main():
    """Demonstrate anomaly detection pipeline."""
    # Configuration
    config = AnomalyConfig(contamination=0.1)
    
    # Generate synthetic data with anomalies
    from data_generator import TimeSeriesGenerator, DataConfig
    
    data_config = DataConfig(
        n_points=1000,
        noise_level=0.1,
        anomaly_probability=0.05,
        anomaly_magnitude=3.0
    )
    generator = TimeSeriesGenerator(data_config)
    t, data = generator.generate_complex_time_series(generator.generate_time_points())
    
    # Add some known anomalies for evaluation
    anomaly_indices = np.random.choice(len(data), size=20, replace=False)
    data[anomaly_indices] += np.random.normal(0, 2, 20)
    
    # Create ground truth
    ground_truth = np.zeros(len(data))
    ground_truth[anomaly_indices] = 1
    
    # Initialize pipeline
    pipeline = AnomalyDetectionPipeline(config)
    
    # Add detectors
    pipeline.add_detector("Isolation Forest", IsolationForestDetector(config))
    pipeline.add_detector("One-Class SVM", OneClassSVMDetector(config))
    pipeline.add_detector("Autoencoder", AutoencoderDetector(config))
    pipeline.add_detector("LSTM Autoencoder", LSTMAutoencoderDetector(config))
    pipeline.add_detector("Statistical", StatisticalDetector(config))
    pipeline.add_detector("DBSCAN", DBSCANDetector(config))
    
    # Fit all models
    pipeline.fit_all(data)
    
    # Detect anomalies
    predictions = pipeline.predict_all(data)
    
    # Create ensemble prediction
    ensemble_pred = pipeline.evaluate_ensemble(predictions)
    predictions["Ensemble"] = ensemble_pred
    
    # Evaluate performance
    logger.info("Anomaly Detection Results:")
    for name, pred in predictions.items():
        if len(pred) == len(ground_truth):
            accuracy = np.mean(pred == ground_truth)
            precision = np.sum((pred == 1) & (ground_truth == 1)) / np.sum(pred == 1) if np.sum(pred == 1) > 0 else 0
            recall = np.sum((pred == 1) & (ground_truth == 1)) / np.sum(ground_truth == 1) if np.sum(ground_truth == 1) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            logger.info(f"{name}:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
    
    # Plot results
    pipeline.plot_results(data, predictions)
    
    return predictions


if __name__ == "__main__":
    main()
