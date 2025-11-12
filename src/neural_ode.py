"""
Neural Ordinary Differential Equations for Time Series Analysis.

This module implements Neural ODEs for time series modeling, providing
continuous-time dynamics instead of discrete layers. Neural ODEs define
the derivative of the hidden state with a neural network and solve it
using an ODE solver.

This approach is particularly effective for:
- Irregular time series
- Physics-informed modeling
- Data-efficient learning
- Smooth trajectory prediction
"""

import logging
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NeuralODEConfig:
    """Configuration class for Neural ODE model."""
    hidden_size: int = 64
    num_layers: int = 3
    learning_rate: float = 0.01
    epochs: int = 500
    batch_size: int = 32
    dropout: float = 0.1
    activation: str = "tanh"


class ODEFunc(nn.Module):
    """
    Neural network that defines the ODE function dy/dt = f(t, y).
    
    This class implements the right-hand side of the ODE system,
    where the derivative of the state is computed by a neural network.
    
    Args:
        hidden_size: Number of hidden units in each layer
        num_layers: Number of hidden layers
        dropout: Dropout probability for regularization
        activation: Activation function to use
    """
    
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "tanh"
    ) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Build the neural network layers
        layers = []
        input_size = 1
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            
            if activation.lower() == "tanh":
                layers.append(nn.Tanh())
            elif activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(hidden_size, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ODE function.
        
        Args:
            t: Time tensor (not used in this implementation)
            y: State tensor of shape (batch_size, 1)
            
        Returns:
            Derivative tensor of shape (batch_size, 1)
        """
        return self.net(y)


class NeuralODE(nn.Module):
    """
    Neural ODE model for time series prediction.
    
    This class wraps the ODE function and provides methods for
    training and prediction using differentiable ODE solvers.
    
    Args:
        ode_func: The ODE function defining the dynamics
        solver: ODE solver method to use
        rtol: Relative tolerance for ODE solver
        atol: Absolute tolerance for ODE solver
    """
    
    def __init__(
        self,
        ode_func: ODEFunc,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4
    ) -> None:
        super().__init__()
        
        self.ode_func = ode_func
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
    
    def forward(
        self,
        y0: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Solve the ODE system from initial condition y0 over time points t.
        
        Args:
            y0: Initial condition tensor of shape (batch_size, 1)
            t: Time points tensor of shape (n_points,)
            
        Returns:
            Solution tensor of shape (n_points, batch_size, 1)
        """
        return odeint(
            self.ode_func,
            y0,
            t,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol
        )
    
    def predict(
        self,
        y0: torch.Tensor,
        t: torch.Tensor,
        return_derivatives: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Make predictions using the trained model.
        
        Args:
            y0: Initial condition tensor
            t: Time points tensor
            return_derivatives: Whether to return derivatives
            
        Returns:
            Tuple of (predictions, derivatives)
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(y0, t)
            
            derivatives = None
            if return_derivatives:
                derivatives = self.ode_func(t, predictions.squeeze(-1))
        
        return predictions, derivatives


class NeuralODETrainer:
    """
    Trainer class for Neural ODE models.
    
    Provides methods for training, validation, and model persistence.
    """
    
    def __init__(
        self,
        model: NeuralODE,
        config: NeuralODEConfig,
        device: str = "cpu"
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        self.loss_fn = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def train(
        self,
        y0: torch.Tensor,
        t: torch.Tensor,
        y_target: torch.Tensor,
        val_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Train the Neural ODE model.
        
        Args:
            y0: Initial condition tensor
            t: Time points tensor
            y_target: Target values tensor
            val_data: Optional validation data tuple (y0_val, t_val, y_val)
            
        Returns:
            Dictionary containing training history
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        # Move data to device
        y0 = y0.to(self.device)
        t = t.to(self.device)
        y_target = y_target.to(self.device)
        
        for epoch in range(self.config.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(y0, t).squeeze(-1)
            loss = self.loss_fn(predictions, y_target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Store training loss
            self.train_losses.append(loss.item())
            
            # Validation
            if val_data is not None and epoch % 10 == 0:
                val_loss = self._validate(val_data)
                self.val_losses.append(val_loss)
            
            # Logging
            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        logger.info("Training completed")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "final_loss": self.train_losses[-1]
        }
    
    def _validate(
        self,
        val_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> float:
        """Perform validation step."""
        y0_val, t_val, y_val = val_data
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(y0_val.to(self.device), t_val.to(self.device))
            val_loss = self.loss_fn(predictions.squeeze(-1), y_val.to(self.device))
        
        return val_loss.item()
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        logger.info(f"Model loaded from {filepath}")


def generate_synthetic_data(
    n_points: int = 1000,
    noise_level: float = 0.1,
    trend_strength: float = 0.5,
    seasonality_period: int = 50,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic time series data with trend, seasonality, and noise.
    
    Args:
        n_points: Number of data points
        noise_level: Standard deviation of noise
        trend_strength: Strength of linear trend
        seasonality_period: Period of seasonal component
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (time_points, true_signal, noisy_signal)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate time points
    t = torch.linspace(0, 10, n_points)
    
    # Generate true signal with trend and seasonality
    trend = trend_strength * t
    seasonal = torch.sin(2 * np.pi * t / seasonality_period)
    y_true = trend + seasonal
    
    # Add noise
    noise = noise_level * torch.randn_like(t)
    y_noisy = y_true + noise
    
    return t, y_true, y_noisy


def plot_results(
    t: torch.Tensor,
    y_true: torch.Tensor,
    y_noisy: torch.Tensor,
    y_pred: torch.Tensor,
    title: str = "Neural ODE Time Series Prediction",
    save_path: Optional[str] = None
) -> None:
    """
    Plot the results of Neural ODE prediction.
    
    Args:
        t: Time points
        y_true: True signal
        y_noisy: Noisy observations
        y_pred: Model predictions
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    plt.plot(t.numpy(), y_noisy.numpy(), 
             label="Noisy Observations", alpha=0.6, linewidth=1)
    plt.plot(t.numpy(), y_pred.numpy(), 
             label="Neural ODE Prediction", linewidth=2, color='red')
    plt.plot(t.numpy(), y_true.numpy(), 
             label="True Signal", linestyle="--", linewidth=2, color='green')
    
    plt.title(title, fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    """Main function to demonstrate Neural ODE usage."""
    # Configuration
    config = NeuralODEConfig(
        hidden_size=64,
        num_layers=3,
        learning_rate=0.01,
        epochs=200,
        dropout=0.1
    )
    
    # Generate synthetic data
    logger.info("Generating synthetic data...")
    t, y_true, y_noisy = generate_synthetic_data(
        n_points=1000,
        noise_level=0.1,
        trend_strength=0.5,
        seasonality_period=50
    )
    
    # Initialize model
    logger.info("Initializing Neural ODE model...")
    ode_func = ODEFunc(
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    model = NeuralODE(ode_func)
    
    # Initialize trainer
    trainer = NeuralODETrainer(model, config)
    
    # Prepare training data
    y0 = y_noisy[0].unsqueeze(0).unsqueeze(-1)  # Initial condition
    
    # Train model
    logger.info("Training Neural ODE...")
    history = trainer.train(y0, t, y_noisy)
    
    # Make predictions
    logger.info("Making predictions...")
    predictions, _ = model.predict(y0, t)
    y_pred = predictions.squeeze()
    
    # Plot results
    plot_results(t, y_true, y_noisy, y_pred)
    
    # Save model
    trainer.save_model("models/neural_ode_model.pth")
    
    logger.info("Neural ODE demonstration completed successfully!")


if __name__ == "__main__":
    main()
