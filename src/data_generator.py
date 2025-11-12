"""
Data Generation Module for Time Series Analysis.

This module provides comprehensive data generation capabilities for creating
realistic synthetic time series with various characteristics including trends,
seasonality, noise, anomalies, and different patterns commonly found in
real-world time series data.
"""

import logging
from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from enum import Enum
import yaml

logger = logging.getLogger(__name__)


class TimeSeriesType(Enum):
    """Enumeration of different time series types."""
    SINE_WAVE = "sine_wave"
    COSINE_WAVE = "cosine_wave"
    LINEAR_TREND = "linear_trend"
    EXPONENTIAL_TREND = "exponential_trend"
    LOGARITHMIC_TREND = "logarithmic_trend"
    RANDOM_WALK = "random_walk"
    ARMA = "arma"
    SEASONAL = "seasonal"
    COMPLEX = "complex"


@dataclass
class DataConfig:
    """Configuration for data generation."""
    n_points: int = 1000
    noise_level: float = 0.1
    trend_strength: float = 0.5
    seasonality_period: int = 50
    anomaly_probability: float = 0.05
    anomaly_magnitude: float = 3.0
    seed: int = 42
    time_range: Tuple[float, float] = (0, 10)
    frequency: float = 1.0


class TimeSeriesGenerator:
    """
    Comprehensive time series generator for creating synthetic data.
    
    This class provides methods to generate various types of time series
    including simple patterns, complex combinations, and realistic scenarios
    with anomalies and noise.
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._set_random_seed()
    
    def _set_random_seed(self) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
    
    def generate_time_points(self) -> np.ndarray:
        """Generate time points based on configuration."""
        return np.linspace(
            self.config.time_range[0],
            self.config.time_range[1],
            self.config.n_points
        )
    
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to the signal."""
        noise = self.config.noise_level * np.random.randn(len(signal))
        return signal + noise
    
    def add_anomalies(
        self,
        signal: np.ndarray,
        anomaly_indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Add anomalies to the signal.
        
        Args:
            signal: Input signal
            anomaly_indices: Optional predefined anomaly positions
            
        Returns:
            Tuple of (signal_with_anomalies, anomaly_indices)
        """
        signal_with_anomalies = signal.copy()
        
        if anomaly_indices is None:
            # Randomly select anomaly positions
            n_anomalies = int(self.config.anomaly_probability * len(signal))
            anomaly_indices = np.random.choice(
                len(signal), size=n_anomalies, replace=False
            )
        
        # Add anomalies
        for idx in anomaly_indices:
            anomaly_value = self.config.anomaly_magnitude * np.random.randn()
            signal_with_anomalies[idx] += anomaly_value
        
        return signal_with_anomalies, anomaly_indices
    
    def generate_sine_wave(self, t: np.ndarray) -> np.ndarray:
        """Generate sine wave time series."""
        return np.sin(2 * np.pi * self.config.frequency * t)
    
    def generate_cosine_wave(self, t: np.ndarray) -> np.ndarray:
        """Generate cosine wave time series."""
        return np.cos(2 * np.pi * self.config.frequency * t)
    
    def generate_linear_trend(self, t: np.ndarray) -> np.ndarray:
        """Generate linear trend time series."""
        return self.config.trend_strength * t
    
    def generate_exponential_trend(self, t: np.ndarray) -> np.ndarray:
        """Generate exponential trend time series."""
        return self.config.trend_strength * np.exp(0.1 * t)
    
    def generate_logarithmic_trend(self, t: np.ndarray) -> np.ndarray:
        """Generate logarithmic trend time series."""
        return self.config.trend_strength * np.log(1 + t)
    
    def generate_random_walk(self, t: np.ndarray) -> np.ndarray:
        """Generate random walk time series."""
        steps = np.random.randn(len(t))
        return np.cumsum(steps) * self.config.trend_strength
    
    def generate_arma(
        self,
        t: np.ndarray,
        ar_params: List[float] = [0.7],
        ma_params: List[float] = [0.3]
    ) -> np.ndarray:
        """
        Generate ARMA time series.
        
        Args:
            t: Time points
            ar_params: Autoregressive parameters
            ma_params: Moving average parameters
            
        Returns:
            ARMA time series
        """
        n = len(t)
        ar_order = len(ar_params)
        ma_order = len(ma_params)
        
        # Initialize
        series = np.zeros(n)
        errors = np.random.randn(n)
        
        # Generate ARMA process
        for i in range(max(ar_order, ma_order), n):
            # AR component
            ar_component = sum(
                ar_params[j] * series[i - j - 1] 
                for j in range(ar_order)
            )
            
            # MA component
            ma_component = sum(
                ma_params[j] * errors[i - j - 1] 
                for j in range(ma_order)
            )
            
            series[i] = ar_component + ma_component + errors[i]
        
        return series
    
    def generate_seasonal(
        self,
        t: np.ndarray,
        periods: List[int] = None
    ) -> np.ndarray:
        """
        Generate seasonal time series with multiple periods.
        
        Args:
            t: Time points
            periods: List of seasonal periods
            
        Returns:
            Seasonal time series
        """
        if periods is None:
            periods = [self.config.seasonality_period]
        
        seasonal_component = np.zeros(len(t))
        
        for period in periods:
            seasonal_component += np.sin(2 * np.pi * t / period)
            seasonal_component += 0.5 * np.cos(4 * np.pi * t / period)
        
        return seasonal_component
    
    def generate_complex_time_series(
        self,
        t: np.ndarray,
        components: Dict[str, Any] = None
    ) -> np.ndarray:
        """
        Generate complex time series with multiple components.
        
        Args:
            t: Time points
            components: Dictionary specifying components to include
            
        Returns:
            Complex time series
        """
        if components is None:
            components = {
                "trend": True,
                "seasonal": True,
                "noise": True,
                "anomalies": True
            }
        
        signal = np.zeros(len(t))
        
        # Add trend component
        if components.get("trend", False):
            signal += self.generate_linear_trend(t)
        
        # Add seasonal component
        if components.get("seasonal", False):
            signal += self.generate_seasonal(t)
        
        # Add noise
        if components.get("noise", False):
            signal = self.add_noise(signal)
        
        # Add anomalies
        if components.get("anomalies", False):
            signal, _ = self.add_anomalies(signal)
        
        return signal
    
    def generate_multivariate_time_series(
        self,
        n_variables: int = 3,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate multivariate time series with correlations.
        
        Args:
            n_variables: Number of variables
            correlation_matrix: Optional correlation matrix
            
        Returns:
            Tuple of (time_points, multivariate_data)
        """
        t = self.generate_time_points()
        
        if correlation_matrix is None:
            # Generate random correlation matrix
            correlation_matrix = np.random.rand(n_variables, n_variables)
            correlation_matrix = correlation_matrix @ correlation_matrix.T
            np.fill_diagonal(correlation_matrix, 1.0)
        
        # Generate independent components
        independent_series = []
        for i in range(n_variables):
            series = self.generate_complex_time_series(t)
            independent_series.append(series)
        
        independent_series = np.array(independent_series).T
        
        # Apply correlation structure
        chol = np.linalg.cholesky(correlation_matrix)
        correlated_series = independent_series @ chol.T
        
        return t, correlated_series
    
    def generate_realistic_scenarios(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate realistic time series scenarios.
        
        Returns:
            Dictionary of scenario names and (time, data) tuples
        """
        t = self.generate_time_points()
        scenarios = {}
        
        # Stock price simulation
        scenarios["stock_price"] = self._generate_stock_price(t)
        
        # Energy consumption
        scenarios["energy_consumption"] = self._generate_energy_consumption(t)
        
        # Weather temperature
        scenarios["temperature"] = self._generate_temperature(t)
        
        # Traffic flow
        scenarios["traffic_flow"] = self._generate_traffic_flow(t)
        
        # Sales data
        scenarios["sales"] = self._generate_sales_data(t)
        
        return scenarios
    
    def _generate_stock_price(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic stock price time series."""
        # Geometric Brownian motion
        dt = t[1] - t[0]
        returns = np.random.normal(0.001, 0.02, len(t))  # Daily returns
        log_prices = np.cumsum(returns)
        prices = 100 * np.exp(log_prices)  # Starting price of 100
        
        return t, prices
    
    def _generate_energy_consumption(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic energy consumption time series."""
        # Base consumption with daily and weekly patterns
        daily_pattern = 50 + 30 * np.sin(2 * np.pi * t / 1)  # Daily cycle
        weekly_pattern = 20 * np.sin(2 * np.pi * t / 7)  # Weekly cycle
        trend = 0.1 * t  # Slight upward trend
        
        consumption = daily_pattern + weekly_pattern + trend
        consumption = self.add_noise(consumption)
        
        return t, consumption
    
    def _generate_temperature(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic temperature time series."""
        # Annual cycle with trend
        annual_cycle = 15 + 10 * np.sin(2 * np.pi * t / 365)  # Annual cycle
        trend = 0.01 * t  # Climate warming trend
        daily_variation = 2 * np.sin(2 * np.pi * t)  # Daily variation
        
        temperature = annual_cycle + trend + daily_variation
        temperature = self.add_noise(temperature)
        
        return t, temperature
    
    def _generate_traffic_flow(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic traffic flow time series."""
        # Rush hour patterns
        rush_hour = 100 + 50 * np.sin(2 * np.pi * t / 1)  # Daily rush hours
        weekly_pattern = 20 * np.sin(2 * np.pi * t / 7)  # Weekly pattern
        random_events = 10 * np.random.randn(len(t))  # Random events
        
        traffic = rush_hour + weekly_pattern + random_events
        traffic = np.maximum(traffic, 0)  # Ensure non-negative
        
        return t, traffic
    
    def _generate_sales_data(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic sales data time series."""
        # Seasonal sales with promotions
        base_sales = 1000
        seasonal = 200 * np.sin(2 * np.pi * t / 365)  # Annual seasonality
        trend = 5 * t  # Growth trend
        
        # Random promotions
        promotions = np.zeros(len(t))
        promo_indices = np.random.choice(len(t), size=20, replace=False)
        promotions[promo_indices] = np.random.uniform(200, 500, 20)
        
        sales = base_sales + seasonal + trend + promotions
        sales = self.add_noise(sales)
        sales = np.maximum(sales, 0)  # Ensure non-negative
        
        return t, sales
    
    def save_data(
        self,
        data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        filepath: str
    ) -> None:
        """Save generated data to files."""
        for name, (t, values) in data.items():
            df = pd.DataFrame({
                'time': t,
                'value': values
            })
            df.to_csv(f"{filepath}_{name}.csv", index=False)
        
        logger.info(f"Data saved to {filepath}")


def load_config(config_path: str) -> DataConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)['data']
    
    return DataConfig(**config_dict)


def main():
    """Demonstrate data generation capabilities."""
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Initialize generator
    generator = TimeSeriesGenerator(config)
    
    # Generate time points
    t = generator.generate_time_points()
    
    # Generate different types of time series
    logger.info("Generating various time series types...")
    
    # Simple patterns
    sine_wave = generator.generate_sine_wave(t)
    linear_trend = generator.generate_linear_trend(t)
    seasonal = generator.generate_seasonal(t)
    
    # Complex time series
    complex_series = generator.generate_complex_time_series(t)
    
    # Multivariate data
    t_multi, multi_data = generator.generate_multivariate_time_series(n_variables=3)
    
    # Realistic scenarios
    scenarios = generator.generate_realistic_scenarios()
    
    logger.info("Data generation completed successfully!")
    
    # Save data
    generator.save_data(scenarios, "data/generated_scenarios")
    
    return {
        "time_points": t,
        "sine_wave": sine_wave,
        "linear_trend": linear_trend,
        "seasonal": seasonal,
        "complex_series": complex_series,
        "multivariate": (t_multi, multi_data),
        "scenarios": scenarios
    }


if __name__ == "__main__":
    main()
