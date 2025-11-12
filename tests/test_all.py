"""
Unit Tests for Time Series Analysis Project.

This module contains comprehensive unit tests for all components
including data generation, forecasting, anomaly detection, Neural ODE,
and visualization modules.
"""

import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generator import TimeSeriesGenerator, DataConfig, TimeSeriesType
from neural_ode import NeuralODE, NeuralODEConfig, NeuralODETrainer, ODEFunc
from forecasting import (
    ForecastingPipeline, ForecastConfig,
    LSTMForecaster, GRUForecaster, TransformerForecaster
)
from anomaly_detection import (
    AnomalyDetectionPipeline, AnomalyConfig,
    IsolationForestDetector, AutoencoderDetector, StatisticalDetector
)
from visualization import VisualizationConfig, TimeSeriesVisualizer


class TestDataGenerator(unittest.TestCase):
    """Test cases for data generation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DataConfig(
            n_points=100,
            noise_level=0.1,
            trend_strength=0.5,
            seasonality_period=20,
            seed=42
        )
        self.generator = TimeSeriesGenerator(self.config)
    
    def test_data_config_initialization(self):
        """Test DataConfig initialization."""
        config = DataConfig()
        self.assertEqual(config.n_points, 1000)
        self.assertEqual(config.noise_level, 0.1)
        self.assertEqual(config.seed, 42)
    
    def test_time_points_generation(self):
        """Test time points generation."""
        time_points = self.generator.generate_time_points()
        self.assertEqual(len(time_points), self.config.n_points)
        self.assertAlmostEqual(time_points[0], 0.0)
        self.assertAlmostEqual(time_points[-1], 10.0)
    
    def test_sine_wave_generation(self):
        """Test sine wave generation."""
        t = self.generator.generate_time_points()
        sine_wave = self.generator.generate_sine_wave(t)
        
        self.assertEqual(len(sine_wave), len(t))
        self.assertGreaterEqual(np.min(sine_wave), -1.1)
        self.assertLessEqual(np.max(sine_wave), 1.1)
    
    def test_linear_trend_generation(self):
        """Test linear trend generation."""
        t = self.generator.generate_time_points()
        trend = self.generator.generate_linear_trend(t)
        
        self.assertEqual(len(trend), len(t))
        self.assertGreater(trend[-1], trend[0])  # Upward trend
    
    def test_noise_addition(self):
        """Test noise addition."""
        original_signal = np.ones(100)
        noisy_signal = self.generator.add_noise(original_signal)
        
        self.assertEqual(len(noisy_signal), len(original_signal))
        self.assertNotEqual(np.sum(noisy_signal), np.sum(original_signal))
    
    def test_anomaly_addition(self):
        """Test anomaly addition."""
        signal = np.ones(100)
        signal_with_anomalies, anomaly_indices = self.generator.add_anomalies(signal)
        
        self.assertEqual(len(signal_with_anomalies), len(signal))
        self.assertIsInstance(anomaly_indices, list)
    
    def test_complex_time_series_generation(self):
        """Test complex time series generation."""
        t = self.generator.generate_time_points()
        complex_series = self.generator.generate_complex_time_series(t)
        
        self.assertEqual(len(complex_series), len(t))
        self.assertIsInstance(complex_series, np.ndarray)
    
    def test_multivariate_time_series_generation(self):
        """Test multivariate time series generation."""
        t, multi_data = self.generator.generate_multivariate_time_series(n_variables=3)
        
        self.assertEqual(len(t), self.config.n_points)
        self.assertEqual(multi_data.shape[1], 3)
        self.assertEqual(multi_data.shape[0], self.config.n_points)


class TestNeuralODE(unittest.TestCase):
    """Test cases for Neural ODE module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NeuralODEConfig(
            hidden_size=32,
            num_layers=2,
            learning_rate=0.01,
            epochs=10,
            batch_size=16
        )
        self.ode_func = ODEFunc(
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers
        )
        self.model = NeuralODE(self.ode_func)
    
    def test_ode_func_initialization(self):
        """Test ODEFunc initialization."""
        self.assertIsInstance(self.ode_func, ODEFunc)
        self.assertEqual(self.ode_func.hidden_size, self.config.hidden_size)
        self.assertEqual(self.ode_func.num_layers, self.config.num_layers)
    
    def test_ode_func_forward(self):
        """Test ODEFunc forward pass."""
        t = torch.tensor([0.0])
        y = torch.tensor([[1.0]])
        
        output = self.ode_func(t, y)
        
        self.assertEqual(output.shape, y.shape)
        self.assertIsInstance(output, torch.Tensor)
    
    def test_neural_ode_initialization(self):
        """Test NeuralODE initialization."""
        self.assertIsInstance(self.model, NeuralODE)
        self.assertIsInstance(self.model.ode_func, ODEFunc)
    
    def test_neural_ode_forward(self):
        """Test NeuralODE forward pass."""
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 1.0, 2.0])
        
        output = self.model(y0, t)
        
        self.assertEqual(output.shape, (len(t), 1, 1))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_neural_ode_trainer_initialization(self):
        """Test NeuralODETrainer initialization."""
        trainer = NeuralODETrainer(self.model, self.config)
        
        self.assertIsInstance(trainer, NeuralODETrainer)
        self.assertIsInstance(trainer.optimizer, torch.optim.Adam)
        self.assertIsInstance(trainer.loss_fn, torch.nn.MSELoss)
    
    def test_neural_ode_trainer_training(self):
        """Test NeuralODETrainer training."""
        trainer = NeuralODETrainer(self.model, self.config)
        
        # Generate simple training data
        t = torch.linspace(0, 2, 50)
        y_true = torch.sin(t)
        y0 = y_true[0].unsqueeze(0).unsqueeze(-1)
        
        # Train model
        history = trainer.train(y0, t, y_true)
        
        self.assertIsInstance(history, dict)
        self.assertIn('train_losses', history)
        self.assertIn('final_loss', history)
        self.assertEqual(len(history['train_losses']), self.config.epochs)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        trainer = NeuralODETrainer(self.model, self.config)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            filepath = tmp_file.name
        
        try:
            # Save model
            trainer.save_model(filepath)
            self.assertTrue(os.path.exists(filepath))
            
            # Load model
            new_trainer = NeuralODETrainer(self.model, self.config)
            new_trainer.load_model(filepath)
            
            self.assertEqual(len(trainer.train_losses), len(new_trainer.train_losses))
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestForecasting(unittest.TestCase):
    """Test cases for forecasting module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ForecastConfig(
            sequence_length=10,
            hidden_size=32,
            num_layers=2,
            epochs=5,
            batch_size=16,
            learning_rate=0.001
        )
        self.train_data = np.random.randn(100)
        self.test_data = np.random.randn(20)
    
    def test_lstm_forecaster_initialization(self):
        """Test LSTM forecaster initialization."""
        forecaster = LSTMForecaster(self.config)
        
        self.assertIsInstance(forecaster, LSTMForecaster)
        self.assertFalse(forecaster.is_fitted)
    
    def test_lstm_forecaster_fitting(self):
        """Test LSTM forecaster fitting."""
        forecaster = LSTMForecaster(self.config)
        
        # Mock the training to avoid long execution time
        with patch.object(forecaster, '_prepare_data') as mock_prepare:
            mock_prepare.return_value = (
                torch.randn(50, self.config.sequence_length),
                torch.randn(50)
            )
            
            forecaster.fit(self.train_data)
            
            self.assertTrue(forecaster.is_fitted)
            self.assertIsNotNone(forecaster.model)
    
    def test_gru_forecaster_initialization(self):
        """Test GRU forecaster initialization."""
        forecaster = GRUForecaster(self.config)
        
        self.assertIsInstance(forecaster, GRUForecaster)
        self.assertFalse(forecaster.is_fitted)
    
    def test_transformer_forecaster_initialization(self):
        """Test Transformer forecaster initialization."""
        forecaster = TransformerForecaster(self.config)
        
        self.assertIsInstance(forecaster, TransformerForecaster)
        self.assertFalse(forecaster.is_fitted)
    
    def test_forecasting_pipeline_initialization(self):
        """Test forecasting pipeline initialization."""
        pipeline = ForecastingPipeline(self.config)
        
        self.assertIsInstance(pipeline, ForecastingPipeline)
        self.assertEqual(len(pipeline.forecasters), 0)
    
    def test_forecasting_pipeline_add_forecaster(self):
        """Test adding forecasters to pipeline."""
        pipeline = ForecastingPipeline(self.config)
        forecaster = LSTMForecaster(self.config)
        
        pipeline.add_forecaster("LSTM", forecaster)
        
        self.assertEqual(len(pipeline.forecasters), 1)
        self.assertIn("LSTM", pipeline.forecasters)


class TestAnomalyDetection(unittest.TestCase):
    """Test cases for anomaly detection module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AnomalyConfig(
            contamination=0.1,
            epochs=5,
            learning_rate=0.001,
            hidden_size=32
        )
        self.data = np.random.randn(100)
    
    def test_isolation_forest_detector_initialization(self):
        """Test Isolation Forest detector initialization."""
        detector = IsolationForestDetector(self.config)
        
        self.assertIsInstance(detector, IsolationForestDetector)
        self.assertFalse(detector.is_fitted)
    
    def test_isolation_forest_detector_fitting(self):
        """Test Isolation Forest detector fitting."""
        detector = IsolationForestDetector(self.config)
        
        detector.fit(self.data)
        
        self.assertTrue(detector.is_fitted)
        self.assertIsNotNone(detector.model)
    
    def test_isolation_forest_detector_prediction(self):
        """Test Isolation Forest detector prediction."""
        detector = IsolationForestDetector(self.config)
        detector.fit(self.data)
        
        predictions = detector.predict(self.data)
        
        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
    
    def test_autoencoder_detector_initialization(self):
        """Test Autoencoder detector initialization."""
        detector = AutoencoderDetector(self.config)
        
        self.assertIsInstance(detector, AutoencoderDetector)
        self.assertFalse(detector.is_fitted)
    
    def test_statistical_detector_initialization(self):
        """Test Statistical detector initialization."""
        detector = StatisticalDetector(self.config)
        
        self.assertIsInstance(detector, StatisticalDetector)
        self.assertFalse(detector.is_fitted)
    
    def test_statistical_detector_fitting(self):
        """Test Statistical detector fitting."""
        detector = StatisticalDetector(self.config)
        
        detector.fit(self.data)
        
        self.assertTrue(detector.is_fitted)
        self.assertIsNotNone(detector.z_threshold)
        self.assertIsNotNone(detector.iqr_bounds)
    
    def test_statistical_detector_prediction(self):
        """Test Statistical detector prediction."""
        detector = StatisticalDetector(self.config)
        detector.fit(self.data)
        
        predictions = detector.predict(self.data)
        
        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
    
    def test_anomaly_detection_pipeline_initialization(self):
        """Test anomaly detection pipeline initialization."""
        pipeline = AnomalyDetectionPipeline(self.config)
        
        self.assertIsInstance(pipeline, AnomalyDetectionPipeline)
        self.assertEqual(len(pipeline.detectors), 0)
    
    def test_anomaly_detection_pipeline_add_detector(self):
        """Test adding detectors to pipeline."""
        pipeline = AnomalyDetectionPipeline(self.config)
        detector = IsolationForestDetector(self.config)
        
        pipeline.add_detector("Isolation Forest", detector)
        
        self.assertEqual(len(pipeline.detectors), 1)
        self.assertIn("Isolation Forest", pipeline.detectors)


class TestVisualization(unittest.TestCase):
    """Test cases for visualization module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = VisualizationConfig()
        self.visualizer = TimeSeriesVisualizer(self.config)
        self.data = np.random.randn(100)
        self.time_points = np.arange(100)
    
    def test_visualization_config_initialization(self):
        """Test VisualizationConfig initialization."""
        config = VisualizationConfig()
        
        self.assertEqual(config.figure_size, (12, 8))
        self.assertEqual(config.dpi, 100)
        self.assertEqual(config.style, "seaborn-v0_8")
    
    def test_time_series_visualizer_initialization(self):
        """Test TimeSeriesVisualizer initialization."""
        visualizer = TimeSeriesVisualizer(self.config)
        
        self.assertIsInstance(visualizer, TimeSeriesVisualizer)
        self.assertEqual(visualizer.config, self.config)
    
    def test_plot_time_series_without_show(self):
        """Test time series plotting without showing."""
        # This should not raise an exception
        self.visualizer.plot_time_series(
            self.data, 
            self.time_points, 
            show=False
        )
    
    def test_plot_multiple_series_without_show(self):
        """Test multiple series plotting without showing."""
        data_dict = {
            'series1': self.data,
            'series2': self.data + 1
        }
        
        # This should not raise an exception
        self.visualizer.plot_multiple_series(
            data_dict,
            self.time_points,
            show=False
        )
    
    def test_plot_distribution_without_show(self):
        """Test distribution plotting without showing."""
        # This should not raise an exception
        self.visualizer.plot_distribution(
            self.data,
            show=False
        )


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_config = DataConfig(n_points=100, seed=42)
        self.generator = TimeSeriesGenerator(self.data_config)
        self.time_points = self.generator.generate_time_points()
        self.data = self.generator.generate_complex_time_series(self.time_points)
    
    def test_end_to_end_data_generation_to_visualization(self):
        """Test complete pipeline from data generation to visualization."""
        # Generate data
        self.assertIsNotNone(self.data)
        self.assertEqual(len(self.data), self.data_config.n_points)
        
        # Create visualizer
        viz_config = VisualizationConfig()
        visualizer = TimeSeriesVisualizer(viz_config)
        
        # Plot data (should not raise exception)
        visualizer.plot_time_series(self.data, self.time_points, show=False)
    
    def test_end_to_end_anomaly_detection_pipeline(self):
        """Test complete anomaly detection pipeline."""
        # Add some anomalies
        anomaly_indices = [10, 20, 30]
        data_with_anomalies = self.data.copy()
        data_with_anomalies[anomaly_indices] += 3.0
        
        # Initialize anomaly detection
        anomaly_config = AnomalyConfig(contamination=0.1)
        pipeline = AnomalyDetectionPipeline(anomaly_config)
        
        # Add detectors
        pipeline.add_detector("Statistical", StatisticalDetector(anomaly_config))
        
        # Fit and predict
        pipeline.fit_all(data_with_anomalies)
        predictions = pipeline.predict_all(data_with_anomalies)
        
        # Check results
        self.assertIn("Statistical", predictions)
        self.assertEqual(len(predictions["Statistical"]), len(data_with_anomalies))


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataGenerator,
        TestNeuralODE,
        TestForecasting,
        TestAnomalyDetection,
        TestVisualization,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
