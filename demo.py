#!/usr/bin/env python3
"""
Time Series Analysis Project - Complete Demonstration Script

This script demonstrates the complete functionality of the modernized
time series analysis project including Neural ODEs, forecasting,
anomaly detection, and visualization.
"""

import os
import sys
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import our modules
from data_generator import TimeSeriesGenerator, DataConfig
from neural_ode import NeuralODE, NeuralODEConfig, NeuralODETrainer, ODEFunc
from forecasting import ForecastingPipeline, ForecastConfig, LSTMForecaster, GRUForecaster
from anomaly_detection import AnomalyDetectionPipeline, AnomalyConfig, IsolationForestDetector, StatisticalDetector
from visualization import TimeSeriesVisualizer, VisualizationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main demonstration function."""
    logger.info("Starting Time Series Analysis Project Demonstration")
    
    # 1. Data Generation
    logger.info("1. Generating synthetic time series data...")
    data_config = DataConfig(
        n_points=1000,
        noise_level=0.1,
        trend_strength=0.5,
        seasonality_period=50,
        anomaly_probability=0.05,
        seed=42
    )
    
    generator = TimeSeriesGenerator(data_config)
    t = generator.generate_time_points()
    data = generator.generate_complex_time_series(t)
    
    logger.info(f"   Generated {len(data)} data points")
    logger.info(f"   Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    
    # 2. Neural ODE Training
    logger.info("2. Training Neural ODE model...")
    neural_ode_config = NeuralODEConfig(
        hidden_size=64,
        num_layers=3,
        learning_rate=0.01,
        epochs=100,  # Reduced for demo
        dropout=0.1
    )
    
    ode_func = ODEFunc(
        hidden_size=neural_ode_config.hidden_size,
        num_layers=neural_ode_config.num_layers,
        dropout=neural_ode_config.dropout
    )
    model = NeuralODE(ode_func)
    trainer = NeuralODETrainer(model, neural_ode_config)
    
    # Train model
    y0 = torch.tensor(data[0]).unsqueeze(0).unsqueeze(-1)
    t_tensor = torch.tensor(t)
    y_target = torch.tensor(data)
    history = trainer.train(y0, t_tensor, y_target)
    
    logger.info(f"   Neural ODE training completed. Final loss: {history['final_loss']:.6f}")
    logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Forecasting Comparison
    logger.info("3. Running forecasting comparison...")
    forecast_config = ForecastConfig(
        sequence_length=60,
        hidden_size=50,
        num_layers=2,
        epochs=50,  # Reduced for demo
        learning_rate=0.001
    )
    
    # Split data
    train_size = int(len(data) * 0.8)
    test_size = int(len(data) * 0.1)
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    
    # Initialize pipeline
    pipeline = ForecastingPipeline(forecast_config)
    pipeline.add_forecaster("LSTM", LSTMForecaster(forecast_config))
    pipeline.add_forecaster("GRU", GRUForecaster(forecast_config))
    
    # Train and predict
    pipeline.fit_all(train_data)
    predictions = pipeline.predict_all(len(test_data))
    
    # Evaluate
    results = pipeline.evaluate_models(test_data, predictions)
    
    logger.info("   Forecasting results:")
    for model_name, metrics in results.items():
        logger.info(f"     {model_name} RMSE: {metrics['RMSE']:.4f}")
    
    # 4. Anomaly Detection
    logger.info("4. Running anomaly detection...")
    anomaly_config = AnomalyConfig(
        contamination=0.1,
        epochs=50,  # Reduced for demo
        learning_rate=0.001
    )
    
    anomaly_pipeline = AnomalyDetectionPipeline(anomaly_config)
    anomaly_pipeline.add_detector("Isolation Forest", IsolationForestDetector(anomaly_config))
    anomaly_pipeline.add_detector("Statistical", StatisticalDetector(anomaly_config))
    
    anomaly_pipeline.fit_all(data)
    anomaly_predictions = anomaly_pipeline.predict_all(data)
    
    logger.info("   Anomaly detection results:")
    for method_name, predictions in anomaly_predictions.items():
        n_anomalies = np.sum(predictions)
        logger.info(f"     {method_name}: {n_anomalies} anomalies detected")
    
    # 5. Visualization
    logger.info("5. Creating visualizations...")
    viz_config = VisualizationConfig()
    visualizer = TimeSeriesVisualizer(viz_config)
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Plot Neural ODE results
    predictions_neural_ode, _ = model.predict(y0, t_tensor)
    y_pred_neural_ode = predictions_neural_ode.squeeze().detach().numpy()
    
    plt.figure(figsize=(12, 8))
    plt.plot(t, data, label='Original Data', alpha=0.7, linewidth=1)
    plt.plot(t, y_pred_neural_ode, label='Neural ODE Prediction', linewidth=2, color='red')
    plt.title('Neural ODE Time Series Smoothing')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'neural_ode_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot forecasting results
    plt.figure(figsize=(12, 8))
    plt.plot(range(train_size), train_data, label='Training Data', alpha=0.7)
    
    test_start = train_size
    test_end = test_start + len(test_data)
    plt.plot(range(test_start, test_end), test_data, label='Actual', linewidth=2, color='green')
    
    colors = ['red', 'blue', 'orange']
    for i, (model_name, pred) in enumerate(predictions.items()):
        plt.plot(range(test_start, test_end), pred, 
                label=f'{model_name} Prediction', 
                linestyle='--', linewidth=2, color=colors[i])
    
    plt.title('Forecasting Results Comparison')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'forecasting_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot anomaly detection results
    n_methods = len(anomaly_predictions)
    fig, axes = plt.subplots(n_methods + 1, 1, figsize=(12, 4 * (n_methods + 1)))
    
    # Original data
    axes[0].plot(t, data, label='Original Data', alpha=0.7)
    axes[0].set_title('Original Time Series')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Anomaly detection results
    for i, (method_name, predictions) in enumerate(anomaly_predictions.items()):
        ax = axes[i + 1]
        
        ax.plot(t, data, label='Data', alpha=0.7, color='blue')
        
        anomaly_indices = np.where(predictions == 1)[0]
        if len(anomaly_indices) > 0:
            ax.scatter(anomaly_indices, data[anomaly_indices], 
                      color='red', s=50, label='Anomalies', zorder=5)
        
        ax.set_title(f'{method_name} Anomaly Detection')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'anomaly_detection_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   Visualizations saved to {output_dir}/")
    
    # 6. Model Persistence
    logger.info("6. Saving models...")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save Neural ODE model
    neural_ode_path = models_dir / 'neural_ode_model.pth'
    trainer.save_model(str(neural_ode_path))
    logger.info(f"   Neural ODE model saved to {neural_ode_path}")
    
    # 7. Summary
    logger.info("=== PROJECT DEMONSTRATION COMPLETED ===")
    logger.info(f"✓ Generated {len(data)} data points")
    logger.info(f"✓ Trained Neural ODE with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"✓ Compared {len(predictions)} forecasting models")
    logger.info(f"✓ Detected anomalies using {len(anomaly_predictions)} methods")
    logger.info(f"✓ Created visualizations in {output_dir}/")
    logger.info(f"✓ Saved models in {models_dir}/")
    
    logger.info("\n=== NEXT STEPS ===")
    logger.info("1. Run the Streamlit web interface: streamlit run app.py")
    logger.info("2. Explore the Jupyter notebook: notebooks/complete_analysis_example.ipynb")
    logger.info("3. Run the test suite: python -m pytest tests/ -v")
    logger.info("4. Check the comprehensive README.md for detailed usage instructions")
    
    logger.info("\n🎉 Time Series Analysis Project is ready for use!")


if __name__ == "__main__":
    main()
