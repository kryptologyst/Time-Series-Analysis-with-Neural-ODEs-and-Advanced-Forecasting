# Time Series Analysis with Neural ODEs and Advanced Forecasting

A comprehensive Python project for time series analysis featuring Neural Ordinary Differential Equations (Neural ODEs), state-of-the-art forecasting methods, anomaly detection, and interactive visualization capabilities.

## Features

### Core Capabilities
- **Neural ODEs**: Continuous-time modeling using differentiable ODE solvers
- **Advanced Forecasting**: ARIMA, Prophet, LSTM, GRU, and Transformer models
- **Anomaly Detection**: Isolation Forest, Autoencoders, Statistical methods, and ensemble approaches
- **Data Generation**: Realistic synthetic time series with trends, seasonality, and noise
- **Interactive Visualization**: Matplotlib, Seaborn, and Plotly-based plots
- **Web Interface**: Streamlit dashboard for interactive exploration

### Technical Highlights
- **Modern Python**: Type hints, docstrings, PEP8 compliance
- **Comprehensive Testing**: Unit tests for all modules
- **Configuration Management**: YAML-based configuration
- **Model Persistence**: Checkpoint saving and loading
- **Performance Monitoring**: Training metrics and model comparison
- **Extensible Architecture**: Modular design for easy extension

## Installation

### Prerequisites
- Python 3.10+
- pip or conda package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Time-Series-Analysis-with-Neural-ODEs-and-Advanced-Forecasting.git
cd Time-Series-Analysis-with-Neural-ODEs-and-Advanced-Forecasting

# Install dependencies
pip install -r requirements.txt
```

### Optional GPU Support
For GPU acceleration (recommended for deep learning models):

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Web Interface (Recommended)
Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to access the full-featured web interface.

### 2. Command Line Usage

#### Generate Synthetic Data
```python
from src.data_generator import TimeSeriesGenerator, DataConfig

# Configure data generation
config = DataConfig(
    n_points=1000,
    noise_level=0.1,
    trend_strength=0.5,
    seasonality_period=50
)

# Generate complex time series
generator = TimeSeriesGenerator(config)
t, data = generator.generate_complex_time_series(generator.generate_time_points())
```

#### Neural ODE Training
```python
from src.neural_ode import NeuralODE, NeuralODEConfig, NeuralODETrainer, ODEFunc

# Configure Neural ODE
config = NeuralODEConfig(
    hidden_size=64,
    num_layers=3,
    learning_rate=0.01,
    epochs=200
)

# Initialize model
ode_func = ODEFunc(hidden_size=config.hidden_size, num_layers=config.num_layers)
model = NeuralODE(ode_func)
trainer = NeuralODETrainer(model, config)

# Train model
y0 = torch.tensor(data[0]).unsqueeze(0).unsqueeze(-1)
t_tensor = torch.tensor(t)
y_target = torch.tensor(data)
history = trainer.train(y0, t_tensor, y_target)
```

#### Forecasting Comparison
```python
from src.forecasting import ForecastingPipeline, ForecastConfig, LSTMForecaster, GRUForecaster

# Configure forecasting
config = ForecastConfig(epochs=100, learning_rate=0.001)

# Initialize pipeline
pipeline = ForecastingPipeline(config)
pipeline.add_forecaster("LSTM", LSTMForecaster(config))
pipeline.add_forecaster("GRU", GRUForecaster(config))

# Train and predict
train_data = data[:800]
test_data = data[800:900]
pipeline.fit_all(train_data)
predictions = pipeline.predict_all(50)
```

#### Anomaly Detection
```python
from src.anomaly_detection import AnomalyDetectionPipeline, AnomalyConfig, IsolationForestDetector

# Configure anomaly detection
config = AnomalyConfig(contamination=0.1)

# Initialize pipeline
pipeline = AnomalyDetectionPipeline(config)
pipeline.add_detector("Isolation Forest", IsolationForestDetector(config))

# Detect anomalies
pipeline.fit_all(data)
anomalies = pipeline.predict_all(data)
```

## Project Structure

```
time-series-analysis/
├── src/                          # Source code
│   ├── neural_ode.py            # Neural ODE implementation
│   ├── data_generator.py        # Synthetic data generation
│   ├── forecasting.py           # Forecasting methods
│   ├── anomaly_detection.py     # Anomaly detection
│   └── visualization.py         # Visualization utilities
├── tests/                        # Unit tests
│   └── test_all.py              # Comprehensive test suite
├── config/                       # Configuration files
│   └── config.yaml              # Main configuration
├── data/                         # Data storage
│   ├── raw/                      # Raw data files
│   └── processed/                # Processed data files
├── models/                       # Model storage
│   └── checkpoints/              # Saved model checkpoints
├── notebooks/                     # Jupyter notebooks
├── logs/                         # Log files
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Configuration

The project uses YAML configuration files for easy customization. Key configuration options:

### Data Generation (`config/config.yaml`)
```yaml
data:
  n_points: 1000
  noise_level: 0.1
  trend_strength: 0.5
  seasonality_period: 50
  anomaly_probability: 0.05
```

### Model Settings
```yaml
models:
  neural_ode:
    hidden_size: 64
    num_layers: 3
    learning_rate: 0.01
    epochs: 500
  
  lstm:
    hidden_size: 50
    num_layers: 2
    dropout: 0.2
    epochs: 100
```

## Advanced Usage

### Custom Data Sources
```python
# Load your own time series data
import pandas as pd

# Load CSV data
df = pd.read_csv('your_data.csv')
data = df['value'].values
time_points = df['timestamp'].values

# Use with Neural ODE
config = NeuralODEConfig()
ode_func = ODEFunc(hidden_size=64)
model = NeuralODE(ode_func)
trainer = NeuralODETrainer(model, config)

# Train on your data
y0 = torch.tensor(data[0]).unsqueeze(0).unsqueeze(-1)
t_tensor = torch.tensor(time_points)
y_target = torch.tensor(data)
trainer.train(y0, t_tensor, y_target)
```

### Model Comparison Dashboard
```python
from src.visualization import InteractiveVisualizer

# Create interactive comparison
viz = InteractiveVisualizer(VisualizationConfig())
fig = viz.create_model_comparison_dashboard(forecast_results)
fig.show()
```

### Ensemble Methods
```python
# Create ensemble predictions
ensemble_pred = pipeline.evaluate_ensemble(anomaly_predictions, threshold=0.5)
```

## Performance Optimization

### GPU Acceleration
```python
# Check GPU availability
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Models automatically use GPU when available
```

### Memory Management
```python
# For large datasets, use smaller batch sizes
config = ForecastConfig(batch_size=16)  # Reduce from default 32

# Use gradient accumulation for effective larger batch sizes
# (implemented in training loops)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_all.py::TestNeuralODE -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/
```

### Adding New Models
1. Create a new forecaster class inheriting from `BaseForecaster`
2. Implement required methods: `fit()`, `predict()`, `get_model_name()`
3. Add to the forecasting pipeline
4. Write unit tests
5. Update documentation

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python version (3.10+ required)
python --version
```

#### Memory Issues
```python
# Reduce batch size and sequence length
config = ForecastConfig(
    batch_size=16,      # Reduce from 32
    sequence_length=30  # Reduce from 60
)
```

#### GPU Issues
```python
# Check CUDA installation
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# Fallback to CPU
device = torch.device("cpu")
```

### Performance Tips
- Use GPU acceleration for deep learning models
- Adjust batch sizes based on available memory
- Use smaller sequence lengths for faster training
- Enable mixed precision training for large models

## Examples

### Complete Workflow Example
```python
import numpy as np
import torch
from src.data_generator import TimeSeriesGenerator, DataConfig
from src.neural_ode import NeuralODE, NeuralODEConfig, NeuralODETrainer, ODEFunc
from src.forecasting import ForecastingPipeline, ForecastConfig, LSTMForecaster
from src.anomaly_detection import AnomalyDetectionPipeline, AnomalyConfig, IsolationForestDetector
from src.visualization import TimeSeriesVisualizer, VisualizationConfig

# 1. Generate synthetic data
data_config = DataConfig(n_points=1000, noise_level=0.1)
generator = TimeSeriesGenerator(data_config)
t, data = generator.generate_complex_time_series(generator.generate_time_points())

# 2. Train Neural ODE
neural_ode_config = NeuralODEConfig(epochs=200)
ode_func = ODEFunc(hidden_size=64, num_layers=3)
model = NeuralODE(ode_func)
trainer = NeuralODETrainer(model, neural_ode_config)

y0 = torch.tensor(data[0]).unsqueeze(0).unsqueeze(-1)
t_tensor = torch.tensor(t)
y_target = torch.tensor(data)
trainer.train(y0, t_tensor, y_target)

# 3. Forecasting comparison
forecast_config = ForecastConfig(epochs=100)
pipeline = ForecastingPipeline(forecast_config)
pipeline.add_forecaster("LSTM", LSTMForecaster(forecast_config))

train_data = data[:800]
test_data = data[800:900]
pipeline.fit_all(train_data)
predictions = pipeline.predict_all(50)

# 4. Anomaly detection
anomaly_config = AnomalyConfig(contamination=0.1)
anomaly_pipeline = AnomalyDetectionPipeline(anomaly_config)
anomaly_pipeline.add_detector("Isolation Forest", IsolationForestDetector(anomaly_config))
anomaly_pipeline.fit_all(data)
anomalies = anomaly_pipeline.predict_all(data)

# 5. Visualization
viz_config = VisualizationConfig()
visualizer = TimeSeriesVisualizer(viz_config)
visualizer.plot_time_series(data, t, title="Complete Analysis Results")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{time_series_analysis,
  title={Time Series Analysis with Neural ODEs and Advanced Forecasting},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Time-Series-Analysis-with-Neural-ODEs-and-Advanced-Forecasting}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Streamlit team for the web interface capabilities
- Plotly team for interactive visualization
- The open-source community for various time series libraries

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the test examples
- Contact the maintainers

 
# Time-Series-Analysis-with-Neural-ODEs-and-Advanced-Forecasting
