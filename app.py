"""
Streamlit Web Interface for Time Series Analysis.

This module provides a comprehensive web interface for exploring time series
analysis capabilities including data generation, forecasting, anomaly detection,
and visualization using Streamlit.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, Any, Optional
import yaml
import io
import base64

# Import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_generator import TimeSeriesGenerator, DataConfig, TimeSeriesType
from src.neural_ode import NeuralODE, NeuralODEConfig, NeuralODETrainer, ODEFunc
from src.forecasting import (
    ForecastingPipeline, ForecastConfig,
    ARIMAForecaster, ProphetForecaster, LSTMForecaster,
    GRUForecaster, TransformerForecaster
)
from src.anomaly_detection import (
    AnomalyDetectionPipeline, AnomalyConfig,
    IsolationForestDetector, AutoencoderDetector, StatisticalDetector
)
from src.visualization import VisualizationConfig, InteractiveVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Time Series Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        self.data = None
        self.time_points = None
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open('config/config.yaml', 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            st.warning("Configuration file not found. Using default settings.")
            return {}
    
    def run(self):
        """Run the Streamlit application."""
        # Header
        st.markdown('<h1 class="main-header">📈 Time Series Analysis Dashboard</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self._create_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏠 Home", "📊 Data Generation", "🔮 Forecasting", "🚨 Anomaly Detection", "📈 Neural ODE"
        ])
        
        with tab1:
            self._show_home()
        
        with tab2:
            self._show_data_generation()
        
        with tab3:
            self._show_forecasting()
        
        with tab4:
            self._show_anomaly_detection()
        
        with tab5:
            self._show_neural_ode()
    
    def _create_sidebar(self):
        """Create sidebar with configuration options."""
        st.sidebar.title("⚙️ Configuration")
        
        # Data parameters
        st.sidebar.subheader("📊 Data Parameters")
        self.n_points = st.sidebar.slider("Number of Points", 100, 2000, 1000)
        self.noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)
        self.trend_strength = st.sidebar.slider("Trend Strength", 0.0, 2.0, 0.5)
        self.seasonality_period = st.sidebar.slider("Seasonality Period", 10, 200, 50)
        
        # Model parameters
        st.sidebar.subheader("🤖 Model Parameters")
        self.epochs = st.sidebar.slider("Training Epochs", 50, 500, 200)
        self.learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
        self.hidden_size = st.sidebar.slider("Hidden Size", 16, 128, 64)
        
        # Anomaly detection
        st.sidebar.subheader("🚨 Anomaly Detection")
        self.contamination = st.sidebar.slider("Contamination", 0.01, 0.3, 0.1, 0.01)
        self.anomaly_probability = st.sidebar.slider("Anomaly Probability", 0.0, 0.2, 0.05, 0.01)
    
    def _show_home(self):
        """Show home page."""
        st.markdown('<h2 class="section-header">Welcome to Time Series Analysis Dashboard</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🔮 Forecasting</h3>
                <p>Compare multiple forecasting methods including ARIMA, Prophet, LSTM, GRU, and Transformer models.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🚨 Anomaly Detection</h3>
                <p>Detect anomalies using Isolation Forest, Autoencoders, Statistical methods, and ensemble approaches.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📈 Neural ODE</h3>
                <p>Explore Neural Ordinary Differential Equations for continuous-time modeling of time series data.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        ## 🚀 Getting Started
        
        1. **Generate Data**: Use the Data Generation tab to create synthetic time series with various characteristics
        2. **Forecast**: Apply multiple forecasting models and compare their performance
        3. **Detect Anomalies**: Identify unusual patterns in your data using state-of-the-art methods
        4. **Neural ODE**: Experiment with continuous-time neural networks for smooth trajectory modeling
        
        ## 📚 Features
        
        - **Multiple Data Types**: Generate realistic time series with trends, seasonality, and noise
        - **Advanced Forecasting**: ARIMA, Prophet, Deep Learning models (LSTM, GRU, Transformer)
        - **Anomaly Detection**: Statistical, Machine Learning, and Deep Learning approaches
        - **Interactive Visualizations**: Plotly-based interactive plots
        - **Model Comparison**: Side-by-side performance evaluation
        - **Export Results**: Download plots and model results
        """)
    
    def _show_data_generation(self):
        """Show data generation interface."""
        st.markdown('<h2 class="section-header">📊 Data Generation</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configuration")
            
            # Data type selection
            data_type = st.selectbox(
                "Data Type",
                ["Complex", "Sine Wave", "Linear Trend", "Random Walk", "Stock Price", "Energy Consumption"]
            )
            
            # Generate button
            if st.button("Generate Data", type="primary"):
                with st.spinner("Generating data..."):
                    self._generate_data(data_type)
        
        with col2:
            if self.data is not None:
                st.subheader("Generated Data")
                
                # Plot the data
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=self.time_points,
                    y=self.data,
                    mode='lines',
                    name='Generated Data',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    title=f"Generated {data_type} Time Series",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Data statistics
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                
                with col_stats1:
                    st.metric("Mean", f"{np.mean(self.data):.3f}")
                
                with col_stats2:
                    st.metric("Std Dev", f"{np.std(self.data):.3f}")
                
                with col_stats3:
                    st.metric("Min", f"{np.min(self.data):.3f}")
                
                with col_stats4:
                    st.metric("Max", f"{np.max(self.data):.3f}")
                
                # Download data
                if st.button("Download Data"):
                    df = pd.DataFrame({
                        'time': self.time_points,
                        'value': self.data
                    })
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{data_type.lower().replace(' ', '_')}_data.csv",
                        mime="text/csv"
                    )
    
    def _generate_data(self, data_type: str):
        """Generate data based on selected type."""
        try:
            config = DataConfig(
                n_points=self.n_points,
                noise_level=self.noise_level,
                trend_strength=self.trend_strength,
                seasonality_period=self.seasonality_period,
                anomaly_probability=self.anomaly_probability
            )
            
            generator = TimeSeriesGenerator(config)
            self.time_points = generator.generate_time_points()
            
            if data_type == "Complex":
                self.data = generator.generate_complex_time_series(self.time_points)
            elif data_type == "Sine Wave":
                self.data = generator.generate_sine_wave(self.time_points)
            elif data_type == "Linear Trend":
                self.data = generator.generate_linear_trend(self.time_points)
            elif data_type == "Random Walk":
                self.data = generator.generate_random_walk(self.time_points)
            elif data_type == "Stock Price":
                _, self.data = generator._generate_stock_price(self.time_points)
            elif data_type == "Energy Consumption":
                _, self.data = generator._generate_energy_consumption(self.time_points)
            
            st.success(f"Successfully generated {data_type} data with {len(self.data)} points!")
            
        except Exception as e:
            st.error(f"Error generating data: {e}")
    
    def _show_forecasting(self):
        """Show forecasting interface."""
        st.markdown('<h2 class="section-header">🔮 Forecasting</h2>', unsafe_allow_html=True)
        
        if self.data is None:
            st.warning("Please generate data first in the Data Generation tab.")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Model Selection")
            
            # Model selection
            selected_models = st.multiselect(
                "Select Models",
                ["ARIMA", "Prophet", "LSTM", "GRU", "Transformer"],
                default=["LSTM", "GRU"]
            )
            
            # Forecast horizon
            forecast_horizon = st.slider("Forecast Horizon", 10, 200, 50)
            
            # Train/test split
            train_ratio = st.slider("Training Ratio", 0.5, 0.9, 0.8)
            
            if st.button("Run Forecasting", type="primary"):
                with st.spinner("Training models and making predictions..."):
                    self._run_forecasting(selected_models, forecast_horizon, train_ratio)
        
        with col2:
            if hasattr(self, 'forecast_results'):
                st.subheader("Forecasting Results")
                
                # Plot results
                fig = go.Figure()
                
                # Training data
                train_size = int(len(self.data) * train_ratio)
                fig.add_trace(go.Scatter(
                    x=list(range(train_size)),
                    y=self.data[:train_size],
                    mode='lines',
                    name='Training Data',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Test data
                test_data = self.data[train_size:train_size + forecast_horizon]
                fig.add_trace(go.Scatter(
                    x=list(range(train_size, train_size + len(test_data))),
                    y=test_data,
                    mode='lines',
                    name='Actual',
                    line=dict(color='#ff7f0e', width=2)
                ))
                
                # Predictions
                colors = px.colors.qualitative.Set2
                for i, (model_name, predictions) in enumerate(self.forecast_results.items()):
                    fig.add_trace(go.Scatter(
                        x=list(range(train_size, train_size + len(predictions))),
                        y=predictions,
                        mode='lines',
                        name=f'{model_name} Prediction',
                        line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    title="Forecasting Results Comparison",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    height=500,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics
                if hasattr(self, 'forecast_metrics'):
                    st.subheader("Performance Metrics")
                    
                    metrics_df = pd.DataFrame(self.forecast_metrics).T
                    st.dataframe(metrics_df, use_container_width=True)
    
    def _run_forecasting(self, selected_models: list, forecast_horizon: int, train_ratio: float):
        """Run forecasting with selected models."""
        try:
            # Split data
            train_size = int(len(self.data) * train_ratio)
            train_data = self.data[:train_size]
            test_data = self.data[train_size:train_size + forecast_horizon]
            
            # Configuration
            config = ForecastConfig(
                train_ratio=train_ratio,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                hidden_size=self.hidden_size
            )
            
            # Initialize pipeline
            pipeline = ForecastingPipeline(config)
            
            # Add selected models
            if "ARIMA" in selected_models:
                pipeline.add_forecaster("ARIMA", ARIMAForecaster(config))
            if "Prophet" in selected_models:
                pipeline.add_forecaster("Prophet", ProphetForecaster(config))
            if "LSTM" in selected_models:
                pipeline.add_forecaster("LSTM", LSTMForecaster(config))
            if "GRU" in selected_models:
                pipeline.add_forecaster("GRU", GRUForecaster(config))
            if "Transformer" in selected_models:
                pipeline.add_forecaster("Transformer", TransformerForecaster(config))
            
            # Fit models
            pipeline.fit_all(train_data)
            
            # Make predictions
            predictions = pipeline.predict_all(forecast_horizon)
            self.forecast_results = predictions
            
            # Evaluate models
            if len(test_data) > 0:
                metrics = pipeline.evaluate_models(test_data, predictions)
                self.forecast_metrics = metrics
            
            st.success(f"Successfully trained {len(selected_models)} models and made predictions!")
            
        except Exception as e:
            st.error(f"Error in forecasting: {e}")
    
    def _show_anomaly_detection(self):
        """Show anomaly detection interface."""
        st.markdown('<h2 class="section-header">🚨 Anomaly Detection</h2>', unsafe_allow_html=True)
        
        if self.data is None:
            st.warning("Please generate data first in the Data Generation tab.")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Detection Methods")
            
            # Method selection
            selected_methods = st.multiselect(
                "Select Methods",
                ["Isolation Forest", "Autoencoder", "Statistical", "LSTM Autoencoder"],
                default=["Isolation Forest", "Statistical"]
            )
            
            if st.button("Detect Anomalies", type="primary"):
                with st.spinner("Detecting anomalies..."):
                    self._run_anomaly_detection(selected_methods)
        
        with col2:
            if hasattr(self, 'anomaly_results'):
                st.subheader("Anomaly Detection Results")
                
                # Plot results
                fig = go.Figure()
                
                # Normal data
                normal_indices = []
                anomaly_indices = []
                
                for i, (method_name, predictions) in enumerate(self.anomaly_results.items()):
                    method_anomalies = np.where(predictions == 1)[0]
                    if i == 0:  # Use first method for plotting
                        normal_indices = np.where(predictions == 0)[0]
                        anomaly_indices = method_anomalies
                
                # Plot normal data
                fig.add_trace(go.Scatter(
                    x=normal_indices,
                    y=self.data[normal_indices],
                    mode='markers',
                    name='Normal Data',
                    marker=dict(color='#1f77b4', size=4)
                ))
                
                # Plot anomalies
                if len(anomaly_indices) > 0:
                    fig.add_trace(go.Scatter(
                        x=anomaly_indices,
                        y=self.data[anomaly_indices],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='#d62728', size=8)
                    ))
                
                fig.update_layout(
                    title="Anomaly Detection Results",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    height=500,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly counts
                st.subheader("Anomaly Counts by Method")
                anomaly_counts = {method: np.sum(predictions) for method, predictions in self.anomaly_results.items()}
                counts_df = pd.DataFrame(list(anomaly_counts.items()), columns=['Method', 'Anomaly Count'])
                st.dataframe(counts_df, use_container_width=True)
    
    def _run_anomaly_detection(self, selected_methods: list):
        """Run anomaly detection with selected methods."""
        try:
            # Configuration
            config = AnomalyConfig(
                contamination=self.contamination,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                hidden_size=self.hidden_size
            )
            
            # Initialize pipeline
            pipeline = AnomalyDetectionPipeline(config)
            
            # Add selected methods
            if "Isolation Forest" in selected_methods:
                pipeline.add_detector("Isolation Forest", IsolationForestDetector(config))
            if "Autoencoder" in selected_methods:
                pipeline.add_detector("Autoencoder", AutoencoderDetector(config))
            if "Statistical" in selected_methods:
                pipeline.add_detector("Statistical", StatisticalDetector(config))
            if "LSTM Autoencoder" in selected_methods:
                from src.anomaly_detection import LSTMAutoencoderDetector
                pipeline.add_detector("LSTM Autoencoder", LSTMAutoencoderDetector(config))
            
            # Fit models
            pipeline.fit_all(self.data)
            
            # Detect anomalies
            predictions = pipeline.predict_all(self.data)
            self.anomaly_results = predictions
            
            st.success(f"Successfully detected anomalies using {len(selected_methods)} methods!")
            
        except Exception as e:
            st.error(f"Error in anomaly detection: {e}")
    
    def _show_neural_ode(self):
        """Show Neural ODE interface."""
        st.markdown('<h2 class="section-header">📈 Neural ODE</h2>', unsafe_allow_html=True)
        
        if self.data is None:
            st.warning("Please generate data first in the Data Generation tab.")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Neural ODE Configuration")
            
            # Model parameters
            num_layers = st.slider("Number of Layers", 1, 5, 3)
            dropout = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
            activation = st.selectbox("Activation", ["tanh", "relu", "elu"])
            
            if st.button("Train Neural ODE", type="primary"):
                with st.spinner("Training Neural ODE..."):
                    self._run_neural_ode(num_layers, dropout, activation)
        
        with col2:
            if hasattr(self, 'neural_ode_results'):
                st.subheader("Neural ODE Results")
                
                # Plot results
                fig = go.Figure()
                
                # Original data
                fig.add_trace(go.Scatter(
                    x=self.time_points,
                    y=self.data,
                    mode='lines',
                    name='Original Data',
                    line=dict(color='#1f77b4', width=1, opacity=0.7)
                ))
                
                # Neural ODE prediction
                fig.add_trace(go.Scatter(
                    x=self.time_points,
                    y=self.neural_ode_results['predictions'],
                    mode='lines',
                    name='Neural ODE Prediction',
                    line=dict(color='#d62728', width=2)
                ))
                
                fig.update_layout(
                    title="Neural ODE Time Series Smoothing",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    height=500,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Training metrics
                st.subheader("Training Metrics")
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                
                with col_metric1:
                    st.metric("Final Loss", f"{self.neural_ode_results['final_loss']:.6f}")
                
                with col_metric2:
                    st.metric("Training Epochs", self.epochs)
                
                with col_metric3:
                    st.metric("Model Parameters", f"{self.neural_ode_results['num_params']:,}")
    
    def _run_neural_ode(self, num_layers: int, dropout: float, activation: str):
        """Run Neural ODE training."""
        try:
            # Configuration
            config = NeuralODEConfig(
                hidden_size=self.hidden_size,
                num_layers=num_layers,
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                dropout=dropout,
                activation=activation
            )
            
            # Initialize model
            ode_func = ODEFunc(
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                activation=config.activation
            )
            model = NeuralODE(ode_func)
            
            # Initialize trainer
            trainer = NeuralODETrainer(model, config)
            
            # Prepare data
            y0 = torch.tensor(self.data[0]).unsqueeze(0).unsqueeze(-1)
            t = torch.tensor(self.time_points)
            y_target = torch.tensor(self.data)
            
            # Train model
            history = trainer.train(y0, t, y_target)
            
            # Make predictions
            predictions, _ = model.predict(y0, t)
            y_pred = predictions.squeeze().detach().numpy()
            
            # Store results
            self.neural_ode_results = {
                'predictions': y_pred,
                'final_loss': history['final_loss'],
                'num_params': sum(p.numel() for p in model.parameters())
            }
            
            st.success("Neural ODE training completed successfully!")
            
        except Exception as e:
            st.error(f"Error in Neural ODE training: {e}")


def main():
    """Main function to run the Streamlit app."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
