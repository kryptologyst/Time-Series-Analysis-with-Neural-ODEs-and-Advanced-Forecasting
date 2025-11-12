"""
Comprehensive Visualization Module for Time Series Analysis.

This module provides extensive visualization capabilities for time series data,
including trend analysis, seasonality decomposition, forecasting results,
anomaly detection visualization, and interactive plots using Matplotlib,
Seaborn, and Plotly.
"""

import logging
from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
import warnings

# Statistical visualization
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Some decomposition plots will be disabled.")

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    # General settings
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    style: str = "seaborn-v0_8"
    color_palette: str = "Set2"
    
    # Plotly settings
    plotly_template: str = "plotly_white"
    plotly_height: int = 600
    plotly_width: int = 1000
    
    # Color settings
    primary_color: str = "#1f77b4"
    secondary_color: str = "#ff7f0e"
    anomaly_color: str = "#d62728"
    forecast_color: str = "#2ca02c"
    
    # Font settings
    title_font_size: int = 16
    label_font_size: int = 12
    legend_font_size: int = 10


class TimeSeriesVisualizer:
    """
    Comprehensive time series visualization class.
    
    Provides methods for creating various types of plots including
    basic time series plots, decomposition, forecasting comparisons,
    anomaly detection results, and interactive visualizations.
    """
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self._setup_style()
    
    def _setup_style(self) -> None:
        """Setup matplotlib and seaborn styles."""
        plt.style.use(self.config.style)
        sns.set_palette(self.config.color_palette)
    
    def plot_time_series(
        self,
        data: np.ndarray,
        time_points: Optional[np.ndarray] = None,
        title: str = "Time Series Plot",
        xlabel: str = "Time",
        ylabel: str = "Value",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Create a basic time series plot.
        
        Args:
            data: Time series data
            time_points: Optional time points (defaults to indices)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        if time_points is None:
            time_points = np.arange(len(data))
        
        plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        plt.plot(time_points, data, color=self.config.primary_color, linewidth=1.5)
        plt.title(title, fontsize=self.config.title_font_size)
        plt.xlabel(xlabel, fontsize=self.config.label_font_size)
        plt.ylabel(ylabel, fontsize=self.config.label_font_size)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_multiple_series(
        self,
        data_dict: Dict[str, np.ndarray],
        time_points: Optional[np.ndarray] = None,
        title: str = "Multiple Time Series",
        xlabel: str = "Time",
        ylabel: str = "Value",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot multiple time series on the same plot.
        
        Args:
            data_dict: Dictionary of series names and data
            time_points: Optional time points
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        if time_points is None:
            time_points = np.arange(len(next(iter(data_dict.values()))))
        
        plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        colors = sns.color_palette(self.config.color_palette, len(data_dict))
        
        for i, (name, series) in enumerate(data_dict.items()):
            plt.plot(time_points, series, label=name, color=colors[i], linewidth=1.5)
        
        plt.title(title, fontsize=self.config.title_font_size)
        plt.xlabel(xlabel, fontsize=self.config.label_font_size)
        plt.ylabel(ylabel, fontsize=self.config.label_font_size)
        plt.legend(fontsize=self.config.legend_font_size)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_decomposition(
        self,
        data: np.ndarray,
        period: int = 12,
        model: str = "additive",
        title: str = "Time Series Decomposition",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot seasonal decomposition of time series.
        
        Args:
            data: Time series data
            period: Seasonal period
            model: Decomposition model ('additive' or 'multiplicative')
            title: Plot title
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available. Skipping decomposition plot.")
            return
        
        try:
            decomposition = seasonal_decompose(data, model=model, period=period)
            
            fig, axes = plt.subplots(4, 1, figsize=(self.config.figure_size[0], self.config.figure_size[1] * 1.5))
            
            # Original data
            axes[0].plot(data, color=self.config.primary_color)
            axes[0].set_title(f'{title} - Original')
            axes[0].set_ylabel('Original')
            axes[0].grid(True, alpha=0.3)
            
            # Trend
            axes[1].plot(decomposition.trend, color=self.config.secondary_color)
            axes[1].set_title('Trend')
            axes[1].set_ylabel('Trend')
            axes[1].grid(True, alpha=0.3)
            
            # Seasonal
            axes[2].plot(decomposition.seasonal, color=self.config.forecast_color)
            axes[2].set_title('Seasonal')
            axes[2].set_ylabel('Seasonal')
            axes[2].grid(True, alpha=0.3)
            
            # Residual
            axes[3].plot(decomposition.resid, color=self.config.anomaly_color)
            axes[3].set_title('Residual')
            axes[3].set_ylabel('Residual')
            axes[3].set_xlabel('Time')
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error in decomposition: {e}")
    
    def plot_forecasting_results(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        predictions: Dict[str, np.ndarray],
        title: str = "Forecasting Results",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot forecasting results comparison.
        
        Args:
            train_data: Training data
            test_data: Test data
            predictions: Dictionary of model predictions
            title: Plot title
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Plot training data
        train_end = len(train_data)
        plt.plot(range(train_end), train_data, label='Training Data', 
                color=self.config.primary_color, alpha=0.7)
        
        # Plot test data
        test_start = train_end
        test_end = test_start + len(test_data)
        plt.plot(range(test_start, test_end), test_data, label='Actual', 
                color=self.config.secondary_color, linewidth=2)
        
        # Plot predictions
        colors = sns.color_palette(self.config.color_palette, len(predictions))
        for i, (name, pred) in enumerate(predictions.items()):
            plt.plot(range(test_start, test_end), pred, label=f'{name} Prediction', 
                    color=colors[i], linestyle='--', linewidth=2)
        
        plt.title(title, fontsize=self.config.title_font_size)
        plt.xlabel('Time', fontsize=self.config.label_font_size)
        plt.ylabel('Value', fontsize=self.config.label_font_size)
        plt.legend(fontsize=self.config.legend_font_size)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_anomaly_detection(
        self,
        data: np.ndarray,
        anomaly_indices: np.ndarray,
        title: str = "Anomaly Detection Results",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot anomaly detection results.
        
        Args:
            data: Time series data
            anomaly_indices: Indices of detected anomalies
            title: Plot title
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Plot data
        plt.plot(data, color=self.config.primary_color, alpha=0.7, label='Data')
        
        # Highlight anomalies
        if len(anomaly_indices) > 0:
            plt.scatter(anomaly_indices, data[anomaly_indices], 
                       color=self.config.anomaly_color, s=50, 
                       label='Anomalies', zorder=5)
        
        plt.title(title, fontsize=self.config.title_font_size)
        plt.xlabel('Time', fontsize=self.config.label_font_size)
        plt.ylabel('Value', fontsize=self.config.label_font_size)
        plt.legend(fontsize=self.config.legend_font_size)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_correlation_matrix(
        self,
        data: np.ndarray,
        title: str = "Correlation Matrix",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot correlation matrix for multivariate time series.
        
        Args:
            data: Multivariate time series data
            title: Plot title
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        if data.ndim == 1:
            logger.warning("Data is univariate. Correlation matrix not applicable.")
            return
        
        correlation_matrix = np.corrcoef(data.T)
        
        plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title(title, fontsize=self.config.title_font_size)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_distribution(
        self,
        data: np.ndarray,
        title: str = "Data Distribution",
        bins: int = 50,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot data distribution histogram.
        
        Args:
            data: Time series data
            title: Plot title
            bins: Number of histogram bins
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        plt.hist(data, bins=bins, alpha=0.7, color=self.config.primary_color, edgecolor='black')
        plt.title(title, fontsize=self.config.title_font_size)
        plt.xlabel('Value', fontsize=self.config.label_font_size)
        plt.ylabel('Frequency', fontsize=self.config.label_font_size)
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        plt.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'+1 Std: {mean_val + std_val:.2f}')
        plt.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7, label=f'-1 Std: {mean_val - std_val:.2f}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_acf_pacf(
        self,
        data: np.ndarray,
        lags: int = 40,
        title: str = "ACF and PACF",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).
        
        Args:
            data: Time series data
            lags: Number of lags to plot
            title: Plot title
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available. Skipping ACF/PACF plot.")
            return
        
        try:
            from statsmodels.tsa.stattools import acf, pacf
            
            fig, axes = plt.subplots(2, 1, figsize=(self.config.figure_size[0], self.config.figure_size[1] * 1.2))
            
            # ACF
            acf_values = acf(data, nlags=lags)
            axes[0].bar(range(len(acf_values)), acf_values, alpha=0.7, color=self.config.primary_color)
            axes[0].set_title('Autocorrelation Function (ACF)')
            axes[0].set_xlabel('Lag')
            axes[0].set_ylabel('ACF')
            axes[0].grid(True, alpha=0.3)
            
            # PACF
            pacf_values = pacf(data, nlags=lags)
            axes[1].bar(range(len(pacf_values)), pacf_values, alpha=0.7, color=self.config.secondary_color)
            axes[1].set_title('Partial Autocorrelation Function (PACF)')
            axes[1].set_xlabel('Lag')
            axes[1].set_ylabel('PACF')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error in ACF/PACF plot: {e}")


class InteractiveVisualizer:
    """
    Interactive visualization class using Plotly.
    
    Provides methods for creating interactive plots that can be
    used in web applications or Jupyter notebooks.
    """
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
    
    def create_interactive_time_series(
        self,
        data: np.ndarray,
        time_points: Optional[np.ndarray] = None,
        title: str = "Interactive Time Series",
        xlabel: str = "Time",
        ylabel: str = "Value"
    ) -> go.Figure:
        """
        Create interactive time series plot.
        
        Args:
            data: Time series data
            time_points: Optional time points
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            Plotly figure object
        """
        if time_points is None:
            time_points = np.arange(len(data))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=data,
            mode='lines',
            name='Time Series',
            line=dict(color=self.config.primary_color, width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            template=self.config.plotly_template,
            height=self.config.plotly_height,
            width=self.config.plotly_width,
            hovermode='x unified'
        )
        
        return fig
    
    def create_interactive_forecasting(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        predictions: Dict[str, np.ndarray],
        title: str = "Interactive Forecasting Results"
    ) -> go.Figure:
        """
        Create interactive forecasting comparison plot.
        
        Args:
            train_data: Training data
            test_data: Test data
            predictions: Dictionary of model predictions
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Training data
        train_end = len(train_data)
        fig.add_trace(go.Scatter(
            x=list(range(train_end)),
            y=train_data,
            mode='lines',
            name='Training Data',
            line=dict(color=self.config.primary_color, width=2)
        ))
        
        # Test data
        test_start = train_end
        test_end = test_start + len(test_data)
        fig.add_trace(go.Scatter(
            x=list(range(test_start, test_end)),
            y=test_data,
            mode='lines',
            name='Actual',
            line=dict(color=self.config.secondary_color, width=2)
        ))
        
        # Predictions
        colors = px.colors.qualitative.Set2
        for i, (name, pred) in enumerate(predictions.items()):
            fig.add_trace(go.Scatter(
                x=list(range(test_start, test_end)),
                y=pred,
                mode='lines',
                name=f'{name} Prediction',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            template=self.config.plotly_template,
            height=self.config.plotly_height,
            width=self.config.plotly_width,
            hovermode='x unified'
        )
        
        return fig
    
    def create_interactive_anomaly_detection(
        self,
        data: np.ndarray,
        anomaly_indices: np.ndarray,
        title: str = "Interactive Anomaly Detection"
    ) -> go.Figure:
        """
        Create interactive anomaly detection plot.
        
        Args:
            data: Time series data
            anomaly_indices: Indices of detected anomalies
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Normal data
        normal_indices = np.setdiff1d(np.arange(len(data)), anomaly_indices)
        fig.add_trace(go.Scatter(
            x=normal_indices,
            y=data[normal_indices],
            mode='markers',
            name='Normal Data',
            marker=dict(color=self.config.primary_color, size=4)
        ))
        
        # Anomalies
        if len(anomaly_indices) > 0:
            fig.add_trace(go.Scatter(
                x=anomaly_indices,
                y=data[anomaly_indices],
                mode='markers',
                name='Anomalies',
                marker=dict(color=self.config.anomaly_color, size=8)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            template=self.config.plotly_template,
            height=self.config.plotly_height,
            width=self.config.plotly_width,
            hovermode='x unified'
        )
        
        return fig
    
    def create_model_comparison_dashboard(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "Model Comparison Dashboard"
    ) -> go.Figure:
        """
        Create interactive model comparison dashboard.
        
        Args:
            results: Dictionary of model results with metrics
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        models = list(results.keys())
        metrics = list(next(iter(results.values())).keys())
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics,
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = px.colors.qualitative.Set2
        
        for i, metric in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            values = [results[model][metric] for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric,
                    marker_color=colors[i % len(colors)],
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=title,
            template=self.config.plotly_template,
            height=self.config.plotly_height,
            width=self.config.plotly_width
        )
        
        return fig


def main():
    """Demonstrate visualization capabilities."""
    # Configuration
    config = VisualizationConfig()
    
    # Generate sample data
    from data_generator import TimeSeriesGenerator, DataConfig
    
    data_config = DataConfig(n_points=1000, noise_level=0.1)
    generator = TimeSeriesGenerator(data_config)
    t, data = generator.generate_complex_time_series(generator.generate_time_points())
    
    # Initialize visualizers
    static_viz = TimeSeriesVisualizer(config)
    interactive_viz = InteractiveVisualizer(config)
    
    # Create various plots
    logger.info("Creating visualization examples...")
    
    # Basic time series plot
    static_viz.plot_time_series(data, t, title="Sample Time Series")
    
    # Multiple series comparison
    data_dict = {
        'Original': data,
        'Smoothed': np.convolve(data, np.ones(10)/10, mode='same'),
        'Trend': np.linspace(data[0], data[-1], len(data))
    }
    static_viz.plot_multiple_series(data_dict, t, title="Multiple Series Comparison")
    
    # Decomposition
    static_viz.plot_decomposition(data, period=50, title="Seasonal Decomposition")
    
    # Distribution
    static_viz.plot_distribution(data, title="Data Distribution")
    
    # ACF/PACF
    static_viz.plot_acf_pacf(data, title="ACF and PACF")
    
    # Interactive plots
    interactive_fig = interactive_viz.create_interactive_time_series(
        data, t, title="Interactive Time Series"
    )
    interactive_fig.show()
    
    logger.info("Visualization examples completed successfully!")
    
    return static_viz, interactive_viz


if __name__ == "__main__":
    main()
