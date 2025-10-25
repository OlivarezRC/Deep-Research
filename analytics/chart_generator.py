"""
Chart generation module for BSP AI Assistant
Handles creation of various chart types from data using Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import pandas as pd
from utils.utils import get_logger

logger = get_logger()


class ChartGenerator:
    """Generate interactive Plotly charts from data"""
    
    SUPPORTED_CHART_TYPES = [
        "line", "bar", "scatter", "pie", "histogram", 
        "box", "heatmap", "area", "funnel", "waterfall"
    ]
    
    def __init__(self):
        self.default_layout = {
            "template": "plotly_white",
            "font": {"family": "Arial, sans-serif", "size": 12},
            "margin": {"l": 50, "r": 50, "t": 50, "b": 50},
        }
    
    def create_chart(
        self, 
        chart_type: str,
        data: Dict[str, Any],
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create a chart based on type and data
        
        Args:
            chart_type: Type of chart to create
            data: Dictionary containing chart data
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            **kwargs: Additional chart-specific parameters
            
        Returns:
            Plotly Figure object
        """
        chart_type = chart_type.lower()
        
        if chart_type not in self.SUPPORTED_CHART_TYPES:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        try:
            # Convert data to DataFrame if needed
            df = self._prepare_dataframe(data)
            
            # Generate chart based on type
            if chart_type == "line":
                fig = self._create_line_chart(df, **kwargs)
            elif chart_type == "bar":
                fig = self._create_bar_chart(df, **kwargs)
            elif chart_type == "scatter":
                fig = self._create_scatter_chart(df, **kwargs)
            elif chart_type == "pie":
                fig = self._create_pie_chart(df, **kwargs)
            elif chart_type == "histogram":
                fig = self._create_histogram(df, **kwargs)
            elif chart_type == "box":
                fig = self._create_box_chart(df, **kwargs)
            elif chart_type == "heatmap":
                fig = self._create_heatmap(df, **kwargs)
            elif chart_type == "area":
                fig = self._create_area_chart(df, **kwargs)
            elif chart_type == "funnel":
                fig = self._create_funnel_chart(df, **kwargs)
            elif chart_type == "waterfall":
                fig = self._create_waterfall_chart(df, **kwargs)
            
            # Apply common layout settings
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                **self.default_layout
            )
            
            logger.info(f"Created {chart_type} chart successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            raise
    
    def _prepare_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Convert various data formats to DataFrame"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise ValueError("Data must be dict, list, or DataFrame")
    
    def _create_line_chart(self, df: pd.DataFrame, x: str = None, y: str = None, **kwargs) -> go.Figure:
        """Create line chart"""
        x = x or df.columns[0]
        y = y or df.columns[1]
        
        fig = px.line(df, x=x, y=y, **kwargs)
        return fig
    
    def _create_bar_chart(self, df: pd.DataFrame, x: str = None, y: str = None, **kwargs) -> go.Figure:
        """Create bar chart"""
        x = x or df.columns[0]
        y = y or df.columns[1]
        
        orientation = kwargs.pop('orientation', 'v')
        fig = px.bar(df, x=x, y=y, orientation=orientation, **kwargs)
        return fig
    
    def _create_scatter_chart(self, df: pd.DataFrame, x: str = None, y: str = None, **kwargs) -> go.Figure:
        """Create scatter plot"""
        x = x or df.columns[0]
        y = y or df.columns[1]
        
        fig = px.scatter(df, x=x, y=y, **kwargs)
        return fig
    
    def _create_pie_chart(self, df: pd.DataFrame, values: str = None, names: str = None, **kwargs) -> go.Figure:
        """Create pie chart"""
        values = values or df.columns[1]
        names = names or df.columns[0]
        
        fig = px.pie(df, values=values, names=names, **kwargs)
        return fig
    
    def _create_histogram(self, df: pd.DataFrame, x: str = None, **kwargs) -> go.Figure:
        """Create histogram"""
        x = x or df.columns[0]
        
        fig = px.histogram(df, x=x, **kwargs)
        return fig
    
    def _create_box_chart(self, df: pd.DataFrame, x: str = None, y: str = None, **kwargs) -> go.Figure:
        """Create box plot"""
        y = y or df.columns[0]
        
        fig = px.box(df, x=x, y=y, **kwargs)
        return fig
    
    def _create_heatmap(self, df: pd.DataFrame, **kwargs) -> go.Figure:
        """Create heatmap"""
        fig = px.imshow(df, **kwargs)
        return fig
    
    def _create_area_chart(self, df: pd.DataFrame, x: str = None, y: str = None, **kwargs) -> go.Figure:
        """Create area chart"""
        x = x or df.columns[0]
        y = y or df.columns[1]
        
        fig = px.area(df, x=x, y=y, **kwargs)
        return fig
    
    def _create_funnel_chart(self, df: pd.DataFrame, x: str = None, y: str = None, **kwargs) -> go.Figure:
        """Create funnel chart"""
        x = x or df.columns[1]
        y = y or df.columns[0]
        
        fig = px.funnel(df, x=x, y=y, **kwargs)
        return fig
    
    def _create_waterfall_chart(self, df: pd.DataFrame, x: str = None, y: str = None, **kwargs) -> go.Figure:
        """Create waterfall chart"""
        x = x or df.columns[0]
        y = y or df.columns[1]
        
        fig = go.Figure(go.Waterfall(
            x=df[x].tolist(),
            y=df[y].tolist(),
            **kwargs
        ))
        return fig
    
    def create_multi_series_chart(
        self,
        chart_type: str,
        data: Dict[str, List[Any]],
        series_names: List[str],
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create chart with multiple data series
        
        Args:
            chart_type: Type of chart
            data: Dictionary with x values and multiple y series
            series_names: Names for each data series
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        x_values = data.get('x', [])
        
        for series_name in series_names:
            y_values = data.get(series_name, [])
            
            if chart_type == "line":
                fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=series_name))
            elif chart_type == "bar":
                fig.add_trace(go.Bar(x=x_values, y=y_values, name=series_name))
            elif chart_type == "area":
                fig.add_trace(go.Scatter(x=x_values, y=y_values, fill='tozeroy', name=series_name))
        
        fig.update_layout(title=title, **self.default_layout)
        return fig