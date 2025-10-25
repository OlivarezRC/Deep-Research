"""
Analytics module for BSP AI Assistant
Provides chart generation and AI-powered insights
"""

from analytics.chart_generator import ChartGenerator
from analytics.insight_generator import InsightGenerator
from analytics.analytics_handler import AnalyticsHandler, handle_analytics_command

__all__ = [
    'ChartGenerator',
    'InsightGenerator',
    'AnalyticsHandler',
    'handle_analytics_command'
]

__version__ = '1.0.0'