"""
Analytics handler for BSP AI Assistant
Orchestrates chart generation and insight generation
Now with LLM-powered chart recommendations and memory support
"""

import chainlit as cl
import pandas as pd
from typing import Dict, Any, Optional, List
from analytics.chart_generator import ChartGenerator
from analytics.insight_generator import InsightGenerator
from utils.utils import get_logger, append_message
import json
import time
logger = get_logger()


class AnalyticsHandler:
    """Main handler for analytics features"""
    
    def __init__(self):
        self.chart_gen = ChartGenerator()
        self.insight_gen = InsightGenerator()
    
    async def process_analytics_request(
        self,
        data: Any,
        user_prompt: str = "",
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process analytics request with LLM-powered chart recommendations
        
        Args:
            data: Input data (DataFrame, dict, or file path)
            user_prompt: User's prompt describing what they want to analyze
            params: Additional parameters for processing
            
        Returns:
            Dictionary containing results
        """
        params = params or {}
        
        try:
            # Convert data to DataFrame
            df = self._prepare_data(data)
            
            if df is None or df.empty:
                return {"error": "No valid data provided"}
            
            # Store data info in session for memory
            cl.user_session.set("analytics_data", {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict()
            })
            
            results = {}
            
            # Get LLM recommendations for charts
            chart_specs = await self._get_chart_recommendations(df, user_prompt)
            
            # Generate recommended charts
            if chart_specs:
                charts_result = await self._generate_multiple_charts(df, chart_specs)
                results["charts"] = charts_result
            
            # Generate insights
            insights_result = await self._generate_insights(df, user_prompt, params)
            results["insights"] = insights_result
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing analytics request: {e}")
            return {"error": str(e)}
    
    async def _get_chart_recommendations(
        self,
        df: pd.DataFrame,
        user_prompt: str
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to recommend appropriate charts based on data and user intent
        
        Args:
            df: DataFrame to analyze
            user_prompt: User's description of what they want to see
            
        Returns:
            List of chart specifications
        """
        try:
            # Prepare data summary
            data_summary = {
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "shape": df.shape,
                "sample": df.head(3).to_dict(),
                "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
            }
            
            prompt = f"""User query: {query} 
            You are a data visualization expert. Based on the data and user request, recommend appropriate charts.

Data Summary:
- Shape: {data_summary['shape'][0]} rows, {data_summary['shape'][1]} columns
- Columns: {', '.join(data_summary['columns'])}
- Numeric columns: {', '.join(data_summary['numeric_columns'])}
- Categorical columns: {', '.join(data_summary['categorical_columns'])}

User Request: "{user_prompt if user_prompt else 'Show me insights about this data'}"

Available chart types: line, bar, scatter, pie, histogram, box, heatmap, area, funnel, waterfall

Recommend 2-4 charts that would best visualize this data based on the user's request.
For each chart, specify:
1. Chart type
2. Which columns to use (x and y axis)
3. Chart title
4. Brief reason for recommendation

Respond in JSON format:
{{
  "charts": [
    {{
      "type": "bar",
      "x": "column_name",
      "y": "column_name",
      "title": "Chart Title",
      "reason": "why this chart is useful"
    }}
  ]
}}"""
            chat_settings = cl.user_session.get("chat_settings", {})
            provider = chat_settings.get("model_provider", "litellm")
            cl.user_session.set("start_time", time.time())
            if provider == "foundry":
                # Use Foundry agent for recommendations
                from utils.foundry import chat_agent
                
                # Store current mode and switch to analytics mode
                original_mode = cl.user_session.get("analytics_mode", False)
                cl.user_session.set("analytics_mode", True)
                
                response = await chat_agent(prompt)
                
                # Restore original mode
                cl.user_session.set("analytics_mode", original_mode)
                
                # Parse response
                try:
                    # Clean markdown code blocks
                    clean_response = response.strip()
                    if clean_response.startswith("```json"):
                        clean_response = clean_response[7:]
                    if clean_response.startswith("```"):
                        clean_response = clean_response[3:]
                    if clean_response.endswith("```"):
                        clean_response = clean_response[:-3]
                    clean_response = clean_response.strip()
                    
                    chart_data = json.loads(clean_response)
                    return chart_data.get("charts", [])
                except json.JSONDecodeError:
                    logger.warning("Could not parse chart recommendations as JSON")
                    # Fallback to default charts
                    return self._get_default_charts(df)
                    
        except Exception as e:
            logger.error(f"Error getting chart recommendations: {e}")
            return self._get_default_charts(df)
    
    def _get_default_charts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate default chart specifications based on data types"""
        charts = []
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # If we have categorical and numeric columns, create a bar chart
        if categorical_cols and numeric_cols:
            charts.append({
                "type": "bar",
                "x": categorical_cols[0],
                "y": numeric_cols[0],
                "title": f"{numeric_cols[0]} by {categorical_cols[0]}",
                "reason": "Comparing values across categories"
            })
        
        # If we have multiple numeric columns, create a line chart
        if len(numeric_cols) >= 2:
            charts.append({
                "type": "line",
                "x": df.columns[0],
                "y": numeric_cols[0],
                "title": f"{numeric_cols[0]} Trend",
                "reason": "Showing trend over time/sequence"
            })
        
        # Add histogram for distribution
        if numeric_cols:
            charts.append({
                "type": "histogram",
                "x": numeric_cols[0],
                "title": f"Distribution of {numeric_cols[0]}",
                "reason": "Understanding data distribution"
            })
        
        return charts[:3]  # Limit to 3 charts
    
    async def _generate_multiple_charts(
        self,
        df: pd.DataFrame,
        chart_specs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate multiple charts based on specifications"""
        try:
            generated_charts = []
            
            for spec in chart_specs:
                try:
                    fig = self.chart_gen.create_chart(
                        chart_type=spec.get("type", "bar"),
                        data=df,
                        title=spec.get("title", "Data Visualization"),
                        x=spec.get("x"),
                        y=spec.get("y"),
                        x_label=spec.get("x_label"),
                        y_label=spec.get("y_label")
                    )
                    
                    generated_charts.append({
                        "figure": fig,
                        "type": spec.get("type"),
                        "title": spec.get("title"),
                        "reason": spec.get("reason", "")
                    })
                    
                except Exception as chart_error:
                    logger.error(f"Error generating chart {spec.get('type')}: {chart_error}")
                    continue
            
            return {
                "success": True,
                "charts": generated_charts,
                "count": len(generated_charts)
            }
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _prepare_data(self, data: Any) -> Optional[pd.DataFrame]:
        """Convert various data formats to DataFrame"""
        try:
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, dict):
                return pd.DataFrame(data)
            elif isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, str):
                # Try to parse as JSON
                try:
                    parsed = json.loads(data)
                    return pd.DataFrame(parsed)
                except:
                    pass
                
                # Try to read as file
                if data.endswith('.csv'):
                    return pd.read_csv(data)
                elif data.endswith('.xlsx') or data.endswith('.xls'):
                    return pd.read_excel(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None
    
    async def _generate_insights(
        self,
        df: pd.DataFrame,
        user_prompt: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate insights from data"""
        try:
            context = f"User request: {user_prompt}" if user_prompt else None
            focus_areas = params.get("focus_areas")
            
            insights = await self.insight_gen.generate_insights(
                data=df,
                context=context,
                focus_areas=focus_areas
            )
            
            return {
                "success": True,
                "data": insights
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def display_analytics(
        self,
        results: Dict[str, Any],
        show_data_preview: bool = True
    ):
        """Display analytics results in Chainlit UI"""
        try:
            # Display multiple charts
            if "charts" in results and results["charts"].get("success"):
                charts_data = results["charts"].get("charts", [])
                
                await cl.Message(
                    content=f"📊 **Generated {len(charts_data)} Visualizations**"
                ).send()
                
                for i, chart_info in enumerate(charts_data, 1):
                    fig = chart_info.get("figure")
                    title = chart_info.get("title", f"Chart {i}")
                    reason = chart_info.get("reason", "")
                    
                    if fig:
                        plotly_element = cl.Plotly(
                            name=f"chart_{i}",
                            figure=fig,
                            display="inline"
                        )
                        
                        chart_description = f"**{title}**"
                        if reason:
                            chart_description += f"\n\n_{reason}_"
                        
                        await cl.Message(
                            content=chart_description,
                            elements=[plotly_element]
                        ).send()
            
           # Display insights with enhanced statistics
            if "insights" in results and results["insights"].get("success"):
                insights_data = results["insights"]["data"]
                
                # Send statistics as separate message for better visibility
                if "statistics" in insights_data and insights_data["statistics"]:
                    stats_md = self._format_statistics_table(insights_data["statistics"])
                    await cl.Message(
                        content=stats_md,
                        author="Statistics"
                    ).send()
                
                # Format and send insights (without stats, since we sent them separately)
                insights_md = self._format_insights_markdown(insights_data, include_stats=False)
                
                await cl.Message(
                    content=insights_md,
                    author="Analytics"
                ).send()
            
            # Display errors if any
            if "error" in results:
                await cl.Message(
                    content=f"⚠️ Error: {results['error']}",
                    author="Error"
                ).send()
                
        except Exception as e:
            logger.error(f"Error displaying analytics: {e}")
            await cl.Message(
                content=f"Error displaying results: {str(e)}",
                author="Error"
            ).send()
    
    def _format_statistics_table(self, statistics: Dict[str, Any]) -> str:
        """Format comprehensive statistics as a markdown table"""
        md = ["## 📊 Statistical Summary\n"]
        
        # Main statistics table
        md.append("### Descriptive Statistics")
        md.append("")
        md.append("| Column | Count | Missing | Mean | Median | Std Dev | Min | Q1 | Q3 | Max |")
        md.append("|--------|-------|---------|------|--------|---------|-----|----|----|-----|")
        
        for col, stats in statistics.items():
            count = stats.get('count', 0)
            missing = stats.get('missing', 0)
            mean = stats.get('mean', 0)
            median = stats.get('median', 0)
            std = stats.get('std', 0)
            min_val = stats.get('min', 0)
            q1 = stats.get('q25', 0)
            q3 = stats.get('q75', 0)
            max_val = stats.get('max', 0)
            
            md.append(
                f"| **{col}** | {count} | {missing} | {mean:.2f} | {median:.2f} | {std:.2f} | "
                f"{min_val:.2f} | {q1:.2f} | {q3:.2f} | {max_val:.2f} |"
            )
        
        md.append("")
        
        # Additional metrics table
        md.append("### Distribution Metrics")
        md.append("")
        md.append("| Column | Range | IQR | CV (%) | Skewness |")
        md.append("|--------|-------|-----|--------|----------|")
        
        for col, stats in statistics.items():
            range_val = stats.get('range', 0)
            iqr = stats.get('iqr', 0)
            cv = stats.get('cv', 0)
            skewness = stats.get('skewness', 0)
            
            # Interpret skewness
            if abs(skewness) < 0.5:
                skew_label = "Symmetric"
            elif skewness > 0:
                skew_label = "Right-skewed"
            else:
                skew_label = "Left-skewed"
            
            md.append(
                f"| **{col}** | {range_val:.2f} | {iqr:.2f} | {cv:.2f} | "
                f"{skewness:.2f} ({skew_label}) |"
            )
        
        md.append("")
        md.append("_**Legend:**_")
        md.append("- _Q1/Q3: 25th/75th percentiles_")
        md.append("- _IQR: Interquartile Range (Q3 - Q1)_")
        md.append("- _CV: Coefficient of Variation (relative variability)_")
        md.append("- _Skewness: Distribution asymmetry (0 = symmetric)_")
        md.append("")
        
        return "\n".join(md)
    
    def _format_insights_markdown(self, insights: Dict[str, Any], include_stats: bool = True) -> str:
        """Format insights as markdown with optional statistics"""
        md = ["## 📈 Data Insights\n"]
        
        # Key findings
        if "key_findings" in insights:
            md.append(f"### Key Findings\n{insights['key_findings']}\n")
        
        # Detailed insights
        if "insights" in insights and insights["insights"]:
            md.append("### Detailed Analysis")
            for i, insight in enumerate(insights["insights"], 1):
                md.append(f"{i}. {insight}")
            md.append("")
        
        # Recommendations
        if "recommendations" in insights and insights["recommendations"]:
            md.append("### Recommendations")
            for i, rec in enumerate(insights["recommendations"], 1):
                md.append(f"{i}. {rec}")
            md.append("")
        
        # Include statistics only if requested (for backward compatibility)
        if include_stats and "statistics" in insights and insights["statistics"]:
            md.append(self._format_statistics_table(insights["statistics"]))
        
        return "\n".join(md)


# Global instance
analytics_handler = AnalyticsHandler()


async def handle_analytics_command(message_content: str, elements: List = None):
    """
    Handle analytics commands from chat - always generates both charts and insights
    
    Command format:
    /analytics [your question or description]
    
    Args:
        message_content: The message content containing the command
        elements: Any file elements attached to the message
    """
    try:
        # Enable analytics mode for memory
        cl.user_session.set("analytics_mode", True)
        
        # Extract the user prompt (everything after /analytics)
        user_prompt = message_content[len("/analytics"):].strip()
        
        # Try to get data from uploaded files
        data = None
        if elements:
            for element in elements:
                if hasattr(element, 'path'):
                    data = element.path
                    break
        
        # If no file, check if data is provided as JSON in the prompt
        if not data and user_prompt:
            # Try to extract JSON from the prompt
            try:
                # Look for JSON-like structures
                if '{' in user_prompt and '}' in user_prompt:
                    json_start = user_prompt.index('{')
                    json_end = user_prompt.rindex('}') + 1
                    potential_json = user_prompt[json_start:json_end]
                    data = json.loads(potential_json)
                    # Remove JSON from prompt
                    user_prompt = user_prompt[:json_start].strip()
            except:
                pass
        
        if not data:
            await cl.Message(
                content="📁 Please attach a CSV or Excel file, or provide data as JSON.\n\nExample: `/analytics Show me sales trends` (with file attached)"
            ).send()
            # Disable analytics mode
            cl.user_session.set("analytics_mode", False)
            return
        
        # Add analytics request to message history for memory
        append_message("user", f"[Analytics Request] {user_prompt if user_prompt else 'Analyze this data'}", elements)
        
        # Process analytics request
        await cl.Message(content="🔄 Analyzing your data and generating visualizations...").send()
        
        results = await analytics_handler.process_analytics_request(
            data=data,
            user_prompt=user_prompt
        )
        
        # Display results
        await analytics_handler.display_analytics(results)
        
        # Add results to message history for memory
        summary = f"Generated {results.get('charts', {}).get('count', 0)} charts and analytical insights for the data."
        append_message("assistant", summary, [])
        
        # Keep analytics mode enabled for follow-up questions
        await cl.Message(
            content="💡 You can ask follow-up questions about this data, and I'll remember the context!"
        ).send()
        
    except Exception as e:
        logger.error(f"Error handling analytics command: {e}")
        await cl.Message(
            content=f"Error processing analytics: {str(e)}",
            author="Error"
        ).send()
        # Disable analytics mode on error
        cl.user_session.set("analytics_mode", False)