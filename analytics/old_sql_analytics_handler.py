
"""
Analytics handler for BSP AI Assistant
Orchestrates chart generation and insight generation
Now with LLM-powered chart recommendations and memory support
+ Generic SQL agent (DuckDB) for NL/SQL filtering & aggregation
"""

import time
import json
import chainlit as cl
import pandas as pd
from typing import Dict, Any, Optional, List

from analytics.chart_generator import ChartGenerator
from analytics.insight_generator import InsightGenerator
from analytics.sql_agent import SQLAgent

from utils.utils import get_logger, append_message

logger = get_logger()


class AnalyticsHandler:
    """Main handler for analytics features"""

    def __init__(self):
        self.chart_gen = ChartGenerator()
        self.insight_gen = InsightGenerator()
        self.sql_agent = SQLAgent()  # NEW

    async def process_analytics_request(
        self,
        data: Any,
        user_prompt: str = "",
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process analytics request with intelligent chart selection + SQL Agent
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
                "columns": [str(c) for c in df.columns.tolist()],
                "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()}
            })

            # ---- SQL Agent: register & transform via LLM SQL (fallback to heuristic) ----
            reg_meta = self.sql_agent.register_dataframe("data", df)

            try:
                df_sql, fa_meta = await self.sql_agent.query_via_llm("data", user_prompt)
            except Exception as _:
                # LLM path unavailable → fallback to heuristic SQL/NL
                prefer_metric = "Rate" if "Rate" in df.columns else None
                df_sql, fa_meta = self.sql_agent.query("data", user_prompt, prefer_metric)

            # Use the SQL-transformed dataframe downstream
            df = df_sql
            cl.user_session.set("analytics_sql_meta", {"registry": reg_meta, **fa_meta})

            results: Dict[str, Any] = {}

            # Check explicit chart requests
            explicit_charts = self._extract_explicit_chart_requests(user_prompt, df)

            if explicit_charts:
                logger.info(f"Using {len(explicit_charts)} explicit chart request(s)")
                charts_result = await self._generate_multiple_charts(df, explicit_charts)
                results["charts"] = charts_result
                results["chart_source"] = "explicit"
            else:
                logger.info("No explicit charts requested, using LLM recommendations")
                chart_specs = await self._get_chart_recommendations(df, user_prompt)
                if chart_specs:
                    charts_result = await self._generate_multiple_charts(df, chart_specs)
                    results["charts"] = charts_result
                    results["chart_source"] = "llm"

            # Insights on the transformed slice
            insights_result = await self._generate_insights(df, user_prompt, params)
            results["insights"] = insights_result

            return results

        except Exception as e:
            logger.error(f"Error processing analytics request: {e}", exc_info=True)
            return {"error": str(e)}

    def _extract_explicit_chart_requests(
        self, 
        user_prompt: str, 
        df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Extract explicit chart requests from user prompt
        """
        if not user_prompt:
            return []

        prompt_lower = user_prompt.lower()
        explicit_charts = []

        chart_patterns = {
            "line": ["line chart", "line graph", "trend line", "time series"],
            "bar": ["bar chart", "bar graph", "column chart", "vertical bar"],
            "scatter": ["scatter plot", "scatter chart", "scatterplot", "point plot"],
            "pie": ["pie chart", "pie graph", "donut chart"],
            "histogram": ["histogram", "distribution chart", "frequency chart"],
            "box": ["box plot", "boxplot", "box and whisker"],
            "heatmap": ["heatmap", "heat map", "correlation matrix"],
            "area": ["area chart", "area graph", "filled line"],
            "funnel": ["funnel chart", "funnel graph"],
            "waterfall": ["waterfall chart", "waterfall graph"]
        }

        requested_types = []
        for chart_type, patterns in chart_patterns.items():
            for pattern in patterns:
                if pattern in prompt_lower:
                    requested_types.append(chart_type)
                    break

        requested_types = list(dict.fromkeys(requested_types))
        if not requested_types:
            return []

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # safe string casting for column-name checks
        mentioned_columns = [col for col in df.columns if str(col).lower() in prompt_lower]

        for chart_type in requested_types:
            spec = self._build_chart_spec_from_prompt(
                chart_type, df, user_prompt, numeric_cols, categorical_cols, mentioned_columns
            )
            if spec:
                explicit_charts.append(spec)

        return explicit_charts

    def _build_chart_spec_from_prompt(
        self,
        chart_type: str,
        df: pd.DataFrame,
        prompt: str,
        numeric_cols: List[str],
        categorical_cols: List[str],
        mentioned_columns: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Build chart specification based on chart type and available columns"""

        spec = {
            "type": chart_type,
            "title": f"{chart_type.capitalize()} Chart",
            "reason": f"User explicitly requested {chart_type} chart"
        }

        if chart_type in ["line", "bar", "scatter", "area"]:
            if len(mentioned_columns) >= 2:
                spec["x"] = mentioned_columns[0]
                spec["y"] = mentioned_columns[1]
            elif len(mentioned_columns) == 1 and mentioned_columns[0] in numeric_cols:
                spec["x"] = df.columns[0]
                spec["y"] = mentioned_columns[0]
            elif categorical_cols and numeric_cols:
                spec["x"] = categorical_cols[0] if categorical_cols else df.columns[0]
                spec["y"] = numeric_cols[0]
            else:
                spec["x"] = df.columns[0] if len(df.columns) > 0 else None
                spec["y"] = df.columns[1] if len(df.columns) > 1 else None

        elif chart_type == "pie":
            if len(mentioned_columns) >= 2:
                spec["names"] = mentioned_columns[0]
                spec["values"] = mentioned_columns[1]
            elif categorical_cols and numeric_cols:
                spec["names"] = categorical_cols[0]
                spec["values"] = numeric_cols[0]
            else:
                spec["names"] = df.columns[0] if len(df.columns) > 0 else None
                spec["values"] = df.columns[1] if len(df.columns) > 1 else None

        elif chart_type == "histogram":
            if mentioned_columns and mentioned_columns[0] in numeric_cols:
                spec["x"] = mentioned_columns[0]
            elif numeric_cols:
                spec["x"] = numeric_cols[0]
            else:
                spec["x"] = df.columns[0] if len(df.columns) > 0 else None

        elif chart_type == "box":
            if len(mentioned_columns) >= 1 and mentioned_columns[0] in numeric_cols:
                spec["y"] = mentioned_columns[0]
                if len(categorical_cols) > 0:
                    spec["x"] = categorical_cols[0]
            elif numeric_cols:
                spec["y"] = numeric_cols[0]
                if categorical_cols:
                    spec["x"] = categorical_cols[0]

        if spec.get("x") and spec.get("y"):
            spec["title"] = f"{spec['y']} by {spec['x']}"
        elif spec.get("y"):
            spec["title"] = f"Distribution of {spec['y']}"
        elif spec.get("x"):
            spec["title"] = f"Analysis of {spec['x']}"

        return spec if (spec.get("x") or spec.get("y") or spec.get("names")) else None

    async def _get_chart_recommendations(
        self,
        df: pd.DataFrame,
        user_prompt: str
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to recommend appropriate charts based on data and user intent
        """
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            data_summary = {
                "columns": [str(c) for c in df.columns.tolist()],
                "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
                "shape": df.shape,
                "sample": df.head(3).to_dict() if len(df) > 0 else {},
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols
            }

            prompt = f"""You are a data visualization expert. Based on the data and user request, recommend appropriate charts.

Data Summary:
- Shape: {data_summary['shape'][0]} rows, {data_summary['shape'][1]} columns
- Columns: {', '.join(data_summary['columns'][:10])}{'...' if len(data_summary['columns']) > 10 else ''}
- Numeric columns: {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}
- Categorical columns: {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}

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
                from utils.foundry import chat_agent
                original_mode = cl.user_session.get("analytics_mode", False)
                cl.user_session.set("analytics_mode", True)
                response = await chat_agent(prompt)
                cl.user_session.set("analytics_mode", original_mode)
                if not response or response.strip() == "":
                    logger.warning("Empty response from chart recommendation")
                    return self._get_default_charts(df)
                try:
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
                    return self._get_default_charts(df)
            return self._get_default_charts(df)
        except Exception as e:
            logger.error(f"Error getting chart recommendations: {e}")
            return self._get_default_charts(df)

    def _get_default_charts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate default chart specifications based on data types"""
        charts = []
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if categorical_cols and numeric_cols:
            charts.append({
                "type": "bar",
                "x": categorical_cols[0],
                "y": numeric_cols[0],
                "title": f"{numeric_cols[0]} by {categorical_cols[0]}",
                "reason": "Comparing values across categories"
            })

        if len(numeric_cols) >= 2:
            charts.append({
                "type": "line",
                "x": df.columns[0],
                "y": numeric_cols[0],
                "title": f"{numeric_cols[0]} Trend",
                "reason": "Showing trend over time/sequence"
            })

        if numeric_cols:
            charts.append({
                "type": "histogram",
                "x": numeric_cols[0],
                "title": f"Distribution of {numeric_cols[0]}",
                "reason": "Understanding data distribution"
            })

        return charts[:3]

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
            return {"success": True, "charts": generated_charts, "count": len(generated_charts)}
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            return {"success": False, "error": str(e)}

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
                try:
                    parsed = json.loads(data)
                    return pd.DataFrame(parsed)
                except Exception:
                    pass
                if data.endswith('.csv'):
                    return pd.read_csv(data)
                elif data.endswith(('.xlsx', '.xls')):
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
                data=df, context=context, focus_areas=focus_areas
            )
            return {"success": True, "data": insights}
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {"success": False, "error": str(e)}

    async def display_analytics(
        self,
        results: Dict[str, Any],
        show_data_preview: bool = True
    ):
        """Display analytics results in Chainlit UI"""
        try:
            # Charts
            if "charts" in results and results["charts"].get("success"):
                charts_data = results["charts"].get("charts", [])
                await cl.Message(content=f"📊 **Generated {len(charts_data)} Visualizations**").send()
                for i, chart_info in enumerate(charts_data, 1):
                    fig = chart_info.get("figure")
                    title = chart_info.get("title", f"Chart {i}")
                    reason = chart_info.get("reason", "")
                    if fig:
                        plotly_element = cl.Plotly(name=f"chart_{i}", figure=fig, display="inline")
                        desc = f"**{title}**"
                        if reason:
                            desc += f"\n\n_{reason}_"
                        await cl.Message(content=desc, elements=[plotly_element]).send()

            # Insights + stats
            if "insights" in results and results["insights"].get("success"):
                insights_data = results["insights"]["data"]
                if "statistics" in insights_data and insights_data["statistics"]:
                    stats_md = self._format_statistics_table(insights_data["statistics"])
                    await cl.Message(content=stats_md, author="Statistics").send()
                insights_md = self._format_insights_markdown(insights_data, include_stats=False)
                await cl.Message(content=insights_md, author="Analytics").send()

            # Show executed SQL / meta
            fa_meta = cl.user_session.get("analytics_sql_meta")
            if fa_meta:
                sql_txt = fa_meta.get("sql", "-- no sql --")
                meta_det = fa_meta.get("detected", {})
                await cl.Message(
                    content=f"🧮 **SQL used**\n```sql\n{sql_txt}\n```\n\n**Meta:** `{json.dumps(meta_det, default=str)}`"
                ).send()

            # Errors
            if "error" in results:
                await cl.Message(content=f"⚠️ Error: {results['error']}", author="Error").send()

        except Exception as e:
            logger.error(f"Error displaying analytics: {e}")
            await cl.Message(content=f"Error displaying results: {str(e)}", author="Error").send()

    def _format_statistics_table(self, statistics: Dict[str, Any]) -> str:
        """Format statistics as a markdown table"""
        if not statistics:
            return "## 📊 Statistical Summary\n\n_No numeric columns found in the data._\n"

        md = ["## 📊 Statistical Summary\n"]
        try:
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
            md.append("### Distribution Metrics")
            md.append("")
            md.append("| Column | Range | IQR | CV (%) | Skewness |")
            md.append("|--------|-------|-----|--------|----------|")
            for col, stats in statistics.items():
                range_val = stats.get('range', 0)
                iqr = stats.get('iqr', 0)
                cv = stats.get('cv', 0)
                skewness = stats.get('skewness', 0)
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
        except Exception as e:
            logger.error(f"Error formatting statistics table: {e}", exc_info=True)
            md.append("\n_Error formatting statistics. Raw data available on request._\n")
        return "\n".join(md)

    def _format_insights_markdown(self, insights: Dict[str, Any], include_stats: bool = True) -> str:
        """Format insights as markdown with optional statistics"""
        md = ["## 📈 Data Insights\n"]
        if "key_findings" in insights:
            md.append(f"### Key Findings\n{insights['key_findings']}\n")
        if "insights" in insights and insights["insights"]:
            md.append("### Detailed Analysis")
            for i, insight in enumerate(insights["insights"], 1):
                md.append(f"{i}. {insight}")
            md.append("")
        if "recommendations" in insights and insights["recommendations"]:
            md.append("### Recommendations")
            for i, rec in enumerate(insights["recommendations"], 1):
                md.append(f"{i}. {rec}")
            md.append("")
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
    """
    try:
        cl.user_session.set("analytics_mode", True)
        user_prompt = message_content[len("/analytics"):].strip()

        data = None
        if elements:
            for element in elements:
                if hasattr(element, 'path'):
                    data = element.path
                    break

        if not data and user_prompt:
            try:
                if '{' in user_prompt and '}' in user_prompt:
                    json_start = user_prompt.index('{')
                    json_end = user_prompt.rindex('}') + 1
                    potential_json = user_prompt[json_start:json_end]
                    data = json.loads(potential_json)
                    user_prompt = user_prompt[:json_start].strip()
            except Exception:
                pass

        if not data:
            await cl.Message(
                content="📁 Please attach a CSV or Excel file, or provide data as JSON.\n\nExample: `/analytics Show me sales trends` (with file attached)"
            ).send()
            cl.user_session.set("analytics_mode", False)
            return

        append_message("user", f"[Analytics Request] {user_prompt if user_prompt else 'Analyze this data'}", elements)

        await cl.Message(content="🔄 Analyzing your data and generating visualizations...").send()

        results = await analytics_handler.process_analytics_request(
            data=data,
            user_prompt=user_prompt
        )

        await analytics_handler.display_analytics(results)

        summary = f"Generated {results.get('charts', {}).get('count', 0)} charts and analytical insights for the data."
        append_message("assistant", summary, [])

        await cl.Message(
            content="💡 You can ask follow-up questions about this data, and I'll remember the context!"
        ).send()

    except Exception as e:
        logger.error(f"Error handling analytics command: {e}")
        await cl.Message(
            content=f"Error processing analytics: {str(e)}",
            author="Error"
        ).send()
        cl.user_session.set("analytics_mode", False)
