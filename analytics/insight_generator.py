"""
Insight generation module for BSP AI Assistant
Uses LLM to generate analytical insights from data
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from utils.utils import get_logger
import chainlit as cl
import json
import time
logger = get_logger()


async def get_llm_response(prompt: str) -> str:
    """
    Get LLM response for insights generation using Foundry agent
    
    Args:
        prompt: The prompt to send to LLM
        
    Returns:
        LLM response text
    """
    try:
        # Always use Foundry agent for analytics insights
        cl.user_session.set("start_time", time.time())
        chat_settings = cl.user_session.get("chat_settings", {})
        provider = chat_settings.get("model_provider", "litellm")
        if provider == "foundry":
            from utils.foundry import chat_agent
            
            # Store current analytics mode state
            analytics_mode = cl.user_session.get("analytics_mode", False)
            if not analytics_mode:
                cl.user_session.set("analytics_mode", True)
            
            response = await chat_agent(prompt)
            
            # Restore analytics mode state if it was off
            if not analytics_mode:
                cl.user_session.set("analytics_mode", False)
            
            return response
        
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        # Return a fallback JSON structure
        return _get_fallback_response()


def _get_fallback_response() -> str:
    """Return a fallback JSON response when LLM fails"""
    return json.dumps({
        "key_findings": "Statistical analysis completed successfully. The data has been analyzed for patterns, trends, and anomalies.",
        "insights": [
            "Descriptive statistics have been calculated for all numeric columns.",
            "Distribution metrics show the spread and variability of your data.",
            "Review the statistical tables above for detailed metrics."
        ],
        "recommendations": [
            "Examine the coefficient of variation (CV) to understand data consistency.",
            "Check skewness values to identify distribution patterns.",
            "Investigate any outliers identified in the analysis."
        ]
    })


class InsightGenerator:
    """Generate AI-powered insights from data"""
    
    def __init__(self):
        # Don't get LLM details in __init__ to avoid context issues
        pass
    
    async def generate_insights(
        self,
        data: pd.DataFrame,
        context: Optional[str] = None,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive insights from data
        
        Args:
            data: DataFrame containing the data to analyze
            context: Additional context about the data
            focus_areas: Specific areas to focus on (trends, outliers, etc.)
            
        Returns:
            Dictionary containing insights and recommendations
        """
        try:
            # Perform statistical analysis
            stats = self._calculate_statistics(data)
            
            # Detect patterns
            patterns = self._detect_patterns(data)
            
            # Generate LLM-powered insights
            llm_insights = await self._generate_llm_insights(data, stats, patterns, context, focus_areas)
            
            return {
                "statistics": stats,
                "patterns": patterns,
                "insights": llm_insights.get("insights", []),
                "recommendations": llm_insights.get("recommendations", []),
                "key_findings": llm_insights.get("key_findings", "")
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {"error": str(e)}
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistics for numeric columns"""
        stats = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                col_data = data[col].dropna()  # Remove NaN values
                
                if len(col_data) == 0:
                    continue
                
                # Calculate basic stats with None checks
                mean_val = float(col_data.mean()) if not pd.isna(col_data.mean()) else 0.0
                median_val = float(col_data.median()) if not pd.isna(col_data.median()) else 0.0
                std_val = float(col_data.std()) if not pd.isna(col_data.std()) else 0.0
                min_val = float(col_data.min()) if not pd.isna(col_data.min()) else 0.0
                max_val = float(col_data.max()) if not pd.isna(col_data.max()) else 0.0
                q25 = float(col_data.quantile(0.25)) if not pd.isna(col_data.quantile(0.25)) else 0.0
                q75 = float(col_data.quantile(0.75)) if not pd.isna(col_data.quantile(0.75)) else 0.0
                
                # Calculate derived metrics with safety checks
                iqr = q75 - q25 if (q75 is not None and q25 is not None) else 0.0
                range_val = max_val - min_val if (max_val is not None and min_val is not None) else 0.0
                
                # CV with division by zero check
                if mean_val != 0 and not pd.isna(mean_val) and not pd.isna(std_val):
                    cv = (std_val / abs(mean_val)) * 100
                else:
                    cv = 0.0
                
                # Skewness with enough data points
                try:
                    skewness = float(col_data.skew()) if len(col_data) > 2 and not pd.isna(col_data.skew()) else 0.0
                except:
                    skewness = 0.0
                
                stats[col] = {
                    "mean": mean_val,
                    "median": median_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                    "q25": q25,
                    "q75": q75,
                    "iqr": iqr,
                    "range": range_val,
                    "cv": cv,
                    "skewness": skewness,
                    "count": int(col_data.count()),
                    "missing": int(data[col].isna().sum())
                }
                
            except Exception as e:
                logger.warning(f"Error calculating statistics for column {col}: {e}")
                continue
        
        return stats
    
    def _detect_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect patterns in the data"""
        patterns = {
            "trends": [],
            "outliers": [],
            "correlations": []
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Detect trends (simple linear regression slope)
        for col in numeric_cols:
            if len(data) > 1:
                x = np.arange(len(data))
                y = data[col].values
                
                # Remove NaN values
                mask = ~np.isnan(y)
                if mask.sum() > 1:
                    slope = np.polyfit(x[mask], y[mask], 1)[0]
                    
                    if abs(slope) > data[col].std() * 0.1:
                        direction = "increasing" if slope > 0 else "decreasing"
                        patterns["trends"].append({
                            "column": col,
                            "direction": direction,
                            "strength": abs(float(slope))
                        })
        
        # Detect outliers using IQR method
        for col in numeric_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            
            if len(outliers) > 0:
                patterns["outliers"].append({
                    "column": col,
                    "count": len(outliers),
                    "values": outliers[col].tolist()[:5]  # Limit to 5 examples
                })
        
        # Detect correlations
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr_value = corr_matrix.loc[col1, col2]
                    
                    if abs(corr_value) > 0.5:
                        patterns["correlations"].append({
                            "columns": [col1, col2],
                            "correlation": float(corr_value),
                            "strength": "strong" if abs(corr_value) > 0.7 else "moderate"
                        })
        
        return patterns
    
    async def _generate_llm_insights(
        self,
        data: pd.DataFrame,
        stats: Dict[str, Any],
        patterns: Dict[str, Any],
        context: Optional[str] = None,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Use LLM to generate natural language insights"""
        
        # Prepare data summary for LLM
        data_summary = {
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "dtypes": data.dtypes.astype(str).to_dict(),
            "sample": data.head(5).to_dict(),
            "statistics": stats,
            "patterns": patterns
        }
        
        prompt = f"""You are a data analyst. Analyze the following data and provide insights.

Data Summary:
- Shape: {data_summary['shape'][0]} rows, {data_summary['shape'][1]} columns
- Columns: {', '.join(data_summary['columns'])}

Statistics:
{self._format_stats_for_prompt(stats)}

Detected Patterns:
{self._format_patterns_for_prompt(patterns)}

{f'Context: {context}' if context else ''}
{f'Focus on: {", ".join(focus_areas)}' if focus_areas else ''}

Please provide:
1. Key findings (2-3 sentences summary)
2. 3-5 specific insights about the data
3. 2-3 actionable recommendations

Format your response as JSON:
{{
  "key_findings": "summary here",
  "insights": ["insight 1", "insight 2", ...],
  "recommendations": ["recommendation 1", "recommendation 2", ...]
}}"""

        try:
            # Use the helper function that handles context properly
            response_text = await get_llm_response(prompt)
            
            # Try to parse JSON from response
            # Clean up markdown code blocks if present
            clean_response = response_text.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.startswith("```"):
                clean_response = clean_response[3:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            result = json.loads(clean_response)
            return result
            
        except json.JSONDecodeError as je:
            logger.warning(f"Failed to parse LLM response as JSON: {je}")
            # Fallback: try to extract insights from text
            return self._parse_text_insights(response_text)
        except Exception as e:
            logger.error(f"Error generating LLM insights: {e}")
            return {
                "key_findings": "Analysis complete. Statistical patterns detected in the data.",
                "insights": [
                    f"Dataset contains {data.shape[0]} records across {data.shape[1]} columns.",
                    "Statistical analysis completed successfully.",
                    "Review the patterns section for detailed findings."
                ],
                "recommendations": [
                    "Examine the detected trends for actionable insights.",
                    "Investigate any outliers identified in the analysis."
                ]
            }
    
    def _parse_text_insights(self, text: str) -> Dict[str, Any]:
        """Fallback parser for non-JSON responses"""
        lines = text.split('\n')
        insights = []
        recommendations = []
        key_findings = ""
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            lower_line = line.lower()
            if 'key finding' in lower_line or 'summary' in lower_line:
                current_section = 'findings'
            elif 'insight' in lower_line:
                current_section = 'insights'
            elif 'recommendation' in lower_line:
                current_section = 'recommendations'
            elif current_section == 'findings' and not key_findings:
                key_findings = line
            elif current_section == 'insights' and line.startswith(('-', '•', '*', str)):
                insights.append(line.lstrip('-•* 0123456789.'))
            elif current_section == 'recommendations' and line.startswith(('-', '•', '*', str)):
                recommendations.append(line.lstrip('-•* 0123456789.'))
        
        return {
            "key_findings": key_findings or "Analysis completed successfully.",
            "insights": insights[:5] if insights else ["Data analyzed with detected patterns."],
            "recommendations": recommendations[:3] if recommendations else ["Review detailed statistics for insights."]
        }
    
    def _format_stats_for_prompt(self, stats: Dict[str, Any]) -> str:
        """Format statistics for LLM prompt"""
        formatted = []
        for col, values in stats.items():
            formatted.append(f"{col}: mean={values['mean']:.2f}, median={values['median']:.2f}, std={values['std']:.2f}")
        return "\n".join(formatted)
    
    def _format_patterns_for_prompt(self, patterns: Dict[str, Any]) -> str:
        """Format patterns for LLM prompt"""
        formatted = []
        
        if patterns["trends"]:
            formatted.append("Trends:")
            for trend in patterns["trends"]:
                formatted.append(f"  - {trend['column']}: {trend['direction']}")
        
        if patterns["outliers"]:
            formatted.append("Outliers detected:")
            for outlier in patterns["outliers"]:
                formatted.append(f"  - {outlier['column']}: {outlier['count']} outliers")
        
        if patterns["correlations"]:
            formatted.append("Correlations:")
            for corr in patterns["correlations"]:
                formatted.append(f"  - {corr['columns'][0]} & {corr['columns'][1]}: {corr['strength']} ({corr['correlation']:.2f})")
        
        return "\n".join(formatted) if formatted else "No significant patterns detected."