"""
SQL Agent module for BSP AI Assistant
Handles SQL query generation and execution for data filtering and aggregation
"""

import pandas as pd
import sqlite3
import json
from typing import Dict, Any, Optional, List
from utils.utils import get_logger
import chainlit as cl
import time
import re

logger = get_logger()


class SQLAgent:
    """Generate and execute SQL queries for data transformation"""
    
    def __init__(self):
        self.connection = None
        self.table_name = "data_table"
    
    async def process_sql_request(
        self,
        df: pd.DataFrame,
        user_prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process SQL request by generating and executing queries
        
        Args:
            df: Input DataFrame
            user_prompt: User's request describing desired transformation
            context: Additional context (e.g., detected chart types, insight areas)
            
        Returns:
            Dictionary containing transformed data and query info
        """
        try:
            # Create in-memory SQLite database
            self._create_temp_database(df)
            
            # Generate SQL query using LLM
            sql_query = await self._generate_sql_query(df, user_prompt, context)
            
            if not sql_query:
                return {
                    "success": False,
                    "error": "Could not generate SQL query"
                }
            
            # Execute query
            result_df = self._execute_query(sql_query)
            
            # Validate result
            if result_df is None or result_df.empty:
                return {
                    "success": False,
                    "error": "Query returned no results"
                }
            
            return {
                "success": True,
                "data": result_df,
                "query": sql_query,
                "original_shape": df.shape,
                "result_shape": result_df.shape,
                "transformation_summary": self._get_transformation_summary(df, result_df, sql_query)
            }
            
        except Exception as e:
            logger.error(f"Error processing SQL request: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            self._close_connection()
    
    def _create_temp_database(self, df: pd.DataFrame):
        """Create temporary SQLite database with the DataFrame"""
        try:
            # Close existing connection if any
            self._close_connection()
            
            # Create in-memory database
            self.connection = sqlite3.connect(':memory:')
            
            # Load DataFrame into SQLite
            df.to_sql(self.table_name, self.connection, index=False, if_exists='replace')
            
            logger.info(f"Created temp database with table '{self.table_name}' ({df.shape[0]} rows, {df.shape[1]} cols)")
            
        except Exception as e:
            logger.error(f"Error creating temp database: {e}")
            raise
    
    async def _generate_sql_query(
        self,
        df: pd.DataFrame,
        user_prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Use LLM to generate SQL query based on user request
        
        Args:
            df: DataFrame schema reference
            user_prompt: User's request
            context: Additional context about the analysis
            
        Returns:
            SQL query string
        """
        try:
            # Prepare schema information
            schema_info = self._get_schema_info(df)
            
            # Build context string
            context_str = ""
            if context:
                if context.get("chart_types"):
                    context_str += f"\nRequested chart types: {', '.join(context['chart_types'])}"
                if context.get("focus_areas"):
                    context_str += f"\nFocus areas: {', '.join(context['focus_areas'])}"
            
            prompt = f"""You are a SQL expert. Generate a SQL query to transform data based on the user's request.

**Database Schema:**
Table name: {self.table_name}
Columns and types:
{schema_info}

**Sample Data (first 3 rows):**
{df.head(3).to_string()}

**User Request:** "{user_prompt}"
{context_str}

**Guidelines:**
1. Generate ONLY the SQL query, no explanations
2. Use standard SQLite syntax
3. Always use table name: {self.table_name}
4. Column names are case-sensitive
5. For date filtering, use proper date functions (e.g., strftime, date())
6. For aggregations, use appropriate GROUP BY
7. Include ORDER BY if sorting is mentioned
8. Use WHERE clauses for filtering conditions
9. Support HAVING for aggregate filtering
10. Return only SELECT queries (no DDL/DML)

**Common patterns:**
- Filter by year: WHERE strftime('%Y', date_column) = '2025'
- Filter by date range: WHERE date_column BETWEEN 'start' AND 'end'
- Aggregate by group: SELECT category, SUM(amount) FROM {self.table_name} GROUP BY category
- Top N results: ORDER BY column DESC LIMIT N
- Multiple conditions: WHERE condition1 AND condition2

**Response Format:**
Return ONLY the SQL query without any markdown, explanations, or formatting.
Example: SELECT * FROM {self.table_name} WHERE year = 2025

SQL Query:"""

            # Get LLM response
            response = await self._get_llm_response(prompt)
            
            # Clean and validate query
            query = self._clean_sql_query(response)
            
            if not query:
                logger.warning("LLM returned empty query")
                return None
            
            # Validate query
            if not self._validate_query(query):
                logger.warning(f"Generated query failed validation: {query}")
                return None
            
            logger.info(f"Generated SQL query: {query}")
            return query
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            return None
    
    def _get_schema_info(self, df: pd.DataFrame) -> str:
        """Get formatted schema information"""
        schema_lines = []
        for col, dtype in df.dtypes.items():
            # Map pandas dtypes to SQL types
            sql_type = self._map_dtype_to_sql(dtype)
            # Show sample values
            sample = df[col].dropna().head(2).tolist()
            sample_str = f" (e.g., {sample})" if sample else ""
            schema_lines.append(f"  - {col}: {sql_type}{sample_str}")
        
        return "\n".join(schema_lines)
    
    def _map_dtype_to_sql(self, dtype) -> str:
        """Map pandas dtype to SQL type"""
        dtype_str = str(dtype)
        if 'int' in dtype_str:
            return 'INTEGER'
        elif 'float' in dtype_str:
            return 'REAL'
        elif 'datetime' in dtype_str:
            return 'DATETIME'
        elif 'bool' in dtype_str:
            return 'BOOLEAN'
        else:
            return 'TEXT'
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get LLM response for SQL generation"""
        try:
            chat_settings = cl.user_session.get("chat_settings", {})
            provider = chat_settings.get("model_provider", "litellm")
            
            cl.user_session.set("start_time", time.time())
            
            if provider == "foundry":
                from utils.foundry import chat_agent
                
                # Ensure analytics mode is enabled
                analytics_mode = cl.user_session.get("analytics_mode", False)
                if not analytics_mode:
                    cl.user_session.set("analytics_mode", True)
                
                response = await chat_agent(prompt)
                
                # Restore analytics mode state
                if not analytics_mode:
                    cl.user_session.set("analytics_mode", False)
                
                return response
            else:
                # Fallback for other providers
                logger.warning("SQL agent requires Foundry provider")
                return ""
                
        except Exception as e:
            logger.error(f"Error getting LLM response for SQL: {e}")
            return ""
    
    def _clean_sql_query(self, query: str) -> Optional[str]:
        """Clean and extract SQL query from LLM response"""
        if not query:
            return None
        
        query = query.strip()
        
        # Remove markdown code blocks
        if query.startswith("```sql"):
            query = query[6:]
        elif query.startswith("```"):
            query = query[3:]
        
        if query.endswith("```"):
            query = query[:-3]
        
        query = query.strip()
        
        # Remove common prefixes
        prefixes = ["SQL Query:", "Query:", "Here's the query:", "Here is the query:"]
        for prefix in prefixes:
            if query.lower().startswith(prefix.lower()):
                query = query[len(prefix):].strip()
        
        # Ensure query starts with SELECT
        if not query.upper().startswith("SELECT"):
            # Try to find SELECT in the response
            match = re.search(r'(SELECT\s+.*)', query, re.IGNORECASE | re.DOTALL)
            if match:
                query = match.group(1)
            else:
                return None
        
        # Remove trailing semicolon
        query = query.rstrip(';').strip()
        
        return query if query else None
    
    def _validate_query(self, query: str) -> bool:
        """Validate SQL query for safety and correctness"""
        if not query:
            return False
        
        query_upper = query.upper()
        
        # Must be a SELECT query
        if not query_upper.startswith("SELECT"):
            return False
        
        # Disallow dangerous operations
        dangerous_keywords = [
            "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", 
            "CREATE", "TRUNCATE", "EXEC", "EXECUTE"
        ]
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                logger.warning(f"Dangerous keyword '{keyword}' found in query")
                return False
        
        # Must reference the correct table
        if self.table_name not in query:
            logger.warning(f"Query does not reference table '{self.table_name}'")
            return False
        
        return True
    
    def _execute_query(self, query: str) -> Optional[pd.DataFrame]:
        """Execute SQL query and return results as DataFrame"""
        try:
            if not self.connection:
                raise Exception("Database connection not established")
            
            # Execute query
            result_df = pd.read_sql_query(query, self.connection)
            
            logger.info(f"Query executed successfully. Result shape: {result_df.shape}")
            return result_df
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None
    
    def _get_transformation_summary(
        self,
        original_df: pd.DataFrame,
        result_df: pd.DataFrame,
        query: str
    ) -> str:
        """Generate summary of data transformation"""
        summary_parts = []
        
        # Row changes
        orig_rows, result_rows = original_df.shape[0], result_df.shape[0]
        if result_rows < orig_rows:
            reduction = ((orig_rows - result_rows) / orig_rows) * 100
            summary_parts.append(f"Filtered from {orig_rows} to {result_rows} rows ({reduction:.1f}% reduction)")
        elif result_rows > orig_rows:
            summary_parts.append(f"Expanded from {orig_rows} to {result_rows} rows (aggregation or join)")
        else:
            summary_parts.append(f"Maintained {result_rows} rows")
        
        # Column changes
        orig_cols, result_cols = original_df.shape[1], result_df.shape[1]
        if result_cols < orig_cols:
            summary_parts.append(f"Selected {result_cols} of {orig_cols} columns")
        elif result_cols > orig_cols:
            summary_parts.append(f"Generated {result_cols} columns (includes computed columns)")
        
        # Detect operations from query
        query_upper = query.upper()
        operations = []
        
        if "WHERE" in query_upper:
            operations.append("filtering")
        if "GROUP BY" in query_upper:
            operations.append("aggregation")
        if "ORDER BY" in query_upper:
            operations.append("sorting")
        if any(agg in query_upper for agg in ["SUM(", "COUNT(", "AVG(", "MAX(", "MIN("]):
            operations.append("calculation")
        
        if operations:
            summary_parts.append(f"Operations: {', '.join(operations)}")
        
        return " | ".join(summary_parts)
    
    def _close_connection(self):
        """Close database connection"""
        if self.connection:
            try:
                self.connection.close()
                self.connection = None
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        self._close_connection()