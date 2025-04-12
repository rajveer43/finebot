import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import os
import logging
import re
import json
from src.tools.tool_registry import Tool, tool_registry
from src.llm.llm_manager import llm_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVAnalyzerTool(Tool):
    """Tool for analyzing CSV files based on user queries."""
    
    name = "CSVAnalyzerTool"
    description = "Analyzes CSV/Excel files and performs custom analysis based on user queries"
    
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the CSV or Excel file to analyze"
            },
            "query": {
                "type": "string",
                "description": "User query specifying the analysis to perform"
            },
            "max_rows": {
                "type": "integer",
                "description": "Maximum number of rows to analyze",
                "default": 1000
            },
            "output_format": {
                "type": "string",
                "description": "Format for the output (text, json, or html)",
                "enum": ["text", "json", "html"],
                "default": "text"
            }
        },
        "required": ["file_path", "query"]
    }
    
    output_schema = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Summary of the analysis"
            },
            "results": {
                "type": "object",
                "description": "Detailed results of the analysis"
            },
            "python_code": {
                "type": "string",
                "description": "Python code used for the analysis"
            },
            "error": {
                "type": "string",
                "description": "Error message if analysis failed"
            }
        }
    }
    
    def execute(self, file_path: str, query: str, max_rows: int = 1000, 
               output_format: str = "text") -> Dict[str, Any]:
        """
        Execute CSV analysis based on user query.
        
        Args:
            file_path: Path to the CSV/Excel file
            query: User query specifying the analysis to perform
            max_rows: Maximum number of rows to analyze
            output_format: Format for the output (text, json, or html)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Validate file path
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            # Load the data
            df = self._load_data(file_path, max_rows)
            
            # Generate analysis
            analysis_results, python_code = self._analyze_data(df, query)
            
            # Format results according to the requested output format
            formatted_results = self._format_results(analysis_results, output_format)
            
            return {
                "summary": analysis_results.get("summary", "Analysis completed successfully."),
                "results": formatted_results,
                "python_code": python_code
            }
            
        except Exception as e:
            logger.error(f"Error analyzing CSV: {str(e)}", exc_info=True)
            return {"error": f"Failed to analyze CSV: {str(e)}"}
    
    def _load_data(self, file_path: str, max_rows: int = 1000) -> pd.DataFrame:
        """Load data from CSV or Excel file."""
        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path, nrows=max_rows)
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path, nrows=max_rows)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
                
            # Handle date columns automatically
            for col in df.columns:
                if any(date_term in col.lower() for date_term in ['date', 'time', 'year', 'month', 'day']):
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass  # If conversion fails, keep as is
            
            logger.info(f"Loaded data from {file_path} with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    async def _analyze_data(self, df: pd.DataFrame, query: str) -> tuple:
        """Analyze data based on user query using LLM to generate code."""
        # Identify analysis type from query
        analysis_type = self._identify_analysis_type(query)
        
        # Get basic data information
        data_info = self._get_data_info(df)
        
        # Generate Python code for analysis using LLM
        python_code = await self._generate_analysis_code(query, df, data_info, analysis_type)
        
        # Execute the generated code
        analysis_results = self._execute_analysis_code(python_code, df)
        
        # Generate summary of results
        if "summary" not in analysis_results:
            summary = f"Analysis of {len(df)} rows and {len(df.columns)} columns completed."
            analysis_results["summary"] = summary
            
        return analysis_results, python_code
    
    def _identify_analysis_type(self, query: str) -> str:
        """Identify the type of analysis requested in the query."""
        # Common analysis types
        analysis_types = {
            "statistics": ["statistics", "stats", "stat", "mean", "median", "std", "min", "max", "describe"],
            "correlation": ["correlation", "correlate", "relationship", "corr", "relate", "connected"],
            "aggregation": ["aggregate", "group", "sum", "average", "count", "total"],
            "filtering": ["filter", "where", "find", "search", "rows where", "contains"],
            "outliers": ["outlier", "anomaly", "unusual", "extreme", "detect"],
            "missing": ["missing", "null", "na", "empty", "blank"],
            "trends": ["trend", "change", "over time", "growth", "increase", "decrease"],
            "top": ["top", "bottom", "highest", "lowest", "most", "least", "maximum", "minimum"],
            "distribution": ["distribution", "hist", "histogram", "frequency", "spread", "range"],
            "categorization": ["category", "categorize", "classify", "group", "segment"]
        }
        
        # Check for matches
        matched_types = []
        query_lower = query.lower()
        for analysis_type, keywords in analysis_types.items():
            if any(keyword in query_lower for keyword in keywords):
                matched_types.append(analysis_type)
        
        # Default to statistics if no match
        if not matched_types:
            return "statistics"
        
        # Return the most frequent type if multiple matches
        return max(set(matched_types), key=matched_types.count)
    
    def _get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the DataFrame."""
        try:
            # Basic information
            column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            # Missing values
            missing_values = df.isnull().sum().to_dict()
            
            # Sample data (first few rows)
            sample_json = df.head(3).to_json(orient='records')
            
            return {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "column_types": column_types,
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "date_columns": date_cols,
                "missing_values": missing_values,
                "sample_data": sample_json
            }
        except Exception as e:
            logger.error(f"Error getting data info: {str(e)}")
            return {
                "shape": df.shape,
                "columns": df.columns.tolist()
            }
    
    async def _generate_analysis_code(self, query: str, df: pd.DataFrame, 
                                    data_info: Dict[str, Any], 
                                    analysis_type: str) -> str:
        """Generate Python code for analysis using LLM."""
        # Create prompt for code generation
        prompt = f"""
        Generate Python code to analyze a DataFrame based on this query:
        "{query}"
        
        Data information:
        - Rows: {data_info['shape'][0]}, Columns: {data_info['shape'][1]}
        - Column names: {', '.join(data_info['columns'])}
        - Numeric columns: {', '.join(data_info['numeric_columns'])}
        - Categorical columns: {', '.join(data_info['categorical_columns'])}
        - Date columns: {', '.join(data_info['date_columns'])}
        
        Analysis type: {analysis_type}
        
        The DataFrame is already loaded as 'df'. Write Python code that analyzes the data and returns a dictionary with the results.
        Include a 'summary' key in the returned dictionary with a text summary of the findings.
        Do not use any visualizations or external libraries that aren't standard with pandas/numpy.
        Make sure your code handles potential errors (e.g., missing columns, incorrect data types).
        
        Example output format:
        ```python
        def analyze_data(df):
            # Analysis code here
            results = {{}}
            # Store results
            results["summary"] = "Summary of analysis"
            return results
        ```
        
        Only return the Python code, nothing else.
        """
        
        # Generate code using LLM
        code_text, _ = await llm_manager.generate_response(prompt, temperature=0.2)
        
        # Extract just the code part
        code_match = re.search(r'```python\s*(.*?)\s*```', code_text, re.DOTALL)
        if code_match:
            generated_code = code_match.group(1)
        else:
            # If no code block found, use the whole response
            generated_code = code_text
        
        # Ensure it has the right function name and signature
        if not re.search(r'def\s+analyze_data\s*\(\s*df\s*\)\s*:', generated_code):
            # Add the function definition if missing
            generated_code = f"def analyze_data(df):\n{generated_code}"
        
        return generated_code
    
    def _execute_analysis_code(self, python_code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute the generated Python code safely."""
        try:
            # Create local environment with only the DataFrame
            local_env = {"df": df, "pd": pd, "np": np}
            
            # Execute the code
            exec(python_code, {"__builtins__": {}}, local_env)
            
            # Call the analyze_data function
            if "analyze_data" in local_env and callable(local_env["analyze_data"]):
                results = local_env["analyze_data"](df)
                if not isinstance(results, dict):
                    results = {"results": results, "summary": "Analysis completed."}
                return results
            else:
                return {"error": "No analyze_data function found in generated code."}
        except Exception as e:
            logger.error(f"Error executing analysis code: {str(e)}")
            return {"error": str(e), "summary": "Analysis failed to execute."}
    
    def _format_results(self, results: Dict[str, Any], output_format: str) -> Any:
        """Format results according to the requested output format."""
        if output_format == "json":
            # Convert to JSON-serializable format
            try:
                # Convert pandas objects and numpy types
                def convert_to_serializable(obj):
                    if isinstance(obj, pd.DataFrame):
                        return obj.to_dict(orient='records')
                    elif isinstance(obj, pd.Series):
                        return obj.to_dict()
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif np.issubdtype(type(obj), np.number):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_serializable(item) for item in obj]
                    else:
                        return obj
                
                return convert_to_serializable(results)
            except Exception as e:
                logger.error(f"Error converting results to JSON: {str(e)}")
                return {"error": str(e)}
                
        elif output_format == "html":
            # Create HTML representation
            html_output = "<div class='csv-analysis-results'>"
            
            # Add summary
            if "summary" in results:
                html_output += f"<div class='summary'><p>{results['summary']}</p></div>"
            
            # Add tables for DataFrames
            for key, value in results.items():
                if key == "summary" or key == "error":
                    continue
                    
                if isinstance(value, pd.DataFrame):
                    html_output += f"<h3>{key}</h3>"
                    html_output += value.to_html(classes='dataframe')
                elif isinstance(value, dict):
                    html_output += f"<h3>{key}</h3>"
                    html_output += "<table class='dataframe'>"
                    for k, v in value.items():
                        html_output += f"<tr><td>{k}</td><td>{v}</td></tr>"
                    html_output += "</table>"
                elif isinstance(value, list):
                    html_output += f"<h3>{key}</h3>"
                    html_output += "<ul>"
                    for item in value:
                        html_output += f"<li>{item}</li>"
                    html_output += "</ul>"
                else:
                    html_output += f"<h3>{key}</h3>"
                    html_output += f"<p>{value}</p>"
            
            html_output += "</div>"
            return html_output
            
        else:  # Default to text format
            # Return the results as is
            return results

# Register the tool
tool_registry.register(CSVAnalyzerTool) 