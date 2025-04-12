import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, Any, List, Optional, Union
import os
import numpy as np
from datetime import datetime
import logging
import re
import json
from src.tools.tool_registry import Tool, tool_registry
from src.llm.llm_manager import llm_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicVisualizationTool(Tool):
    """Tool for creating dynamic data visualizations based on user queries using LLM-generated code."""
    
    name = "DynamicVisualizationTool"
    description = "Creates custom data visualizations from CSV/Excel files based on natural language queries"
    
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "User query describing the visualization to create"
            },
            "file_path": {
                "type": "string",
                "description": "Path to the CSV or Excel file containing the data"
            },
            "chart_type": {
                "type": "string",
                "description": "Optional chart type hint"
            },
            "max_rows": {
                "type": "integer",
                "description": "Maximum number of rows to process",
                "default": 1000
            }
        },
        "required": ["query", "file_path"]
    }
    
    output_schema = {
        "type": "object",
        "properties": {
            "visualization_data": {
                "type": "string",
                "description": "Base64 encoded image data of the visualization"
            },
            "insights": {
                "type": "string",
                "description": "Textual insights about the visualized data"
            },
            "python_code": {
                "type": "string",
                "description": "Python code used to generate the visualization"
            },
            "chart_type": {
                "type": "string",
                "description": "Type of chart that was created"
            },
            "error": {
                "type": "string",
                "description": "Error message if visualization failed"
            }
        }
    }
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the visualization based on user query and file.
        
        Args:
            **kwargs: Keyword arguments matching the input schema
            
        Returns:
            Dictionary with visualization data, insights, and the code used
        """
        try:
            # Extract parameters
            query = kwargs.get("query")
            file_path = kwargs.get("file_path")
            chart_type = kwargs.get("chart_type")
            max_rows = kwargs.get("max_rows", 1000)
            
            # Validate file path
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            # Load the data
            df = self._load_data(file_path, max_rows)
            
            # Generate and execute visualization code
            result = self._create_visualization(df, query, chart_type)
            
            return result
        
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}", exc_info=True)
            return {"error": f"Failed to create visualization: {str(e)}"}
    
    def _load_data(self, file_path: str, max_rows: int = 1000) -> pd.DataFrame:
        """Load data from CSV or Excel file."""
        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path, nrows=max_rows)
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path, nrows=max_rows)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
                
            # Handle date columns
            for col in df.columns:
                # Check if column name contains date-related terms
                if any(date_term in col.lower() for date_term in ['date', 'time', 'year', 'month', 'day']):
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass  # If conversion fails, keep as is
                        
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    async def _generate_visualization_code(self, df: pd.DataFrame, query: str, chart_type: Optional[str] = None) -> str:
        """Generate Python code for visualization using LLM."""
        # Get data information
        data_info = self._get_data_info(df)
        
        # Detect chart type from query if not provided
        detected_chart_type = chart_type or self._detect_chart_type_from_query(query)
        
        # Create a prompt for code generation
        prompt = f"""
        Generate Python code to create a {detected_chart_type} visualization based on this query:
        "{query}"
        
        Data information:
        - Rows: {data_info['shape'][0]}, Columns: {data_info['shape'][1]}
        - Column names: {', '.join(data_info['columns'])}
        - Numeric columns: {', '.join(data_info['numeric_columns'])}
        - Categorical columns: {', '.join(data_info['categorical_columns'])}
        - Date columns: {', '.join(data_info['date_columns'])}
        - First few rows: {data_info['sample_data']}
        
        Write Python code that:
        1. Creates a detailed and professional-looking {detected_chart_type} visualization using matplotlib/seaborn
        2. Makes appropriate decisions about which columns to use based on the query
        3. Includes proper titles, labels, and annotations
        4. Uses the 'plt.figure(figsize=(12, 7))' for sizing
        5. Handles any necessary data transformations
        6. Generates insights about the data pattern or trend shown in the visualization
        7. Returns both the figure and a summary of insights
        
        Example output format:
        ```python
        def create_visualization(df):
            # Data preparation
            prepared_df = df.copy()
            
            # Create the visualization
            plt.figure(figsize=(12, 7))
            # Visualization code here...
            
            # Add title and labels
            plt.title("Title")
            plt.xlabel("X Label")
            plt.ylabel("Y Label")
            
            # Generate insights
            insights = "Key observations about the data..."
            
            # Return the figure and insights
            return plt.gcf(), insights
        ```
        
        Only return the Python code, nothing else. Make sure the code handles potential errors.
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
        if not re.search(r'def\s+create_visualization\s*\(\s*df\s*\)\s*:', generated_code):
            # Add the function definition if missing
            generated_code = f"def create_visualization(df):\n{generated_code}"
        
        return generated_code, detected_chart_type
    
    def _get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the DataFrame."""
        try:
            # Basic information
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            # Sample data (first few rows)
            sample_data = df.head(3).to_dict(orient='records')
            sample_json = json.dumps(sample_data, default=str)
            
            return {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "date_columns": date_cols,
                "sample_data": sample_json
            }
        except Exception as e:
            logger.error(f"Error getting data info: {str(e)}")
            return {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "sample_data": "{}"
            }
    
    def _detect_chart_type_from_query(self, query: str) -> str:
        """Detect chart type from the query."""
        query_lower = query.lower()
        
        # Common chart types and their keywords
        chart_types = {
            "line chart": ["line chart", "trend", "over time", "time series", "growth", "progression", "evolution"],
            "bar chart": ["bar chart", "bar graph", "comparison", "compare", "ranking", "rank"],
            "scatter plot": ["scatter plot", "scatter", "correlation", "relationship", "x vs y"],
            "pie chart": ["pie chart", "pie", "proportion", "percentage", "composition", "share"],
            "histogram": ["histogram", "distribution", "frequency", "density"],
            "box plot": ["box plot", "box", "whisker", "distribution comparison", "outliers"],
            "heatmap": ["heatmap", "heat map", "correlation matrix", "matrix"],
            "area chart": ["area chart", "area", "cumulative", "stacked"],
            "bubble chart": ["bubble chart", "bubble", "three variables"]
        }
        
        # Check for direct mentions of chart types
        for chart_type, keywords in chart_types.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return chart_type
        
        # Default to line chart for financial data, otherwise auto-detect
        if any(term in query_lower for term in ["financial", "finance", "stock", "price", "revenue", "profit"]):
            return "line chart"
        
        return "auto-detect"
    
    async def _create_visualization(self, df: pd.DataFrame, query: str, chart_type: Optional[str] = None) -> Dict[str, Any]:
        """Create a visualization based on the query and data."""
        try:
            # Generate visualization code
            python_code, detected_chart_type = await self._generate_visualization_code(df, query, chart_type)
            
            # Execute the visualization code
            fig, insights = self._execute_visualization_code(python_code, df)
            
            # Save figure to a base64 encoded string
            buf = io.BytesIO()
            try:
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            except Exception as save_err:
                logger.error(f"Error saving figure to buffer: {str(save_err)}")
                # Try to save with a different approach
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            finally:
                plt.close(fig)
                
            # Verify we have image data
            if not image_data or len(image_data) < 100:
                logger.warning(f"Generated image data is suspiciously small ({len(image_data) if image_data else 0} bytes)")
            else:
                logger.info(f"Successfully generated visualization, data size: {len(image_data)} bytes")
            
            # Save a backup of the visualization to filesystem
            try:
                viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "visualizations")
                os.makedirs(viz_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{detected_chart_type}_{timestamp}.png"
                filepath = os.path.join(viz_dir, filename)
                
                # Save directly from the figure
                fig.savefig(filepath, format='png', dpi=100, bbox_inches='tight')
                logger.info(f"Backup visualization saved to: {filepath}")
            except Exception as backup_err:
                logger.warning(f"Failed to save backup visualization: {str(backup_err)}")
            
            return {
                "visualization_data": image_data,
                "insights": insights,
                "python_code": python_code,
                "chart_type": detected_chart_type
            }
        except Exception as e:
            logger.error(f"Error in visualization creation: {str(e)}", exc_info=True)
            return {"error": f"Failed to create visualization: {str(e)}"}
    
    def _execute_visualization_code(self, python_code: str, df: pd.DataFrame) -> tuple:
        """Execute the generated Python code for visualization."""
        try:
            # Create a local environment with visualization libraries
            local_env = {
                "df": df, 
                "pd": pd, 
                "np": np, 
                "plt": plt, 
                "sns": sns, 
                "datetime": datetime
            }
            
            # Execute the code
            exec(python_code, local_env)
            
            # Call the create_visualization function
            if "create_visualization" in local_env and callable(local_env["create_visualization"]):
                fig, insights = local_env["create_visualization"](df)
                return fig, insights
            else:
                raise ValueError("No create_visualization function found in generated code.")
        except Exception as e:
            logger.error(f"Error executing visualization code: {str(e)}", exc_info=True)
            # Create a fallback visualization with error message
            fallback_fig = plt.figure(figsize=(12, 7))
            plt.text(0.5, 0.5, f"Error creating visualization:\n{str(e)}", 
                    horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.tight_layout()
            return fallback_fig, f"Failed to generate visualization: {str(e)}"
            
# Register the tool with the registry
tool_registry.register(DynamicVisualizationTool) 