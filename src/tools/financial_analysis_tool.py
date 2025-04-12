import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import json
import os
import logging
from datetime import datetime

from src.tools.tool_registry import Tool, tool_registry
from src.tools.file_processor import FileProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialAnalysisTool(Tool):
    """Tool for analyzing financial data from various sources."""
    
    name = "FinancialAnalysisTool"
    description = "Analyze financial data to extract insights, trends, and statistics"
    
    input_schema = {
        "type": "object",
        "properties": {
            "data_source": {
                "type": "string",
                "description": "Path to file containing financial data or JSON string with data"
            },
            "analysis_type": {
                "type": "string",
                "description": "Type of analysis to perform",
                "enum": ["summary", "trend", "comparison", "statistics", "custom"]
            },
            "metrics": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of metrics to analyze"
            },
            "time_period": {
                "type": "string",
                "description": "Time period to analyze (e.g., 'Q1 2023', 'monthly', 'yearly')"
            },
            "custom_query": {
                "type": "string",
                "description": "Custom analysis query (for analysis_type='custom')"
            }
        },
        "required": ["data_source", "analysis_type"]
    }
    
    output_schema = {
        "type": "object",
        "properties": {
            "analysis_results": {
                "type": "object",
                "description": "Results of the financial analysis"
            },
            "charts": {
                "type": "array",
                "description": "List of chart configurations for visualization"
            },
            "insights": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Key insights extracted from the analysis"
            }
        }
    }
    
    def __init__(self):
        super().__init__()
        self.file_processor = FileProcessor()
    
    def execute(self, data_source: str, analysis_type: str, 
                metrics: Optional[List[str]] = None, 
                time_period: Optional[str] = None,
                custom_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute financial analysis on the provided data.
        
        Args:
            data_source: Path to file or JSON string with data
            analysis_type: Type of analysis to perform
            metrics: List of metrics to analyze
            time_period: Time period to analyze
            custom_query: Custom analysis query
            
        Returns:
            Dictionary with analysis results, charts, and insights
        """
        try:
            # Load the data
            df = self._load_data(data_source)
            
            # Perform the requested analysis
            if analysis_type == "summary":
                return self._perform_summary_analysis(df, metrics)
            elif analysis_type == "trend":
                return self._perform_trend_analysis(df, metrics, time_period)
            elif analysis_type == "comparison":
                return self._perform_comparison_analysis(df, metrics, time_period)
            elif analysis_type == "statistics":
                return self._perform_statistical_analysis(df, metrics)
            elif analysis_type == "custom":
                if not custom_query:
                    raise ValueError("Custom analysis requires a custom_query parameter")
                return self._perform_custom_analysis(df, custom_query)
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
                
        except Exception as e:
            logger.error(f"Error in financial analysis: {str(e)}")
            return {
                "error": str(e),
                "analysis_type": analysis_type,
                "data_source": data_source
            }
    
    def _load_data(self, data_source: str) -> pd.DataFrame:
        """
        Load data from file or JSON string.
        
        Args:
            data_source: Path to file or JSON string
            
        Returns:
            Pandas DataFrame with the data
        """
        # Check if data_source is a file path
        if os.path.exists(data_source):
            # Use file processor to load the file
            result = self.file_processor.execute(file_path=data_source)
            
            if "error" in result:
                raise ValueError(f"Error processing file: {result['error']}")
            
            # Check for tables in the response
            if "tables" in result and result["tables"]:
                return pd.DataFrame(result["tables"])
            elif result.get("content_type") == "tabular" and "tables" in result:
                return pd.DataFrame(result["tables"])
            else:
                raise ValueError("The file does not contain tabular data")
        else:
            # Try to parse as JSON string
            try:
                data = json.loads(data_source)
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Convert nested dictionaries to DataFrame
                    return pd.DataFrame([data])
                else:
                    raise ValueError("JSON data must be a list of objects or a single object")
            except json.JSONDecodeError:
                raise ValueError(f"Invalid data source: {data_source}")
    
    def _perform_summary_analysis(self, df: pd.DataFrame, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform summary analysis on the data."""
        if metrics:
            # Filter to requested metrics if they exist in the dataframe
            available_metrics = [m for m in metrics if m in df.columns]
            
            if not available_metrics:
                raise ValueError(f"None of the requested metrics {metrics} found in data")
            
            # Focus on these metrics
            analysis_df = df[available_metrics]
        else:
            # Use all numeric columns
            analysis_df = df.select_dtypes(include=np.number)
            
            if analysis_df.empty:
                raise ValueError("No numeric columns found in data for analysis")
        
        # Calculate summary statistics
        summary_stats = analysis_df.describe().to_dict()
        
        # Identify key columns and their trends
        insights = []
        for col in analysis_df.columns:
            insights.append(f"{col}: Mean value is {analysis_df[col].mean():.2f}")
            if len(analysis_df) > 1:
                insights.append(f"{col}: Trend is {'increasing' if analysis_df[col].iloc[-1] > analysis_df[col].iloc[0] else 'decreasing'}")
        
        # Create chart configurations
        charts = []
        for col in analysis_df.columns[:5]:  # Limit to first 5 columns to avoid too many charts
            charts.append({
                "chart_type": "bar",
                "title": f"Summary of {col}",
                "x_axis": "Statistic",
                "y_axis": "Value",
                "data": [
                    {"Statistic": "Mean", "Value": summary_stats[col]['mean']},
                    {"Statistic": "Median", "Value": summary_stats[col]['50%']},
                    {"Statistic": "Min", "Value": summary_stats[col]['min']},
                    {"Statistic": "Max", "Value": summary_stats[col]['max']}
                ]
            })
        
        return {
            "analysis_results": {
                "summary_statistics": summary_stats,
                "column_count": len(analysis_df.columns),
                "row_count": len(analysis_df)
            },
            "charts": charts,
            "insights": insights
        }
    
    def _perform_trend_analysis(self, df: pd.DataFrame, metrics: Optional[List[str]] = None, 
                               time_period: Optional[str] = None) -> Dict[str, Any]:
        """Perform trend analysis on time-series data."""
        # Try to identify date/time columns
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # If no datetime columns, try to convert string columns that might contain dates
        if not date_cols:
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_cols.append(col)
                    except:
                        pass
        
        if not date_cols:
            # If still no date columns, add an index column
            df['_index'] = range(len(df))
            date_cols = ['_index']
        
        # Select the first date column as x-axis
        x_col = date_cols[0]
        
        # Select metrics to analyze
        if metrics:
            # Filter to requested metrics if they exist in the dataframe
            available_metrics = [m for m in metrics if m in df.columns]
            
            if not available_metrics:
                raise ValueError(f"None of the requested metrics {metrics} found in data")
            
            y_cols = available_metrics
        else:
            # Use all numeric columns except date cols
            y_cols = df.select_dtypes(include=np.number).columns.tolist()
            y_cols = [col for col in y_cols if col not in date_cols and col != '_index']
            
            if not y_cols:
                raise ValueError("No numeric columns found for trend analysis")
        
        # Prepare data for trend analysis
        trend_data = {}
        for col in y_cols:
            if x_col == '_index':
                trend_data[col] = df[col].tolist()
            else:
                trend_data[col] = df[[x_col, col]].sort_values(by=x_col).to_dict(orient='records')
        
        # Calculate trends
        trends = {}
        insights = []
        
        for col in y_cols:
            values = df[col].tolist()
            if len(values) > 1:
                first_val = values[0] if not pd.isna(values[0]) else 0
                last_val = values[-1] if not pd.isna(values[-1]) else 0
                
                if first_val != 0:
                    pct_change = ((last_val - first_val) / first_val) * 100
                else:
                    pct_change = 0
                
                trend_direction = "increasing" if pct_change > 0 else "decreasing"
                
                trends[col] = {
                    "start_value": first_val,
                    "end_value": last_val,
                    "change": last_val - first_val,
                    "pct_change": pct_change,
                    "trend": trend_direction
                }
                
                insights.append(f"{col} showed a {trend_direction} trend of {abs(pct_change):.2f}%")
        
        # Create chart configurations
        charts = []
        for col in y_cols[:3]:  # Limit to first 3 metrics
            chart_data = []
            
            if x_col == '_index':
                x_values = list(range(len(df)))
                y_values = df[col].tolist()
                
                for i, val in enumerate(y_values):
                    chart_data.append({"x": i, "y": val})
            else:
                for _, row in df.iterrows():
                    chart_data.append({
                        "x": row[x_col].strftime('%Y-%m-%d') if isinstance(row[x_col], datetime) else str(row[x_col]),
                        "y": row[col]
                    })
            
            charts.append({
                "chart_type": "line",
                "title": f"Trend of {col}",
                "x_axis": x_col,
                "y_axis": col,
                "data": chart_data
            })
        
        return {
            "analysis_results": {
                "trends": trends,
                "time_column": x_col,
                "metrics_analyzed": y_cols
            },
            "charts": charts,
            "insights": insights
        }
    
    def _perform_comparison_analysis(self, df: pd.DataFrame, metrics: Optional[List[str]] = None, 
                                   time_period: Optional[str] = None) -> Dict[str, Any]:
        """Perform comparison analysis between different metrics or time periods."""
        # Determine which metrics to compare
        if metrics and len(metrics) >= 2:
            # Use the provided metrics for comparison
            comparison_cols = [m for m in metrics if m in df.columns]
            
            if len(comparison_cols) < 2:
                raise ValueError(f"Need at least 2 metrics for comparison, found: {comparison_cols}")
        else:
            # Auto-select numeric columns for comparison
            comparison_cols = df.select_dtypes(include=np.number).columns.tolist()[:5]  # Limit to 5
            
            if len(comparison_cols) < 2:
                raise ValueError("Need at least 2 numeric columns for comparison")
        
        # Calculate correlation matrix
        corr_matrix = df[comparison_cols].corr().round(3).to_dict()
        
        # Calculate ratios between pairs
        ratios = {}
        for i, col1 in enumerate(comparison_cols):
            for col2 in comparison_cols[i+1:]:
                # Skip if any column has zeros to avoid division by zero
                if (df[col2] == 0).any():
                    continue
                    
                ratio_name = f"{col1}_to_{col2}"
                ratios[ratio_name] = (df[col1] / df[col2]).mean()
        
        # Generate insights
        insights = []
        for i, col1 in enumerate(comparison_cols):
            for col2 in comparison_cols[i+1:]:
                if col1 in corr_matrix and col2 in corr_matrix[col1]:
                    corr = corr_matrix[col1][col2]
                    if abs(corr) > 0.7:
                        strength = "strong"
                    elif abs(corr) > 0.3:
                        strength = "moderate"
                    else:
                        strength = "weak"
                        
                    direction = "positive" if corr > 0 else "negative"
                    insights.append(f"{col1} and {col2} have a {strength} {direction} correlation ({corr:.2f}).")
        
        # Compare trends
        trends = {}
        for col in comparison_cols:
            values = df[col].tolist()
            if len(values) > 1:
                first_val = values[0] if not pd.isna(values[0]) else 0
                last_val = values[-1] if not pd.isna(values[-1]) else 0
                
                if first_val != 0:
                    pct_change = ((last_val - first_val) / first_val) * 100
                else:
                    pct_change = 0
                
                trend_direction = "increasing" if pct_change > 0 else "decreasing"
                
                trends[col] = {
                    "change": last_val - first_val,
                    "pct_change": pct_change,
                    "trend": trend_direction
                }
        
        # Create comparison chart
        chart_data = []
        for col in comparison_cols:
            # Normalize values for easier comparison
            normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) if df[col].max() != df[col].min() else df[col]
            chart_data.append({
                "name": col,
                "values": normalized.tolist()
            })
        
        charts = [{
            "chart_type": "comparison",
            "title": "Comparison of Metrics (Normalized)",
            "data": chart_data
        }]
        
        return {
            "analysis_results": {
                "correlation_matrix": corr_matrix,
                "ratios": ratios,
                "trends": trends,
                "metrics_compared": comparison_cols
            },
            "charts": charts,
            "insights": insights
        }
    
    def _perform_statistical_analysis(self, df: pd.DataFrame, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform detailed statistical analysis on the data."""
        # Select metrics to analyze
        if metrics:
            # Filter to requested metrics if they exist in the dataframe
            available_metrics = [m for m in metrics if m in df.columns]
            
            if not available_metrics:
                raise ValueError(f"None of the requested metrics {metrics} found in data")
            
            analysis_df = df[available_metrics]
        else:
            # Use all numeric columns
            analysis_df = df.select_dtypes(include=np.number)
            
            if analysis_df.empty:
                raise ValueError("No numeric columns found in data for analysis")
        
        # Calculate detailed statistics
        stats = {}
        for col in analysis_df.columns:
            col_stats = {}
            series = analysis_df[col].dropna()
            
            # Basic statistics
            col_stats["count"] = len(series)
            col_stats["mean"] = series.mean()
            col_stats["median"] = series.median()
            col_stats["std"] = series.std()
            col_stats["min"] = series.min()
            col_stats["max"] = series.max()
            col_stats["range"] = series.max() - series.min()
            
            # Quartiles
            col_stats["25%"] = series.quantile(0.25)
            col_stats["75%"] = series.quantile(0.75)
            col_stats["iqr"] = col_stats["75%"] - col_stats["25%"]
            
            # Additional statistics
            col_stats["variance"] = series.var()
            col_stats["skew"] = series.skew()
            col_stats["kurtosis"] = series.kurtosis()
            
            stats[col] = col_stats
        
        # Generate insights
        insights = []
        for col, col_stats in stats.items():
            insights.append(f"{col}: Mean={col_stats['mean']:.2f}, Median={col_stats['median']:.2f}, Std={col_stats['std']:.2f}")
            
            # Check for skewness
            if abs(col_stats['skew']) > 1:
                skew_direction = "positive" if col_stats['skew'] > 0 else "negative"
                insights.append(f"{col} shows {skew_direction} skewness ({col_stats['skew']:.2f}), indicating non-normal distribution.")
            
            # Check for outliers using IQR method
            lower_bound = col_stats['25%'] - 1.5 * col_stats['iqr']
            upper_bound = col_stats['75%'] + 1.5 * col_stats['iqr']
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            if len(outliers) > 0:
                insights.append(f"{col} has {len(outliers)} potential outliers ({(len(outliers)/len(series))*100:.1f}% of data).")
        
        # Create chart configurations
        charts = []
        for col in analysis_df.columns[:3]:  # Limit to first 3 columns
            # Box plot data
            box_plot_data = {
                "min": stats[col]["min"],
                "q1": stats[col]["25%"],
                "median": stats[col]["median"],
                "q3": stats[col]["75%"],
                "max": stats[col]["max"]
            }
            
            charts.append({
                "chart_type": "boxplot",
                "title": f"Distribution of {col}",
                "data": box_plot_data
            })
        
        return {
            "analysis_results": {
                "statistics": stats,
                "columns_analyzed": list(analysis_df.columns)
            },
            "charts": charts,
            "insights": insights
        }
    
    def _perform_custom_analysis(self, df: pd.DataFrame, custom_query: str) -> Dict[str, Any]:
        """
        Perform a custom analysis based on a query.
        Note: For real-world use, this would need security measures to prevent code injection.
        """
        # For demonstration, use simple query parsing rather than eval
        # In a production system, this would use a dedicated query language or SQL
        
        results = {}
        insights = []
        
        # Simple example of handling a few custom query types
        if "average" in custom_query.lower() or "mean" in custom_query.lower():
            for col in df.select_dtypes(include=np.number).columns:
                results[f"mean_{col}"] = df[col].mean()
                insights.append(f"The average of {col} is {df[col].mean():.2f}")
        
        elif "sum" in custom_query.lower() or "total" in custom_query.lower():
            for col in df.select_dtypes(include=np.number).columns:
                results[f"sum_{col}"] = df[col].sum()
                insights.append(f"The total of {col} is {df[col].sum():.2f}")
        
        elif "count" in custom_query.lower():
            results["row_count"] = len(df)
            results["column_count"] = len(df.columns)
            insights.append(f"The dataset contains {len(df)} rows and {len(df.columns)} columns")
        
        elif "correlation" in custom_query.lower():
            numeric_df = df.select_dtypes(include=np.number)
            if len(numeric_df.columns) >= 2:
                results["correlation"] = numeric_df.corr().to_dict()
                
                # Find highest correlation
                corr_matrix = numeric_df.corr()
                np.fill_diagonal(corr_matrix.values, 0)  # Exclude self-correlations
                max_corr = corr_matrix.max().max()
                max_corr_idx = np.where(corr_matrix == max_corr)
                
                if len(max_corr_idx[0]) > 0:
                    col1 = corr_matrix.index[max_corr_idx[0][0]]
                    col2 = corr_matrix.columns[max_corr_idx[1][0]]
                    insights.append(f"The highest correlation is between {col1} and {col2}: {max_corr:.2f}")
        
        else:
            # Default to summary stats
            results["summary"] = df.describe().to_dict()
            insights.append("Performed default summary analysis")
        
        # Simple chart based on query
        charts = []
        if "trend" in custom_query.lower() and len(df) > 1:
            # Create a line chart for the first numeric column
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                charts.append({
                    "chart_type": "line",
                    "title": f"Trend of {col}",
                    "data": df[col].tolist()
                })
        
        return {
            "analysis_results": results,
            "charts": charts,
            "insights": insights,
            "query": custom_query
        }


# Register the tool
tool_registry.register(FinancialAnalysisTool) 