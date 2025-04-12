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
from src.tools.tool_registry import Tool, tool_registry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataVisualizationTool(Tool):
    """Tool for visualizing financial data from CSV/Excel files."""
    
    name = "DataVisualizationTool"
    description = "Creates data visualizations from CSV/Excel files based on user queries"
    
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "User query about what visualization to create"
            },
            "file_path": {
                "type": "string",
                "description": "Path to the CSV or Excel file containing the data"
            },
            "chart_type": {
                "type": "string",
                "enum": ["line", "bar", "pie", "scatter", "histogram", "box", "heatmap", "auto"],
                "description": "Type of chart to create (auto will try to determine the best chart)"
            },
            "x_column": {
                "type": "string",
                "description": "Column to use for x-axis"
            },
            "y_column": {
                "type": "string",
                "description": "Column to use for y-axis"
            },
            "groupby_column": {
                "type": "string",
                "description": "Column to group data by"
            },
            "title": {
                "type": "string",
                "description": "Title for the chart"
            },
            "limit": {
                "type": "integer",
                "description": "Limit the number of rows to process"
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
            "chart_type": {
                "type": "string",
                "description": "Type of chart that was created"
            },
            "data_summary": {
                "type": "object",
                "description": "Statistical summary of the data"
            },
            "error": {
                "type": "string",
                "description": "Error message if visualization failed"
            }
        }
    }
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the visualization based on the query and file.
        
        Args:
            **kwargs: Keyword arguments matching the input schema
            
        Returns:
            Dictionary with visualization data and insights
        """
        try:
            # Extract parameters
            query = kwargs.get("query")
            file_path = kwargs.get("file_path")
            chart_type = kwargs.get("chart_type", "auto")
            x_column = kwargs.get("x_column")
            y_column = kwargs.get("y_column")
            groupby_column = kwargs.get("groupby_column")
            title = kwargs.get("title")
            limit = kwargs.get("limit", 1000)
            
            # Validate file path
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            # Load the data
            df = self._load_data(file_path, limit)
            
            # If x_column or y_column not provided, try to determine from query
            if not x_column or not y_column:
                x_column, y_column, groupby_column, chart_type, title = self._analyze_query(query, df)
            
            # Create visualization
            plt.figure(figsize=(12, 7))
            fig, image_data, insights = self._create_visualization(
                df, chart_type, x_column, y_column, groupby_column, title
            )
            
            # Generate data summary
            data_summary = self._generate_data_summary(df, x_column, y_column)
            
            return {
                "visualization_data": image_data,
                "insights": insights,
                "chart_type": chart_type,
                "data_summary": data_summary
            }
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}", exc_info=True)
            return {"error": f"Failed to create visualization: {str(e)}"}
    
    def _load_data(self, file_path: str, limit: int = 1000) -> pd.DataFrame:
        """Load data from CSV or Excel file."""
        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path, nrows=limit)
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path, nrows=limit)
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
    
    def _analyze_query(self, query: str, df: pd.DataFrame) -> tuple:
        """Analyze the query to determine visualization parameters."""
        # Default values
        x_col = None
        y_col = None
        group_col = None
        chart = "auto"
        chart_title = "Data Visualization"
        detected_lang = 'en'  # Initialize detected_lang with default value
        
        # Translation support for non-English queries
        original_query = query
        
        try:
            # Check if query is not in English and translate if needed
            import langdetect
            
            detected_lang = langdetect.detect(query)
            if detected_lang != 'en':
                # Log the language detection
                logger.info(f"Detected query language: {detected_lang}")
                # Note: We're not doing the translation here as it requires async
                # The detect_and_translate function is async and would need proper handling
            
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}, proceeding with original query")
            detected_lang = 'en'  # Reset to English on failure
        
        # Pre-identify data types in the dataframe
        date_cols = []
        numeric_cols = []
        categorical_cols = []
        
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or any(term in col.lower() for term in 
                                                          ['date', 'time', 'year', 'month', 'quarter', 'period']):
                date_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        # Prefer date columns for x-axis in financial data (time series)
        if date_cols:
            x_col = date_cols[0]
            logger.info(f"Using date column as x-axis by default: {x_col}")
        
        # Enhanced financial terms detection
        financial_terms = {
            'revenue': ['revenue', 'sales', 'income', 'earnings', 'turnover', 'proceeds', 'ingresos', 'ventas'],
            'expense': ['expense', 'cost', 'expenditure', 'spending', 'payment', 'outlay', 'gastos', 'costos'],
            'profit': ['profit', 'margin', 'gain', 'earnings', 'return', 'yield', 'beneficio', 'ganancia'],
            'growth': ['growth', 'increase', 'rise', 'gain', 'expansion', 'appreciation', 'crecimiento', 'aumento'],
            'comparison': ['compare', 'comparison', 'versus', 'vs', 'against', 'contrast', 'comparar', 'comparación'],
            'trend': ['trend', 'pattern', 'movement', 'direction', 'progression', 'tendencia', 'patrón'],
            'average': ['average', 'mean', 'median', 'typical', 'normal', 'promedio', 'media'],
            'top': ['top', 'best', 'highest', 'maximum', 'leading', 'superior', 'principal', 'mejor']
        }
        
        # Financial specific analyses
        analysis_types = {
            'trend': ['trend', 'over time', 'progression', 'movement', 'direction', 'development', 'tendencia'],
            'comparison': ['compare', 'comparison', 'versus', 'vs', 'contrast', 'difference', 'comparación'],
            'distribution': ['distribution', 'spread', 'range', 'variation', 'dispersion', 'distribución'],
            'correlation': ['correlation', 'relationship', 'connection', 'association', 'link', 'correlación'],
            'proportion': ['proportion', 'percentage', 'ratio', 'share', 'portion', 'composition', 'proporción'],
            'summary': ['summary', 'overview', 'statistics', 'metrics', 'indicators', 'resumen', 'estadísticas']
        }
        
        # Determine analysis type
        detected_analysis = None
        for analysis, terms in analysis_types.items():
            if any(term in query.lower() for term in terms):
                detected_analysis = analysis
                break
        
        # Financial metrics detection for y-axis
        detected_metrics = []
        for metric, terms in financial_terms.items():
            if any(term in query.lower() for term in terms):
                detected_metrics.append(metric)
                # Find matching columns for metrics
                for col in numeric_cols:
                    if any(term in col.lower() for term in terms):
                        if not y_col:
                            y_col = col
                            logger.info(f"Using financial metric column as y-axis: {y_col}")
                            break
        
        # Find column matches in the query and df
        for col in df.columns:
            col_lower = col.lower()
            # Check if column name is mentioned in the query
            if col_lower in query.lower():
                # Ensure we prioritize date columns for x-axis
                if col in date_cols and not x_col:
                    x_col = col
                    logger.info(f"Column '{col}' explicitly mentioned in query set as x_col")
                # If we have a date column as x and this is numeric, use as y
                elif col in numeric_cols and x_col and not y_col:
                    y_col = col
                    logger.info(f"Column '{col}' explicitly mentioned in query set as y_col")
                # Otherwise follow the standard logic
                elif not x_col:
                    x_col = col
                    logger.info(f"Column '{col}' explicitly mentioned in query set as x_col")
                elif not y_col:
                    y_col = col
                    logger.info(f"Column '{col}' explicitly mentioned in query set as y_col")
                elif not group_col:
                    group_col = col
                    logger.info(f"Column '{col}' explicitly mentioned in query set as group_col")
        
        # Time period detection for financial analysis
        time_periods = ['year', 'quarter', 'month', 'week', 'day', 'annual', 'quarterly', 'monthly', 'weekly', 'daily']
        time_period_mentions = [period for period in time_periods if period in query.lower()]
        
        # If we detect time periods but no x column, look for matching time columns
        if time_period_mentions and not x_col:
            for period in time_period_mentions:
                matching_cols = [col for col in df.columns if period.lower() in col.lower()]
                if matching_cols:
                    x_col = matching_cols[0]
                    logger.info(f"Time period '{period}' mentioned, using column '{x_col}' as x-axis")
                    break
        
        # Q1, Q2, etc. detection in financial queries
        quarter_pattern = r'Q[1-4]'
        quarters = re.findall(quarter_pattern, query, re.IGNORECASE)
        if quarters and not group_col:
            quarter_cols = [col for col in df.columns if 'quarter' in col.lower() or 'q' in col.lower()]
            if quarter_cols:
                # If there's a specific quarter column, use it for grouping
                group_col = quarter_cols[0]
                logger.info(f"Quarter pattern detected, using '{group_col}' for grouping")
        
        # Handle "top N" queries (e.g., "top 5 products by sales")
        top_n_pattern = r'top\s+(\d+)'
        top_n_match = re.search(top_n_pattern, query.lower())
        if top_n_match:
            try:
                # Get the value (e.g., 5 from "top 5")
                n = int(top_n_match.group(1))
                # Sort logic will be handled in visualization creation
                chart = 'bar'  # Top N is typically shown as bar chart
                logger.info(f"Top {n} pattern detected, using bar chart")
            except:
                pass
        
        # Determine chart type based on analysis type
        if detected_analysis:
            if detected_analysis == 'trend':
                chart = 'line'
            elif detected_analysis == 'comparison':
                chart = 'bar'
            elif detected_analysis == 'distribution':
                chart = 'histogram'
            elif detected_analysis == 'correlation':
                chart = 'scatter'
            elif detected_analysis == 'proportion':
                chart = 'pie'
            logger.info(f"Analysis type '{detected_analysis}' detected, using {chart} chart")
        
        # Extract title from query
        title_match = re.search(r'show (.*?)( for | from )', query.lower())
        if title_match:
            chart_title = title_match.group(1).strip().title() + " Visualization"
        else:
            # Alternative title extraction
            title_match = re.search(r'(compare|show|visualize|analyze|plot|graph) (.*?)( in | from | between )', query.lower())
            if title_match:
                chart_title = title_match.group(2).strip().title() + " Visualization"
        
        # Fallbacks if we couldn't determine columns
        if not x_col and len(df.columns) > 0:
            # Prefer date column for x-axis
            if date_cols:
                x_col = date_cols[0]
                logger.info(f"No explicit x-axis column detected, using date column '{x_col}' as default")
            else:
                x_col = df.columns[0]
                logger.info(f"No explicit x-axis column detected, using first column '{x_col}' as default")
                
        if not y_col and len(df.columns) > 1:
            # Try to find a numeric column for y
            if numeric_cols:
                # For financial data, prefer columns with financial terms
                financial_numeric_cols = [col for col in numeric_cols if 
                                        any(term in col.lower() for term in 
                                           ['revenue', 'profit', 'sales', 'cost', 'income', 'expense'])]
                if financial_numeric_cols:
                    y_col = financial_numeric_cols[0]
                    logger.info(f"No explicit y-axis column detected, using financial column '{y_col}' as default")
                else:
                    y_col = numeric_cols[0]
                    logger.info(f"No explicit y-axis column detected, using numeric column '{y_col}' as default")
            else:
                y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                logger.info(f"No numeric columns available, using column '{y_col}' as default y-axis")
        
        # For multilingual support, check if we should return response in original language
        # We'll store the original language in the chart title for later use
        if detected_lang != 'en' and detected_lang:
            chart_title = f"{chart_title} [{detected_lang}]"
        
        logger.info(f"Query analysis results: x={x_col}, y={y_col}, group={group_col}, chart={chart}, title={chart_title}")
        return x_col, y_col, group_col, chart, chart_title
    
    def _create_visualization(self, df: pd.DataFrame, chart_type: str, 
                             x_column: str, y_column: str,
                             groupby_column: Optional[str] = None,
                             title: Optional[str] = None) -> tuple:
        """Create the visualization based on the parameters."""
        fig = plt.figure(figsize=(12, 7))
        
        # Detect top N pattern in title
        top_n = None
        top_n_pattern = r'top\s+(\d+)'
        if title:
            top_n_match = re.search(top_n_pattern, title.lower())
            if top_n_match:
                try:
                    top_n = int(top_n_match.group(1))
                except:
                    pass
        
        # Check for language marker in the title
        target_language = 'en'
        lang_pattern = r'\[(.*?)\]'
        if title:
            lang_match = re.search(lang_pattern, title)
            if lang_match:
                target_language = lang_match.group(1)
                # Remove the language marker from the title
                title = re.sub(r'\s*\[.*?\]', '', title)
                
        # Auto-determine chart type if set to 'auto'
        if chart_type == 'auto':
            chart_type = self._determine_chart_type(df, x_column, y_column)
        
        # Set a default title if none provided
        if not title:
            title = f"{y_column} by {x_column}"
            if groupby_column:
                title += f" grouped by {groupby_column}"
                
        # Handle data preparation and sorting
        prepared_df = df.copy()
        
        # Convert date columns if needed
        if not pd.api.types.is_datetime64_any_dtype(prepared_df[x_column]) and any(date_term in x_column.lower() 
                for date_term in ['date', 'time', 'year', 'month', 'day', 'quarter']):
            try:
                prepared_df[x_column] = pd.to_datetime(prepared_df[x_column])
            except:
                pass  # If conversion fails, continue with original data
        
        # Sort data temporally if x is a datetime
        if pd.api.types.is_datetime64_any_dtype(prepared_df[x_column]):
            prepared_df = prepared_df.sort_values(by=x_column)
            
        # Apply Top N filtering if specified
        if top_n and top_n > 0:
            if groupby_column:
                # For grouped data, get top N groups by sum of y_column
                if pd.api.types.is_datetime64_any_dtype(prepared_df[y_column]):
                    # For datetime columns, use count instead of sum
                    grouped_counts = prepared_df.groupby(groupby_column)[y_column].count().sort_values(ascending=False)
                    top_groups = grouped_counts.head(top_n).index.tolist()
                else:
                    # For non-datetime columns, use sum as before
                    grouped_sums = prepared_df.groupby(groupby_column)[y_column].sum().sort_values(ascending=False)
                    top_groups = grouped_sums.head(top_n).index.tolist()
                prepared_df = prepared_df[prepared_df[groupby_column].isin(top_groups)]
            else:
                # For ungrouped data, get top N x values by y_column
                if pd.api.types.is_datetime64_any_dtype(prepared_df[y_column]):
                    # For datetime columns, sort directly rather than trying to sum
                    prepared_df = prepared_df.sort_values(by=y_column).head(top_n)
                else:
                    # For non-datetime columns, use sum as before
                    grouped_data = prepared_df.groupby(x_column)[y_column].sum().reset_index()
                    grouped_data = grouped_data.sort_values(by=y_column, ascending=False).head(top_n)
                    prepared_df = prepared_df[prepared_df[x_column].isin(grouped_data[x_column])]
        
        # Set aesthetics - use financial themed style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(x_column, fontsize=12)
        plt.ylabel(y_column, fontsize=12)
        
        # Generate insights placeholder
        insights = "Data visualization insights will be provided here."
        
        # Create the visualization based on chart type
        if chart_type == 'line':
            # Check if data needs to be resampled (for financial time series)
            if pd.api.types.is_datetime64_any_dtype(prepared_df[x_column]):
                # Determine appropriate frequency based on data
                date_range = prepared_df[x_column].max() - prepared_df[x_column].min()
                
                if groupby_column:
                    for group, group_df in prepared_df.groupby(groupby_column):
                        plt.plot(group_df[x_column], group_df[y_column], marker='o', label=str(group))
                    plt.legend(title=groupby_column)
                else:
                    plt.plot(prepared_df[x_column], prepared_df[y_column], marker='o', linewidth=2)
                    
                # Add trendline for financial data
                try:
                    from scipy import stats
                    if not groupby_column:  # Only add trendline for ungrouped data
                        # Convert datetime to ordinal for regression
                        x_numeric = pd.to_numeric(prepared_df[x_column])
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            range(len(x_numeric)), prepared_df[y_column])
                        trendline = [intercept + slope * i for i in range(len(x_numeric))]
                        plt.plot(prepared_df[x_column], trendline, 'r--', linewidth=1, 
                                 label=f'Trend (r²={r_value**2:.2f})')
                        plt.legend()
                except Exception as e:
                    logger.warning(f"Could not add trendline: {str(e)}")
                    
                # Format x-axis for dates
                plt.gcf().autofmt_xdate()
            else:
                if groupby_column:
                    for group, group_df in prepared_df.groupby(groupby_column):
                        plt.plot(group_df[x_column], group_df[y_column], marker='o', label=str(group))
                    plt.legend(title=groupby_column)
                else:
                    plt.plot(prepared_df[x_column], prepared_df[y_column], marker='o', linewidth=2)
            
            # Generate financial trend insights
            insights = self._generate_trend_insights(prepared_df, x_column, y_column)
            
        elif chart_type == 'bar':
            # For financial bar charts, sort by values if not time-based
            if not pd.api.types.is_datetime64_any_dtype(prepared_df[x_column]) and top_n is None:
                # Sort by value unless it's a time series
                if not groupby_column:
                    sorted_df = prepared_df.sort_values(by=y_column, ascending=False)
                    if len(sorted_df) > 15:  # Limit to top 15 if there are too many
                        sorted_df = sorted_df.head(15)
                    plt.bar(sorted_df[x_column], sorted_df[y_column], 
                           color='#1f77b4', alpha=0.8)
                else:
                    # For grouped data
                    pivot_data = prepared_df.pivot_table(
                        index=x_column, 
                        columns=groupby_column, 
                        values=y_column, 
                        aggfunc='sum'
                    )
                    # Sort by total if too many categories
                    if len(pivot_data) > 15:
                        pivot_data['total'] = pivot_data.sum(axis=1)
                        pivot_data = pivot_data.sort_values('total', ascending=False).head(15)
                        pivot_data = pivot_data.drop('total', axis=1)
                    pivot_data.plot(kind='bar', ax=plt.gca(), rot=45)
            else:
                if groupby_column:
                    # Use pivot tables for grouped financial data
                    pivot_data = prepared_df.pivot_table(
                        index=x_column, 
                        columns=groupby_column, 
                        values=y_column, 
                        aggfunc='sum'
                    )
                    pivot_data.plot(kind='bar', ax=plt.gca())
                else:
                    prepared_df.plot(x=x_column, y=y_column, kind='bar', ax=plt.gca(), 
                                    color='#1f77b4', rot=45)
            
            # Add data labels for financial data
            for p in plt.gca().patches:
                value = p.get_height()
                # Format large numbers with K, M suffix
                if value >= 1e6:
                    value_str = f'{value/1e6:.1f}M'
                elif value >= 1e3:
                    value_str = f'{value/1e3:.1f}K'
                else:
                    value_str = f'{value:.1f}'
                plt.gca().annotate(value_str, (p.get_x() + p.get_width()/2., value),
                                  ha='center', va='bottom', rotation=0, fontsize=8)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            insights = self._generate_comparison_insights(prepared_df, x_column, y_column, groupby_column)
            
        elif chart_type == 'pie':
            # For financial pie charts, we want to ensure we're showing percentages
            # and typically want to aggregate data
            if groupby_column:
                # Use groupby_column for pies if specified
                pie_data = prepared_df.groupby(groupby_column)[y_column].sum()
            else:
                pie_data = prepared_df.groupby(x_column)[y_column].sum()
            
            # Limit to top categories if there are too many
            if len(pie_data) > 8:
                others_sum = pie_data.nsmallest(len(pie_data) - 8).sum()
                pie_data = pie_data.nlargest(8)
                pie_data['Others'] = others_sum
                
            # Sort for better visualization
            pie_data = pie_data.sort_values(ascending=False)
            
            # Calculate percentages for labels
            total = pie_data.sum()
            pie_labels = [f'{idx} ({val/total:.1%})' for idx, val in zip(pie_data.index, pie_data.values)]
            
            plt.pie(pie_data, labels=pie_labels, autopct='', 
                   shadow=False, startangle=90, 
                   wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.ylabel('')  # Remove y-label for pie charts
            
            insights = self._generate_proportion_insights(pie_data)
            
        elif chart_type == 'scatter':
            if groupby_column:
                # Use a different color for each group
                for group, group_df in prepared_df.groupby(groupby_column):
                    plt.scatter(group_df[x_column], group_df[y_column], 
                               label=str(group), alpha=0.7, s=50)
                plt.legend(title=groupby_column)
            else:
                plt.scatter(prepared_df[x_column], prepared_df[y_column], alpha=0.7, s=50)
            
            # Add trend line for financial analysis
            try:
                if pd.api.types.is_numeric_dtype(prepared_df[y_column]) and \
                   pd.api.types.is_numeric_dtype(prepared_df[x_column].astype(float)):
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        prepared_df[x_column].astype(float), prepared_df[y_column])
                    
                    # Add regression line
                    x_range = np.linspace(prepared_df[x_column].min(), prepared_df[x_column].max(), 100)
                    y_pred = intercept + slope * x_range
                    plt.plot(x_range, y_pred, 'r--', 
                           label=f'y = {slope:.2f}x + {intercept:.2f} (r²={r_value**2:.2f})')
                    plt.legend()
            except Exception as e:
                logger.warning(f"Could not calculate regression: {str(e)}")
                
            insights = self._generate_correlation_insights(prepared_df, x_column, y_column)
            
        elif chart_type == 'histogram':
            if pd.api.types.is_numeric_dtype(prepared_df[y_column]):
                if groupby_column and prepared_df[groupby_column].nunique() <= 5:
                    # Create multiple histograms for different groups
                    for group, group_df in prepared_df.groupby(groupby_column):
                        plt.hist(group_df[y_column], bins=15, alpha=0.5, label=str(group))
                    plt.legend(title=groupby_column)
                else:
                    # Calculate optimal bin size using Freedman-Diaconis rule
                    q75, q25 = np.percentile(prepared_df[y_column], [75, 25])
                    iqr = q75 - q25
                    bin_width = 2 * iqr / (len(prepared_df) ** (1/3))
                    bin_count = int(np.ceil((prepared_df[y_column].max() - prepared_df[y_column].min()) / bin_width))
                    bin_count = max(5, min(50, bin_count))  # Keep bins between 5 and 50
                    
                    plt.hist(prepared_df[y_column], bins=bin_count, alpha=0.7, color='#1f77b4',
                            edgecolor='black', linewidth=0.5)
                    
                # Add vertical line for mean and median
                plt.axvline(prepared_df[y_column].mean(), color='red', linestyle='--', 
                           linewidth=1, label=f'Mean: {prepared_df[y_column].mean():.2f}')
                plt.axvline(prepared_df[y_column].median(), color='green', linestyle='-', 
                           linewidth=1, label=f'Median: {prepared_df[y_column].median():.2f}')
                plt.legend()
            
            insights = self._generate_distribution_insights(prepared_df, y_column)
            
        elif chart_type == 'box':
            if groupby_column:
                # Sort categories by median for better visualization
                grouped_medians = prepared_df.groupby(groupby_column)[y_column].median().sort_values(ascending=False)
                category_order = grouped_medians.index.tolist()
                
                # Create box plot with ordered categories
                sns.boxplot(x=groupby_column, y=y_column, data=prepared_df, 
                          order=category_order, ax=plt.gca())
            else:
                # If no groupby, create boxplot by x_column
                if prepared_df[x_column].nunique() <= 10:  # Only use boxplot if reasonable number of categories
                    sns.boxplot(x=x_column, y=y_column, data=prepared_df, ax=plt.gca())
                else:
                    # Fall back to single boxplot if too many categories
                    plt.boxplot(prepared_df[y_column])
                    plt.xticks([1], [y_column])
                    
            insights = self._generate_distribution_insights(prepared_df, y_column, x_column)
            
        elif chart_type == 'heatmap':
            if groupby_column:
                # Create pivot table for heatmap
                pivot_data = prepared_df.pivot_table(index=x_column, columns=groupby_column, values=y_column)
                # Limit size for readability
                if len(pivot_data) > 15 or pivot_data.shape[1] > 10:
                    if len(pivot_data) > 15:
                        pivot_data = pivot_data.iloc[:15, :]
                    if pivot_data.shape[1] > 10:
                        pivot_data = pivot_data.iloc[:, :10]
                
                # Use nicer color map and add annotations
                sns.heatmap(pivot_data, annot=True, cmap="YlGnBu", fmt=".1f", 
                          linewidths=0.5, ax=plt.gca())
            else:
                # For financial data, correlation heatmap is often useful
                numeric_df = prepared_df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) >= 3:
                    corr = numeric_df.corr()
                    mask = np.triu(np.ones_like(corr, dtype=bool))  # Create mask for upper triangle
                    sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", 
                              linewidths=0.5, ax=plt.gca())
                else:
                    # Fallback for limited numeric columns
                    sns.heatmap(pd.crosstab(prepared_df[x_column], prepared_df[y_column]), 
                              cmap="YlGnBu", ax=plt.gca())
            
            plt.xticks(rotation=45, ha='right')
            insights = "Heatmap showing patterns and relationships between variables."
            
        # Add financial-specific annotations
        if pd.api.types.is_numeric_dtype(prepared_df[y_column]) and chart_type not in ['pie', 'heatmap']:
            # Add some stats to the plot
            stats_text = (
                f"Min: {prepared_df[y_column].min():.2f}\n"
                f"Max: {prepared_df[y_column].max():.2f}\n"
                f"Mean: {prepared_df[y_column].mean():.2f}\n"
                f"Median: {prepared_df[y_column].median():.2f}"
            )
            plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        va='top', fontsize=9)
        
        # Add timestamp to plot for financial reporting
        plt.figtext(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                   ha='right', va='bottom', fontsize=8, alpha=0.7)
            
        # Tight layout for better spacing
        plt.tight_layout()
        
        # Save figure to a base64 encoded string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # Translate insights if needed
        if target_language != 'en':
            try:
                from src.tools.language_tool import translate_text
                insights = translate_text(insights, target_language)
            except Exception as e:
                logger.warning(f"Could not translate insights: {str(e)}")
        
        return fig, image_data, insights
    
    def _determine_chart_type(self, df: pd.DataFrame, x_column: str, y_column: str) -> str:
        """Determine the best chart type based on the data."""
        # Check data types
        x_dtype = df[x_column].dtype
        y_dtype = df[y_column].dtype
        
        # Time series data is best shown as a line chart
        if pd.api.types.is_datetime64_any_dtype(df[x_column]):
            return 'line'
            
        # Categorical vs. numeric often works well as a bar chart
        if pd.api.types.is_categorical_dtype(df[x_column]) or df[x_column].nunique() < 15:
            if pd.api.types.is_numeric_dtype(df[y_column]):
                return 'bar'
                
        # Two numeric variables can use scatter plot
        if (pd.api.types.is_numeric_dtype(df[x_column]) and 
            pd.api.types.is_numeric_dtype(df[y_column])):
            return 'scatter'
            
        # If x has many unique values, it may be better as a histogram
        if pd.api.types.is_numeric_dtype(df[y_column]) and df[x_column].nunique() > 20:
            return 'histogram'
            
        # Default to bar chart
        return 'bar'
    
    def _generate_data_summary(self, df: pd.DataFrame, x_column: str, y_column: str) -> Dict[str, Any]:
        """Generate statistical summary of the data."""
        summary = {}
        
        # Basic summary stats for y column if numeric
        if pd.api.types.is_numeric_dtype(df[y_column]):
            summary["y_stats"] = {
                "mean": float(df[y_column].mean()),
                "median": float(df[y_column].median()),
                "min": float(df[y_column].min()),
                "max": float(df[y_column].max()),
                "std": float(df[y_column].std())
            }
            
        # Count unique values in x column
        summary["x_unique_count"] = int(df[x_column].nunique())
        
        # Check for missing values
        summary["missing_values"] = {
            "x_column": int(df[x_column].isna().sum()),
            "y_column": int(df[y_column].isna().sum())
        }
        
        # Total row count
        summary["row_count"] = len(df)
        
        return summary
    
    def _generate_trend_insights(self, df: pd.DataFrame, x_column: str, y_column: str) -> str:
        """Generate insights for trend/line charts."""
        if not pd.api.types.is_numeric_dtype(df[y_column]):
            return "The trend visualization shows patterns over the selected categories."
            
        insights = []
        
        # Check if x is a datetime
        is_time_series = pd.api.types.is_datetime64_any_dtype(df[x_column])
        
        # Calculate overall trend
        try:
            # Get trend direction using linear regression
            if is_time_series:
                # Convert datetime to ordinal for regression
                x_numeric = np.array(range(len(df)))
            else:
                x_numeric = df[x_column].astype(float)
                
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, df[y_column])
            
            # Calculate percent change from first to last value
            first_value = df[y_column].iloc[0]
            last_value = df[y_column].iloc[-1]
            if first_value != 0:
                percent_change = ((last_value - first_value) / abs(first_value)) * 100
            else:
                percent_change = np.inf if last_value > 0 else -np.inf if last_value < 0 else 0
                
            # Determine significant change
            if abs(percent_change) < 1:
                change_desc = "remained stable"
            elif percent_change > 0:
                if percent_change > 50:
                    change_desc = f"increased significantly by {percent_change:.1f}%"
                else:
                    change_desc = f"increased by {percent_change:.1f}%"
            else:
                if percent_change < -50:
                    change_desc = f"decreased significantly by {abs(percent_change):.1f}%"
                else:
                    change_desc = f"decreased by {abs(percent_change):.1f}%"
                    
            # Determine trend strength
            if abs(r_value) > 0.7:
                trend_strength = "strong"
            elif abs(r_value) > 0.4:
                trend_strength = "moderate"
            else:
                trend_strength = "weak"
                
            # Create trend insight
            if is_time_series:
                time_period = "the observed period"
                time_unit = "time"
                
                # Try to determine appropriate time unit
                time_range = df[x_column].max() - df[x_column].min()
                if time_range.days > 365*2:
                    time_unit = "years"
                elif time_range.days > 90:
                    time_unit = "quarters"
                elif time_range.days > 30:
                    time_unit = "months"
                elif time_range.days > 7:
                    time_unit = "weeks"
                else:
                    time_unit = "days"
                    
                trend_insight = f"{y_column} {change_desc} over {time_period}, showing a {trend_strength} trend (r²={r_value**2:.2f})."
            else:
                trend_insight = f"{y_column} shows a {trend_strength} {'positive' if slope > 0 else 'negative'} correlation with {x_column} (r²={r_value**2:.2f})."
                
            insights.append(trend_insight)
        except Exception as e:
            logger.warning(f"Could not calculate trend stats: {str(e)}")
            insights.append(f"The visualization shows the relationship between {x_column} and {y_column}.")
            
        # Check for volatility
        if len(df) >= 3:
            try:
                # Calculate volatility using rolling changes
                if is_time_series:
                    pct_changes = df[y_column].pct_change().dropna()
                    volatility = pct_changes.std() * 100  # Convert to percentage
                    
                    if volatility > 20:
                        vol_desc = "highly volatile"
                    elif volatility > 10:
                        vol_desc = "moderately volatile"
                    elif volatility > 5:
                        vol_desc = "somewhat volatile"
                    else:
                        vol_desc = "relatively stable"
                        
                    insights.append(f"The data shows {vol_desc} behavior with {volatility:.1f}% average fluctuation between consecutive periods.")
            except Exception as e:
                logger.warning(f"Could not calculate volatility: {str(e)}")
                
        # Identify peaks and troughs for financial data
        try:
            if len(df) >= 5:
                from scipy.signal import find_peaks
                
                # Find peaks (high points)
                peaks, _ = find_peaks(df[y_column], distance=len(df)//10)
                # Find troughs (low points) by inverting the data
                troughs, _ = find_peaks(-df[y_column], distance=len(df)//10)
                
                if len(peaks) > 0 and is_time_series:
                    peak_times = df.iloc[peaks][x_column]
                    peak_values = df.iloc[peaks][y_column]
                    max_peak_idx = peak_values.idxmax()
                    max_peak_time = df.loc[max_peak_idx, x_column]
                    max_peak_value = df.loc[max_peak_idx, y_column]
                    
                    if isinstance(max_peak_time, pd.Timestamp):
                        max_peak_time_str = max_peak_time.strftime('%b %Y' if max_peak_time.year != df[x_column].min().year else '%b')
                    else:
                        max_peak_time_str = str(max_peak_time)
                        
                    insights.append(f"Peak {y_column} of {max_peak_value:.2f} was observed in {max_peak_time_str}.")
                    
                if len(troughs) > 0 and is_time_series:
                    trough_times = df.iloc[troughs][x_column]
                    trough_values = df.iloc[troughs][y_column]
                    min_trough_idx = trough_values.idxmin()
                    min_trough_time = df.loc[min_trough_idx, x_column]
                    min_trough_value = df.loc[min_trough_idx, y_column]
                    
                    if isinstance(min_trough_time, pd.Timestamp):
                        min_trough_time_str = min_trough_time.strftime('%b %Y' if min_trough_time.year != df[x_column].min().year else '%b')
                    else:
                        min_trough_time_str = str(min_trough_time)
                        
                    insights.append(f"Lowest {y_column} of {min_trough_value:.2f} was observed in {min_trough_time_str}.")
        except Exception as e:
            logger.warning(f"Could not identify peaks and troughs: {str(e)}")
            
        # Add recent trend for financial reporting
        if is_time_series and len(df) >= 3:
            try:
                # Get most recent periods (last 3 or 30% of data, whichever is less)
                recent_cutoff = max(len(df) - 3, int(len(df) * 0.7))
                recent_data = df.iloc[recent_cutoff:]
                
                if len(recent_data) >= 2:
                    recent_first = recent_data[y_column].iloc[0]
                    recent_last = recent_data[y_column].iloc[-1]
                    
                    if recent_first != 0:
                        recent_pct_change = ((recent_last - recent_first) / abs(recent_first)) * 100
                    else:
                        recent_pct_change = np.inf if recent_last > 0 else -np.inf if recent_last < 0 else 0
                        
                    if abs(recent_pct_change) < 1:
                        recent_trend = "remained stable"
                    elif recent_pct_change > 0:
                        recent_trend = f"increased by {recent_pct_change:.1f}%"
                    else:
                        recent_trend = f"decreased by {abs(recent_pct_change):.1f}%"
                        
                    insights.append(f"In the most recent period, {y_column} has {recent_trend}.")
            except Exception as e:
                logger.warning(f"Could not calculate recent trend: {str(e)}")
                
        return " ".join(insights)
    
    def _generate_comparison_insights(self, df: pd.DataFrame, x_column: str, y_column: str, 
                                     groupby_column: Optional[str] = None) -> str:
        """Generate insights for comparison charts."""
        insights = []
        
        try:
            if groupby_column:
                # Generate insights for grouped data
                pivot_data = df.pivot_table(index=x_column, columns=groupby_column, 
                                           values=y_column, aggfunc='mean')
                
                # Identify highest and lowest categories for each group
                for group in pivot_data.columns:
                    max_category = pivot_data[group].idxmax()
                    min_category = pivot_data[group].idxmin()
                    max_value = pivot_data[group].max()
                    min_value = pivot_data[group].min()
                    
                    insights.append(f"For {group}, {max_category} has the highest {y_column} ({max_value:.2f}) and {min_category} has the lowest ({min_value:.2f}).")
                    
                # Compare between groups if there are exactly 2 groups
                if len(pivot_data.columns) == 2:
                    group1, group2 = pivot_data.columns
                    avg1 = pivot_data[group1].mean()
                    avg2 = pivot_data[group2].mean()
                    
                    if avg1 != 0 and avg2 != 0:
                        pct_diff = ((avg1 - avg2) / abs(avg2)) * 100
                        if pct_diff > 0:
                            group_insight = f"On average, {group1} is {pct_diff:.1f}% higher than {group2}."
                        else:
                            group_insight = f"On average, {group1} is {abs(pct_diff):.1f}% lower than {group2}."
                        insights.append(group_insight)
                        
                # Add variance analysis for financial data
                for group in pivot_data.columns:
                    variance = pivot_data[group].var()
                    std_dev = pivot_data[group].std()
                    mean = pivot_data[group].mean()
                    
                    if mean != 0:
                        cv = (std_dev / abs(mean)) * 100  # Coefficient of variation as percentage
                        
                        if cv > 50:
                            variance_insight = f"{group} shows high variability with a coefficient of variation of {cv:.1f}%."
                        elif cv > 25:
                            variance_insight = f"{group} shows moderate variability with a coefficient of variation of {cv:.1f}%."
                        else:
                            variance_insight = f"{group} shows relatively consistent values with a coefficient of variation of {cv:.1f}%."
                            
                        insights.append(variance_insight)
            
            else:
                # Generate insights for ungrouped data
                if pd.api.types.is_numeric_dtype(df[y_column]):
                    # Find highest and lowest categories
                    if len(df) > 0:
                        # Handle case with too many categories by limiting
                        if df[x_column].nunique() > 20:
                            top_categories = df.groupby(x_column)[y_column].mean().nlargest(3)
                            bottom_categories = df.groupby(x_column)[y_column].mean().nsmallest(3)
                            
                            top_insight = "Top categories: " + ", ".join([f"{idx} ({val:.2f})" for idx, val in top_categories.items()])
                            bottom_insight = "Bottom categories: " + ", ".join([f"{idx} ({val:.2f})" for idx, val in bottom_categories.items()])
                            
                            insights.append(f"{top_insight}")
                            insights.append(f"{bottom_insight}")
                        else:
                            max_idx = df.groupby(x_column)[y_column].mean().idxmax()
                            min_idx = df.groupby(x_column)[y_column].mean().idxmin()
                            max_val = df.groupby(x_column)[y_column].mean().max()
                            min_val = df.groupby(x_column)[y_column].mean().min()
                            
                            insights.append(f"{max_idx} has the highest {y_column} at {max_val:.2f}, while {min_idx} has the lowest at {min_val:.2f}.")
                            
                            # Calculate difference between highest and lowest
                            if min_val != 0:
                                diff_pct = ((max_val - min_val) / abs(min_val)) * 100
                                insights.append(f"The highest value is {diff_pct:.1f}% greater than the lowest value.")
                                
                    # Calculate distribution statistics for financial analysis
                    mean = df[y_column].mean()
                    median = df[y_column].median()
                    
                    if mean != 0:
                        mean_median_diff = ((mean - median) / abs(mean)) * 100
                        
                        if abs(mean_median_diff) > 20:
                            insights.append(f"The distribution is skewed, with the mean {mean:.2f} differing from the median {median:.2f} by {abs(mean_median_diff):.1f}%.")
                        
                    # Check for outliers in financial data
                    q1 = df[y_column].quantile(0.25)
                    q3 = df[y_column].quantile(0.75)
                    iqr = q3 - q1
                    upper_bound = q3 + 1.5 * iqr
                    lower_bound = q1 - 1.5 * iqr
                    
                    outliers = df[(df[y_column] > upper_bound) | (df[y_column] < lower_bound)]
                    
                    if len(outliers) > 0:
                        if len(outliers) == 1:
                            outlier_idx = outliers.iloc[0][x_column]
                            outlier_val = outliers.iloc[0][y_column]
                            insights.append(f"There is 1 notable outlier: {outlier_idx} with value {outlier_val:.2f}.")
                        else:
                            insights.append(f"There are {len(outliers)} outliers in the dataset that significantly differ from the typical values.")
        
        except Exception as e:
            logger.warning(f"Error generating comparison insights: {str(e)}", exc_info=True)
            insights.append(f"The chart compares {y_column} across different {x_column} categories.")
            
        return " ".join(insights)
    
    def _generate_correlation_insights(self, df: pd.DataFrame, x_column: str, y_column: str) -> str:
        """Generate insights for scatter plots showing correlations."""
        insights = f"Correlation analysis between {x_column} and {y_column}:\n"
        
        # Only applicable if both columns are numeric
        if pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column]):
            corr = df[[x_column, y_column]].corr().iloc[0, 1]
            
            # Interpret correlation coefficient
            if abs(corr) > 0.8:
                strength = "very strong"
            elif abs(corr) > 0.6:
                strength = "strong"
            elif abs(corr) > 0.4:
                strength = "moderate"
            elif abs(corr) > 0.2:
                strength = "weak"
            else:
                strength = "very weak"
                
            direction = "positive" if corr >= 0 else "negative"
            
            insights += f"• Correlation coefficient: {corr:.2f}\n"
            insights += f"• This indicates a {strength} {direction} correlation between {x_column} and {y_column}.\n"
            
            if direction == "positive":
                insights += f"• As {x_column} increases, {y_column} tends to increase as well.\n"
            else:
                insights += f"• As {x_column} increases, {y_column} tends to decrease.\n"
                
            # R-squared value
            r_squared = corr ** 2
            insights += f"• R-squared value: {r_squared:.2f} - indicating that {r_squared*100:.1f}% of the variance in {y_column} can be explained by {x_column}.\n"
        else:
            insights += "• Correlation analysis requires numeric data. One or both columns are non-numeric.\n"
            
        return insights
    
    def _generate_distribution_insights(self, df: pd.DataFrame, y_column: str, 
                                      x_column: Optional[str] = None) -> str:
        """Generate insights for distributions (histograms, box plots)."""
        if not pd.api.types.is_numeric_dtype(df[y_column]):
            return f"Distribution of {y_column} shows the frequency of different categories."
            
        insights = f"Distribution analysis of {y_column}:\n"
        
        # Basic statistics
        mean = df[y_column].mean()
        median = df[y_column].median()
        std = df[y_column].std()
        skewness = df[y_column].skew()
        
        insights += f"• Mean: {mean:.2f}, Median: {median:.2f}\n"
        insights += f"• Standard Deviation: {std:.2f}\n"
        
        # Analyze distribution shape
        if abs(skewness) < 0.5:
            insights += "• The distribution appears relatively symmetric.\n"
        elif skewness > 0:
            insights += f"• The distribution is positively skewed (skewness: {skewness:.2f}), with a longer tail to the right.\n"
        else:
            insights += f"• The distribution is negatively skewed (skewness: {skewness:.2f}), with a longer tail to the left.\n"
            
        # Check for outliers
        q1 = df[y_column].quantile(0.25)
        q3 = df[y_column].quantile(0.75)
        iqr = q3 - q1
        outlier_count = ((df[y_column] < (q1 - 1.5 * iqr)) | (df[y_column] > (q3 + 1.5 * iqr))).sum()
        
        if outlier_count > 0:
            outlier_pct = outlier_count / len(df) * 100
            insights += f"• Potential outliers detected: {outlier_count} points ({outlier_pct:.1f}% of data) fall outside the expected range.\n"
            
        return insights
    
    def _generate_proportion_insights(self, series: pd.Series) -> str:
        """Generate insights for pie charts showing proportions."""
        insights = f"Proportion analysis:\n"
        
        # Sort values from largest to smallest
        sorted_data = series.sort_values(ascending=False)
        
        # Top categories
        top_category = sorted_data.index[0]
        top_value = sorted_data.iloc[0]
        top_pct = top_value / sorted_data.sum() * 100
        
        insights += f"• Largest segment: {top_category} represents {top_pct:.1f}% of the total.\n"
        
        # Concentration analysis
        top_three_pct = sorted_data.iloc[:3].sum() / sorted_data.sum() * 100
        insights += f"• Top 3 categories account for {top_three_pct:.1f}% of the total.\n"
        
        # Distribution evenness
        if len(sorted_data) > 1:
            smallest_pct = sorted_data.iloc[-1] / sorted_data.sum() * 100
            ratio = top_pct / smallest_pct if smallest_pct > 0 else float('inf')
            
            if ratio > 10:
                insights += "• The distribution is highly concentrated, with significant disparities between segments.\n"
            elif ratio > 3:
                insights += "• The distribution shows moderate concentration, with some segments significantly larger than others.\n"
            else:
                insights += "• The distribution is relatively even across the different segments.\n"
                
        return insights

# Register the tool with the registry
tool_registry.register(DataVisualizationTool) 