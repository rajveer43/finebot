import os
import re
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from src.llm.llm_manager import llm_manager
from src.tools.tool_registry import tool_registry
# Import this to ensure all tools are registered
import src.tools.ensure_tools
from src.config.config import UPLOAD_FOLDER

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialAgent:
    """
    Financial Intelligence Agent that processes queries, coordinates tools,
    and generates responses.
    """
    
    def __init__(self):
        """Initialize the agent with necessary components."""
        self.active_documents = {}
        self.system_prompt = """
        You are a financial intelligence chatbot. Your role is to help users analyze financial
        documents and data. You can process various file formats including CSV, Excel, PDFs,
        and Word documents. You should extract relevant financial information, identify trends,
        and provide insightful analysis.
        
        When a user uploads a document, you should:
        1. Acknowledge the document and identify its type
        2. Extract the key financial information
        3. Provide a summary of the contents
        4. Be ready to answer specific questions about the document
        
        You should be helpful, clear, and accurate in your responses. If you are unsure about
        something, acknowledge your uncertainty. You should be able to handle documents in
        multiple languages.
        
        You have access to various tools to help you analyze documents:
        - File processor for handling different document types
        - Financial analysis for identifying trends and patterns
        - Text summarization for condensing large documents
        - Language translation for multilingual support
        
        You should adapt your analysis based on the type of document. For spreadsheets,
        focus on numerical analysis. For text documents, extract key financial information
        and summarize important points.
        """
        # Set the system prompt for the LLM
        llm_manager.available_providers["gemini"].set_system_prompt(self.system_prompt)
        llm_manager.available_providers["groq"].set_system_prompt(self.system_prompt)
        
        # Make sure we import the tool_registry and ensure_tools module
        try:
            # Import to ensure tools are registered
            import src.tools.ensure_tools
        except Exception as e:
            logger.error(f"Error importing ensure_tools module: {str(e)}")
        
        # Get the available tools with error handling
        try:
            tool_metadatas = tool_registry.get_all_tools()
            self.available_tools = [tool["name"] for tool in tool_metadatas]
        except Exception as e:
            logger.error(f"Error getting tools from registry: {str(e)}")
            self.available_tools = []  # Default to empty list if there's an error
            
        # Log what tools are available
        logger.info(f"Available tools: {', '.join(self.available_tools)}")
        
    async def process_query(self, user_query: str, uploaded_files: Optional[List] = None, 
                          language: str = "en", chat_history: Optional[List] = None) -> Dict[str, Any]:
        """
        Process a query from the user and generate a response.
        
        Args:
            user_query: The user's query text
            uploaded_files: List of files uploaded by the user
            language: The language to use for the response
            chat_history: List of previous messages in the conversation
            
        Returns:
            A dictionary containing the response text and any additional data
        """
        try:
            # Process any uploaded files first
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    await self._process_uploaded_file(uploaded_file)
            
            # Analyze intent to determine the best tools to use
            intent_analysis, provider = await llm_manager.analyze_intent(user_query)
            logger.info(f"Intent analysis from {provider}: {intent_analysis}")
            
            # Extract key information from intent analysis
            primary_intent = intent_analysis.get("intent", "question_answering")
            tools_needed = intent_analysis.get("tools_needed", [])
            
            # Store results from various tools
            results = {}
            
            # Check for web search or online information need
            web_search_keywords = ['latest', 'current', 'recent', 'news', 'update', 'online', 'internet', 'web', 'search']
            needs_web_search = any(keyword in user_query.lower() for keyword in web_search_keywords)
            
            # Look for web search intent or explicit URL
            if needs_web_search or primary_intent == "extraction" or "SearchAPITool" in tools_needed:
                # Check if a URL is explicitly mentioned
                url_pattern = re.compile(r'https?://\S+')
                urls = re.findall(url_pattern, user_query)
                
                # If URLs are found, use WebSearchTool
                if urls:
                    web_search_tool = tool_registry.create_tool_instance("WebSearchTool")
                    if web_search_tool:
                        for url in urls:
                            # Determine if we should download linked documents
                            download_docs = "download" in user_query.lower() or "get linked" in user_query.lower()
                            
                            try:
                                # Execute web search
                                search_result = web_search_tool.execute(
                                    url=url,
                                    extract_type="all",
                                    download_linked_docs=download_docs
                                )
                                
                                results["web_search"] = search_result
                                
                                # If documents were downloaded, process them
                                if search_result.get("downloaded_files"):
                                    for file_path in search_result["downloaded_files"]:
                                        await self._process_uploaded_file(file_path)
                                
                                # If tables were extracted, store them for analysis
                                if search_result.get("tables"):
                                    # Store tables as JSON for financial analysis
                                    table_data = json.dumps(search_result["tables"])
                                    temp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                                          "uploads", f"extracted_tables_{len(self.active_documents)}.json")
                                    
                                    with open(temp_path, 'w') as f:
                                        f.write(table_data)
                                    
                                    # Process the table file
                                    await self._process_uploaded_file(temp_path)
                                    
                                # Add tools used
                                if "tools_used" not in results:
                                    results["tools_used"] = []
                                results["tools_used"].append("WebSearchTool")
                            except Exception as e:
                                logger.error(f"Error executing web search tool: {str(e)}")
                    else:
                        logger.error("Failed to create WebSearchTool tool instance")
                # If no URLs but needs web search, use SearchAPITool
                elif needs_web_search:
                    try:
                        search_api_tool = tool_registry.create_tool_instance("SearchAPITool")
                        if search_api_tool:
                            try:
                                # Extract a search query from the user query
                                search_query = user_query
                                search_terms = ['search for', 'look up', 'find information', 'latest on', 'recent news about']
                                
                                for term in search_terms:
                                    if term in user_query.lower():
                                        search_parts = user_query.lower().split(term)
                                        if len(search_parts) > 1:
                                            search_query = search_parts[1].strip()
                                            break
                                
                                # Execute search
                                search_result = search_api_tool.execute(
                                    query=search_query,
                                    num_results=5,
                                    fetch_full_content=True
                                )
                                
                                # Add to results
                                results["search_api"] = search_result
                                
                                # Add tools used
                                if "tools_used" not in results:
                                    results["tools_used"] = []
                                results["tools_used"].append("SearchAPITool")
                                
                                logger.info(f"Executed search API with query: {search_query}")
                            except Exception as e:
                                logger.error(f"Error executing search API tool: {str(e)}")
                        else:
                            logger.error("Failed to create SearchAPITool tool instance")
                    except Exception as e:
                        logger.error(f"Failed to create SearchAPITool tool instance: {str(e)}")
                        # Continue with other tools even if search fails

            # Check for CSV analysis requests
            csv_analysis_keywords = ['analyze', 'analysis', 'examine', 'statistics', 'csv data', 'excel data', 'get statistics', 'summarize']
            needs_csv_analysis = any(keyword in user_query.lower() for keyword in csv_analysis_keywords)
            
            # Check for visualization requests
            visualization_keywords = ['visualize', 'chart', 'plot', 'graph', 'show me', 'trend', 'compare', 'display', 'analyze']
            needs_visualization = any(keyword in user_query.lower() for keyword in visualization_keywords)
            
            # Handle CSV analysis
            if needs_csv_analysis or primary_intent in ['data_analysis', 'statistical_analysis']:
                # Find active documents that could be analyzed (CSV/Excel)
                analyzable_docs = {}
                for name, info in self.active_documents.items():
                    if name.lower().endswith(('.csv', '.xlsx', '.xls')):
                        analyzable_docs[name] = info
                
                if analyzable_docs:
                    # Select the most appropriate file based on the query
                    selected_doc_name = None
                    
                    # Look for file names in the query
                    for doc_name in analyzable_docs.keys():
                        # Try to find mentions of the file in the query
                        doc_base = os.path.basename(doc_name)
                        doc_without_ext = os.path.splitext(doc_base)[0]
                        
                        if doc_base.lower() in user_query.lower() or doc_without_ext.lower() in user_query.lower():
                            selected_doc_name = doc_name
                            logger.info(f"Selected document for analysis based on name match: {selected_doc_name}")
                            break
                    
                    # If no specific file was mentioned, use the most recently added one
                    if not selected_doc_name:
                        selected_doc_name = list(analyzable_docs.keys())[0]
                        # For multiple files, try to find the most recent one
                        if len(analyzable_docs) > 1:
                            selected_doc_name = max(
                                analyzable_docs.keys(),
                                key=lambda k: analyzable_docs[k].get("timestamp", 0)
                            )
                            logger.info(f"Selected most recent document for analysis: {selected_doc_name}")
                    
                    file_path = analyzable_docs[selected_doc_name]['path']
                    
                    # Create CSV analyzer tool
                    csv_analyzer_tool = tool_registry.create_tool_instance("CSVAnalyzerTool")
                    if csv_analyzer_tool:
                        # Execute CSV analysis
                        analysis_params = {
                            "file_path": file_path,
                            "query": user_query,
                            "output_format": "html",  # Rich output for Streamlit
                            "max_rows": 5000  # Reasonable limit
                        }
                        
                        logger.info(f"Executing CSV analysis with parameters: {analysis_params}")
                        try:
                            analysis_result = csv_analyzer_tool.execute(**analysis_params)
                            results["csv_analysis"] = analysis_result
                            
                            # Add tools used
                            if "tools_used" not in results:
                                results["tools_used"] = []
                            results["tools_used"].append("CSVAnalyzerTool")
                        except Exception as e:
                            logger.error(f"Error executing CSV analyzer tool: {str(e)}")
                    else:
                        logger.error("Failed to create CSVAnalyzerTool instance")
            
            # Handle visualization with dynamic tool
            if needs_visualization or primary_intent in ['visualization', 'trend_analysis', 'comparison']:
                # Find active documents that could be visualized (CSV/Excel)
                visualizable_docs = {}
                for name, info in self.active_documents.items():
                    if name.lower().endswith(('.csv', '.xlsx', '.xls')):
                        visualizable_docs[name] = info
                
                if visualizable_docs:
                    # Select the most appropriate file based on the query
                    selected_doc_name = None
                    
                    # Look for file names in the query
                    for doc_name in visualizable_docs.keys():
                        # Try to find mentions of the file in the query
                        doc_base = os.path.basename(doc_name)
                        doc_without_ext = os.path.splitext(doc_base)[0]
                        
                        if doc_base.lower() in user_query.lower() or doc_without_ext.lower() in user_query.lower():
                            selected_doc_name = doc_name
                            logger.info(f"Selected document for visualization based on name match: {selected_doc_name}")
                            break
                    
                    # If no specific file was mentioned, use the most recently added one
                    if not selected_doc_name:
                        selected_doc_name = list(visualizable_docs.keys())[0]
                        # For multiple files, try to find the most recent one
                        if len(visualizable_docs) > 1:
                            selected_doc_name = max(
                                visualizable_docs.keys(),
                                key=lambda k: visualizable_docs[k].get("timestamp", 0)
                            )
                            logger.info(f"Selected most recent document for visualization: {selected_doc_name}")
                    
                    file_path = visualizable_docs[selected_doc_name]['path']
                    
                    # Try to determine chart type from query
                    chart_type = None
                    chart_types = {
                        'line': ['line', 'trend', 'over time', 'timeseries'],
                        'bar': ['bar', 'column', 'compare', 'comparison'],
                        'pie': ['pie', 'proportion', 'percentage', 'share'],
                        'scatter': ['scatter', 'correlation', 'relationship'],
                        'histogram': ['histogram', 'distribution', 'frequency'],
                        'box': ['box', 'boxplot', 'range', 'outliers'],
                        'heatmap': ['heatmap', 'heat map', 'matrix']
                    }
                    
                    for ctype, keywords in chart_types.items():
                        if any(kw in user_query.lower() for kw in keywords):
                            chart_type = ctype
                            logger.info(f"Detected chart type from query: {chart_type}")
                            break
                    
                    # First try with DynamicVisualizationTool for LLM-generated charts
                    dynamic_viz_tool = tool_registry.create_tool_instance("DynamicVisualizationTool")
                    if dynamic_viz_tool:
                        # Execute dynamic visualization
                        viz_params = {
                            "query": user_query,
                            "file_path": file_path,
                            "chart_type": chart_type,
                            "max_rows": 5000  # Reasonable limit
                        }
                        
                        logger.info(f"Executing dynamic visualization with parameters: {viz_params}")
                        try:
                            # Load the data first
                            try:
                                if file_path.lower().endswith('.csv'):
                                    df = pd.read_csv(file_path)
                                elif file_path.lower().endswith(('.xlsx', '.xls')):
                                    df = pd.read_excel(file_path)
                                else:
                                    raise ValueError(f"Unsupported file format: {file_path}")
                                logger.info(f"Successfully loaded data from {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
                            except Exception as load_err:
                                logger.error(f"Error loading data for visualization: {str(load_err)}")
                                raise ValueError(f"Error loading data: {str(load_err)}")
                            
                            # Create visualization with safeguards
                            viz_result = await dynamic_viz_tool._create_visualization(
                                df=df,
                                query=user_query,
                                chart_type=chart_type
                            )
                            
                            if "error" not in viz_result and "visualization_data" in viz_result:
                                # Quick validation of image data
                                viz_data = viz_result["visualization_data"] 
                                if not viz_data or len(viz_data) < 100:  # Very small, likely invalid
                                    logger.warning(f"Visualization data is suspiciously small: {len(viz_data) if viz_data else 0} bytes")
                                    raise ValueError("Generated visualization appears to be invalid or empty")
                                
                                # All good!
                                logger.info(f"Successfully created visualization, size: {len(viz_data)} bytes")
                                results["visualization"] = viz_result
                                # Add tools used
                                if "tools_used" not in results:
                                    results["tools_used"] = []
                                results["tools_used"].append("DynamicVisualizationTool")
                            else:
                                # Fallback to standard visualization tool
                                error_msg = viz_result.get("error", "Unknown error")
                                logger.warning(f"Dynamic visualization failed: {error_msg}. Falling back to standard tool.")
                                viz_tool = tool_registry.create_tool_instance("DataVisualizationTool")
                                if viz_tool:
                                    std_viz_params = {
                                        "query": user_query,
                                        "file_path": file_path,
                                        "chart_type": chart_type or "auto"
                                    }
                                    std_viz_result = viz_tool.execute(**std_viz_params)
                                    results["visualization"] = std_viz_result
                                    
                                    # Add tools used
                                    if "tools_used" not in results:
                                        results["tools_used"] = []
                                    results["tools_used"].append("DataVisualizationTool")
                        except Exception as e:
                            logger.error(f"Error executing dynamic visualization tool: {str(e)}", exc_info=True)
                            # Try standard visualization as fallback
                            viz_tool = tool_registry.create_tool_instance("DataVisualizationTool")
                            if viz_tool:
                                std_viz_params = {
                                    "query": user_query,
                                    "file_path": file_path,
                                    "chart_type": chart_type or "auto"
                                }
                                std_viz_result = viz_tool.execute(**std_viz_params)
                                results["visualization"] = std_viz_result
                                
                                # Add tools used
                                if "tools_used" not in results:
                                    results["tools_used"] = []
                                results["tools_used"].append("DataVisualizationTool")
                    else:
                        logger.error("Failed to create DynamicVisualizationTool instance")
                        # Fall back to regular visualization
                        viz_tool = tool_registry.create_tool_instance("DataVisualizationTool")
                        if viz_tool:
                            std_viz_params = {
                                "query": user_query,
                                "file_path": file_path,
                                "chart_type": chart_type or "auto"
                            }
                            std_viz_result = viz_tool.execute(**std_viz_params)
                            results["visualization"] = std_viz_result
                            
                            # Add tools used
                            if "tools_used" not in results:
                                results["tools_used"] = []
                            results["tools_used"].append("DataVisualizationTool")
            
            # Handle data analysis intents
            if primary_intent in ["data_analysis", "trend_analysis", "comparison"]:
                # Check if we have active documents
                if self.active_documents:
                    # For comparison intent, try to compare two documents if available
                    if primary_intent == "comparison" and len(self.active_documents) >= 2:
                        # Get the two most recent documents
                        sorted_docs = sorted(self.active_documents.values(), 
                                           key=lambda x: x["timestamp"], reverse=True)
                        doc1 = sorted_docs[0]
                        doc2 = sorted_docs[1]
                        
                        logger.info(f"Comparing documents: {os.path.basename(doc1['path'])} and {os.path.basename(doc2['path'])}")
                        
                        # Read the content of both documents
                        try:
                            # Use our helper method to get content
                            doc1_content = self._get_document_content(doc1)
                            doc2_content = self._get_document_content(doc2)
                            
                            # Generate comparison using LLM
                            if doc1_content and doc2_content:
                                doc1_name = os.path.basename(doc1['path'])
                                doc2_name = os.path.basename(doc2['path'])
                                
                                # Extract document types
                                doc1_type = doc1['processed_data'].get('metadata', {}).get('type', 'unknown')
                                doc2_type = doc2['processed_data'].get('metadata', {}).get('type', 'unknown')
                                
                                # Truncate content if too long
                                max_len = 10000
                                if len(doc1_content) > max_len:
                                    doc1_content = doc1_content[:max_len] + "... [truncated]"
                                if len(doc2_content) > max_len:
                                    doc2_content = doc2_content[:max_len] + "... [truncated]"
                                
                                # Create comparison prompt
                                comparison_prompt = f"""
                                Please compare these two financial documents and generate a detailed comparison:
                                
                                DOCUMENT 1 ({doc1_name}, type: {doc1_type}):
                                {doc1_content}
                                
                                DOCUMENT 2 ({doc2_name}, type: {doc2_type}):
                                {doc2_content}
                                
                                Generate a structured comparison highlighting:
                                1. Key similarities between the documents
                                2. Important differences and discrepancies
                                3. Financial metrics present in both and how they differ
                                4. Overall assessment of what these documents together reveal
                                
                                Focus particularly on financial insights, trends, and metrics.
                                """
                                
                                comparison_text, provider = await llm_manager.generate_response(comparison_prompt, temperature=0.2)
                                
                                # Add to results
                                results["comparison"] = {
                                    "summary": comparison_text,
                                    "document1": doc1_name,
                                    "document2": doc2_name,
                                    "provider": provider
                                }
                                
                                logger.info(f"Generated document comparison using {provider}")
                        except Exception as e:
                            logger.error(f"Error comparing documents: {str(e)}")
                    
                    # Continue with regular analysis if comparison wasn't performed or for other intents
                    if "comparison" not in results:
                        # Use the most recently added document
                        latest_file = max(self.active_documents.values(), 
                                        key=lambda x: x["timestamp"])
                        file_path = latest_file["path"]
                        
                        # Determine analysis type based on intent
                        analysis_type = "summary"
                        if primary_intent == "trend_analysis":
                            analysis_type = "trend"
                        elif primary_intent == "comparison":
                            analysis_type = "comparison"
                        
                        # Extract potential metrics mentioned in the query
                        metrics = intent_analysis.get("metrics_mentioned", [])
                        
                        # Execute financial analysis
                        analysis_tool = tool_registry.create_tool_instance("FinancialAnalysisTool")
                        if analysis_tool:
                            try:
                                results["analysis"] = analysis_tool.execute(
                                    data_source=file_path,
                                    analysis_type=analysis_type,
                                    metrics=metrics
                                )
                            except Exception as e:
                                logger.error(f"Error executing financial analysis tool: {str(e)}")
                        else:
                            logger.error("Failed to create FinancialAnalysisTool tool instance")
            
            # Handle summarization intent
            if primary_intent == "summarization" or "TextSummarizationTool" in tools_needed:
                # Determine if we should summarize a document or text in the query
                text_to_summarize = ""
                
                # First check if we need to summarize web content
                if "web_search" in results and results["web_search"].get("text_content"):
                    text_to_summarize = results["web_search"]["text_content"]
                # Otherwise check active documents
                elif self.active_documents:
                    # Use the most recently added document
                    latest_file = max(self.active_documents.values(), 
                                     key=lambda x: x["timestamp"])
                    file_path = latest_file["path"]
                    
                    # Read the file content first
                    try:
                        # Try to get content from processed data first
                        file_content = latest_file['processed_data'].get('content', '')
                        
                        # If no content in processed data, try to read the file
                        if not file_content:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                                
                        summarization_tool = tool_registry.create_tool_instance("TextSummarizationTool")
                        if summarization_tool:
                            try:
                                results["summarization"] = await summarization_tool.execute(
                                    text=file_content,
                                    summary_type="detailed",
                                    max_length=500
                                )
                            except Exception as e:
                                logger.error(f"Error executing text summarization tool: {str(e)}")
                        else:
                            logger.error("Failed to create TextSummarizationTool tool instance")
                    except Exception as e:
                        logger.error(f"Error reading file for summarization: {str(e)}")
            
            # Build the final response prompt
            response_prompt = f"""User query: {user_query}\n\n"""
            
            # Include chat history for context if available
            if chat_history and len(chat_history) > 0:
                response_prompt += "Previous conversation:\n"
                # Add the last 5 messages (or fewer if not available)
                history_to_include = chat_history[-5:] if len(chat_history) > 5 else chat_history
                for msg in history_to_include:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    response_prompt += f"{role.capitalize()}: {content}\n"
                response_prompt += "\n"
            
            # Include search API results if present
            if "search_api" in results:
                search_result = results["search_api"]
                if "summary" in search_result and search_result["summary"]:
                    response_prompt += f"\nWeb Search Results:\n{search_result['summary']}\n"
                
                # Add sources information
                if "sources" in search_result and search_result["sources"]:
                    response_prompt += "\nSources:\n"
                    for i, source in enumerate(search_result["sources"][:3]):  # Limit to first 3 sources
                        response_prompt += f"- {source.get('title', 'Source ' + str(i+1))}: {source.get('url', '')}\n"

            # Include any visualization results
            if "visualization" in results:
                viz_result = results["visualization"]
                if "insights" in viz_result and viz_result["insights"]:
                    response_prompt += f"\nVisualization insights:\n{viz_result['insights']}\n"
            
            # Include any CSV analysis results
            if "csv_analysis" in results:
                analysis_result = results["csv_analysis"]
                if "summary" in analysis_result and analysis_result["summary"]:
                    response_prompt += f"\nCSV Analysis Summary:\n{analysis_result['summary']}\n"
                    
                # Optionally include detailed results if not too long
                if "results" in analysis_result and isinstance(analysis_result["results"], str):
                    if len(analysis_result["results"]) < 1000:  # Limit length
                        response_prompt += f"\nDetailed Analysis:\n{analysis_result['results']}\n"
                elif "results" in analysis_result:
                    try:
                        results_str = json.dumps(analysis_result["results"], indent=2)
                        if len(results_str) < 1000:
                            response_prompt += f"\nDetailed Analysis:\n{results_str}\n"
                    except TypeError:
                        pass # Ignore if not JSON serializable
            
            # Include any comparison results
            if "comparison" in results:
                comparison_result = results["comparison"]
                doc1 = comparison_result.get("document1", "Document 1")
                doc2 = comparison_result.get("document2", "Document 2")
                if "summary" in comparison_result and comparison_result["summary"]:
                    response_prompt += f"\nComparison of {doc1} and {doc2}:\n{comparison_result['summary']}\n"
            
            # Include any (other) analysis results (e.g., from FinancialAnalysisTool)
            if "analysis" in results:
                analysis_result = results["analysis"]
                if "summary" in analysis_result and analysis_result["summary"]:
                    response_prompt += f"\nAnalysis results:\n{analysis_result['summary']}\n"
            
            # Include any web search results
            if "web_search" in results:
                web_result = results["web_search"]
                if "summary" in web_result and web_result["summary"]:
                    response_prompt += f"\nWeb content summary:\n{web_result['summary']}\n"
            
            # Include any summarization results
            if "summarization" in results:
                summarization_result = results["summarization"]
                if "summary" in summarization_result and summarization_result["summary"]:
                    response_prompt += f"\nDocument summary:\n{summarization_result['summary']}\n"
            
            # Include information about active documents
            active_docs_info = self.get_active_documents()
            if active_docs_info:
                doc_description = "\nActive documents:\n"
                for name, details in active_docs_info.items():
                    doc_description += f"- {name} ({details['type']})\n"
                response_prompt += doc_description
            
            response_prompt += """
            Please provide a helpful, clear, and informative response to the user based on the above information.
            Focus on answering their specific query and providing valuable insights from the data.
            If web search results are included, make sure to reference the sources of information.
            """
            
            # Generate response with LLM manager
            try:
                # Try with chat history first
                response_text, provider = await llm_manager.generate_response(
                    response_prompt, 
                    temperature=0.7, 
                    chat_history=chat_history
                )
            except TypeError as e:
                # Fall back to calling without chat_history if needed
                logger.warning(f"Error using chat_history with LLM: {str(e)}. Falling back to basic prompt.")
                response_text, provider = await llm_manager.generate_response(
                    response_prompt,
                    temperature=0.7
                )
            
            logger.info(f"Generated response using provider: {provider}")
            
            # Translate response if necessary (if language is not English)
            if language != "en":
                language_tool = tool_registry.create_tool_instance("LanguageTool")
                if language_tool:
                    try:
                        translation_result = await language_tool.execute(
                            text=response_text,
                            source_language="en",
                            target_language=language
                        )
                        response_text = translation_result["translated_text"]
                    except Exception as e:
                        logger.error(f"Error executing language translation tool: {str(e)}")
                else:
                    logger.error("Failed to create LanguageTool tool instance")
            
            # Build the final response object
            response = {
                "text": response_text,
                "active_documents": active_docs_info,
                "provider_used": provider
            }
            
            # Add visualization data if present
            if "visualization" in results:
                response["visualization"] = results["visualization"]
            
            # Add search API results if present to final response
            if "search_api" in results:
                response["search_api"] = {
                    "summary": results["search_api"].get("summary", ""),
                    "sources": results["search_api"].get("sources", []),
                    "query": results["search_api"].get("query", "")
                }
            
            # Add tools used from results
            if "tools_used" in results:
                response["tools_used"] = results["tools_used"]
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "text": f"I encountered an error while processing your request: {str(e)}. Please try again or rephrase your query.",
                "error": str(e),
                "active_documents": self.get_active_documents()
            }
    
    async def _process_uploaded_file(self, file_path: str) -> Dict[str, Any]:
        """Process an uploaded file and store the results."""
        try:
            # Try to get FileProcessor first
            file_processor = tool_registry.create_tool_instance("FileProcessor")
            
            # Fall back to DummyTool if FileProcessor is not available
            if not file_processor:
                logger.warning("FileProcessor not available, falling back to DummyTool")
                dummy_tool = tool_registry.create_tool_instance("DummyTool")
                
                if dummy_tool:
                    # Read file content as text
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {str(e)}")
                        file_content = f"Error reading file: {str(e)}"
                    
                    # Use DummyTool to process the file
                    result = dummy_tool.execute(text=file_content)
                    
                    # Convert to expected format
                    result = {
                        "content": result.get("result", ""),
                        "metadata": {
                            "type": os.path.splitext(file_path)[1],
                            "stats": result.get("stats", {})
                        },
                        "tables": []
                    }
                else:
                    logger.error("DummyTool not available either")
                    return {"error": "No tools available for processing", "content": "", "metadata": {"type": "unknown"}, "tables": []}
            else:
                # Use FileProcessor if available
                result = file_processor.execute(file_path=file_path)
            
            # Store the processed file data
            file_name = os.path.basename(file_path)
            self.active_documents[file_name] = {
                "path": file_path,
                "processed_data": result,
                "timestamp": time.time()
            }
            
            return result
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {"error": str(e), "content": "", "metadata": {"type": "unknown"}, "tables": []}
    
    def get_active_documents(self):
        """Get information about active documents."""
        documents_info = {}
        for doc_name, doc_info in self.active_documents.items():
            doc_path = doc_info.get('path', '')
            processed_data = doc_info.get('processed_data', {})
            doc_type = processed_data.get('metadata', {}).get('type', 'unknown')
            doc_size = len(processed_data.get('content', ''))
            has_tables = len(processed_data.get('tables', [])) > 0
            
            documents_info[doc_name] = {
                'path': doc_path,
                'type': doc_type,
                'size': doc_size,
                'has_tables': has_tables
            }
        
        return documents_info
        
    def get_available_tools(self):
        """Get the list of available tools for the agent."""
        return self.available_tools 

    def clear_documents(self):
        """Clear all active documents from the agent."""
        self.active_documents = {}
        logger.info("All documents cleared from the agent")

    def _get_document_content(self, doc_info: Dict[str, Any]) -> str:
        """
        Safely extract content from a document regardless of its type.
        
        Args:
            doc_info: The document info dictionary from active_documents
            
        Returns:
            The document content as a string
        """
        # Try to get content from processed data first
        if 'processed_data' in doc_info:
            # Get content directly if available
            if 'content' in doc_info['processed_data']:
                return doc_info['processed_data']['content']
                
            # If it's a tabular document, format the tables
            if doc_info['processed_data'].get('content_type') == 'tabular' and 'tables' in doc_info['processed_data']:
                tables = doc_info['processed_data']['tables']
                if tables:
                    # For tabular data, create a text representation
                    try:
                        import pandas as pd
                        df = pd.DataFrame(tables)
                        return df.to_string(index=False)
                    except:
                        # Fallback if pandas fails
                        return str(tables)
        
        # Fallback: try to read the file directly
        try:
            if 'path' in doc_info:
                file_path = doc_info['path']
                doc_type = doc_info.get('processed_data', {}).get('metadata', {}).get('type', '').lower()
                
                # For binary files like PDF, we need specialized handling
                if doc_type in ['pdf', 'docx', 'doc']:
                    # Create a FileProcessor to handle the file
                    file_processor = tool_registry.create_tool_instance("FileProcessor")
                    if file_processor:
                        result = file_processor.execute(file_path=file_path)
                        return result.get('content', f"Could not extract content from {file_path}")
                
                # For text files, simply read them
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except UnicodeDecodeError:
                    # If it fails with UTF-8, it's probably a binary file
                    return f"Binary file: {file_path}"
                except Exception as e:
                    return f"Error reading file: {str(e)}"
        except Exception as e:
            return f"Error extracting document content: {str(e)}"
            
        # If all else fails, return an empty string
        return "" 