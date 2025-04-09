import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
import os
import json
import re
import time

from src.llm.gemini_integration import gemini
from src.tools.tool_registry import tool_registry
from src.tools.file_processor import FileProcessor
from src.tools.financial_analysis_tool import FinancialAnalysisTool
from src.tools.text_summarization import TextSummarizationTool
from src.tools.language_tool import LanguageTool
from src.tools.web_search_tool import WebSearchTool
from src.config.config import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialAgent:
    """
    Agentic AI coordinator for the financial chatbot.
    Manages intent detection, tool selection, and response generation.
    """
    
    def __init__(self):
        """Initialize the Financial Agent."""
        # Set default language
        self.current_language = DEFAULT_LANGUAGE
        
        # Initialize context storage
        self.chat_history = []
        self.active_documents = {}  # Keeps track of processed documents
        
        # Set the system prompt for the LLM
        self.system_prompt = """
        You are a Financial Intelligence Assistant with expertise in:
        1. Analyzing financial data and documents
        2. Extracting insights from financial reports
        3. Performing calculations on financial metrics
        4. Generating summaries of financial content
        5. Translating financial information across languages
        6. Searching and extracting content from web URLs related to finance
        
        You can understand and process:
        - Financial statements (income statements, balance sheets, cash flow statements)
        - Financial reports and analysis
        - Economic data and indicators
        - Market trends and stock information
        - Banking and investment documents
        
        Your objective is to help users understand their financial information and provide valuable insights.
        
        When responding to queries:
        - Prioritize accuracy and clarity in your analysis
        - Provide concise and structured information
        - Highlight key patterns or insights in the data
        - Use your tools when appropriate to analyze documents or process specific requests
        
        You have several tools at your disposal and will select the appropriate one based on the user's needs.
        """
        
        gemini.set_system_prompt(self.system_prompt)
        
    async def process_query(self, user_query: str, 
                           uploaded_files: Optional[List[str]] = None,
                           language: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query, determine intent, execute tools, and generate a response.
        
        Args:
            user_query: The user's query text
            uploaded_files: List of paths to any files uploaded with this query
            language: Optional language code to override detection
            
        Returns:
            Dictionary with the response and additional context
        """
        try:
            # Track query in history
            self.chat_history.append({"role": "user", "content": user_query})
            
            # Handle language detection or override
            if language and language in SUPPORTED_LANGUAGES:
                self.current_language = language
            else:
                # Detect language if not explicitly provided
                lang_tool = tool_registry.create_tool_instance("LanguageTool")
                lang_result = await lang_tool.execute(text=user_query, action="detect")
                
                if "error" not in lang_result:
                    detected_lang = lang_result.get("detected_language", DEFAULT_LANGUAGE)
                    
                    # Only switch if the detected language is supported
                    if detected_lang in SUPPORTED_LANGUAGES:
                        self.current_language = detected_lang
            
            # Process any uploaded files
            if uploaded_files:
                for file_path in uploaded_files:
                    await self._process_uploaded_file(file_path)
            
            # Analyze intent to determine which tools to use
            intent_analysis = await gemini.analyze_intent(user_query)
            logger.info(f"Intent analysis: {intent_analysis}")
            
            # Execute tools based on intent
            tool_results = await self._execute_tools_for_intent(user_query, intent_analysis)
            
            # Generate final response
            response = await self._generate_response(user_query, intent_analysis, tool_results)
            
            # Track response in history
            self.chat_history.append({"role": "assistant", "content": response["text"]})
            
            # Ensure response is in the user's language
            if self.current_language != DEFAULT_LANGUAGE:
                response = await self._translate_response(response, self.current_language)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            error_message = f"I apologize, but I encountered an error while processing your request: {str(e)}"
            
            self.chat_history.append({"role": "assistant", "content": error_message})
            
            return {
                "text": error_message,
                "error": str(e),
                "success": False
            }
    
    async def _process_uploaded_file(self, file_path: str) -> Dict[str, Any]:
        """Process an uploaded file and store the results."""
        file_processor = tool_registry.create_tool_instance("FileProcessor")
        result = file_processor.execute(file_path=file_path)
        
        # Store the processed file data
        file_name = os.path.basename(file_path)
        self.active_documents[file_name] = {
            "path": file_path,
            "processed_data": result,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        return result
    
    async def _execute_tools_for_intent(self, query: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute relevant tools based on detected intent.
        
        Args:
            query: The user's query
            intent: The intent analysis results
            
        Returns:
            Dictionary of tool execution results
        """
        results = {}
        
        primary_intent = intent.get("intent", "")
        tools_needed = intent.get("tools_needed", [])
        
        # Log the execution plan
        logger.info(f"Executing tools for intent: {primary_intent}, tools: {tools_needed}")
        
        # Check for URL in query to determine if web search is needed
        url_pattern = re.compile(r'https?://\S+')
        urls = re.findall(url_pattern, query)
        
        # Handle web search if URLs are present or web search is explicitly needed
        if urls or "WebSearchTool" in tools_needed or primary_intent == "extraction":
            web_search_tool = tool_registry.create_tool_instance("WebSearchTool")
            
            for url in urls:
                # Determine if we should download linked documents
                download_docs = "download" in query.lower() or "get linked" in query.lower()
                
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
        
        # Handle data analysis intents
        if primary_intent in ["data_analysis", "trend_analysis", "comparison"]:
            # Check if we have active documents
            if self.active_documents:
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
                metrics = intent.get("metrics_mentioned", [])
                
                # Execute financial analysis
                analysis_tool = tool_registry.create_tool_instance("FinancialAnalysisTool")
                results["analysis"] = analysis_tool.execute(
                    data_source=file_path,
                    analysis_type=analysis_type,
                    metrics=metrics
                )
        
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
                
                if "text_content" in latest_file["processed_data"]:
                    text_to_summarize = latest_file["processed_data"]["text_content"]
            
            if text_to_summarize:
                # Determine summary type from query
                summary_type = "detailed"
                if re.search(r"brief|short|quick", query, re.IGNORECASE):
                    summary_type = "brief"
                elif re.search(r"bullet|point|list", query, re.IGNORECASE):
                    summary_type = "bullet_points"
                elif re.search(r"executive|management", query, re.IGNORECASE):
                    summary_type = "executive_summary"
                
                # Execute summarization
                summarization_tool = tool_registry.create_tool_instance("TextSummarizationTool")
                results["summarization"] = await summarization_tool.execute(
                    text=text_to_summarize,
                    summary_type=summary_type
                )
        
        # Check if language translation is needed
        if "LanguageTool" in tools_needed:
            if re.search(r"translate|translation", query, re.IGNORECASE):
                # Extract target language from query
                target_lang = DEFAULT_LANGUAGE
                
                lang_matches = re.search(r"to (spanish|french|german|chinese|japanese|english|italian|arabic)", 
                                       query, re.IGNORECASE)
                if lang_matches:
                    language_name = lang_matches.group(1).lower()
                    
                    # Map language name to code
                    lang_map = {
                        "english": "en",
                        "spanish": "es",
                        "french": "fr",
                        "german": "de",
                        "chinese": "zh",
                        "japanese": "ja",
                        "italian": "it",
                        "arabic": "ar"
                    }
                    
                    if language_name in lang_map:
                        target_lang = lang_map[language_name]
                
                # Find text to translate - check web search, documents, or query itself
                text_to_translate = query
                
                if "web_search" in results and results["web_search"].get("text_content"):
                    text_to_translate = results["web_search"]["text_content"]
                elif self.active_documents:
                    latest_file = max(self.active_documents.values(), 
                                     key=lambda x: x["timestamp"])
                    if "text_content" in latest_file["processed_data"]:
                        text_to_translate = latest_file["processed_data"]["text_content"]
                
                # Execute translation
                lang_tool = tool_registry.create_tool_instance("LanguageTool")
                results["translation"] = await lang_tool.execute(
                    text=text_to_translate,
                    action="translate",
                    target_language=target_lang
                )
        
        return results
    
    async def _generate_response(self, query: str, intent: Dict[str, Any], 
                              tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a final response based on query, intent and tool results.
        
        Args:
            query: The user's query
            intent: The intent analysis 
            tool_results: Results from tool executions
            
        Returns:
            Dictionary with the final response
        """
        # Create context for response generation
        context = {
            "query": query,
            "intent": intent,
            "tool_results": tool_results
        }
        
        # Prompt the LLM to generate a response
        prompt = f"""
        Please review the user's query and the results from our analysis tools,
        and generate a helpful response that addresses their financial question.
        
        User Query: {query}
        
        Detected Intent: {intent.get('intent', 'unknown')}
        Confidence: {intent.get('confidence', 0)}
        
        Tool Results: {json.dumps(tool_results, indent=2)}
        
        Generate a clear, concise, and helpful response that:
        1. Directly addresses the user's query
        2. Incorporates relevant insights from the tool results
        3. Provides actionable financial information
        4. Uses professional but accessible language
        """
        
        # Generate the response
        response_text = await gemini.generate_response(prompt)
        
        return {
            "text": response_text,
            "intent": intent.get("intent", "unknown"),
            "tools_used": list(tool_results.keys()),
            "success": True
        }
    
    async def _translate_response(self, response: Dict[str, Any], target_language: str) -> Dict[str, Any]:
        """
        Translate the response to the target language.
        
        Args:
            response: The response to translate
            target_language: The target language code
            
        Returns:
            The translated response
        """
        if target_language == DEFAULT_LANGUAGE:
            return response
        
        lang_tool = tool_registry.create_tool_instance("LanguageTool")
        translation_result = await lang_tool.execute(
            text=response["text"],
            action="translate",
            target_language=target_language
        )
        
        if "error" not in translation_result:
            response["text"] = translation_result["translated_text"]
            response["language"] = target_language
        
        return response
    
    def clear_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []
        logger.info("Chat history cleared")
    
    def clear_documents(self) -> None:
        """Clear the active documents."""
        self.active_documents = {}
        logger.info("All documents cleared")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the chat history."""
        return self.chat_history
    
    def get_active_documents(self) -> Dict[str, Any]:
        """Get information about the active documents."""
        active_docs = {}
        for name, doc in self.active_documents.items():
            active_docs[name] = {
                "path": doc["path"],
                "type": doc["processed_data"].get("metadata", {}).get("type", "unknown"),
                "size": len(doc["processed_data"].get("content", "")),
                "has_tables": len(doc["processed_data"].get("tables", [])) > 0
            }
        return active_docs 