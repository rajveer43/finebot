import logging
import importlib
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # Import the tool registry
    logger.info("Importing tool_registry...")
    from src.tools.tool_registry import tool_registry
    
    # Import all tools to ensure they are registered
    logger.info("Importing tool modules...")
    try:
        import src.tools.file_processor
        logger.info("✅ Imported file_processor")
    except Exception as e:
        logger.error(f"❌ Error importing file_processor: {str(e)}")
    
    try:
        import src.tools.financial_analysis_tool
        logger.info("✅ Imported financial_analysis_tool")
    except Exception as e:
        logger.error(f"❌ Error importing financial_analysis_tool: {str(e)}")
    
    try:
        import src.tools.text_summarization
        logger.info("✅ Imported text_summarization")
    except Exception as e:
        logger.error(f"❌ Error importing text_summarization: {str(e)}")
    
    try:
        import src.tools.language_tool
        logger.info("✅ Imported language_tool")
    except Exception as e:
        logger.error(f"❌ Error importing language_tool: {str(e)}")
    
    try:
        import src.tools.web_search_tool
        logger.info("✅ Imported web_search_tool")
    except Exception as e:
        logger.error(f"❌ Error importing web_search_tool: {str(e)}")
    
    try:
        import src.tools.data_visualization_tool
        logger.info("✅ Imported data_visualization_tool")
    except Exception as e:
        logger.error(f"❌ Error importing data_visualization_tool: {str(e)}")
    
    # Print all registered tools
    all_tools = tool_registry.get_all_tools()
    logger.info(f"Number of registered tools: {len(all_tools)}")
    
    for i, tool in enumerate(all_tools):
        logger.info(f"Tool {i+1}: {tool['name']} - {tool['description']}")
    
    # List of expected tool names
    expected_tools = [
        "FileProcessor",
        "FinancialAnalysisTool",
        "TextSummarizationTool",
        "LanguageTool",
        "WebSearchTool",
        "DataVisualizationTool"
    ]
    
    # Check for missing tools
    registered_names = [tool["name"] for tool in all_tools]
    for expected_tool in expected_tools:
        if expected_tool not in registered_names:
            logger.error(f"MISSING TOOL: {expected_tool} is not registered!")
        else:
            logger.info(f"✅ {expected_tool} is properly registered")
    
    # Try to create instances of each tool
    for tool_name in registered_names:
        tool_instance = tool_registry.create_tool_instance(tool_name)
        if tool_instance:
            logger.info(f"✅ Successfully created instance of {tool_name}")
        else:
            logger.error(f"❌ Failed to create instance of {tool_name}")
    
    # Test the FinancialAgent initialization
    try:
        from src.agents.financial_agent import FinancialAgent
        agent = FinancialAgent()
        logger.info(f"✅ FinancialAgent initialized successfully")
        logger.info(f"Available tools in agent: {', '.join(agent.available_tools)}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize FinancialAgent: {str(e)}")

if __name__ == "__main__":
    main() 