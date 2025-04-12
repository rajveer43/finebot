"""
Utility module to ensure all tools are properly registered.
Import this module early in the application startup to make sure
all tools are properly registered with the tool registry.
"""

import logging
from src.tools.tool_registry import tool_registry, Tool

logger = logging.getLogger(__name__)

# Define the known tools and their module paths
TOOL_MODULES = {
    "FileProcessor": "src.tools.file_processor",
    "FinancialAnalysisTool": "src.tools.financial_analysis_tool",
    "TextSummarizationTool": "src.tools.text_summarization",
    "LanguageTool": "src.tools.language_tool",
    "WebSearchTool": "src.tools.web_search_tool", 
    "DataVisualizationTool": "src.tools.data_visualization_tool",
    "CSVAnalyzerTool": "src.tools.csv_analyzer_tool",
    "DynamicVisualizationTool": "src.tools.dynamic_visualization_tool",
    "SearchAPITool": "src.tools.search_api_tool",
    "DummyTool": "src.tools.dummy_tool"
}

# Define a fallback dummy tool that can be created directly if imports fail
class FallbackDummyTool(Tool):
    """A simple fallback dummy tool for testing."""
    
    name = "DummyTool"
    description = "A simple tool for testing that doesn't depend on external libraries"
    
    def execute(self, **kwargs):
        """Process text input."""
        text = kwargs.get("text", "")
        return {
            "result": f"Processed: {text}",
            "stats": {"length": len(text)}
        }

def register_tools():
    """Register all tools manually if they aren't already registered."""
    registered_tools = 0
    
    # Always register the built-in FallbackDummyTool to ensure at least one tool is available
    try:
        tool_registry.register(FallbackDummyTool)
        logger.info("✅ Registered built-in FallbackDummyTool")
        registered_tools += 1
    except Exception as e:
        logger.error(f"❌ Failed to register built-in FallbackDummyTool: {str(e)}")
    
    # Try to import the external dummy tool if available
    try:
        from src.tools.dummy_tool import DummyTool
        logger.info("✅ Imported DummyTool")
    except Exception as e:
        logger.warning(f"⚠️ Error importing DummyTool: {str(e)}")
    
    # Check which tools need to be registered
    all_tools = tool_registry.get_all_tools()
    registered_names = [tool["name"] for tool in all_tools]
    
    for tool_name, module_path in TOOL_MODULES.items():
        if tool_name not in registered_names or tool_name == "DummyTool": # Always try to register DummyTool
            try:
                # Try to import the module
                module = __import__(module_path, fromlist=['*'])
                
                # Get the tool class
                tool_class = getattr(module, tool_name)
                
                # Register the tool
                tool_registry.register(tool_class)
                logger.info(f"✅ Manually registered {tool_name}")
                registered_tools += 1
            except ImportError as e:
                logger.warning(f"⚠️ Could not import module {module_path}: {str(e)}")
            except AttributeError as e:
                logger.warning(f"⚠️ Could not find tool class {tool_name} in module {module_path}: {str(e)}")
            except Exception as e:
                logger.warning(f"⚠️ Error registering {tool_name}: {str(e)}")
    
    logger.info(f"Registered {registered_tools} tools manually.")
    return registered_tools

# Register tools when this module is imported
register_tools() 