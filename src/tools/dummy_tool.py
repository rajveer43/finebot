"""
A simple dummy tool that doesn't have external dependencies.
This can be used for testing when other tools have import errors.
"""

import logging
from typing import Dict, Any
from src.tools.tool_registry import Tool, tool_registry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyTool(Tool):
    """A simple dummy tool for testing."""
    
    name = "DummyTool"
    description = "A simple tool for testing that doesn't depend on external libraries"
    
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to process"
            },
            "repeat_count": {
                "type": "integer",
                "description": "Number of times to repeat the text",
                "default": 1
            }
        },
        "required": ["text"]
    }
    
    output_schema = {
        "type": "object",
        "properties": {
            "result": {
                "type": "string",
                "description": "Processed result"
            },
            "stats": {
                "type": "object",
                "description": "Statistics about the processed text"
            }
        }
    }
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool functionality.
        
        Args:
            text: Text to process
            repeat_count: Number of times to repeat the text (default: 1)
            
        Returns:
            Dictionary with the processed result and statistics
        """
        # Extract parameters
        text = kwargs.get("text", "")
        repeat_count = kwargs.get("repeat_count", 1)
        
        # Process the text
        result = text * repeat_count
        
        # Calculate some statistics
        stats = {
            "original_length": len(text),
            "result_length": len(result),
            "repeat_count": repeat_count,
            "word_count": len(text.split()),
            "uppercase_count": sum(1 for c in text if c.isupper()),
            "lowercase_count": sum(1 for c in text if c.islower()),
            "digit_count": sum(1 for c in text if c.isdigit())
        }
        
        # Log the operation
        logger.info(f"DummyTool processed text of length {len(text)}, repeated {repeat_count} times")
        
        # Return the result
        return {
            "result": result,
            "stats": stats
        }

# Register the tool
tool_registry.register(DummyTool) 