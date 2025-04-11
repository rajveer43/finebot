from typing import Dict, Any, List, Type, Callable, Optional
import inspect
import json

class Tool:
    """Base class for all tools in the financial chatbot."""
    
    name: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = {}
    output_schema: Dict[str, Any] = {}
    
    def __init__(self):
        if not self.name:
            self.name = self.__class__.__name__
    
    def execute(self, **kwargs) -> Any:
        """
        Execute the tool's functionality.
        This method should be overridden by all tool implementations.
        
        Args:
            **kwargs: Keyword arguments for the tool execution
            
        Returns:
            Any: The result of the tool execution
        """
        raise NotImplementedError("Tool must implement execute method")
    
    @classmethod
    def get_input_schema(cls) -> Dict[str, Any]:
        """Return the input schema for this tool."""
        return cls.input_schema
    
    @classmethod
    def get_output_schema(cls) -> Dict[str, Any]:
        """Return the output schema for this tool."""
        return cls.output_schema
    
    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Get tool metadata for registration."""
        return {
            "name": cls.name,
            "description": cls.description,
            "input_schema": cls.input_schema,
            "output_schema": cls.output_schema
        }


class ToolRegistry:
    """Registry for all available tools in the system."""
    
    def __init__(self):
        self.tools: Dict[str, Type[Tool]] = {}
        
    def register(self, tool_class: Type[Tool]) -> None:
        """
        Register a tool class with the registry.
        
        Args:
            tool_class: The tool class to register
        """
        self.tools[tool_class.name] = tool_class
        
    def get_tool(self, tool_name: str) -> Optional[Type[Tool]]:
        """
        Get a tool class by name.
        
        Args:
            tool_name: The name of the tool to get
            
        Returns:
            The tool class or None if not found
        """
        return self.tools.get(tool_name)
    
    def create_tool_instance(self, tool_name: str) -> Optional[Tool]:
        """
        Create an instance of a tool by name.
        
        Args:
            tool_name: The name of the tool to instantiate
            
        Returns:
            An instance of the tool or None if not found
        """
        tool_class = self.get_tool(tool_name)
        if tool_class:
            return tool_class()
        return None
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool by name with the provided arguments.
        
        Args:
            tool_name: The name of the tool to execute
            **kwargs: Keyword arguments to pass to the tool
            
        Returns:
            The result of the tool execution
            
        Raises:
            ValueError: If the tool is not found
        """
        tool = self.create_tool_instance(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
        
        return tool.execute(**kwargs)
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Get metadata for all registered tools.
        
        Returns:
            List of tool metadata dictionaries
        """
        return [tool_class.get_metadata() for tool_class in self.tools.values()]
    
    def get_tools_as_json_schema(self) -> Dict[str, Any]:
        """
        Get all tools formatted as a JSON schema for LLM function calling.
        
        Returns:
            JSON schema describing all registered tools
        """
        tools_schema = []
        
        for tool_name, tool_class in self.tools.items():
            metadata = tool_class.get_metadata()
            
            # Create function schema
            function_schema = {
                "name": tool_name,
                "description": metadata["description"],
                "parameters": {
                    "type": "object",
                    "properties": metadata["input_schema"].get("properties", {}),
                    "required": metadata["input_schema"].get("required", [])
                }
            }
            
            tools_schema.append({"type": "function", "function": function_schema})
            
        return tools_schema


# Create singleton instance
tool_registry = ToolRegistry() 