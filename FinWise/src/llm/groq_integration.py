import requests
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.config import GROQ_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroqLLM:
    """
    Class for interacting with Groq models through their API.
    """
    
    def __init__(self, model_name: str = "llama3-8b-8192"):
        """
        Initialize the Groq LLM with the specified model.
        
        Args:
            model_name: The name of the Groq model to use
                        Options include: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b, etc.
        """
        self.model_name = model_name
        self.api_key = GROQ_API_KEY
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.system_prompt = None
        
    def set_system_prompt(self, system_prompt: str):
        """
        Set a system prompt that will be included in all requests to the model.
        
        Args:
            system_prompt: The system prompt text
        """
        self.system_prompt = system_prompt
        
    async def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate a response from the Groq model based on the provided prompt.
        
        Args:
            prompt: The user's input prompt
            temperature: Controls randomness (0.0 to 1.0)
            
        Returns:
            The generated response as a string
        """
        # Prepare messages
        messages = []
        
        # Add system prompt if it exists
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
            
        # Add the user's message
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Send request to Groq API
        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()  # Raise exception for error status codes
            
            # Parse the response
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            raise
    
    async def generate_structured_response(self, 
                                          prompt: str, 
                                          output_schema: Dict[str, Any],
                                          temperature: float = 0.2) -> Dict[str, Any]:
        """
        Generate a structured response from Groq based on a provided JSON schema.
        
        Args:
            prompt: The user's input prompt
            output_schema: The JSON schema for structuring the output
            temperature: Controls randomness (0.0 to 1.0)
            
        Returns:
            A dictionary containing the structured response
        """
        schema_prompt = f"""
        Please analyze the following query and provide a response structured according to this JSON schema:
        {json.dumps(output_schema, indent=2)}
        
        Query: {prompt}
        
        Ensure your response is valid JSON matching the provided schema exactly.
        """
        
        # Prepare messages
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        messages.append({"role": "user", "content": schema_prompt})
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Send request to Groq API
        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()  # Raise exception for error status codes
            
            # Parse the response
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
            
            # Extract and parse JSON from the response
            try:
                # Look for JSON block in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    return json.loads(json_str)
                else:
                    raise ValueError("No valid JSON found in response")
            except json.JSONDecodeError:
                # Fallback to attempt to fix malformed JSON
                try:
                    import json_repair # type: ignore
                    repaired_json = json_repair.repair_json(response_text)
                    return json.loads(repaired_json)
                except:
                    # Return error information if JSON parsing fails
                    return {
                        "error": "Failed to parse structured response",
                        "raw_response": response_text
                    }
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            raise
                
    async def analyze_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user's query to determine intent and needed tools.
        
        Args:
            query: The user's input query
            
        Returns:
            A dictionary containing the intent analysis
        """
        intent_schema = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "The primary intent of the query",
                    "enum": ["data_analysis", "summarization", "comparison", "question_answering", 
                             "visualization", "extraction", "trend_analysis", "other"]
                },
                "tools_needed": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of tools needed to fulfill this request"
                },
                "file_types_mentioned": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "File types mentioned in the query"
                },
                "metrics_mentioned": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Financial metrics mentioned in the query"
                },
                "visualization_type": {
                    "type": "string",
                    "description": "Type of visualization that might be needed",
                    "enum": ["trend", "comparison", "distribution", "proportion", "correlation", "none"]
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score for this intent classification (0-1)"
                }
            },
            "required": ["intent", "tools_needed", "confidence"]
        }
        
        return await self.generate_structured_response(query, intent_schema)

# Create a default instance for easy import elsewhere
groq = GroqLLM() 