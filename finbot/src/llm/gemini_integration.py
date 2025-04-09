import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Any, Optional
import json

from src.config.config import GEMINI_API_KEY

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

class GeminiLLM:
    """
    Class for interacting with Google's Gemini model through the Google Generative AI API
    and LangChain integration.
    """
    
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        """
        Initialize the Gemini LLM with the specified model.
        
        Args:
            model_name: The name of the Gemini model to use
        """
        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(model=model_name)
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
        Generate a response from the Gemini model based on the provided prompt.
        
        Args:
            prompt: The user's input prompt
            temperature: Controls randomness (0.0 to 1.0)
            
        Returns:
            The generated response as a string
        """
        messages = []
        
        # Add system prompt if it exists
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
            
        # Add the user's message
        messages.append(HumanMessage(content=prompt))
        
        # Generate response - temperature is set when creating the LLM instance
        # since the latest API doesn't accept it in the invoke method
        temp_llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=temperature)
        response = await temp_llm.ainvoke(messages)
        
        return response.content
    
    async def generate_structured_response(self, 
                                          prompt: str, 
                                          output_schema: Dict[str, Any],
                                          temperature: float = 0.2) -> Dict[str, Any]:
        """
        Generate a structured response from Gemini based on a provided JSON schema.
        
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
        
        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        
        messages.append(HumanMessage(content=schema_prompt))
        
        # Generate response - create a new instance with the temperature parameter
        temp_llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=temperature)
        response = await temp_llm.ainvoke(messages)
        
        # Extract and parse JSON from the response
        try:
            # Look for JSON block in the response
            response_text = response.content
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
                import json_repair
                repaired_json = json_repair.repair_json(response.content)
                return json.loads(repaired_json)
            except:
                # Return error information if JSON parsing fails
                return {
                    "error": "Failed to parse structured response",
                    "raw_response": response.content
                }
                
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
                "confidence": {
                    "type": "number",
                    "description": "Confidence score for this intent classification (0-1)"
                }
            },
            "required": ["intent", "tools_needed", "confidence"]
        }
        
        return await self.generate_structured_response(query, intent_schema)

# Create a default instance for easy import elsewhere
gemini = GeminiLLM() 