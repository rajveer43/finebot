import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Any, Optional
import json
import logging
import time

from src.config.config import GEMINI_API_KEY, GEMINI_2_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key management
class APIKeyManager:
    """
    Manages multiple API keys with rotation on quota errors
    """
    def __init__(self, api_keys: List[str]):
        """
        Initialize with a list of API keys
        
        Args:
            api_keys: List of API keys to use
        """
        self.api_keys = api_keys
        self.current_index = 0
        self.key_status = {key: {"active": True, "last_error": None, "cooldown_until": 0} for key in api_keys}
        
    def get_current_key(self) -> str:
        """Get the current active API key"""
        # Try to find an active key
        for _ in range(len(self.api_keys)):
            key = self.api_keys[self.current_index]
            key_info = self.key_status[key]
            
            # If key is in cooldown, check if cooldown period has expired
            current_time = time.time()
            if not key_info["active"] and key_info["cooldown_until"] <= current_time:
                logger.info(f"API key {self.mask_key(key)} cooldown expired, marking as active again")
                key_info["active"] = True
                
            # If key is active, return it
            if key_info["active"]:
                return key
                
            # Try next key
            self.current_index = (self.current_index + 1) % len(self.api_keys)
        
        # If we get here, all keys are inactive
        logger.warning("All API keys are in cooldown. Using first key anyway.")
        return self.api_keys[0]
    
    def rotate_key(self) -> str:
        """Rotate to the next API key"""
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        return self.get_current_key()
        
    def mark_key_error(self, key: str, error_message: str):
        """Mark a key as having an error"""
        if key in self.key_status:
            # Check if it's a quota error
            if "429" in error_message and "quota" in error_message:
                # Extract retry delay if available
                retry_seconds = 60  # Default cooldown
                import re
                retry_match = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)\s*}", error_message)
                if retry_match:
                    try:
                        retry_seconds = int(retry_match.group(1))
                    except:
                        pass
                
                cooldown_until = time.time() + retry_seconds
                self.key_status[key] = {
                    "active": False,
                    "last_error": error_message,
                    "cooldown_until": cooldown_until
                }
                logger.warning(f"API key {self.mask_key(key)} hit quota limit, in cooldown for {retry_seconds} seconds")
            else:
                # Non-quota error, just log it
                self.key_status[key]["last_error"] = error_message
                logger.warning(f"API key {self.mask_key(key)} experienced non-quota error: {error_message}")
    
    def mask_key(self, key: str) -> str:
        """Return a masked version of the API key for logging"""
        if len(key) <= 8:
            return "****"
        return key[:4] + "..." + key[-4:]
    
    def __str__(self) -> str:
        """String representation showing key status"""
        status_str = "API Key Status:\n"
        for i, key in enumerate(self.api_keys):
            key_info = self.key_status[key]
            current_marker = "→ " if i == self.current_index else "  "
            status_str += f"{current_marker}{self.mask_key(key)}: {'Active' if key_info['active'] else 'Cooldown'}\n"
        return status_str

# Initialize API key manager with the default key from config and the new key
# Add your second API key here
API_KEYS = [GEMINI_API_KEY, GEMINI_2_API_KEY]
key_manager = APIKeyManager(API_KEYS)

# Configure the Gemini API with the initial key
genai.configure(api_key=key_manager.get_current_key())

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
        self.key_manager = key_manager
        self.llm = self._create_llm_instance()
        self.system_prompt = None
        
    def _create_llm_instance(self, temperature: float = 0.7):
        """Create a new LLM instance with the current API key"""
        api_key = self.key_manager.get_current_key()
        return ChatGoogleGenerativeAI(model=self.model_name, temperature=temperature, google_api_key=api_key)
        
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
        
        # Try with rotation on quota errors
        for attempt in range(len(API_KEYS) + 1):  # +1 to allow for retrying first key if all fail
            try:
                # Create a new LLM instance with current API key
                current_key = self.key_manager.get_current_key()
                temp_llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=temperature, google_api_key=current_key)
                
                # Generate response
                response = await temp_llm.ainvoke(messages)
                return response.content
                
            except Exception as e:
                error_msg = str(e)
                current_key = self.key_manager.get_current_key()
                
                # Mark the current key as having an error
                self.key_manager.mark_key_error(current_key, error_msg)
                
                # If we've tried all keys, re-raise the exception
                if attempt >= len(API_KEYS) - 1:
                    raise
                
                # Otherwise try the next key
                logger.info(f"Rotating API key after error: {error_msg}")
                self.key_manager.rotate_key()
                logger.info(f"Now using API key: {self.key_manager.mask_key(self.key_manager.get_current_key())}")
        
        # This should never be reached due to the exception handling above
        raise RuntimeError("Failed to generate response after trying all API keys")
    
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
        
        # Try with rotation on quota errors
        for attempt in range(len(API_KEYS) + 1):  # +1 to allow for retrying first key if all fail
            try:
                # Create a new LLM instance with current API key
                current_key = self.key_manager.get_current_key()
                temp_llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=temperature, google_api_key=current_key)
                
                # Generate response
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
                
            except Exception as e:
                error_msg = str(e)
                current_key = self.key_manager.get_current_key()
                
                # Mark the current key as having an error
                self.key_manager.mark_key_error(current_key, error_msg)
                
                # If we've tried all keys, re-raise the exception
                if attempt >= len(API_KEYS) - 1:
                    raise
                
                # Otherwise try the next key
                logger.info(f"Rotating API key after error: {error_msg}")
                self.key_manager.rotate_key()
                logger.info(f"Now using API key: {self.key_manager.mask_key(self.key_manager.get_current_key())}")
                
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
gemini = GeminiLLM() 