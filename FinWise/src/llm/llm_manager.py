import logging
from typing import List, Dict, Any, Optional, Tuple
import time

from src.llm.gemini_integration import gemini
from src.llm.groq_integration import groq

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMManager:
    """
    Manages multiple LLM providers with fallback capability.
    """
    
    def __init__(self, providers=None, default_provider="gemini"):
        """
        Initialize the LLM manager with specified providers.
        
        Args:
            providers: List of provider names to use, in order of preference
            default_provider: The default provider to use first
        """
        # Available providers
        self.available_providers = {
            "gemini": gemini,
            "groq": groq
        }
        
        # Set providers list from input or use all available
        self.providers = providers if providers else list(self.available_providers.keys())
        
        # Ensure default provider is first in the list
        if default_provider in self.providers:
            self.providers.remove(default_provider)
            self.providers.insert(0, default_provider)
        
        # Provider status tracking
        self.provider_status = {
            provider: {
                "active": True, 
                "error_count": 0,
                "last_error": None,
                "cooldown_until": 0
            } for provider in self.providers
        }
        
        logger.info(f"LLM Manager initialized with providers: {self.providers}")
    
    def _get_active_provider(self) -> str:
        """Get the current active provider name."""
        current_time = time.time()
        
        # Check each provider in order of preference
        for provider in self.providers:
            status = self.provider_status[provider]
            
            # Check if provider is in cooldown
            if not status["active"] and status["cooldown_until"] <= current_time:
                logger.info(f"Provider {provider} cooldown expired, marking as active again")
                status["active"] = True
            
            # If provider is active, return it
            if status["active"]:
                return provider
        
        # If all providers are inactive, use the first one anyway
        logger.warning("All providers are in cooldown. Using first provider anyway.")
        return self.providers[0]
    
    def _mark_provider_error(self, provider: str, error_message: str) -> None:
        """Mark a provider as having an error."""
        if provider not in self.provider_status:
            return
            
        status = self.provider_status[provider]
        status["error_count"] += 1
        status["last_error"] = error_message
        
        # Check if it's a quota/rate limit error
        is_quota_error = False
        
        if provider == "gemini" and "429" in error_message and "quota" in error_message:
            is_quota_error = True
        elif provider == "groq" and ("429" in error_message or "rate limit" in error_message.lower()):
            is_quota_error = True
            
        # Apply cooldown if it's a quota error or too many errors
        if is_quota_error or status["error_count"] >= 3:
            cooldown_time = 60  # Default cooldown in seconds
            
            # Extract retry delay if available for Gemini
            if provider == "gemini" and "retry_delay" in error_message:
                import re
                retry_match = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)\s*}", error_message)
                if retry_match:
                    try:
                        cooldown_time = int(retry_match.group(1))
                    except:
                        pass
            
            status["active"] = False
            status["cooldown_until"] = time.time() + cooldown_time
            logger.warning(f"Provider {provider} marked inactive for {cooldown_time} seconds due to errors")
        
    async def generate_response(self, prompt: str, temperature: float = 0.7, 
                              max_attempts: int = 3) -> Tuple[str, str]:
        """
        Generate a response using available providers with fallback.
        
        Args:
            prompt: The user's input prompt
            temperature: Controls randomness (0.0 to 1.0)
            max_attempts: Maximum number of attempts across providers
            
        Returns:
            Tuple of (response text, provider used)
        """
        attempts = 0
        providers_tried = set()
        
        while attempts < max_attempts and len(providers_tried) < len(self.providers):
            provider_name = self._get_active_provider()
            
            # Skip if we've already tried this provider
            if provider_name in providers_tried:
                # Try the next provider
                index = self.providers.index(provider_name)
                provider_name = self.providers[(index + 1) % len(self.providers)]
                
            providers_tried.add(provider_name)
            provider = self.available_providers[provider_name]
            
            try:
                logger.info(f"Attempting to generate response with provider: {provider_name}")
                response = await provider.generate_response(prompt, temperature)
                return response, provider_name
                
            except Exception as e:
                error_message = str(e)
                logger.error(f"Error with provider {provider_name}: {error_message}")
                self._mark_provider_error(provider_name, error_message)
                attempts += 1
        
        raise RuntimeError(f"Failed to generate response after {attempts} attempts across {len(providers_tried)} providers")
    
    async def generate_structured_response(self, prompt: str, output_schema: Dict[str, Any],
                                        temperature: float = 0.2, max_attempts: int = 3) -> Tuple[Dict[str, Any], str]:
        """
        Generate a structured response using available providers with fallback.
        
        Args:
            prompt: The user's input prompt
            output_schema: The JSON schema for structuring the output
            temperature: Controls randomness (0.0 to 1.0)
            max_attempts: Maximum number of attempts across providers
            
        Returns:
            Tuple of (structured response dict, provider used)
        """
        attempts = 0
        providers_tried = set()
        
        while attempts < max_attempts and len(providers_tried) < len(self.providers):
            provider_name = self._get_active_provider()
            
            # Skip if we've already tried this provider
            if provider_name in providers_tried:
                # Try the next provider
                index = self.providers.index(provider_name)
                provider_name = self.providers[(index + 1) % len(self.providers)]
                
            providers_tried.add(provider_name)
            provider = self.available_providers[provider_name]
            
            try:
                logger.info(f"Attempting to generate structured response with provider: {provider_name}")
                response = await provider.generate_structured_response(prompt, output_schema, temperature)
                return response, provider_name
                
            except Exception as e:
                error_message = str(e)
                logger.error(f"Error with provider {provider_name}: {error_message}")
                self._mark_provider_error(provider_name, error_message)
                attempts += 1
        
        raise RuntimeError(f"Failed to generate structured response after {attempts} attempts across {len(providers_tried)} providers")
    
    async def analyze_intent(self, query: str, max_attempts: int = 3) -> Tuple[Dict[str, Any], str]:
        """
        Analyze user intent using available providers with fallback.
        
        Args:
            query: The user's input query
            max_attempts: Maximum number of attempts across providers
            
        Returns:
            Tuple of (intent analysis dict, provider used)
        """
        attempts = 0
        providers_tried = set()
        
        while attempts < max_attempts and len(providers_tried) < len(self.providers):
            provider_name = self._get_active_provider()
            
            # Skip if we've already tried this provider
            if provider_name in providers_tried:
                # Try the next provider
                index = self.providers.index(provider_name)
                provider_name = self.providers[(index + 1) % len(self.providers)]
                
            providers_tried.add(provider_name)
            provider = self.available_providers[provider_name]
            
            try:
                logger.info(f"Attempting to analyze intent with provider: {provider_name}")
                response = await provider.analyze_intent(query)
                return response, provider_name
                
            except Exception as e:
                error_message = str(e)
                logger.error(f"Error with provider {provider_name}: {error_message}")
                self._mark_provider_error(provider_name, error_message)
                attempts += 1
        
        raise RuntimeError(f"Failed to analyze intent after {attempts} attempts across {len(providers_tried)} providers")

# Create a default instance for easy import elsewhere
llm_manager = LLMManager() 