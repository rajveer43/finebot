from typing import Dict, Any, List, Optional
import logging
import langdetect
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

from src.tools.tool_registry import Tool, tool_registry
from src.llm.gemini_integration import gemini
from src.config.config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize language detector
langdetect.DetectorFactory.seed = 0  # For consistent results

class LanguageTool(Tool):
    """Tool for language detection and translation in the financial chatbot."""
    
    name = "LanguageTool"
    description = "Detect language of text and translate content between languages"
    
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to process for language detection or translation"
            },
            "action": {
                "type": "string",
                "description": "Action to perform",
                "enum": ["detect", "translate"],
                "default": "detect"
            },
            "target_language": {
                "type": "string",
                "description": "Target language code for translation (e.g., 'en', 'es', 'fr')"
            }
        },
        "required": ["text", "action"]
    }
    
    output_schema = {
        "type": "object",
        "properties": {
            "detected_language": {
                "type": "string",
                "description": "Detected language code"
            },
            "translated_text": {
                "type": "string",
                "description": "Translated text (if translation was requested)"
            },
            "confidence": {
                "type": "number",
                "description": "Confidence level of language detection"
            }
        }
    }
    
    def __init__(self):
        super().__init__()
        self.language_names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ru': 'Russian',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'pt': 'Portuguese',
            'it': 'Italian'
        }
    
    async def execute(self, text: str, action: str = "detect", 
                target_language: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute language detection or translation.
        
        Args:
            text: Text to process
            action: Action to perform (detect or translate)
            target_language: Target language for translation
            
        Returns:
            Dictionary with detection or translation results
        """
        try:
            if action == "detect":
                return await self._detect_language(text)
            elif action == "translate":
                if not target_language:
                    raise ValueError("Target language is required for translation")
                
                # First detect the source language
                detection_result = await self._detect_language(text)
                source_language = detection_result["detected_language"]
                
                # Then perform translation
                return await self._translate_text(text, source_language, target_language)
            else:
                raise ValueError(f"Unsupported action: {action}")
                
        except Exception as e:
            logger.error(f"Error in language processing: {str(e)}")
            return {
                "error": str(e),
                "action": action
            }
    
    async def _detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Use langdetect library for initial detection
            lang_code = detect(text)
            confidence = 0.8  # langdetect doesn't provide confidence scores, using default
            
            # For very short texts or ambiguous cases, use LLM to verify
            if len(text.split()) < 5:
                lang_code, confidence = await self._llm_language_detection(text)
            
            language_name = self.language_names.get(lang_code, f"Unknown ({lang_code})")
            
            return {
                "detected_language": lang_code,
                "language_name": language_name,
                "confidence": confidence,
                "supported": lang_code in SUPPORTED_LANGUAGES
            }
            
        except LangDetectException:
            # If standard detection fails, try LLM-based detection
            lang_code, confidence = await self._llm_language_detection(text)
            language_name = self.language_names.get(lang_code, f"Unknown ({lang_code})")
            
            return {
                "detected_language": lang_code,
                "language_name": language_name,
                "confidence": confidence,
                "supported": lang_code in SUPPORTED_LANGUAGES
            }
    
    async def _llm_language_detection(self, text: str) -> tuple:
        """
        Use LLM to detect language for cases where standard detection might fail.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (language_code, confidence)
        """
        prompt = f"""
        Please identify the language of the following text. Respond with the ISO 639-1 language code 
        (e.g., 'en' for English, 'es' for Spanish) and a confidence score between 0 and 1.
        
        Text: "{text}"
        
        Format your response as a JSON object with 'language_code' and 'confidence' fields.
        """
        
        detection_schema = {
            "type": "object",
            "properties": {
                "language_code": {
                    "type": "string",
                    "description": "ISO 639-1 language code"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence level between 0 and 1"
                }
            }
        }
        
        result = await gemini.generate_structured_response(prompt, detection_schema, temperature=0.1)
        
        if "error" in result:
            # Default to English if detection fails
            return DEFAULT_LANGUAGE, 0.5
        
        return result.get("language_code", DEFAULT_LANGUAGE), result.get("confidence", 0.5)
    
    async def _translate_text(self, text: str, source_language: str, target_language: str) -> Dict[str, Any]:
        """
        Translate text from source to target language.
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Dictionary with translation results
        """
        # Check if the target language is supported
        if target_language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Target language '{target_language}' is not supported. Supported languages: {SUPPORTED_LANGUAGES}")
        
        # If source and target are the same, no translation needed
        if source_language == target_language:
            return {
                "detected_language": source_language,
                "translated_text": text,
                "target_language": target_language,
                "confidence": 1.0,
                "note": "Source and target languages are the same, no translation performed."
            }
        
        # Use Gemini for translation
        source_name = self.language_names.get(source_language, source_language)
        target_name = self.language_names.get(target_language, target_language)
        
        prompt = f"""
        Translate the following text from {source_name} to {target_name}. 
        Preserve the formatting, line breaks, and special characters where appropriate.
        
        Original text ({source_name}):
        {text}
        
        Translated text ({target_name}):
        """
        
        translated_text = await gemini.generate_response(prompt, temperature=0.2)
        
        return {
            "detected_language": source_language,
            "translated_text": translated_text,
            "target_language": target_language,
            "source_language_name": source_name,
            "target_language_name": target_name
        }


# Register the tool
tool_registry.register(LanguageTool) 