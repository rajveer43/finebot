from typing import Dict, Any, List, Optional
import logging

from src.tools.tool_registry import Tool, tool_registry
from src.llm.gemini_integration import gemini

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextSummarizationTool(Tool):
    """Tool for summarizing text content from financial documents."""
    
    name = "TextSummarizationTool"
    description = "Summarize text content from financial documents with different levels of detail"
    
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to summarize"
            },
            "summary_type": {
                "type": "string",
                "description": "Type of summary to generate",
                "enum": ["brief", "detailed", "bullet_points", "executive_summary"],
                "default": "detailed"
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum length of the summary in words",
                "default": 200
            },
            "focus_areas": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Specific areas or topics to focus on in the summary"
            }
        },
        "required": ["text"]
    }
    
    output_schema = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Generated summary of the text"
            },
            "key_points": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Key points extracted from the text"
            },
            "metadata": {
                "type": "object",
                "description": "Additional information about the summary"
            }
        }
    }
    
    def __init__(self):
        super().__init__()
    
    async def execute(self, text: str, summary_type: str = "detailed", 
                max_length: int = 200, focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a summary of the provided text.
        
        Args:
            text: The text to summarize
            summary_type: Type of summary to generate (brief, detailed, bullet_points, executive_summary)
            max_length: Maximum length of the summary in words
            focus_areas: Specific areas or topics to focus on in the summary
            
        Returns:
            Dictionary with the summary, key points, and metadata
        """
        try:
            # Truncate very long texts for LLM processing
            if len(text) > 25000:
                text = text[:25000] + "... [text truncated for processing]"
            
            # Generate the summary based on the requested type
            if summary_type == "brief":
                summary, key_points = await self._generate_brief_summary(text, max_length, focus_areas)
            elif summary_type == "detailed":
                summary, key_points = await self._generate_detailed_summary(text, max_length, focus_areas)
            elif summary_type == "bullet_points":
                summary, key_points = await self._generate_bullet_points(text, max_length, focus_areas)
            elif summary_type == "executive_summary":
                summary, key_points = await self._generate_executive_summary(text, max_length, focus_areas)
            else:
                # Default to detailed summary
                summary, key_points = await self._generate_detailed_summary(text, max_length, focus_areas)
            
            # Calculate metadata
            word_count_original = len(text.split())
            word_count_summary = len(summary.split())
            
            metadata = {
                "summary_type": summary_type,
                "original_length": word_count_original,
                "summary_length": word_count_summary,
                "compression_ratio": round(word_count_summary / word_count_original, 2) if word_count_original > 0 else 0,
                "focus_areas": focus_areas if focus_areas else []
            }
            
            return {
                "summary": summary,
                "key_points": key_points,
                "metadata": metadata
            }
                
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {
                "error": str(e),
                "summary_type": summary_type,
                "text_length": len(text)
            }
    
    async def _generate_brief_summary(self, text: str, max_length: int, 
                                    focus_areas: Optional[List[str]] = None) -> tuple:
        """Generate a brief summary focused on key financial insights."""
        focus_text = ""
        if focus_areas:
            focus_text = f" Focus specifically on the following areas: {', '.join(focus_areas)}."
        
        prompt = f"""
        Please provide a brief summary (maximum {max_length} words) of the following financial text. 
        Highlight only the most critical financial information and key insights.{focus_text}
        
        Text to summarize:
        {text}
        """
        
        summary_schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of the text"
                },
                "key_points": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "3-5 key points from the text"
                }
            }
        }
        
        result = await gemini.generate_structured_response(prompt, summary_schema, temperature=0.2)
        
        if "error" in result:
            # Fallback to simpler approach if structured response fails
            summary_text = await gemini.generate_response(prompt, temperature=0.2)
            return summary_text, ["Key points not available due to processing error"]
        
        return result.get("summary", ""), result.get("key_points", [])
    
    async def _generate_detailed_summary(self, text: str, max_length: int, 
                                      focus_areas: Optional[List[str]] = None) -> tuple:
        """Generate a detailed summary with comprehensive financial analysis."""
        focus_text = ""
        if focus_areas:
            focus_text = f" Pay special attention to the following areas: {', '.join(focus_areas)}."
        
        prompt = f"""
        Please provide a comprehensive summary (maximum {max_length} words) of the following financial text.
        Include detailed analysis of financial metrics, trends, and insights.{focus_text}
        
        Text to summarize:
        {text}
        """
        
        summary_schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Detailed summary of the text"
                },
                "key_points": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "5-8 key points from the text"
                }
            }
        }
        
        result = await gemini.generate_structured_response(prompt, summary_schema, temperature=0.2)
        
        if "error" in result:
            # Fallback to simpler approach if structured response fails
            summary_text = await gemini.generate_response(prompt, temperature=0.2)
            return summary_text, ["Key points not available due to processing error"]
        
        return result.get("summary", ""), result.get("key_points", [])
    
    async def _generate_bullet_points(self, text: str, max_length: int, 
                                   focus_areas: Optional[List[str]] = None) -> tuple:
        """Generate a bullet-point summary of key financial information."""
        focus_text = ""
        if focus_areas:
            focus_text = f" Ensure you include points about the following areas: {', '.join(focus_areas)}."
        
        prompt = f"""
        Please provide a bullet-point summary of the following financial text.
        Extract the most important financial data points, metrics, and insights.
        Limit the response to approximately {max_length} words in total.{focus_text}
        
        Text to summarize:
        {text}
        """
        
        summary_schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief introductory paragraph"
                },
                "key_points": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Bullet points with key information from the text (8-12 points)"
                }
            }
        }
        
        result = await gemini.generate_structured_response(prompt, summary_schema, temperature=0.2)
        
        if "error" in result:
            # Fallback to simpler approach if structured response fails
            bullet_points = await gemini.generate_response(prompt, temperature=0.2)
            return "", bullet_points.split("\n")
        
        return result.get("summary", ""), result.get("key_points", [])
    
    async def _generate_executive_summary(self, text: str, max_length: int, 
                                       focus_areas: Optional[List[str]] = None) -> tuple:
        """Generate an executive summary for high-level decision makers."""
        focus_text = ""
        if focus_areas:
            focus_text = f" Pay particular attention to the following areas of interest: {', '.join(focus_areas)}."
        
        prompt = f"""
        Please provide an executive summary (maximum {max_length} words) of the following financial text.
        This summary should be suitable for senior executives and decision makers.
        Focus on strategic insights, major financial indicators, and actionable recommendations.
        Use clear, concise language suitable for high-level business communication.{focus_text}
        
        Text to summarize:
        {text}
        """
        
        summary_schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string", 
                    "description": "Executive summary of the text"
                },
                "key_points": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Key strategic points and recommendations"
                }
            }
        }
        
        result = await gemini.generate_structured_response(prompt, summary_schema, temperature=0.2)
        
        if "error" in result:
            # Fallback to simpler approach if structured response fails
            summary_text = await gemini.generate_response(prompt, temperature=0.2)
            return summary_text, ["Key points not available due to processing error"]
        
        return result.get("summary", ""), result.get("key_points", [])


# Register the tool
tool_registry.register(TextSummarizationTool) 