import requests
import logging
import os
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import time

from src.tools.tool_registry import Tool, tool_registry
from src.tools.web_search_tool import WebSearchTool
from src.config.config import SERPAPI_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchAPITool(Tool):
    """Tool for searching web content using search APIs and summarizing results."""
    
    name = "SearchAPITool"
    description = "Search the web for information using a search API, retrieve content, and provide summaries and insights"
    
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find information"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of search results to fetch",
                "default": 5
            },
            "include_urls": {
                "type": "boolean",
                "description": "Whether to include URLs in the response",
                "default": True
            },
            "fetch_full_content": {
                "type": "boolean",
                "description": "Whether to fetch full content from search results",
                "default": True
            }
        },
        "required": ["query"]
    }
    
    output_schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "description": "Search results with content"
            },
            "summary": {
                "type": "string",
                "description": "Summarized information from search results"
            },
            "sources": {
                "type": "array",
                "description": "Sources of information"
            },
            "error": {
                "type": "string",
                "description": "Error message if search failed"
            }
        }
    }
    
    def __init__(self):
        super().__init__()
        self.web_search_tool = WebSearchTool()
        # Check if API key is available
        self.serpapi_key = os.environ.get("SERPAPI_KEY", SERPAPI_KEY)
        self.use_mockdata = not self.serpapi_key
        if self.use_mockdata:
            logger.warning("SERPAPI_KEY not found. Using mock search data. For production, set the SERPAPI_KEY environment variable.")
    
    def execute(self, query: str, num_results: int = 5, 
               include_urls: bool = True, 
               fetch_full_content: bool = True) -> Dict[str, Any]:
        """
        Execute search and summarize results.
        
        Args:
            query: Search query to find information
            num_results: Number of search results to fetch
            include_urls: Whether to include URLs in the response
            fetch_full_content: Whether to fetch full content from search results
            
        Returns:
            Dictionary with search results, summary, and sources
        """
        try:
            # Step 1: Perform the search using API or mock data
            search_results = self._perform_search(query, num_results)
            
            if "error" in search_results:
                return search_results
            
            # Step 2: Process and enrich search results
            enriched_results = []
            sources = []
            
            for result in search_results["results"]:
                url = result.get("link")
                if not url:
                    continue
                
                source = {
                    "title": result.get("title", "Untitled"),
                    "url": url,
                    "snippet": result.get("snippet", "")
                }
                
                # Add source to list
                sources.append(source)
                
                # Fetch full content if requested
                if fetch_full_content:
                    try:
                        content_result = self.web_search_tool.execute(
                            url=url,
                            extract_type="text",
                            download_linked_docs=False
                        )
                        
                        # Add full content to result
                        if "text_content" in content_result and content_result["text_content"]:
                            # Limit content length to avoid overwhelming
                            text_content = content_result["text_content"]
                            if len(text_content) > 10000:
                                text_content = text_content[:10000] + "... [truncated]"
                            
                            result["full_content"] = text_content
                    except Exception as e:
                        logger.error(f"Error fetching content from {url}: {str(e)}")
                        result["error_fetching_content"] = str(e)
                
                enriched_results.append(result)
            
            # Step 3: Create a summary of all content
            summary = self._create_summary(enriched_results, query)
            
            # Prepare final response
            response = {
                "results": enriched_results if include_urls else [],
                "summary": summary,
                "sources": sources,
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error executing search for query '{query}': {str(e)}")
            return {
                "error": str(e),
                "query": query
            }
    
    def _perform_search(self, query: str, num_results: int) -> Dict[str, Any]:
        """
        Perform search using SerpAPI or mock data.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Dictionary with search results
        """
        if self.use_mockdata:
            # Use mock data for testing
            return self._generate_mock_search_results(query, num_results)
        
        try:
            # Use SerpAPI for real searches
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.serpapi_key,
                "num": num_results
            }
            
            response = requests.get(
                "https://serpapi.com/search", 
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract organic results
            organic_results = data.get("organic_results", [])
            
            return {
                "results": organic_results[:num_results],
                "search_metadata": data.get("search_metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Error calling search API: {str(e)}")
            return {
                "error": f"Search API error: {str(e)}",
                "query": query
            }
    
    def _generate_mock_search_results(self, query: str, num_results: int) -> Dict[str, Any]:
        """
        Generate mock search results for testing when API key is not available.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Dictionary with mock search results
        """
        logger.info(f"Generating mock search results for query: {query}")
        
        # Create mock results based on the query
        mock_results = []
        current_time = datetime.now().strftime("%Y-%m-%d")
        
        topics = [
            "financial market trends", 
            "economic forecast", 
            "investment strategies", 
            "stock market analysis",
            "financial technology innovation"
        ]
        
        for i in range(min(num_results, 5)):
            topic = topics[i % len(topics)]
            mock_results.append({
                "position": i + 1,
                "title": f"{topic.title()} - {current_time}",
                "link": f"https://example.com/finance/{topic.replace(' ', '-')}-{i + 1}",
                "snippet": f"Latest analysis on {topic} as of {current_time}. This mock result discusses key insights related to your query: '{query}'.",
                "source": "MockFinancialTimes"
            })
        
        return {
            "results": mock_results,
            "search_metadata": {
                "status": "Success",
                "created_at": current_time,
                "processed_at": current_time,
                "total_time_taken": 0.5,
                "engine_url": "https://www.mockgoogle.com/search?q=" + query.replace(" ", "+")
            }
        }
    
    def _create_summary(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Create a summary of search results.
        
        Args:
            results: List of search results with content
            query: Original search query
            
        Returns:
            Summary text
        """
        # Start with a header
        summary = f"# Search Results Summary for: {query}\n\n"
        
        # Count how many results have full content
        results_with_content = sum(1 for r in results if "full_content" in r)
        
        summary += f"Found {len(results)} relevant sources"
        if results_with_content < len(results):
            summary += f" (full content extracted from {results_with_content})"
        summary += ".\n\n"
        
        # Add key points from each source
        summary += "## Key Information Sources:\n\n"
        
        for i, result in enumerate(results):
            title = result.get("title", "Untitled")
            summary += f"### {i+1}. {title}\n\n"
            
            if "snippet" in result:
                summary += f"**Snippet**: {result['snippet']}\n\n"
            
            if "full_content" in result:
                # Extract a relevant excerpt from the full content
                excerpt = self._extract_relevant_excerpt(result["full_content"], query, max_length=300)
                summary += f"**Excerpt**: {excerpt}\n\n"
            
            summary += f"**Source**: [{title}]({result.get('link', '#')})\n\n"
        
        # Add timestamp
        summary += f"\n\n*Information updated as of {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
        
        return summary
    
    def _extract_relevant_excerpt(self, content: str, query: str, max_length: int = 300) -> str:
        """
        Extract a relevant excerpt from content based on the query.
        
        Args:
            content: Full content text
            query: Search query to find relevant section
            max_length: Maximum length of excerpt
            
        Returns:
            Relevant excerpt from content
        """
        # Convert query to lower case and split into terms
        query_terms = query.lower().split()
        
        # Split content into paragraphs
        paragraphs = [p for p in content.split('\n') if p.strip()]
        
        # Score paragraphs based on query term frequency
        scored_paragraphs = []
        for para in paragraphs:
            if len(para) < 30:  # Skip very short paragraphs
                continue
                
            para_lower = para.lower()
            score = sum(1 for term in query_terms if term in para_lower)
            scored_paragraphs.append((score, para))
        
        # Sort by score (highest first)
        scored_paragraphs.sort(reverse=True)
        
        # Take the highest scoring paragraph
        if scored_paragraphs:
            best_para = scored_paragraphs[0][1]
            
            # Truncate if needed
            if len(best_para) > max_length:
                # Try to find a good breaking point
                cutoff = max_length
                while cutoff > max_length - 50 and cutoff < len(best_para) and best_para[cutoff] != ' ':
                    cutoff += 1
                return best_para[:cutoff] + "..."
            return best_para
        
        # Fallback if no good paragraph found
        return content[:max_length] + "..." if len(content) > max_length else content

# Register the tool
tool_registry.register(SearchAPITool) 