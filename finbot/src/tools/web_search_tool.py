import requests
import os
import logging
import re
import tempfile
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import json

from src.tools.tool_registry import Tool, tool_registry
from src.tools.file_processor import FileProcessor
from src.config.config import UPLOAD_FOLDER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchTool(Tool):
    """Tool for searching web content and processing links."""
    
    name = "WebSearchTool"
    description = "Search and extract content from web URLs, download linked documents, and process online financial information"
    
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch content from"
            },
            "extract_type": {
                "type": "string",
                "description": "Type of extraction to perform",
                "enum": ["text", "tables", "links", "document", "all"],
                "default": "all"
            },
            "search_term": {
                "type": "string",
                "description": "Term to search for within the content (optional)"
            },
            "download_linked_docs": {
                "type": "boolean",
                "description": "Whether to download linked documents (PDF, Excel, etc.)",
                "default": False
            }
        },
        "required": ["url"]
    }
    
    output_schema = {
        "type": "object",
        "properties": {
            "text_content": {
                "type": "string",
                "description": "Extracted text content from the URL"
            },
            "tables": {
                "type": "array",
                "description": "Tables extracted from the content"
            },
            "links": {
                "type": "array",
                "description": "Links found in the content"
            },
            "downloaded_files": {
                "type": "array",
                "description": "Paths to downloaded files"
            },
            "metadata": {
                "type": "object",
                "description": "Metadata about the extraction"
            }
        }
    }
    
    def __init__(self):
        super().__init__()
        self.file_processor = FileProcessor()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def execute(self, url: str, extract_type: str = "all", 
                search_term: Optional[str] = None,
                download_linked_docs: bool = False) -> Dict[str, Any]:
        """
        Execute web search and content extraction.
        
        Args:
            url: URL to fetch content from
            extract_type: Type of extraction to perform (text, tables, links, document, all)
            search_term: Term to search for within the content
            download_linked_docs: Whether to download linked documents
            
        Returns:
            Dictionary with extraction results
        """
        try:
            # Check if URL is a direct link to a document (PDF, Excel, etc.)
            if self._is_document_url(url):
                return self._process_document_url(url)
            
            # Otherwise process as HTML
            return self._process_html_url(url, extract_type, search_term, download_linked_docs)
                
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            return {
                "error": str(e),
                "url": url,
                "extract_type": extract_type
            }
    
    def _is_document_url(self, url: str) -> bool:
        """Check if URL points directly to a document."""
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        return path.endswith(('.pdf', '.xlsx', '.xls', '.csv', '.doc', '.docx'))
    
    def _process_document_url(self, url: str) -> Dict[str, Any]:
        """Process a URL that points directly to a document."""
        try:
            # Extract filename from URL
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            
            # Create a temporary file to save the document
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp_file:
                # Download file
                response = requests.get(url, headers=self.headers, stream=True)
                response.raise_for_status()
                
                # Write content to temporary file
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                
                tmp_path = tmp_file.name
            
            # Process the downloaded file using the FileProcessor
            processing_result = self.file_processor.execute(file_path=tmp_path)
            
            # Move file to uploads folder for persistence
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            os.rename(tmp_path, upload_path)
            
            return {
                "text_content": processing_result.get("text_content", ""),
                "tables": processing_result.get("table_data", []),
                "links": [],
                "downloaded_files": [upload_path],
                "metadata": {
                    "url": url,
                    "document_type": os.path.splitext(filename)[1][1:],
                    "filename": filename,
                    "processed_data": processing_result.get("metadata", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error downloading document from {url}: {str(e)}")
            return {
                "error": str(e),
                "url": url,
                "text_content": f"Failed to download or process document from {url}"
            }
    
    def _process_html_url(self, url: str, extract_type: str, 
                         search_term: Optional[str], 
                         download_linked_docs: bool) -> Dict[str, Any]:
        """Process URL as HTML content."""
        try:
            # Fetch the content
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Initialize results
            result = {
                "text_content": "",
                "tables": [],
                "links": [],
                "downloaded_files": [],
                "metadata": {
                    "url": url,
                    "title": soup.title.string if soup.title else url,
                    "extract_type": extract_type
                }
            }
            
            # Extract based on type
            if extract_type in ["text", "all"]:
                # Extract main text content
                text_content = self._extract_text_content(soup)
                
                # Filter by search term if provided
                if search_term and text_content:
                    # Highlight search term in content
                    text_content = self._highlight_search_term(text_content, search_term)
                
                result["text_content"] = text_content
            
            if extract_type in ["tables", "all"]:
                # Extract tables
                tables = self._extract_tables(soup)
                result["tables"] = tables
            
            if extract_type in ["links", "all"]:
                # Extract links
                links = self._extract_links(soup, url)
                result["links"] = links
                
                # Download linked documents if requested
                if download_linked_docs:
                    downloaded_files = self._download_linked_documents(links)
                    result["downloaded_files"] = downloaded_files
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing HTML URL {url}: {str(e)}")
            return {
                "error": str(e),
                "url": url,
                "extract_type": extract_type
            }
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from the page."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get main content (heuristic approach)
        main_content = ""
        
        # Try to find main content containers
        main_elements = soup.find_all(["article", "main", "div"], 
                                     class_=re.compile(r"content|article|main|post"))
        
        if main_elements:
            # Use the longest content section
            main_content = max([elem.get_text(strip=True, separator="\n") for elem in main_elements], 
                              key=len)
        else:
            # Fallback to full page text
            main_content = soup.get_text(strip=True, separator="\n")
        
        # Clean up the text (remove excessive whitespace)
        main_content = re.sub(r'\n+', '\n', main_content)
        main_content = re.sub(r' +', ' ', main_content)
        
        return main_content.strip()
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract tables from the page."""
        tables = []
        
        for i, table in enumerate(soup.find_all('table')):
            # Extract table headers
            headers = []
            header_row = table.find('thead')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
            else:
                # Try first row as header
                first_row = table.find('tr')
                if first_row:
                    headers = [th.get_text(strip=True) for th in first_row.find_all(['th', 'td'])]
            
            # Extract table rows
            rows = []
            for tr in table.find_all('tr')[1:] if headers else table.find_all('tr'):
                row = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if row:  # Skip empty rows
                    rows.append(row)
            
            if rows:  # Only add tables with data
                tables.append({
                    "table_id": i + 1,
                    "headers": headers,
                    "rows": rows,
                    "row_count": len(rows),
                    "column_count": len(headers) if headers else (len(rows[0]) if rows else 0)
                })
        
        return tables
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Extract links from the page."""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Skip empty, javascript or anchor links
            if not href or href.startswith(('javascript:', '#')):
                continue
            
            # Handle relative URLs
            if not href.startswith(('http://', 'https://')):
                parsed_base = urlparse(base_url)
                base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
                
                if href.startswith('/'):
                    href = f"{base_domain}{href}"
                else:
                    href = f"{base_domain}/{href}"
            
            # Get link text
            link_text = link.get_text(strip=True)
            
            # Determine if it's a document link
            is_document = self._is_document_url(href)
            
            links.append({
                "url": href,
                "text": link_text if link_text else href,
                "is_document": is_document,
                "document_type": os.path.splitext(href)[1][1:] if is_document else None
            })
        
        return links
    
    def _highlight_search_term(self, text: str, search_term: str) -> str:
        """Highlight search term in text content."""
        # Simple case-insensitive replacement
        pattern = re.compile(re.escape(search_term), re.IGNORECASE)
        highlighted = pattern.sub(f"**{search_term}**", text)
        
        return highlighted
    
    def _download_linked_documents(self, links: List[Dict[str, Any]]) -> List[str]:
        """Download linked documents."""
        downloaded_files = []
        
        for link in links:
            # Only download document links
            if link.get("is_document", False):
                try:
                    # Extract filename from URL
                    url = link["url"]
                    parsed_url = urlparse(url)
                    filename = os.path.basename(parsed_url.path)
                    
                    # Create file path in uploads folder
                    file_path = os.path.join(UPLOAD_FOLDER, filename)
                    
                    # Skip if file already exists
                    if os.path.exists(file_path):
                        downloaded_files.append(file_path)
                        continue
                    
                    # Download file
                    response = requests.get(url, headers=self.headers, stream=True)
                    response.raise_for_status()
                    
                    # Save file
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    downloaded_files.append(file_path)
                    
                except Exception as e:
                    logger.error(f"Error downloading {link['url']}: {str(e)}")
        
        return downloaded_files


# Register the tool
tool_registry.register(WebSearchTool) 