import os
import asyncio
import streamlit as st
import tempfile
import logging
from pathlib import Path
from datetime import datetime
import time
import re
import json
import base64
from io import BytesIO
import pandas as pd
import uuid

# Import tool registry and ensure tools are registered
from src.tools.tool_registry import tool_registry
import src.tools.ensure_tools

# Import database manager
from src.db.db_connection import db_manager

# Ensure the base directory is in the Python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.financial_agent import FinancialAgent
from src.config.config import ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH, SUPPORTED_LANGUAGES

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Financial Chatbot")

# Create uploads directory if it doesn't exist
Path("uploads").mkdir(exist_ok=True)

# Helper function to save visualizations
def save_visualization_to_filesystem(viz_data, chart_type, query):
    """Save visualization to filesystem and update session state."""
    try:
        if not viz_data:
            logger.warning("Attempted to save empty visualization data")
            return
            
        # Create a directory for visualizations if it doesn't exist
        viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        logger.info(f"Visualization directory: {viz_dir}")
        
        # Get timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sanitize chart type for filename
        if not chart_type or not isinstance(chart_type, str):
            chart_type = "chart"
        # Remove any special characters that might cause issues in filenames
        chart_type = re.sub(r'[^\w\-_]', '_', chart_type)
        
        # Create a filename with timestamp and chart type
        filename = f"{chart_type}_{timestamp}.png"
        filepath = os.path.join(viz_dir, filename)
        logger.info(f"Saving visualization to: {filepath}")
        
        # Decode the base64 image and save it
        try:
            img_data = base64.b64decode(viz_data)
        except Exception as decode_err:
            logger.error(f"Error decoding base64 data: {str(decode_err)}")
            raise ValueError(f"Invalid base64 image data: {str(decode_err)}")
        
        # Validate that we have actual image data
        if len(img_data) < 100:  # Extremely small file, likely not valid
            logger.error(f"Image data too small ({len(img_data)} bytes), might be invalid")
        
        # Write the file
        with open(filepath, "wb") as f:
            f.write(img_data)
            logger.info(f"Successfully wrote {len(img_data)} bytes to {filepath}")
        
        # Add the saved file path to the session state for reference
        if "saved_visualizations" not in st.session_state:
            st.session_state.saved_visualizations = []
        
        # Store visualization metadata
        viz_metadata = {
            "path": filepath,
            "type": chart_type,
            "timestamp": timestamp,
            "query": query,
            "size_bytes": len(img_data)
        }
        st.session_state.saved_visualizations.append(viz_metadata)
        
        # Also save to database
        if "session_id" in st.session_state:
            try:
                db_manager.save_visualization(
                    session_id=st.session_state.session_id,
                    visualization_data={
                        "chart_type": chart_type,
                        "query": query,
                        "file_path": filepath,
                        "size_bytes": len(img_data),
                        "image_data": viz_data,
                        "timestamp": datetime.now()
                    }
                )
                logger.info(f"Saved visualization to database: {chart_type}_{timestamp}")
            except Exception as db_err:
                logger.error(f"Error saving visualization to database: {str(db_err)}")
                # Continue execution even if database save fails
        
        # Show success message about saved visualization
        st.success(f"Visualization saved to {filename}")
        logger.info(f"Visualization metadata added to session state: {viz_metadata}")
        
        return filepath
    except Exception as e:
        logger.error(f"Error saving visualization: {str(e)}", exc_info=True)
        st.warning(f"Could not save visualization: {str(e)}")
        print(f"DEBUG: Error saving visualization: {str(e)}")
        return None

# Helper function to format timestamps
def format_timestamp(dt):
    """Format datetime for display."""
    now = datetime.now()
    
    # If same day, just show time
    if dt.date() == now.date():
        return dt.strftime("%I:%M %p")
    # If within a week, show day and time
    elif (now - dt).days < 7:
        return dt.strftime("%A, %I:%M %p")
    # Otherwise show full date
    else:
        return dt.strftime("%b %d, %Y, %I:%M %p")

# Set page configuration
st.set_page_config(
    page_title="FinWise -Financial Intelligence Chatbot",
    page_icon="💰",
    layout="wide",
    # initial_sidebar_state="expanded"  
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .chat-container {
        margin-bottom: 5rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #e6f3ff;
    }
    .chat-message .avatar {
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 0.25rem;
        margin-right: 1rem;
        background-size: cover;
    }
    .chat-message .avatar.user {
        background-color: #3498db;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    .chat-message .avatar.assistant {
        background-color: #2ecc71;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .timestamp {
        font-size: 0.75rem;
        color: #aaa;
        margin-bottom: 0.5rem;
    }
    .chat-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 1rem;
        box-shadow: 0 -4px 6px -1px rgba(0, 0, 0, 0.1);
        z-index: 100;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.25rem;
    }
    .url-input-container {
        margin-bottom: 1rem;
        border: 1px solid #ddd;
        border-radius: 0.25rem;
        padding: 1rem;
    }
    .history-item {
        cursor: pointer;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .history-item:hover {
        background-color: #f0f2f6;
    }
    .active-doc-item {
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        border-radius: 0.25rem;
        background-color: #f0f2f6;
    }
    .dashboard-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .tool-badge {
        font-size: 0.8rem;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        background-color: #3498db;
        color: white;
        margin-right: 0.5rem;
        display: inline-block;
    }
    .viz-container {
        margin-top: 1rem;
        border: 1px solid #eee;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    /* Enhanced chat input fixed at bottom */
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 1rem;
        box-shadow: 0 -4px 6px -1px rgba(0, 0, 0, 0.1);
        z-index: 100;
        display: flex;
        align-items: center;
        margin-left: 1rem;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Ensure main content has padding at bottom to prevent content being hidden behind fixed chat input */
    .main .block-container {
        padding-bottom: 5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'financial_agent' not in st.session_state:
    from src.agents.financial_agent import FinancialAgent  # Ensure the correct import
    st.session_state.financial_agent = FinancialAgent() 

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'language' not in st.session_state:
    st.session_state.language = "en"  # Default language is English
    
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "en"  # Default language is English
    
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
    
if 'active_documents' not in st.session_state:
    st.session_state.active_documents = {}
    
if 'session_timestamp' not in st.session_state:
    st.session_state.session_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
if 'show_url_input' not in st.session_state:
    st.session_state.show_url_input = False

if 'saved_visualizations' not in st.session_state:
    st.session_state.saved_visualizations = []

# Generate a unique session ID if not already set
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"Created new session with ID: {st.session_state.session_id}")

# Also initialize visualizations directory
visualizations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")
os.makedirs(visualizations_dir, exist_ok=True)

# App title and header
st.title("FinWise - Financial Intelligence Chatbot")
st.markdown("""
This AI assistant can process financial documents, web content, and data. 
Upload files, provide URLs, or ask questions to get insights on financial information.
""")

# Sidebar content
with st.sidebar:
    st.header("Settings & Tools")
    
    # Create tabs for different sidebar sections
    tab1, tab2, tab3, tab4 = st.tabs(["Chat Settings", "Documents", "Visualizations", "History"])
    
    with tab1:
        # Language selection
        language_options = {
            "en": "🇺🇸 English",
            "es": "🇪🇸 Spanish",
            "fr": "🇫🇷 French",
            "de": "🇩🇪 German",
            "zh": "🇨🇳 Chinese",
            "ja": "🇯🇵 Japanese"
        }
        
        selected_language = st.selectbox(
            "Select Language", 
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            index=list(language_options.keys()).index(st.session_state.language)
        )
        
        # Update language in session state if changed
        if selected_language != st.session_state.language:
            st.session_state.language = selected_language
            st.session_state.selected_language = selected_language
        
        # New chat button
        if st.button("New Chat", key="new_chat_button"):
            # Clear chat history
            st.session_state.chat_history = []
            # Generate a new session ID
            st.session_state.session_id = str(uuid.uuid4())
            logger.info(f"Started new chat session with ID: {st.session_state.session_id}")
            st.rerun()
        
        # Show session ID for debugging
        st.caption(f"Session ID: {st.session_state.session_id}")
    
    with tab2:
        # Active documents section
        st.subheader("Active Documents")
        
        # File upload area
        st.write("Upload financial documents for analysis:")
        uploaded_files = st.file_uploader(
            "Upload Files",
            accept_multiple_files=True,
            type=list(ALLOWED_EXTENSIONS.keys()),
            key="sidebar_file_uploader"
        )
        
        # Process uploaded files
        if uploaded_files:
            st.session_state.uploaded_files = []
            
            for uploaded_file in uploaded_files:
                # Create a temporary file
                UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                
                # Save the file to disk
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Add to list for processing
                st.session_state.uploaded_files.append(file_path)
                
                # Save document reference to database
                try:
                    db_manager.save_document_reference(
                        session_id=st.session_state.session_id,
                        document_data={
                            "name": uploaded_file.name,
                            "path": file_path,
                            "type": os.path.splitext(uploaded_file.name)[1],
                            "size": len(uploaded_file.getbuffer()),
                            "timestamp": datetime.now()
                        }
                    )
                    logger.info(f"Saved document reference to database: {uploaded_file.name}")
                except Exception as e:
                    logger.error(f"Error saving document reference to database: {str(e)}")
                
                # Show message about processed file
                st.success(f"File uploaded: {uploaded_file.name}")
        
        # URL input section - moved from Chat Settings to Documents tab
        st.markdown("---")
        st.write("Process URL to extract content:")
        url_input = st.text_input("Enter URL", key="sidebar_url_input")
        col1, col2 = st.columns([1, 1])
        with col1:
            extract_tables = st.checkbox("Extract Tables", value=True)
        with col2:
            download_linked = st.checkbox("Download Linked Docs", value=False)

        if st.button("Process URL", key="process_url_button"):
            if url_input:
                # Store URL in session state for processing
                st.session_state.url_query = url_input
                st.rerun()
            else:
                st.warning("Please enter a URL to process.")
        
        st.markdown("---")
        
        # Get active documents from the financial agent
        active_documents = st.session_state.financial_agent.get_active_documents()
        
        # Display active documents
        if active_documents:
            for doc_name, doc_info in active_documents.items():
                st.markdown(f"""
                <div class="active-doc-item">
                    <strong>{doc_name}</strong><br/>
                    Type: {doc_info['type']}<br/>
                    Size: {doc_info['size']} bytes<br/>
                    Has Tables: {"Yes" if doc_info['has_tables'] else "No"}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("No active documents. Upload files or process URLs to start analyzing.")
        
        # Clear documents button
        if st.button("Clear Documents", key="clear_docs_button"):
            st.session_state.financial_agent.clear_documents()
            st.session_state.active_documents = {}
            st.success("All documents cleared!")
            st.rerun()
    
    with tab3:
        # Visualizations section
        st.subheader("Saved Visualizations")
        
        # Get visualizations from database for current session
        try:
            db_visualizations = db_manager.get_visualizations(st.session_state.session_id)
            
            if db_visualizations:
                for idx, viz in enumerate(db_visualizations):
                    st.markdown(f"### {viz.get('chart_type', 'Chart')} - {format_timestamp(viz['timestamp'])}")
                    
                    # Only show if image_data is available
                    if "image_data" in viz:
                        st.image(
                            f"data:image/png;base64,{viz['image_data']}", 
                            caption=viz.get('query', 'Chart'),
                            use_container_width=True
                        )
                    else:
                        st.write("Image data not available")
                    
                    with st.expander("Query Details", expanded=False):
                        st.write(viz.get('query', 'No query info'))
            else:
                st.write("No visualizations saved in this session.")
        except Exception as e:
            logger.error(f"Error retrieving visualizations from database: {str(e)}")
            st.warning("Could not retrieve visualizations from database. Using local cache instead.")
            
            # Fallback to session state visualizations
            if "saved_visualizations" in st.session_state and st.session_state.saved_visualizations:
                for viz in st.session_state.saved_visualizations:
                    st.markdown(f"### {viz.get('type', 'Chart')} - {viz.get('timestamp', 'Unknown')}")
                    if os.path.exists(viz.get('path', '')):
                        st.image(viz['path'], caption=viz.get('query', 'Chart'), use_container_width=True)
                    else:
                        st.write("Image file not found")
            else:
                st.info("No visualizations available.")
    
    with tab4:
        # Chat history section
        st.subheader("Chat History")
        
        # Get chat history from database for current session
        try:
            db_chat_history = db_manager.get_chat_history(st.session_state.session_id)
            
            if db_chat_history:
                # Group by date
                dates = {}
                for msg in db_chat_history:
                    date_str = msg['timestamp'].strftime("%Y-%m-%d")
                    if date_str not in dates:
                        dates[date_str] = []
                    dates[date_str].append(msg)
                
                # Display by date
                for date_str, messages in dates.items():
                    with st.expander(date_str, expanded=False):
                        for msg in messages:
                            time_str = msg['timestamp'].strftime("%I:%M %p")
                            role = msg.get('role', 'unknown')
                            content = msg.get('content', 'No content')
                            st.markdown(f"**{time_str} - {role.capitalize()}**: {content[:100]}...")
                
                # Add download button for chat history
                if st.button("Download Chat History", key="download_history_button"):
                    # Convert chat history to DataFrame
                    history_data = []
                    for msg in db_chat_history:
                        history_data.append({
                            "timestamp": msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                            "role": msg.get("role", "unknown"),
                            "content": msg.get("content", ""),
                            "provider": msg.get("provider", "") if msg.get("role") == "assistant" else ""
                        })
                    
                    if history_data:
                        df = pd.DataFrame(history_data)
                        csv = df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="chat_history_{st.session_state.session_id[:8]}.csv">Download CSV File</a>'
                        st.markdown(href, unsafe_allow_html=True)
            else:
                st.write("No chat history found.")
        except Exception as e:
            logger.error(f"Error retrieving chat history from database: {str(e)}")
            st.warning("Could not retrieve chat history from database. Using local history instead.")
            
            # Fallback to session state chat history
            if st.session_state.chat_history:
                # Group messages by date for better organization
                messages_by_date = {}
                for msg in st.session_state.chat_history:
                    # Ensure timestamp exists
                    if "timestamp" not in msg:
                        msg["timestamp"] = datetime.now()
                        
                    # Get date string
                    date_str = msg["timestamp"].strftime("%Y-%m-%d")
                    
                    # Add to date group
                    if date_str not in messages_by_date:
                        messages_by_date[date_str] = []
                    messages_by_date[date_str].append(msg)
                
                # Display by date
                for date_str, messages in messages_by_date.items():
                    with st.expander(date_str, expanded=False):
                        for msg in messages:
                            time_str = msg["timestamp"].strftime("%I:%M %p")
                            role = msg.get('role', 'unknown')
                            content = msg.get('content', 'No content')
                            st.markdown(f"**{time_str} - {role.capitalize()}**: {content[:100]}...")
                
                # Add download button for local chat history
                if st.button("Download Chat History", key="download_local_history_button"):
                    # Convert chat history to DataFrame
                    history_data = []
                    for msg in st.session_state.chat_history:
                        # Ensure timestamp exists
                        if "timestamp" not in msg:
                            msg["timestamp"] = datetime.now()
                            
                        history_data.append({
                            "timestamp": msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                            "role": msg.get("role", "unknown"),
                            "content": msg.get("content", "")
                        })
                    
                    if history_data:
                        df = pd.DataFrame(history_data)
                        csv = df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="chat_history_{st.session_state.session_id[:8]}.csv">Download CSV File</a>'
                        st.markdown(href, unsafe_allow_html=True)
            else:
                st.info("No chat history available.")

    # Create additional sidebar tabs using a container
    sidebar_tabs = st.sidebar.tabs(["Document History", "Search History"])
    
    # Document History tab
    with sidebar_tabs[0]:
        st.subheader("Document Processing History")
        
        # Try to get document history from database
        try:
            doc_references = db_manager.get_document_references(st.session_state.session_id)
            doc_history = db_manager.get_document_history(st.session_state.session_id)
            
            if doc_references:
                # Create a dictionary of document names for selection
                doc_names = {doc['name']: doc for doc in doc_references}
                selected_doc = st.selectbox(
                    "Select Document", 
                    options=list(doc_names.keys()),
                    key="doc_history_selector"
                )
                
                if selected_doc:
                    # Show document details
                    selected_doc_data = doc_names[selected_doc]
                    st.write(f"**Type:** {selected_doc_data.get('type', 'Unknown')}")
                    st.write(f"**Size:** {selected_doc_data.get('size', 0)} bytes")
                    st.write(f"**Uploaded:** {format_timestamp(selected_doc_data.get('timestamp', datetime.now()))}")
                    
                    # Filter history for this document
                    doc_specific_history = [h for h in doc_history if h.get('document_name') == selected_doc]
                    
                    if doc_specific_history:
                        st.markdown("### Processing History")
                        for entry in doc_specific_history:
                            with st.expander(f"{entry.get('action', 'Action')} - {format_timestamp(entry.get('timestamp', datetime.now()))}"):
                                # Show action details
                                if 'details' in entry:
                                    for key, value in entry['details'].items():
                                        if isinstance(value, dict):
                                            st.json(value)
                                        else:
                                            st.write(f"**{key.capitalize()}:** {value}")
                
                # Add export button for document history
                if doc_history:
                    if st.button("Export Document History", key="export_doc_history"):
                        # Convert document history to DataFrame
                        history_data = []
                        for entry in doc_history:
                            if isinstance(entry.get('details'), dict):
                                # Flatten details for CSV
                                details_str = ", ".join([f"{k}: {v}" for k, v in entry.get('details', {}).items() 
                                                    if not isinstance(v, (dict, list))])
                            else:
                                details_str = str(entry.get('details', ''))
                                
                            history_data.append({
                                "timestamp": entry.get('timestamp', datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
                                "document_name": entry.get('document_name', ''),
                                "action": entry.get('action', ''),
                                "details": details_str
                            })
                        
                        if history_data:
                            df = pd.DataFrame(history_data)
                            csv = df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="document_history_{st.session_state.session_id[:8]}.csv">Download CSV File</a>'
                            st.markdown(href, unsafe_allow_html=True)
            else:
                st.info("No document history available. Upload documents to start tracking their processing.")
        except Exception as e:
            logger.error(f"Error retrieving document history from database: {str(e)}")
            st.warning("Could not retrieve document history from database.")
    
    # Search History tab
    with sidebar_tabs[1]:
        st.subheader("Web Search History")
        
        # Try to get search history from database
        try:
            search_history = db_manager.get_search_history(st.session_state.session_id)
            
            if search_history:
                for entry in search_history:
                    with st.expander(f"{entry.get('query', 'Search')} - {format_timestamp(entry.get('timestamp', datetime.now()))}"):
                        # Show query details
                        st.write(f"**Query:** {entry.get('query', '')}")
                        
                        # Show sources if available
                        if 'sources' in entry and entry['sources']:
                            st.markdown("#### Sources:")
                            for source in entry['sources']:
                                st.markdown(f"- [{source.get('title', 'Source')}]({source.get('url', '#')})")
                        
                        # Show summary if available
                        if 'summary' in entry and entry['summary']:
                            with st.expander("View Summary", expanded=False):
                                st.markdown(entry['summary'])
                
                # Add export button for search history
                if st.button("Export Search History", key="export_search_history"):
                    # Convert search history to DataFrame
                    history_data = []
                    for entry in search_history:
                        # Count sources
                        source_count = len(entry.get('sources', []))
                        source_list = [s.get('url', '') for s in entry.get('sources', [])]
                            
                        history_data.append({
                            "timestamp": entry.get('timestamp', datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
                            "query": entry.get('query', ''),
                            "sources_count": source_count,
                            "sources": ", ".join(source_list[:3])  # Limit to first 3 sources
                        })
                    
                    if history_data:
                        df = pd.DataFrame(history_data)
                        csv = df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="search_history_{st.session_state.session_id[:8]}.csv">Download CSV File</a>'
                        st.markdown(href, unsafe_allow_html=True)
            else:
                st.info("No search history available. Perform web searches to see history here.")
        except Exception as e:
            logger.error(f"Error retrieving search history from database: {str(e)}")
            st.warning("Could not retrieve search history from database.")

    # About section
    with st.expander("About"):
        st.markdown("""
        ### Financial Intelligence Chatbot
        
        This AI assistant can:
        - Process CSV, Excel, PDF, and Word documents
        - Extract data from financial websites
        - Analyze trends and generate insights
        - Answer questions about financial data
        - Summarize financial information
        - Translate content to multiple languages
        
        © 2025 FinWiseFinancial Intelligence Chatbot
        """)

# Main chat interface
chat_container = st.container()

# Remove URL Input area from main content as it's now in the sidebar
if "show_url_input" in st.session_state:
    # Remove this variable as we no longer need the toggle
    del st.session_state.show_url_input

# Chat area
with chat_container:
    # Display chat history (from session state)
    for i, message in enumerate(st.session_state.chat_history):
        # Add timestamps if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now()
        
        if message["role"] == "user":
            with st.chat_message("user"):
                # Add formatted timestamp
                st.markdown(f"<div class='timestamp'>{format_timestamp(message['timestamp'])}</div>", 
                           unsafe_allow_html=True)
                st.write(message["content"])
        else:  # Assistant message
            with st.chat_message("assistant"):
                # Add formatted timestamp
                st.markdown(f"<div class='timestamp'>{format_timestamp(message['timestamp'])}</div>", 
                           unsafe_allow_html=True)
                st.write(message["content"])

# Move chat input to the bottom and make it fixed using CSS
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
# User input area - fixed at bottom
user_input = st.chat_input("Ask me something about your financial data...")
st.markdown('</div>', unsafe_allow_html=True)

# Handle URL processing if previously submitted
if "url_query" in st.session_state:
    url_query = st.session_state.url_query
    
    # Create a message placeholder
    message_placeholder = st.empty()
    message_placeholder.chat_message("user").write(f"Process URL: {url_query}")
    
    # Add to chat history
    user_timestamp = datetime.now()
    st.session_state.chat_history.append({"role": "user", "content": f"Process URL: {url_query}", "timestamp": user_timestamp})
    
    # Save to database
    try:
        db_manager.save_chat_message(
            session_id=st.session_state.session_id,
            message={
                "role": "user",
                "content": f"Process URL: {url_query}",
                "timestamp": user_timestamp,
                "type": "url_query"
            }
        )
    except Exception as e:
        logger.error(f"Error saving chat message to database: {str(e)}")
    
    # Process the query
    with st.spinner("Processing URL..."):
        # Convert this into an asyncio task and await it
        async def process_query():
            # Get language from session state safely
            language = st.session_state.get("selected_language", "en")
            return await st.session_state.financial_agent.process_query(
                user_query=f"Analyze this URL: {url_query}",
                language=language
            )

        # Run async code
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(process_query())
        
        # Add debug logging to check response structure
        print("DEBUG: Response keys:", response.keys())
        
        # Display assistant message with tools used and provider info
        tools_used = ""
        if "tools_used" in response and response["tools_used"]:
            tools_list = ", ".join(response["tools_used"])
            tools_used = f"<div style='font-size: 0.8rem; color: #888; margin-top: 0.5rem;'>Tools used: {tools_list}</div>"
        
        # Add provider information if available
        provider_info = ""
        if "provider_used" in response:
            provider_name = response["provider_used"].capitalize()
            provider_info = f"<div style='font-size: 0.8rem; color: #888; margin-top: 0.5rem;'>Powered by: {provider_name}</div>"
        
        # Add assistant message to chat history
        assistant_timestamp = datetime.now()
        st.session_state.chat_history.append({"role": "assistant", "content": response["text"], "timestamp": assistant_timestamp})
        
        # Save to database
        db_manager.save_chat_message(
            session_id=st.session_state.session_id,
            message={
                "role": "assistant",
                "content": response["text"],
                "timestamp": assistant_timestamp,
                "provider": response.get("provider_used", "unknown"),
                "tools_used": response.get("tools_used", [])
            }
        )
        
        st.chat_message("assistant").markdown(
            f"""
            {response["text"]}
            {tools_used}
            {provider_info}
            """,
            unsafe_allow_html=True
        )
        
        # Display CSV analysis results if present
        if "csv_analysis" in response:
            print("DEBUG: CSV analysis found in response")
            analysis_result = response["csv_analysis"]
            
            if "error" in analysis_result:
                st.error(f"CSV Analysis Error: {analysis_result['error']}")
            else:
                # Display summary
                if "summary" in analysis_result:
                    st.info(analysis_result["summary"])
                
                # Display HTML results if available
                if "results" in analysis_result and isinstance(analysis_result["results"], str) and analysis_result["results"].startswith("<div"):
                    st.components.v1.html(analysis_result["results"], height=400, scrolling=True)
                # Display other results if available
                elif "results" in analysis_result:
                    st.json(analysis_result["results"])
                
                # Display code if requested with a toggle
                if "python_code" in analysis_result:
                    with st.expander("View Analysis Code", expanded=False):
                        st.code(analysis_result["python_code"], language="python")
        
        # Display any visualizations if present
        if "visualization" in response:
            print("DEBUG: Visualization found in response")
            print("DEBUG: Visualization keys:", response["visualization"].keys())
            
            if "visualization_data" in response["visualization"]:
                viz_data = response["visualization"]["visualization_data"]
                print(f"DEBUG: Visualization data length: {len(viz_data) if viz_data else 0}")
                
                if viz_data:
                    try:
                        # Display the visualization in Streamlit
                        st.image(f"data:image/png;base64,{viz_data}", caption="Data Visualization", use_container_width=True)
                        
                        # Display insights below the visualization
                        if "insights" in response["visualization"]:
                            with st.expander("View Data Insights", expanded=True):
                                st.markdown(response["visualization"]["insights"])
                        
                        # Display code if available (for DynamicVisualizationTool)
                        if "python_code" in response["visualization"]:
                            with st.expander("View Visualization Code", expanded=False):
                                st.code(response["visualization"]["python_code"], language="python")
                        
                        # Save visualization to filesystem and database
                        save_visualization_to_filesystem(viz_data, 
                                                     response["visualization"].get("chart_type", "chart"), 
                                                     url_query)
                        
                    except Exception as e:
                        st.error(f"Error displaying visualization: {str(e)}")
                        print(f"DEBUG: Visualization error: {str(e)}")
                else:
                    st.warning("Visualization data is empty")
            elif "error" in response["visualization"]:
                st.error(f"Visualization Error: {response['visualization']['error']}")
            else:
                st.warning("No visualization data found in response")
    
    # Clear URL to process
    del st.session_state.url_query

# Handle user input if provided
if user_input:
    # Display user message
    message_placeholder = st.empty()
    message_placeholder.chat_message("user").write(user_input)
    
    # Add to chat history
    user_timestamp = datetime.now()
    st.session_state.chat_history.append({"role": "user", "content": user_input, "timestamp": user_timestamp})
    
    # Save to database
    try:
        db_manager.save_chat_message(
            session_id=st.session_state.session_id,
            message={
                "role": "user",
                "content": user_input,
                "timestamp": user_timestamp,
                "type": "chat"
            }
        )
    except Exception as e:
        logger.error(f"Error saving chat message to database: {str(e)}")
    
    # Process any pending file uploads
    uploaded_files = []
    if hasattr(st.session_state, 'uploaded_files') and st.session_state.uploaded_files:
        uploaded_files = st.session_state.uploaded_files
        
        # Create upload directory if not exists
        UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Show message about processing files
        if uploaded_files:
            with st.spinner(f"Processing {len(uploaded_files)} files..."):
                for file_path in uploaded_files:
                    st.info(f"Processing file: {os.path.basename(file_path)}")
        
        # Clear the uploaded files after processing
        st.session_state.uploaded_files = []
    
    # Process the query
    with st.spinner("Thinking..."):
        # Convert this into an asyncio task and await it
        async def process_query():
            # Get language from session state safely
            language = st.session_state.get("selected_language", "en")
            return await st.session_state.financial_agent.process_query(
                user_query=user_input,
                uploaded_files=uploaded_files,
                language=language,
                chat_history=st.session_state.chat_history  # Pass chat history
            )

        # Run async code
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(process_query())
        
        # Add debug logging to check response structure
        print("DEBUG: Response keys:", response.keys())
        
        # Display assistant message with tools used and provider info
        tools_used = ""
        if "tools_used" in response and response["tools_used"]:
            tools_list = ", ".join(response["tools_used"])
            tools_used = f"<div style='font-size: 0.8rem; color: #888; margin-top: 0.5rem;'>Tools used: {tools_list}</div>"
        
        # Add provider information if available
        provider_info = ""
        if "provider_used" in response:
            provider_name = response["provider_used"].capitalize()
            provider_info = f"<div style='font-size: 0.8rem; color: #888; margin-top: 0.5rem;'>Powered by: {provider_name}</div>"
        
        # Add assistant message to chat history
        assistant_timestamp = datetime.now()
        st.session_state.chat_history.append({"role": "assistant", "content": response["text"], "timestamp": assistant_timestamp})
        
        # Save to database
        db_manager.save_chat_message(
            session_id=st.session_state.session_id,
            message={
                "role": "assistant",
                "content": response["text"],
                "timestamp": assistant_timestamp,
                "provider": response.get("provider_used", "unknown"),
                "tools_used": response.get("tools_used", [])
            }
        )
        
        st.chat_message("assistant").markdown(
            f"""
            <div class='timestamp'>{format_timestamp(assistant_timestamp)}</div>
            {response["text"]}
            {tools_used}
            {provider_info}
            """,
            unsafe_allow_html=True
        )
        
        # Display CSV analysis results if present
        if "csv_analysis" in response:
            print("DEBUG: CSV analysis found in response")
            analysis_result = response["csv_analysis"]
            
            if "error" in analysis_result:
                st.error(f"CSV Analysis Error: {analysis_result['error']}")
            else:
                # Display summary
                if "summary" in analysis_result:
                    st.info(analysis_result["summary"])
                
                # Display HTML results if available
                if "results" in analysis_result and isinstance(analysis_result["results"], str) and analysis_result["results"].startswith("<div"):
                    st.components.v1.html(analysis_result["results"], height=400, scrolling=True)
                # Display other results if available
                elif "results" in analysis_result:
                    st.json(analysis_result["results"])
                
                # Display code if requested with a toggle
                if "python_code" in analysis_result:
                    with st.expander("View Analysis Code", expanded=False):
                        st.code(analysis_result["python_code"], language="python")
        
        # Display any visualizations if present
        if "visualization" in response:
            print("DEBUG: Visualization found in response")
            print("DEBUG: Visualization keys:", response["visualization"].keys())
            
            if "visualization_data" in response["visualization"]:
                viz_data = response["visualization"]["visualization_data"]
                print(f"DEBUG: Visualization data length: {len(viz_data) if viz_data else 0}")
                
                if viz_data:
                    try:
                        # Display the visualization in Streamlit
                        st.image(f"data:image/png;base64,{viz_data}", caption="Data Visualization", use_container_width=True)
                        
                        # Display insights below the visualization
                        if "insights" in response["visualization"]:
                            with st.expander("View Data Insights", expanded=True):
                                st.markdown(response["visualization"]["insights"])
                        
                        # Display code if available (for DynamicVisualizationTool)
                        if "python_code" in response["visualization"]:
                            with st.expander("View Visualization Code", expanded=False):
                                st.code(response["visualization"]["python_code"], language="python")
                        
                        # Save visualization to filesystem and database
                        save_visualization_to_filesystem(viz_data, 
                                                     response["visualization"].get("chart_type", "chart"), 
                                                     user_input)
                        
                    except Exception as e:
                        st.error(f"Error displaying visualization: {str(e)}")
                        print(f"DEBUG: Visualization error: {str(e)}")
                else:
                    st.warning("Visualization data is empty")
            elif "error" in response["visualization"]:
                st.error(f"Visualization Error: {response['visualization']['error']}")
            else:
                st.warning("No visualization data found in response")

        # Display search API results if present
        if "search_api" in response:
            print("DEBUG: Search API results found in response")
            search_result = response["search_api"]
            
            if "error" in search_result:
                st.error(f"Search Error: {search_result['error']}")
            else:
                # Save search history to database
                try:
                    db_manager.save_search_history(
                        session_id=st.session_state.session_id,
                        search_data=search_result
                    )
                    logger.info(f"Saved search history for query: {search_result.get('query', 'Unknown')}")
                except Exception as e:
                    logger.error(f"Error saving search history to database: {str(e)}")
                
                # Display sources
                if "sources" in search_result and search_result["sources"]:
                    with st.expander("View Sources", expanded=False):
                        st.markdown("### Information Sources")
                        for source in search_result["sources"]:
                            st.markdown(f"- [{source.get('title', 'Source')}]({source.get('url', '#')})")
                
                # Display summary
                if "summary" in search_result and search_result["summary"]:
                    with st.expander("View Full Search Results", expanded=False):
                        st.markdown(search_result["summary"])
