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

# Import tool registry and ensure tools are registered
from src.tools.tool_registry import tool_registry
import src.tools.ensure_tools

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
        
        # Show success message about saved visualization
        st.success(f"Visualization saved to {filename}")
        logger.info(f"Visualization metadata added to session state: {viz_metadata}")
        
        return filepath
    except Exception as e:
        logger.error(f"Error saving visualization: {str(e)}", exc_info=True)
        st.warning(f"Could not save visualization: {str(e)}")
        print(f"DEBUG: Error saving visualization: {str(e)}")
        return None

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
    tab1, tab2, tab3 = st.tabs(["Chat Settings", "Documents", "Visualizations"])
    
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
            index=0
        )
        st.session_state.language = selected_language
    
    with tab2:
        # File upload section
        st.subheader("Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Financial Documents", 
            accept_multiple_files=True,
            type=list(ALLOWED_EXTENSIONS.keys())
        )
        
        # URL input toggle
        if st.button("➕ Add URL" if not st.session_state.show_url_input else "➖ Hide URL Input"):
            st.session_state.show_url_input = not st.session_state.show_url_input
        
        # URL input section
        if st.session_state.show_url_input:
            with st.expander("URL Input", expanded=True):
                url_input = st.text_input("Enter financial website URL:", placeholder="https://example.com/financial-report")
                download_linked = st.checkbox("Download linked documents", value=True)
                
                if st.button("Process URL"):
                    if url_input and re.match(r'https?://\S+', url_input):
                        # Add URL to query
                        url_query = f"Extract information from this URL: {url_input}"
                        if download_linked:
                            url_query += " and download any linked documents."
                        
                        # Add to chat history
                        st.session_state.chat_history.append({"role": "user", "content": url_query})
                        
                        # Trigger processing on rerun
                        st.session_state.url_to_process = url_input
                        st.session_state.url_download_linked = download_linked
                        st.rerun()
                    else:
                        st.error("Please enter a valid URL starting with http:// or https://")
    
    with tab3:
        # Visualizations Dashboard
        st.subheader("Visualizations Dashboard")
        
        # Show visualization stats
        if "saved_visualizations" in st.session_state:
            num_viz = len(st.session_state.saved_visualizations)
            st.markdown(f"**{num_viz} visualization{'s' if num_viz != 1 else ''} saved**")
            
            # Group visualizations by type
            viz_types = {}
            for viz in st.session_state.saved_visualizations:
                chart_type = viz.get("type", "unknown")
                if chart_type in viz_types:
                    viz_types[chart_type] += 1
                else:
                    viz_types[chart_type] = 1
            
            # Show chart types distribution
            if viz_types:
                st.write("Chart Types:")
                for chart_type, count in viz_types.items():
                    st.write(f"- {chart_type}: {count}")
            
            if num_viz > 0:
                # Browser section
                with st.expander("Browse Visualizations", expanded=True):
                    # Create a dropdown to select from saved visualizations
                    viz_options = {}
                    for i, viz in enumerate(st.session_state.saved_visualizations):
                        # Truncate query if too long for display
                        query_text = viz["query"]
                        if len(query_text) > 30:
                            query_text = query_text[:27] + "..."
                        
                        # Format display text
                        viz_options[i] = f"{viz['type']} - {viz['timestamp']} - {query_text}"
                    
                    viz_filter = st.selectbox(
                        "Filter by Chart Type",
                        ["All Types"] + list(viz_types.keys())
                    )
                    
                    # Filter visualizations based on selected type
                    filtered_indices = list(viz_options.keys())
                    if viz_filter != "All Types":
                        filtered_indices = [i for i, viz in enumerate(st.session_state.saved_visualizations) 
                                          if viz.get("type") == viz_filter]
                    
                    if filtered_indices:
                        selected_viz_idx = st.selectbox(
                            "Select Visualization",
                            filtered_indices,
                            format_func=lambda x: viz_options[x]
                        )
                        
                        # Display the selected visualization
                        if selected_viz_idx is not None:
                            selected_viz = st.session_state.saved_visualizations[selected_viz_idx]
                            
                            # Check if file exists
                            if os.path.exists(selected_viz["path"]):
                                # Display the image
                                st.image(selected_viz["path"], caption=f"{selected_viz['type']} Chart")
                                
                                # Display metadata
                                st.write("**Details:**")
                                st.caption(f"Created: {selected_viz['timestamp']}")
                                st.caption(f"Query: {selected_viz['query']}")
                                
                                # Add option to delete the visualization
                                if st.button("Delete This Visualization"):
                                    try:
                                        # Remove the file
                                        os.remove(selected_viz["path"])
                                        # Remove from session state
                                        st.session_state.saved_visualizations.pop(selected_viz_idx)
                                        st.success("Visualization deleted")
                                        # Force refresh
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error deleting visualization: {str(e)}")
                            else:
                                st.warning("Visualization file not found.")
                    else:
                        st.info(f"No visualizations of type '{viz_filter}' found.")
                
                # Export section
                with st.expander("Export Visualizations"):
                    export_all = st.button("Prepare All Visualizations for Download")
                    
                    if export_all:
                        for i, viz in enumerate(st.session_state.saved_visualizations):
                            if os.path.exists(viz["path"]):
                                filename = os.path.basename(viz["path"])
                                with open(viz["path"], "rb") as f:
                                    viz_bytes = f.read()
                                    st.download_button(
                                        label=f"Download {filename}",
                                        data=viz_bytes,
                                        file_name=filename,
                                        mime="image/png",
                                        key=f"download_{i}"
                                    )
                
                # Cleanup option
                with st.expander("Maintenance"):
                    if st.button("Clear All Visualizations"):
                        confirm = st.checkbox("I understand this will delete all saved visualizations")
                        
                        if confirm and st.button("Confirm Delete All"):
                            try:
                                # Delete all visualization files
                                for viz in st.session_state.saved_visualizations:
                                    if os.path.exists(viz["path"]):
                                        os.remove(viz["path"])
                                
                                # Clear the session state
                                st.session_state.saved_visualizations = []
                                st.success("All visualizations deleted")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting visualizations: {str(e)}")
            else:
                st.info("No visualizations saved yet. Create visualizations by asking queries about your data.")
    
    # Session information
    st.subheader("Session Info")
    st.info(f"Session started: {st.session_state.session_timestamp}")
    
    # Active documents section
    active_docs = st.session_state.financial_agent.get_active_documents()
    st.session_state.active_documents = active_docs  # Update session state with current active documents
    
    if active_docs:
        st.subheader(f"Active Documents ({len(active_docs)})")
        for name, info in active_docs.items():
            st.markdown(f"📄 **{name}**", help=f"Path: {info['path']}")
    
    # Clear buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
    
    with col2:
        if st.button("Clear Documents"):
            st.session_state.processed_files = []
            st.session_state.financial_agent.clear_documents()
            st.success("Documents cleared!")
    
    # Export chat history
    if st.session_state.chat_history:
        if st.download_button(
            label="Export Chat History",
            data=json.dumps(st.session_state.chat_history, indent=2),
            file_name=f"financial_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        ):
            st.success("Chat history exported!")
    
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
        
        © 2025 Financial Intelligence Chatbot
        """)

# Main chat area
chat_container = st.container()

# Process uploaded files
if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        for uploaded_file in uploaded_files:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                # Write the uploaded file to the temporary file
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Add to list of processed files
            st.session_state.processed_files.append(tmp_path)
            
            # Show success message
            st.success(f"Uploaded: {uploaded_file.name}")

# Input area at the bottom
with st.container():
    # Chat input
    user_input = st.chat_input("Type your message:", key="user_input")

# Process URL if requested
if hasattr(st.session_state, 'url_to_process'):
    url_query = f"Extract information from this URL: {st.session_state.url_to_process}"
    if hasattr(st.session_state, 'url_download_linked') and st.session_state.url_download_linked:
        url_query += " and download any linked documents."
    
    with chat_container:
        # Display user message
        st.chat_message("user").markdown(url_query)
        
        # Process with spinner
        with st.spinner("Processing URL..."):
            # Process user query
            async def process_query():
                response = await st.session_state.financial_agent.process_query(
                    user_query=url_query,
                    uploaded_files=st.session_state.processed_files,
                    language=st.session_state.language
                )
                return response
                
            # Run async code
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(process_query())
            
            # Clear processed files to avoid reprocessing
            st.session_state.processed_files = []
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response["text"]})
            
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
                            
                            # Save visualization to filesystem
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
    del st.session_state.url_to_process
    if hasattr(st.session_state, 'url_download_linked'):
        del st.session_state.url_download_linked

# Process user input
if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with chat_container:
        # Display user message
        st.chat_message("user").markdown(user_input)
        
        # Process with spinner
        with st.spinner("Thinking..."):
            # Process user query
            async def process_query():
                response = await st.session_state.financial_agent.process_query(
                    user_query=user_input,
                    uploaded_files=st.session_state.processed_files,
                    language=st.session_state.language
                )
                return response
                
            # Run async code
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(process_query())
            
            # Clear processed files to avoid reprocessing
            st.session_state.processed_files = []
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response["text"]})
            
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
                            
                            # Save visualization to filesystem
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

# Display chat history
with chat_container:
    # Reverse the history to show latest messages at the bottom
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.chat_message("user").markdown(content)
        else:
            st.chat_message("assistant").markdown(content)

if __name__ == "__main__":
    st.sidebar.markdown(f"© {datetime.now().year} Financial Intelligence Chatbot")

async def process_user_message():
    """Process the user message and generate a response."""
    user_query = st.session_state.user_query.strip()
    
    if not user_query:
        return
    
    with st.chat_message("user"):
        st.markdown(user_query)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        # Process uploaded files if any
        uploaded_files = []
        if st.session_state.uploaded_files:
            for uploaded_file in st.session_state.uploaded_files:
                # Save the file to a temporary location
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                uploaded_files.append(file_path)
            
            # Clear the uploaded files after processing
            st.session_state.uploaded_files = []
        
        # Process the query
        response = await st.session_state.financial_agent.process_query(
            user_query=user_query,
            uploaded_files=uploaded_files,
            language=st.session_state.selected_language
        )
        
        # Display the response text
        message_placeholder.markdown(response["text"])
        
        # Display provider information if available
        if "provider_used" in response:
            provider_name = response["provider_used"].capitalize()
            st.caption(f"Powered by: {provider_name}")
        
        # Display any visualizations if present
        if "visualization" in response and "visualization_data" in response["visualization"]:
            viz_data = response["visualization"]["visualization_data"]
            if viz_data:
                # Display the visualization in Streamlit
                st.image(f"data:image/png;base64,{viz_data}", caption="Data Visualization", use_container_width=True)
                
                # Save the visualization to the filesystem
                try:
                    import base64
                    import os
                    from datetime import datetime
                    
                    # Create a directory for visualizations if it doesn't exist
                    viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")
                    os.makedirs(viz_dir, exist_ok=True)
                    
                    # Get chart type and timestamp for the filename
                    chart_type = response["visualization"].get("chart_type", "chart")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Create a filename with timestamp and chart type
                    filename = f"{chart_type}_{timestamp}.png"
                    filepath = os.path.join(viz_dir, filename)
                    
                    # Decode the base64 image and save it
                    img_data = base64.b64decode(viz_data)
                    with open(filepath, "wb") as f:
                        f.write(img_data)
                    
                    # Add the saved file path to the session state for reference
                    if "saved_visualizations" not in st.session_state:
                        st.session_state.saved_visualizations = []
                    
                    # Store visualization metadata
                    viz_metadata = {
                        "path": filepath,
                        "type": chart_type,
                        "timestamp": timestamp,
                        "query": user_query
                    }
                    st.session_state.saved_visualizations.append(viz_metadata)
                    
                    # Show success message about saved visualization
                    st.success(f"Visualization saved to {filename}")
                except Exception as e:
                    st.warning(f"Could not save visualization: {str(e)}")
                
                # Display insights below the visualization
                if "insights" in response["visualization"]:
                    with st.expander("View Data Insights", expanded=True):
                        st.markdown(response["visualization"]["insights"])
                
                # Show data summary
                if "data_summary" in response["visualization"]:
                    with st.expander("Data Summary", expanded=False):
                        st.json(response["visualization"]["data_summary"])
