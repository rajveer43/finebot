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

# Set page configuration
st.set_page_config(
    page_title="Financial Intelligence Chatbot",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
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

# App title and header
st.title("Financial Intelligence Chatbot")
st.markdown("""
This AI assistant can process financial documents, web content, and data. 
Upload files, provide URLs, or ask questions to get insights on financial information.
""")

# Sidebar content
with st.sidebar:
    st.header("Settings & Tools")
    
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
            st.session_state.financial_agent.clear_history()
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
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_area("Type your message:", key="user_input", height=100)
    with col2:
        submit_button = st.button("Send", key="submit")

# Process URL if requested
if hasattr(st.session_state, 'url_to_process'):
    url_query = f"Extract information from this URL: {st.session_state.url_to_process}"
    if hasattr(st.session_state, 'url_download_linked') and st.session_state.url_download_linked:
        url_query += " and download any linked documents."
    
    with chat_container:
        # Display user message
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar user">U</div>
                <div class="message">{url_query}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
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
            
            # Display assistant message
            st.markdown(
                f"""
                <div class="chat-message assistant">
                    <div class="avatar assistant">AI</div>
                    <div class="message">{response["text"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Clear URL to process
    del st.session_state.url_to_process
    if hasattr(st.session_state, 'url_download_linked'):
        del st.session_state.url_download_linked

# Process user input
if submit_button and user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with chat_container:
        # Display user message
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar user">U</div>
                <div class="message">{user_input}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
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
            
            # Display assistant message with tools used
            tools_used = ""
            if "tools_used" in response and response["tools_used"]:
                tools_list = ", ".join(response["tools_used"])
                tools_used = f"<div style='font-size: 0.8rem; color: #888; margin-top: 0.5rem;'>Tools used: {tools_list}</div>"
            
            st.markdown(
                f"""
                <div class="chat-message assistant">
                    <div class="avatar assistant">AI</div>
                    <div class="message">
                        {response["text"]}
                        {tools_used}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# Display chat history
with chat_container:
    # Reverse the history to show latest messages at the bottom
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(
                f"""
                <div class="chat-message user">
                    <div class="avatar user">U</div>
                    <div class="message">{content}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="chat-message assistant">
                    <div class="avatar assistant">AI</div>
                    <div class="message">{content}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    st.sidebar.markdown(f"© {datetime.now().year} Financial Intelligence Chatbot")
