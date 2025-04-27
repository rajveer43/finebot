import streamlit as st
import pandas as pd
import os
import asyncio
import tempfile
import logging
from datetime import datetime

from src.db.postgres_connection import postgres_manager
from src.tools.data_upload_tool import data_upload_tool
# from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def render_data_upload_page():
    """Render the data upload and SQL query page."""
    st.title("Data Management")
    
    # Initialize session state variables if they don't exist
    if "pg_connected" not in st.session_state:
        st.session_state.pg_connected = False
    
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    # Sidebar for PostgreSQL connection
    with st.sidebar:
        st.subheader("PostgreSQL Connection")
        
        with st.form("postgres_connection_form"):
            username = st.text_input("Username", value=os.getenv("POSTGRES_USERNAME", "postgres"))
            password = st.text_input("Password", type="password", value=os.getenv("POSTGRES_PASSWORD", ""))
            hostname = st.text_input("Host", value=os.getenv("POSTGRES_HOST", "localhost"))
            port = st.text_input("Port", value=os.getenv("POSTGRES_PORT", "5432"))
            database = st.text_input("Database", value=os.getenv("POSTGRES_DB", "finbot"))
            
            connect_button = st.form_submit_button("Connect")
        
        if connect_button:
            try:
                # Construct PostgreSQL URI
                postgres_uri = f"postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{database}"
                
                # Create a new PostgreSQL manager with the provided URI
                pg_manager = postgres_manager.__class__(uri=postgres_uri)
                
                if pg_manager.connected:
                    # Update the global instance
                    postgres_manager.uri = pg_manager.uri
                    postgres_manager.engine = pg_manager.engine
                    postgres_manager.connected = pg_manager.connected
                    postgres_manager.metadata = pg_manager.metadata
                    
                    st.session_state.pg_connected = True
                    st.success(f"Connected to PostgreSQL database: {database}")
                else:
                    st.error(f"Failed to connect to PostgreSQL database")
                    st.session_state.pg_connected = False
            except Exception as e:
                st.error(f"Error connecting to PostgreSQL: {str(e)}")
                st.session_state.pg_connected = False
        
        # Show database tables when connected
        if st.session_state.pg_connected:
            st.subheader("Database Tables")
            tables = postgres_manager.list_tables()
            
            if tables:
                for table in tables:
                    with st.expander(f"📊 {table}"):
                        # Get table info
                        table_info = postgres_manager.get_table_info(table)[0]
                        
                        # Display columns
                        st.write("**Columns:**")
                        cols_df = pd.DataFrame([
                            {"Column": col["name"], "Type": col["type"]} 
                            for col in table_info["columns"]
                        ])
                        st.dataframe(cols_df, use_container_width=True)
                        
                        # Add button to preview data
                        if st.button(f"Preview {table} data", key=f"preview_{table}"):
                            try:
                                query_result = postgres_manager.execute_query(f"SELECT * FROM {table} LIMIT 10")
                                if query_result:
                                    st.dataframe(pd.DataFrame(query_result), use_container_width=True)
                                else:
                                    st.info("No data found or error executing query")
                            except Exception as e:
                                st.error(f"Error previewing data: {str(e)}")
            else:
                st.info("No tables available in the database. Upload data to create tables.")
    
    # Main content area with tabs
    if st.session_state.pg_connected:
        tab1, tab2 = st.tabs(["Upload Data", "Query Data"])
        
        # Tab 1: Data Upload
        with tab1:
            st.header("Upload Excel or CSV Files")
            
            uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
            
            if uploaded_file is not None:
                # Preview the file
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        file_type = "CSV"
                    else:
                        df = pd.read_excel(uploaded_file)
                        file_type = "Excel"
                    
                    st.subheader(f"Preview: {uploaded_file.name}")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # File stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", len(df))
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        st.metric("File Type", file_type)
                    
                    # Upload form
                    with st.form("upload_form"):
                        default_table_name = os.path.splitext(uploaded_file.name)[0].lower().replace(" ", "_")
                        table_name = st.text_input("Table Name", value=default_table_name)
                        
                        if_exists_option = st.selectbox(
                            "If Table Exists",
                            options=["replace", "append", "fail"],
                            index=0,
                            help="Choose what to do if the table already exists"
                        )
                        
                        submit_button = st.form_submit_button("Upload to Database")
                    
                    if submit_button:
                        with st.spinner("Uploading data to PostgreSQL..."):
                            # Reset file pointer to beginning
                            uploaded_file.seek(0)
                            
                            # Call the data upload tool
                            result = asyncio.run(data_upload_tool.save_to_database(
                                file=uploaded_file,
                                table_name=table_name, 
                                if_exists=if_exists_option
                            ))
                            
                            if result["status"] == "success":
                                st.success(result["message"])
                                
                                # Show updated info
                                if "result" in result:
                                    st.info(f"Uploaded {result['result']['rows_count']} rows and {result['result']['columns_count']} columns to table '{table_name}'")
                            else:
                                st.error(f"Error: {result['message']}")
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        # Tab 2: Query Data
        with tab2:
            st.header("Query Your Data")
            
            # Check if any tables exist
            tables = postgres_manager.list_tables()
            if not tables:
                st.warning("No tables found in the database. Please upload data first.")
            else:
                st.write("Ask questions about your data using natural language.")
                
                # Query examples
                with st.expander("Example Questions"):
                    st.markdown("""
                    - Show me the first 5 rows from the table X
                    - What is the average value of column Y?
                    - How many unique values are in column Z?
                    - What is the correlation between columns A and B?
                    - Show me the distribution of values in column C
                    """)
                
                # Query input
                query = st.text_area("Your Question", 
                                    placeholder="Ask a question about your data...")
                
                if st.button("Run Query"):
                    if query:
                        with st.spinner("Analyzing your query..."):
                            try:
                                # Use the OpenAI API key from environment
                                api_key = os.getenv("GROQ_API_KEY")
                                
                                # Execute the query
                                response = asyncio.run(data_upload_tool.query_database(
                                    query=query,
                                    llm_api_key=api_key
                                ))
                                
                                # Add to history
                                st.session_state.query_history.append({
                                    "query": query,
                                    "response": response,
                                    "timestamp": datetime.now()
                                })
                                
                                # Display results
                                if response["status"] == "success":
                                    st.markdown("### Results")
                                    st.markdown(response["result"]["output"])
                                else:
                                    st.error(f"Error: {response['message']}")
                            
                            except Exception as e:
                                st.error(f"Error executing query: {str(e)}")
                    else:
                        st.warning("Please enter a question to query your data.")
                
                # Show query history
                if st.session_state.query_history:
                    with st.expander("Query History", expanded=False):
                        for i, item in enumerate(reversed(st.session_state.query_history)):
                            st.markdown(f"**Query {i+1}:** {item['query']}  \n"
                                        f"**Time:** {item['timestamp'].strftime('%H:%M:%S')}  \n"
                                        f"**Response:**  \n{item['response']['result']['output'] if item['response']['status'] == 'success' else item['response']['message']}")
                            st.divider()
    
    else:
        # Not connected to database
        st.info("Please connect to a PostgreSQL database using the sidebar form.")
        
        # Show example capabilities
        st.markdown("""
        ## Data Upload & Query Features
        
        This tool allows you to:
        1. **Upload CSV and Excel files** directly to a PostgreSQL database
        2. **Query your data using natural language**
        3. **Analyze and visualize** the database content
        4. Perform **text-to-SQL** conversions automatically
        
        Connect to your PostgreSQL database to get started!
        """)
        
        # Example image
        st.image("https://miro.medium.com/max/1400/1*J_Q12t5Z90AIIp_tNfuOdQ.png", 
                 caption="Example: Natural Language to SQL Conversion") 