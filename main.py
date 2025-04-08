import os
import tempfile
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from datetime import datetime
import langdetect # type: ignore
import time
from typing import List, Dict, Any, Optional, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# CrewAI imports
from crewai import Agent, Task, Crew, Process # type: ignore
from langchain.tools import BaseTool, StructuredTool
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "openai api key"

# Create temporary directory for file uploads
if not os.path.exists('temp_uploads'):
    os.makedirs('temp_uploads')

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# LLM definitions
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
fast_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

class CSVAnalyzerInput(BaseModel):
    file_path: str = Field(description="Path to the CSV file")
    analysis_type: str = Field(default="summary", description="Type of analysis (summary, trends, statistics)")

class ExcelProcessorInput(BaseModel):
    file_path: str = Field(description="Path to the Excel file")
    analysis_type: str = Field(default="summary", description="Type of analysis (summary, trends, statistics)")

class TextSummarizerInput(BaseModel):
    text: str = Field(description="Text to be summarized")
    summary_type: str = Field(default="brief", description="Type of summary (brief, detailed)")

class TableQuerierInput(BaseModel):
    query: str = Field(description="Query to execute on the table data")
    table_name: str = Field(description="Name of the table to query")

class VisualizationInput(BaseModel):
    data: List[Dict[str, Any]] = Field(description="Data to visualize")
    chart_type: str = Field(default="bar", description="Type of chart (bar, line, scatter, pie, heatmap)")


class CSVAnalyzerTool(BaseTool):
    name = "csv_analyzer"
    description = "Analyzes CSV files to extract insights, trends, and statistics"
    args_schema: Type[BaseModel] = CSVAnalyzerInput

    def _run(self, file_path: str, analysis_type: str = "summary") -> Dict[str, Any]:
        try:
            df = pd.read_csv(file_path)
            
            if analysis_type == "summary":
                summary = {
                    "columns": df.columns.tolist(),
                    "shape": df.shape,
                    "head": df.head(5).to_dict(),
                    "info": df.describe().to_dict()
                }
                return summary
            
            elif analysis_type == "trends":
                plt.figure(figsize=(10, 6))
                date_cols = [col for col in df.columns if 'date' in col.lower()]
                if date_cols:
                    date_col = date_cols[0]
                    df[date_col] = pd.to_datetime(df[date_col])
                    df.set_index(date_col, inplace=True)
                
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 4:
                    numeric_cols = numeric_cols[:4]
                
                for col in numeric_cols:
                    plt.plot(df.index if len(date_cols) > 0 else range(len(df)), df[col], label=col)
                
                plt.title(f"Trends Analysis")
                plt.legend()
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                image = base64.b64encode(buffer.read()).decode('utf-8')
                
                return {
                    "type": "trend_analysis",
                    "image": image,
                    "columns_analyzed": numeric_cols.tolist()
                }
                
            elif analysis_type == "statistics":
                return {
                    "mean": df.mean().to_dict(),
                    "median": df.median().to_dict(),
                    "std": df.std().to_dict(),
                    "correlation": df.corr().to_dict()
                }
            
            else:
                return {"error": f"Analysis type '{analysis_type}' not supported"}
                
        except Exception as e:
            return {"error": str(e)}

    def _arun(self, file_path: str, analysis_type: str = "summary"):
        return self._run(file_path, analysis_type)

class ExcelProcessorTool(BaseTool):
    name: str = "excel_processor"
    description: str = "Processes Excel files to extract sheets, tables, and insights"
    args_schema: Type[BaseModel] = ExcelProcessorInput

    def _run(self, file_path: str, sheet_name: str = None, analysis_type: str = "summary") -> Dict[str, Any]:
        """
        Process an Excel file
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet to analyze (optional)
            analysis_type: Type of analysis (summary, tables, statistics)
            
        Returns:
            Analysis results as a dictionary
        """
        try:
            # Get list of sheet names
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            if sheet_name is not None and sheet_name not in sheet_names:
                return {"error": f"Sheet '{sheet_name}' not found in the Excel file"}
            
            # If no sheet specified, use the first one
            if sheet_name is None:
                sheet_name = sheet_names[0]
            
            # Load the sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            if analysis_type == "summary":
                # Summary of Excel file
                return {
                    "file_path": file_path,
                    "sheet_names": sheet_names,
                    "current_sheet": sheet_name,
                    "columns": df.columns.tolist(),
                    "rows": len(df),
                    "sample_data": df.head(5).to_dict()
                }
                
            elif analysis_type == "tables":
                # Extract tables from the sheet
                tables = []
                # This is a simplified approach - in real implementation, you might
                # want to use libraries like xlrd for more complex table detection
                
                # For now, we'll just return the entire sheet as a table
                tables.append({
                    "table_name": sheet_name,
                    "columns": df.columns.tolist(),
                    "rows": len(df),
                    "sample": df.head(5).to_dict()
                })
                
                return {"tables": tables}
                
            elif analysis_type == "statistics":
                # Statistical analysis
                return {
                    "mean": df.mean().to_dict(),
                    "median": df.median().to_dict(),
                    "std": df.std().to_dict(),
                    "correlation": df.corr().to_dict()
                }
                
            elif analysis_type == "visualization":
                # Create visualization for numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                if len(numeric_cols) > 0:
                    plt.figure(figsize=(12, 8))
                    
                    # Create a simple heatmap of correlations
                    corr = df[numeric_cols].corr()
                    sns.heatmap(corr, annot=True, cmap='coolwarm')
                    
                    # Convert to base64
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    image = base64.b64encode(buffer.read()).decode('utf-8')
                    
                    return {
                        "type": "correlation_heatmap",
                        "image": image,
                        "columns_analyzed": numeric_cols.tolist()
                    }
                else:
                    return {"error": "No numeric columns found for visualization"}
            
            else:
                return {"error": f"Analysis type '{analysis_type}' not supported"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def _arun(self, file_path: str, sheet_name: str = None, analysis_type: str = "summary"):
        # Async version would go here
        return self._run(file_path, sheet_name, analysis_type)

class TextSummarizerTool(BaseTool):
    name: str = "text_summarizer"
    description: str = "Summarizes text content from documents"
    args_schema: Type[BaseModel] = TextSummarizerInput

    def _run(self, text: str, max_length: int = 1000) -> Dict[str, Any]:
        """
        Summarize text content
        
        Args:
            text: Text content to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary as a dictionary
        """
        try:
            # Use LLM to summarize text
            prompt = f"""
            Please provide a comprehensive summary of the following text. 
            Focus on key financial insights, metrics, and important points.
            
            TEXT:
            {text[:50000]}  # Limit input to avoid token limits
            
            Your summary should be concise but capture the essential information.
            """
            
            response = fast_llm.invoke(prompt)
            summary = response.content
            
            return {
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary)
            }
                
        except Exception as e:
            return {"error": str(e)}
    
    def _arun(self, text: str, max_length: int = 1000):
        # Async version would go here
        return self._run(text, max_length)

class TableQuerierTool(BaseTool):
    name: str = "table_querier"
    description: str = "Executes natural language queries against tabular data"
    args_schema: Type[BaseModel] = TableQuerierInput

    def _run(self, file_path: str, query: str, file_type: str = "csv") -> Dict[str, Any]:
        """
        Query tabular data using natural language
        
        Args:
            file_path: Path to the file with tabular data
            query: Natural language query
            file_type: Type of file (csv, excel)
            
        Returns:
            Query results as a dictionary
        """
        try:
            # Load data based on file type
            if file_type.lower() == "csv":
                df = pd.read_csv(file_path)
            elif file_type.lower() in ["excel", "xlsx", "xls"]:
                df = pd.read_excel(file_path)
            else:
                return {"error": f"Unsupported file type: {file_type}"}
            
            # Run SQL-like query using LLM
            prompt = f"""
            I have a tabular dataset with the following columns:
            {', '.join(df.columns.tolist())}
            
            Here's a sample of the data (first 5 rows):
            {df.head(5).to_string()}
            
            Based on this dataset, answer the following query:
            "{query}"
            
            First reason about how to approach this query, then provide your answer.
            If the query requires calculation or filtering, explain your process.
            The final answer should be clearly marked as ANSWER: at the end.
            """
            
            response = llm.invoke(prompt)
            
            # Extract the answer part
            result = response.content
            
            return {
                "query": query,
                "result": result,
                "columns": df.columns.tolist()
            }
                
        except Exception as e:
            return {"error": str(e)}
    
    def _arun(self, file_path: str, query: str, file_type: str = "csv"):
        # Async version would go here
        return self._run(file_path, query, file_type)

class VisualizationTool(BaseTool):
    name: str = "visualization_tool"
    description: str = "Creates visual representations of financial data"
    args_schema: Type[BaseModel] = VisualizationInput

    def _run(self, file_path: str, viz_type: str, columns: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Create visualizations from financial data
        
        Args:
            file_path: Path to the data file
            viz_type: Type of visualization (bar, line, scatter, pie, heatmap)
            columns: Columns to include in visualization
            
        Returns:
            Visualization as base64 encoded image
        """
        try:
            # Determine file type and load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return {"error": "Unsupported file type for visualization"}
            
            # If no columns specified, use all numeric columns
            if columns is None:
                columns = df.select_dtypes(include=['number']).columns.tolist()
                if len(columns) > 5:  # Limit to first 5 numeric columns
                    columns = columns[:5]
            
            # Filter to only include existing columns
            columns = [col for col in columns if col in df.columns]
            
            if not columns:
                return {"error": "No valid columns for visualization"}
            
            # Create visualization based on type
            plt.figure(figsize=(10, 6))
            
            if viz_type == "bar":
                df[columns].sum().plot(kind='bar')
                plt.title("Bar Chart")
                plt.ylabel("Sum")
                
            elif viz_type == "line":
                # Check for date column
                date_cols = [col for col in df.columns if 'date' in col.lower()]
                if date_cols:
                    date_col = date_cols[0]
                    df[date_col] = pd.to_datetime(df[date_col])
                    df.set_index(date_col, inplace=True)
                
                df[columns].plot(kind='line')
                plt.title("Line Chart")
                
            elif viz_type == "scatter":
                if len(columns) < 2:
                    return {"error": "Scatter plot requires at least 2 columns"}
                plt.scatter(df[columns[0]], df[columns[1]])
                plt.xlabel(columns[0])
                plt.ylabel(columns[1])
                plt.title("Scatter Plot")
                
            elif viz_type == "pie":
                df[columns[0]].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.title(f"Pie Chart: {columns[0]}")
                
            elif viz_type == "heatmap":
                corr = df[columns].corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm')
                plt.title("Correlation Heatmap")
                
            else:
                return {"error": f"Unsupported visualization type: {viz_type}"}
            
            # Convert plot to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image = base64.b64encode(buffer.read()).decode('utf-8')
            
            return {
                "type": viz_type,
                "image": image,
                "columns": columns
            }
                
        except Exception as e:
            return {"error": str(e)}
    
    def _arun(self, file_path: str, viz_type: str, columns: List[str] = None, **kwargs):
        # Async version would go here
        return self._run(file_path, viz_type, columns, **kwargs)

class WebExtractorTool(BaseTool):
    name: str = "web_extractor"
    description: str = "Extracts and processes content from web URLs"
    
    def _run(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a URL
        
        Args:
            url: URL to extract content from
            
        Returns:
            Extracted content as dictionary
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Send request to URL
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content
            title = soup.title.string if soup.title else "No title"
            paragraphs = [p.text for p in soup.find_all('p')]
            tables = []
            
            # Extract tables
            for table in soup.find_all('table'):
                table_data = []
                for row in table.find_all('tr'):
                    row_data = [cell.text.strip() for cell in row.find_all(['td', 'th'])]
                    table_data.append(row_data)
                tables.append(table_data)
            
            # Summarize content
            text_content = "\n".join(paragraphs)
            summarize_prompt = f"""
            Please summarize the following web content, focusing on financial information:
            
            Title: {title}
            
            Content:
            {text_content[:3000]}  # Limit for token efficiency
            
            Provide a brief summary focusing on key financial insights, data points, and trends.
            """
            
            # Get summary
            response = fast_llm.invoke(summarize_prompt)
            summary = response.content
            
            return {
                "url": url,
                "title": title,
                "summary": summary,
                "paragraph_count": len(paragraphs),
                "table_count": len(tables)
            }
                
        except Exception as e:
            return {"error": str(e)}
    
    def _arun(self, url: str):
        # Async version would go here
        return self._run(url)

class LanguageDetectionInput(BaseModel):
    text: str = Field(description="Text to detect language")
    
class LanguageDetectionTool(BaseTool):
    name: str = "language_detector"
    description: str = "Detects the language of input text"
    args_schema: Type[BaseModel] = LanguageDetectionInput
    def _run(self, text: str) -> Dict[str, str]:
        """
        Detect the language of input text
        
        Args:
            text: Text to detect language
            
        Returns:
            Detected language information
        """
        try:
            # Detect language
            lang_code = langdetect.detect(text)
            
            # Map language codes to full names
            language_map = {
                'en': 'English',
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'it': 'Italian',
                'pt': 'Portuguese',
                'ru': 'Russian',
                'zh-cn': 'Chinese',
                'ja': 'Japanese',
                'ko': 'Korean',
                'ar': 'Arabic',
                'hi': 'Hindi'
            }
            
            language_name = language_map.get(lang_code, f"Unknown ({lang_code})")
            
            return {
                "detected_code": lang_code,
                "detected_language": language_name
            }
                
        except Exception as e:
            return {"error": str(e)}
    
    def _arun(self, text: str):
        # Async version would go here
        return self._run(text)

# Create vector store and document processing functions
def process_document(file_path, file_type):
    """Process a document and store in vector DB"""
    
    try:
        # Choose appropriate loader based on file type
        if file_type == 'pdf':
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_type == 'docx':
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
        elif file_type == 'csv':
            # For CSV, we'll create a text representation
            df = pd.read_csv(file_path)
            text = f"CSV File Analysis:\nColumns: {', '.join(df.columns)}\n"
            text += f"Number of rows: {len(df)}\n"
            text += f"Sample data:\n{df.head(5).to_string()}\n"
            text += f"Statistical summary:\n{df.describe().to_string()}"
            
            # Create document manually
            from langchain.docstore.document import Document as LangchainDocument
            documents = [LangchainDocument(page_content=text, metadata={"source": file_path})]
        elif file_type in ['xlsx', 'xls']:
            # Try to use UnstructuredExcelLoader, fall back to manual processing
            try:
                loader = UnstructuredExcelLoader(file_path)
                documents = loader.load()
            except:
                # Manual Excel processing
                excel_file = pd.ExcelFile(file_path)
                sheet_names = excel_file.sheet_names
                
                text = f"Excel File Analysis:\nSheet names: {', '.join(sheet_names)}\n"
                
                # Process first sheet
                df = pd.read_excel(file_path, sheet_name=sheet_names[0])
                text += f"First sheet ({sheet_names[0]}):\n"
                text += f"Columns: {', '.join(df.columns)}\n"
                text += f"Number of rows: {len(df)}\n"
                text += f"Sample data:\n{df.head(5).to_string()}\n"
                
                # Create document manually
                from langchain.docstore.document import Document as LangchainDocument
                documents = [LangchainDocument(page_content=text, metadata={"source": file_path})]
        else:
            return {"error": f"Unsupported file type: {file_type}"}
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"financial_docs_{int(time.time())}"
        )
        
        return {
            "status": "success",
            "vectorstore": vectorstore,
            "document_path": file_path,
            "chunks": len(chunks)
        }
    
    except Exception as e:
        return {"error": str(e)}

def create_financial_crew(document_info=None):
    """Create the crew of agents for financial analysis"""
    
    # Create tool instances
    csv_analyzer = CSVAnalyzerTool()
    excel_processor = ExcelProcessorTool()
    text_summarizer = TextSummarizerTool()
    table_querier = TableQuerierTool()
    visualizer = VisualizationTool()
    web_extractor = WebExtractorTool()
    language_detector = LanguageDetectionTool()
    
    # Create the agents
    document_processor = Agent(
        role="Document Processing Specialist",
        goal="Process and extract relevant information from financial documents",
        backstory="An expert in handling various document formats and extracting structured data",
        verbose=True,
        llm=llm,
        tools=[csv_analyzer, excel_processor]
    )
    
    financial_analyst = Agent(
        role="Financial Analyst",
        goal="Analyze financial data and provide insights",
        backstory="A seasoned financial analyst with expertise in identifying trends and patterns",
        verbose=True,
        llm=llm,
        tools=[text_summarizer, table_querier]
    )
    
    data_visualizer = Agent(
        role="Data Visualization Expert",
        goal="Create informative visualizations from financial data",
        backstory="An expert in transforming complex financial data into clear visual representations",
        verbose=True,
        llm=llm,
        tools=[visualizer]
    )
    
    language_specialist = Agent(
        role="Multilingual Communication Specialist",
        goal="Handle multilingual queries and provide responses in appropriate languages",
        backstory="An expert linguist proficient in multiple languages and financial terminology",
        verbose=True,
        llm=llm,
        tools=[language_detector]
    )
    
    # Create the crew
    crew = Crew(
        agents=[document_processor, financial_analyst, data_visualizer, language_specialist],
        tasks=[],  # Tasks will be added dynamically based on user queries
        verbose=True,
        process=Process.sequential
    )
    
    return crew

csv_analyzer = CSVAnalyzerTool()
excel_processor = ExcelProcessorTool()
text_summarizer = TextSummarizerTool()
table_querier = TableQuerierTool()
visualizer = VisualizationTool()
web_extractor = WebExtractorTool()
language_detector = LanguageDetectionTool()
    
    # Create the agents
document_processor = Agent(
        role="Document Processing Specialist",
        goal="Process and extract relevant information from financial documents",
        backstory="An expert in handling various document formats and extracting structured data",
        verbose=True,
        llm=llm,
        tools=[csv_analyzer, excel_processor]
    )
    
financial_analyst = Agent(
        role="Financial Analyst",
        goal="Analyze financial data and provide insights",
        backstory="A seasoned financial analyst with expertise in identifying trends and patterns",
        verbose=True,
        llm=llm,
        tools=[text_summarizer, table_querier]
    )
    
data_visualizer = Agent(
        role="Data Visualization Expert",
        goal="Create informative visualizations from financial data",
        backstory="An expert in transforming complex financial data into clear visual representations",
        verbose=True,
        llm=llm,
        tools=[visualizer]
    )
    
language_specialist = Agent(
        role="Multilingual Communication Specialist",
        goal="Handle multilingual queries and provide responses in appropriate languages",
        backstory="An expert linguist proficient in multiple languages and financial terminology",
        verbose=True,
        llm=llm,
        tools=[language_detector]
    )
    
    # Create the crew
crew = Crew(
        agents=[document_processor, financial_analyst, data_visualizer, language_specialist],
        tasks=[],  # Tasks will be added dynamically based on user queries
        verbose=True,
        process=Process.sequential
    )
# Streamlit UI
def main():
    st.set_page_config(
        page_title="CIMCopilot Agent Chatbot",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("CIMCopilot Agent Chatbot")
    
    # Initialize session state for chat history and document info
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "document_info" not in st.session_state:
        st.session_state.document_info = None
    
    if "current_language" not in st.session_state:
        st.session_state.current_language = "English"
    
    # Sidebar for document upload and language selection
    with st.sidebar:
        st.header("Settings")
        
        # Language selection
        languages = ["English", "Spanish", "French", "German", "Chinese"]
        st.session_state.current_language = st.selectbox(
            "Select Language", 
            languages,
            index=languages.index(st.session_state.current_language)
        )
        
        st.header("Upload Financial Document")
        uploaded_file = st.file_uploader(
            "Upload a financial document",
            type=["csv", "xlsx", "xls", "pdf", "docx"]
        )
        
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            file_ext = uploaded_file.name.split(".")[-1].lower()
            temp_file_path = os.path.join("temp_uploads", f"{int(time.time())}_{uploaded_file.name}")
            
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.info(f"Processing {uploaded_file.name}...")
            
            # Process the document
            document_info = process_document(temp_file_path, file_ext)
            
            if "error" in document_info:
                st.error(f"Error processing document: {document_info['error']}")
            else:
                st.session_state.document_info = document_info
                st.success(f"Document processed successfully! {document_info['chunks']} chunks extracted.")
                
                # Add system message about the document
                system_msg = {
                    "role": "system", 
                    "content": f"Processed document: {uploaded_file.name} with {document_info['chunks']} chunks of text."
                }
                st.session_state.messages.append(system_msg)
    
    # Chat interface
    st.header("Chat")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display image if available
            if "image" in message:
                st.image(f"data:image/png;base64,{message['image']}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your financial data..."):
        # Add user message to chat history
        user_msg = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_msg)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process user query
        with st.chat_message("assistant"):
            with st.spinner("Processing your question..."):
                # Check if we need to translate the query
                if st.session_state.current_language != "English":
                    # Detect language or use selected language
                    lang_tool = LanguageDetectionTool()
                    lang_result = lang_tool._run(prompt)
                    
                    # Create a language-specific response task
                    language_info = f"The user is communicating in {st.session_state.current_language}. " + \
                                    f"You must provide your final response in {st.session_state.current_language}."
                else:
                    language_info = "The user is communicating in English."
                
                # Create the crew
                crew = create_financial_crew(st.session_state.document_info)
                
                # Determine the task based on the query
                query_analysis_prompt = f"""
                Based on the following user query, determine the type of financial analysis task needed:
                
                USER QUERY: {prompt}
                
                Classify into one of:
                1. DOCUMENT_SUMMARY - General summary of the document
                2. TREND_ANALYSIS - Analysis of trends in the data
                3. STATISTICAL_ANALYSIS - Statistical insights from the data
                4. VISUALIZATION - Creating charts or visual representations
                5. TABLE_QUERY - Specific query against tabular data
                6. GENERAL_QUESTION - General financial question
                
                Return only the classification without explanation.
                """
                
                task_type_response = fast_llm.invoke(query_analysis_prompt)
                task_type = task_type_response.content.strip()
                
                # Create appropriate task
                document_context = ""
                if st.session_state.document_info:
                    document_context = f"The user has uploaded a document: {st.session_state.document_info['document_path']}. "
                
                task_description = f"""
                {language_info}
                
                {document_context}
                
                USER QUERY: {prompt}
                
                Analyze the user query and provide a comprehensive response. If the query is about:
                - Document content: Extract and summarize relevant information
                - Financial analysis: Provide insightful analysis with key metrics
                - Data visualization: Generate appropriate visualizations
                - Specific data questions: Query the data and provide clear answers
                
                Your response should be professional, accurate, and helpful.
                """
                
                # Create task based on query type
                if "VISUALIZATION" in task_type:
                    # Task for visualization
                    task = Task(
                        description=f"""
                        {task_description}
                        
                        Create appropriate visualizations to address the user's query.
                        Determine the best visualization type (bar, line, scatter, pie, heatmap) 
                        based on the data and query.
                        """,
                        agent=data_visualizer
                    )
                    
                elif "TABLE_QUERY" in task_type:
                    # Task for specific data query
                    task = Task(
                        description=f"""
                        {task_description}
                        
                        Extract specific information from the tabular data to answer the user's query.
                        Use the table_querier tool to formulate and execute the query.
                        """,
                        agent=financial_analyst
                    )
                    
                elif "TREND_ANALYSIS" in task_type:
                    # Task for trend analysis
                    task = Task(
                        description=f"""
                        {task_description}
                        
                        Perform trend analysis on the financial data to identify patterns,
                        growth trajectories, and significant changes over time.
                        """,
                        agent=financial_analyst
                    )
                    
                elif "STATISTICAL_ANALYSIS" in task_type:
                    # Task for statistical analysis
                    task = Task(
                        description=f"""
                        {task_description}
                        
                        Perform statistical analysis on the financial data to extract
                        meaningful insights, correlations, and statistical significance.
                        """,
                        agent=financial_analyst
                    )
                    
                elif "DOCUMENT_SUMMARY" in task_type:
                    # Task for document summary
                    task = Task(
                        description=f"""
                        {task_description}
                        
                        Extract and summarize key information from the document.
                        Focus on the most important financial information and insights.
                        """,
                        agent=document_processor
                    )
                    
                else:  # General question
                    # Default task for general questions
                    task = Task(
                        description=f"""
                        {task_description}
                        
                        Provide a comprehensive response to the user's general financial question.
                        Use your financial knowledge and the document context if relevant.
                        """,
                        agent=financial_analyst
                    )
                
                # Add the task to the crew
                crew.tasks = [task]
                
                # Run the crew to get the result
                result = crew.kickoff()
                
                # Check if we need to translate the response
                if st.session_state.current_language != "English" and "en" in lang_result.get("detected_code", ""):
                    # Create translation task
                    translation_task = Task(
                        description=f"""
                        Translate the following response to {st.session_state.current_language}:
                        
                        {result}
                        
                        Ensure the translation maintains all technical financial terminology accurately.
                        """,
                        agent=language_specialist
                    )
                    
                    # Create a new crew for translation
                    translation_crew = Crew(
                        agents=[language_specialist],
                        tasks=[translation_task],
                        verbose=True,
                        process=Process.sequential
                    )
                    
                    # Get translated result
                    result = translation_crew.kickoff()
                
                # Process result for image content
                image_data = None
                content = result
                
                # Check if the result contains image data
                if "image" in result.lower():
                    # Simple regex to find base64 images
                    import re
                    image_match = re.search(r'base64,([^"\'\\]+)', result)
                    if image_match:
                        image_data = image_match.group(1)
                        # Clean up the result text by removing the image data
                        content = re.sub(r'data:image/png;base64,[^"\'\\]+', "[Image]", result)
                
                # Add assistant message to chat history
                assistant_msg = {
                    "role": "assistant", 
                    "content": content
                }
                
                if image_data:
                    assistant_msg["image"] = image_data
                
                st.session_state.messages.append(assistant_msg)
                
                # Display the response
                st.markdown(content)
                
                if image_data:
                    st.image(f"data:image/png;base64,{image_data}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
