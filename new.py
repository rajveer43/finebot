# Single-file Financial Summarizer App with CrewAI, PDF/DOCX/CSV Search Tools, FastAPI & Streamlit

import base64
from io import BytesIO
import os
import time
from typing import Type
import logging  # Added logger
# import fitz  # PyMuPDF
# import docx
from matplotlib import pyplot as plt
import pandas as pd
import streamlit as st
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
from crewai import Agent, Task, Crew, LLM
from crewai_tools import PDFSearchTool, DOCXSearchTool, CSVSearchTool
from langdetect import detect
import uvicorn
import requests
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = LLM(
    model="gemini/gemini-2.0-flash-exp",  # Updated model to Gemini 2.0
    temperature=0.8,
    max_tokens=4000,
)

os.environ["OPENAI_API_KEY"] = ""
# os.environ["GEMINI_API_KEY"] = "AIzaSyDYmei4WbkEX9p_6rF_RaANkl72DgxEIBQ"
# ==== Language Detection Tool ====
class LanguageDetectionTool:
    def _run(self, text: str) -> str:
        try:
            return detect(text)
        except Exception as e:
            logger.error(f"Language detection error: {str(e)}")  # Log error
            return f"Error: {str(e)}"
        

class CSVPlotInput(BaseModel):
    file_path: str = Field(..., description="Path to the CSV file")
    x_column: str = Field(..., description="Column name for X-axis")
    y_column: str = Field(..., description="Column name for Y-axis")
    chart_type: str = Field(..., description="Type of chart: line, bar, histogram")

class CSVPlotTool(BaseTool):
    name: str = "CSV Plot Tool"
    description: str = "Plots a graph (line, bar, or histogram) from a CSV file."
    args_schema: Type[BaseModel] = CSVPlotInput

    def _run(self, file_path: str, x_column: str, y_column: str, chart_type: str) -> str:
        try:
            df = pd.read_csv(file_path)
            if x_column not in df.columns or y_column not in df.columns:
                logger.error(f"Columns '{x_column}' or '{y_column}' not found in CSV.")  # Log error
                return f"Error: Columns '{x_column}' or '{y_column}' not found in CSV."

            plt.figure(figsize=(10, 6))

            if chart_type.lower() == "line":
                plt.plot(df[x_column], df[y_column], marker='o')
            elif chart_type.lower() == "bar":
                plt.bar(df[x_column], df[y_column])
            elif chart_type.lower() == "histogram":
                plt.hist(df[y_column], bins=10)
                plt.xlabel(y_column)  # In histogram, X-axis is value bins
                plt.ylabel("Frequency")
            else:
                logger.error(f"Unsupported chart type '{chart_type}'.")  # Log error
                return f"Error: Unsupported chart type '{chart_type}'. Use 'line', 'bar', or 'histogram'."

            if chart_type.lower() != "histogram":
                plt.xlabel(x_column)
                plt.ylabel(y_column)

            plt.title(f"{chart_type.capitalize()} Chart: {y_column} vs {x_column}")
            plt.grid(True)

            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            plt.close()

            return f"data:image/png;base64,{image_base64}"

        except Exception as e:
            logger.error(f"Error generating graph: {str(e)}")  # Log error
            return f"Error generating graph: {str(e)}"

# ==== FastAPI Setup ====
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Helper: Create Crew for PDF/DOCX/CSV ====
def create_financial_crew(document_info):
    doc_path = document_info["document_path"]
    file_type = document_info["file_type"]

    # Load appropriate tool
    if file_type == "pdf":
        search_tool = PDFSearchTool(pdf=doc_path)
    elif file_type == "docx":
        search_tool = DOCXSearchTool(docx=doc_path)
    elif file_type == "csv":
        search_tool = CSVSearchTool(csv=doc_path)  # Initialize with the CSV file path
        visualizer_tool = CSVPlotTool(file=doc_path)
    else:
        logger.error("Unsupported file type")  # Log error
        raise ValueError("Unsupported file type")

    agent = Agent(
        role="Financial Analyst",
        goal="Understand and summarize financial documents",
        backstory="Expert in financial document analysis.",
        llm=llm,  # Updated LLM to Gemini 2.0
        tools=[search_tool, visualizer_tool],
        verbose=True
    )

    return Crew(agents=[agent], tasks=[], verbose=True)

# ==== FastAPI: Dummy Document Processor ====
@app.post("/process")
async def process_doc(file: UploadFile = File(...)):
    os.makedirs("temp_uploads", exist_ok=True)
    temp_path = os.path.join("temp_uploads", f"{int(time.time())}_{file.filename}")

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ["pdf", "docx", "csv"]:
        logger.error("Unsupported format")  # Log error
        return {"error": "Unsupported format"}

    return {"document_path": temp_path, "file_type": file_ext}

# ==== Streamlit UI ====
def main():
    st.set_page_config(page_title="CIMCopilot Agent Chatbot", page_icon="💰", layout="wide")
    st.title("CIMCopilot Agent Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_info" not in st.session_state:
        st.session_state.document_info = None
    if "current_language" not in st.session_state:
        st.session_state.current_language = "English"

    with st.sidebar:
        st.header("Settings")
        languages = ["English", "Spanish", "French", "German", "Chinese"]
        st.session_state.current_language = st.selectbox("Select Language", languages, index=0)

        st.header("Upload Document")
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "csv"])  # Added CSV support

        if uploaded_file:
            temp_file_path = os.path.join("temp_uploads", f"{int(time.time())}_{uploaded_file.name}")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Processing document..."):
                res = requests.post("http://localhost:8000/process", files={"file": open(temp_file_path, "rb")})
                result = res.json()

                if "error" in result:
                    st.error(result["error"])
                    logger.error(result["error"])  # Log error
                else:
                    st.session_state.document_info = result
                    st.success("Document processed successfully!")
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"Processed document: {uploaded_file.name}  with  text."
                    })

    st.header("Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing your question..."):
                lang_info = "English"
                if st.session_state.current_language != "English":
                    lang_tool = LanguageDetectionTool()
                    lang_info = lang_tool._run(prompt)

                document_context = ""
                if st.session_state.document_info:
                    document_context = f"Document path: {st.session_state.document_info['document_path']}"

                crew = create_financial_crew(st.session_state.document_info)
                task = Task(
                    description=f"The user asked: '{prompt}' in language {lang_info}. Use the document at {document_context} to answer.",
                    expected_output="A comprehensive answer to the user's question based on the financial document analysis return response in the language of the question",
                    agent=crew.agents[0],
                )
                crew.tasks = [task]
                
                # Check for context length before invoking the model
                if len(st.session_state.messages) > 10:  # Example threshold
                    st.session_state.messages = st.session_state.messages[-10:]  # Keep only the last 10 messages

                response = crew.kickoff()

                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)

# ==== Launch both FastAPI and Streamlit ====
if __name__ == "__main__":
    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    Thread(target=run_api).start()
    time.sleep(1)
    main()
