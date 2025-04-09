# Financial Intelligence Chatbot

A robust, scalable financial chatbot capable of processing both structured and unstructured financial documents. The system extracts key insights, summarizes content, and answers user queries based on uploaded data.

## Agentic AI Architecture

This implementation uses an agentic AI architecture powered by Google's Gemini models. The system consists of:

1. **Core Agent**: A central coordinator that manages:
   - Intent detection
   - Tool selection
   - Response generation
   - Language detection and translation

2. **Tool Registry System**: Modular tools that can be dynamically invoked:
   - Tools register capabilities and schemas
   - The agent selects appropriate tools based on user intent
   - Loose coupling allows for easy extension

3. **LLM Integration**: Google Gemini integration with:
   - Intent detection
   - Structured output generation
   - Response generation

## Features

- Process multiple file formats (CSV, Excel, PDF, DOCX)
- Extract financial data and insights
- Analyze trends and metrics
- Support for multiple languages
- Real-time chat interface
- Document history management

## Project Structure

```
financial-intelligence-chatbot/
├── src/
│   ├── agents/
│   │   └── financial_agent.py  # Core agent coordinator
│   ├── tools/
│   │   ├── file_processor.py   # File processing tool
│   │   ├── financial_analysis_tool.py  # Financial analysis tool
│   │   ├── text_summarization.py  # Text summarization tool
│   │   ├── language_tool.py   # Language detection/translation tool
│   │   └── tool_registry.py   # Tool registry system
│   ├── llm/
│   │   └── gemini_integration.py  # Google Gemini LLM integration
│   ├── config/
│   │   └── config.py  # Configuration settings
│   ├── models/        # Data models
│   ├── utils/         # Utility functions
│   └── routes/        # API routes
├── uploads/           # Directory for uploaded files
├── main.py            # Streamlit application
├── requirements.txt   # Dependencies
└── .env               # Environment variables
```

## Technology Stack

- **Language**: Python 3.8+
- **LLM**: Google Gemini Pro
- **Frontend**: Streamlit
- **Data Processing**:
  - Pandas for tabular data
  - PyPDF for PDF parsing
  - python-docx for DOCX processing
- **Libraries**:
  - langdetect for language detection
  - streamlit for web UI
  - langchain for LLM integrations

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd financial-intelligence-chatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with the following variables:
```
PORT=5000
MONGODB_URI=mongodb://localhost:27017/financial_chatbot
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

5. Get a Gemini API key:
   - Visit [Google AI Studio](https://makersuite.google.com/)
   - Create an account and generate an API key
   - Add the key to your `.env` file

## Running the Application

Start the Streamlit app:
```bash
streamlit run main.py
```

The application will be available at http://localhost:8501

## How It Works

1. **User Query Processing**:
   - The system analyzes the user's query to determine intent
   - It selects the appropriate tools based on the detected intent
   - The agent coordinates tool execution

2. **Document Processing**:
   - Files are uploaded through the Streamlit interface
   - The file processor tool extracts content based on file type
   - Different processing strategies are applied for different formats

3. **Analysis & Insights**:
   - The financial analysis tool extracts trends, metrics, and insights
   - Results are formatted for easy understanding

4. **Response Generation**:
   - The agent generates a natural language response using Gemini
   - Responses incorporate insights from the tool executions

5. **Multilingual Support**:
   - Language is detected automatically
   - Users can select their preferred language
   - Responses are translated to the selected language

## Extending the System

To add new tools:

1. Create a new tool class that inherits from the `Tool` base class
2. Implement the `execute` method with your tool's functionality
3. Define input and output schemas
4. Register the tool with the `tool_registry`

## License

This project is licensed under the MIT License. 