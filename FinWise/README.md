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

1. Clone the repository
2. Install the required dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with the following variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `GEMINI_API_KEY`: Your Google Gemini API key
   - Additional LLM API keys as needed

### API Key Management

#### Multiple API Keys
You can configure multiple API keys for the same LLM provider to enable key rotation and load balancing. To add multiple keys:

1. In your `.env` file, add multiple keys using comma separation:
   ```
   OPENAI_API_KEY=key1,key2,key3
   GEMINI_API_KEY=key1,key2
   ```

2. The system will automatically rotate between these keys to:
   - Distribute API calls across multiple keys
   - Avoid rate limiting issues
   - Provide failover if one key encounters errors

#### API Key Rotation
The application implements an intelligent key rotation strategy:
- Keys are used in sequence, rotating to the next key after each API call
- If a key encounters an error (like rate limiting), it's temporarily skipped
- The system tracks usage metrics for each key to optimize distribution

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

## API Key Rotation

The application supports API key rotation for Google Gemini to help manage rate limits and ensure uninterrupted service:

1. **Multiple API Keys**:
   - The system can use multiple Gemini API keys
   - Keys are automatically rotated when rate limits are reached

2. **Adding API Keys**:
   - Open `src/llm/gemini_integration.py`
   - Add your additional API keys to the `API_KEYS` list
   - Example: `API_KEYS = [GEMINI_API_KEY, "YOUR_SECOND_API_KEY_HERE", "YOUR_THIRD_API_KEY_HERE"]`

3. **Benefits**:
   - Increased resilience against rate limiting
   - Higher throughput for LLM requests
   - Seamless fallback when one key reaches its quota

## Multi-Provider LLM Support

The chatbot supports multiple LLM providers with automatic fallback functionality:

1. **Available Providers**:
   - **Google Gemini**: Primary provider for most queries
   - **Groq**: Used as a fallback when Gemini encounters errors or rate limits

2. **Setting Up Groq**:
   - Create a Groq account at [groq.com](https://console.groq.com/keys)
   - Get your API key from the Groq console
   - Add it to your `.env` file: `GROQ_API_KEY=your_groq_api_key_here`

3. **Provider Management**:
   - The system automatically switches between providers when errors occur
   - Error detection identifies rate limits and quota issues
   - Cooldown periods are managed for each provider

4. **Benefits**:
   - Increased reliability through provider redundancy
   - Continued operation during provider outages
   - Ability to leverage the unique strengths of each model

## License

This project is licensed under the MIT License. 