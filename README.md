# Financial Intelligence Chatbot

A robust, scalable financial chatbot capable of processing both structured and unstructured financial documents. The system extracts key insights, summarizes content, and answers user queries based on uploaded data.

![Financial Intelligence Chatbot](docs/screenshots/dashboard.png)

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
- Web search and online content analysis
- Advanced data visualization capabilities
- Export functionality for session data
- Robust database with local fallback storage
- Comprehensive history tracking
- Third-party charting library integration

## Project Structure
```
FinWise/
├── keys/              # API keys and credentials
├── src/
│   ├── agents/
│   │   └── financial_agent.py  # Core agent coordinator
│   ├── tools/
│   │   ├── file_processor.py   # File processing tool
│   │   ├── financial_analysis_tool.py  # Financial analysis tool
│   │   ├── text_summarization.py  # Text summarization tool
│   │   ├── language_tool.py   # Language detection/translation tool
│   │   ├── web_search_tool.py # Web content extraction tool
│   │   ├── search_api_tool.py # Search API integration
│   │   ├── data_visualization_tool.py # Visualization tool
│   │   ├── csv_analyzer_tool.py # CSV analysis tool
│   │   ├── dynamic_visualization_tool.py # Advanced visualization
│   │   └── tool_registry.py   # Tool registry system
│   ├── llm/
│   │   ├── llm_manager.py     # LLM provider manager
│   │   ├── gemini_integration.py  # Google Gemini integration
│   │   └── groq_integration.py  # Groq LLM integration
│   ├── db/
│   │   └── db_connection.py   # Database connection manager
│   ├── config/
│   │   └── config.py  # Configuration settings
│   ├── models/        # Data models
│   ├── utils/         # Utility functions
│   └── routes/        # API routes
├── docs/
│   ├── screenshots/           # Application screenshots
│   └── diagrams/             # System architecture diagrams
├── uploads/           # Directory for uploaded files
├── local_storage/     # Local storage for database fallback
├── main.py            # Streamlit application
├── system_architecture.md # Architecture diagrams 
├── visualization/ # Visualization capabilities documentation
├── evaluation_report.md # System performance and evaluation report
├── debug_tools.py     # Debugging and tool validation utilities
├── requirements.txt   # Dependencies
└── .env               # Environment variables```
```

## Technology Stack

- **Language**: Python 3.8+
- **LLM**: Google Gemini Pro, Groq
- **Frontend**: Streamlit
- **Database**: MongoDB with local storage fallback
- **Data Processing**:
  - Pandas for tabular data
  - PyPDF for PDF parsing
  - python-docx for DOCX processing
- **Data Visualization**:
  - Matplotlib
  - Plotly
  - Altair
  - Seaborn
- **Web Search**:
  - SerpAPI integration
  - BeautifulSoup for web scraping
- **Libraries**:
  - langdetect for language detection
  - streamlit for web UI
  - langchain for LLM integrations

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/financial-intelligence-chatbot.git
cd financial-intelligence-chatbot
```

2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required dependencies: 
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with the following variables:
```
# LLM API Keys
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# SerpAPI for web search
SERPAPI_KEY=your_serpapi_key_here

# MongoDB connection (optional)
MONGODB_CONNECTION_STRING=your_mongodb_connection_string
```

### ✅ Steps to Get JSON Key from Google Cloud Console

1. **Go to Google Cloud Console**  
   Open: [https://console.cloud.google.com/](https://console.cloud.google.com/)

2. **Select your project**  
   At the top, click the project dropdown and choose the project you're working with (`finbot` in your case).

3. **Enable the required API** (if not already enabled):  
   - For Gen Language models or Vertex AI:  
     Search for and enable **Vertex AI API** or **Generative Language API**.

4. **Go to IAM & Admin > Service Accounts**  
   - Navigation Menu (☰) → **IAM & Admin** → **Service Accounts**

5. **Create a New Service Account** *(or select an existing one)*  
   - Click **“Create Service Account”**
   - Give it a name like `gen-lang-client`
   - Click **“Create and Continue”**

6. **Assign Role(s)**  
   - Choose appropriate roles. For Gen AI use, select:  
     `Vertex AI User` or `Generative Language API User`  
   - Click **“Continue”** and then **“Done”**

7. **Generate the Key File (JSON)**  
   - After creation, click the service account name
   - Go to the **“Keys” tab**
   - Click **“Add Key” → “Create new key”**
   - Select **“JSON”** → Click **“Create”**
   - ✅ The `.json` key file will automatically download.

8. **Move or Save the Key File to Your Path**  
   - Save it to:  
     `C:\your\path\keys\`  
     (create the folders if they don’t exist)

Great question! Once you’ve downloaded the JSON key file, you need to **set an environment variable** so your code can authenticate using that key.

---

### ✅ Terminal Command to Use the Key (Linux/macOS/WSL)

```bash
export GOOGLE_APPLICATION_CREDENTIALS="C:/path/to/project/finbot/keys/gen-lang-client-0049803850-db4864d6249d.json"
```

### ✅ Windows Command Prompt (cmd)

```cmd
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\project\finbot\keys\gen-lang-client-0049803850-db4864d6249d.json
```

---

### ✅ Windows PowerShell

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\project\finbot\keys\gen-lang-client-0049803850-db4864d6249d.json"
```


### ⚠️ Important Security Note:
Never share this key publicly. It gives access to your Google Cloud resources. Treat it like a password.

### Database Setup

The application supports two database options:

#### Option 1: MongoDB (Recommended for Production)
1. Set up a MongoDB database (local or cloud-based like MongoDB Atlas)
2. Add your connection string to the `.env` file:
   ```
   MONGODB_CONNECTION_STRING=mongodb+srv://username:password@cluster.mongodb.net/financial_chatbot
   ```

#### Option 2: Local File Storage (Default)
- If no MongoDB connection is provided, the system automatically falls back to local file storage
- Local storage files are saved in the `local_storage/` directory
- This option works out of the box with no additional setup

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

### Command-line Options

You can customize the application startup with these options:

```bash
streamlit run main.py -- --port 8502 --no-db-connection --mock-llm
```

Available options:
- `--port`: Specify a custom port (default: 8501)
- `--no-db-connection`: Run without attempting database connection
- `--mock-llm`: Use mock LLM responses for testing without API costs

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

6. **Web Research**:
   - Extract content from URLs mentioned in queries
   - Perform web searches for relevant financial information
   - Summarize online content and provide source citations

7. **Data Visualization**:
   - Analyze data to determine appropriate visualization types
   - Generate charts based on financial data patterns
   - Support multiple visualization libraries for different chart types
   - Save visualizations for future reference

8. **Export & History Management**:
   - Track session history including chat, documents, and searches
   - Export data in multiple formats (JSON, CSV)
   - Generate specialized reports for different aspects of analysis
   - Access document processing history for audit purposes

## UI Features and Navigation

The application interface is designed for intuitive navigation:

### Main Areas
- **Chat Interface**: Central area for conversation with the AI
- **Sidebar**: Contains settings, document management, and history
- **Fixed Chat Input**: Always accessible at the bottom of the screen

### Sidebar Sections
1. **Chat Settings**: Language selection and session management
2. **Documents**: Upload and manage financial documents
3. **Visualizations**: View saved charts and graphs
4. **History**: Browse conversation history
5. **Document History**: Track document processing actions
6. **Search History**: View past web searches
7. **Export Data**: Export session data in various formats
8. **About**: Information about the application

### Export Features

The application provides comprehensive export capabilities:

1. **Export Formats**:
   - JSON: Complete data export with all details
   - CSV: Spreadsheet-friendly format for easy analysis
   - Full Archive: Complete session backup

2. **Report Types**:
   - Chat History: Export conversations with timestamps
   - Document Summary: Summary of all processed documents
   - Visualization History: Export of charts and analysis
   - Activity Log: Complete audit trail of system usage

3. **Export Process**:
   1. Navigate to the "Export Data" tab in the sidebar
   2. Select your preferred export format
   3. Choose whether to include visualizations
   4. Click "Export Current Session" or select a specific report
   5. Download the generated file

## Advanced Visualization

The system supports multiple charting libraries for different visualization needs:

1. **Built-in Libraries**:
   - **Matplotlib**: Standard static charts
   - **Plotly**: Interactive visualizations
   - **Altair**: Declarative charts
   - **Seaborn**: Statistical visualizations

2. **Chart Types**:
   - Line charts for trend analysis
   - Bar charts for comparisons
   - Scatter plots for correlation analysis
   - Pie charts for composition analysis
   - Heatmaps for complex data patterns
   - Box plots for distribution analysis

3. **Dynamic Chart Selection**:
   - The system automatically determines the most appropriate chart type based on data characteristics
   - Users can specify preferred chart types in their queries
   - Charts adapt to data size and dimensionality

## Extending the System

To add new tools:

1. Create a new tool class that inherits from the `Tool` base class
2. Implement the `execute` method with your tool's functionality
3. Define input and output schemas
4. Register the tool with the `tool_registry`

Example of a new tool implementation:

```python
from src.tools.tool_registry import Tool, tool_registry

class NewFinancialTool(Tool):
    """Tool for specialized financial analysis."""
    
    name = "NewFinancialTool"
    description = "Performs specialized financial analysis"
    
    input_schema = {
        "type": "object",
        "properties": {
            "data": {
                "type": "string",
                "description": "Financial data to analyze"
            }
        },
        "required": ["data"]
    }
    
    output_schema = {
        "type": "object",
        "properties": {
            "result": {
                "type": "string",
                "description": "Analysis result"
            }
        }
    }
    
    def execute(self, data: str) -> dict:
        """Execute the tool with the provided data."""
        # Your implementation here
        result = self._analyze_data(data)
        return {"result": result}
    
    def _analyze_data(self, data: str) -> str:
        # Custom analysis logic
        return "Analysis results"

# Register the tool
tool_registry.register(NewFinancialTool)
```

## System Architecture

The system architecture is documented in detail in the `system_architecture.md` file, which contains Mermaid diagrams illustrating:

1. **System Architecture Diagram**: High-level components and their relationships
2. **Block Diagram**: Logical organization of the system's layers
3. **Workflow Diagram**: Sequence of operations for typical user interactions
4. **Component Hierarchy**: Organization and nesting of system components
5. **Data Flow Diagram**: How data moves through the system
6. **Entity-Relationship Diagram**: Data relationships
7. **State Diagram**: System states during different operations
8. **Deployment Diagram**: Physical deployment structure

To view these diagrams, open the file in a Markdown viewer that supports Mermaid, such as:
- GitHub's web interface
- VS Code with the Markdown Preview Mermaid Support extension
- Various online Mermaid viewers

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

## Troubleshooting

### Database Connection Issues
- Check your MongoDB connection string in the `.env` file
- Ensure your IP address is whitelisted in MongoDB Atlas (if using Atlas)
- The system will automatically fall back to local storage if the database is unavailable

### LLM API Key Issues
- Verify your API keys in the `.env` file
- Check quota limits on your LLM provider accounts
- The system will attempt to use alternative providers if available

### File Processing Errors
- Ensure uploaded files are in supported formats
- Check file encoding (UTF-8 is recommended)
- Large files may need to be split into smaller pieces

### Web Search Issues
- Verify your SerpAPI key
- Check your internet connection
- Some websites may block content extraction





## Contributing

We welcome contributions to improve the Financial Intelligence Chatbot:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

## License

This project is licensed under the MIT License. 


