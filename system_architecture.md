# FinWise System Architecture Diagrams

## System Architecture Diagram

```mermaid
graph TD
    subgraph Frontend["Streamlit Frontend"]
        UI[UI Components]
        SessionState[Session State]
        Sidebar[Sidebar Components]
        Chat[Chat Interface]
        Viz[Visualization Display]
    end

    subgraph Core["Core System"]
        FinAgent[Financial Agent]
        LLMManager[LLM Manager]
        DBManager[Database Manager]
    end

    subgraph Tools["Tool Registry"]
        FileProc[File Processor]
        WebSearch[Web Search Tool]
        SearchAPI[Search API Tool]
        FinAnalysis[Financial Analysis Tool]
        DataViz[Data Visualization Tool]
        DynamicViz[Dynamic Visualization Tool]
        TextSum[Text Summarization Tool]
        Lang[Language Tool]
        CSVAnalyzer[CSV Analyzer Tool]
    end

    subgraph External["External Services"]
        MongoDB[(MongoDB Database)]
        Gemini[Gemini AI]
        Groq[Groq API]
        SerpAPI[SerpAPI]
    end

    UI --> FinAgent
    SessionState --> FinAgent
    Chat --> FinAgent
    FinAgent --> Viz
    FinAgent --> Tools
    FinAgent --> LLMManager
    LLMManager --> Gemini
    LLMManager --> Groq
    DBManager --> MongoDB
    DBManager --> LocalFiles[(Local File Storage)]
    FinAgent --> DBManager
    SearchAPI --> SerpAPI
    SearchAPI --> WebSearch
    
    class Frontend,Core,Tools,External cluster
```

## Block Diagram

```mermaid
flowchart TD
    subgraph "UI Layer"
        S[Streamlit Application]
        S --> |User Inputs| SB[Sidebar]
        S --> |Displays| CH[Chat History]
        S --> |Shows| VD[Visualization Display]
        S --> |Fixed Bottom| CI[Chat Input]
    end

    subgraph "Business Logic Layer"
        FA[Financial Agent] --> |Handles| QP[Query Processing]
        FA --> |Manages| DH[Document Handling]
        FA --> |Controls| TR[Tool Registry]
        LM[LLM Manager] --> |Manages| MP[Model Providers]
        LM --> |Handles| FR[Fallbacks]
    end

    subgraph "Data Layer"
        DB[Database Manager] --> |Stores| CM[Chat Messages]
        DB --> |Manages| VZ[Visualizations]
        DB --> |Catalogs| DR[Document References]
        DB --> |Tracks| SH[Search History]
        DB --> |Uses| MDB[(MongoDB)]
        DB --> |Fallback| MFS[Mock File Storage]
    end

    subgraph "Tool Layer"
        TR --> FP[File Processor]
        TR --> WS[Web Search]
        TR --> SA[Search API]
        TR --> FA2[Financial Analysis]
        TR --> DVT[Data Visualization]
        TR --> TS[Text Summarization]
        TR --> LT[Language Translation]
        TR --> CA[CSV Analysis]
    end

    S --> FA
    FA --> LM
    FA --> DB
    
    class "UI Layer","Business Logic Layer","Data Layer","Tool Layer" cluster
```

## Workflow Diagram

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant FA as Financial Agent
    participant TR as Tool Registry
    participant LLM as LLM Manager
    participant DB as Database Manager
    
    User->>UI: Upload Document
    UI->>FA: Process Document
    FA->>TR: Use File Processor
    TR-->>FA: Document Content
    FA->>DB: Save Document Reference
    DB-->>FA: Confirmation
    FA-->>UI: Document Processed
    
    User->>UI: Send Chat Message
    UI->>FA: Process Query
    FA->>LLM: Analyze Intent
    LLM-->>FA: Intent Analysis
    
    alt Data Analysis Intent
        FA->>TR: Use Financial Analysis Tool
        TR-->>FA: Analysis Results
    else Web Search Intent
        FA->>TR: Use Search API Tool
        TR-->>FA: Search Results
    else Visualization Intent
        FA->>TR: Use Visualization Tool
        TR-->>FA: Visualization Data
    end
    
    FA->>LLM: Generate Response
    LLM-->>FA: Text Response
    FA->>DB: Save Chat Message
    DB-->>FA: Confirmation
    FA-->>UI: Display Response
    
    alt Visualization Generated
        FA->>DB: Save Visualization
        DB-->>FA: Confirmation
        FA-->>UI: Display Visualization
    end
    
    User->>UI: Export History
    UI->>DB: Get Session Data
    DB-->>UI: Session History
    UI-->>User: Download File
```

## Component Hierarchy

```mermaid
graph TD
    subgraph Application
        Main[Main Application]
        Main --> Sidebar
        Main --> MainContent
    end
    
    subgraph Sidebar
        Settings[Chat Settings]
        Documents[Document Management]
        Visualizations[Saved Visualizations]
        History[Chat History]
        DocHistory[Document History]
        SearchHistory[Search History]
        Export[Export Features]
        About[About Section]
    end
    
    subgraph MainContent
        ChatInterface[Chat Interface]
        VisualizationDisplay[Visualization Display]
        AnalysisResults[Analysis Results]
        Sources[Source References]
    end
    
    subgraph Agents
        FinancialAgent[Financial Agent]
    end
    
    subgraph Managers
        LLMManager[LLM Manager]
        DatabaseManager[Database Manager]
        MockDatabaseManager[Mock DB Manager]
    end
    
    subgraph Tools
        FileProcessor[File Processor]
        WebSearchTool[Web Search Tool]
        SearchAPITool[Search API Tool]
        FinancialAnalysisTool[Financial Analysis]
        TextSummarizationTool[Text Summarization]
        LanguageTool[Language Tool]
        DataVisualizationTool[Data Visualization]
        CSVAnalyzerTool[CSV Analyzer]
        DynamicVisualizationTool[Dynamic Visualization]
    end
    
    MainContent --> FinancialAgent
    Sidebar --> DatabaseManager
    FinancialAgent --> Tools
    FinancialAgent --> LLMManager
    DatabaseManager --> MockDatabaseManager
    
    class Application,Sidebar,MainContent,Agents,Managers,Tools cluster
``` 

## Data Flow Diagram

```mermaid
graph LR
    User((User)) --> |Uploads Files| IN[Input Processor]
    User --> |Queries| IN
    IN --> |Documents| DB[(Database)]
    IN --> |Requests| Agent[Financial Agent]
    DB --> |Historical Data| Agent
    Agent --> |Tool Request| Tools[Tool Registry]
    Tools --> |Analysis Results| Agent
    Agent --> |Query| LLM[LLM Service]
    LLM --> |Response| Agent
    Agent --> |Results| OUT[Output Formatter]
    OUT --> |Visualizations| User
    OUT --> |Text Response| User
    OUT --> |Export Data| User
``` 

## Feature Breakdown Structure

```mermaid
graph TD
    FinWise[FinWise Financial Chatbot] --> DataAnalysis[Data Analysis]
    FinWise --> WebResearch[Web Research]
    FinWise --> InterfaceFeatures[Interface Features]
    FinWise --> Export[Export & History]
    
    DataAnalysis --> FA1[CSV/Excel Analysis]
    DataAnalysis --> FA2[PDF Analysis]
    DataAnalysis --> FA3[Trend Detection]
    DataAnalysis --> FA4[Visualization]
    
    WebResearch --> WR1[Search API Integration]
    WebResearch --> WR2[URL Content Extraction]
    WebResearch --> WR3[Summarization]
    WebResearch --> WR4[Source Citation]
    
    InterfaceFeatures --> IF1[Chat Interface]
    InterfaceFeatures --> IF2[Document Management]
    InterfaceFeatures --> IF3[Visualization Display]
    InterfaceFeatures --> IF4[Language Translation]
    
    Export --> EX1[Chat History Export]
    Export --> EX2[Document History Tracking]
    Export --> EX3[Activity Reports]
    Export --> EX4[Full Session Archive]
    
    class FinWise,DataAnalysis,WebResearch,InterfaceFeatures,Export feature
``` 