import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/financial_chatbot")

# Server configuration
PORT = int(os.getenv("PORT", "5000"))

# File Upload Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")
ALLOWED_EXTENSIONS = {
    'csv': 'text/csv',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'xls': 'application/vnd.ms-excel',
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'doc': 'application/msword',
    'txt': 'text/plain'
}

# Maximum file size (10 MB)
MAX_CONTENT_LENGTH = 10 * 1024 * 1024

# Language settings
DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "zh", "ja"] 