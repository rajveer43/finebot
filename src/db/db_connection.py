import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import pymongo, but provide a fallback if not available
try:
    import pymongo
    from bson import ObjectId
    MONGODB_AVAILABLE = True
except ImportError:
    logger.warning("pymongo or bson not available - using mock database manager")
    MONGODB_AVAILABLE = False

# Import the MongoDB URI from config
try:
    from src.config.config import MONGODB_URI
except ImportError:
    logger.warning("config.py not found or missing MONGODB_URI - using default connection string")
    MONGODB_URI = "mongodb+srv://rajveer43:pSajp7kXhbKd1KYz@dev-cluster.78ajkfz.mongodb.net/?retryWrites=true&w=majority&appName=dev-cluster"

class MockDatabaseManager:
    """Mock database manager for when MongoDB is not available."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the mock database."""
        self.connected = False
        self.chat_messages = []
        self.visualizations = []
        self.document_references = []
        logger.info("Using mock database manager - data will not be persisted")
    
    def save_chat_message(self, session_id: str, message: Dict[str, Any]) -> str:
        """Save a chat message to the in-memory list."""
        try:
            # Make sure we have a timestamp
            if "timestamp" not in message:
                message["timestamp"] = datetime.now()
            
            # Add the session ID
            message["session_id"] = session_id
            
            # Add a mock ID
            message["_id"] = f"mock_{len(self.chat_messages)}"
            
            # Append to the list
            self.chat_messages.append(message)
            logger.info(f"Saved chat message with mock ID: {message['_id']}")
            return message["_id"]
        except Exception as e:
            logger.error(f"Error saving chat message to mock DB: {str(e)}")
            return None
    
    def save_visualization(self, session_id: str, visualization_data: Dict[str, Any]) -> str:
        """Save a visualization to the in-memory list."""
        try:
            # Make sure we have a timestamp
            if "timestamp" not in visualization_data:
                visualization_data["timestamp"] = datetime.now()
            
            # Add the session ID
            visualization_data["session_id"] = session_id
            
            # Add a mock ID
            visualization_data["_id"] = f"mock_{len(self.visualizations)}"
            
            # Append to the list
            self.visualizations.append(visualization_data)
            logger.info(f"Saved visualization with mock ID: {visualization_data['_id']}")
            return visualization_data["_id"]
        except Exception as e:
            logger.error(f"Error saving visualization to mock DB: {str(e)}")
            return None
    
    def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a session from the in-memory list."""
        try:
            # Filter messages by session ID
            messages = [m for m in self.chat_messages if m.get("session_id") == session_id]
            
            # Sort by timestamp
            messages.sort(key=lambda x: x.get("timestamp", datetime.min))
            
            return messages
        except Exception as e:
            logger.error(f"Error getting chat history from mock DB: {str(e)}")
            return []
    
    def get_visualizations(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get visualizations from the in-memory list."""
        try:
            # Filter by session ID if provided
            if session_id:
                visualizations = [v for v in self.visualizations if v.get("session_id") == session_id]
            else:
                visualizations = self.visualizations.copy()
            
            # Sort by timestamp (most recent first)
            visualizations.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
            
            return visualizations
        except Exception as e:
            logger.error(f"Error getting visualizations from mock DB: {str(e)}")
            return []
    
    def save_document_reference(self, session_id: str, document_data: Dict[str, Any]) -> str:
        """Save a document reference to the in-memory list."""
        try:
            # Make sure we have a timestamp
            if "timestamp" not in document_data:
                document_data["timestamp"] = datetime.now()
            
            # Add the session ID
            document_data["session_id"] = session_id
            
            # Add a mock ID
            document_data["_id"] = f"mock_{len(self.document_references)}"
            
            # Append to the list
            self.document_references.append(document_data)
            logger.info(f"Saved document reference with mock ID: {document_data['_id']}")
            return document_data["_id"]
        except Exception as e:
            logger.error(f"Error saving document reference to mock DB: {str(e)}")
            return None
    
    def get_document_references(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get document references from the in-memory list."""
        try:
            # Filter by session ID if provided
            if session_id:
                doc_refs = [d for d in self.document_references if d.get("session_id") == session_id]
            else:
                doc_refs = self.document_references.copy()
            
            # Sort by timestamp (most recent first)
            doc_refs.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
            
            return doc_refs
        except Exception as e:
            logger.error(f"Error getting document references from mock DB: {str(e)}")
            return []
    
    def close(self):
        """No-op for the mock database."""
        logger.info("Mock database 'closed'")


class DatabaseManager:
    """Manager class for database operations."""
    
    def __init__(self, uri=None):
        """Initialize database connection."""
        if not MONGODB_AVAILABLE:
            raise ImportError("pymongo or bson not available - cannot create DatabaseManager")
            
        self.client = None
        self.db = None
        self.connected = False
        self.uri = uri or MONGODB_URI
        
        # Connect to MongoDB
        self._connect()
    
    def _connect(self):
        """Establish connection to MongoDB."""
        try:
            self.client = pymongo.MongoClient(self.uri)
            # Ping the server to confirm connection
            self.client.admin.command('ping')
            logger.info("Connected to MongoDB successfully")
            
            # Set the database
            self.db = self.client["finbot_db"]
            self.connected = True
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            self.connected = False
    
    def save_chat_message(self, session_id: str, message: Dict[str, Any]) -> str:
        """
        Save a chat message to the database.
        
        Args:
            session_id: The session identifier
            message: Dictionary containing message data
            
        Returns:
            ID of the saved message
        """
        if not self.connected:
            logger.warning("Not connected to database, can't save chat message")
            return None
        
        try:
            # Make sure we have a timestamp
            if "timestamp" not in message:
                message["timestamp"] = datetime.now()
            
            # Add the session ID
            message["session_id"] = session_id
            
            # Insert the message
            result = self.db.chat_messages.insert_one(message)
            logger.info(f"Saved chat message with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving chat message: {str(e)}")
            return None
    
    def save_visualization(self, session_id: str, visualization_data: Dict[str, Any]) -> str:
        """
        Save a visualization to the database.
        
        Args:
            session_id: The session identifier
            visualization_data: Dictionary containing visualization data
            
        Returns:
            ID of the saved visualization
        """
        if not self.connected:
            logger.warning("Not connected to database, can't save visualization")
            return None
        
        try:
            # Make sure we have a timestamp
            if "timestamp" not in visualization_data:
                visualization_data["timestamp"] = datetime.now()
            
            # Add the session ID
            visualization_data["session_id"] = session_id
            
            # Insert the visualization
            result = self.db.visualizations.insert_one(visualization_data)
            logger.info(f"Saved visualization with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
            return None
    
    def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get chat history for a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            List of chat messages
        """
        if not self.connected:
            logger.warning("Not connected to database, can't get chat history")
            return []
        
        try:
            # Find messages for the session and sort by timestamp
            messages = list(self.db.chat_messages.find(
                {"session_id": session_id}
            ).sort("timestamp", 1))
            
            # Convert ObjectId to string for JSON serialization
            for message in messages:
                if "_id" in message:
                    message["_id"] = str(message["_id"])
            
            return messages
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return []
    
    def get_visualizations(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get visualizations, optionally filtered by session.
        
        Args:
            session_id: The session identifier (optional)
            
        Returns:
            List of visualizations
        """
        if not self.connected:
            logger.warning("Not connected to database, can't get visualizations")
            return []
        
        try:
            # Create query filter
            query = {}
            if session_id:
                query["session_id"] = session_id
            
            # Find visualizations and sort by timestamp (most recent first)
            visualizations = list(self.db.visualizations.find(
                query
            ).sort("timestamp", -1))
            
            # Convert ObjectId to string for JSON serialization
            for viz in visualizations:
                if "_id" in viz:
                    viz["_id"] = str(viz["_id"])
            
            return visualizations
        except Exception as e:
            logger.error(f"Error getting visualizations: {str(e)}")
            return []
    
    def save_document_reference(self, session_id: str, document_data: Dict[str, Any]) -> bool:
        """
        Save a document reference to the database.
        
        Args:
            session_id: Session ID
            document_data: Document data including name, path, type, etc.
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not document_data:
                return False
                
            # Add session ID and timestamp
            document_data['session_id'] = session_id
            if 'timestamp' not in document_data:
                document_data['timestamp'] = datetime.now()
                
            # Save to database
            self.db['documents'].insert_one(document_data)
            return True
        except Exception as e:
            logger.error(f"Error saving document reference: {str(e)}")
            return False
    
    def save_document_history(self, session_id: str, document_name: str, 
                            action: str, details: Dict[str, Any]) -> bool:
        """
        Save document processing history.
        
        Args:
            session_id: Session ID
            document_name: Name of the document
            action: Action performed (e.g., 'upload', 'analyze', 'visualize')
            details: Details of the action
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not session_id or not document_name or not action:
                return False
                
            history_entry = {
                'session_id': session_id,
                'document_name': document_name,
                'action': action,
                'details': details,
                'timestamp': datetime.now()
            }
                
            # Save to database
            self.db['document_history'].insert_one(history_entry)
            return True
        except Exception as e:
            logger.error(f"Error saving document history: {str(e)}")
            return False
    
    def get_document_history(self, session_id: str, document_name: Optional[str] = None) -> List:
        """
        Get document processing history.
        
        Args:
            session_id: Session ID
            document_name: Optional document name to filter by
            
        Returns:
            List of document history entries
        """
        try:
            # Prepare query
            query = {'session_id': session_id}
            if document_name:
                query['document_name'] = document_name
                
            # Get history from database
            cursor = self.db['document_history'].find(query).sort('timestamp', -1)
            
            # Convert to list and return
            return list(cursor)
        except Exception as e:
            logger.error(f"Error getting document history: {str(e)}")
            return []
    
    def get_document_references(self, session_id: str) -> List:
        """
        Get document references for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of document references
        """
        try:
            # Prepare query
            query = {'session_id': session_id}
                
            # Get documents from database
            cursor = self.db['documents'].find(query).sort('timestamp', -1)
            
            # Convert to list and return
            return list(cursor)
        except Exception as e:
            logger.error(f"Error getting document references: {str(e)}")
            return []
            
    def save_search_history(self, session_id: str, search_data: Dict[str, Any]) -> bool:
        """
        Save search history to the database.
        
        Args:
            session_id: Session ID
            search_data: Search data including query, results, etc.
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not search_data:
                return False
                
            # Add session ID and timestamp
            search_data['session_id'] = session_id
            if 'timestamp' not in search_data:
                search_data['timestamp'] = datetime.now()
                
            # Save to database
            self.db['search_history'].insert_one(search_data)
            return True
        except Exception as e:
            logger.error(f"Error saving search history: {str(e)}")
            return False
    
    def get_search_history(self, session_id: str) -> List:
        """
        Get search history for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of search history entries
        """
        try:
            # Prepare query
            query = {'session_id': session_id}
                
            # Get search history from database
            cursor = self.db['search_history'].find(query).sort('timestamp', -1)
            
            # Convert to list and return
            return list(cursor)
        except Exception as e:
            logger.error(f"Error getting search history: {str(e)}")
            return []
    
    def close(self):
        """Close the database connection."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")

# Create a singleton instance - use MockDatabaseManager as a fallback
try:
    if MONGODB_AVAILABLE:
        db_manager = DatabaseManager()
    else:
        db_manager = MockDatabaseManager()
except Exception as e:
    logger.error(f"Error creating database manager: {str(e)}")
    logger.info("Falling back to mock database")
    db_manager = MockDatabaseManager() 