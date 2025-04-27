import os
import logging
import pandas as pd
from sqlalchemy import create_engine, inspect, MetaData, Table, Column, String, Integer, DateTime, Text, text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import Dict, List, Any, Optional
from langchain_community.utilities import SQLDatabase

# Load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PostgreSQL connection parameters
PG_USERNAME = os.getenv("POSTGRES_USERNAME", "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
PG_DB = os.getenv("POSTGRES_DB", "finbot")

# SQLAlchemy Base
Base = declarative_base()

class PostgresManager:
    """Manager class for PostgreSQL database operations."""
    
    def __init__(self, uri=None):
        """Initialize database connection."""
        self.engine = None
        self.connected = False
        
        if uri:
            self.uri = uri
        else:
            self.uri = f"postgresql+psycopg2://{PG_USERNAME}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
        
        # Connect to PostgreSQL
        self._connect()
    
    def _connect(self):
        """Establish connection to PostgreSQL."""
        try:
            self.engine = create_engine(self.uri)
            # Test connection - use SQLAlchemy text() for proper SQL execution
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("Connected to PostgreSQL successfully")
            self.connected = True
            
            # Initialize metadata
            self.metadata = MetaData()
            self.metadata.reflect(bind=self.engine)
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            self.connected = False
    
    def get_engine(self):
        """Return the SQLAlchemy engine."""
        return self.engine
    
    def get_langchain_db(self):
        """Return a LangChain SQLDatabase instance for the connected database."""
        if not self.connected:
            raise ConnectionError("Not connected to PostgreSQL database")
        
        return SQLDatabase.from_uri(self.uri)
    
    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace') -> bool:
        """
        Save a pandas DataFrame to a PostgreSQL table.
        
        Args:
            df: The pandas DataFrame to save
            table_name: The name of the table
            if_exists: How to behave if the table exists ('fail', 'replace', or 'append')
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to database, can't save DataFrame")
            return False
        
        try:
            # Save DataFrame to PostgreSQL
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
            logger.info(f"Successfully saved DataFrame to table: {table_name}")
            
            # Update metadata to reflect the new table
            self.metadata.reflect(bind=self.engine, only=[table_name])
            
            return True
        except Exception as e:
            logger.error(f"Error saving DataFrame to PostgreSQL: {str(e)}")
            return False
    
    def get_table_info(self, table_name: str = None) -> List[Dict]:
        """
        Get information about tables in the database.
        
        Args:
            table_name: Optional specific table name to query
            
        Returns:
            List of dictionaries containing table information
        """
        if not self.connected:
            logger.warning("Not connected to database, can't get table info")
            return []
        
        try:
            inspector = inspect(self.engine)
            tables_info = []
            
            tables = [table_name] if table_name else inspector.get_table_names()
            
            for table in tables:
                columns = inspector.get_columns(table)
                tables_info.append({
                    "table_name": table,
                    "columns": [{"name": col["name"], "type": str(col["type"])} for col in columns]
                })
            
            return tables_info
        except Exception as e:
            logger.error(f"Error getting table info from PostgreSQL: {str(e)}")
            return []
    
    def execute_query(self, query: str) -> Optional[List[Dict]]:
        """
        Execute a SQL query and return the results.
        
        Args:
            query: SQL query to execute
            
        Returns:
            List of dictionaries containing query results, or None if error
        """
        if not self.connected:
            logger.warning("Not connected to database, can't execute query")
            return None
        
        try:
            with self.engine.connect() as conn:
                # Use text() function to convert the string to a SQL expression
                result = conn.execute(text(query))
                if result.returns_rows:
                    # Convert to list of dictionaries
                    columns = result.keys()
                    return [dict(zip(columns, row)) for row in result]
                return []
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return None
    
    def list_tables(self) -> List[str]:
        """
        List all tables in the database.
        
        Returns:
            List of table names
        """
        if not self.connected:
            logger.warning("Not connected to database, can't list tables")
            return []
        
        try:
            inspector = inspect(self.engine)
            return inspector.get_table_names()
        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            return []
    
    def close(self):
        """Close the database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("PostgreSQL connection closed")
            self.connected = False

# Create a singleton instance
postgres_manager = PostgresManager() 