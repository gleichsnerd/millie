"""Milvus connection management."""
import os
from typing import Dict
from pymilvus import Collection, connections
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class MilvusConnection:
    """Manages Milvus database connection."""
    
    _instance = None
    _collections: Dict[str, Collection] = {}
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, host: str | None = None, port: int | None = None, db_name: str | None = None):
        if not hasattr(self, 'initialized'):
            # Load environment variables
            load_dotenv()
            
            # Set connection parameters with priority: explicit args > env vars > defaults
            self.host = host or os.getenv('MILVUS_HOST', 'localhost')
            self.port = port or int(os.getenv('MILVUS_PORT', '19530'))
            self.db_name = db_name or os.getenv('MILVUS_DB_NAME', 'default')
            
            self.initialized = True
            self._connect()
    
    def _connect(self):
        """Establish connection to Milvus."""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=str(self.port),  # Convert port to string for pymilvus
                db_name=self.db_name
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus at {self.host}:{self.port}: {str(e)}")
            raise
    
    def close(self):
        """Close all collections and connection."""
        for collection in self._collections.values():
            collection.release()
        connections.disconnect("default")
        logger.info("Disconnected from Milvus")
    
    @classmethod
    def get_collection(cls, name: str) -> Collection:
        """Get a cached collection instance."""
        if name not in cls._collections:
            cls._collections[name] = Collection(name)
        return cls._collections[name]
    
    @classmethod
    def remove_collection(cls, name: str):
        """Remove a collection from cache."""
        if name in cls._collections:
            cls._collections.pop(name) 