"""Milvus session management."""
import os
from typing import Type, List, Dict, Optional, TypeVar, Union
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility
import logging
from dotenv import load_dotenv

from millie.db.connection import MilvusConnection
from millie.orm.milvus_model import MilvusModel

load_dotenv()

T = TypeVar('T', bound=MilvusModel)
logger = logging.getLogger(__name__)

class MilvusSession:
    """Manages Milvus database operations."""
    
    def __init__(self, host: str = os.getenv('MILVUS_HOST', 'localhost'), port: int = int(os.getenv('MILVUS_PORT', 19530)), db_name: str = os.getenv('MILVUS_DB_NAME', 'default')):
        self.connection = MilvusConnection(host, port, db_name)
    
    def get_milvus_collection(self, model_class: Type[T]) -> Collection:
        """Get or create a collection for a model class."""
        collection_name = model_class.collection_name()
        
        # Check if collection exists
        if not utility.has_collection(collection_name):
            # Create collection with schema
            schema = model_class.schema()
            collection = Collection(
                name=collection_name,
                schema=CollectionSchema(schema.fields)
            )
            collection.create_index(
                field_name="embedding",
                index_params={
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
            )
            logger.info(f"Created collection {collection_name}")
        
        # Get collection from cache
        return MilvusConnection.get_collection(collection_name)
    
    def collection_exists(self, model_class: Type[T]) -> bool:
        """Check if a collection exists."""
        return utility.has_collection(model_class.collection_name())
    
    def collection(self, model_class: Type[T]) -> Collection:
        return self.get_milvus_collection(model_class)
    
    def drop_all_collections(self):
        """Drop all collections inside Milvus"""
        collections = utility.list_collections()
        if not collections:
            logger.info("No collections found to drop")
            return
        
        for collection in collections:
            utility.drop_collection(collection)
            logger.info(f"Dropped collection: {collection}")
    
    def close(self):
        """Close the session."""
        self.connection.close()

    def init_collection(self, model_class: Type[T]):
        """Initialize all collections for registered models."""
        logger.info(f"Initializing collection {model_class.collection_name()}...")
        
        try:
            self.get_milvus_collection(model_class)
            logger.info(f"Initialized collection for {model_class.collection_name()}")
        except Exception as e:
            logger.error(f"Failed to initialize collection for {model_class.collection_name()}: {str(e)}")
            raise
    
    def load_collection(self, model_class: Type[T]):
        """Load a collection."""
        logger.info(f"Loading collection {model_class.collection_name()}...")
        try:
            milvus_collection = self.get_milvus_collection(model_class)
            milvus_collection.load()
            logger.info(f"Loaded collection for {model_class.collection_name()}")
        except Exception as e:
            logger.error(f"Failed to load collection for {model_class.collection_name()}: {str(e)}")
            raise

    def unload_collection(self, model_class: Type[T]):
        """Unload a collection."""
        logger.info(f"Unloading collection {model_class.collection_name()}...")
        try:
            milvus_collection = Collection(model_class.collection_name())
            milvus_collection.release()
            logger.info(f"Unloaded collection for {model_class.collection_name()}")
        except Exception as e:
            logger.error(f"Failed to unload collection for {model_class.collection_name()}: {str(e)}")
            raise 
