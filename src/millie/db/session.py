"""Milvus session management."""
import os
from typing import Type, List, Dict, Optional, TypeVar, Union
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility
import logging
from dotenv import load_dotenv

from millie.db.connection import MilvusConnection
from millie.orm.base_model import BaseMilvusModel

load_dotenv()

T = TypeVar('T', bound=BaseMilvusModel)
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
                schema=CollectionSchema(schema['fields'])
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

    def insert(self, model: T) -> str:
        """Insert a model instance."""
        collection = self.get_milvus_collection(type(model))
        data = model.to_dict()
        
        try:
            collection.insert([data])
            collection.flush()
            return model.id
        except Exception as e:
            logger.error(f"Failed to insert {type(model).__name__}: {str(e)}")
            raise
    
    def search(self, 
              model_class: Type[T], 
              vector: List[float], 
              limit: int = 10, 
              expr: Optional[str] = None) -> List[T]:
        """Search for similar vectors."""
        collection = self.get_milvus_collection(model_class)
        collection.load()
        
        try:
            results = collection.search(
                data=[vector],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=limit,
                expr=expr,
                output_fields=list(model_class.__annotations__.keys())
            )
            
            return [
                model_class.from_dict(hit.entity)
                for hit in results[0]
            ]
        finally:
            collection.release()
    
    def query(self, 
             model_class: Type[T], 
             expr: Optional[str] = "",
             output_fields: Optional[List[str]] = None,
             limit: Optional[int] = None) -> List[T]:
        """Query models by expression."""
        collection = self.get_milvus_collection(model_class)
        collection.load()
        
        try:
            query_params = {
                "expr": expr,
                "output_fields": output_fields or list(model_class.__annotations__.keys())
            }
            
            # Add limit if expression is empty or limit is specified
            if not expr or limit is not None:
                query_params["limit"] = min(limit or 16384, 16384)  # Use max allowed limit as default
            
            results = collection.query(**query_params)
            
            return [model_class.from_dict(entity) for entity in results]
        finally:
            collection.release()
    
    def update(self, model: T, expr: str):
        """Update model fields."""
        collection = self.get_milvus_collection(type(model))
        data = model.to_dict()
        
        try:
            collection.upsert(
                data=[data]
            )
            collection.flush()
        except Exception as e:
            logger.error(f"Failed to update {type(model).__name__}: {str(e)}")
            raise
    
    def delete(self, model_class: Type[T], expr: str):
        """Delete models by expression."""
        collection = self.get_milvus_collection(model_class)
        
        try:
            collection.delete(expr)
            collection.flush()
        except Exception as e:
            logger.error(f"Failed to delete {model_class.__name__}: {str(e)}")
            raise
    
    def load_relationships(self, model: T):
        """Load related models for a model instance."""
        for field, related_model_name in model.relationships.items():
            # Import the related model class dynamically to avoid circular imports
            module = __import__('db.models', fromlist=[related_model_name])
            related_model = getattr(module, related_model_name)
            
            # Handle different relationship types
            if isinstance(getattr(model, field), list):
                # One-to-many relationship
                results = self.query(
                    related_model,
                    f'parent_id == "{model.id}"'
                )
                setattr(model, field, results)
            else:
                # One-to-one relationship
                results = self.query(
                    related_model,
                    f'id == "{getattr(model, field + "_id")}"'
                )
                if results:
                    setattr(model, field, results[0])
    
    def save_relationships(self, model: T):
        """Save related models for a model instance."""
        for field, _ in model.relationships.items():
            related_data = getattr(model, field)
            if related_data is None:
                continue
            
            if isinstance(related_data, list):
                # One-to-many relationship
                for related_model in related_data:
                    setattr(related_model, 'parent_id', model.id)
                    self.insert(related_model)
            else:
                # One-to-one relationship
                self.insert(related_data) 
