from typing import Dict, Any, Type, TypeVar, Optional, List, ClassVar, get_type_hints, Union
from abc import ABC, abstractmethod, ABCMeta
from datetime import datetime
import json
from dataclasses import dataclass, field as dataclass_field, fields as dataclass_fields, is_dataclass, MISSING
from typeguard import check_type
from pymilvus import DataType, FieldSchema, Hit, Collection

from millie.orm.fields import milvus_field
from millie.orm.base_model import BaseMilvusModel, MilvusModelMeta
from millie.db.connection import MilvusConnection

T = TypeVar('T', bound='MilvusModel')

class TypeCheckError(TypeError):
    """Raised when a type check fails."""
    pass

class MilvusModel(BaseMilvusModel, metaclass=MilvusModelMeta):
    """Milvus model with database operations."""
    embedding: Optional[List[float]] = milvus_field(DataType.FLOAT_VECTOR, dim=1536, default=None)
    metadata: Dict[str, Any] = milvus_field(DataType.JSON, default_factory=dict)
    
    # Class variable to mark migration collections
    is_migration_collection: ClassVar[bool] = False
    
    @classmethod
    def _get_collection(cls) -> Collection:
        """Get the Milvus collection for this model."""
        return MilvusConnection.get_collection(cls.collection_name())
    
    @classmethod
    def load(cls) -> None:
        """Load the collection into memory for faster queries.
        
        This is useful when you plan to perform multiple queries on the collection.
        The collection will stay in memory until you call unload() or the session ends.
        """
        collection = cls._get_collection()
        collection.load()
    
    @classmethod
    def unload(cls) -> None:
        """Unload the collection from memory.
        
        This frees up memory by removing the collection from RAM.
        Subsequent queries will be slower until you call load() again.
        """
        collection = cls._get_collection()
        collection.release()
    
    def save(self) -> bool:
        """Save this model instance to Milvus.
        If the model has an ID and exists, it will be updated.
        If it doesn't exist or has no ID, it will be inserted.
        
        Returns:
            True if successful, False otherwise
        """
        collection = self._get_collection()
        
        # Convert model to dictionary
        data = self.to_dict()
        
        # If we have an ID, try to update
        if hasattr(self, 'id') and getattr(self, 'id'):
            # Delete existing record
            expr = f'id == "{self.id}"'
            collection.delete(expr)
        
        # Insert the record
        try:
            collection.insert(data)
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    @classmethod
    def bulk_insert(cls: Type[T], models: List[T], batch_size: int = 100) -> bool:
        """Insert multiple models in batches.
        
        Args:
            models: List of model instances to insert
            batch_size: Number of records to insert at once
            
        Returns:
            True if all inserts were successful, False if any failed
        """
        collection = cls._get_collection()
        
        # Convert all models to dictionaries
        data = [model.to_dict() for model in models]
        
        # Insert in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            try:
                collection.insert(batch)
            except Exception as e:
                print(f"Error inserting batch: {str(e)}")
                return False
        
        return True
    
    @classmethod
    def bulk_upsert(cls: Type[T], models: List[T], batch_size: int = 100) -> bool:
        """Update or insert multiple models in batches.
        
        Args:
            models: List of model instances to upsert
            batch_size: Number of records to process at once
            
        Returns:
            True if all operations were successful, False if any failed
        """
        collection = cls._get_collection()
        
        # Group models by whether they have IDs
        updates = []
        inserts = []
        for model in models:
            if hasattr(model, 'id') and getattr(model, 'id'):
                updates.append(model)
            else:
                inserts.append(model)
        
        # Process updates in batches
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            # Get IDs for this batch
            ids = [model.id for model in batch]
            # Delete existing records
            if hasattr(self, 'id') and getattr(self, 'id'):
                # Delete existing record
                expr = 'id in ["' + '","'.join(ids) + '"]'
                collection.delete(expr)
            # Insert updated records
            collection.insert([model.to_dict() for model in batch])
        
        # Process inserts in batches
        if inserts:
            try:
                return cls.bulk_insert(inserts, batch_size)
            except Exception as e:
                print(f"Error inserting new records: {str(e)}")
                return False
        
        return True
    
    def delete(self) -> bool:
        """Delete this model instance from Milvus.
        
        Returns:
            True if successful, False otherwise
        """
        if not hasattr(self, 'id') or not getattr(self, 'id'):
            return False
            
        collection = self._get_collection()
        
        try:
            expr = f'id == "{self.id}"'
            collection.delete(expr)
            return True
        except Exception as e:
            print(f"Error deleting model: {str(e)}")
            return False
    
    @classmethod
    def delete_many(cls, expr: str) -> bool:
        """Delete multiple models matching an expression.
        
        Args:
            expr: Expression to match records to delete
            
        Returns:
            True if successful, False otherwise
        """
        collection = cls._get_collection()
        
        try:
            collection.delete(expr)
            return True
        except Exception as e:
            print(f"Error deleting models: {str(e)}")
            return False
    
    @classmethod
    def get_all(
        cls: Type[T],
        offset: int = 0,
        limit: Optional[int] = None,
        output_fields: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        order_desc: bool = False
    ) -> List[T]:
        """Get all models with optional pagination and ordering.
        
        Args:
            offset: Number of records to skip
            limit: Maximum number of records to return. If None, returns up to 16384 records.
            output_fields: Optional list of fields to return. If None, returns all fields.
            order_by: Optional field to order by
            order_desc: If True, order in descending order
            
        Returns:
            List of model instances
        """
        collection = cls._get_collection()
        
        # Build query parameters
        query_params = {
            "expr": "",  # Empty string for no filtering
            "output_fields": output_fields or ["*"],
            "offset": offset,
            "limit": min(limit or 16384, 16384)  # Always include a limit, max 16384
        }
            
        # Add ordering if specified
        if order_by:
            order = "DESC" if order_desc else "ASC"
            query_params["order_by"] = f"{order_by} {order}"
        
        # Execute query
        results = collection.query(**query_params)
        
        return [cls.from_dict(cls._convert_hit_to_dict(result)) for result in results]
    
    @classmethod
    def get_by_id(cls: Type[T], id: str, output_fields: Optional[List[str]] = None) -> Optional[T]:
        """Get a single model by its ID.
        
        Args:
            id: The ID to look up
            output_fields: Optional list of fields to return. If None, returns all fields.
            
        Returns:
            Model instance if found, None otherwise
        """
        collection = cls._get_collection()
        
        # Query by ID
        expr = f'id == "{id}"'
        results = collection.query(
            expr=expr,
            output_fields=output_fields or ['*']
        )
        
        if not results:
            return None
            
        return cls.from_dict(cls._convert_hit_to_dict(results[0]))
    
    @classmethod
    def filter(cls: Type[T], output_fields: Optional[List[str]] = None, **kwargs) -> List[T]:
        """Get models matching the given filters.
        
        Args:
            output_fields: Optional list of fields to return. If None, returns all fields.
            **kwargs: Field names and values to filter by
            
        Returns:
            List of matching models
        """
        collection = cls._get_collection()
        
        # Build filter expression
        conditions = []
        for field, value in kwargs.items():
            if isinstance(value, str):
                conditions.append(f'{field} == "{value}"')
            else:
                conditions.append(f'{field} == {value}')
        expr = ' && '.join(conditions)
        
        results = collection.query(
            expr=expr,
            output_fields=output_fields or ['*']
        )
        
        return [cls.from_dict(cls._convert_hit_to_dict(result)) for result in results]
    
    @classmethod
    def search_by_similarity(
        cls: Type[T],
        query_embedding: List[float],
        limit: int = 5,
        expr: Optional[str] = None,
        metric_type: str = "L2",
        search_params: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[T]:
        """Search for models by vector similarity.
        
        Args:
            query_embedding: The embedding vector to search with
            limit: Maximum number of results to return
            expr: Optional filter expression
            metric_type: Distance metric to use (L2 or IP)
            search_params: Optional dictionary of search parameters (e.g. {"nlist": 1024, "nprobe": 10})
                         If not provided, defaults to {"nprobe": 10}
            output_fields: Optional list of fields to return. If None, returns all fields.
            
        Returns:
            List of models sorted by similarity
        """
        collection = cls._get_collection()
        
        # Build search parameters
        params = {
            "metric_type": metric_type,
            "params": search_params or {"nprobe": 10}
        }
        
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=params,
            limit=limit,
            expr=expr,
            output_fields=output_fields or ["*"]
        )
        
        return [cls.from_dict(cls._convert_hit_to_dict(hit)) for hit in results[0]]
    
    @classmethod
    @abstractmethod
    def collection_name(cls) -> str:
        """Get the collection name for this model."""
        ...
    
    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """Get the Milvus schema for this model by inspecting decorated fields."""
        fields = []
        
        # Get all dataclass fields including from parent classes
        for field in dataclass_fields(cls):
            if str(field.type).startswith('typing.ClassVar'):
                continue
                
            # Check if field has Milvus metadata
            if field.metadata and 'milvus' in field.metadata:
                milvus_info = field.metadata['milvus']
                fields.append(
                    FieldSchema(
                        field.name,
                        milvus_info.data_type,
                        **milvus_info.kwargs
                    )
                )
        
        return {"fields": fields}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for Milvus insertion."""
        result = {}
        for field in dataclass_fields(self):
            if not str(field.type).startswith('typing.ClassVar'):
                value = getattr(self, field.name)
                if value is not None and value != {}:
                    # Serialize complex types
                    if field.name in ['metadata', 'extra_data']:
                        value = json.dumps(self._serialize_complex_type(value))
                    else:
                        value = self._serialize_complex_type(value)
                    result[field.name] = value
        return result
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create model instance from dictionary."""
        # Parse JSON strings back to dicts
        if isinstance(data.get('metadata'), str):
            data['metadata'] = json.loads(data['metadata'])
        
        # Handle any other JSON fields
        for key, value in data.items():
            if isinstance(value, str):
                # Try to parse JSON fields
                if value.startswith('{') and value.endswith('}'):
                    try:
                        data[key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass  # Not valid JSON, leave as string
                
                # Try to parse datetime fields
                if key in ['created_at', 'updated_at'] or key.endswith('_at'):
                    try:
                        data[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        pass  # Not a valid datetime string, leave as string
        
        # Filter out class variables
        instance_data = {
            k: v for k, v in data.items()
            if k in cls.__annotations__ and not str(cls.__annotations__[k]).startswith('typing.ClassVar')
        }
        
        return cls(**instance_data)
    
    def _serialize_complex_type(self, value: Any) -> Any:
        """Serialize complex data types."""
        if isinstance(value, list):
            return [self._serialize_complex_type(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_complex_type(v) for k, v in value.items()}
        elif isinstance(value, datetime):
            return value.isoformat()
        elif hasattr(value, 'to_dict'):
            return value.to_dict()
        return value
    
    def serialize_for_json(self) -> str:
        """Serialize model to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def deserialize_from_json(cls: Type[T], json_str: str) -> T:
        """Create model instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @staticmethod
    def _convert_hit_to_dict(data: Union[Dict[str, Any], Hit]) -> Dict[str, Any]:
        """Convert Hit object or dictionary to properly typed dictionary.
        
        Args:
            data: Raw data from Milvus, either a dict or Hit object
            
        Returns:
            Dictionary with properly typed values
        """
        # If data is a Hit object, get its fields
        if isinstance(data, Hit):
            data = dict(data.fields)
        
        if 'embedding' in data:
            # Convert embedding values to float
            data['embedding'] = [float(x) for x in data['embedding']]
        return data 