"""Base class for Milvus models."""
from typing import Dict, Any, Type, TypeVar, Optional, List, ClassVar, get_type_hints, Union, get_origin, get_args
from abc import ABC, abstractmethod
from datetime import datetime
import json
import logging
from dataclasses import dataclass, field as dataclass_field, fields, MISSING
import uuid
from pymilvus import DataType, FieldSchema, Hit, Collection
from typeguard import check_type, TypeCheckError

from millie.db.schema import Schema
from millie.orm.fields import milvus_field
from millie.db.connection import MilvusConnection

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='MilvusModel')

# Registry of all model classes
MODEL_REGISTRY: Dict[str, Type['MilvusModel']] = {}

def register_model(cls: Type[T]) -> Type[T]:
    """Register a model class in the registry."""
    MODEL_REGISTRY[cls.__name__] = cls
    return cls

def eval_type(type_hint: Any) -> Type:
    """Evaluate a type hint to get its concrete type.
    
    Args:
        type_hint: The type hint to evaluate
        
    Returns:
        The concrete type
    """
    # Handle Optional types
    origin = get_origin(type_hint)
    if origin is Union:
        args = get_args(type_hint)
        # Optional[T] is Union[T, None]
        if len(args) == 2 and args[1] is type(None):
            return eval_type(args[0])
    # Handle List, Dict, etc.
    elif origin is not None:
        return origin
    # Handle simple types
    return type_hint

@dataclass(kw_only=True)
class MilvusModel(ABC):
    """Base class for all Milvus models.
    
    This is a dataclass that provides:
    1. Schema generation for Milvus
    2. Serialization/deserialization
    3. Model registration for tracking
    4. Milvus collection operations (CRUD)
    
    """
    
    # Class variable to mark migration collections
    is_migration_collection: ClassVar[bool] = False
    
    def __init_subclass__(cls, **kwargs):
        """Make sure subclasses are also dataclasses and registered."""
        super().__init_subclass__(**kwargs)
        # Register the model class
        register_model(cls)
        # Make sure subclasses are also dataclasses
        dataclass(cls, kw_only=True)
        
        # Add type hints for the constructor
        cls.__init__.__annotations__ = get_type_hints(cls)
    
    @classmethod
    def __class_getitem__(cls, key):
        """Support for type hints like MilvusModel[T]."""
        return cls
    
    def __post_init__(self):
        """Validate field types after initialization."""
        for field_name, field in self.__class__.__dataclass_fields__.items():
            if str(field.type).startswith('typing.ClassVar'):
                continue
                
            value = getattr(self, field_name)
            
            # Skip None values for Optional fields or fields with default_factory
            if value is None:
                if field.default_factory is not MISSING or 'Optional' in str(field.type):
                    continue
                if not field.metadata.get('required', True):
                    continue
            
            try:
                # Special handling for List[float] type
                if eval_type(field.type) == list and field_name == 'embedding' and value is not None:
                    if not isinstance(value, list) or not all(isinstance(x, (int, float)) for x in value):
                        raise TypeCheckError(f"{type(value).__name__} did not match any element in the union")
                # Basic type checking
                elif not isinstance(value, eval_type(field.type)):
                    raise TypeCheckError(f"{type(value).__name__} is not an instance of {field.type}")
            except Exception as e:
                raise TypeCheckError(str(e))
    
    @classmethod
    @abstractmethod
    def collection_name(cls) -> str:
        """Get the collection name for this model."""
        ...
    
    @classmethod
    def schema(cls) -> Schema:
        """Generate the schema for this model."""
        fields = []
        for field_name, field in cls.__dataclass_fields__.items():
            if not str(field.type).startswith('typing.ClassVar'):
                milvus_info = field.metadata.get('milvus')
                if milvus_info:
                    # Get field parameters from milvus_info
                    kwargs = {}
                    if hasattr(milvus_info, 'max_length'):
                        kwargs['max_length'] = milvus_info.max_length
                    if hasattr(milvus_info, 'dim'):
                        kwargs['dim'] = milvus_info.dim
                    if hasattr(milvus_info, 'is_primary'):
                        kwargs['is_primary'] = milvus_info.is_primary
                    
                    fields.append(
                        FieldSchema(
                            name=field_name,
                            dtype=milvus_info.dtype,
                            **kwargs
                        )
                    )
        return Schema(
            name=cls.__name__,
            collection_name=cls.collection_name(),
            fields=fields,
            is_migration_collection=cls.is_migration_collection
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for Milvus insertion."""
        result = {}
        for field_name, field in self.__class__.__dataclass_fields__.items():
            if str(field.type).startswith('typing.ClassVar'):
                continue
                
            value = getattr(self, field_name)
            
            # Handle JSON fields
            if field.metadata and 'milvus' in field.metadata and field.metadata['milvus'].dtype == DataType.JSON:
                # Ensure we have a dict to serialize
                if value is None:
                    value = {}
                
                # Serialize complex types first
                value = self._serialize_complex_type(value)
                # Then convert to JSON string
                value = json.dumps(value)
            else:
                value = self._serialize_complex_type(value)
                
            # Always include the field, even if None
            result[field_name] = value
        
        return result
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create model instance from dictionary."""
        # Parse JSON fields and convert types
        parsed_data = {}
        for key, value in data.items():
            # Get field type from annotations
            field_type = get_type_hints(cls).get(key)
            
            # Handle JSON fields
            if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                try:
                    parsed_data[key] = json.loads(value)
                except json.JSONDecodeError:
                    parsed_data[key] = value
            # Handle datetime fields
            elif field_type == datetime and isinstance(value, str):
                try:
                    parsed_data[key] = datetime.fromisoformat(value)
                except ValueError:
                    parsed_data[key] = value
            else:
                parsed_data[key] = value
        
        return cls(**parsed_data)
    
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
    
    @classmethod
    def get_all_models(cls) -> List[Type['MilvusModel']]:
        """Get all registered model classes."""
        return list(MODEL_REGISTRY.values())
    
    @classmethod
    def get_model(cls, name: str) -> Optional[Type['MilvusModel']]:
        """Get a registered model by name."""
        return MODEL_REGISTRY.get(name)
    
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
        """Convert Hit object or dictionary to properly typed dictionary."""
        # If data is a Hit object, get its fields
        if isinstance(data, Hit):
            data = dict(data.fields)
        
        if 'embedding' in data:
            # Convert embedding values to float
            data['embedding'] = [float(x) for x in data['embedding']]
        return data
    
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
            expr = f'id in ["{id}" for id in {ids}]'
            try:
                collection.delete(expr)
            except Exception as e:
                print(f"Error deleting existing records: {str(e)}")
                return False
            
            # Insert updated records
            try:
                data = [model.to_dict() for model in batch]
                collection.insert(data)
            except Exception as e:
                print(f"Error inserting updated records: {str(e)}")
                return False
        
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