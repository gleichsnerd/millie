from typing import Dict, Any, Type, TypeVar, Optional, List, ClassVar, get_type_hints
from abc import ABC, abstractmethod, ABCMeta
from datetime import datetime
import json
from dataclasses import dataclass, field as dataclass_field, fields as dataclass_fields, is_dataclass, MISSING
from typeguard import check_type
from pymilvus import DataType, FieldSchema

from millie.orm.fields import milvus_field

T = TypeVar('T', bound='MilvusModel')

class MilvusModelMeta(ABCMeta):
    """Metaclass for Milvus models that handles registration and dataclass conversion."""
    _models: Dict[str, Type['MilvusModel']] = {}
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Add base fields if not already present
        if name != 'MilvusModel':
            annotations = namespace.setdefault('__annotations__', {})
            
            # Get parent annotations
            parent_annotations = {}
            for base in bases:
                if hasattr(base, '__annotations__'):
                    parent_annotations.update(base.__annotations__)
            
            # Update annotations
            for field_name, field_type in parent_annotations.items():
                if field_name not in annotations:
                    annotations[field_name] = field_type
                    # Only add field to namespace if it has a default in parent
                    for base in bases:
                        if hasattr(base, '__dataclass_fields__'):
                            if field_name in base.__dataclass_fields__:
                                field = base.__dataclass_fields__[field_name]
                                if field.default is not dataclass_field or field.default_factory is not dataclass_field:
                                    namespace[field_name] = field.default if field.default is not dataclass_field else field.default_factory
            
            # Add base fields only if they have explicit defaults
            if 'embedding' not in namespace:
                namespace['embedding'] = dataclass_field(default=None)
                annotations['embedding'] = Optional[List[float]]
            if 'metadata' not in namespace:
                namespace['metadata'] = dataclass_field(default_factory=dict)
                annotations['metadata'] = Dict[str, Any]
        
        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Skip registration for MilvusModel itself
        if name == 'MilvusModel':
            return cls
            
        # Convert to dataclass if not already
        if not hasattr(cls, '__dataclass_fields__'):
            # Create a new dataclass that includes all fields
            cls = dataclass(cls, kw_only=True)
            
            # Initialize default fields
            for field_name, field in cls.__dataclass_fields__.items():
                if field.default_factory is not dataclass_field:
                    setattr(cls, field_name, field.default)
        
        # Add type checking to __init__
        original_init = cls.__init__
        def type_checked_init(self, *args, **kwargs):
            # Get all fields from this class and parent classes
            all_fields = {}
            for base in reversed(cls.__mro__):
                if hasattr(base, '__dataclass_fields__'):
                    all_fields.update(base.__dataclass_fields__)
            
            # Check for missing required fields first
            required_fields = []
            for name, field in all_fields.items():
                # A field is required if both default and default_factory are MISSING
                if (field.default is MISSING and 
                    field.default_factory is MISSING and 
                    name not in kwargs):
                    required_fields.append(name)
            
            if required_fields:
                raise TypeError(f"Missing required fields: {', '.join(required_fields)}")
            
            # Check types of all arguments
            hints = get_type_hints(cls)  # Use get_type_hints to resolve forward references
            for name, value in kwargs.items():
                if name in hints and value is not None:
                    # Skip type checking for metadata if it's a string - will be handled in __post_init__
                    if name == 'metadata' and isinstance(value, str):
                        continue
                    try:
                        check_type(value, hints[name])
                    except TypeError as e:
                        raise TypeCheckError(str(e))
            
            # Initialize all fields with their defaults first
            for field_name, field in all_fields.items():
                if field_name not in kwargs:
                    if hasattr(field.default_factory, '__call__'):
                        # Field has a callable default_factory
                        setattr(self, field_name, field.default_factory())
                    elif field.default is not dataclass_field:
                        # Field has a default value
                        setattr(self, field_name, field.default)
                    else:
                        # Field has no default and is not provided
                        setattr(self, field_name, None)
            
            # Then update with provided values
            for name, value in kwargs.items():
                setattr(self, name, value)
            
            # Call parent's __init__ with filtered kwargs
            parent_fields = set()
            for base in bases:
                if hasattr(base, '__dataclass_fields__'):
                    parent_fields.update(base.__dataclass_fields__.keys())
            
            parent_kwargs = {k: v for k, v in kwargs.items() if k in parent_fields}
            super(cls, self).__init__(**parent_kwargs)
            
            # Call post_init to handle JSON parsing
            if hasattr(self, '__post_init__'):
                self.__post_init__()
        cls.__init__ = type_checked_init
        
        # Register the model
        mcs._models[name] = cls
        
        return cls
    
    @classmethod
    def get_model(mcs, name: str) -> Optional[Type['MilvusModel']]:
        """Get a registered model by name."""
        return mcs._models.get(name)
    
    @classmethod
    def get_all_models(mcs) -> List[Type['MilvusModel']]:
        """Get all registered models."""
        return list(mcs._models.values())

class TypeCheckError(TypeError):
    """Raised when a type check fails."""
    pass

class MilvusModel(ABC, metaclass=MilvusModelMeta):
    """Base class for all Milvus models."""
    embedding: Optional[List[float]] = milvus_field(DataType.FLOAT_VECTOR, dim=1536, default=None)
    metadata: Dict[str, Any] = milvus_field(DataType.JSON, default_factory=dict)
    
    # Class variable to mark migration collections
    is_migration_collection: ClassVar[bool] = False
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model with the given fields.
        
        Args:
            **kwargs: Field values to set
        """
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __post_init__(self):
        """Convert metadata and extra_data to dict if they're JSON strings."""
        # Handle metadata field
        if isinstance(self.metadata, str):
            try:
                self.metadata = json.loads(self.metadata)
            except json.JSONDecodeError:
                self.metadata = {}
        
        # Handle extra_data field if it exists
        if hasattr(self, 'extra_data') and isinstance(self.extra_data, str):
            try:
                self.extra_data = json.loads(self.extra_data)
            except json.JSONDecodeError:
                self.extra_data = {}
    
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
            if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                try:
                    data[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass  # Not valid JSON, leave as string
        
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