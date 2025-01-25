from typing import Dict, Any, Type, TypeVar, Optional, List
from abc import ABC
from datetime import datetime
import json
from dataclasses import dataclass, field

T = TypeVar('T', bound='BaseModel')

@dataclass(kw_only=True)
class BaseModel(ABC):
    """Base class for all Milvus models."""
    # Required fields that all models must have
    id: str
    embedding: List[float]
    
    # Optional fields with defaults
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Convert metadata to dict if it's a JSON string."""
        if isinstance(self.metadata, str):
            try:
                self.metadata = json.loads(self.metadata)
            except json.JSONDecodeError:
                self.metadata = {}
    
    @classmethod
    def collection_name(cls) -> str:
        """Get the collection name for this model."""
        return cls.__name__.lower().replace('model', '')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for Milvus insertion."""
        data = {
            'id': self.id,
            'embedding': self.embedding,
        }
        
        # Only include metadata if it's not empty and not an empty dict
        if self.metadata and (not isinstance(self.metadata, dict) or self.metadata):
            data['metadata'] = self.metadata if isinstance(self.metadata, str) else json.dumps(self.metadata)
        
        # Add any additional fields from child classes
        for key, value in self.__dict__.items():
            if key not in data and not key.startswith('_'):
                if isinstance(value, dict):
                    # Only include non-empty dicts
                    if value:
                        data[key] = json.dumps(value)
                else:
                    data[key] = value
        
        return data
    
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
        
        return cls(**data)
    
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