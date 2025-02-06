from dataclasses import dataclass, field
from typing import Dict, List, Any, ClassVar, Type
from .fields import field as mfield

@dataclass
class MilvusModel:
    """Base class for Milvus models with field support."""
    
    def __init_subclass__(cls, **kwargs):
        """Process field definitions when a subclass is created."""
        super().__init_subclass__(**kwargs)
        # Ensure class is treated as a dataclass
        return dataclass(cls, **kwargs)
    
    @classmethod
    def collection_name(cls) -> str:
        """Get the collection name for this model."""
        raise NotImplementedError
    
    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """Get the Milvus schema for this model."""
        raise NotImplementedError 