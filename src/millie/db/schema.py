from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pymilvus import DataType, FieldSchema

@dataclass
class SchemaField:
    """Represents a field in a Milvus schema."""
    name: str
    dtype: str
    max_length: Optional[int] = None
    dim: Optional[int] = None
    is_primary: bool = False
    
    @classmethod
    def from_field_schema(cls, field: FieldSchema) -> 'SchemaField':
        """Create from a Milvus FieldSchema object."""
        return cls(
            name=field.name,
            dtype=field.dtype.name,
            max_length=field.max_length if field.max_length != -1 else None,
            dim=field.dim if hasattr(field, 'dim') and field.dim != -1 else None,
            is_primary=getattr(field, 'is_primary', False)
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaField':
        """Create from a dictionary representation."""
        return cls(
            name=data['name'],
            dtype=data['dtype'],
            max_length=data.get('max_length'),
            dim=data.get('dim'),
            is_primary=data.get('is_primary', False)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = {
            'name': self.name,
            'dtype': self.dtype
        }
        if self.max_length is not None:
            data['max_length'] = self.max_length
        if self.dim is not None:
            data['dim'] = self.dim
        if self.is_primary:
            data['is_primary'] = True
        return data
    
    def to_field_schema(self) -> FieldSchema:
        """Convert to Milvus FieldSchema object."""
        kwargs = {
            'name': self.name,
            'dtype': getattr(DataType, self.dtype)
        }
        if self.max_length is not None:
            kwargs['max_length'] = self.max_length
        if self.dim is not None:
            kwargs['dim'] = self.dim
        if self.is_primary:
            kwargs['is_primary'] = True
        return FieldSchema(**kwargs)

@dataclass
class Schema:
    """Represents a Milvus collection schema."""
    
    def __init__(self, name: str, collection_name: str, fields: List[SchemaField], is_migration_collection: bool = False):
        self.name = name
        self.collection_name = collection_name
        self.fields = fields
        self.is_migration_collection = is_migration_collection
        self.version = 0  # Initialize version to 0
    
    @classmethod
    def from_model(cls, model_class: Any) -> 'Schema':
        """Create schema from a MilvusModel class."""
        if not hasattr(model_class, 'schema'):
            raise ValueError(f"Model {model_class.__name__} missing required schema() method")
        
        schema_def = model_class.schema()
        if schema_def is None:
            raise ValueError(f"Model {model_class.__name__} schema() method returned None")
        
        if not isinstance(schema_def, dict) or 'fields' not in schema_def:
            raise ValueError(f"Model {model_class.__name__} schema() must return a dict with 'fields' key")
            
        fields = [SchemaField.from_field_schema(f) for f in schema_def['fields']]
        
        return cls(
            name=model_class.__name__,
            collection_name=model_class.collection_name(),
            fields=fields,
            is_migration_collection=getattr(model_class, 'is_migration_collection', False)
        )
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Schema':
        """Create schema from dictionary representation."""
        schema = cls(
            name=data["name"],
            collection_name=data["collection_name"],
            fields=[SchemaField.from_dict(field) for field in data["schema"]["fields"]],
            is_migration_collection=data.get("is_migration_collection", False)
        )
        schema.version = data.get("version", 0)  # Load version from dict
        return schema
    
    def to_dict(self) -> Dict:
        """Convert schema to dictionary representation."""
        return {
            "name": self.name,
            "collection_name": self.collection_name,
            "schema": {
                "fields": [field.to_dict() for field in self.fields]
            },
            "is_migration_collection": self.is_migration_collection,
            "version": self.version  # Include version in dict
        }
    
    def get_field(self, name: str) -> Optional[SchemaField]:
        """Get a field by name."""
        for field in self.fields:
            if field.name == name:
                return field
        return None 