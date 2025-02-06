from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import field as dataclass_field
from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel

class RuleSchema(TypedDict):
    """Type hints for RuleModel fields."""
    id: str
    type: str
    section: str
    description: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]

class RuleModel(MilvusModel):
    """Model for game rules."""
    # Required fields
    id: str
    type: str
    section: str
    description: str
    priority: int
    
    # Optional fields with defaults
    embedding: Optional[List[float]] = dataclass_field(default=None)
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)
    
    @classmethod
    def collection_name(cls) -> str:
        return "rules"
    
    @classmethod
    def schema(cls) -> Dict[str, Any]:
        return {
            "fields": [
                FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema("type", DataType.VARCHAR, max_length=50),
                FieldSchema("section", DataType.VARCHAR, max_length=50),
                FieldSchema("description", DataType.VARCHAR, max_length=1000),
                FieldSchema("priority", DataType.INT64),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=1536),
                FieldSchema("metadata", DataType.JSON)
            ]
        }
    
    def format_rule(self) -> str:
        """Format rule for display."""
        return f"{self.section.upper()} - {self.type}: {self.description}"
