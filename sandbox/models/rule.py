from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import field as dataclass_field
from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel
from millie.orm.fields import milvus_field

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
    id: str = milvus_field(DataType.VARCHAR, max_length=100, is_primary=True)
    type: str = milvus_field(DataType.VARCHAR, max_length=50)
    section: str = milvus_field(DataType.VARCHAR, max_length=50)
    description: str = milvus_field(DataType.VARCHAR, max_length=1000)
    priority: int = milvus_field(DataType.INT64)
    
    # Optional fields with defaults
    embedding: Optional[List[float]] = milvus_field(DataType.FLOAT_VECTOR, dim=1536, default=None)
    metadata: Dict[str, Any] = milvus_field(DataType.JSON, default_factory=dict)
    
    @classmethod
    def collection_name(cls) -> str:
        return "rules"
    
    def format_rule(self) -> str:
        """Format rule for display."""
        return f"{self.section.upper()} - {self.type}: {self.description}"
