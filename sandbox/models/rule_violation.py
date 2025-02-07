from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import field as dataclass_field
from pymilvus import DataType, FieldSchema
from millie.orm.fields import milvus_field
from millie.orm.milvus_model import MilvusModel

class RuleViolationSchema(TypedDict):
    """Type hints for RuleViolationModel fields."""
    id: str
    rule_id: str
    penalty: str
    description: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]

class RuleViolationModel(MilvusModel):
    """Model for rule violations."""
    # Required fields
    id: str = milvus_field(DataType.VARCHAR, max_length=100, is_primary=True)
    rule_id: str = milvus_field(DataType.VARCHAR, max_length=100)
    penalty: str = milvus_field(DataType.VARCHAR, max_length=50)
    description: str = milvus_field(DataType.VARCHAR, max_length=1000)
    
    # Optional fields with defaults
    embedding: Optional[List[float]] = milvus_field(DataType.FLOAT_VECTOR, dim=1536, default=None)
    metadata: Dict[str, Any] = milvus_field(DataType.JSON, default_factory=dict)
    
    @classmethod
    def collection_name(cls) -> str:
        return "rule_violations"
    
    def format_violation(self) -> str:
        """Format violation for display."""
        return f"{self.penalty.upper()}: {self.description}"
