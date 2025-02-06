from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import field as dataclass_field
from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel

class RuleViolationSchema(TypedDict):
    """Type hints for RuleViolationModel fields."""
    id: str
    rule_id: str
    penalty: str
    description: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]

# Intentionally not inheriting from MilvusModel
class RuleViolationModel():
    """Model for rule violations."""
    # Required fields
    id: str
    rule_id: str  # Reference to the rule this violation is for
    penalty: str
    description: str
    
    # Optional fields with defaults
    embedding: Optional[List[float]] = dataclass_field(default=None)
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)
    
    @classmethod
    def collection_name(cls) -> str:
        return "rule_violations"
    
    @classmethod
    def schema(cls) -> Dict[str, Any]:
        return {
            "fields": [
                FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema("rule_id", DataType.VARCHAR, max_length=100),
                FieldSchema("penalty", DataType.VARCHAR, max_length=50),
                FieldSchema("description", DataType.VARCHAR, max_length=1000),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=1536),
                FieldSchema("metadata", DataType.JSON)
            ]
        }
    
    def format_violation(self) -> str:
        """Format violation for display."""
        return f"{self.penalty.upper()}: {self.description}"
