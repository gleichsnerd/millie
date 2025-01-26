from typing import Dict, Any, List
from dataclasses import dataclass, field
from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel

@MilvusModel()
class RuleModel():
    """Model for game rules."""
    # Required fields from base class
    id: str
    
    # Optional fields with defaults
    embedding: List[float] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Additional required fields
    type: str
    section: str
    description: str
    
    @classmethod
    def collection_name(cls) -> str:
        return "rules"
    
    @classmethod
    def schema(cls) -> Dict:
        return {
            "fields": [
                FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema("type", DataType.VARCHAR, max_length=50),
                FieldSchema("section", DataType.VARCHAR, max_length=50),
                FieldSchema("description", DataType.VARCHAR, max_length=1000),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=1536),
                FieldSchema("metadata", DataType.JSON)
            ]
        }
    
    def format_rule(self) -> str:
        """Format rule for display."""
        return f"{self.section.upper()} - {self.type}: {self.description}"
