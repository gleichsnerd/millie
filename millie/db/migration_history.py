from typing import Dict, Any
from dataclasses import field
from datetime import datetime
from pymilvus import DataType, FieldSchema

from millie.orm.milvus_model import MilvusModel

@MilvusModel(is_migration_collection=True)
class MigrationHistoryModel():
    """Tracks applied database migrations."""
    id: str  # Migration filename/identifier
    checksum: str  # Hash of migration contents for detecting changes
    applied_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional info like duration, errors, etc.
    
    @classmethod
    def collection_name(cls) -> str:
        return "migration_history"
    
    @classmethod
    def schema(cls) -> Dict:
        return {
            "fields": [
                FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema("checksum", DataType.VARCHAR, max_length=64),
                FieldSchema("applied_at", DataType.VARCHAR, max_length=30),
                FieldSchema("metadata", DataType.JSON)
            ]
        }
    