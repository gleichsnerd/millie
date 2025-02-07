"""Model for tracking migration history."""
from typing import Dict, Any, List, Optional
from dataclasses import field as dataclass_field
from datetime import datetime
from pymilvus import DataType, FieldSchema

from millie.orm.fields import milvus_field
from millie.orm.milvus_model import MilvusModel
from millie.orm.decorators import MillieMigrationModel

@MillieMigrationModel
class MigrationHistoryModel(MilvusModel):
    """Model for tracking migration history."""
    id: str = milvus_field(DataType.VARCHAR, max_length=100, is_primary=True)
    name: str = milvus_field(DataType.VARCHAR, max_length=100)
    version: str = milvus_field(DataType.VARCHAR, max_length=50)
    applied_at: str = milvus_field(DataType.VARCHAR, max_length=50)
    
    @classmethod
    def collection_name(cls) -> str:
        return "migration_history"
    
    @classmethod
    def schema(cls) -> Dict[str, Any]:
        return {
            "fields": [
                FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema("name", DataType.VARCHAR, max_length=100),
                FieldSchema("version", DataType.VARCHAR, max_length=50),
                FieldSchema("applied_at", DataType.VARCHAR, max_length=50),
                FieldSchema("metadata", DataType.JSON)
            ]
        }
    