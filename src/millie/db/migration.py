"""Base class for Milvus migrations."""
from abc import ABC, abstractmethod
import logging
from typing import List
from pymilvus import Collection, FieldSchema, DataType, CollectionSchema, utility

from millie.orm.decorators import MillieMigrationModel

logger = logging.getLogger(__name__)

class Migration(ABC):
    """Base class for all migrations.
    
    All migrations should inherit from this class and implement methods for
    upgrading and downgrading the schema.
    """
    
    @abstractmethod
    def up(self):
        """Upgrade to this version. Must be implemented by subclasses."""
        raise NotImplementedError
    
    @abstractmethod
    def down(self):
        """Downgrade from this version. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def __init__(self):
        """Initialize the migration."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def apply(self):
        """Apply this migration."""
        try:
            self.logger.info(f"Applying migration {self.__class__.__name__}")
            self.up()
            self.logger.info(f"Successfully applied migration {self.__class__.__name__}")
        except Exception as e:
            self.logger.error(f"Failed to apply migration {self.__class__.__name__}: {str(e)}")
            raise
    
    def rollback(self):
        """Roll back this migration."""
        try:
            self.logger.info(f"Rolling back migration {self.__class__.__name__}")
            self.down()
            self.logger.info(f"Successfully rolled back migration {self.__class__.__name__}")
        except Exception as e:
            self.logger.error(f"Failed to roll back migration {self.__class__.__name__}: {str(e)}")
            raise

    def ensure_collection(self, name: str, fields: List[FieldSchema]) -> Collection:
        """Ensure a collection exists with the given fields.
        
        Args:
            name: The name of the collection
            fields: The fields to create the collection with
            
        Returns:
            The collection object
            
        Raises:
            ValueError: If no fields are provided
        """
        if not fields:
            raise ValueError("No fields provided for collection creation")

        try:
            collection = Collection(name)
            return collection
        except Exception:
            schema = CollectionSchema(fields=fields)
            collection = Collection(name, schema=schema)
            
            # Create auto index for vector fields
            for field in fields:
                if field.dtype == DataType.FLOAT_VECTOR:
                    index_params = {
                        "metric_type": "L2",
                        "index_type": "IVF_FLAT",
                        "params": {"nlist": 1024}
                    }
                    collection.create_index(
                        field_name=field.name,
                        index_params=index_params
                    )
                    self.logger.info(f"Created auto index for vector field {field.name}")
            
            return collection 